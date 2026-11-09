"""Manual integration controls using the installed FlashInfer tuner/cache."""

import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from flashinfer import autotune_v2
from flashinfer.autotuner import AutoTuner, TunableRunner, TuningConfig

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_executor.runner import flashinfer_autotune as sg
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
from sglang.srt.runtime_context import get_context


class Runner(TunableRunner):
    def get_valid_tactics(self, inputs, profile):
        return [1, 2]

    def forward(self, *a, **k):
        raise AssertionError("controlled profiling must not execute kernels")


def snapshot(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*.json")
    }


def run():
    checks = []

    def check(ok, label):
        assert ok, label
        checks.append(label)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tuner, runner = AutoTuner(), Runner()
        config, inputs = TuningConfig(), [torch.ones(2)]
        profiles, resets = [], []

        def profile(runner, inputs, tactic, config, **kwargs):
            policy = tuner._effective_measure_policy
            winner = 2 if policy is not None else 1
            profiles.append(tactic)
            return 1.0 if tactic == winner else 10.0

        def choose(op="sglang_scoped_control"):
            return tuner.choose_one(op, [runner], config, inputs)[1]

        def state():
            return (
                tuner.is_tuning_mode,
                tuner._winner_cache_identity(),
                tuner._effective_skip_ops,
                tuner._managed_cache,
                tuner._v2_local.active,
                tuner._v2_local.strict_replay,
            )

        def reset():
            resets.append(True)

        with patch.object(AutoTuner, "get", lambda: tuner):
            tuner._profile_single_kernel = profile
            with autotune_v2(persistent_cache=False):
                check(choose() == 1, "legacy winner selected")
            legacy = root / "legacy.json"
            tuner.save_configs(str(legacy))
            before = state()
            scope = sg.FlashInferGraphAutotune(root / "graph", set())
            with scope.capture_context(choose, reset):
                check(choose() == 2, "graph winner differs from legacy")
                check(tuner._v2_local.strict_replay, "strict capture lookup")
            check(state() == before and choose() == 1, "bare legacy lookup restored")
            check(len(resets) == 1, "preparation resets attention state")
            tuner.save_configs(str(root / "after.json"))
            check(
                legacy.read_bytes() == (root / "after.json").read_bytes(),
                "v1 persistence excludes graph winner",
            )
            cold_files = snapshot(root / "graph")
            check(bool(cold_files), "graph winner persisted")
            count = len(profiles)
            tuner = AutoTuner()
            tuner._profile_single_kernel = profile
            tuner.load_configs(str(legacy))
            with scope.capture_context(choose, reset):
                check(choose() == 2, "fresh tuner restores graph winner")
            check(len(profiles) == count, "warm preparation and capture do not profile")
            check(snapshot(root / "graph") == cold_files, "warm cache bytes unchanged")
            check(choose() == 1, "fresh bare serving still uses v1")
            before = state()
            try:
                with scope.capture_context(choose, reset):
                    choose("uncovered")
                raise AssertionError("strict miss was accepted")
            except RuntimeError:
                pass
            check(
                state() == before and len(profiles) == count,
                "strict miss restores state without profiling",
            )
            for where in ("prepare", "capture"):

                def fail():
                    raise ValueError("injected " + where)

                n = len(resets)
                try:
                    with scope.capture_context(
                        fail if where == "prepare" else choose, reset
                    ):
                        fail()
                    raise AssertionError("failure not raised")
                except ValueError:
                    pass
                check(
                    state() == before and len(resets) == n + 1,
                    where + " error restores scope and resets preparation",
                )
            skipped = sg.FlashInferGraphAutotune(
                root / "graph", {"sglang_scoped_control"}
            )
            with skipped.capture_context(choose, reset):
                check(choose() == -1, "skip overrides populated graph winner")
            check(
                choose() == 1 and len(profiles) == count,
                "skip restores legacy without profiling",
            )

            mr = SimpleNamespace(
                device="cuda",
                is_generation=True,
                is_draft_worker=False,
                spec_algorithm=SimpleNamespace(is_speculative=lambda: False),
                model_config=SimpleNamespace(quantization="modelopt_fp8"),
                model=torch.nn.Linear(2, 2),
            )
            graph = SimpleNamespace(
                model_runner=mr,
                backend=FullCudaGraphBackend.__new__(FullCudaGraphBackend),
                capture_forward_mode=ForwardMode.DECODE,
                enable_torch_compile=False,
                enable_two_batch_overlap=False,
            )
            ctx = get_context()
            with ctx.override_server_args(
                model_path="dummy", device="cuda", quantization="modelopt_fp8"
            ):
                with envs.SGLANG_ENABLE_FLASHINFER_GRAPH_AUTOTUNE.override(False):
                    check(
                        sg.FlashInferGraphAutotune.create(graph) is None,
                        "opt-out has no scope",
                    )
                with (
                    envs.SGLANG_ENABLE_FLASHINFER_GRAPH_AUTOTUNE.override(True),
                    patch.object(
                        sg, "flashinfer_autotune_cache_path", lambda _: legacy
                    ),
                ):
                    check(
                        sg.FlashInferGraphAutotune.create(graph).cache_root
                        == legacy.with_suffix(".decode_graph"),
                        "stable deployment cache",
                    )
                    with envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.override(False):
                        a, b = (
                            sg.FlashInferGraphAutotune.create(graph) for _ in range(2)
                        )
                        check(
                            a.cache_root != b.cache_root
                            and a.cache_root.parent.name == "runs",
                            "reuse-off isolates each capture session",
                        )
                        with a.capture_context(choose, reset):
                            check(choose() == 2, "reuse-off graph scope works")
                        check(
                            snapshot(root / "graph") == cold_files,
                            "reuse-off stable store untouched",
                        )
                    for field, value in [
                        ("disable_flashinfer_autotune", True),
                        ("enable_deterministic_inference", True),
                        ("tp_size", 2),
                        ("pp_size", 2),
                        ("enable_lora", True),
                        ("moe_runner_backend", "flashinfer_megamoe"),
                    ]:
                        previous = ctx.config_leaf(field)
                        ctx.override("graph-control", **{field: value})
                        try:
                            check(
                                sg.FlashInferGraphAutotune.create(graph) is None,
                                field + " preserves existing path",
                            )
                        finally:
                            ctx.override("graph-control-restore", **{field: previous})
                    for field, value in [
                        ("enable_torch_compile", True),
                        ("enable_two_batch_overlap", True),
                        ("capture_forward_mode", ForwardMode.TARGET_VERIFY),
                    ]:
                        previous = getattr(graph, field)
                        setattr(graph, field, value)
                        check(
                            sg.FlashInferGraphAutotune.create(graph) is None,
                            field + " remains outside scope",
                        )
                        setattr(graph, field, previous)
            with patch.object(torch.cuda, "is_current_stream_capturing", lambda: True):
                count = len(profiles)
                try:
                    with scope.capture_context(choose, reset):
                        pass
                    raise AssertionError("active capture was accepted")
                except RuntimeError:
                    pass
                check(len(profiles) == count, "tuning forbidden during active capture")
    return {"checks": checks, "count": len(checks), "status": "passed"}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
