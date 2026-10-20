"""Compare natural default/v1/v2 selection on one pinned descriptor source.

This manual experiment uses the actual startup wrapper and existing v1 JSON
persistence. Independent candidate measurements preserve the natural lookup
source: memory after cold tuning, loaded files after a fresh warm restart.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import ctypes
import inspect
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from flashinfer_autotune_descriptor_qualification import (
    prepare_native,
    tactic_json,
    tensor_digest,
)
from flashinfer_autotune_eager_setup import (
    MAX_ITERATIONS,
    digest,
    gpu_status,
    measure,
    positive_float,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("default", "v1", "v2"), required=True)
    parser.add_argument("--phase", choices=("cold", "warm"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--m", type=int, choices=(128, 144), required=True)
    parser.add_argument("--n", type=int, choices=(1024,), default=1024)
    parser.add_argument("--k", type=int, choices=(1024,), default=1024)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup-ms", type=positive_float, default=500)
    parser.add_argument("--min-sample-ms", type=positive_float, default=20)
    parser.add_argument("--oracle", action="store_true")
    args = parser.parse_args()
    if min(args.samples, args.iterations) < 1 or args.iterations > MAX_ITERATIONS:
        parser.error("sample/iteration counts must be positive and within the bound")
    return args


class SelectionExperiment:
    def __init__(self, args):
        self.args = args
        if os.environ.get("LD_PRELOAD") or getattr(
            ctypes.CDLL(None), "fi_probe_set_phase", None
        ):
            raise RuntimeError("timing workers must run without native preload/hooks")
        if args.phase == "cold" and args.cache.exists():
            raise RuntimeError("cold worker refuses to overwrite an existing cache")
        if args.phase == "warm" and not args.cache.is_file():
            raise RuntimeError("warm worker requires the preceding cold worker's cache")
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        self.cache_before = digest(args.cache) if args.cache.is_file() else None
        os.environ["FLASHINFER_AUTOTUNER_LOAD_FROM_FILE"] = "0"
        os.environ["SGLANG_FLASHINFER_AUTOTUNE_CACHE"] = "1"

        import flashinfer
        import torch
        from flashinfer.autotuner import AutoTuner
        from flashinfer.gemm import gemm_base

        self.torch, self.flashinfer, self.gemm_base = torch, flashinfer, gemm_base
        if torch.__version__ != "2.14.1+cu130" or torch.version.cuda != "13.0":
            raise RuntimeError(
                "the pinned Torch 2.14.1+cu130/CUDA13.0 stack is required"
            )
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            9,
            0,
        ):
            raise RuntimeError("the preregistered device is an SM90 H100")
        if flashinfer.__version__ != "0.7.2":
            raise RuntimeError("the qualified descriptor overlay over0.7.2 is required")
        self.manifest = json.loads(args.manifest.read_text())
        self.native_artifact = prepare_native("candidate", self.manifest)
        self.tuner = AutoTuner.get()
        if (
            self.tuner._winner_cache()
            or self.tuner._file_configs
            or self.tuner._active_managed_store
            or self.tuner.is_tuning_mode
        ):
            raise RuntimeError("fresh worker unexpectedly has tuner state")
        if self.tuner._winner_cache() is not self.tuner.profiling_cache:
            raise RuntimeError(
                "unspecified policy must use the existing v1 winner partition"
            )
        self.initial_gpu_status = gpu_status()
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.a = torch.randn((1, args.m, args.k), device="cuda").to(torch.float8_e4m3fn)
        self.b = (
            torch.randn((1, args.n, args.k), device="cuda")
            .to(torch.float8_e4m3fn)
            .transpose(1, 2)
        )
        self.scale_a, self.scale_b = (
            torch.ones(1, device="cuda"),
            torch.ones(1, device="cuda"),
        )
        self.out = torch.empty((1, args.m, args.n), device="cuda", dtype=torch.bfloat16)
        self.reference = (
            torch.bmm(self.a.float(), self.b.float()) * self.scale_a * self.scale_b
        )
        self.operands = self.operand_digests()
        self.model_runner = SimpleNamespace(
            device="cuda", forward_stream=torch.cuda.Stream(), is_draft_worker=False
        )
        self.caller_stream = torch.cuda.current_stream()
        self.original_choose, self.original_search = (
            self.tuner.choose_one,
            self.tuner.search_cache,
        )
        self.original_profile = self.tuner._profile_single_kernel
        self.profile_attempts = 0
        self.forbidden_profile_attempts = []
        self.profiles, self.selections, self.correctness = [], [], []
        self.binding_checks = []
        self.stage = "setup"
        self.allow_profiling = False
        self.captured = {}
        self.last_hit = False
        self.last_tier = "miss"
        self.last_tactic = None
        self.oracle = []

    def operand_digests(self):
        return {
            name: tensor_digest(self.torch, getattr(self, name))
            for name in ("a", "b", "scale_a", "scale_b")
        }

    def operation(self):
        return self.flashinfer.bmm_fp8(
            self.a,
            self.b,
            self.scale_a,
            self.scale_b,
            self.torch.bfloat16,
            out=self.out,
            backend="cublas",
        )

    def check_output(self, label):
        self.torch.cuda.synchronize()
        self.torch.testing.assert_close(
            self.out.float(), self.reference, rtol=0.02, atol=0.125
        )
        self.correctness.append(
            {
                "check": label,
                "max_abs_error": float(
                    (self.out.float() - self.reference).abs().max().item()
                ),
            }
        )

    def profile_hook(self, runner, inputs, tactic, config, *more, **kwargs):
        # Profiling errors may be swallowed by the tuner: record BEFORE raising.
        self.profile_attempts += 1
        if not self.allow_profiling:
            self.forbidden_profile_attempts.append(self.stage)
            raise RuntimeError("unexpected profiling outside cold tuned startup")
        shapes = [list(t.shape) if t is not None else None for t in inputs]
        row = {
            "stage": self.stage,
            "runner": type(runner).__name__,
            "tactic": tactic_json(tactic),
            "input_shapes": shapes,
            "normalized_profile": [
                list(s)
                for s in self.tuner._find_nearest_profile(
                    tuple(tuple(s) if s is not None else (0,) for s in shapes), config
                )
            ],
            "use_cuda_graph": config.use_cuda_graph,
            "use_cold_l2_cache": config.use_cold_l2_cache,
        }
        self.profiles.append(row)
        try:
            measured = self.original_profile(
                runner, inputs, tactic, config, *more, **kwargs
            )
            row["profiler_ms"] = measured if math.isfinite(measured) else None
            return measured
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            raise

    def search_hook(self, op, runners, shapes, config, *more, **kwargs):
        result = self.original_search(op, runners, shapes, config, *more, **kwargs)
        if op != "fp8_gemm":
            raise RuntimeError("unexpected tuned operation")
        self.last_hit = bool(result[0])
        self.last_tier = "miss"
        if self.last_hit:
            inputs = kwargs.get("inputs", more[0] if more else None)
            selected = runners[result[1]]
            key = self.tuner._get_cache_key(
                op,
                selected,
                shapes,
                config,
                selected.get_cache_key_extras(inputs) if inputs is not None else (),
            )
            if (
                key in self.tuner._winner_cache()
                and self.tuner._winner_cache()[key][0] == result[2]
            ):
                self.last_tier = "memory"
            elif (
                key.file_key in self.tuner._file_configs
                and self.tuner._file_configs[key.file_key][1] == result[2]
            ):
                self.last_tier = "file"
            else:
                raise RuntimeError("hit came from an unexpected cache source")
        return result

    def choose_hook(self, op, runners, config, inputs, **kwargs):
        self.last_hit, self.last_tier = False, "miss"
        selected, tactic = self.original_choose(op, runners, config, inputs, **kwargs)
        if (
            op != "fp8_gemm"
            or len(runners) != 1
            or type(selected).__name__ != "CublasFp8GemmRunner"
        ):
            raise RuntimeError("unexpected operation or candidate runner pool")
        self.captured.update(runner=selected, config=config, inputs=inputs)
        self.last_tactic = tactic
        stream = self.torch.cuda.current_stream()
        capturing = self.torch.cuda.is_current_stream_capturing()
        if self.stage == "startup" and (
            stream != self.model_runner.forward_stream or capturing
        ):
            raise RuntimeError("startup did not use the wrapper's forward stream")
        if self.stage != "startup" and not capturing and stream != self.caller_stream:
            raise RuntimeError("bare operation did not restore the caller stream")
        self.selections.append(
            {
                "stage": self.stage,
                "tactic": tactic_json(tactic),
                "cache_hit": self.last_hit,
                "lookup_tier": self.last_tier,
                "capturing": capturing,
                "on_forward_stream": stream == self.model_runner.forward_stream,
                "on_caller_stream": stream == self.caller_stream,
            }
        )
        return selected, tactic

    @contextlib.contextmanager
    def trace_lookup(self):
        with (
            patch.object(self.tuner, "search_cache", self.search_hook),
            patch.object(self.tuner, "choose_one", self.choose_hook),
        ):
            yield

    @contextlib.contextmanager
    def native_observation(self):
        original = self.native_module
        calls = []

        class ObservedNative:
            def bmm_fp8_run_with_descriptor(self, *args, **kwargs):
                used = original.bmm_fp8_run_with_descriptor(*args, **kwargs)
                calls.append(int(used))
                return used

            def __getattr__(self, name):
                return getattr(original, name)

        if self.native_cell.cell_contents is not original:
            raise RuntimeError("nested native observation")
        self.native_cell.cell_contents = ObservedNative()
        try:
            yield calls
        finally:
            self.native_cell.cell_contents = original

    def observed_call(self, label, expected=None, tier=None, allow_rejection=False):
        self.stage = label
        self.out.fill_(float("nan"))
        before = len(self.selections)
        with self.trace_lookup(), self.native_observation() as calls:
            self.operation()
        if len(self.selections) != before + 1:
            raise RuntimeError("missing actual public selection observation")
        row = self.selections[-1]
        tactic = self.last_tactic
        if expected is not None and tactic != expected:
            raise RuntimeError("public operation did not consume the expected tactic")
        if tier is not None and row["lookup_tier"] != tier:
            raise RuntimeError("public operation used the wrong cache lookup source")
        if row["cache_hit"] != (row["lookup_tier"] != "miss"):
            raise RuntimeError("cache source/hit observations disagree")
        if row["lookup_tier"] == "file" and self.tuner._winner_cache():
            raise RuntimeError("file lookup unexpectedly populated in-memory winners")
        if not self.captured["runner"].validate_tactic(self.captured["inputs"], tactic):
            raise RuntimeError("schema-invalid candidate was consumed")
        if tactic == -1:
            if calls:
                raise RuntimeError(
                    "default unexpectedly called the descriptor entry point"
                )
            accepted = True
        else:
            if len(calls) != 1 or calls[0] not in (0, 1):
                raise RuntimeError("missing or malformed native descriptor acceptance")
            accepted = calls == [1]
        acceptance = {
            "stage": label,
            "tactic": tactic_json(tactic),
            "accepted": accepted,
            "used_descriptor": calls,
        }
        self.binding_checks.append(acceptance)
        if not accepted and not allow_rejection:
            raise RuntimeError(
                "natural or previously accepted explicit winner fell back to the heuristic"
            )
        self.check_output(label)
        return tactic, row, acceptance

    def capture(self, tactic, tier, label):
        self.stage = label
        graph = self.torch.cuda.CUDAGraph()
        before = len(self.selections)
        with (
            self.trace_lookup(),
            self.native_observation() as calls,
            self.torch.cuda.graph(graph),
        ):
            self.operation()
        if len(self.selections) != before + 1:
            raise RuntimeError("capture did not perform a public tactic lookup")
        selected = self.selections[-1]
        if (
            self.last_tactic != tactic
            or selected["lookup_tier"] != tier
            or not selected["capturing"]
        ):
            raise RuntimeError("capture changed the selected tactic or lookup source")
        if tier == "file" and self.tuner._winner_cache():
            raise RuntimeError("file capture unexpectedly populated in-memory winners")
        expected = [] if tactic == -1 else [1]
        if calls != expected:
            raise RuntimeError(
                "capture fell back instead of accepting the selected descriptor"
            )
        self.binding_checks.append(
            {
                "stage": label,
                "tactic": tactic_json(tactic),
                "accepted": True,
                "used_descriptor": calls,
            }
        )
        return graph

    def check_graph(self, graph, label):
        saved_a, saved_reference = self.a.clone(), self.reference
        try:
            self.a.zero_()
            self.reference = self.torch.zeros_like(saved_reference)
            self.out.fill_(float("nan"))
            graph.replay()
            self.check_output(label + "_changed_input")
        finally:
            self.a.copy_(saved_a)
            self.reference = saved_reference
        self.out.fill_(float("nan"))
        graph.replay()
        self.check_output(label + "_restored_input")

    def measurements(self, graph, label):
        result = {}
        for execution, call in (
            ("eager", self.operation),
            ("cuda_graph", graph.replay),
        ):
            self.stage = label + "_measure_" + execution
            self.out.fill_(float("nan"))
            result[execution] = measure(self.torch, call, self.args)
            self.check_output(label + "_measured_" + execution)
        return result

    def setup(self):
        from flashinfer.autotuner import autotune

        started = time.perf_counter()
        with autotune(False), self.trace_lookup():
            self.operation()
        self.torch.cuda.synchronize()
        self.setup_ms = (time.perf_counter() - started) * 1000
        self.check_output("initial_untuned")
        if (
            self.tuner._winner_cache()
            or self.tuner._file_configs
            or self.last_hit
            or self.last_tactic != -1
        ):
            raise RuntimeError(
                "initial untuned call unexpectedly consumed or populated a winner"
            )
        self.module = self.gemm_base.get_gemm_module()
        factory = self.module.cublas_fp8_gemm_runner
        closure = dict(zip(factory.__code__.co_freevars, factory.__closure__))
        if "module" not in closure:
            raise RuntimeError("audited native module closure was not found")
        self.native_cell = closure["module"]
        self.native_module = self.native_cell.cell_contents

    def startup(self):
        import flashinfer.autotuner as autotuner_module

        from sglang.srt.model_executor.runner import flashinfer_autotune as startup
        from sglang.srt.runtime_context import get_context, get_parallel

        self.startup_source = Path(startup.__file__).resolve()
        source_root = Path(__file__).resolve().parents[3]
        if not self.startup_source.is_relative_to(source_root / "python"):
            raise RuntimeError("SGLang wrapper imported from another checkout")
        original_autotune = autotuner_module.autotune
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                get_context().override_server_args(
                    moe_runner_backend="triton", flashinfer_autotune_skip_ops=[]
                )
            )
            stack.enter_context(
                get_parallel().override(tp_group=SimpleNamespace(world_size=1))
            )
            stack.enter_context(
                patch.object(
                    startup, "flashinfer_autotune_cache_path", lambda _: self.args.cache
                )
            )
            if self.args.arm in ("default", "v1"):
                stack.enter_context(
                    patch.object(
                        self.flashinfer,
                        "autotune_v2",
                        lambda **_: contextlib.nullcontext(),
                    )
                )
            if self.args.arm == "default":

                def untuned_context(_mode=True, **kwargs):
                    return original_autotune(False, **kwargs)

                stack.enter_context(
                    patch.object(autotuner_module, "autotune", untuned_context)
                )
            self.stage = "startup"
            self.allow_profiling = (
                self.args.phase == "cold" and self.args.arm != "default"
            )
            try:
                self.torch.cuda.synchronize()
                started = time.perf_counter()
                with (
                    self.trace_lookup(),
                    startup.flashinfer_autotune_context(
                        self.model_runner, run_lm_head=False
                    ),
                ):
                    self.operation()
                self.torch.cuda.synchronize()
                self.startup_ms = (time.perf_counter() - started) * 1000
            finally:
                self.allow_profiling = False
        self.startup_profile_attempts = self.profile_attempts
        if (
            self.tuner.is_tuning_mode
            or self.tuner._active_managed_store
            or self.torch.cuda.current_stream() != self.caller_stream
        ):
            raise RuntimeError("startup leaked tuning/store/stream state")
        if self.args.arm == "default" or self.args.phase == "warm":
            if self.profile_attempts:
                raise RuntimeError("default or warm startup attempted profiling")
        elif not self.profile_attempts:
            raise RuntimeError("cold tuned startup never entered the actual profiler")
        self.cache_after_startup = digest(self.args.cache)
        if self.args.phase == "warm" and self.cache_before != self.cache_after_startup:
            raise RuntimeError("warm startup changed the cold cache bytes")

    def natural_measurements(self):
        expected_tier = (
            "miss"
            if self.args.arm == "default"
            else ("memory" if self.args.phase == "cold" else "file")
        )
        tactic, selected, acceptance = self.observed_call(
            "natural_before", tier=expected_tier
        )
        if self.args.arm == "default" and tactic != -1:
            raise RuntimeError("untuned control did not consume the true-miss default")
        self.runner, self.config, self.inputs = (
            self.captured[k] for k in ("runner", "config", "inputs")
        )
        self.cache_key = self.tuner._get_cache_key(
            "fp8_gemm",
            self.runner,
            tuple(tuple(t.shape) for t in self.inputs),
            self.config,
            self.runner.get_cache_key_extras(self.inputs),
        )
        graph = self.capture(tactic, expected_tier, "natural_capture")
        self.check_graph(graph, "natural")
        measured = self.measurements(graph, "natural")
        self.observed_call("natural_after", expected=tactic, tier=expected_tier)
        self.natural = {
            "tactic": tactic_json(tactic),
            "cache_hit": selected["cache_hit"],
            "lookup_tier": expected_tier,
            "acceptance": acceptance,
            **measured,
        }
        self.natural_tactic = tactic
        runtime_profiles = [
            p
            for p in self.profiles
            if tuple(tuple(s) for s in p["normalized_profile"])
            == self.cache_key.nearest_profile
        ]
        finite = [p for p in runtime_profiles if p.get("profiler_ms") is not None]
        self.runtime_profile_summary = {
            "normalized_profile": [list(s) for s in self.cache_key.nearest_profile],
            "profiles": len(runtime_profiles),
            "finite_profiles": len(finite),
            "minimum_ms": min((p["profiler_ms"] for p in finite), default=None),
            "minimum_tactics": [],
        }
        if finite:
            minimum = self.runtime_profile_summary["minimum_ms"]
            self.runtime_profile_summary["minimum_tactics"] = [
                p["tactic"] for p in finite if p["profiler_ms"] == minimum
            ]
            if (
                tactic_json(tactic)
                not in self.runtime_profile_summary["minimum_tactics"]
            ):
                raise RuntimeError(
                    "natural runtime winner differs from the actual finite profiler minimum"
                )
        if (
            self.args.arm == "v2"
            and self.args.phase == "cold"
            and not any(
                p["tactic"] == -1 and p.get("profiler_ms") is not None
                for p in runtime_profiles
            )
        ):
            raise RuntimeError("v2 default had no finite runtime-bucket profile")
        if self.args.arm == "v1" and any(p["tactic"] == -1 for p in self.profiles):
            raise RuntimeError("v1 control unexpectedly profiled the default")

    def snapshot_state(self):
        # These are the fields clear_cache mutates in the pinned implementation.
        names = (
            "profiling_cache",
            "_ranked_tactics_cache",
            "_profiling_cache_policies",
            "_file_configs",
            "_file_config_policies",
            "_namespaced_records",
            "_dirty_namespaces",
            "_observed_cache_generations",
            "_logged_file_hits",
            "_logged_cache_miss_oor",
            "_winner_partitions",
            "_managed_decoded",
            "_preloaded_stores",
            "_dirty",
            "_dirty_seq",
        )
        return {name: copy.copy(getattr(self.tuner, name)) for name in names}

    def restore_state(self, snapshot):
        self.tuner.clear_cache()
        for name, value in snapshot.items():
            target = getattr(self.tuner, name)
            if isinstance(target, (dict, set)):
                target.clear()
                target.update(value)
            else:
                setattr(self.tuner, name, value)
        if any(getattr(self.tuner, name) != value for name, value in snapshot.items()):
            raise RuntimeError("natural tuner cache state was not restored")

    @contextlib.contextmanager
    def oracle_scope(self, tactic, index):
        saved = self.snapshot_state()
        path = None
        try:
            if self.args.phase == "warm":
                self.tuner.clear_cache()
                if not self.tuner.load_configs(str(self.args.cache)):
                    raise RuntimeError(
                        "oracle failed to load the original natural cache"
                    )
                path = self.args.output.with_name(
                    self.args.output.stem + f".oracle-{index:02d}.json"
                )
                if path.exists():
                    raise RuntimeError("oracle control file already exists")
            self.tuner._winner_cache()[self.cache_key] = (tactic, None)
            if self.args.phase == "warm":
                original_policy = self.tuner._file_config_policies.get(
                    self.cache_key.file_key
                )
                if original_policy is not None:
                    self.tuner._profiling_cache_policies[(None, self.cache_key)] = (
                        original_policy
                    )
                self.tuner.save_configs(str(path))
                self.tuner.clear_cache()
                if not self.tuner.load_configs(str(path)):
                    raise RuntimeError("oracle failed actual candidate file reload")
                if self.tuner._winner_cache():
                    raise RuntimeError("warm oracle unexpectedly has in-memory winners")
            yield {
                "lookup_tier": "memory" if self.args.phase == "cold" else "file",
                "control_file": str(path) if path else None,
                "control_sha256": digest(path) if path else None,
            }
        finally:
            self.restore_state(saved)
            if digest(self.args.cache) != self.cache_after_startup:
                raise RuntimeError("oracle changed the persisted natural winner file")

    def oracle_measurements(self):
        from flashinfer.autotuner.autotuner import OptimizationProfile, StaticDim

        profile = OptimizationProfile(
            [[StaticDim(d) for d in t.shape] for t in self.inputs],
            [None] * len(self.inputs),
        )
        eligible = self.tuner._blocklist.filter(
            "fp8_gemm", self.runner, self.runner.get_valid_tactics(self.inputs, profile)
        )
        if not eligible or any(
            not self.runner.validate_tactic(self.inputs, t) for t in eligible
        ):
            raise RuntimeError(
                "runtime enumeration is empty or contains malformed descriptors"
            )
        self.runtime_eligible_tactics = [tactic_json(t) for t in eligible]
        if self.natural_tactic != -1 and self.natural_tactic not in eligible:
            raise RuntimeError(
                "natural winner is absent from runtime descriptor enumeration"
            )
        candidates = list(dict.fromkeys([-1, *eligible]))
        random.Random(self.args.seed).shuffle(candidates)
        self.oracle_order = [tactic_json(t) for t in candidates]
        if not self.args.oracle:
            return
        for index, tactic in enumerate(candidates):
            label = f"oracle_{index:02d}"
            with self.oracle_scope(tactic, index) as provenance:
                _, _, acceptance = self.observed_call(
                    label + "_before",
                    expected=tactic,
                    tier=provenance["lookup_tier"],
                    allow_rejection=True,
                )
                row = {
                    "tactic": tactic_json(tactic),
                    "accepted": acceptance["accepted"],
                    "exclusion_reason": None,
                    "acceptance": acceptance,
                    **provenance,
                }
                if acceptance["accepted"]:
                    graph = self.capture(
                        tactic, provenance["lookup_tier"], label + "_capture"
                    )
                    self.check_graph(graph, label)
                    row.update(self.measurements(graph, label))
                    self.observed_call(
                        label + "_after",
                        expected=tactic,
                        tier=provenance["lookup_tier"],
                    )
                else:
                    row["exclusion_reason"] = (
                        "native descriptor rejected at the actual runtime shape; public API fell back to the heuristic; no timings accepted"
                    )
                self.oracle.append(row)
        self.observed_call(
            "natural_after_oracle",
            expected=self.natural_tactic,
            tier=self.natural["lookup_tier"],
        )

    def run(self):
        with patch.object(self.tuner, "_profile_single_kernel", self.profile_hook):
            self.setup()
            self.startup()
            self.natural_measurements()
            self.oracle_measurements()
        if (
            self.profile_attempts != self.startup_profile_attempts
            or self.forbidden_profile_attempts
        ):
            raise RuntimeError(
                "profiling was attempted after startup or during warm reuse"
            )
        if self.tuner.is_tuning_mode or self.tuner._active_managed_store:
            raise RuntimeError("worker leaked tuning or managed-store state")
        if self.operands != self.operand_digests():
            raise RuntimeError("fixed-workload operands changed")
        after_hash = digest(self.args.cache)
        if after_hash != self.cache_after_startup:
            raise RuntimeError("runtime checks changed the natural cache file")
        torch = self.torch
        from flashinfer.autotuner import AutoTuner

        source_root = Path(__file__).resolve().parents[3]
        return {
            "status": "success",
            "schema_version": 1,
            "implementation": "candidate",
            "arm": self.args.arm,
            "phase": self.args.phase,
            "shape": [self.args.m, self.args.n, self.args.k],
            "seed": self.args.seed,
            "pid": os.getpid(),
            "worker_sha256": digest(__file__),
            "manifest_sha256": digest(self.args.manifest),
            "environment": {
                "sglang_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
                ).strip(),
                "sglang_wrapper_sha256": digest(self.startup_source),
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "flashinfer": self.flashinfer.__version__,
                "flashinfer_tuner_sha256": digest(inspect.getfile(AutoTuner)),
                "native_artifact": self.native_artifact,
                "helper_sha256": {
                    name: digest(Path(__file__).with_name(name))
                    for name in (
                        "flashinfer_autotune_descriptor_qualification.py",
                        "flashinfer_autotune_eager_setup.py",
                    )
                },
                "gpu": torch.cuda.get_device_properties(0).name,
                "physical_gpu_uuid": str(
                    getattr(torch.cuda.get_device_properties(0), "uuid", None)
                ),
                "capability": list(torch.cuda.get_device_capability()),
                "gpu_status_before": self.initial_gpu_status,
                "gpu_status_after": gpu_status(),
                "LD_PRELOAD": os.environ.get("LD_PRELOAD"),
                "backend": "cublas",
                "measurement_policy": "unspecified",
                "managed_store": False,
                "operand_sha256": self.operands,
                "scope": "single fixed FP8 operation; no model/server",
            },
            "timing_protocol": {
                "warmup_ms": self.args.warmup_ms,
                "min_sample_ms": self.args.min_sample_ms,
                "samples": self.args.samples,
                "requested_iterations": self.args.iterations,
                "oracle_order": self.oracle_order,
                "instrumentation": "none in latency windows except the dormant profiling-attempt guard",
            },
            "initial_public_call_ms": self.setup_ms,
            "startup_ms": self.startup_ms,
            "profile_attempts": self.profile_attempts,
            "forbidden_profile_attempts": self.forbidden_profile_attempts,
            "profiles": self.profiles,
            "selections": self.selections,
            "runtime_profile_summary": self.runtime_profile_summary,
            "natural": self.natural,
            "runtime_eligible_tactics": self.runtime_eligible_tactics,
            "oracle": self.oracle,
            "binding_acceptance_checks": self.binding_checks,
            "cache": {
                "path": str(self.args.cache),
                "runtime_file_key": self.cache_key.file_key,
                "before_sha256": self.cache_before,
                "after_startup_sha256": self.cache_after_startup,
                "after_oracle_sha256": after_hash,
            },
            "correctness": {
                "passed": True,
                "rtol": 0.02,
                "atol": 0.125,
                "checks": self.correctness,
            },
            "scope_notes": [
                "The natural default is an uncached miss; its oracle counterpart is a cached default on the phase-matched lookup source.",
                "Cold oracle uses memory; warm oracle uses real saved/reloaded v1 files. All oracle state is restored and natural cache bytes remain unchanged.",
                "M144 tunes and serves144 while descriptor enumeration uses bucket256; this is not a separate tune256 then serve144 experiment.",
                "Rejected descriptors are reported as fallback exclusions, without accepted performance measurements.",
                "Explicit native binary build/load and the initial untuned public call are excluded from startup_ms; candidate enumeration and profiling performed by the wrapper remain included. Cold startup is diagnostic, not the experiment priority.",
            ],
        }


def main():
    args = parse_args()
    try:
        result = SelectionExperiment(args).run()
    except Exception as error:
        traceback.print_exc()
        result = {
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "pid": os.getpid(),
            "arm": args.arm,
            "phase": args.phase,
            "shape": [args.m, args.n, args.k],
            "seed": args.seed,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
