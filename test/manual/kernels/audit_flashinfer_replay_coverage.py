"""Report-only cache coverage of an actual model's existing decode capture.

Run once with --phase cold, then in a fresh process with --phase warm and the
same --cache-dir. This manual diagnostic changes no production tuning policy.
Numerical forwards run after the observed startup/capture region has finished.
"""

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

MODEL_ID = "nvidia/Llama-3.1-8B-Instruct-FP8"
MODEL_REVISION = "42d9515ebd69eea3a87351d079c671c3c5ff0a31"
PROMPT = "The capital of France is Paris. The capital of Italy is"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def cache_files(cache_dir):
    return {
        str(path.relative_to(cache_dir)): {
            "sha256": digest(path.read_bytes()),
            "contents": json.loads(path.read_text()),
        }
        for path in sorted((cache_dir / "flashinfer" / "autotune").rglob("*.json"))
    }


def tuner_state(tuner):
    return {
        "is_tuning": tuner.is_tuning_mode,
        "active_tuning_contexts": tuner._active_tuning_contexts,
        "managed_store": tuner._active_managed_store is not None,
        "ambient_store": tuner._managed_cache is not None,
        "winner_identity": tuner._winner_cache_identity(),
        "measurement_policy": repr(tuner._effective_measure_policy),
        "skip_ops": sorted(tuner._effective_skip_ops),
        "buckets": tuner._override_tuning_buckets,
        "round_up": tuner._override_round_up,
        "profile_replays": tuner._override_cuda_graph_profile_replays,
    }


def cache_inventory(tuner):
    from flashinfer.autotuner.autotuner import _tactic_to_json

    return {
        "memory": sorted(
            (
                {
                    "key": key.file_key,
                    "runner": key.runner_class_name,
                    "tactic": _tactic_to_json(value[0]),
                }
                for key, value in tuner.profiling_cache.items()
            ),
            key=lambda item: (item["key"], repr(item["tactic"])),
        ),
        "loaded": {
            key: [runner, _tactic_to_json(tactic)]
            for key, (runner, tactic) in sorted(tuner._file_configs.items())
        },
    }


def assert_observable(state):
    if (
        state["is_tuning"]
        or state["active_tuning_contexts"]
        or state["managed_store"]
        or state["ambient_store"]
        or state["winner_identity"] is not None
    ):
        raise RuntimeError(
            "Observer requires completed tuning and the legacy cache identity"
        )


def isolated_report_control():
    """Exercise real CPU lookup paths without touching the model's singleton/cache."""
    import torch
    from flashinfer import autotune_v2
    from flashinfer.autotune_cache import ReplayCoverage
    from flashinfer.autotuner import AutoTuner, autotune
    from flashinfer.autotuner.autotuner import TunableRunner, TuningConfig

    class CpuRunner(TunableRunner):
        def get_valid_tactics(self, inputs, profile):
            return [-1]

        def forward(self, inputs, tactic=-1, **kwargs):
            raise AssertionError("Lookup control must not execute a kernel")

    singleton = AutoTuner.get()
    original_inventory = cache_inventory(singleton)
    tuner, runner, config = AutoTuner(), CpuRunner(), TuningConfig()
    inputs = [torch.ones(2, device="cpu")]
    before = tuner_state(tuner)
    reports = []
    with (
        patch.object(AutoTuner, "get", return_value=tuner),
        patch.object(
            tuner,
            "_profile_single_kernel",
            side_effect=AssertionError("Control profiled"),
        ),
        autotune(
            False,
            tuning_buckets=(1, 4),
            round_up=True,
            skip_ops={"unrelated_control_op"},
        ),
    ):
        expected_state = tuner_state(tuner)
        for seeded in (False, True):
            if seeded:
                key = tuner._get_cache_key(
                    "replay_coverage_cpu_control", runner, ((2,),), config, ()
                )
                tuner._cache_profile_result(key, None, config, -1)
            report = ReplayCoverage()
            with autotune_v2(
                mode="replay",
                persistent_cache=False,
                cache_scope="context",
                replay_report=report,
            ):
                if tuner_state(tuner) != expected_state:
                    raise AssertionError("Control observer changed inherited options")
                selected, tactic = tuner.choose_one(
                    "replay_coverage_cpu_control", [runner], config, inputs
                )
            if (
                selected is not runner
                or tactic != -1
                or tuner_state(tuner) != expected_state
            ):
                raise AssertionError("Control lookup or restoration failed")
            reports.append(dataclasses.asdict(report))
        try:
            with autotune_v2(
                mode="replay",
                persistent_cache=False,
                cache_scope="context",
                replay_report=ReplayCoverage(),
            ):
                raise LookupError("intentional context-exit control")
        except LookupError:
            pass
        exception_restored = (
            tuner_state(tuner) == expected_state and not tuner._in_v2_context
        )
    return {
        "miss": reports[0],
        "cached_default": reports[1],
        "exception_restored": exception_restored,
        "state_restored": tuner_state(tuner) == before,
        "singleton_restored": AutoTuner.get() is singleton,
        "model_cache_unchanged": cache_inventory(singleton) == original_inventory,
        "profile_count": 0,
    }


class Observation:
    def __init__(self, args, result):
        import torch
        from flashinfer import autotune_v2
        from flashinfer.autotune_cache import ReplayCoverage
        from flashinfer.autotuner import AutoTuner

        self.args, self.result = args, result
        self.torch, self.autotune_v2, self.coverage = torch, autotune_v2, ReplayCoverage
        self.tuner_class, self.tuner = AutoTuner, AutoTuner.get()
        self.stage = "startup"
        self.observing = False
        self.profiles = []
        self.protocol_calls = []
        self.replays = 0
        self.load_save = []

    @contextlib.contextmanager
    def install(self):
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )
        from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
            FullCudaGraphBackend,
        )

        original_capture = FullCudaGraphBackend.capture_one
        original_replay = FullCudaGraphBackend.replay
        original_profile = self.tuner_class._profile_single_kernel
        original_protocol = self.tuner_class.choose_with_protocol

        def profile(tuner, runner, inputs, tactic, tuning_config, *args, **kwargs):
            self.profiles.append({"stage": self.stage, "runner": type(runner).__name__})
            return original_profile(
                tuner, runner, inputs, tactic, tuning_config, *args, **kwargs
            )

        def protocol(tuner, *args, **kwargs):
            self.protocol_calls.append(
                {"stage": self.stage, "observed": self.observing}
            )
            if self.observing:
                raise RuntimeError(
                    "Custom protocol selection is outside this observer's neutrality scope"
                )
            return original_protocol(tuner, *args, **kwargs)

        def capture(backend, shape_key, forward_fn, *args, **kwargs):
            runner = backend._cuda_graph_runner
            if type(runner) is not DecodeCudaGraphRunner:
                raise RuntimeError("Expected only the ordinary decode graph runner")
            if runner.model_runner.spec_algorithm.is_speculative():
                raise RuntimeError("Speculative graphs are outside this audit")
            ordinal = 0

            def observed_forward():
                nonlocal ordinal
                capturing = self.torch.cuda.is_current_stream_capturing()
                label = "capture" if capturing else "capture_warmup"
                row = {
                    "ordinal": ordinal,
                    "stage": label,
                    "capturing": capturing,
                    "shape_key": dataclasses.asdict(shape_key),
                    "state_before": tuner_state(self.tuner),
                    "cache_before": cache_inventory(self.tuner),
                    "profile_count_before": len(self.profiles),
                }
                self.result["capture_records"].append(row)
                ordinal += 1
                assert_observable(row["state_before"])
                if self.tuner._in_v2_context:
                    raise RuntimeError("Startup tuning context has not exited")
                report = self.coverage(max_records=8192)
                previous_stage = self.stage
                self.stage, self.observing = label, True
                try:
                    with self.autotune_v2(
                        mode="replay",
                        persistent_cache=False,
                        cache_scope="context",
                        replay_report=report,
                    ):
                        row["state_inside"] = tuner_state(self.tuner)
                        if row["state_inside"] != row["state_before"]:
                            raise RuntimeError(
                                "Observer changed effective lookup state"
                            )
                        return forward_fn()
                finally:
                    self.stage, self.observing = previous_stage, False
                    row["coverage"] = dataclasses.asdict(report)
                    row["state_after"] = tuner_state(self.tuner)
                    row["cache_after"] = cache_inventory(self.tuner)
                    row["profile_count_after"] = len(self.profiles)
                    row["context_restored"] = not self.tuner._in_v2_context
                    if (
                        row["state_after"] != row["state_before"]
                        or not row["context_restored"]
                    ):
                        raise RuntimeError(
                            "Observer did not restore effective lookup state"
                        )

            return original_capture(
                backend, shape_key, observed_forward, *args, **kwargs
            )

        def replay(backend, *args, **kwargs):
            self.replays += 1
            return original_replay(backend, *args, **kwargs)

        def persistence_wrapper(name):
            original = getattr(self.tuner_class, name)

            def invoke(tuner, path, *args, **kwargs):
                before = Path(path).read_bytes() if Path(path).is_file() else None
                returned = original(tuner, path, *args, **kwargs)
                self.load_save.append(
                    {
                        "operation": name,
                        "path": str(Path(path).relative_to(self.args.cache_dir)),
                        "before_sha256": digest(before) if before is not None else None,
                        "after_sha256": digest(Path(path).read_bytes()),
                        "cache_after": cache_inventory(tuner),
                    }
                )
                return returned

            return invoke

        with contextlib.ExitStack() as stack:
            for owner, name, replacement in (
                (FullCudaGraphBackend, "capture_one", capture),
                (FullCudaGraphBackend, "replay", replay),
                (self.tuner_class, "_profile_single_kernel", profile),
                (self.tuner_class, "choose_with_protocol", protocol),
                (self.tuner_class, "load_configs", persistence_wrapper("load_configs")),
                (self.tuner_class, "save_configs", persistence_wrapper("save_configs")),
            ):
                stack.enter_context(patch.object(owner, name, replacement))
            yield


def numerical_checks(bench, tokenizer, observer):
    """Fresh identical prefixes isolate eager/graph KV state for each token case."""
    import torch

    from sglang.benchmark import one_batch

    runner = bench.torch_runner
    graph_runner = runner.decode_cuda_graph_runner
    backend = graph_runner.backend
    if len(backend._graphs) != 1 or len(backend._outputs) != 1:
        raise RuntimeError("Expected exactly one original batch-one graph")
    graph_output = next(iter(backend._outputs.values())).next_token_logits
    if graph_output is None:
        raise RuntimeError("Captured graph has no logits allocation")
    owner_ptr = graph_output.data_ptr()
    graph_object = next(iter(backend._graphs.values()))
    tokens = [
        tokenizer.encode(word, add_special_tokens=False)[-1]
        for word in (" Rome", " Tokyo")
    ]
    if tokens[0] == tokens[1]:
        raise RuntimeError("Changed-input token control is vacuous")
    checks, outputs = [], []
    controls = SimpleNamespace(batch_size=[1], cut_len=0)
    for case, token in zip(
        ("original", "changed", "restored"), (tokens[0], tokens[1], tokens[0])
    ):
        arms = {}
        details = {}
        for arm in ("eager", "graph"):
            bench.clear()
            input_ids, reqs = one_batch.prepare_inputs_for_correctness_test(
                controls, tokenizer, [PROMPT]
            )
            one_batch.prepare_extend_inputs_for_correctness_test(
                controls, input_ids, reqs, None
            )
            _, _, batch = bench.extend(reqs)
            prefix_seq_lens = batch.seq_lens.cpu().tolist()
            before_replays, before_profiles = observer.replays, len(observer.profiles)
            token_tensor = torch.tensor([token], dtype=torch.int64, device="cuda")
            scope = (
                patch.object(graph_runner, "can_run_graph", return_value=False)
                if arm == "eager"
                else contextlib.nullcontext()
            )
            if arm == "graph":
                if graph_output.data_ptr() != owner_ptr:
                    raise RuntimeError("Captured output allocation changed")
                graph_output.fill_(float("nan"))
            with scope:
                _, logits = bench.decode(token_tensor, batch)
                arms[arm] = logits.detach().float().clone()
            torch.cuda.synchronize()
            finite = bool(torch.isfinite(arms[arm]).all().item())
            details[arm] = {
                "finite": finite,
                "prompt_input_ids": input_ids[0],
                "prefix_seq_lens": prefix_seq_lens,
                "decode_token": token,
                "replay_delta": observer.replays - before_replays,
                "profile_delta": len(observer.profiles) - before_profiles,
                "output_shape": list(arms[arm].shape),
                "output_sha256": digest(arms[arm].cpu().numpy().tobytes()),
                "poisoned_graph_output": arm == "graph",
                "graph_output_owner_unchanged": graph_output.data_ptr() == owner_ptr,
                "original_graph_unchanged": next(iter(backend._graphs.values()))
                is graph_object,
                "graph_input_ids": graph_runner.buffers.input_ids[:1].cpu().tolist()
                if arm == "graph"
                else None,
            }
            if not finite:
                raise RuntimeError(
                    "Numerical output remained nonfinite after execution"
                )
        difference = (arms["eager"] - arms["graph"]).abs()
        checks.append(
            {
                "case": case,
                "prompt_input_ids": input_ids[0],
                "decode_token": token,
                "arms": details,
                "atol": 0.05,
                "rtol": 0.01,
                "allclose": bool(
                    torch.allclose(arms["eager"], arms["graph"], atol=0.05, rtol=0.01)
                ),
                "max_abs_error": float(difference.max().item()),
                "argmax_equal": bool(
                    torch.equal(arms["eager"].argmax(-1), arms["graph"].argmax(-1))
                ),
            }
        )
        outputs.append(arms["graph"])
    return {
        "checks": checks,
        "changed_output": not bool(torch.equal(outputs[0], outputs[1])),
        "restored_output_allclose": bool(
            torch.allclose(outputs[0], outputs[2], atol=0.05, rtol=0.01)
        ),
        "scope": "eager_graph_execution_consistency_with_fresh_identical_prefixes",
    }


def run(args, result):
    os.environ["SGLANG_CACHE_DIR"] = str(args.cache_dir)
    os.environ["SGLANG_FLASHINFER_AUTOTUNE_CACHE"] = "1"
    import flashinfer
    import torch

    from sglang.benchmark import one_batch
    from sglang.srt.runtime_context import SpawnRanks, publish
    from sglang.srt.server_args import PortArgs, ServerArgs

    torch.manual_seed(19)
    result["isolated_control"] = isolated_report_control()
    result["environment"] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flashinfer": flashinfer.__version__,
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "pid": os.getpid(),
    }
    result["cache_before_startup"] = cache_files(args.cache_dir)
    if bool(result["cache_before_startup"]) != (args.phase == "warm"):
        raise RuntimeError(
            "Cold needs an empty tuning cache; warm needs its preserved cold cache"
        )
    config = dict(
        model_path=str(args.model_path),
        tokenizer_path=str(args.model_path),
        quantization="modelopt_fp8",
        dtype="bfloat16",
        device="cuda",
        tp_size=1,
        max_running_requests=1,
        max_total_tokens=512,
        context_length=256,
        mem_fraction_static=0.7,
        disable_overlap_schedule=True,
        attention_backend="triton",
        kv_cache_dtype="bfloat16",
        cuda_graph_config={
            "decode": {"backend": "full", "bs": [1], "max_bs": 1},
            "prefill": {"backend": "disabled"},
        },
    )
    result["server_args"] = config
    server_args = ServerArgs(**config)
    server_args.resolve_once()
    one_batch._set_envs_and_config(server_args)
    ports = PortArgs.init_new(server_args)
    publish(server_args, role="scheduler", ranks=SpawnRanks(world_rank=0, gpu_id=0))
    one_batch._init_process_global_configs()
    observer = Observation(args, result)
    try:
        with observer.install(), torch.inference_mode():
            bench, tokenizer = one_batch.load_model(server_args, ports, 0, 0)
            eligible_linears = [
                {
                    "name": name,
                    "weight_shape": list(module.weight.shape),
                    "weight_dtype": str(module.weight.dtype),
                    "input_scale_shape": list(module.input_scale.shape),
                    "weight_scale_shape": list(module.weight_scale.shape),
                }
                for name, module in bench.torch_runner.model.named_modules()
                if getattr(module, "use_flashinfer_bmm", False)
            ]
            result["startup"] = {
                "profile_count": len(observer.profiles),
                "cache": cache_inventory(observer.tuner),
                "state": tuner_state(observer.tuner),
                "files": cache_files(args.cache_dir),
                "eligible_linear_layers": len(eligible_linears),
                "eligible_linears": eligible_linears,
            }
            observer.stage = "numerical"
            result["numerics"] = numerical_checks(bench, tokenizer, observer)
            result["cache_after_numerics"] = cache_files(args.cache_dir)
            result["inventory_after_numerics"] = cache_inventory(observer.tuner)
            result["final_state"] = tuner_state(observer.tuner)
            result["status"] = "completed"
    finally:
        result["profile_calls"] = observer.profiles
        result["protocol_calls"] = observer.protocol_calls
        result["persistence_calls"] = observer.load_save
        result["graph_replays"] = observer.replays
        result["sources"] = {
            name: {
                "path": module.__file__,
                "sha256": digest(Path(module.__file__).read_bytes()),
            }
            for name, module in sorted(sys.modules.copy().items())
            if name
            in {
                "flashinfer.autotune_cache",
                "flashinfer.autotuner.autotuner",
                "flashinfer.gemm.gemm_base",
                "sglang.benchmark.one_batch",
                "sglang.srt.model_executor.runner.flashinfer_autotune",
                "sglang.srt.model_executor.runner.decode_cuda_graph_runner",
                "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend",
                "sglang.srt.layers.quantization.modelopt_quant",
                "sglang.srt.layers.quantization.fp8_utils",
            }
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("cold", "warm"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema": 1,
        "study": "actual_model_decode_replay_coverage",
        "phase": args.phase,
        "status": "started",
        "started_ns": time.time_ns(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "provenance": json.loads(args.provenance.read_text()),
        "provenance_sha256": digest(args.provenance.read_bytes()),
        "worker_sha256": digest(Path(__file__).read_bytes()),
        "capture_records": [],
    }
    try:
        run(args, result)
    except BaseException as error:
        result["status"] = "error"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        result["finished_ns"] = time.time_ns()
        write_json(args.output, result)


if __name__ == "__main__":
    main()
