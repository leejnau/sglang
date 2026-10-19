"""One fresh-process arm of bench_flashinfer_autotune_selection.py.

This is a version-specific experiment, not a model-serving benchmark. It uses
the real SGLang startup context, FlashInfer tuner, v1 JSON persistence and cuBLAS
FP8 kernel. Only the absent model's configuration/cache-path boundaries and the
requested v1/default context controls are supplied by this harness.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("default", "v1", "v2"), required=True)
    parser.add_argument("--phase", choices=("cold", "warm"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--oracle", action="store_true")
    args = parser.parse_args()
    if min(args.m, args.n, args.k, args.samples, args.iterations) < 1:
        parser.error("dimensions, samples and iterations must be positive")
    if any(d % 16 for d in (args.m, args.n, args.k)):
        parser.error("this FP8 experiment requires dimensions divisible by 16")
    return args


def shape_list(inputs):
    return [list(x.shape) if x is not None else None for x in inputs]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision(root):
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def gpu_status():
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,driver_version,pstate,clocks.sm,power.limit",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable: {error}"


def run(args):
    # Set before importing FlashInfer: bundled winners would spoil the baseline.
    os.environ["FLASHINFER_AUTOTUNER_LOAD_FROM_FILE"] = "0"
    os.environ["SGLANG_FLASHINFER_AUTOTUNE_CACHE"] = "1"
    if args.phase == "cold" and args.cache.exists():
        raise RuntimeError(
            "cold cache already exists; use an isolated output directory"
        )
    if args.phase == "warm" and not args.cache.is_file():
        raise RuntimeError("warm phase requires the preceding cold process's cache")
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    before_hash = digest(args.cache) if args.cache.is_file() else None

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("this worker requires an accessible NVIDIA CUDA GPU")
    if torch.__version__.split("+")[0] != "2.14.1":
        raise RuntimeError(f"pinned Torch 2.14.1 required, found {torch.__version__}")
    import flashinfer
    from flashinfer.autotuner import AutoTuner

    if flashinfer.__version__ != "0.7.2":
        raise RuntimeError(
            f"audited FlashInfer 0.7.2 required, found {flashinfer.__version__}; "
            "review the private instrumentation before changing this guard"
        )
    if torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("the first experiment targets Hopper or newer FP8 hardware")
    initial_gpu_status = gpu_status()

    from sglang.srt.model_executor.runner import flashinfer_autotune as startup
    from sglang.srt.runtime_context import get_context, get_parallel

    root = Path(__file__).resolve().parents[3]
    startup_source = Path(startup.__file__).resolve()
    if not startup_source.is_relative_to(root / "python"):
        raise RuntimeError(f"SGLang imported from another checkout: {startup_source}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    tuner = AutoTuner.get()
    if tuner._winner_cache() or tuner._file_configs or tuner._active_managed_store:
        raise RuntimeError("unexpected preexisting tuner state in fresh worker")
    runner = SimpleNamespace(
        device="cuda", forward_stream=torch.cuda.Stream(), is_draft_worker=False
    )

    # Unit dequantization scales deliberately isolate tactic selection from the
    # quantizer. Both arms get exactly the same already-quantized operands.
    a = torch.randn((1, args.m, args.k), device=device).to(torch.float8_e4m3fn)
    b = (
        torch.randn((1, args.n, args.k), device=device)
        .to(torch.float8_e4m3fn)
        .transpose(1, 2)
    )
    scale_a = torch.ones(1, device=device)
    scale_b = torch.ones(1, device=device)
    out = torch.empty((1, args.m, args.n), device=device, dtype=torch.bfloat16)

    def operation():
        return flashinfer.bmm_fp8(
            a, b, scale_a, scale_b, torch.bfloat16, out=out, backend="cublas"
        )

    reference = torch.bmm(a.float(), b.float()) * scale_a * scale_b
    original_choose = tuner.choose_one
    original_search = tuner.search_cache
    original_profile = tuner._profile_single_kernel
    selections, profiles = [], []
    captured = {}
    stage = "setup"
    profile_calls = 0
    disallow_profiling = False

    def profile_hook(r, inputs, tactic, config, *more, **kwargs):
        nonlocal profile_calls
        profile_calls += 1
        if disallow_profiling or args.phase == "warm":
            raise RuntimeError(f"unexpected profiling during {stage}/{args.phase}")
        record = {
            "runner": type(r).__name__,
            "tactic": int(tactic),
            "input_shapes": shape_list(inputs),
            "use_cuda_graph": config.use_cuda_graph,
            "use_cold_l2_cache": config.use_cold_l2_cache,
        }
        profiles.append(record)
        try:
            measured = original_profile(r, inputs, tactic, config, *more, **kwargs)
            record["profiler_ms"] = measured if math.isfinite(measured) else None
            return measured
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise

    def search_hook(op, runners, shapes, config, *more, **kwargs):
        result = original_search(op, runners, shapes, config, *more, **kwargs)
        if op == "fp8_gemm":
            captured["last_hit"] = bool(result[0])
        return result

    def choose_hook(op, runners, config, inputs, **kwargs):
        captured["last_hit"] = False
        selected, tactic = original_choose(op, runners, config, inputs, **kwargs)
        if op != "fp8_gemm":
            raise RuntimeError(f"unexpected tuned operation {op}")
        captured.update(op=op, runners=runners, config=config, inputs=inputs)
        selections.append(
            {
                "stage": stage,
                "operation": op,
                "runner": type(selected).__name__,
                "tactic": int(tactic),
                "cache_hit": captured["last_hit"],
            }
        )
        return selected, tactic

    def check_output():
        # Compare with the actual FP8 inputs, rather than unquantized originals.
        torch.testing.assert_close(out.float(), reference, rtol=0.02, atol=0.125)
        return float((out.float() - reference).abs().max().item())

    def measure(call, execution):
        # No choose/search tracing here. The profiling guard stays installed but
        # is never called on a normal lookup path, adding no per-call hook cost.
        for _ in range(10):
            call()
        torch.cuda.synchronize()
        wall_samples, gpu_samples = [], []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        end_event.record()
        end_event.synchronize()
        for _ in range(args.samples):
            start = time.perf_counter()
            start_event.record()
            for _ in range(args.iterations):
                call()
            end_event.record()
            end_event.synchronize()
            wall_samples.append((time.perf_counter() - start) * 1000 / args.iterations)
            gpu_samples.append(start_event.elapsed_time(end_event) / args.iterations)
        return {
            "samples_ms": wall_samples,
            "gpu_event_samples_ms": gpu_samples,
            "median_ms": statistics.median(wall_samples),
            "iterations_per_sample": args.iterations,
            "scope": (
                "public API eager dispatch; synchronized batched wall time per invocation"
                if execution == "eager"
                else "captured CUDA Graph replay; synchronized batched wall time per replay"
            ),
            "event_scope": "GPU event span including any host submission gaps",
            "l2_condition": "reused hot buffers; no explicit L2 flush",
        }

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            get_context().override_server_args(
                moe_runner_backend="triton", flashinfer_autotune_skip_ops=[]
            )
        )
        stack.enter_context(
            get_parallel().override(tp_group=SimpleNamespace(world_size=1))
        )
        stack.enter_context(patch.object(tuner, "_profile_single_kernel", profile_hook))
        # Initialize the JIT module/CUDA libraries outside the startup interval.
        import flashinfer.autotuner as autotuner_module

        original_autotune = autotuner_module.autotune
        setup_start = time.perf_counter()
        with original_autotune(False):
            operation()
        torch.cuda.synchronize()
        setup_ms = (time.perf_counter() - setup_start) * 1000
        check_output()
        if tuner._winner_cache() or tuner._file_configs:
            raise RuntimeError("initial untuned call unexpectedly populated winners")

        stack.enter_context(
            patch.object(
                startup, "flashinfer_autotune_cache_path", lambda _: args.cache
            )
        )
        if args.arm in ("default", "v1"):
            stack.enter_context(
                patch.object(
                    flashinfer, "autotune_v2", lambda **_: contextlib.nullcontext()
                )
            )
        if args.arm == "default":

            def untuned_context(_mode=True, **kwargs):
                return original_autotune(False, **kwargs)

            stack.enter_context(
                patch.object(autotuner_module, "autotune", untuned_context)
            )

        trace = contextlib.ExitStack()
        stack.callback(trace.close)
        trace.enter_context(patch.object(tuner, "choose_one", choose_hook))
        trace.enter_context(patch.object(tuner, "search_cache", search_hook))
        stage = "startup"
        torch.cuda.synchronize()
        start = time.perf_counter()
        with startup.flashinfer_autotune_context(runner, run_lm_head=False):
            operation()
        torch.cuda.synchronize()
        startup_ms = (time.perf_counter() - start) * 1000
        startup_profile_calls = profile_calls
        cache_hash = digest(args.cache)
        disallow_profiling = True
        if (args.arm == "default" or args.phase == "warm") and profile_calls:
            raise RuntimeError("default/warm startup unexpectedly attempted profiling")
        if tuner.is_tuning_mode:
            raise RuntimeError("startup context left tuning enabled")
        if args.arm != "default" and args.phase == "cold" and not profile_calls:
            raise RuntimeError("cold tuned arm failed to enter the real profiler")
        if args.arm == "v2" and args.phase == "cold":
            if not any(p["tactic"] == -1 for p in profiles):
                raise RuntimeError("v2 default candidate did not participate")
        if args.arm == "v1" and any(p["tactic"] == -1 for p in profiles):
            raise RuntimeError("v1 control unexpectedly profiled the default candidate")

        stage = "bare"
        operation()
        torch.cuda.synchronize()
        natural = selections[-1]
        if args.arm == "default":
            if natural["tactic"] != -1 or natural["cache_hit"]:
                raise RuntimeError("untuned baseline did not use an uncached default")
        elif not natural["cache_hit"]:
            raise RuntimeError("bare serving fell back instead of consuming a winner")
        max_error = check_output()

        from flashinfer.autotuner.autotuner import OptimizationProfile, StaticDim

        inputs = captured["inputs"]
        real_profile = OptimizationProfile(
            [[StaticDim(d) for d in t.shape] for t in inputs], [None] * len(inputs)
        )
        selected_runner = captured["runners"][0]
        eligible = tuner._blocklist.filter(
            "fp8_gemm",
            selected_runner,
            selected_runner.get_valid_tactics(inputs, real_profile),
        )
        if natural["tactic"] >= 0 and natural["tactic"] not in eligible:
            raise RuntimeError(
                "selected ordinal is invalid at runtime shape; cuBLAS would fall back"
            )
        stage = "capture"
        graph = torch.cuda.CUDAGraph()
        before_capture = len(selections)
        with torch.cuda.graph(graph):
            operation()
        if len(selections) == before_capture:
            raise RuntimeError("graph capture did not observe candidate selection")
        if selections[-1]["tactic"] != natural["tactic"]:
            raise RuntimeError("graph capture selected a different winner")
        if args.arm != "default" and not selections[-1]["cache_hit"]:
            raise RuntimeError("graph capture fell back instead of consuming a winner")
        trace.close()
        stage = "measure"
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        max_error = max(max_error, check_output())
        measurements = {
            "eager": measure(operation, "eager"),
            "cuda_graph": measure(graph.replay, "cuda_graph"),
        }
        graph.replay()
        torch.cuda.synchronize()
        max_error = max(max_error, check_output())

        oracle = []
        if args.oracle:
            # Force through the REAL lookup cache so each public call still
            # constructs/enumerates its runner, as it does during serving. A
            # retained direct runner would omit repeated CPU algorithm setup.
            winners = tuner._winner_cache()
            saved = winners.copy()
            key = tuner._get_cache_key(
                "fp8_gemm",
                selected_runner,
                tuple(tuple(t.shape) for t in inputs),
                captured["config"],
                selected_runner.get_cache_key_extras(inputs),
            )
            try:
                candidates = [-1, *eligible]
                random.Random(args.seed).shuffle(candidates)
                for tactic in candidates:
                    winners.clear()
                    winners[key] = (tactic, None)
                    trace_one = contextlib.ExitStack()
                    with trace_one:
                        trace_one.enter_context(
                            patch.object(tuner, "choose_one", choose_hook)
                        )
                        trace_one.enter_context(
                            patch.object(tuner, "search_cache", search_hook)
                        )
                        stage = "oracle_check"
                        operation()
                        observed = selections[-1]
                        if not observed["cache_hit"] or observed["tactic"] != tactic:
                            raise RuntimeError(
                                "forced candidate was not consumed through real lookup"
                            )
                        max_error = max(max_error, check_output())
                        candidate_graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(candidate_graph):
                            operation()
                    stage = "oracle_measure"
                    out.fill_(float("nan"))
                    candidate_graph.replay()
                    torch.cuda.synchronize()
                    max_error = max(max_error, check_output())
                    oracle.append(
                        {
                            "runner": type(selected_runner).__name__,
                            "tactic": tactic,
                            "eager": measure(operation, "eager"),
                            "cuda_graph": measure(candidate_graph.replay, "cuda_graph"),
                        }
                    )
                    candidate_graph.replay()
                    torch.cuda.synchronize()
                    max_error = max(max_error, check_output())
            finally:
                winners.clear()
                winners.update(saved)
            if digest(args.cache) != cache_hash:
                raise RuntimeError(
                    "oracle unexpectedly changed persisted natural winners"
                )
        if profile_calls != startup_profile_calls:
            raise RuntimeError("profiling was attempted after startup")

    props = torch.cuda.get_device_properties(device)
    return {
        "status": "success",
        "arm": args.arm,
        "phase": args.phase,
        "shape": [args.m, args.n, args.k],
        "seed": args.seed,
        "pid": os.getpid(),
        "instrumentation_sha256": digest(Path(__file__)),
        "environment": {
            "sglang_commit": git_revision(root),
            "python": sys.version,
            "sglang_wrapper_source": str(startup_source),
            "sglang_wrapper_sha256": digest(startup_source),
            "flashinfer_tuner_sha256": digest(Path(inspect.getfile(AutoTuner))),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "flashinfer": flashinfer.__version__,
            "gpu": props.name,
            "capability": list(torch.cuda.get_device_capability()),
            "gpu_memory_bytes": props.total_memory,
            "backend": "cublas",
            "gpu_status_before": initial_gpu_status,
            "gpu_status_after": gpu_status(),
            "measurement_policy": "unspecified; operation defaults retained",
            "timer_env": os.environ.get("FLASHINFER_AUTOTUNE_TIMER"),
            "global_timer_enabled": tuner._use_global_timer,
            "startup_timing_config": {
                name: getattr(captured["config"], name)
                for name in (
                    "use_cuda_graph",
                    "use_cold_l2_cache",
                    "profiling_repeat",
                    "cuda_graph_profile_replays",
                    "use_cold_l2_graph_replay",
                )
            },
            "bundled_configs": False,
            "scope": "single-GPU operation; no model/server",
            "strides": {"a": list(a.stride()), "b": list(b.stride())},
            "scales": [1.0, 1.0],
            "output_dtype": "bfloat16",
        },
        "setup_ms": setup_ms,
        "startup_ms": startup_ms,
        "startup_scope": "instrumented wrapper tuning/warmup; initial FP8 module preparation excluded; lazy wrapper/profiler setup included",
        "profile_calls": profile_calls,
        "candidate_profiles": profiles,
        "runtime_eligible_tactics": [int(t) for t in eligible],
        "selections": selections,
        "measurements": measurements,
        "oracle": oracle,
        "oracle_scope": "optional public-API candidate sweep in seeded order; no statistical optimality claim",
        "correctness": {
            "passed": True,
            "max_abs_error": max_error,
            "rtol": 0.02,
            "atol": 0.125,
        },
        "cache_path": str(args.cache),
        "cache_before_sha256": before_hash,
        "cache_sha256": cache_hash,
    }


def main():
    args = parse_args()
    try:
        result = run(args)
    except Exception as error:
        traceback.print_exc()
        result = {
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "pid": os.getpid(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
