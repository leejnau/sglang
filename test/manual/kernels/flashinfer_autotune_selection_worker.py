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

MAX_ITERATIONS = 65536


def positive_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


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
    parser.add_argument("--warmup-ms", type=positive_float, default=500)
    parser.add_argument("--min-sample-ms", type=positive_float, default=20)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--oracle", action="store_true")
    args = parser.parse_args()
    if min(args.m, args.n, args.k, args.samples, args.iterations) < 1:
        parser.error("dimensions, samples and iterations must be positive")
    if any(d % 16 for d in (args.m, args.n, args.k)):
        parser.error("this FP8 experiment requires dimensions divisible by 16")
    if args.iterations > MAX_ITERATIONS:
        parser.error(f"--iterations must be at most {MAX_ITERATIONS}")
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
    caller_stream = torch.cuda.current_stream()

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
    eager_measurement_checks = 0
    changed_input_graph_checks = 0

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
        current_stream = torch.cuda.current_stream()
        on_forward_stream = current_stream == runner.forward_stream
        on_caller_stream = current_stream == caller_stream
        capturing = torch.cuda.is_current_stream_capturing()
        if stage == "startup" and (not on_forward_stream or capturing):
            raise RuntimeError(
                "startup selection must use the forward stream outside capture"
            )
        if stage == "bare" and (not on_caller_stream or capturing):
            raise RuntimeError(
                "bare selection must use the caller stream outside capture"
            )
        if stage in ("capture", "oracle_capture") and not capturing:
            raise RuntimeError("capture selection must run inside CUDA Graph capture")
        selections.append(
            {
                "stage": stage,
                "operation": op,
                "runner": type(selected).__name__,
                "tactic": int(tactic),
                "cache_hit": captured["last_hit"],
                "on_forward_stream": on_forward_stream,
                "on_caller_stream": on_caller_stream,
                "capturing": capturing,
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
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        # Materialize CUDA event resources before warmup and calibration.
        start_event.record()
        end_event.record()
        end_event.synchronize()

        def wall_batch(iterations):
            start = time.perf_counter()
            for _ in range(iterations):
                call()
            torch.cuda.synchronize()
            return (time.perf_counter() - start) * 1000

        warmup_calls = 0
        warmup_start = time.perf_counter()
        while True:
            wall_batch(args.iterations)
            warmup_calls += args.iterations
            warmup_actual_ms = (time.perf_counter() - warmup_start) * 1000
            if warmup_actual_ms >= args.warmup_ms:
                break

        effective_iterations = args.iterations
        calibration_batches = []
        while True:
            elapsed_ms = wall_batch(effective_iterations)
            calibration_batches.append(
                {"iterations": effective_iterations, "wall_ms": elapsed_ms}
            )
            if elapsed_ms >= args.min_sample_ms:
                break
            if effective_iterations == MAX_ITERATIONS:
                raise RuntimeError(
                    f"{execution} could not reach a {args.min_sample_ms:g} ms batch "
                    f"within {MAX_ITERATIONS} iterations (last {elapsed_ms:g} ms)"
                )
            effective_iterations = min(effective_iterations * 2, MAX_ITERATIONS)

        wall_samples, gpu_samples = [], []
        wall_batches, gpu_batches = [], []
        for _ in range(args.samples):
            start = time.perf_counter()
            start_event.record()
            for _ in range(effective_iterations):
                call()
            end_event.record()
            end_event.synchronize()
            wall_ms = (time.perf_counter() - start) * 1000
            gpu_ms = start_event.elapsed_time(end_event)
            wall_batches.append(wall_ms)
            gpu_batches.append(gpu_ms)
            wall_samples.append(wall_ms / effective_iterations)
            gpu_samples.append(gpu_ms / effective_iterations)
        return {
            "samples_ms": wall_samples,
            "gpu_event_samples_ms": gpu_samples,
            "wall_batch_ms": wall_batches,
            "gpu_event_batch_ms": gpu_batches,
            "median_ms": statistics.median(wall_samples),
            "requested_iterations": args.iterations,
            "effective_iterations": effective_iterations,
            "iterations_per_sample": effective_iterations,
            "warmup_actual_ms": warmup_actual_ms,
            "warmup_calls": warmup_calls,
            "warmup_batches": warmup_calls // args.iterations,
            "calibration_batches": calibration_batches,
            "scope": (
                "public API eager dispatch; synchronized batched wall time per invocation"
                if execution == "eager"
                else "captured CUDA Graph replay; synchronized batched wall time per replay"
            ),
            "event_scope": "GPU event span including any host submission gaps",
            "l2_condition": "reused hot buffers; no explicit L2 flush",
        }

    def measure_checked(call, execution):
        nonlocal max_error, eager_measurement_checks
        # Keep correctness work outside all timing windows, and check before a
        # subsequent execution mode can overwrite a bad or unwritten output.
        out.fill_(float("nan"))
        measurement = measure(call, execution)
        torch.cuda.synchronize()
        max_error = max(max_error, check_output())
        if execution == "eager":
            eager_measurement_checks += 1
        return measurement

    def check_graph_inputs(graph):
        nonlocal reference, changed_input_graph_checks
        saved_a = a.clone()
        saved_reference = reference
        try:
            # A captured graph must read updated tensor contents, not merely
            # rewrite an answer previously produced during warmup/capture.
            a.zero_()
            reference = torch.zeros_like(saved_reference)
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            changed_error = check_output()
            changed_input_graph_checks += 1
        finally:
            a.copy_(saved_a)
            reference = saved_reference
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        return max(changed_error, check_output())

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

            def normalized_profile(shapes):
                return tuner._find_nearest_profile(
                    tuple(tuple(s) if s is not None else (0,) for s in shapes),
                    captured["config"],
                )

            runtime_key = normalized_profile(shape_list(captured["inputs"]))
            if not any(
                p["tactic"] == -1
                and normalized_profile(p["input_shapes"]) == runtime_key
                and p.get("profiler_ms") is not None
                for p in profiles
            ):
                raise RuntimeError(
                    "v2 default candidate lacked a finite successful profile "
                    f"at runtime bucket {runtime_key[0]}"
                )
        if args.arm == "v1" and any(p["tactic"] == -1 for p in profiles):
            raise RuntimeError("v1 control unexpectedly profiled the default candidate")

        stage = "bare"
        out.fill_(float("nan"))
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
        max_error = max(max_error, check_graph_inputs(graph))
        measurements = {
            "eager": measure_checked(operation, "eager"),
            "cuda_graph": measure_checked(graph.replay, "cuda_graph"),
        }

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
                        out.fill_(float("nan"))
                        operation()
                        observed = selections[-1]
                        if not observed["cache_hit"] or observed["tactic"] != tactic:
                            raise RuntimeError(
                                "forced candidate was not consumed through real lookup"
                            )
                        max_error = max(max_error, check_output())
                        candidate_graph = torch.cuda.CUDAGraph()
                        stage = "oracle_capture"
                        with torch.cuda.graph(candidate_graph):
                            operation()
                    stage = "oracle_measure"
                    max_error = max(max_error, check_graph_inputs(candidate_graph))
                    oracle.append(
                        {
                            "runner": type(selected_runner).__name__,
                            "tactic": tactic,
                            "eager": measure_checked(operation, "eager"),
                            "cuda_graph": measure_checked(
                                candidate_graph.replay, "cuda_graph"
                            ),
                        }
                    )
            finally:
                winners.clear()
                winners.update(saved)
            if digest(args.cache) != cache_hash:
                raise RuntimeError(
                    "oracle unexpectedly changed persisted natural winners"
                )
        if profile_calls != startup_profile_calls:
            raise RuntimeError("profiling was attempted after startup")
        if eager_measurement_checks != 1 + len(
            oracle
        ) or changed_input_graph_checks != 1 + len(oracle):
            raise RuntimeError("not every natural/oracle execution was checked")

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
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "selected_device_ordinal": torch.cuda.current_device(),
            "physical_gpu_uuid": (
                str(props.uuid) if getattr(props, "uuid", None) is not None else None
            ),
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
        "timing_protocol": {
            "warmup_ms": args.warmup_ms,
            "min_sample_ms": args.min_sample_ms,
            "requested_iterations": args.iterations,
            "max_iterations": MAX_ITERATIONS,
        },
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
            "eager_checks_after_measurement": eager_measurement_checks,
            "changed_input_graph_checks": changed_input_graph_checks,
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
