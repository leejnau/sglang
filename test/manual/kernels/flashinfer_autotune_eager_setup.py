"""Diagnose recurring FP8 eager setup at fixed cached tactics.

This manual, version-specific operation experiment is not a serving benchmark or
a proposed implementation change. Benchmark workers run without instrumentation;
observation workers separately collect Python boundaries, native algorithm
identity and optional CPU/CUDA traces. Both use the existing FlashInfer tuner
lookup path with forced in-memory winners, not new tuning or managed stores.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import gzip
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
from unittest.mock import patch

MAX_ITERATIONS = 65536
EXPECTED_TUNER_SHA256 = (
    "e222f35f8b5a59a99a45fad627a79db8fc45fadc52b667e63b70c0968509109d"
)


def positive_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("benchmark", "observe"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup-ms", type=positive_float, default=500)
    parser.add_argument("--min-sample-ms", type=positive_float, default=20)
    parser.add_argument("--observation-calls", type=int, default=5)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    if (
        min(
            args.m,
            args.n,
            args.k,
            args.samples,
            args.iterations,
            args.observation_calls,
        )
        < 1
    ):
        parser.error("dimensions and counts must be positive")
    if any(d % 16 for d in (args.m, args.n, args.k)):
        parser.error("FP8 dimensions must be divisible by 16")
    if args.iterations > MAX_ITERATIONS:
        parser.error(f"--iterations must be at most {MAX_ITERATIONS}")
    if args.observation_calls > 1000:
        parser.error("--observation-calls must be at most 1000")
    if args.mode == "benchmark" and args.trace:
        parser.error("--trace is only permitted in the separate observe worker")
    return args


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


class BoundaryRecorder:
    """Instrumented CPU spans; exclusive means subtraction of wrapped children."""

    def __init__(self):
        self.frames = []
        self.records = []

    def wrap(self, name, function):
        def observed(*args, **kwargs):
            frame = {"children_ns": 0}
            self.frames.append(frame)
            start = time.perf_counter_ns()
            failed = True
            try:
                result = function(*args, **kwargs)
                failed = False
                return result
            finally:
                inclusive = time.perf_counter_ns() - start
                self.frames.pop()
                if self.frames:
                    self.frames[-1]["children_ns"] += inclusive
                self.records.append(
                    {
                        "boundary": name,
                        "inclusive_ns": inclusive,
                        "exclusive_ns": max(0, inclusive - frame["children_ns"]),
                        "failed": failed,
                    }
                )

        return observed

    def summary(self):
        result = {}
        for name in sorted({r["boundary"] for r in self.records}):
            rows = [r for r in self.records if r["boundary"] == name]
            result[name] = {
                "calls": len(rows),
                "failed_calls": sum(r["failed"] for r in rows),
                "inclusive_total_ns": sum(r["inclusive_ns"] for r in rows),
                "exclusive_total_ns": sum(r["exclusive_ns"] for r in rows),
                "inclusive_samples_ns": [r["inclusive_ns"] for r in rows],
                "exclusive_samples_ns": [r["exclusive_ns"] for r in rows],
            }
        return result


def measure(torch, call, args):
    """Uninstrumented synchronized wall batches; CUDA events include host gaps."""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    end_event.record()
    end_event.synchronize()

    def wall_batch(iterations):
        start = time.perf_counter()
        for _ in range(iterations):
            call()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000

    warmup_start = time.perf_counter()
    warmup_calls = 0
    while True:
        wall_batch(args.iterations)
        warmup_calls += args.iterations
        warmup_ms = (time.perf_counter() - warmup_start) * 1000
        if warmup_ms >= args.warmup_ms:
            break
    iterations = args.iterations
    calibration = []
    while True:
        elapsed = wall_batch(iterations)
        calibration.append({"iterations": iterations, "wall_ms": elapsed})
        if elapsed >= args.min_sample_ms:
            break
        if iterations == MAX_ITERATIONS:
            raise RuntimeError("calibration did not reach the minimum batch duration")
        iterations = min(iterations * 2, MAX_ITERATIONS)
    wall_batches, event_batches = [], []
    for _ in range(args.samples):
        start = time.perf_counter()
        start_event.record()
        for _ in range(iterations):
            call()
        end_event.record()
        end_event.synchronize()
        wall_batches.append((time.perf_counter() - start) * 1000)
        event_batches.append(start_event.elapsed_time(end_event))
    wall = [t / iterations for t in wall_batches]
    events = [t / iterations for t in event_batches]
    return {
        "samples_ms": wall,
        "gpu_event_samples_ms": events,
        "wall_batch_ms": wall_batches,
        "gpu_event_batch_ms": event_batches,
        "median_ms": statistics.median(wall),
        "requested_iterations": args.iterations,
        "effective_iterations": iterations,
        "iterations_per_sample": iterations,
        "warmup_actual_ms": warmup_ms,
        "warmup_calls": warmup_calls,
        "calibration_batches": calibration,
        "event_scope": "GPU event span including host submission gaps",
        "l2_condition": "reused hot buffers; no explicit L2 flush",
    }


def run(args):
    os.environ["FLASHINFER_AUTOTUNER_LOAD_FROM_FILE"] = "0"
    probe = getattr(ctypes.CDLL(None), "fi_probe_set_phase", None)
    if args.mode == "benchmark" and (os.environ.get("LD_PRELOAD") or probe):
        raise RuntimeError("benchmark workers must run without LD_PRELOAD/native hooks")
    if args.mode == "observe" and not probe:
        raise RuntimeError("observe workers require the native probe in LD_PRELOAD")
    if probe:
        probe.argtypes = [ctypes.c_char_p]
        probe.restype = None
        probe(None)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("this experiment requires a CUDA GPU")
    if torch.__version__ != "2.14.1+cu130" or torch.version.cuda != "13.0":
        raise RuntimeError("the pinned Torch 2.14.1+cu130/CUDA 13.0 stack is required")
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("this preregistered diagnostic targets an SM90 H100")

    import flashinfer
    from flashinfer.autotuner import AutoTuner, autotune
    from flashinfer.autotuner.autotuner import OptimizationProfile, StaticDim
    from flashinfer.gemm import gemm_base

    if flashinfer.__version__ != "0.7.2":
        raise RuntimeError("the audited FlashInfer 0.7.2 implementation is required")
    tuner_source = Path(inspect.getfile(AutoTuner))
    if digest(tuner_source) != EXPECTED_TUNER_SHA256:
        raise RuntimeError("FlashInfer tuner source differs from the audited wheel")
    tuner = AutoTuner.get()
    if tuner._winner_cache() or tuner._file_configs or tuner._active_managed_store:
        raise RuntimeError("fresh worker unexpectedly has tuner state")
    if tuner.is_tuning_mode:
        raise RuntimeError("fresh worker unexpectedly has tuning enabled")
    root = Path(__file__).resolve().parents[3]
    initial_gpu_status = gpu_status()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    a = torch.randn((1, args.m, args.k), device="cuda").to(torch.float8_e4m3fn)
    b = (
        torch.randn((1, args.n, args.k), device="cuda")
        .to(torch.float8_e4m3fn)
        .transpose(1, 2)
    )
    scale_a = torch.ones(1, device="cuda")
    scale_b = torch.ones(1, device="cuda")
    out = torch.empty((1, args.m, args.n), device="cuda", dtype=torch.bfloat16)
    reference = torch.bmm(a.float(), b.float()) * scale_a * scale_b
    profile_attempts = []

    def tensor_digest(tensor):
        raw = tensor.detach().contiguous().view(torch.uint8).cpu()
        return hashlib.sha256(bytes(raw.flatten().tolist())).hexdigest()

    operand_digests = {
        "a": tensor_digest(a),
        "b": tensor_digest(b),
        "scale_a": tensor_digest(scale_a),
        "scale_b": tensor_digest(scale_b),
    }

    def forbidden_profile(*_args, **_kwargs):
        profile_attempts.append("unexpected")
        raise RuntimeError("this diagnostic never permits profiling")

    def operation():
        return flashinfer.bmm_fp8(
            a, b, scale_a, scale_b, torch.bfloat16, out=out, backend="cublas"
        )

    correctness = []

    def check_output(label):
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), reference, rtol=0.02, atol=0.125)
        error = float((out.float() - reference).abs().max().item())
        correctness.append({"check": label, "max_abs_error": error})

    original_choose = tuner.choose_one
    original_search = tuner.search_cache
    captured = {}

    def capture_choose(op, runners, config, inputs, **kwargs):
        if op != "fp8_gemm" or len(runners) != 1:
            raise RuntimeError("unexpected operation or multiple runners")
        result = original_choose(op, runners, config, inputs, **kwargs)
        captured.update(op=op, runners=runners, config=config, inputs=inputs)
        return result

    with (
        patch.object(tuner, "_profile_single_kernel", forbidden_profile),
        autotune(False),
    ):
        with patch.object(tuner, "choose_one", capture_choose):
            operation()
        check_output("initial_untuned")
        if tuner._winner_cache() or tuner._file_configs:
            raise RuntimeError("initial untuned call populated winner state")
        selected_runner = captured["runners"][0]
        inputs = captured["inputs"]
        config = captured["config"]
        module = gemm_base.get_gemm_module()
        original_factory = module.cublas_fp8_gemm_runner
        if type(selected_runner).__name__ != "CublasFp8GemmRunner":
            raise RuntimeError("unexpected FP8 runner")

        def signature(tensors):
            return tuple(
                (
                    tuple(t.shape),
                    tuple(t.stride()),
                    str(t.dtype),
                    str(t.device),
                    t.data_ptr(),
                    t.storage_offset(),
                )
                for t in tensors
            )

        expected_signature = signature(inputs)
        runner_extras = selected_runner.get_cache_key_extras(inputs)
        shape_profile = OptimizationProfile(
            [[StaticDim(d) for d in t.shape] for t in inputs], [None] * len(inputs)
        )
        eligible = tuner._blocklist.filter(
            "fp8_gemm",
            selected_runner,
            selected_runner.get_valid_tactics(inputs, shape_profile),
        )
        if 0 not in eligible:
            raise RuntimeError("ordinal zero is not eligible at the runtime shape")
        donor_cache = selected_runner._algo_cache.copy()
        donor_key = selected_runner._algo_cache_key(inputs)
        if len(donor_cache) != 1 or donor_key not in donor_cache:
            raise RuntimeError("unexpected donor algorithm-cache layout")
        algo_buffer, algo_count = donor_cache[donor_key]
        if algo_count <= 0 or algo_buffer.device.type != "cpu":
            raise RuntimeError("donor algorithm enumeration did not return CPU choices")
        donor_buffer_sha = hashlib.sha256(bytes(algo_buffer.tolist())).hexdigest()
        expected_hash = hash(selected_runner)
        cache_key = tuner._get_cache_key(
            "fp8_gemm",
            selected_runner,
            tuple(tuple(t.shape) for t in inputs),
            config,
            runner_extras,
        )
        winners = tuner._winner_cache()
        saved_winners = winners.copy()
        retained = original_factory()
        retained.get_valid_tactics(inputs, shape_profile)
        if hash(retained) != expected_hash:
            raise RuntimeError("algorithm caches changed runner identity")
        if retained._algo_cache[donor_key][1] != algo_count:
            raise RuntimeError("independent retained runner returned a different count")

        # Only an exact fixed workload may reuse these CPU descriptors. This is
        # not a reusable cache design: tensor identities, layouts, device,
        # workspace and extras are checked before and after every variant.
        def check_identity():
            if signature(inputs) != expected_signature:
                raise RuntimeError("fixed-workload tensor identity or layout changed")
            if selected_runner.get_cache_key_extras(inputs) != runner_extras:
                raise RuntimeError("runner cache-key extras changed")
            if hash(selected_runner) != expected_hash:
                raise RuntimeError("runner identity changed")
            if (
                hashlib.sha256(bytes(algo_buffer.tolist())).hexdigest()
                != donor_buffer_sha
            ):
                raise RuntimeError("donor algorithm bytes were modified")

        def seeded_factory():
            fresh = original_factory()
            if type(fresh).__name__ != "CublasFp8GemmRunner" or fresh._algo_cache:
                raise RuntimeError("unexpected fresh runner/cache during intervention")
            fresh._algo_cache.update(donor_cache)
            return fresh

        @contextlib.contextmanager
        def variant_scope(variant):
            check_identity()
            winners.clear()
            winners[cache_key] = (variant["tactic"], None)
            factory_context = (
                patch.object(module, "cublas_fp8_gemm_runner", seeded_factory)
                if variant["path"] == "public_seeded_algorithms"
                else contextlib.nullcontext()
            )
            try:
                with factory_context:
                    yield (
                        (lambda: retained(inputs=inputs, tactic=variant["tactic"]))
                        if variant["path"] == "retained_runner"
                        else operation
                    )
            finally:
                winners.clear()
                winners.update(saved_winners)
                check_identity()
                if module.cublas_fp8_gemm_runner is not original_factory:
                    raise RuntimeError("runner-factory patch was not restored")

        lookup_checks = []

        def check_lookup(call, variant, label):
            observations = []
            hit = False

            def search_hook(*args_, **kwargs_):
                nonlocal hit
                result = original_search(*args_, **kwargs_)
                hit = bool(result[0])
                return result

            def choose_hook(op, runners, config_, inputs_, **kwargs_):
                nonlocal hit
                hit = False
                chosen, tactic = original_choose(
                    op, runners, config_, inputs_, **kwargs_
                )
                if not hit or tactic != variant["tactic"]:
                    raise RuntimeError(
                        "public operation did not consume the forced winner"
                    )
                if signature(inputs_) != expected_signature:
                    raise RuntimeError("public API changed the fixed-workload inputs")
                if chosen.get_cache_key_extras(inputs_) != runner_extras:
                    raise RuntimeError("public API changed runner cache-key extras")
                observations.append({"cache_hit": hit, "tactic": int(tactic)})
                return chosen, tactic

            with (
                patch.object(tuner, "choose_one", choose_hook),
                patch.object(tuner, "search_cache", search_hook),
            ):
                call()
            if variant["path"] != "retained_runner" and len(observations) != 1:
                raise RuntimeError("missing actual tuner lookup observation")
            if variant["path"] == "retained_runner" and observations:
                raise RuntimeError("retained runner unexpectedly called the tuner")
            lookup_checks.append(
                {"variant": variant["name"], "stage": label, "lookups": observations}
            )

        def check_graph(graph, label):
            nonlocal reference
            saved_a, saved_reference = a.clone(), reference
            try:
                a.zero_()
                reference = torch.zeros_like(reference)
                out.fill_(float("nan"))
                graph.replay()
                check_output(label + "_changed_input")
            finally:
                a.copy_(saved_a)
                reference = saved_reference
            out.fill_(float("nan"))
            graph.replay()
            check_output(label + "_restored_input")

        variants = [
            {
                "name": path + ("_default" if tactic == -1 else "_ordinal0"),
                "path": path,
                "tactic": tactic,
            }
            for path in ("public_cached", "retained_runner", "public_seeded_algorithms")
            for tactic in (-1, 0)
        ]
        random.Random(args.seed).shuffle(variants)
        measurements, observations, traces = [], [], []
        try:
            for variant in variants:
                with variant_scope(variant) as call:
                    check_lookup(call, variant, "before")
                    check_output(variant["name"] + "_before")
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        check_lookup(call, variant, "capture")
                    check_graph(graph, variant["name"])
                    if args.mode == "benchmark":
                        timed = dict(variant)
                        for mode, invoke in (
                            ("eager", call),
                            ("cuda_graph", graph.replay),
                        ):
                            out.fill_(float("nan"))
                            timed[mode] = measure(torch, invoke, args)
                            check_output(variant["name"] + "_measured_" + mode)
                        measurements.append(timed)
                    else:
                        observations.append(
                            observe_variant(
                                torch,
                                tuner,
                                module,
                                original_factory,
                                call,
                                graph,
                                variant,
                                args,
                                probe,
                                check_graph,
                            )
                        )
                        check_output(variant["name"] + "_observed")
                        if args.trace:
                            traces.append(
                                trace_variant(torch, call, graph, variant, args)
                            )
                            check_output(variant["name"] + "_traced")
                    check_lookup(call, variant, "after")
                    check_output(variant["name"] + "_after")
        finally:
            winners.clear()
            winners.update(saved_winners)
            if probe:
                probe(None)
        if profile_attempts or tuner.is_tuning_mode or tuner._active_managed_store:
            raise RuntimeError(
                "unexpected profiling/tuning/store state after diagnostic"
            )
        check_identity()
        if {
            "a": tensor_digest(a),
            "b": tensor_digest(b),
            "scale_a": tensor_digest(scale_a),
            "scale_b": tensor_digest(scale_b),
        } != operand_digests:
            raise RuntimeError("input operands or scale contents changed")

    props = torch.cuda.get_device_properties(0)
    return {
        "status": "success",
        "schema_version": 1,
        "mode": args.mode,
        "pid": os.getpid(),
        "seed": args.seed,
        "shape": [args.m, args.n, args.k],
        "worker_sha256": digest(__file__),
        "environment": {
            "sglang_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "sglang_wrapper_sha256": digest(
                root / "python/sglang/srt/model_executor/runner/flashinfer_autotune.py"
            ),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "flashinfer": flashinfer.__version__,
            "flashinfer_tuner_sha256": digest(tuner_source),
            "flashinfer_gemm_source_sha256": digest(inspect.getfile(gemm_base)),
            "gpu": props.name,
            "physical_gpu_uuid": str(getattr(props, "uuid", None)),
            "capability": list(torch.cuda.get_device_capability()),
            "gpu_status_before": initial_gpu_status,
            "gpu_status_after": gpu_status(),
            "LD_PRELOAD": os.environ.get("LD_PRELOAD"),
            "native_probe_available": bool(probe),
            "native_probe_output": os.environ.get("FI_NATIVE_PROBE_OUTPUT"),
            "backend": "cublas",
            "measurement_policy": "unspecified; no tuning occurs",
            "bundled_configs": False,
            "scope": "single operation on one GPU; no model/server",
        },
        "timing_protocol": {
            "warmup_ms": args.warmup_ms,
            "min_sample_ms": args.min_sample_ms,
            "samples": args.samples,
            "requested_iterations": args.iterations,
            "max_iterations": MAX_ITERATIONS,
            "variant_order": [v["name"] for v in variants],
            "benchmark_instrumentation": (
                "none; profiling guard is never called"
                if args.mode == "benchmark"
                else "separate observation worker; no latency estimates reported"
            ),
        },
        "intervention": {
            "scope": "diagnostic only; fresh public runners receive an equivalent retained runner's already-enumerated CPU algorithm list",
            "guards": "fixed exact tensor pointers/shapes/strides/dtypes/devices/storage offsets including scales/output/workspace; runner hash/extras and immutable donor bytes checked before and after each variant",
            "timing_qualification": "seeded factory still constructs a fresh runner and copies one cache entry; retained runner bypasses the public API and tuner",
            "algorithm_count": int(algo_count),
            "donor_cpu_buffer_sha256": donor_buffer_sha,
            "donor_hash_scope": "whole-buffer immutability guard within one worker; unused slots are uninitialized, so hashes must not be compared across workers as algorithm identities",
            "donor_cpu_buffer_bytes": algo_buffer.numel(),
            "runner_extras": [str(value) for value in runner_extras],
            "runtime_eligible_tactics": [int(value) for value in eligible],
            "input_signature": [list(row) for row in expected_signature],
            "operand_sha256": operand_digests,
            "scales": [float(scale_a.item()), float(scale_b.item())],
        },
        "profile_attempts": len(profile_attempts),
        "lookup_checks": lookup_checks,
        "measurements": measurements,
        "observations": observations,
        "traces": traces,
        "correctness": {
            "passed": True,
            "rtol": 0.02,
            "atol": 0.125,
            "checks": correctness,
        },
    }


def observe_variant(
    torch,
    tuner,
    module,
    original_factory,
    call,
    graph,
    variant,
    args,
    probe,
    check_graph,
):
    """Short perturbed pass: boundary timings never enter benchmark summaries."""
    recorder = BoundaryRecorder()
    active_factory = module.cublas_fp8_gemm_runner
    closure = dict(
        zip(original_factory.__code__.co_freevars, original_factory.__closure__)
    )
    native_cell = closure.get("module")
    if native_cell is None:
        raise RuntimeError("audited runner factory module closure was not found")
    native_module = native_cell.cell_contents

    class ObservedModule:
        bmm_fp8_get_algos = staticmethod(
            recorder.wrap(
                "native_binding_get_algorithms", native_module.bmm_fp8_get_algos
            )
        )

        def __getattr__(self, name):
            return getattr(native_module, name)

    def observed_factory():
        runner = recorder.wrap(
            "runner_factory_including_class_definition_and_init", active_factory
        )()
        runner.__class__._get_algos = recorder.wrap(
            "runner_get_algorithms", runner.__class__._get_algos
        )
        return runner

    phase_boundaries = []
    observation_graph = torch.cuda.CUDAGraph()
    try:
        native_cell.cell_contents = ObservedModule()
        with (
            patch.object(module, "cublas_fp8_gemm_runner", observed_factory),
            patch.object(
                tuner, "choose_one", recorder.wrap("tuner_choose_one", tuner.choose_one)
            ),
        ):
            for execution in ("eager", "capture", "graph_replay"):
                label = variant["name"] + "_" + execution
                recorder.records.clear()
                if probe:
                    probe(label.encode())
                try:
                    if execution == "capture":
                        with torch.cuda.graph(observation_graph):
                            call()
                    else:
                        invoke = (
                            call if execution == "eager" else observation_graph.replay
                        )
                        for _ in range(args.observation_calls):
                            invoke()
                    torch.cuda.synchronize()
                finally:
                    if probe:
                        probe(None)
                phase_boundaries.append(
                    {
                        "phase": label,
                        "execution": execution,
                        "calls": 1
                        if execution == "capture"
                        else args.observation_calls,
                        "cpu_boundaries": recorder.summary(),
                    }
                )
    finally:
        native_cell.cell_contents = native_module
    check_graph(observation_graph, variant["name"] + "_observation_graph")
    return {
        **variant,
        "calls_per_execution": args.observation_calls,
        "phase_boundaries": phase_boundaries,
        "scope": "instrumented CPU boundary observations; not uninstrumented latency estimates; native durations are nested and must not be added to Python boundaries",
        "constructor_scope": "runner factory includes class definition and __init__; constructor alone is not isolated",
        "retained_scope": "retained runner construction and enumeration happened before observation; class methods were not patched on that retained object",
    }


def trace_variant(torch, call, graph, variant, args):
    trace_stem = args.trace.name.removesuffix(".gz").removesuffix(".json")
    trace_path = args.trace.with_name(trace_stem + "-" + variant["name"] + ".json")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as profiler:
        for execution, invoke in (("eager", call), ("graph_replay", graph.replay)):
            with torch.profiler.record_function(variant["name"] + "/" + execution):
                for _ in range(args.observation_calls):
                    invoke()
                torch.cuda.synchronize()
    profiler.export_chrome_trace(str(trace_path))
    packed = trace_path.with_suffix(".json.gz")
    with trace_path.open("rb") as source, packed.open("wb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as output:
            output.write(source.read())
    trace_path.unlink()
    return {"variant": variant["name"], "path": str(packed), "sha256": digest(packed)}


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
