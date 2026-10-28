"""Qualify one upstream delay binding change before the frozen MLP study.

The ordering probe is a separate process. MLP mode only attests the loaded
source/native providers and delegates to the unchanged historical worker.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import marshal
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from types import CodeType

UPSTREAM_HEAD = "e6bcdebdefff9ad6dd635cb357ead72b0e7f9da6"
MLP_SHA = "a8532276afc70ae1b70e12eaf9fc722fa7b29ed6102407360dd93708a7789157"
NATIVE_HASHES = {
    "gemm": "72690cacca4ce1d69ccaa6b9190475483d984bd58de0962d06c874c801c678d6",
    "trtllm_utils": "e442a689064fe3af4ab46510b2d5aaf3597b072949b84f13d14e9e53352109e4",
}
_BASELINE = b"""import functools

from .jit.tllm_utils import gen_trtllm_utils_module


@functools.cache
def get_trtllm_utils_module():
    return gen_trtllm_utils_module().build_and_load()


def delay_kernel(stream_delay_micro_secs: int):
    get_trtllm_utils_module().delay_kernel(stream_delay_micro_secs)
"""
_PATCHED = b'''import functools

import tvm_ffi

from .jit.tllm_utils import gen_trtllm_utils_module


@functools.cache
def get_trtllm_utils_module():
    return gen_trtllm_utils_module().build_and_load()


def delay_kernel(stream_delay_micro_secs: int):
    """Enqueue a delay on the current PyTorch CUDA stream."""
    # This scalar-only FFI call cannot infer Torch's stream from tensor arguments.
    # Bind it explicitly so the delay precedes the autotuner's timing events.
    with tvm_ffi.use_torch_stream():
        get_trtllm_utils_module().delay_kernel(stream_delay_micro_secs)
'''


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wrapper_source(arm):
    require(arm in ("baseline", "patched"), "unknown binding arm")
    return _BASELINE if arm == "baseline" else _PATCHED


def wrapper_sha(arm):
    return hashlib.sha256(wrapper_source(arm)).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        json.dump(data, output, indent=2, allow_nan=False)
        output.write("\n")


def attest(arm):
    import objective_values_bootstrap as bootstrap
    import torch
    import tvm_ffi
    from flashinfer import tllm_utils
    from flashinfer.autotuner import autotuner

    source = Path(tllm_utils.__file__).resolve()
    require(
        source == Path(bootstrap.source_root).resolve() / "flashinfer/tllm_utils.py"
        and source.read_bytes() == wrapper_source(arm),
        "loaded delay wrapper source differs",
    )
    compiled = compile(source.read_bytes(), str(source), "exec", dont_inherit=True)
    code = next(
        item
        for item in compiled.co_consts
        if isinstance(item, CodeType) and item.co_name == "delay_kernel"
    )
    require(
        marshal.dumps(code) == marshal.dumps(tllm_utils.delay_kernel.__code__)
        and autotuner.delay_kernel is tllm_utils.delay_kernel,
        "delay callable or profiler import differs from loaded source",
    )
    require(
        tllm_utils.get_trtllm_utils_module() is bootstrap.native_delay,
        "delay native provider was replaced",
    )
    native = {}
    for name, expected in NATIVE_HASHES.items():
        row = bootstrap.evidence["native"]["providers"][name]
        require(
            row["library_sha256"] == expected and sha(row["library_path"]) == expected,
            "native provider changed: " + name,
        )
        native[name] = {key: row[key] for key in ("library_path", "library_sha256")}
    package = Path(tvm_ffi.__file__).resolve().parent
    libraries = sorted(
        {
            Path(module.__file__).resolve()
            for name, module in list(sys.modules.items())
            if name.startswith("tvm_ffi")
            and getattr(module, "__file__", None)
            and ".so" in Path(module.__file__).name
        }
    )
    maps = Path("/proc/self/maps")
    if maps.is_file():
        for line in maps.read_text().splitlines():
            path = line.split()[-1]
            if path.startswith(str(package)) and ".so" in path:
                libraries.append(Path(path).resolve())
    stream_source = Path(inspect.getfile(tvm_ffi.use_torch_stream)).resolve()
    torch_stream = torch.cuda.current_stream()
    device = tvm_ffi.device(str(torch_stream.device))
    return {
        "arm": arm,
        "upstream_head": UPSTREAM_HEAD,
        "wrapper_path": str(source),
        "wrapper_sha256": sha(source),
        "callable_matches_source": True,
        "profiler_alias_matches": True,
        "native_identity_passthrough": True,
        "native": native,
        "torch_version": torch.__version__,
        "tvm_ffi_version": importlib.metadata.version("apache-tvm-ffi"),
        "ffi_init": {
            "path": str(package / "__init__.py"),
            "sha256": sha(package / "__init__.py"),
        },
        "ffi_stream_source": {"path": str(stream_source), "sha256": sha(stream_source)},
        "ffi_stream_function_sha256": hashlib.sha256(
            inspect.getsource(tvm_ffi.use_torch_stream).encode()
        ).hexdigest(),
        "ffi_libraries": [
            {"path": str(path), "sha256": sha(path)} for path in sorted(set(libraries))
        ],
        "device": str(torch_stream.device),
        "torch_stream": torch_stream.cuda_stream,
        "ambient_ffi_stream": tvm_ffi.get_raw_stream(device),
    }


def summarize(samples, reference_ms):
    require(len(samples) == 5, "ordering probe requires five samples")
    for row in samples:
        require(
            all(
                isinstance(row[key], (int, float))
                and math.isfinite(row[key])
                and row[key] >= 0
                for key in ("event_ms", "host_ms", "residual_sync_ms")
            ),
            "invalid ordering sample",
        )
    medians = {
        key: statistics.median(row[key] for row in samples)
        for key in ("event_ms", "host_ms", "residual_sync_ms")
    }
    require(
        math.isfinite(reference_ms) and reference_ms > 0, "invalid native reference"
    )
    return {
        "medians": medians,
        "ordering_observed": medians["event_ms"] >= reference_ms / 2,
        "reference_ms": reference_ms,
    }


def ordering(result):
    import torch
    import tvm_ffi
    from flashinfer import tllm_utils
    from flashinfer.autotuner import AutoTuner, TuningConfig

    arm = result["arm"]
    native = tllm_utils.get_trtllm_utils_module()
    streams = {
        "default": torch.cuda.default_stream(),
        "nondefault": torch.cuda.Stream(),
    }
    other = torch.cuda.Stream()
    device = tvm_ffi.device(str(other.device))
    original_ffi = tvm_ffi.get_raw_stream(device)

    def window(stream, call):
        torch.cuda.synchronize()
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        with torch.cuda.stream(stream):
            start.record()
            begin = time.perf_counter_ns()
            call()
            end.record()
            end.synchronize()
            host_ms = (time.perf_counter_ns() - begin) / 1e6
        event_ms = start.elapsed_time(end)
        begin = time.perf_counter_ns()
        torch.cuda.synchronize()
        return {
            "event_ms": event_ms,
            "host_ms": host_ms,
            "residual_sync_ms": (time.perf_counter_ns() - begin) / 1e6,
        }

    result.update(reference=[], controlled_streams=[], natural_streams=[])
    for label, stream in streams.items():
        with torch.cuda.stream(stream), tvm_ffi.use_torch_stream():
            native.delay_kernel(1)
            stream.synchronize()
            reference = [
                window(stream, lambda: native.delay_kernel(5000)) for _ in range(5)
            ]
        reference_ms = statistics.median(row["event_ms"] for row in reference)
        result["reference"].append(
            {"stream_kind": label, "samples": reference, "median_ms": reference_ms}
        )
        natural = [
            window(stream, lambda: tllm_utils.delay_kernel(5000)) for _ in range(5)
        ]
        result["natural_streams"].append(
            {
                "stream_kind": label,
                "torch_stream": stream.cuda_stream,
                "ambient_ffi_stream": original_ffi,
                "forced_conflict": False,
                "samples": natural,
                **summarize(natural, reference_ms),
            }
        )
        with tvm_ffi.use_raw_stream(device, other.cuda_stream):
            samples = [
                window(stream, lambda: tllm_utils.delay_kernel(5000)) for _ in range(5)
            ]
            restored = tvm_ffi.get_raw_stream(device) == other.cuda_stream
        require(
            restored and tvm_ffi.get_raw_stream(device) == original_ffi,
            "FFI context leaked",
        )
        result["controlled_streams"].append(
            {
                "stream_kind": label,
                "torch_stream": stream.cuda_stream,
                "ambient_ffi_stream": other.cuda_stream,
                "forced_conflict": True,
                "restored": restored,
                "samples": samples,
                **summarize(samples, reference_ms),
            }
        )

    # Observation only: use the real profiling method with its untouched timer,
    # no forced FFI context, and a cheap synthetic callable in this process.
    stream = streams["nondefault"]
    tuner = AutoTuner.get()
    tuner.warmup, tuner.repeat, tuner.stream_delay_micro_secs = 3, 10, 5000
    require(tuner._effective_measure_policy is None, "unexpected measurement policy")
    config = TuningConfig(use_cuda_graph=False, use_cold_l2_cache=False)
    inputs = [
        torch.ones(32, device=stream.device),
        torch.empty(32, device=stream.device),
    ]

    def runner(tensors, tactic):
        return torch.add(tensors[0], 1, out=tensors[1])

    profile_rows = []
    for _ in range(5):
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            start = time.perf_counter_ns()
            measured = tuner._profile_single_kernel(
                runner, inputs, -1, config, input_tensor_batches=[inputs]
            )
            host_ms = (time.perf_counter_ns() - start) / 1e6
        start = time.perf_counter_ns()
        torch.cuda.synchronize()
        residual = (time.perf_counter_ns() - start) / 1e6
        torch.testing.assert_close(inputs[1], torch.full_like(inputs[1], 2))
        profile_rows.append(
            {
                "profiler_ms": measured,
                "host_ms": host_ms,
                "residual_sync_ms": residual,
                "numerics": True,
            }
        )
    result["actual_profiler"] = {
        "forced_conflict": False,
        "torch_stream": stream.cuda_stream,
        "ambient_ffi_stream": original_ffi,
        "samples": profile_rows,
        "settings": {
            "warmup": 3,
            "repeat": 10,
            "stream_delay_micro_secs": 5000,
            "measurement_policy": None,
            "use_cuda_graph": False,
            "cold_l2": False,
        },
    }
    result["graph_replays"] = []
    if arm == "patched":
        output = torch.zeros(1, device=stream.device)
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with tvm_ffi.use_raw_stream(device, other.cuda_stream):
            with torch.cuda.graph(graph, stream=stream):
                tllm_utils.delay_kernel(5000)
                output.add_(1)
            require(
                tvm_ffi.get_raw_stream(device) == other.cuda_stream,
                "graph FFI context leaked",
            )
        for value in (3.0, 9.0):
            with torch.cuda.stream(stream):
                output.fill_(value)
            row = window(stream, graph.replay)
            torch.testing.assert_close(output, torch.full_like(output, value + 1))
            result["graph_replays"].append(
                {
                    **row,
                    "input": value,
                    "output": output.item(),
                    "numerics": True,
                    "restored": True,
                }
            )
    else:
        result["graph_note"] = (
            "Not attempted with the intentionally misbound wrapper: capture can invalidate the CUDA context. The two eager stream cases are the negative controls."
        )
    torch.cuda.synchronize()


def validate_ordering(result):
    arm = result["arm"]
    require(arm in ("baseline", "patched"), "unknown ordering arm")
    references = {}
    require(len(result["reference"]) == 2, "missing native reference")
    for row in result["reference"]:
        reference = summarize(row["samples"], row["median_ms"])
        require(
            reference["medians"]["event_ms"] == row["median_ms"],
            "native reference median differs",
        )
        references[row["stream_kind"]] = row["median_ms"]
    require(
        set(references) == {"default", "nondefault"}, "native reference streams differ"
    )
    require(len(result["controlled_streams"]) == 2, "missing stream case")
    require(
        {r["stream_kind"] for r in result["controlled_streams"]}
        == {"default", "nondefault"},
        "stream cases differ",
    )
    for row in result["controlled_streams"]:
        require(
            row["reference_ms"] == references[row["stream_kind"]],
            "stream reference differs",
        )
        observed = summarize(row["samples"], row["reference_ms"])
        require(
            row["restored"]
            and row["forced_conflict"]
            and row["torch_stream"] != row["ambient_ffi_stream"],
            "invalid conflicting-stream control",
        )
        require(
            observed["ordering_observed"] == (arm == "patched"),
            "old-wrapper defect not reproduced or correction failed",
        )
    require(
        len(result["natural_streams"]) == 2
        and all(not row["forced_conflict"] for row in result["natural_streams"]),
        "natural observations missing",
    )
    require(
        {r["stream_kind"] for r in result["natural_streams"]}
        == {"default", "nondefault"},
        "natural streams differ",
    )
    for row in result["natural_streams"]:
        require(
            row["reference_ms"] == references[row["stream_kind"]],
            "natural reference differs",
        )
        observed = summarize(row["samples"], row["reference_ms"])
        if arm == "patched":
            require(observed["ordering_observed"], "natural stream binding failed")
    require(
        len(result["actual_profiler"]["samples"]) == 5
        and not result["actual_profiler"]["forced_conflict"],
        "actual profiler diagnostic missing",
    )
    require(
        all(
            row["numerics"]
            and math.isfinite(row["profiler_ms"])
            and row["profiler_ms"] > 0
            for row in result["actual_profiler"]["samples"]
        ),
        "profiler diagnostic invalid",
    )
    if arm == "patched":
        reference = references["nondefault"]
        samples = result["actual_profiler"]["samples"]
        require(
            statistics.median(row["host_ms"] for row in samples) >= reference / 2
            and statistics.median(row["residual_sync_ms"] for row in samples)
            < reference / 2,
            "actual profiler returned before its delay completed",
        )
        replays = result["graph_replays"]
        require(
            len(replays) == 2 and [row["input"] for row in replays] == [3.0, 9.0],
            "graph replay inputs differ",
        )
        require(
            all(
                row["event_ms"] >= reference / 2
                and row["output"] == row["input"] + 1
                and row["numerics"]
                and row["restored"]
                for row in replays
            ),
            "graph ordering or numerics failed",
        )
    for witness in (result["attestation_before"], result["attestation_after"]):
        require(
            witness["arm"] == arm
            and witness["upstream_head"] == UPSTREAM_HEAD
            and witness["wrapper_sha256"] == wrapper_sha(arm)
            and witness["callable_matches_source"]
            and witness["profiler_alias_matches"]
            and witness["native_identity_passthrough"],
            "invalid wrapper witness",
        )
        require(
            {name: row["library_sha256"] for name, row in witness["native"].items()}
            == NATIVE_HASHES,
            "native witness changed",
        )
    require(
        result["attestation_before"] == result["attestation_after"],
        "binding changed during ordering",
    )


def check_ordering(raw_root):
    records = {}
    for arm in ("baseline", "patched"):
        output = raw_root / "ordering" / (arm + ".json")
        result = json.loads(output.read_text())
        require(
            result["status"] == "passed" and result["arm"] == arm,
            "ordering worker failed",
        )
        validate_ordering(result)
        bootstrap_path = output.with_suffix(".bootstrap.json")
        bootstrap = json.loads(bootstrap_path.read_text())
        require(
            bootstrap["status"] == "passed"
            and bootstrap["result_sha256"] == sha(output)
            and bootstrap["pid"] == result["pid"]
            and bootstrap["binding_arm"] == arm
            and bootstrap["binding_before"] == result["attestation_before"]
            and bootstrap["binding_after"] == result["attestation_after"],
            "ordering bootstrap differs",
        )
        records[arm] = {
            "result_sha256": sha(output),
            "bootstrap_sha256": sha(bootstrap_path),
            "pid": result["pid"],
            "attestation": result["attestation_before"],
        }
    # Container-local PIDs can both be 1. The launch ledger and distinct retained
    # container IDs establish process isolation outside this numerical gate.
    for field in (
        "native",
        "torch_version",
        "tvm_ffi_version",
        "ffi_init",
        "ffi_stream_source",
        "ffi_stream_function_sha256",
        "ffi_libraries",
        "device",
    ):
        require(
            records["baseline"]["attestation"][field]
            == records["patched"]["attestation"][field],
            "ordering arms changed dependency: " + field,
        )
    return {
        "schema_version": 1,
        "status": "passed",
        "upstream_head": UPSTREAM_HEAD,
        "arms": records,
        "limits": "Forced conflict reproduces susceptibility; natural observations do not retroactively identify the historical binding or prove the ranking cause.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("ordering", "mlp", "check-ordering"))
    parser.add_argument("--arm", choices=("baseline", "patched"))
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args, tail = parser.parse_known_args(argv)
    require(not args.output.exists(), "refusing to overwrite result")
    if args.command == "check-ordering":
        require(
            args.raw_root is not None and not tail, "invalid ordering gate arguments"
        )
        write_json(args.output, check_ordering(args.raw_root))
        return 0
    require(args.arm is not None, "binding arm required")
    result = {
        "schema_version": 1,
        "status": "incomplete",
        "arm": args.arm,
        "pid": os.getpid(),
        "upstream_head": UPSTREAM_HEAD,
        "started_ns": time.time_ns(),
    }
    binding_path = args.output.with_suffix(".binding.json")
    try:
        result["attestation_before"] = attest(args.arm)
        if args.command == "ordering":
            require(not tail, "unexpected ordering arguments")
            ordering(result)
            result["attestation_after"] = attest(args.arm)
            validate_ordering(result)
        else:
            require(not binding_path.exists(), "refusing to overwrite binding witness")
            import bench_flashinfer_mlp_confirmation as inherited

            require(sha(inherited.__file__) == MLP_SHA, "frozen MLP worker changed")
            if tail[:1] == ["--"]:
                tail = tail[1:]
            require("--output" not in tail, "duplicate inherited output")
            result["worker_sha256"] = MLP_SHA
            code = inherited.main([*tail, "--output", str(args.output)])
            result["attestation_after"] = attest(args.arm)
            require(
                result["attestation_before"] == result["attestation_after"],
                "binding changed during MLP worker",
            )
            result["result_sha256"] = sha(args.output)
            require(code == 0, "inherited MLP worker failed")
        result["status"] = "passed"
    except Exception as error:
        result.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    result["completed_ns"] = time.time_ns()
    write_json(binding_path if args.command == "mlp" else args.output, result)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
