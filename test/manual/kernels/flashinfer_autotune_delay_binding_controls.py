"""CPU controls for the manual binding qualification; no GPU claims."""

from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import FunctionType, ModuleType, SimpleNamespace
from unittest.mock import patch

import flashinfer_autotune_delay_binding as binding


def sample(event=5.0, host=5.1, residual=0.01):
    return {"event_ms": event, "host_ms": host, "residual_sync_ms": residual}


def fixture(arm):
    witness = {
        "arm": arm,
        "upstream_head": binding.UPSTREAM_HEAD,
        "wrapper_sha256": binding.wrapper_sha(arm),
        "callable_matches_source": True,
        "profiler_alias_matches": True,
        "native_identity_passthrough": True,
        "native": {
            name: {"library_sha256": value}
            for name, value in binding.NATIVE_HASHES.items()
        },
    }
    result = {
        "arm": arm,
        "attestation_before": witness,
        "attestation_after": copy.deepcopy(witness),
        "reference": [],
        "controlled_streams": [],
        "natural_streams": [],
        "graph_replays": [],
    }
    for label in ("default", "nondefault"):
        result["reference"].append(
            {
                "stream_kind": label,
                "samples": [sample() for _ in range(5)],
                "median_ms": 5.0,
            }
        )
        row = {
            "stream_kind": label,
            "torch_stream": 1,
            "ambient_ffi_stream": 2,
            "forced_conflict": True,
            "restored": True,
            "reference_ms": 5.0,
            "samples": [
                sample() if arm == "patched" else sample(0.02, 0.03, 5.0)
                for _ in range(5)
            ],
        }
        result["controlled_streams"].append(row)
        natural = copy.deepcopy(row)
        natural["forced_conflict"] = False
        result["natural_streams"].append(natural)
    result["actual_profiler"] = {
        "forced_conflict": False,
        "samples": [
            {
                "profiler_ms": 0.002,
                "host_ms": 5.1 if arm == "patched" else 0.1,
                "residual_sync_ms": 0.01 if arm == "patched" else 5.0,
                "numerics": True,
            }
            for _ in range(5)
        ],
    }
    if arm == "patched":
        result["graph_replays"] = [
            {
                **sample(),
                "input": value,
                "output": value + 1,
                "numerics": True,
                "restored": True,
            }
            for value in (3.0, 9.0)
        ]
    return result


def wrapper_control(arm, fail):
    state = {"stream": "ambient", "observed": []}

    @contextmanager
    def use_torch_stream():
        previous = state["stream"]
        state["stream"] = "torch"
        try:
            yield
        finally:
            state["stream"] = previous

    def native_delay(value):
        state["observed"].append((state["stream"], value))
        if fail:
            raise ValueError("native failure")

    native = SimpleNamespace(delay_kernel=native_delay)
    jit = ModuleType("delay_fixture.jit.tllm_utils")
    jit.gen_trtllm_utils_module = lambda: SimpleNamespace(build_and_load=lambda: native)
    ffi = ModuleType("tvm_ffi")
    ffi.use_torch_stream = use_torch_stream
    namespace = {"__name__": "delay_fixture.tllm_utils", "__package__": "delay_fixture"}
    with patch.dict(
        "sys.modules", {"delay_fixture.jit.tllm_utils": jit, "tvm_ffi": ffi}
    ):
        exec(
            compile(
                binding.wrapper_source(arm), "wrapper.py", "exec", dont_inherit=True
            ),
            namespace,
        )
        try:
            namespace["delay_kernel"](5000)
            assert not fail
        except ValueError as error:
            assert fail and str(error) == "native failure"
    assert state["stream"] == "ambient"
    assert state["observed"] == [("torch" if arm == "patched" else "ambient", 5000)]
    assert namespace["get_trtllm_utils_module"]() is native


def callable_control(arm):
    jit = ModuleType("delay_fixture.jit.tllm_utils")
    jit.gen_trtllm_utils_module = lambda: None
    ffi = ModuleType("tvm_ffi")
    with (
        tempfile.TemporaryDirectory() as directory,
        patch.dict(
            "sys.modules", {"delay_fixture.jit.tllm_utils": jit, "tvm_ffi": ffi}
        ),
    ):
        source = Path(directory) / "tllm_utils.py"
        source.write_bytes(binding.wrapper_source(arm))
        spec = importlib.util.spec_from_file_location(
            "delay_fixture.tllm_utils", source
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        function = module.delay_kernel
        assert all(binding.verify_delay_callable(source, function, function).values())

        changed = binding.wrapper_source(arm).replace(
            b".delay_kernel(stream_delay_micro_secs)",
            b".delay_kernel(stream_delay_micro_secs + 1)",
        )
        namespace = {
            "__name__": "delay_fixture.changed",
            "__package__": "delay_fixture",
        }
        exec(compile(changed, str(source), "exec", dont_inherit=True), namespace)
        other_function = namespace["delay_kernel"]
        same_code_alias = FunctionType(function.__code__, function.__globals__)
        for actual, alias, failed_field in (
            (other_function, other_function, "code_matches"),
            (function, same_code_alias, "profiler_alias_matches"),
        ):
            try:
                binding.verify_delay_callable(source, actual, alias)
            except RuntimeError as error:
                assert f'"{failed_field}": false' in str(error)
            else:
                raise AssertionError("modified callable accepted: " + failed_field)


def main():
    for arm in ("baseline", "patched"):
        callable_control(arm)
        for fail in (False, True):
            wrapper_control(arm, fail)
        binding.validate_ordering(fixture(arm))
    controls = [
        (
            "unreproduced old-wrapper control",
            "baseline",
            lambda r: r["controlled_streams"][0].update(
                samples=[sample() for _ in range(5)]
            ),
        ),
        (
            "patched ordering fails",
            "patched",
            lambda r: r["controlled_streams"][0].update(
                samples=[sample(0.01) for _ in range(5)]
            ),
        ),
        (
            "natural binding fails",
            "patched",
            lambda r: r["natural_streams"][0].update(
                samples=[sample(0.01) for _ in range(5)]
            ),
        ),
        (
            "profiler returns before delay",
            "patched",
            lambda r: r["actual_profiler"].update(
                samples=[
                    {
                        "profiler_ms": 0.002,
                        "host_ms": 0.01,
                        "residual_sync_ms": 5,
                        "numerics": True,
                    }
                    for _ in range(5)
                ]
            ),
        ),
        (
            "stale graph output",
            "patched",
            lambda r: r["graph_replays"][1].update(output=4.0),
        ),
        (
            "source replaced",
            "patched",
            lambda r: r["attestation_after"].update(wrapper_sha256="bad"),
        ),
        (
            "native replaced",
            "patched",
            lambda r: r["attestation_after"]["native"]["gemm"].update(
                library_sha256="bad"
            ),
        ),
        (
            "context leaked",
            "patched",
            lambda r: r["controlled_streams"][0].update(restored=False),
        ),
        (
            "duplicate stream",
            "patched",
            lambda r: r["controlled_streams"][0].update(stream_kind="nondefault"),
        ),
        (
            "forged reference",
            "patched",
            lambda r: r["reference"][0].update(median_ms=0.01),
        ),
        (
            "incomplete repetitions",
            "patched",
            lambda r: r["controlled_streams"][0]["samples"].pop(),
        ),
        (
            "nonfinite measurement",
            "patched",
            lambda r: r["controlled_streams"][0]["samples"][0].update(
                event_ms=float("nan")
            ),
        ),
    ]
    for name, arm, mutate in controls:
        record = fixture(arm)
        mutate(record)
        try:
            binding.validate_ordering(record)
        except RuntimeError:
            continue
        raise AssertionError("invalid evidence accepted: " + name)
    print(
        json.dumps(
            {
                "status": "passed",
                "wrapper_protocol_cases": 4,
                "callable_identity_cases": 6,
                "valid_gate_cases": 2,
                "rejected_evidence_cases": [row[0] for row in controls],
                "limits": "CPU boundary simulation and evidence validation only; CUDA ordering remains untested.",
            }
        )
    )


if __name__ == "__main__":
    main()
