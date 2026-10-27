"""Check actual full-model imports in the pinned runtime before model startup."""

import argparse
import hashlib
import importlib
import inspect
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provenance = json.loads(args.provenance.read_text())
    records = {}
    for name, expected in provenance["source_sha256"].items():
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError("unexpected imported source: " + name)
        records[name] = dict(path=str(path), sha256=observed)
    from flashinfer import ReplayCoverage, autotune_v2
    from flashinfer.autotuner.autotuner import AutoTuner

    from sglang.benchmark.one_batch import load_model
    from sglang.srt.server_args import ServerArgs

    assert callable(load_model) and callable(autotune_v2) and callable(ReplayCoverage)
    parameters = inspect.signature(autotune_v2).parameters
    assert {"cache_scope", "strict_replay", "cache_root", "replay_report"} <= set(
        parameters
    )
    # The CPU API controls replace import namespaces and must stay in a child.
    control_relative = "test/manual/kernels/audit_flashinfer_scoped_cache_controls.py"
    controls = Path("/workspace") / control_relative
    control_sha = hashlib.sha256(controls.read_bytes()).hexdigest()
    assert control_sha == provenance["harness_source_sha256"][control_relative]
    control_output = args.output.parent / "worker-api-controls.json"
    with (args.output.parent / "worker-api-controls.log").open("xb") as log:
        subprocess.run(
            [
                sys.executable,
                str(controls),
                "--source",
                "/study/input/current/source",
                "--output",
                str(control_output),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=60,
        )
    control_result = json.loads(control_output.read_text())
    assert control_result["tests"] > 0
    assert control_result["failures"] == control_result["errors"] == 0
    actual_api = control_result["real_tuner_control"]
    assert actual_api["checks"]
    for path, expected in actual_api["source_sha256"].items():
        module = path.removesuffix(".py").replace("/", ".")
        assert provenance["source_sha256"][module] == expected
    tuner = AutoTuner.get()
    for name in (
        "_override_cuda_graph_profile_replays",
        "_effective_measure_policy",
        "_active_tuning_contexts",
        "_cache_profile_result",
        "choose_with_protocol",
    ):
        getattr(tuner, name)
    # Parse the exact configuration without constructing a model or running tuning.
    args_config = ServerArgs(
        model_path="/model",
        tokenizer_path="/model",
        quantization="modelopt_fp8",
        dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        device="cuda",
        tp_size=1,
        max_running_requests=1,
        max_total_tokens=512,
        context_length=256,
        mem_fraction_static=0.7,
        disable_overlap_schedule=True,
        attention_backend="triton",
        cuda_graph_config={
            "decode": {"backend": "full", "bs": [1], "max_bs": 1},
            "prefill": {"backend": "disabled"},
        },
    )
    record = dict(
        status="completed",
        sources=records,
        quantization=args_config.quantization,
        model_loaded=False,
        model_forwards=0,
        tuning_calls=0,
        worker_api_controls_sha256=hashlib.sha256(
            control_output.read_bytes()
        ).hexdigest(),
        worker_api_controls_source_sha256=control_sha,
        worker_api_controls_limits=actual_api["limits"],
    )
    with args.output.open("x") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
