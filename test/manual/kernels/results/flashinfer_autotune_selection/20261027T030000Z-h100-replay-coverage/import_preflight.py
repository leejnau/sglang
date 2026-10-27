"""Check actual full-model imports in the pinned runtime before model startup."""

import argparse
import hashlib
import importlib
import json
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
    )
    with args.output.open("x") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
