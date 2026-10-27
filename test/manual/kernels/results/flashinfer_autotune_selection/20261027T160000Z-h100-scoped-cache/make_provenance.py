"""Bind the frozen source, checkpoint and allocated runtime before worker import."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

MODULES = (
    "flashinfer.autotune_cache",
    "flashinfer.autotuner.autotuner",
    "flashinfer.gemm.gemm_base",
    "sglang.benchmark.one_batch",
    "sglang.srt.model_executor.runner.flashinfer_autotune",
    "sglang.srt.model_executor.runner.decode_cuda_graph_runner",
    "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend",
    "sglang.srt.layers.quantization.modelopt_quant",
    "sglang.srt.layers.quantization.fp8_utils",
)
BATCH = "test/manual/kernels/results/flashinfer_autotune_selection/20261027T160000Z-h100-scoped-cache"
WORKER = "test/manual/kernels/audit_flashinfer_scoped_cache.py"
HARNESS = (
    WORKER,
    "test/manual/kernels/audit_flashinfer_scoped_cache_controls.py",
    "test/manual/kernels/audit_flashinfer_replay_coverage.py",
    "test/manual/kernels/results/flashinfer_autotune_selection/20261027T020000Z-h100-mlp-confirmation/run_current.py",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--phase", choices=("preflight", "cold", "warm"), required=True)
    args = parser.parse_args()
    s = args.study
    source = s / "source"
    r = s / "results/scoped-cache/attempt01"
    plan = json.loads((source / BATCH / "plan.json").read_text())
    images = json.loads((r / f"image-{args.phase}.json").read_text())
    assert (
        images["Id"]
        == "sha256:2e152727f0a5ce999172ec48203b2b292d41b3cb370976827718870557899caa"
    )
    assert plan["environment"]["image"] in images["RepoDigests"]
    assert images["Architecture"] == "amd64" and images["Os"] == "linux"
    with (r / f"gpu-before-{args.phase}.csv").open() as handle:
        gpu = [
            {k.strip(): v.strip() for k, v in row.items()}
            for row in csv.DictReader(handle)
        ]
    assert len(gpu) == 1
    gpu = gpu[0]
    assert gpu["name"] == plan["environment"]["gpu"]
    assert gpu["uuid"] == plan["environment"]["gpu_uuid"]
    assert gpu["driver_version"] == plan["environment"]["driver"]
    assert float(gpu["power.limit [W]"].split()[0]) == 310
    assert int(gpu["memory.used [MiB]"].split()[0]) == 0
    expected = {}
    for name in MODULES:
        path = Path(*name.split(".")).with_suffix(".py")
        root = (
            s / "input/current/source"
            if name.startswith("flashinfer.")
            else source / "python"
        )
        expected[name] = digest(root / path)
    for rel, sha in plan["environment"]["flashinfer_source_sha256"].items():
        assert digest(s / "input/current/source/flashinfer" / rel) == sha
    model = json.loads((s / "model-manifest.json").read_text())
    assert model["status"] == "passed" and model["model"] == plan["model"]["repository"]
    assert model["revision"] == plan["model"]["revision"]
    assert digest(s / "model-manifest.json") == digest(
        source / BATCH / "model-manifest.json"
    )
    assert digest(s / "model-stream-plan.json") == plan["model"]["stream_plan_sha256"]
    record = dict(
        schema_version=1,
        model_stream_plan_sha256=digest(s / "model-stream-plan.json"),
        execution_commit=args.commit,
        plan_sha256=digest(source / BATCH / "plan.json"),
        worker_sha256=digest(source / WORKER),
        source_sha256=expected,
        harness_source_sha256={
            relative: digest(source / relative) for relative in HARNESS
        },
        persistent_cache_roots={
            "legacy": "/study/runtime-attempt01/sglang-cache",
            "managed": "/study/runtime-attempt01/managed-cache",
        },
        negative_root_pattern="/study/runtime-attempt01/negative-{phase}",
        image=plan["environment"]["image"],
        image_id=images["Id"],
        gpu={k: gpu[k] for k in ("name", "uuid", "driver_version", "power.limit [W]")},
        model_manifest_sha256=digest(s / "model-manifest.json"),
        model=model,
        native_reference_sha256=digest(s / "native-reference.bootstrap.json"),
        native_reuse_sha256=digest(s / "native-reuse.json"),
        source_snapshot_sha256=digest(s / "source-snapshot.json"),
    )
    output = r / "provenance.json"
    if output.exists():
        assert json.loads(output.read_text()) == record
    else:
        output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
