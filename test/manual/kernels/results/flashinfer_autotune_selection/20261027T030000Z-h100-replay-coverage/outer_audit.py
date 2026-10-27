"""Verify runtime/provider/checkpoint evidence separately from coverage findings."""

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

NATIVE = {
    "gemm": "72690cacca4ce1d69ccaa6b9190475483d984bd58de0962d06c874c801c678d6",
    "trtllm_utils": "e442a689064fe3af4ab46510b2d5aaf3597b072949b84f13d14e9e53352109e4",
}
REFERENCE = "7c4727aca3d99adcb3fe2f35d06f4408b512baa4fe9a91b975bdf77358fe205d"
PREFIX = "current/.cache/flashinfer/0.0.0+unknown/90a/cached_ops/"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, record):
    with path.open("x") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")


def snapshot(study, phase):
    root = study / "results/replay-coverage/attempt01"
    reuse = json.loads((study / "native-reuse.json").read_text())
    source = study / "input/current"
    for relative, expected in reuse["input_files"].items():
        assert sha(source / relative) == expected, relative
    assert {
        str(p.relative_to(source)): os.readlink(p)
        for p in source.rglob("*")
        if p.is_symlink()
    } == reuse["input_links"]
    cache = study / "runtime-attempt01/cache"
    target = root / "native-build-cache/qualified-providers"
    target.mkdir(exist_ok=True)
    for relative, expected in reuse["native_files"].items():
        original = cache / relative
        assert sha(original) == expected, relative
        copied = target / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        if not copied.exists():
            shutil.copy2(original, copied)
        assert sha(copied) == expected
    for name in NATIVE:
        expected = {
            r for r in reuse["native_files"] if r.startswith(PREFIX + name + "/")
        }
        assert {
            str(p.relative_to(cache))
            for p in (cache / PREFIX / name).rglob("*")
            if p.is_file()
        } == expected
    model = json.loads((study / "model-manifest.json").read_text())
    observed_model = {}
    for row in model["files"]:
        path = study / "model" / row["path"]
        assert path.stat().st_size == row["size_bytes"] and sha(path) == row["sha256"]
        observed_model[row["path"]] = row["sha256"]
    inventory = {
        str(p.relative_to(cache)): dict(size=p.stat().st_size, sha256=sha(p))
        for p in sorted(cache.rglob("*"))
        if p.is_file() and not p.is_symlink()
    }
    write_new(
        root / f"runtime-snapshot-{phase}.json",
        dict(
            status="passed",
            phase=phase,
            native_reuse_sha256=sha(study / "native-reuse.json"),
            input_files_verified=len(reuse["input_files"]),
            model_files=observed_model,
            native_files=reuse["native_files"],
            cache_inventory=inventory,
        ),
    )


def audit(root, phases, commit):
    provenance = json.loads((root / "provenance.json").read_text())
    assert provenance["execution_commit"] == commit
    assert (
        provenance["image_id"]
        == "sha256:2e152727f0a5ce999172ec48203b2b292d41b3cb370976827718870557899caa"
    )
    assert provenance["gpu"]["uuid"] == "GPU-240958a2-e8e1-3bac-b767-365c9e0b97bd"
    assert provenance["gpu"]["driver_version"] == "580.159.04"
    assert (
        provenance["native_reference_sha256"]
        == sha(root / "native-reference.bootstrap.json")
        == REFERENCE
    )
    assert provenance["model_manifest_sha256"] == sha(root / "model-manifest.json")
    model = json.loads((root / "model-manifest.json").read_text())
    assert model == provenance["model"] and model["status"] == "passed"
    assert model["revision"] == "42d9515ebd69eea3a87351d079c671c3c5ff0a31"
    assert model["model"] == "nvidia/Llama-3.1-8B-Instruct-FP8"
    assert len(model["files"]) == 9
    for row in model["files"]:
        if row["expected_lfs_sha256"]:
            assert row["sha256"] == row["expected_lfs_sha256"]
    reuse = json.loads((root / "native-build-cache/reuse-manifest.json").read_text())
    assert reuse["status"] == "passed"
    assert provenance["native_reuse_sha256"] == sha(
        root / "native-build-cache/reuse-manifest.json"
    )
    assert provenance["model_stream_plan_sha256"] == sha(
        root / "native-build-cache/model-stream-plan.json"
    )
    reference = json.loads((root / "native-reference.bootstrap.json").read_text())
    workers = {}
    ids = set()
    for phase in phases:
        assert (root / f"{phase}.exit-code").read_text().strip() == "0"
        assert (root / f"launcher-{phase}.exit-code").read_text().strip() == "0"
        cid = (root / f"{phase}.container-id").read_text().strip()
        assert re.fullmatch("[0-9a-f]{64}", cid) and cid not in ids
        ids.add(cid)
        path = root / f"{phase}.json"
        result = json.loads(path.read_text())
        assert result["status"] == "completed"
        bootstrap_path = root / f"{phase}.bootstrap.json"
        boot = json.loads(bootstrap_path.read_text())
        assert boot["status"] == "passed" and boot["result_sha256"] == sha(path)
        if phase != "preflight":
            assert boot["worker_sha256"] == provenance["worker_sha256"]
            assert result["provenance"] == provenance
        assert boot["native_identity_passthrough"]
        assert boot["native_reuse"]["reference_sha256"] == REFERENCE
        assert (
            boot["native_reuse"]["manifest_sha256"] == provenance["native_reuse_sha256"]
        )
        for boundary in ("before", "after_build", "after_worker"):
            row = boot["native_reuse"][boundary]
            assert row["status"] == "passed" and row["elf_sha256"] == NATIVE
        assert len(boot["native_reuse"]["ninja_dry_runs"]) == 2
        for row in boot["native_reuse"]["ninja_dry_runs"]:
            assert row["returncode"] == 0 and row["stdout"].endswith(
                "ninja: no work to do.\n"
            )
        for name, provider in boot["native"]["providers"].items():
            old = reference["native"]["providers"][name]
            for key in (
                "library_path",
                "library_sha256",
                "ninja_path",
                "ninja_sha256",
                "sources",
                "compile_commands",
            ):
                assert provider[key] == old[key], (name, key)
        observed = json.loads((root / f"runtime-snapshot-{phase}.json").read_text())
        assert (
            observed["status"] == "passed"
            and observed["native_files"] == reuse["native_files"]
        )
        assert observed["model_files"] == {
            row["path"]: row["sha256"] for row in model["files"]
        }
        for relative, expected in reuse["native_files"].items():
            assert (
                sha(root / "native-build-cache/qualified-providers" / relative)
                == expected
            ), relative
        assert not (root / f"gpu-processes-before-{phase}.csv").read_text().strip()
        assert not (root / f"gpu-processes-after-{phase}.csv").read_text().strip()
        assert not (root / f"sglang-status-after-{phase}.txt").read_text().strip()
        workers[phase] = dict(
            result_sha256=sha(path),
            bootstrap_sha256=sha(bootstrap_path),
            container_id=cid,
            native_files_verified=len(reuse["native_files"]),
            model_files_verified=len(observed["model_files"]),
        )
    return dict(
        status="passed",
        execution_commit=commit,
        provenance_sha256=sha(root / "provenance.json"),
        phases=phases,
        workers=workers,
        limitation="Checkpoint/source bytes verified on node; raw archive retains their hashes and the exact controlled native providers. Other imported native kernels are not individually qualified.",
    )


def validate_cold_gate(root):
    """Require valid, byte-bound cold evidence without requiring full coverage."""
    analysis = json.loads((root / "cold-analysis.json").read_text())
    assert analysis["status"] == "valid_evidence"
    assert len(analysis["workers"]) == 1
    outer = json.loads((root / "cold-outer-audit.json").read_text())
    assert outer["status"] == "passed"
    for key, path in (
        ("result_sha256", root / "cold.json"),
        ("bootstrap_sha256", root / "cold.bootstrap.json"),
    ):
        expected = sha(path)
        assert analysis["workers"][0][key] == expected
        assert outer["workers"]["cold"][key] == expected
    result = json.loads((root / "cold.json").read_text())
    bootstrap = json.loads((root / "cold.bootstrap.json").read_text())
    assert result["status"] == "completed" and bootstrap["status"] == "passed"
    assert bootstrap["result_sha256"] == sha(root / "cold.json")
    assert (root / "cold.exit-code").read_text().strip() == "0"
    assert (root / "launcher-cold.exit-code").read_text().strip() == "0"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--cold-gate", type=Path)
    parser.add_argument("--phase", choices=("preflight", "cold", "warm"))
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--execution-commit")
    parser.add_argument("--phases", nargs="+", default=["preflight", "cold", "warm"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cold_gate:
        validate_cold_gate(args.cold_gate)
    elif args.snapshot:
        if not args.phase:
            parser.error("--phase is required for snapshot")
        snapshot(args.snapshot, args.phase)
    else:
        if not all((args.raw_root, args.execution_commit, args.output)):
            parser.error("audit requires --raw-root --execution-commit --output")
        report = audit(args.raw_root, args.phases, args.execution_commit)
        write_new(args.output, report)
        print(json.dumps(dict(status=report["status"], phases=report["phases"])))


if __name__ == "__main__":
    main()
