#!/usr/bin/env python3
"""Materialize two exact source arms from the pinned image package."""

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import tarfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unpack(archive, destination):
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as source:
        source.extractall(destination, filter="data")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", type=Path)
    parser.add_argument("batch_dir", type=Path)
    args = parser.parse_args()
    study, batch = args.study_dir, args.batch_dir
    overlay = study / "source-overlay"
    unpack(batch / "source-overlay.tar.gz", overlay)
    manifest = json.loads((overlay / "source-manifest.json").read_text())
    assert (overlay / "source-manifest.json").read_bytes() == (
        batch / "source-manifest.json"
    ).read_bytes()
    spec = importlib.util.find_spec("flashinfer")
    assert spec and spec.origin
    installed = Path(spec.origin).resolve().parent
    original_hashes = {}
    for row in manifest["files"]:
        if not row["runtime_overlay"]:
            continue
        relative = Path(row["installed_package_path"]).relative_to("flashinfer")
        actual = digest(installed / relative)
        assert actual == row["baseline_sha256"], row["repository_path"]
        original_hashes[row["installed_package_path"]] = actual
    prepared = {}
    for arm in ("baseline", "candidate"):
        package_root = study / "packages" / arm
        destination = package_root / "flashinfer"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            installed,
            destination,
            symlinks=False,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        for row in manifest["files"]:
            if not row["runtime_overlay"]:
                continue
            target = package_root / row["installed_package_path"]
            if row.get("archive_payload", True):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(overlay / arm / row["repository_path"], target)
        if arm == "candidate":
            subprocess.run(
                ["git", "apply", str(overlay / "autotuner-doc-only.diff")],
                cwd=package_root,
                check=True,
            )
        hashes = {}
        for row in manifest["files"]:
            if row["runtime_overlay"]:
                target = package_root / row["installed_package_path"]
                actual = digest(target)
                assert actual == row[arm + "_sha256"], row["repository_path"]
                hashes[row["installed_package_path"]] = actual
        prepared[arm] = {"package_root": str(package_root), "sha256": hashes}
    test_root = study / "upstream-tests"
    unpack(batch / "upstream-test-support.tar.gz", test_root)
    tests_manifest = json.loads((test_root / "test-support-manifest.json").read_text())
    test_file = "tests/autotuner/test_autotuner_cublaslt_tactics.py"
    shutil.copyfile(overlay / "candidate" / test_file, test_root / test_file)
    (study / "prepared-packages.json").write_text(
        json.dumps(
            {
                "source_manifest_sha256": digest(overlay / "source-manifest.json"),
                "original_image_package": str(installed),
                "original_runtime_sha256": original_hashes,
                "arms": prepared,
                "upstream_test_sha256": digest(test_root / test_file),
                "test_support_manifest": tests_manifest,
                "scope": "Image metadata/vendors retained; exact release runtime sources versus exact PR diff over that release. Full upstream test environment is not reproduced.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
