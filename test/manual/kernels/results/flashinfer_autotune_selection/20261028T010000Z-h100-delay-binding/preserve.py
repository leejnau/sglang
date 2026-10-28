"""Archive an attempted phase with exact member hashes and no overwrite."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preserve(study, phase, destination, attempt=1):
    if attempt not in (1, 2):
        raise ValueError("unknown experiment attempt")
    root = study / "results/delay-binding/attempt01"
    stem = f"20261028T010000Z-h100-delay-binding-attempt{attempt:02d}-" + phase
    archive = destination / (stem + "-full-raw.tgz")
    receipt = destination / (stem + "-preservation.json")
    manifest = root / ("preservation-manifest-" + phase + ".json")
    if (
        not root.is_dir()
        or not destination.is_dir()
        or any(p.exists() for p in (archive, receipt, manifest))
    ):
        raise RuntimeError(
            "missing evidence directory or an earlier preservation exists"
        )
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("unexpected evidence symlink: " + str(path))
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(root)),
                    "size": path.stat().st_size,
                    "sha256": sha(path),
                }
            )
    with manifest.open("x") as handle:
        json.dump({"phase": phase, "files": files}, handle, indent=2)
        handle.write("\n")
    with tarfile.open(archive, "x:gz") as stream:
        stream.add(root, arcname="attempt01", recursive=True)
    with tarfile.open(archive, "r:gz") as stream:
        for row in files:
            data = stream.extractfile("attempt01/" + row["path"]).read()
            if (
                len(data) != row["size"]
                or hashlib.sha256(data).hexdigest() != row["sha256"]
            ):
                raise RuntimeError("archive content changed: " + row["path"])
    record = {
        "status": "verified",
        "phase": phase,
        "archive": str(archive),
        "size": archive.stat().st_size,
        "sha256": sha(archive),
        "files": len(files),
        "manifest_sha256": sha(manifest),
    }
    with receipt.open("x") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("ordering", "admission", "timing"), required=True
    )
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    print(
        json.dumps(
            preserve(args.study, args.phase, args.destination, args.attempt), indent=2
        )
    )


if __name__ == "__main__":
    main()
