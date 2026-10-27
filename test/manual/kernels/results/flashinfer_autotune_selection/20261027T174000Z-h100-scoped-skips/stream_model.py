"""Emit verified logical checkpoint files without another login-host disk copy."""

import argparse
import hashlib
import io
import json
import os
import stat
import tarfile
from pathlib import Path


class PieceReader:
    def __init__(self, root, row):
        self.root, self.row = root.resolve(), row
        self.position = self.total = 0
        self.handle = None
        self.full_hash = hashlib.sha256()
        self.piece_hash = None
        self.remaining = 0
        self.identity = None

    def _open(self):
        piece = self.row["pieces"][self.position]
        raw_path = self.root / piece["path"]
        path = raw_path.resolve()
        if (
            raw_path.is_symlink()
            or not path.is_relative_to(self.root)
            or not path.is_file()
        ):
            raise ValueError("piece escapes checkpoint staging root")
        self.handle = path.open("rb")
        metadata = os.fstat(self.handle.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size != piece["size_bytes"]
            or metadata.st_mtime_ns != piece["mtime_ns"]
        ):
            raise ValueError("piece metadata changed before transfer")
        self.identity = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        )
        self.remaining = metadata.st_size
        self.piece_hash = hashlib.sha256()

    def _close(self):
        metadata = os.fstat(self.handle.fileno())
        observed = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        )
        self.handle.close()
        self.handle = None
        if observed != self.identity:
            raise ValueError("piece changed during transfer")
        if self.piece_hash.hexdigest() != self.row["pieces"][self.position]["sha256"]:
            raise ValueError("piece bytes changed during transfer")
        self.position += 1

    def read(self, size):
        chunks = []
        while size and self.position < len(self.row["pieces"]):
            if self.handle is None:
                self._open()
            data = self.handle.read(min(size, self.remaining))
            if not data:
                raise ValueError("truncated checkpoint piece")
            self.piece_hash.update(data)
            self.full_hash.update(data)
            chunks.append(data)
            self.remaining -= len(data)
            self.total += len(data)
            size -= len(data)
            if not self.remaining:
                self._close()
        return b"".join(chunks)

    def finish(self):
        if self.handle is not None or self.position != len(self.row["pieces"]):
            raise ValueError("checkpoint pieces were not fully consumed")
        if (
            self.total != self.row["size_bytes"]
            or self.full_hash.hexdigest() != self.row["sha256"]
        ):
            raise ValueError("logical checkpoint differs from pinned hash")


def emit(root, output, expected_manifest, expected_plan):
    manifest_bytes = (root / "model-manifest.json").read_bytes()
    plan_bytes = (root / "model-stream-plan.json").read_bytes()
    if (
        hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest
        or hashlib.sha256(plan_bytes).hexdigest() != expected_plan
    ):
        raise ValueError("checkpoint manifest differs from frozen source plan")
    manifest = json.loads(manifest_bytes)
    plan = json.loads(plan_bytes)
    if (
        plan["status"] != "passed"
        or plan["model_manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest()
    ):
        raise ValueError("unbound checkpoint stream plan")
    expected = {row["path"]: row for row in manifest["files"]}
    if len(plan["files"]) != len(expected) or {
        row["path"] for row in plan["files"]
    } != set(expected):
        raise ValueError("checkpoint stream file set changed")
    with tarfile.open(fileobj=output, mode="w|") as archive:
        for row in plan["files"]:
            if Path(row["path"]).name != row["path"]:
                raise ValueError("invalid checkpoint filename")
            if any(
                row[key] != expected[row["path"]][key]
                for key in ("size_bytes", "sha256")
            ):
                raise ValueError("stream does not match checkpoint manifest")
            if sum(piece["size_bytes"] for piece in row["pieces"]) != row["size_bytes"]:
                raise ValueError("checkpoint piece lengths differ from logical file")
            info = tarfile.TarInfo("model/" + row["path"])
            info.size, info.mode = row["size_bytes"], 0o644
            reader = PieceReader(root, row)
            archive.addfile(info, reader)
            reader.finish()
        for name, data in (
            ("model-manifest.json", manifest_bytes),
            ("model-stream-plan.json", plan_bytes),
        ):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o644
            archive.addfile(info, io.BytesIO(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-manifest", required=True)
    parser.add_argument("--expected-plan", required=True)
    args = parser.parse_args()
    import sys

    emit(args.root, sys.stdout.buffer, args.expected_manifest, args.expected_plan)


if __name__ == "__main__":
    main()
