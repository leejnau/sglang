"""Restore timestamps for this study's exact archived native cache; never build."""

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

ARCHIVE_SHA = "84e44c807d2a897723e793eeb0dd676d1c254491fdf2c95149511953ee4272f5"
PRESERVATION_SHA = "40e684976202ae6eca47e462e5da92efa35ccd2fa4cf1ccaaa8db0716936b6df"
CACHE_PREFIX = "/study/runtime-attempt01/cache/"
SOURCE_PREFIX = "/study/input/current/source/"
NATIVE_PREFIX = "current/.cache/flashinfer/0.0.0+unknown/90a/cached_ops/"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dependency_records(path):
    data = path.read_bytes()
    assert data[:16] == b"# ninjadeps\n\x04\x00\x00\x00"
    offset, paths, records = 16, {}, []
    while offset < len(data):
        tag = struct.unpack_from("<I", data, offset)[0]
        size = tag & 0x7FFFFFFF
        offset += 4
        assert 4 <= size <= 524287 and size % 4 == 0 and offset + size <= len(data)
        payload = data[offset : offset + size]
        offset += size
        values = struct.unpack("<" + "I" * (size // 4), payload)
        if tag & 0x80000000:
            assert len(values) >= 3
            records.append((values[0], values[1] | values[2] << 32, values[3:]))
        else:
            identifier = (~values[-1]) & 0xFFFFFFFF
            assert identifier == len(paths)
            paths[identifier] = payload[:-4].rstrip(b"\0").decode()
    return [
        (paths[i], ns, [paths[j] for j in dependencies])
        for i, ns, dependencies in records
    ]


def restore(study):
    cache = study / "runtime-attempt01/cache"
    preserved = study / "native-preservation.json"
    assert digest(preserved) == PRESERVATION_SHA
    prefix = "native-build-cache/timing/"
    files = {
        row["path"][len(prefix) :]: row["sha256"]
        for row in json.loads(preserved.read_text())["files"]
        if row["path"].startswith(prefix)
    }
    assert files
    for relative, value in files.items():
        assert digest(cache / relative) == value, relative
    changes = []
    for name, count in (("gemm", 3), ("trtllm_utils", 5)):
        directory = cache / NATIVE_PREFIX / name
        lines = (directory / ".ninja_log").read_text().splitlines()
        assert lines[0] == "# ninja log v7"
        logs = {row.split("\t")[3]: int(row.split("\t")[2]) for row in lines[1:]}
        records = dependency_records(directory / ".ninja_deps")
        assert len(records) == count and len(logs) == count + 1
        latest_object = 0
        for output, timestamp, inputs in records:
            assert output.startswith(CACHE_PREFIX + NATIVE_PREFIX + name + "/")
            assert output.endswith(".o") and output in logs
            target = cache / output[len(CACHE_PREFIX) :]
            # Only source-owned inputs can be checked on the staging host.
            # The pinned container's system headers remain guarded by Ninja.
            for dependency in inputs:
                if dependency.startswith(SOURCE_PREFIX):
                    source = (
                        study
                        / "input/current/source"
                        / dependency[len(SOURCE_PREFIX) :]
                    )
                    assert source.is_file() and source.stat().st_mtime_ns <= min(
                        timestamp, logs[output]
                    ), dependency
            old = target.stat().st_mtime_ns
            os.utime(target, ns=(target.stat().st_atime_ns, timestamp))
            assert target.stat().st_mtime_ns == timestamp
            changes.append(
                dict(
                    path=str(target.relative_to(study)),
                    old_mtime_ns=old,
                    new_mtime_ns=timestamp,
                    basis="exact object timestamp stored in unchanged .ninja_deps",
                )
            )
            latest_object = max(latest_object, timestamp)
        library = directory / (name + ".so")
        recorded = logs[CACHE_PREFIX + str(library.relative_to(cache))]
        assert recorded >= latest_object
        old = library.stat().st_mtime_ns
        os.utime(library, ns=(library.stat().st_atime_ns, recorded))
        assert library.stat().st_mtime_ns == recorded
        changes.append(
            dict(
                path=str(library.relative_to(study)),
                old_mtime_ns=old,
                new_mtime_ns=recorded,
                basis="reconstructed library filesystem timestamp from .ninja_log; not claimed original stat timestamp",
            )
        )
    assert len(changes) == 10
    for relative, value in files.items():
        assert digest(cache / relative) == value, relative
    return dict(
        status="passed",
        method="archive bytes unchanged; restore eight object timestamps from deps and reconstruct two library timestamps from build logs",
        archive_sha256=ARCHIVE_SHA,
        preservation_manifest_sha256=PRESERVATION_SHA,
        files_verified=len(files),
        source_mtimes_modified=False,
        native_bytes_modified=False,
        changes=changes,
        limitation="Filesystem timestamp reconstruction is operational metadata; actual pinned-container Ninja no-work and exact native identity remain mandatory before measurement.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    report = restore(args.study)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "files_verified": report["files_verified"],
                "timestamps_restored": len(report["changes"]),
            }
        )
    )


if __name__ == "__main__":
    main()
