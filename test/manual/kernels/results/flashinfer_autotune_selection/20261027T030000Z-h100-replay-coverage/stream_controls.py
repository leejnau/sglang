"""CPU checks of actual piece streaming and its byte-integrity failure paths."""

import hashlib
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path

from stream_model import emit


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    checks = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        chunks = [b"prefix\x00data", b"range\xffdata"]
        pieces = []
        for index, data in enumerate(chunks):
            p = root / str(index)
            p.write_bytes(data)
            pieces.append(
                dict(
                    path=p.name,
                    size_bytes=len(data),
                    sha256=digest(data),
                    mtime_ns=p.stat().st_mtime_ns,
                )
            )
        row = dict(
            path="weights.safetensors",
            size_bytes=sum(map(len, chunks)),
            sha256=digest(b"".join(chunks)),
        )
        manifest = json.dumps(dict(files=[row])).encode()
        plan = json.dumps(
            dict(
                status="passed",
                model_manifest_sha256=digest(manifest),
                files=[dict(**row, pieces=pieces)],
            )
        ).encode()
        (root / "model-manifest.json").write_bytes(manifest)
        (root / "model-stream-plan.json").write_bytes(plan)

        def run(manifest_sha=None, plan_sha=None):
            output = io.BytesIO()
            emit(
                root, output, manifest_sha or digest(manifest), plan_sha or digest(plan)
            )
            return output.getvalue()

        archive = run()
        with tarfile.open(fileobj=io.BytesIO(archive)) as handle:
            assert handle.extractfile("model/weights.safetensors").read() == b"".join(
                chunks
            )
            assert handle.extractfile("model-manifest.json").read() == manifest
        checks.append("exact ordered piece bytes and manifest streamed")
        for name, action in (
            ("frozen manifest mismatch", lambda: run("0" * 64)),
            ("frozen stream-plan mismatch", lambda: run(plan_sha="0" * 64)),
        ):
            try:
                action()
            except ValueError:
                checks.append("reject " + name)
            else:
                raise RuntimeError(name + " accepted")
        target = root / "0"
        old = target.stat()
        target.write_bytes(b"x" * len(chunks[0]))
        os.utime(target, ns=(old.st_atime_ns, old.st_mtime_ns))
        try:
            run()
        except ValueError:
            checks.append("reject modified piece with original length and timestamp")
        else:
            raise RuntimeError("corrupted bytes accepted")
        target.write_bytes(chunks[0])
        os.utime(target, ns=(old.st_atime_ns, old.st_mtime_ns))
        target.unlink()
        target.symlink_to(root / "1")
        try:
            run()
        except ValueError:
            checks.append("reject symlink piece")
        else:
            raise RuntimeError("symlink accepted")
    print(json.dumps(dict(status="passed", checks=len(checks), cases=checks), indent=2))


if __name__ == "__main__":
    main()
