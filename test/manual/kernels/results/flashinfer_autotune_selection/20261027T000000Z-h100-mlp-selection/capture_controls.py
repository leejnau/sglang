"""CPU provenance controls for the manual MLP fixture capture."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = args.repo / "test/manual/kernels/capture_flashinfer_mlp_fixture.py"
    spec = importlib.util.spec_from_file_location("mlp_capture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if "torch" in sys.modules:
        raise AssertionError("capture helpers imported the accelerator dependency")
    checks = []

    def checked(label, predicate):
        if not predicate:
            raise AssertionError(label)
        checks.append(label)

    expected = module.previous_evidence(args.repo)
    checked("real historical input manifest is accepted", expected["shape"][0] == 144)
    source_hashes = module.source_evidence(args.repo)
    checked("real SGLang projection is accepted", len(source_hashes) == 7)
    for label, name, before, after in (
        (
            "reject changed implementation alias",
            "qwen3.py",
            "Qwen2MLP as Qwen3MLP",
            "Qwen2MLP as DifferentMLP",
        ),
        (
            "reject changed linear class",
            "qwen2.py",
            "self.down_proj = RowParallelLinear(",
            "self.down_proj = ColumnParallelLinear(",
        ),
        (
            "reject reversed projection dimensions",
            "qwen2.py",
            "intermediate_size,\n            hidden_size,",
            "hidden_size,\n            intermediate_size,",
        ),
        (
            "reject projection bias",
            "qwen2.py",
            "hidden_size,\n            bias=False,",
            "hidden_size,\n            bias=True,",
        ),
        (
            "reject changed activation",
            "qwen2.py",
            "self.act_fn = SiluAndMul()",
            "self.act_fn = GeluAndMul()",
        ),
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in source_hashes:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(args.repo / relative, target)
            target = root / "python/sglang/srt/models" / name
            text = target.read_text()
            if before not in text:
                raise AssertionError(f"control mutation no longer applies: {label}")
            target.write_text(text.replace(before, after))
            try:
                module.source_evidence(root)
            except RuntimeError:
                checks.append(label)
            else:
                raise AssertionError(label)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        target = root / module.PREVIOUS_MANIFEST
        target.parent.mkdir(parents=True)
        target.write_text((args.repo / module.PREVIOUS_MANIFEST).read_text() + " ")
        try:
            module.previous_evidence(root)
        except RuntimeError:
            checks.append("reject historical manifest byte drift")
        else:
            raise AssertionError("mutated prior manifest was accepted")
    result = subprocess.run(
        [sys.executable, str(path), "--help"], capture_output=True, text=True
    )
    checked(
        "help requires no accelerator dependency",
        result.returncode == 0 and "--model-cache" in result.stdout,
    )
    compile(path.read_bytes(), str(path), "exec")
    checks.append("capture source compiles")
    forbidden = bytes.fromhex("766c6c6d")
    checked(
        "capture has no prohibited framework token",
        forbidden not in path.read_bytes().lower(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "passed",
                "checks": len(checks),
                "passed": checks,
                "scope": "CPU-only provenance and source-drift controls; no model capture, numerical GPU validation or timing",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"status": "passed", "checks": len(checks)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
