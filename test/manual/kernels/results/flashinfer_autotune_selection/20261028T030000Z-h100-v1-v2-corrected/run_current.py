"""Run a fresh study process with the previously qualified native provider."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
import traceback
from pathlib import Path
from types import ModuleType

import container_preflight as provider
import flashinfer_autotune_delay_binding as binding

HELPER_SHA = "b8d0f6bbc85eaca0e3ef1419ab99a97d5340c96b3da382cba81d7896768287ba"
MANIFEST_SHA = "9a4c51562d7410d785cc544c004bc4f92486f40311cb5c7a7328c2c1213160d5"
REFERENCE_SHA = "7c4727aca3d99adcb3fe2f35d06f4408b512baa4fe9a91b975bdf77358fe205d"
NATIVE_HASHES = {
    "gemm": "72690cacca4ce1d69ccaa6b9190475483d984bd58de0962d06c874c801c678d6",
    "trtllm_utils": "e442a689064fe3af4ab46510b2d5aaf3597b072949b84f13d14e9e53352109e4",
}


def verify_native_files(cache, manifest):
    if (
        manifest["status"] != "passed"
        or manifest["native_reference_sha256"] != REFERENCE_SHA
    ):
        raise RuntimeError("unqualified native reuse manifest")
    files = manifest["native_files"]
    prefix = "current/.cache/flashinfer/0.0.0+unknown/90a/cached_ops/"
    for name, expected in NATIVE_HASHES.items():
        if files.get(f"{prefix}{name}/{name}.so") != expected:
            raise RuntimeError("native reference hash changed: " + name)
        directory = cache / prefix / name
        actual = {
            str(p.relative_to(cache)) for p in directory.rglob("*") if p.is_file()
        }
        required = {p for p in files if p.startswith(prefix + name + "/")}
        if actual != required:
            raise RuntimeError("native file inventory changed: " + name)
    for relative, expected in files.items():
        if provider.digest(cache / relative) != expected:
            raise RuntimeError("native build artifact changed: " + relative)
    return {
        "status": "passed",
        "files_verified": len(files),
        "elf_sha256": NATIVE_HASHES,
    }


def require_ninja_no_work(spec):
    completed = subprocess.run(
        ["ninja", "-n", "-C", str(spec.build_dir), "-f", str(spec.ninja_path)],
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    lines = completed.stdout.splitlines()
    expected = [
        f"ninja: Entering directory `{spec.build_dir}'",
        "ninja: no work to do.",
    ]
    if completed.stderr or lines != expected:
        raise RuntimeError(
            "native build requires work: " + completed.stdout + completed.stderr
        )
    return {
        "name": spec.name,
        "stdout": completed.stdout,
        "returncode": completed.returncode,
    }


def check_arm_source(source, manifest_path, arm):
    """Verify the historical critical sources with exactly one declared overlay."""
    manifest = json.loads(manifest_path.read_text())
    if manifest["base_commit"] != "27c62fdf0ab1cd4014df8a15b66a975541fda83c":
        raise RuntimeError("qualified source commit changed")
    candidate = next(
        row for row in manifest["candidates"] if row["name"] == "fp8-runtime-callback"
    )
    base = json.loads(
        (manifest_path.parent / "complete-source-manifest.json").read_text()
    )
    hashes = {row["path"]: row["sha256"] for row in base["files"]}
    hashes.update(
        {row["path"]: row["candidate_sha256"] for row in candidate["tracked_changes"]}
    )
    wrapper = "flashinfer/tllm_utils.py"
    if hashes[wrapper] != binding.wrapper_sha("baseline"):
        raise RuntimeError("baseline wrapper does not match qualified source")
    hashes[wrapper] = binding.wrapper_sha(arm)
    critical = {
        "flashinfer/autotuner/autotuner.py",
        "flashinfer/autotune_cache.py",
        "flashinfer/autotuner/_graph_timing.py",
        "flashinfer/jit/env.py",
        "flashinfer/jit/core.py",
        "flashinfer/jit/cpp_ext.py",
        "flashinfer/jit/gemm/core.py",
        wrapper,
        "flashinfer/jit/tllm_utils.py",
        "build_backend.py",
        "requirements.txt",
        "version.txt",
    }
    observed = {}
    for relative in critical | {row["path"] for row in candidate["tracked_changes"]}:
        actual = provider.digest(source / relative)
        if actual != hashes[relative]:
            raise RuntimeError("source mismatch: " + relative)
        observed[relative] = actual
    if (source / "flashinfer/_build_meta.py").exists():
        raise RuntimeError("unexpected build metadata")
    for name, relative in (
        ("csrc", "csrc"),
        ("include", "include"),
        ("cutlass", "3rdparty/cutlass"),
        ("cccl", "3rdparty/cccl"),
        ("spdlog", "3rdparty/spdlog"),
    ):
        link = source / "flashinfer/data" / name
        if not link.is_symlink() or link.resolve() != (source / relative).resolve():
            raise RuntimeError("source packaging link changed: " + name)
    return {
        "source_commit": manifest["base_commit"],
        "variant": "fp8-runtime-callback",
        "manifest_sha256": provider.digest(manifest_path),
        "source_sha256": observed,
        "runtime_metadata": "absent; official fallback0.0.0+unknown is expected",
        "declared_overlay": {
            "arm": arm,
            "path": wrapper,
            "baseline_sha256": binding.wrapper_sha("baseline"),
            "actual_sha256": binding.wrapper_sha(arm),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("patched",), required=True)
    parser.add_argument("--tuning-arm", choices=("v1", "v2"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-reference", type=Path, required=True)
    parser.add_argument("--reuse-manifest", type=Path, required=True)
    parser.add_argument("worker_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    tail = args.worker_args
    if tail[:1] == ["--"]:
        tail = tail[1:]
    if tail.count("--arm") != 1 or tail[tail.index("--arm") + 1] != args.tuning_arm:
        raise ValueError("bootstrap and worker tuning arms differ")
    output = args.output.resolve()
    bootstrap_path = output.with_suffix(".bootstrap.json")
    if output.exists() or bootstrap_path.exists():
        raise FileExistsError("refusing to overwrite an attempted worker")
    if tail.count("--output") != 1:
        raise ValueError("worker must name exactly one output")
    if Path(tail[tail.index("--output") + 1]).resolve() != output:
        raise ValueError("bootstrap and worker output paths differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "status": "incomplete",
        "pid": os.getpid(),
        "phase": "bootstrap",
        "binding_arm": args.arm,
        "tuning_arm": args.tuning_arm,
        "helper_sha256": HELPER_SHA,
        "candidate_manifest_sha256": MANIFEST_SHA,
        "worker": str(args.target.resolve()),
        "worker_args": tail,
    }
    try:
        if provider.digest(provider.__file__) != HELPER_SHA:
            raise RuntimeError("qualified native helper changed")
        if provider.digest(args.manifest) != MANIFEST_SHA:
            raise RuntimeError("qualified source manifest changed")
        source = args.source.resolve()
        if provider.digest(args.native_reference) != REFERENCE_SHA:
            raise RuntimeError("prior native bootstrap changed")
        reference = json.loads(args.native_reference.read_text())
        reuse = json.loads(args.reuse_manifest.read_text())
        cache = Path("/study/runtime-attempt01/cache")
        if os.environ.get("FLASHINFER_WORKSPACE_BASE") != str(cache / "current"):
            raise RuntimeError("native cache container path changed")
        record["native_reuse"] = {
            "reference_sha256": REFERENCE_SHA,
            "manifest_sha256": provider.digest(args.reuse_manifest),
            "before": verify_native_files(cache, reuse),
        }
        record["source"] = check_arm_source(source, args.manifest, args.arm)
        record["import"] = provider.import_current(source)
        from flashinfer.gemm import gemm_base
        from flashinfer.jit.tllm_utils import gen_trtllm_utils_module

        specs = [gemm_base.gen_gemm_module(), gen_trtllm_utils_module()]
        # Compare generated recipes before the unchanged helper can run Ninja.
        for spec in specs:
            spec.write_ninja()
        verify_native_files(cache, reuse)
        record["native_reuse"]["ninja_dry_runs"] = [
            require_ninja_no_work(spec) for spec in specs
        ]
        native_modules = {}

        def retain_native(name, spec, native):
            if name in native_modules:
                raise RuntimeError("native provider was bound more than once")
            native_modules[name] = native
            return native

        record["native"] = provider.explicit_native_build(
            source, native_module_callback=retain_native
        )
        record["native_reuse"]["after_build"] = verify_native_files(cache, reuse)
        for name, native in record["native"]["providers"].items():
            previous = reference["native"]["providers"][name]
            for field in (
                "library_path",
                "library_sha256",
                "ninja_path",
                "ninja_sha256",
                "sources",
                "compile_commands",
            ):
                if native[field] != previous[field]:
                    raise RuntimeError(
                        "native provenance changed: " + name + "." + field
                    )

        raw = native_modules["gemm"]
        # The public module wraps the native entry point. Validate the one-time
        # factory binding without replacing any forwarding method or timer.
        bound = gemm_base.gen_gemm_module().build_and_load()
        if bound is not raw:
            raise RuntimeError("native provider factory identity changed")
        record["native_identity_passthrough"] = True
        state = ModuleType("objective_values_bootstrap")
        state.native_gemm = raw
        state.native_delay = native_modules["trtllm_utils"]
        state.evidence = record
        state.source_root = source
        state.candidate_manifest_sha256 = MANIFEST_SHA
        state.helper_sha256 = HELPER_SHA
        if state.__name__ in sys.modules:
            raise RuntimeError("bootstrap state already exists")
        sys.modules[state.__name__] = state
        record["binding_before"] = binding.attest(args.arm)
        record["worker_sha256"] = provider.digest(args.target)
        record["phase"] = "worker"
        bootstrap_path.write_text(json.dumps(record, indent=2) + "\n")
        sys.argv = [str(args.target), *tail]
        try:
            runpy.run_path(str(args.target), run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        record["binding_after"] = binding.attest(args.arm)
        record["native_reuse"]["after_worker"] = verify_native_files(cache, reuse)
        if check_arm_source(source, args.manifest, args.arm) != record["source"]:
            raise RuntimeError("source changed during worker")
        record["status"] = "passed"
    except SystemExit as exc:
        record["status"] = "passed" if exc.code in (None, 0) else "failed"
        if record["status"] == "failed":
            record["error"] = repr(exc)
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = repr(exc)
        record["traceback"] = traceback.format_exc()
    if record["status"] == "passed" and not output.is_file():
        record.update(status="failed", error="worker returned without a result")
    if output.is_file():
        record["result_sha256"] = provider.digest(output)
    bootstrap_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "output": str(output)}))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
