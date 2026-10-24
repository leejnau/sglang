"""Run a fresh study process with the previously qualified native provider."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import traceback
from pathlib import Path
from types import ModuleType

import container_preflight as provider

HELPER_SHA = "b8d0f6bbc85eaca0e3ef1419ab99a97d5340c96b3da382cba81d7896768287ba"
MANIFEST_SHA = "9a4c51562d7410d785cc544c004bc4f92486f40311cb5c7a7328c2c1213160d5"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-data-links", action="store_true")
    parser.add_argument("worker_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    tail = args.worker_args
    if tail[:1] == ["--"]:
        tail = tail[1:]
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
        if args.prepare_data_links:
            record["packaging_preparation"] = provider.prepare_data_links(source)
        record["source"] = provider.check_source(
            source, args.manifest, "fp8-runtime-callback"
        )
        record["import"] = provider.import_current(source)
        native_modules = {}

        def retain_native(name, spec, native):
            if name in native_modules:
                raise RuntimeError("native provider was bound more than once")
            native_modules[name] = native
            return native

        record["native"] = provider.explicit_native_build(
            source, native_module_callback=retain_native
        )
        from flashinfer.gemm import gemm_base

        raw = native_modules["gemm"]
        # The public module wraps the native entry point. Validate the one-time
        # factory binding without replacing any forwarding method or timer.
        bound = gemm_base.gen_gemm_module().build_and_load()
        if bound is not raw:
            raise RuntimeError("native provider factory identity changed")
        record["native_identity_passthrough"] = True
        state = ModuleType("objective_values_bootstrap")
        state.native_gemm = raw
        state.evidence = record
        state.source_root = source
        state.candidate_manifest_sha256 = MANIFEST_SHA
        state.helper_sha256 = HELPER_SHA
        if state.__name__ in sys.modules:
            raise RuntimeError("bootstrap state already exists")
        sys.modules[state.__name__] = state
        record["worker_sha256"] = provider.digest(args.target)
        record["phase"] = "worker"
        bootstrap_path.write_text(json.dumps(record, indent=2) + "\n")
        sys.argv = [str(args.target), *tail]
        runpy.run_path(str(args.target), run_name="__main__")
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
