"""Run the exact upstream CPU and H100 FP8 tests with an explicit native build."""

import argparse
import hashlib
import importlib
import json
import os
import sys
import traceback
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"status": "error", "expected_test_count": 19, "reports": []}
    exit_code = 1
    try:
        root = args.test_root.resolve()
        manifest = json.loads(args.manifest.read_text())
        support = json.loads((root / "test-support-manifest.json").read_text())
        hashes = {}
        for row in support["files"]:
            path = root / row["repository_path"]
            actual = sha256(path)
            if actual != row["sha256"]:
                raise RuntimeError(
                    "upstream helper hash mismatch: " + row["repository_path"]
                )
            hashes[row["repository_path"]] = actual
        target_rel = "tests/autotuner/test_autotuner_cublaslt_tactics.py"
        target = root / target_rel
        expected = next(
            row["candidate_sha256"]
            for row in manifest["files"]
            if row["repository_path"] == target_rel
        )
        if sha256(target) != expected:
            raise RuntimeError("the upstream test differs from the frozen candidate")
        hashes[target_rel] = expected
        if (root / "tests/conftest.py").exists():
            raise RuntimeError(
                "the focused driver requires the recorded minimal helper tree"
            )
        # Exact upstream package initializers resolve relative imports and avoid
        # unrelated installed packages named tests.
        sys.path.insert(0, str(root))
        tests = importlib.import_module("tests")
        if Path(tests.__file__).resolve() != root / "tests/__init__.py":
            raise RuntimeError("a different tests package was already imported")
        os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        from flashinfer_autotune_descriptor_qualification import prepare_native

        native = prepare_native("candidate", manifest)
        import pytest

        class Reports:
            def pytest_runtest_logreport(self, report):
                if report.when == "call" or report.failed or report.skipped:
                    result["reports"].append(
                        {
                            "nodeid": report.nodeid,
                            "phase": report.when,
                            "outcome": report.outcome,
                            "duration_seconds": report.duration,
                            "detail": str(report.longrepr) if report.longrepr else None,
                        }
                    )

        names = [
            "test_descriptor_bytes_roundtrip",
            "test_descriptor_json_roundtrip",
            "test_non_descriptor_tactics_rejected",
            "test_descriptor_config_file_roundtrip",
            "test_descriptor_managed_cache_roundtrip",
            "test_tuned_descriptor_reused_across_bucket[cublas]",
            "test_graph_capture_inside_outer_autotune[cublas]",
            "test_invalid_descriptor_runs_heuristic_default[cublas]",
            "test_index_tactics_rejected[cublas]",
            "test_tuned_descriptor_persists[cublas]",
        ]
        targets = [str(target) + "::" + name for name in names]
        pytest_code = int(
            pytest.main(
                [
                    "-q",
                    "-c",
                    "/dev/null",
                    "--rootdir",
                    str(root),
                    "--confcutdir",
                    str(root),
                    "--import-mode=importlib",
                    *targets,
                ],
                plugins=[Reports()],
            )
        )
        passed = [
            r
            for r in result["reports"]
            if r["phase"] == "call" and r["outcome"] == "passed"
        ]
        bad = [r for r in result["reports"] if r["outcome"] != "passed"]
        result.update(
            {
                "pytest_exit_code": pytest_code,
                "passed": len(passed),
                "failed_or_skipped": len(bad),
                "native_artifact": native,
                "test_source_sha256": hashes,
                "source_manifest_sha256": sha256(args.manifest),
                "targets": targets,
                "scope": support["scope"],
            }
        )
        if pytest_code or len(passed) != 19 or bad:
            raise RuntimeError("the exact 19 selected upstream tests did not all pass")
        result["status"] = "success"
        exit_code = 0
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
