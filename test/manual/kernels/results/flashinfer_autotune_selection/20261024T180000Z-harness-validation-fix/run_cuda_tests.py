"""Run the separate regression tree after the caller's qualified CUDA bootstrap."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import io
import json
import os
from pathlib import Path

HARNESS_SHA = "9f6f8db43d5b584fa07852e42c640f4c22a2ac8b535519bde637858841142efe"
TEST_SHA = "8c69bce9eeae4b123992a85e9a475e352481c063856ffadb908c021a81a7463b"
TUNER_SHA = "0db42ad90d1a3fec6f81509a2c520c1892cea0a433f5dbd7f43468a2a10243ca"
SELECTION = (
    "graph_study_validates_owned_output or "
    "fp8_validation_refresh_preserves_views or "
    "fp8_validation_rejects_unsafe_roles or "
    "graph_study_fixture_refresh_is_optional"
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Results:
    def __init__(self):
        self.rows = []

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or report.skipped:
            self.rows.append({"nodeid": report.nodeid, "outcome": report.outcome})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        "--candidate-root",
        dest="candidate_root",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".log").exists():
        raise FileExistsError("do not overwrite a gate")
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    import pytest
    import torch
    from flashinfer.autotuner import AutoTuner

    source = args.candidate_root.resolve()
    harness = source / "benchmarks/bench_autotuner_accuracy.py"
    target = source / "tests/autotuner/test_accuracy_harness.py"
    if not torch.cuda.is_available():
        raise RuntimeError("actual CUDA execution is required")
    if (sha(harness), sha(target), sha(inspect.getfile(AutoTuner))) != (
        HARNESS_SHA,
        TEST_SHA,
        TUNER_SHA,
    ):
        raise ValueError("candidate test or qualified runtime source changed")
    original = AutoTuner._instance
    records, captured = Results(), io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        code = pytest.main(
            [
                "-q",
                "-rA",
                "--noconftest",
                "--rootdir",
                str(source),
                "--confcutdir",
                str(source / "tests"),
                str(target),
                "-k",
                SELECTION,
            ],
            plugins=[records],
        )
    counts = {
        name: sum(row["outcome"] == name for row in records.rows)
        for name in ("passed", "failed", "skipped")
    }
    restored = AutoTuner._instance is original and not torch.is_inference_mode_enabled()
    passed = (
        int(code) == 0
        and counts == {"passed": 28, "failed": 0, "skipped": 0}
        and restored
    )
    result = {
        "status": "passed" if passed else "failed",
        "pytest_exit_code": int(code),
        "counts": counts,
        "tests": records.rows,
        "worker_sha256": sha(__file__),
        "harness_sha256": sha(harness),
        "test_sha256": sha(target),
        "runtime_tuner_path": inspect.getfile(AutoTuner),
        "runtime_tuner_sha256": TUNER_SHA,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "tuner_and_inference_state_restored": restored,
        "scope": "Actual CUDA tensor/capture/replay/validation tests with controlled profiling durations. Candidate files are separate from the qualified runtime tree. No descriptor port, selection/performance or production-policy change.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.output.with_suffix(".log").write_text(captured.getvalue())
    print(json.dumps({"status": result["status"], "counts": counts}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
