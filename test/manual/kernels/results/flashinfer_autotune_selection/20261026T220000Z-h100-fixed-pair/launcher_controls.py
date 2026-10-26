"""CPU controls for the actual launcher's schedule and resume guards."""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parent
launcher = (root / "launch.sh").read_text()
batch = Path(sys.argv[1])


def heredoc(marker):
    return launcher.split("<<'" + marker + "'\n", 1)[1].split("\n" + marker, 1)[0]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


schedule = subprocess.run(
    [sys.executable, "-", str(batch / "schedule.json")],
    input=heredoc("PY_SCHEDULE"),
    text=True,
    capture_output=True,
    check=True,
)
rows = [line.split("\t") for line in schedule.stdout.splitlines()]
assert len(rows) == 8
assert [row[0] for row in rows] == list(map(str, range(8)))
assert [row[3] for row in rows] == [f"block{i}-auto_synthetic-timing" for i in range(8)]
checks = {"actual_schedule": "passed"}
with tempfile.TemporaryDirectory(prefix="fi-fixed-pair-launcher-") as temporary:
    folder = Path(temporary)
    result = folder / "admission-auto_synthetic.json"
    bootstrap = result.with_suffix(".bootstrap.json")
    analysis = folder / "admission-analysis.json"
    for case in (
        "valid",
        "missing_analysis",
        "failed_analysis",
        "changed_result",
        "changed_bootstrap",
        "bad_bootstrap_result_binding",
        "failed_worker",
    ):
        result.write_text(json.dumps({"status": "success"}) + "\n")
        bootstrap.write_text(
            json.dumps({"status": "passed", "result_sha256": digest(result)}) + "\n"
        )
        report = {
            "status": "passed",
            "admission_result_sha256": digest(result),
            "admission_bootstrap_sha256": digest(bootstrap),
        }
        if case == "failed_analysis":
            report["status"] = "failed"
        elif case == "changed_result":
            result.write_text(result.read_text() + " ")
        elif case == "changed_bootstrap":
            bootstrap.write_text(bootstrap.read_text() + " ")
        elif case == "bad_bootstrap_result_binding":
            bootstrap.write_text(
                json.dumps({"status": "passed", "result_sha256": "0" * 64})
            )
            report["admission_bootstrap_sha256"] = digest(bootstrap)
        elif case == "failed_worker":
            result.write_text(json.dumps({"status": "failed"}))
            bootstrap.write_text(
                json.dumps({"status": "passed", "result_sha256": digest(result)})
            )
            report["admission_result_sha256"] = digest(result)
            report["admission_bootstrap_sha256"] = digest(bootstrap)
        analysis.write_text(json.dumps(report))
        if case == "missing_analysis":
            analysis.unlink()
        completed = subprocess.run(
            [sys.executable, "-", temporary],
            input=heredoc("ADMISSION"),
            text=True,
            capture_output=True,
        )
        assert (completed.returncode == 0) == (case == "valid"), case
        checks[case] = "passed"

report = {
    "status": "passed",
    "checks": checks,
    "launcher_sha256": digest(root / "launch.sh"),
    "scope": "Actual embedded launcher code with controlled JSON and the frozen schedule; no Docker, GPU, allocation or worker execution.",
}
(root / "launcher-cpu-controls.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
