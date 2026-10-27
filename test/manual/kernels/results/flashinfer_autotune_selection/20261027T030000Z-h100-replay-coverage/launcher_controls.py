"""CPU controls for the exact cold-to-warm evidence gate; no model or GPU."""

import copy
import json
import tempfile
from pathlib import Path

from outer_audit import sha, validate_cold_gate


def write(path, value):
    path.write_text(json.dumps(value))


def main():
    checks = []
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        result = dict(status="completed", phase="cold")
        write(root / "cold.json", result)
        bootstrap = dict(status="passed", result_sha256=sha(root / "cold.json"))
        write(root / "cold.bootstrap.json", bootstrap)
        pair = dict(
            result_sha256=sha(root / "cold.json"),
            bootstrap_sha256=sha(root / "cold.bootstrap.json"),
        )
        analysis = dict(
            status="valid_evidence",
            coverage_complete=False,
            findings=[dict(reason="retained cache miss")],
            workers=[pair],
        )
        outer = dict(status="passed", workers=dict(cold=pair))

        def reset():
            write(root / "cold.json", result)
            write(root / "cold.bootstrap.json", bootstrap)
            write(root / "cold-analysis.json", analysis)
            write(root / "cold-outer-audit.json", outer)
            (root / "cold.exit-code").write_text("0\n")
            (root / "launcher-cold.exit-code").write_text("0\n")

        reset()
        validate_cold_gate(root)
        checks.append("valid incomplete coverage proceeds to fresh warm")
        mutations = []
        for name, original, key, value in (
            ("cold.json", result, "status", "error"),
            ("cold.bootstrap.json", bootstrap, "status", "failed"),
            (
                "cold-analysis.json",
                analysis,
                "status",
                "invalid_or_incomplete_evidence",
            ),
            ("cold-outer-audit.json", outer, "status", "failed"),
        ):
            modified = copy.deepcopy(original)
            modified[key] = value
            mutations.append((name, modified))
        modified = copy.deepcopy(analysis)
        modified["workers"][0]["result_sha256"] = "0" * 64
        mutations.append(("cold-analysis.json", modified))
        modified = copy.deepcopy(outer)
        modified["workers"]["cold"]["bootstrap_sha256"] = "0" * 64
        mutations.append(("cold-outer-audit.json", modified))
        for index, (name, modified) in enumerate(mutations):
            reset()
            write(root / name, modified)
            try:
                validate_cold_gate(root)
            except AssertionError:
                checks.append("reject stale/invalid cold record " + str(index))
            else:
                raise RuntimeError("invalid gate accepted: " + str(index))
        for name in ("cold.exit-code", "launcher-cold.exit-code"):
            reset()
            (root / name).write_text("1\n")
            try:
                validate_cold_gate(root)
            except AssertionError:
                checks.append("reject failed process " + name)
            else:
                raise RuntimeError("failed process accepted")
    print(json.dumps(dict(status="passed", checks=len(checks), cases=checks), indent=2))


if __name__ == "__main__":
    main()
