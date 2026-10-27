"""CPU controls for the exact cold-to-warm evidence gate; no model or GPU."""

import copy
import json
import tempfile
from pathlib import Path

from outer_audit import cache_before, persistent_caches, sha, validate_cold_gate


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
            qualification_complete=False,
            phase_requirements_met=True,
            findings=[],
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
        checks.append("complete cold qualification proceeds to fresh warm")
        mutations = []
        incomplete = copy.deepcopy(analysis)
        incomplete["phase_requirements_met"] = False
        incomplete["findings"] = [dict(reason="retained cold lifecycle finding")]
        mutations.append(("cold-analysis.json", incomplete))
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
        reset()
        runtime = root / "runtime"
        runtime.mkdir()
        (runtime / "sglang-cache").mkdir()
        (runtime / "managed-cache").mkdir()
        (runtime / "sglang-cache/winners.json").write_text("legacy winner")
        (runtime / "managed-cache/winner.json").write_text("scoped winner")
        write(
            root / "runtime-snapshot-cold.json",
            dict(persistent_caches=persistent_caches(runtime)),
        )
        validate_cold_gate(root, runtime)
        checks.append("both warm persistent roots equal cold snapshots")
        for relative in ("sglang-cache/winners.json", "managed-cache/winner.json"):
            path = runtime / relative
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            try:
                validate_cold_gate(root, runtime)
            except AssertionError:
                checks.append("reject altered warm bytes " + relative)
            else:
                raise RuntimeError("mutated warm cache accepted")
            path.write_bytes(original)
        link = runtime / "managed-cache/link"
        link.symlink_to(runtime / "sglang-cache/winners.json")
        try:
            persistent_caches(runtime)
        except AssertionError:
            checks.append("reject persistent-cache symlink")
        else:
            raise RuntimeError("cache symlink accepted")
        link.unlink()
        study = root / "study"
        (study / "results/scoped-cache/attempt01").mkdir(parents=True)
        cache_before(study, "cold")
        checks.append("cold requires both persistent roots absent")
        (study / "runtime-attempt01/managed-cache").mkdir(parents=True)
        try:
            cache_before(study, "cold")
        except AssertionError:
            checks.append("reject preexisting cold managed root")
        else:
            raise RuntimeError("preexisting managed root accepted")
    print(json.dumps(dict(status="passed", checks=len(checks), cases=checks), indent=2))


if __name__ == "__main__":
    main()
