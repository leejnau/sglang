"""Research-only cache transplant to isolate selection from cache-path costs.

This edits disposable copies of the observed caches. It is not migration code
for SGLang and must not be used as evidence that natural tuning chose these
winners.
"""

import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path("/study/final-study")
    source = root / "matrix-legacy-1-cache"
    legacy = root / "matched-attempt02-legacy-cache"
    scoped = root / "matched-attempt02-scoped-cache"
    assert not legacy.exists() and not scoped.exists()
    shutil.copytree(source, legacy)
    shutil.copytree(root / "matrix-scoped-1-cache", scoped)
    files = list(source.rglob("rank_tp0_pp0_dp0.json"))
    assert len(files) == 1
    src = files[0]
    original = json.loads(src.read_text())
    relative = src.relative_to(source)
    shutil.copy2(src, scoped / relative)
    changes = []
    for path in sorted(scoped.glob("**/entries/*.json")):
        entry = json.loads(path.read_text())
        winner = original[entry["key"]]
        assert entry["runner"] == winner[0] == "CublasFp8GemmRunner"
        assert entry["profiling_policy"] == winner[2]
        before = digest(path)
        old_tactic = entry["tactic"]
        entry["tactic"] = winner[1]
        path.write_text(json.dumps(entry, sort_keys=True) + "\n")
        changes.append(
            {
                "key": entry["key"],
                "old_tactic": old_tactic,
                "new_tactic": winner[1],
                "before": before,
                "after": digest(path),
            }
        )
    expected = {
        key
        for key, value in original.items()
        if isinstance(value, list) and value and value[0] == "CublasFp8GemmRunner"
    }
    assert {entry["key"] for entry in changes} == expected and len(expected) == 20
    assert digest(src) == digest(legacy / relative) == digest(scoped / relative)
    receipt = {
        "purpose": "diagnostic only; matched winners are deliberately seeded",
        "source_v1_sha256": digest(src),
        "untouched_v1_metadata_keys": sorted(set(original) - expected),
        "entries": changes,
    }
    (root / "matched-cache-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print("Copied 20 observed baseline winners into an isolated managed cache")


if __name__ == "__main__":
    main()
