"""Isolate one selected tactic using disposable cache copies, never production."""

import hashlib
import json
import shutil
from pathlib import Path


def snapshot(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*.json")
    }


def main():
    source = Path("/study/prepared-study/matrix-scoped-1-cache")
    output = Path("/study/reviewed-study")
    original_hashes = snapshot(source)
    v1_path = next(source.rglob("rank_tp0_pp0_dp0.json"))
    v1 = json.loads(v1_path.read_text())
    matches = []
    for path in source.glob("**/entries/*.json"):
        entry = json.loads(path.read_text())
        shapes = entry["key_fields"]["__tuple__"][2]["__tuple__"]
        if shapes[0]["__tuple__"] == [1, 1, 14336] and shapes[1]["__tuple__"] == [
            1,
            14336,
            4096,
        ]:
            matches.append((path, entry))
    assert len(matches) == 1
    path, entry = matches[0]
    key = entry["key"]
    assert entry["tactic"] == 1 and v1[key][1] == -1
    receipts = []
    for name in ("scoped-default", "legacy-explicit"):
        dest = output / ("numerical-" + name + "-cache")
        assert not dest.exists()
        shutil.copytree(source, dest)
        before = snapshot(dest)
        if name == "scoped-default":
            target = dest / path.relative_to(source)
            value = json.loads(target.read_text())
            value["tactic"] = -1
        else:
            target = dest / v1_path.relative_to(source)
            value = json.loads(target.read_text())
            value[key][1] = 1
        target.write_text(json.dumps(value, sort_keys=True) + "\n")
        after = snapshot(dest)
        changes = [k for k in before if before[k] != after[k]]
        assert changes == [str(target.relative_to(dest))]
        receipts.append(
            {
                "control": name,
                "changed_file": changes[0],
                "before": before,
                "after": after,
            }
        )
    assert snapshot(source) == original_hashes
    (output / "numerical-cache-control.json").write_text(
        json.dumps(
            {
                "purpose": "causal diagnostic only; not a fix or natural-selection result",
                "key": key,
                "original_cache_hashes": original_hashes,
                "controls": receipts,
            },
            indent=2,
        )
        + "\n"
    )
    print("Prepared two copies with one tactic changed; original unchanged")


if __name__ == "__main__":
    main()
