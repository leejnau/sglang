"""Analyze the predeclared diagnostic; never pool seeded and natural choices."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("root", type=Path)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
runs = {}
for path in sorted(a.root.glob("matched-[0-9][0-9]-*.json")):
    row = json.loads(path.read_text())
    ordinal = path.name.split("-")[1]
    assert ordinal not in runs
    assert row["status"] == "passed" and not row["profiles"]
    assert (
        row["cache_before"] == row["cache_after_startup"] == row["cache_after_serving"]
    )
    assert row["bare_identity"] == "None"
    assert all(c["profile_delta"] == 0 for c in row["captures"])
    assert len(row["numerics"]) == 30
    assert all(c["allclose"] and c["argmax_equal"] for c in row["numerics"])
    runs[ordinal] = row
assert sorted(runs) == ["01", "02", "03"]
receipt = json.loads((a.root / "matched-cache-receipt.json").read_text())
assert all(
    hashlib.sha256((a.root / name).read_bytes()).hexdigest() == value
    for name, value in receipt["original_cache_hashes"].items()
)
metrics = []
for before, scoped, after in zip(*(runs[k]["performance"] for k in ["01", "02", "03"])):
    assert (
        (before["batch"], before["length"])
        == (scoped["batch"], scoped["length"])
        == (after["batch"], after["length"])
    )
    for metric in ["graph_ms", "decode_ms"]:
        ratio = statistics.median(scoped[metric]) / math.sqrt(
            statistics.median(before[metric]) * statistics.median(after[metric])
        )
        metrics.append(
            {
                "batch": before["batch"],
                "length": before["length"],
                "metric": metric,
                "bracketed_ratio": ratio,
                "change_percent": 100 * (ratio - 1),
            }
        )
paths = {p.name.split("-")[1]: p for p in a.root.glob("matched-[0-9][0-9]-*.npz")}
comparisons = []
for left, right in [("01", "02"), ("02", "03"), ("01", "03")]:
    with np.load(paths[left]) as x, np.load(paths[right]) as y:
        assert set(x.files) == set(y.files) and len(x.files) == 60
        for key in sorted(x.files):
            assert np.isfinite(x[key]).all() and np.isfinite(y[key]).all()
            comparisons.append(
                {
                    "left": left,
                    "right": right,
                    "key": key,
                    "exact_equal": bool(np.array_equal(x[key], y[key])),
                    "allclose": bool(np.allclose(x[key], y[key], atol=0.05, rtol=0.01)),
                    "argmax_equal": bool(
                        np.array_equal(x[key].argmax(-1), y[key].argmax(-1))
                    ),
                    "max_abs_error": float(np.abs(x[key] - y[key]).max()),
                }
            )
result = {
    "purpose": receipt["purpose"],
    "original_caches_unchanged": True,
    "numerical_comparisons": sum(len(r["numerics"]) for r in runs.values()),
    "metrics": metrics,
    "comparisons": comparisons,
    "allclose_failures": sum(not r["allclose"] for r in comparisons),
    "exact_comparisons": sum(r["exact_equal"] for r in comparisons),
    "limitations": "One scoped warm process bracketed by two legacy warm processes; seeded cache copies are a diagnostic, not natural selection results or model accuracy equivalence.",
}
a.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "comparisons"}, indent=2))
