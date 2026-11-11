"""Report every declared run; keep numerical discrepancies separate from lifecycle checks."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np


def analyze(root):
    runs = {}
    for path in sorted(root.glob("refresh-[0-9][0-9]-*.json")):
        row = json.loads(path.read_text())
        ordinal = path.name.split("-")[1]
        assert ordinal not in runs
        assert row["status"] in ("passed", "completed_with_numerical_differences"), path
        assert len(row["numerics"]) == 30
        assert all(c["argmax_equal"] for c in row["numerics"])
        assert row["cache_after_startup"] == row["cache_after_serving"]
        assert all(c["profile_delta"] == 0 for c in row["captures"])
        assert row["bare_identity"] == "None"
        if row["arguments"]["phase"] == "warm":
            assert not row["profiles"]
            assert row["cache_before"] == row["cache_after_startup"]
        runs[ordinal] = row
    assert sorted(runs) == ["01", "02", "03", "04", "05", "06"]
    summaries = []
    for ordinal, row in runs.items():
        summaries.append(
            {
                "ordinal": ordinal,
                "arm": row["arguments"]["arm"],
                "phase": row["arguments"]["phase"],
                "status": row["status"],
                "startup_seconds": row["startup_seconds"],
                "profiles": len(row["profiles"]),
                "captures": len(row["captures"]),
                "cache_files": len(row["cache_after_startup"]),
                "numerical_comparisons": len(row["numerics"]),
                "allclose_failures": sum(not c["allclose"] for c in row["numerics"]),
                "max_eager_graph_abs_error": max(
                    c["max_abs_error"] for c in row["numerics"]
                ),
                "strict_capture_hits": sum(
                    (c.get("coverage") or {}).get("hits", 0) for c in row["captures"]
                ),
                "source_files": row["source_files"],
            }
        )
    metrics = []
    pairs = [("03", "04"), ("06", "05")]
    indexed = {
        k: {(r["batch"], r["length"]): r for r in v["performance"]}
        for k, v in runs.items()
        if v["arguments"]["phase"] == "warm"
    }
    for shape in [(b, l) for b in [1, 4, 16] for l in [32, 256]]:
        for metric in ["graph_ms", "decode_ms"]:
            ratios = [
                statistics.median(indexed[b][shape][metric])
                / statistics.median(indexed[a][shape][metric])
                for a, b in pairs
            ]
            ratio = math.exp(statistics.mean(math.log(r) for r in ratios))
            metrics.append(
                {
                    "batch": shape[0],
                    "length": shape[1],
                    "metric": metric,
                    "paired_ratios": ratios,
                    "geomean_ratio": ratio,
                    "change_percent": 100 * (ratio - 1),
                    "investigate": ratio > 1.03,
                }
            )
    paths = {p.name.split("-")[1]: p for p in root.glob("refresh-[0-9][0-9]-*.npz")}
    comparisons = []
    for left, right in [
        ("01", "03"),
        ("03", "06"),
        ("02", "04"),
        ("04", "05"),
        ("03", "04"),
        ("06", "05"),
    ]:
        with np.load(paths[left]) as a, np.load(paths[right]) as b:
            assert set(a.files) == set(b.files) and len(a.files) == 60
            for key in sorted(a.files):
                x, y = a[key], b[key]
                assert np.isfinite(x).all() and np.isfinite(y).all()
                comparisons.append(
                    {
                        "left": left,
                        "right": right,
                        "key": key,
                        "max_abs_error": float(np.abs(x - y).max()),
                        "allclose": bool(np.allclose(x, y, atol=0.05, rtol=0.01)),
                        "argmax_equal": bool(
                            np.array_equal(x.argmax(-1), y.argmax(-1))
                        ),
                        "exact_equal": bool(np.array_equal(x, y)),
                    }
                )
    return {
        "runs": summaries,
        "metrics": metrics,
        "comparisons": comparisons,
        "performance_investigation": any(m["investigate"] for m in metrics),
        "cross_process_allclose_failures": sum(not c["allclose"] for c in comparisons),
        "cross_process_argmax_failures": sum(
            not c["argmax_equal"] for c in comparisons
        ),
        "limits": "One H100/shared host, one model, one cold cache pair, two warm process observations per arm. Descriptive latency; no equivalence, universal quality or serving-throughput claim.",
        "source_result_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.glob("refresh-[0-9][0-9]-*.json")
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ["comparisons", "source_result_hashes"]
            },
            indent=2,
        )
    )
