"""Analyze the predeclared paired model-runner cohort without dropping runs."""

import argparse
import json
import math
import random
import statistics as st
from pathlib import Path


def geo(values):
    return math.exp(st.mean(math.log(x) for x in values))


def analyze(root):
    rows = {}
    for p in sorted(root.glob("matrix-*.json")):
        row = json.loads(p.read_text())
        assert row["status"] == "passed", p.name
        ordinal = p.name.split("-")[1]
        assert ordinal not in rows
        rows[ordinal] = row
    assert len(rows) == 8, f"incomplete cohort: {len(rows)}/8"
    pairs = [("01", "02"), ("04", "03"), ("06", "05"), ("07", "08")]
    result = {
        "pairs": pairs,
        "runs": [],
        "metrics": [],
        "candidate_choices": [],
        "limitations": [
            "One H100 on a shared host, one dense ModelOpt FP8 model.",
            "Four process pairs reuse two independent cache seeds per arm; repeats are not independent tuning trials.",
            "Intervals below resample the two seed clusters, preserving each cold/warm pair; limited statistical power.",
            "ModelRunner decode wall time does not measure HTTP serving throughput.",
        ],
    }
    for ordinal, row in rows.items():
        args = row["arguments"]
        checks = row["numerics"]
        assert len(checks) == 30 and all(
            c["allclose"] and c["argmax_equal"] for c in checks
        )
        assert row["cache_after_startup"] == row["cache_after_serving"]
        if args["phase"] == "warm":
            assert not row["profiles"]
            assert row["cache_before"] == row["cache_after_startup"]
        capture = [s for s in row["selections"] if s["capturing"]]
        assert len(capture) == 384, (ordinal, len(capture))
        assert all(
            s["tactic"] == "-1"
            or (
                s["native_tactic_count"] is not None
                and int(s["tactic"]) < s["native_tactic_count"]
            )
            for s in capture
        )
        choices = {}
        for s in capture:
            assert s["op"] == "fp8_gemm"
            a, b, *_ = s["input_shapes"]
            key = f"M{a[-2]}_N{b[-1]}_K{a[-1]}"
            if key in choices:
                assert choices[key] == s["tactic"]
            choices[key] = s["tactic"]
        result["candidate_choices"].append(
            {
                "ordinal": ordinal,
                "arm": args["arm"],
                "phase": args["phase"],
                "choices": choices,
            }
        )
        result["runs"].append(
            {
                "ordinal": ordinal,
                "arm": args["arm"],
                "phase": args["phase"],
                "startup_seconds": row["startup_seconds"],
                "profile_count": len(row["profiles"]),
                "profile_wall_seconds": sum(v["seconds"] for v in row["profiles"]),
                "startup_peak_memory_bytes": row["startup_peak_memory_bytes"],
                "numerical_comparisons": len(checks),
                "max_eager_graph_abs_error": max(c["max_abs_error"] for c in checks),
                "capture_lookups": len(capture),
                "explicit_capture_choices": sum(s["tactic"] != "-1" for s in capture),
                "strict_hits": sum(
                    (c.get("coverage") or {}).get("hits", 0) for c in row["captures"]
                ),
                "scoped": args["arm"] == "scoped",
            }
        )
    indexed = {
        n: {(r["batch"], r["length"]): r for r in row["performance"]}
        for n, row in rows.items()
    }
    random.seed(20261109)
    for key in ((b, l) for b in (1, 4, 16) for l in (32, 256)):
        for metric in ("graph_ms", "decode_ms"):
            pair_rows = []
            for a, b in pairs:
                av, bv = (st.median(indexed[n][key][metric]) for n in (a, b))
                pair_rows.append(
                    {
                        "baseline": a,
                        "scoped": b,
                        "baseline_ms": av,
                        "scoped_ms": bv,
                        "ratio": bv / av,
                    }
                )
            ratios = [r["ratio"] for r in pair_rows]
            seed_logs = [
                st.mean(math.log(x) for x in ratios[start : start + 2])
                for start in (0, 2)
            ]
            boots = sorted(
                math.exp(st.mean(random.choices(seed_logs, k=2))) for _ in range(10000)
            )
            ci = [boots[249], boots[9749]]
            ratio = geo(ratios)
            result["metrics"].append(
                {
                    "batch": key[0],
                    "length": key[1],
                    "metric": metric,
                    "paired_geomean_ratio": ratio,
                    "cluster_bootstrap_95": ci,
                    "pairs": pair_rows,
                    "requires_investigation": ratio > 1.03 or ci[1] > 1.03,
                }
            )
    result["investigation_needed"] = any(
        x["requires_investigation"] for x in result["metrics"]
    )
    result["numerical_total"] = sum(r["numerical_comparisons"] for r in result["runs"])
    result["source_hashes_by_run"] = {k: v["source_files"] for k, v in rows.items()}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = analyze(a.root)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["metrics"]:
        print(
            row["batch"],
            row["length"],
            row["metric"],
            round((row["paired_geomean_ratio"] - 1) * 100, 3),
            [round((x - 1) * 100, 3) for x in row["cluster_bootstrap_95"]],
            row["requires_investigation"],
        )
    print("investigation needed", result["investigation_needed"])


if __name__ == "__main__":
    main()
