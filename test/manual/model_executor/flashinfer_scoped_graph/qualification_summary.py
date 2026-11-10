"""Summarize paired answers and descriptive timings without dropping failures."""

import argparse
import json
import math
import statistics
from pathlib import Path


def accuracy(root):
    a, b = [
        json.loads((root / f"accuracy-{arm}.json").read_text())
        for arm in ("legacy", "scoped")
    ]
    assert all(
        d["status"] == "passed"
        and len(d["requests"]) == 200
        and d["cache_before"] == d["cache_after"]
        for d in (a, b)
    )
    changes = []
    for x, y in zip(a["requests"], b["requests"]):
        assert (x["index"], x["prompt_sha256"], x["expected"]) == (
            y["index"],
            y["prompt_sha256"],
            y["expected"],
        )
        changes.append(
            {
                "index": x["index"],
                "legacy_correct": x["correct"],
                "scoped_correct": y["correct"],
                "same_answer": x["predicted"] == y["predicted"],
                "same_text": x["text_sha256"] == y["text_sha256"],
            }
        )
    return {
        "questions": 200,
        "legacy_correct": a["correct"],
        "scoped_correct": b["correct"],
        "legacy_score": a["score"],
        "scoped_score": b["score"],
        "scoped_gains": sum(
            not r["legacy_correct"] and r["scoped_correct"] for r in changes
        ),
        "scoped_losses": sum(
            r["legacy_correct"] and not r["scoped_correct"] for r in changes
        ),
        "same_answers": sum(r["same_answer"] for r in changes),
        "same_texts": sum(r["same_text"] for r in changes),
        "paired_results": changes,
        "limits": "One preserved natural cache pair, one model and a fixed 200-question subset. Different from the repository CI benchmark; no general accuracy superiority or equivalence claim.",
    }


def timing(root):
    paths = [
        "timing-01-legacy.json",
        "timing-02-scoped.json",
        "timing-03-scoped.json",
        "timing-04-legacy.json",
    ]
    runs = [json.loads((root / p).read_text()) for p in paths]
    summaries = []
    for name, d in zip(paths, runs):
        assert d["status"] in {"passed", "completed_with_numerical_differences"}
        assert (
            not d["profiles"]
            and d["cache_before"]
            == d["cache_after_startup"]
            == d["cache_after_serving"]
        )
        assert (
            all(x["argmax_equal"] for x in d["numerics"]) and len(d["numerics"]) == 30
        )
        summaries.append(
            {
                "file": name,
                "status": d["status"],
                "allclose_failures": sum(not x["allclose"] for x in d["numerics"]),
                "max_abs_logit_difference": max(
                    x["max_abs_error"] for x in d["numerics"]
                ),
                "profile_count": len(d["profiles"]),
                "capture_preparation_profile_deltas": d[
                    "capture_preparation_profile_deltas"
                ],
            }
        )
    metrics = []
    indexed = [{(v["batch"], v["length"]): v for v in d["performance"]} for d in runs]
    for key in indexed[0]:
        for metric in ("graph_ms", "decode_ms"):
            pairs = []
            for a, b in [(0, 1), (3, 2)]:
                left, right = (
                    statistics.median(indexed[a][key][metric]),
                    statistics.median(indexed[b][key][metric]),
                )
                pairs.append(
                    {
                        "legacy_file": paths[a],
                        "scoped_file": paths[b],
                        "legacy_ms": left,
                        "scoped_ms": right,
                        "ratio": right / left,
                    }
                )
            metrics.append(
                {
                    "batch": key[0],
                    "length": key[1],
                    "metric": metric,
                    "paired_geomean_ratio": math.exp(
                        statistics.mean(math.log(p["ratio"]) for p in pairs)
                    ),
                    "pairs": pairs,
                }
            )
    return {
        "runs": summaries,
        "metrics": metrics,
        "limits": "Two process observations per arm, one preserved cache pair; descriptive warm timings, no independent retuning trials, HTTP throughput or statistical equivalence claim. Logit differences remain explicitly flagged.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--accuracy-only", action="store_true")
    args = p.parse_args()
    result = {"accuracy": accuracy(args.root)}
    if not args.accuracy_only:
        result["timing"] = timing(args.root)
    name = (
        "answer-accuracy-comparison.json"
        if args.accuracy_only
        else "qualification-summary.json"
    )
    (args.root / name).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result["accuracy"].items() if k != "paired_results"},
            indent=2,
        )
    )
    if "timing" in result:
        for row in result["timing"]["metrics"]:
            print(
                row["batch"],
                row["length"],
                row["metric"],
                round(100 * (row["paired_geomean_ratio"] - 1), 3),
            )


if __name__ == "__main__":
    main()
