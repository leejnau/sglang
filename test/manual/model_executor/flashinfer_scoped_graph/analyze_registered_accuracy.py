"""Audit the unchanged registered accuracy test and observed serving paths."""

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--results", type=Path, required=True)
p.add_argument("--source", type=Path, required=True)
p.add_argument("--dataset", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
source = a.source / "python/sglang/test/simple_eval_mixed_prefix_gsm8k.py"
tree = ast.parse(source.read_text())
names = {"get_one_example", "get_few_shot_examples", "get_answer_value"}
module = ast.Module(
    body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names],
    type_ignores=[],
)
namespace = {"ast": ast, "re": re, "INVALID": -9999999}
exec(compile(module, str(source), "exec"), namespace)
rows = [json.loads(s) for s in a.dataset.read_text().splitlines()]
prefix = namespace["get_few_shot_examples"](rows, 5)
expected = rows[5:205]
results = {}
for arm in ["legacy", "scoped"]:
    root = a.results / f"step1-{arm}"
    test = json.loads((root / "test-result.json").read_text())
    data = json.loads((root / "evaluation.json").read_text())
    assert test["tests_run"] == 1 and not test["skipped"]
    assert len(data["convos"]) == 200
    predictions, correct = [], []
    for index, convo in enumerate(data["convos"]):
        prompt = prefix + namespace["get_one_example"](expected, index, False)
        assert convo[0]["content"] == prompt
        value = namespace["get_answer_value"](convo[-1]["content"])
        predictions.append(value)
        correct.append(
            value == namespace["get_answer_value"](expected[index]["answer"])
        )
    assert sum(correct) / 200 == data["score"]
    events = [
        json.loads(line)
        for f in root.glob("observer-*.jsonl")
        for line in f.read_text().splitlines()
    ]
    captures = [r for r in events if r["event"] == "capture"]
    replays = [r for r in events if r["event"] == "graph_replay"]
    selections = [r for r in events if r["event"] == "selection"]
    strict = [r for r in selections if r["strict"]]
    assert captures and replays
    assert all(r["profile_delta"] == 0 for r in captures)
    if arm == "scoped":
        assert strict and all("decode_graph" in r["identity"] for r in strict)
        assert all(r["strict"] and r["report"]["misses"] == 0 for r in captures)
    else:
        assert not strict
    results[arm] = {
        "test_successful": test["successful"],
        "score": data["score"],
        "correct_count": sum(correct),
        "correct": correct,
        "predictions": predictions,
        "capture_count": len(captures),
        "strict_selection_count": len(strict),
        "graph_replays_observed_at_least": max(r["count"] for r in replays),
        "zero_profiles_during_capture": True,
        "test_source_sha256": test["source_sha256"],
        "evaluation_sha256": hashlib.sha256(
            (root / "evaluation.json").read_bytes()
        ).hexdigest(),
    }
left, right = results["legacy"], results["scoped"]
summary = {
    "arms": results,
    "gains": sum(not x and y for x, y in zip(left["correct"], right["correct"])),
    "losses": sum(x and not y for x, y in zip(left["correct"], right["correct"])),
    "identical_answers": sum(
        x == y for x, y in zip(left["predictions"], right["predictions"])
    ),
    "all_200_prompts_match_in_both_arms": True,
    "scorer_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "dataset_sha256": hashlib.sha256(a.dataset.read_bytes()).hexdigest(),
    "limits": "One cold launch per arm with independent selection and default server seeds; observer instrumentation is not a serving-performance measurement. The established floor does not prove statistical equivalence.",
}
a.output.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "arms"}, indent=2))
print(
    {
        arm: {
            k: r[k]
            for k in [
                "test_successful",
                "score",
                "capture_count",
                "strict_selection_count",
                "graph_replays_observed_at_least",
            ]
        }
        for arm, r in results.items()
    }
)
