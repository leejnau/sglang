#!/usr/bin/env python3
"""CPU schema mutations and independent two-candidate arithmetic controls.

Historical records supply schema only; artificial values never enter inference.
"""

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def run(args):
    here = Path(__file__).resolve().parent
    a = module(args.analyzer, "mlp_analysis_controls")
    old_dir = here.parent / "20261026T230000Z-h100-finalist-confirmation"
    old_controls = module(old_dir / "analyzer_controls.py", "inherited_controls")
    # Recheck the unchanged decision/cache/timing controls before new geometry.
    inherited = old_controls.run(
        SimpleNamespace(
            analyzer=old_dir / "analyzer.py",
            prior_analyzer=here.parent
            / "20261026T220000Z-h100-fixed-pair"
            / "analyzer.py",
            historical_contract=old_dir / "historical-contract.json",
            historical_result=args.historical_result,
            schedule=old_dir / "schedule.json",
        )
    )
    passed = ["unchanged inherited confirmation controls: " + str(inherited["count"])]

    def reject(label, value, mutate, validate):
        changed = copy.deepcopy(value)
        mutate(changed)
        try:
            validate(changed)
        except (ValueError, KeyError):
            passed.append(label)
        else:
            raise AssertionError(label + " incorrectly accepted")

    prior, leaf = a.helpers(
        here.parent / "20261026T220000Z-h100-fixed-pair" / "analyzer.py"
    )
    contract = a.load(here / "historical-contract.json")
    schedule = a.load(here / "schedule.json")
    data = a.load(args.historical_result)
    rows = data["startup"]["profiles"]
    for row in rows:
        m, ordinal = row["m"], row["index"] % 9
        row["tactic"] = contract["candidate_order"][str(m)][ordinal]
        row["key"] = contract["profile_keys"][row["index"]]
        row["elapsed_ms"] = 1.0 if ordinal == 0 else 2.0 + ordinal
        row["metadata"][0]["shape"] = [1, m, 3072]
        row["metadata"][0]["stride"] = [m * 3072, 3072, 1]
        row["metadata"][1]["shape"] = [1, 3072, 1024]
        row["metadata"][1]["stride"] = [3072, 1, 3072]
        row["metadata"][4]["shape"] = [1, m, 1024]
        row["metadata"][4]["stride"] = [m * 1024, 1024, 1]
    data["startup"]["candidate_order"] = copy.deepcopy(contract["candidate_order"])
    for row in data["startup"]["initializer_calls"]:
        row["shape"][2] = 3072
        row.update(cloned=False, copied=False, returned_same_object=True)
    data["natural"].update(tactic=-1, key=rows[72]["key"])
    data["original_history"]["default_ms"] = rows[72]["elapsed_ms"]
    data["retained_profile"]["metadata"] = copy.deepcopy(rows[80]["metadata"])
    data = old_controls.artificial_fixture(a, data, schedule)
    check_startup = lambda d: a.validate_startup(d, contract, leaf)
    check_startup(data)
    passed.append("actual MLP candidate/key contract with no-copy synthetic schema")
    reject(
        "old projection geometry rejected",
        data,
        lambda d: d["startup"]["profiles"][0]["metadata"][0].update(shape=[1, 1, 1024]),
        check_startup,
    )
    reject(
        "matched clone rejected",
        data,
        lambda d: d["startup"]["initializer_calls"][8].update(cloned=True),
        check_startup,
    )
    reject(
        "matched copy rejected",
        data,
        lambda d: d["startup"]["initializer_calls"][8].update(copied=True),
        check_startup,
    )
    reject(
        "new generated object rejected",
        data,
        lambda d: d["startup"]["initializer_calls"][8].update(
            returned_same_object=False
        ),
        check_startup,
    )
    reject(
        "old candidate substitution rejected",
        data,
        lambda d: d["startup"]["profiles"][73].update(tactic=[0] * 8),
        check_startup,
    )
    reject(
        "timing startup byte observation rejected",
        data,
        lambda d: d["startup"]["initializer_calls"][0].update(
            generated_sha256="0" * 64
        ),
        check_startup,
    )

    graph_data = copy.deepcopy(data)
    explicit = data["finalist"]["tactic"]
    for graph in graph_data["graphs"]:
        tactic = explicit if graph["id"] == "winner" else -1
        graph["tactic"] = graph["selection"]["tactic"] = tactic
        graph["inputs"][0].update(shape=[1, 144, 3072], stride=[144 * 3072, 3072, 1])
        graph["inputs"][1].update(shape=[1, 3072, 1024], stride=[3072, 1, 3072])
        graph["output"].update(shape=[144, 1024], stride=[1024, 1])
        graph["read_operand_hashes"][0] = contract["fixture"]["quantized_input_sha256"]
    graph_contract = dict(contract, fixed_descriptor=explicit)
    graph_checks = leaf.correctness_index(graph_data)
    check_graphs = lambda d: a.validate_graphs(d, graph_contract, leaf, graph_checks)
    check_graphs(graph_data)
    passed.append("MLP graph geometry uses actual finalist and owned outputs")
    reject(
        "old graph projection rejected",
        graph_data,
        lambda d: d["graphs"][0]["inputs"][0].update(shape=[1, 144, 1024]),
        check_graphs,
    )
    reject(
        "wrong MLP weight layout rejected",
        graph_data,
        lambda d: d["graphs"][0]["inputs"][1].update(stride=[3072, 1024, 1]),
        check_graphs,
    )
    reject(
        "different captured activation rejected",
        graph_data,
        lambda d: d["graphs"][0]["read_operand_hashes"].__setitem__(0, "0" * 64),
        check_graphs,
    )

    def process(block, original, confirmed, graph_ratio):
        log = math.log(graph_ratio)
        return {
            "block": block,
            "capture_order": "winner-first" if block % 2 == 0 else "default-first",
            "original_role": original,
            "confirmed_role": confirmed,
            "original_profiler_ratio": 1.1,
            "graph_log_ratio": log,
            "confirmation_log_ratio": math.log(0.99),
            **a.choice_effect(original, confirmed, log),
        }

    cases = [
        ("winner", "default", 1.2),
        ("default", "winner", 1.2),
        ("winner", "winner", 1.2),
        ("default", "default", 1.2),
        ("default", "winner", 0.8),
        ("winner", "default", 0.8),
        ("default", "default", 0.8),
        ("winner", "winner", 0.8),
    ]
    mixed = [process(i, *case) for i, case in enumerate(cases)]
    counts = a.slow_choice_counts(mixed)
    assert counts == {
        "original_slow": 4,
        "confirmed_slow": 4,
        "corrected": 2,
        "introduced": 2,
        "remaining_slow": 2,
        "neither_slow": 2,
        "graph_point_ties": 0,
    }
    passed.append("all eight choice directions counted independently")
    result = a.summarize(mixed, replicates=100)
    a.close(
        result["metrics"]["hypothetical_over_original"]["geometric_mean_ratio"],
        1,
        "reciprocal corrected/introduced effects balance",
    )
    assert result["checkpoint_disposition"]["result"] == "introduced_slow_choices_stop"
    passed.append("adverse mixed outcomes force stop and remain in estimates")
    improved = [process(i, "winner", "default", 1.2) for i in range(8)]
    result = a.summarize(improved, replicates=100)
    assert result["slow_choice_counts"]["corrected"] == 8
    assert result["checkpoint_disposition"]["result"] == "supports_focused_review"
    a.close(
        result["metrics"]["hypothetical_over_original"]["geometric_mean_ratio"],
        1 / 1.2,
        "correction ratio matches independent arithmetic",
    )
    passed.append("uniform correction passes predeclared review gate")
    for role in ("default", "winner"):
        unchanged = [process(i, role, role, 1.2) for i in range(8)]
        result = a.summarize(unchanged, replicates=100)
        assert (
            result["metrics"]["hypothetical_over_original"]["geometric_mean_ratio"] == 1
        )
        assert (
            result["checkpoint_disposition"]["result"] == "neutral_or_ineffective_stop"
        )
    passed.append("unchanged default and unchanged slow winner both stop")
    ties = [process(i, "winner", "default", 1.0) for i in range(8)]
    counts = a.slow_choice_counts(ties)
    assert counts["graph_point_ties"] == counts["neither_slow"] == 8
    assert counts["corrected"] == counts["introduced"] == 0
    passed.append("exact graph ties are not counted as corrections")
    metric = {"exploratory_process_bootstrap_95pct": [0.9, 1.0]}
    counts = a.slow_choice_counts(improved)
    assert a.disposition(counts, metric)["result"] == "promising_but_inconclusive_stop"
    passed.append("CI touching one cannot pass review gate")
    metric["exploratory_process_bootstrap_95pct"][1] = 0.999999
    assert a.disposition(counts, metric)["result"] == "supports_focused_review"
    passed.append("strict CI rule has no rounded timing margin")
    reject(
        "manipulated choice effect rejected",
        improved,
        lambda p: p[0].update(log_hypothetical_over_original=0),
        a.summarize,
    )
    reject("missing process rejected", improved, lambda p: p.pop(), a.summarize)
    reject(
        "duplicate process rejected",
        improved,
        lambda p: p[7].update(block=0),
        a.summarize,
    )
    reject(
        "nonfinite replay score rejected",
        improved,
        lambda p: p[0].update(graph_log_ratio=float("nan")),
        a.summarize,
    )
    a.validate_schedule(schedule, prior)
    passed.append("eight-process balanced schedule retained")
    return {
        "status": "passed",
        "cpu_only": True,
        "historical_timing_pooled": False,
        "artificial_values_for_schema_and_math_only": True,
        "inherited_controls": inherited["count"],
        "new_controls": len(passed) - 1,
        "count": inherited["count"] + len(passed) - 1,
        "checks": passed,
        "analyzer_sha256": a.sha256(args.analyzer),
        "control_sha256": a.sha256(__file__),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--analyzer", type=Path, default=here / "analyzer.py")
    parser.add_argument("--historical-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "count": result["count"]}))
