#!/usr/bin/env python3
"""CPU mutation controls for finalist identity, decision freeze and paired scoring.

The historical raw file supplies schema only. Artificial confirmation times
exercise rejection and arithmetic; none enter performance inference.
"""

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path


def read_module(path):
    spec = importlib.util.spec_from_file_location("finalist_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def artificial_fixture(a, original, schedule):
    data = copy.deepcopy(original)
    rows = data["startup"]["profiles"]
    explicit = min(rows[73:], key=lambda r: r["elapsed_ms"])
    natural = min(rows[72:], key=lambda r: r["elapsed_ms"])
    end = data["startup"]["ended_ns"]
    data["retained_profile"]["retained_after_ns"] = end - 20
    data["startup_choices"] = [
        {
            "op": "fp8_gemm",
            "profile_attempts": 81,
            "tactic": natural["tactic"],
            "key": natural["key"],
            "requested_runner_identity": 100,
            "returned_runner_identity": 100,
            "returned_ns": end - 10,
        }
    ]
    data["finalist"] = {
        "tactic": explicit["tactic"],
        "profile_index": explicit["index"],
        "key": explicit["key"],
        "elapsed_ms": explicit["elapsed_ms"],
    }
    data["natural"].update(
        first_pass_profile_index=natural["index"],
        frozen_ns=end + 10,
        cache_sha256=data["startup"]["cache_sha256"],
        cache_state_sha256=data["startup"]["cache_state_sha256"],
        cache_preserved_after_confirmation=True,
    )
    data["original_history"].update(
        explicit_index=explicit["index"],
        explicit_ms=explicit["elapsed_ms"],
        explicit_over_default=explicit["elapsed_ms"] / rows[72]["elapsed_ms"],
    )
    now = end + 10

    def tick():
        nonlocal now
        now += 10
        return now

    retained = data["retained_profile"]
    retained.update(
        reference_preparation_started_ns=tick(),
        reference_preparation_ended_ns=tick(),
        after_confirmation_metadata=copy.deepcopy(retained["metadata"]),
    )
    group = {
        "objective": "auto",
        "settings": copy.deepcopy(rows[72]["settings"]),
        "started_ns": tick(),
        "preparation": {
            "ended_ns": tick(),
            "public_schedule_uses_exact_retained_objects": True,
        },
        "rounds": [],
    }
    data["confirmation"] = group
    data["diagnostic_profile_attempts"], data["profile_correctness"] = [], []
    for index, order in enumerate(schedule["confirmation_pair_schedules"]["0"]):
        pair = {"index": index, "order": order, "measurements": []}
        for role in order:
            label = f"confirmation_{index:02d}_{role}"
            tactic = explicit["tactic"] if role == "winner" else -1
            elapsed = 0.005 if role == "winner" else 0.01
            cursor = len(data["diagnostic_profile_attempts"])
            data["diagnostic_profile_attempts"].append(
                {
                    "index": cursor,
                    "stage": label,
                    "objective": "auto",
                    "tactic": tactic,
                    "settings": copy.deepcopy(group["settings"]),
                    "metadata": copy.deepcopy(retained["metadata"]),
                    "tensor_object_ids": retained["tensor_object_ids"][:],
                    "started_ns": tick(),
                    "ended_ns": tick(),
                    "elapsed_ms": elapsed,
                    "error": None,
                }
            )
            data["profile_correctness"].append(
                {
                    "check": label,
                    "passed": True,
                    "output_data_ptr": retained["metadata"][4]["data_ptr"],
                    "max_abs_error": 0,
                    "exact_zero_required": False,
                    "checked_ns": tick(),
                }
            )
            pair["measurements"].append(
                {
                    "role": role,
                    "tactic": tactic,
                    "elapsed_ms": elapsed,
                    "attempt_index": cursor,
                    "correctness_check": label,
                }
            )
        group["rounds"].append(pair)
    group["ended_ns"] = tick()
    payload = a.expected_decision(group["rounds"], explicit["tactic"], tick())
    group.update(payload)
    group["decision_payload"] = copy.deepcopy(payload)
    group["decision_sha256"] = group["initial_decision_sha256"] = a.digest(payload)
    data["final"]["decision_sha256"] = a.digest(payload)
    data["natural"]["confirmed_after_ns"] = tick()
    for graph in data["graphs"]:
        graph["confirmation_decision_sha256"] = a.digest(payload)
    data["graph_choices"] = {
        "natural_role": "default" if natural["tactic"] == -1 else "winner",
        "confirmation_role": payload["decision_role"],
        "both_evaluated_on_same_retained_graphs": True,
    }
    data["profile_correctness"].extend(
        copy.deepcopy(
            [
                r
                for r in original["profile_correctness"]
                if r["check"].startswith("profile_accept_")
            ]
        )
    )
    data["cache_partition_restorations"] = data["cache_partition_restorations"][:1]
    return data


def run(args):
    a = read_module(args.analyzer)
    prior, leaf = a.helpers(args.prior_analyzer)
    contract, schedule = a.load(args.historical_contract), a.load(args.schedule)
    original = a.load(args.historical_result)
    data = artificial_fixture(a, original, schedule)
    passed = []

    def reject(label, value, mutate, validate):
        changed = copy.deepcopy(value)
        mutate(changed)
        try:
            validate(changed)
        except ValueError:
            passed.append(label)
        else:
            raise AssertionError(label + " incorrectly accepted")

    startup = lambda d: a.validate_startup(d, contract, leaf)
    startup(data)
    passed.append(
        "real startup schema accepts dynamic best explicit different from old fixed descriptor"
    )
    a.require(
        data["finalist"]["tactic"] != contract["fixed_descriptor"],
        "fixture exercises dynamic finalist",
    )
    reject(
        "old fixed descriptor substituted",
        data,
        lambda d: d["finalist"].update(tactic=contract["fixed_descriptor"]),
        startup,
    )
    reject(
        "nonminimum explicit selected",
        data,
        lambda d: d["finalist"].update(profile_index=73),
        startup,
    )
    reject(
        "natural selection overwritten",
        data,
        lambda d: d["natural"].update(tactic=d["finalist"]["tactic"]),
        startup,
    )
    reject(
        "chooser return changed",
        data,
        lambda d: d["startup_choices"][0].update(tactic=d["finalist"]["tactic"]),
        startup,
    )
    reject(
        "chooser runner changed",
        data,
        lambda d: d["startup_choices"][0].update(returned_runner_identity=101),
        startup,
    )
    reject(
        "original candidate history changed",
        data,
        lambda d: d["startup"]["profiles"].pop(),
        startup,
    )
    reject(
        "candidate failure silently skipped",
        data,
        lambda d: d["startup"]["profiles"][80].update(error="failure"),
        startup,
    )
    reject(
        "natural cache not preserved",
        data,
        lambda d: d["natural"].update(cache_preserved_after_confirmation=False),
        startup,
    )

    confirmation = lambda d: a.validate_confirmation(
        d, schedule, contract, d["finalist"]["tactic"]
    )
    confirmation(data)
    passed.append(
        "artificial24-call confirmation validates without performance inference"
    )
    reject(
        "finalist silently replaced during confirmation",
        data,
        lambda d: d["diagnostic_profile_attempts"][0].update(tactic=[0] * 8),
        confirmation,
    )
    reject(
        "retained tensor replaced",
        data,
        lambda d: d["diagnostic_profile_attempts"][0]["tensor_object_ids"].__setitem__(
            0, 1
        ),
        confirmation,
    )
    reject(
        "auto objective silently changed",
        data,
        lambda d: d["diagnostic_profile_attempts"][0]["settings"]["config"].update(
            use_cuda_graph=True
        ),
        confirmation,
    )
    reject(
        "missing confirmation pair",
        data,
        lambda d: d["confirmation"]["rounds"].pop(),
        confirmation,
    )
    reject(
        "duration substituted",
        data,
        lambda d: d["confirmation"]["rounds"][0]["measurements"][0].update(
            elapsed_ms=0.123
        ),
        confirmation,
    )
    reject(
        "nonfinite profiling result",
        data,
        lambda d: d["diagnostic_profile_attempts"][0].update(elapsed_ms=float("nan")),
        confirmation,
    )
    reject(
        "no-write numerical control fails",
        data,
        lambda d: d["profile_correctness"][0].update(passed=False),
        confirmation,
    )
    reject(
        "stale output check owner",
        data,
        lambda d: d["profile_correctness"][0].update(output_data_ptr=1),
        confirmation,
    )
    reject(
        "decision timestamp after graph capture",
        data,
        lambda d: d["confirmation"].update(
            decided_ns=d["graphs"][0]["capture_ended_ns"]
        ),
        confirmation,
    )
    reject(
        "decision role rewritten after confirmation",
        data,
        lambda d: d["confirmation"].update(decision_role="default"),
        confirmation,
    )
    reject(
        "initial decision digest changed",
        data,
        lambda d: d["confirmation"].update(initial_decision_sha256="changed"),
        confirmation,
    )
    reject(
        "graph bound to different decision",
        data,
        lambda d: d["graphs"][0].update(confirmation_decision_sha256="changed"),
        confirmation,
    )
    reject(
        "final decision differs",
        data,
        lambda d: d["final"].update(decision_sha256="changed"),
        confirmation,
    )
    reject(
        "hypothetical winner published to cache",
        data,
        lambda d: d["cache_partition_restorations"][0].update(after_sha256="changed"),
        confirmation,
    )

    rounds = copy.deepcopy(data["confirmation"]["rounds"])
    for i, pair in enumerate(rounds):
        for row in pair["measurements"]:
            row["elapsed_ms"] = (0.01 if i < 6 else 3) if row["role"] == "winner" else 1
    decision = a.expected_decision(rounds, data["finalist"]["tactic"], 1)
    a.require(
        decision["decision_role"] == "winner",
        "even sample median-log differs from arithmetic median ratio",
    )
    passed.append("median-log rule rejects arithmetic-median substitution")
    for i, pair in enumerate(rounds):
        for row in pair["measurements"]:
            row["elapsed_ms"] = (0.5 if i < 6 else 2) if row["role"] == "winner" else 1
    a.require(
        a.expected_decision(rounds, [], 1)["decision_role"] == "default",
        "exact log tie selects default",
    )
    passed.append("exact no-margin tie selects default")
    reject(
        "incomplete decision input",
        rounds,
        lambda d: d.pop(),
        lambda d: a.expected_decision(d, [], 1),
    )
    reject(
        "duplicated candidate role",
        rounds,
        lambda d: d[0]["measurements"][0].update(role=d[0]["measurements"][1]["role"]),
        lambda d: a.expected_decision(d, [], 1),
    )

    for log_ratio in (-0.2, 0.2):
        change = a.choice_effect("default", "winner", log_ratio)
        reverse = a.choice_effect("winner", "default", log_ratio)
        a.close(
            change["hypothetical_over_original"]
            * reverse["hypothetical_over_original"],
            1,
            "reciprocal policy transitions",
        )
        for role in ("winner", "default"):
            same = a.choice_effect(role, role, log_ratio)
            a.require(
                same["hypothetical_over_original"] == 1
                and same["outcome"] == "unchanged_choice",
                "unchanged choice exact1 regardless finalist loss",
            )
        a.require(
            change["original_log_regret_two_candidate_only"] >= 0
            and change["confirmed_log_regret_two_candidate_only"] >= 0,
            "pair regret is nonnegative",
        )
    a.require(
        a.choice_effect("default", "winner", 0.2)["outcome"] == "measured_worse"
        and a.choice_effect("winner", "default", 0.2)["outcome"] == "measured_better",
        "counterfactual direction",
    )
    passed.extend(
        [
            "reciprocal choice transitions",
            "unchanged decisions exact1",
            "measured help/hurt direction",
            "nonnegative two-candidate regret",
        ]
    )

    processes = [
        {
            "block": b,
            "capture_order": "winner-first" if b % 2 == 0 else "default-first",
            "original_profiler_ratio": 2,
            "graph_log_ratio": math.log(2),
            "confirmation_log_ratio": math.log(0.5),
            **a.choice_effect("default", "winner", math.log(2)),
        }
        for b in range(8)
    ]
    summary = a.summarize(processes, replicates=100)
    a.close(
        summary["metrics"]["hypothetical_over_original"]["geometric_mean_ratio"],
        2,
        "whole-process geometric mean",
    )
    for endpoint in summary["metrics"]["hypothetical_over_original"][
        "exploratory_process_bootstrap_95pct"
    ]:
        a.close(endpoint, 2, "degenerate process bootstrap")
    a.require(
        summary["outcomes"] == {"measured_worse": 8}, "adverse processes retained"
    )
    passed.append("all eight adverse processes retained in paired process bootstrap")
    reject("dropped default-winning process", processes, lambda d: d.pop(), a.summarize)
    reject("duplicated process", processes, lambda d: d[7].update(block=0), a.summarize)
    a.validate_schedule(schedule, prior)
    reject(
        "unbalanced confirmation order",
        schedule,
        lambda d: d["confirmation_pair_schedules"].update(
            {"0": [["winner", "default"]] * 12}
        ),
        lambda d: a.validate_schedule(d, prior),
    )
    return {
        "status": "passed",
        "cpu_only": True,
        "historical_timing_pooled": False,
        "artificial_confirmation_for_schema_only": True,
        "checks": passed,
        "count": len(passed),
        "analyzer_sha256": a.sha256(args.analyzer),
        "control_sha256": a.sha256(__file__),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--analyzer", type=Path, default=here / "analyzer.py")
    parser.add_argument(
        "--prior-analyzer",
        type=Path,
        default=here.parent / "20261026T220000Z-h100-fixed-pair" / "analyzer.py",
    )
    parser.add_argument("--historical-result", type=Path, required=True)
    parser.add_argument(
        "--historical-contract", type=Path, default=here / "historical-contract.json"
    )
    parser.add_argument("--schedule", type=Path, default=here / "schedule.json")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "count": result["count"]}))
