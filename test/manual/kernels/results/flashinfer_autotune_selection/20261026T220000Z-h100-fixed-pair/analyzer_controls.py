#!/usr/bin/env python3
"""CPU mutation controls for the manual fixed-pair admission analyzer.

The historical artifact supplies real schema, not additional timing observations.
Every changed copy is deliberately invalid and must be rejected before inference.
"""

import argparse
import copy
import importlib.util
import json
from pathlib import Path


def module(path):
    spec = importlib.util.spec_from_file_location("fixed_pair_analysis", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def late_fixture(original, schedule, contract):
    """Declared artificial timing/schema fixture, only for CPU fault injection."""
    data = copy.deepcopy(original)
    metadata = data["startup"]["profiles"][80]["metadata"]
    end_graph = data["rounds"][-1]["measurements"][-1]["ended_ns"]
    clock = end_graph

    def tick():
        nonlocal clock
        clock += 10
        return clock

    data["profiler_objective_order"] = schedule["entries"][0][
        "profiler_objective_order"
    ]
    data["retained_profile"] = {
        "retained_after_index": 80,
        "metadata": metadata,
        "after_pairs_metadata": metadata,
        "tensor_object_ids": list(range(100, 106)),
        "retained_after_ns": data["startup"]["ended_ns"] - 1,
        "reference_preparation_started_ns": tick(),
        "reference_preparation_ended_ns": tick(),
        "read_operand_hashes": ["a" * 64] * 4,
    }
    (
        data["profile_correctness"],
        data["diagnostic_profile_attempts"],
        data["paired_profiles"],
    ) = [], [], []
    data["cache_partition_restorations"], data["synthetic_acceptance"] = [], []
    for objective in data["profiler_objective_order"]:
        settings = copy.deepcopy(contract["original_settings"])
        if objective == "graph":
            settings["config"]["use_cuda_graph"] = True
            settings["measurement_policy"] = {
                "execution_mode": "cuda_graph",
                "cold_l2": None,
                "_timer": "auto",
                "graph_timing": None,
            }
            settings["effective_policy_timer"] = "events"
        group = {
            "objective": objective,
            "settings": settings,
            "started_ns": tick(),
            "preparation": {
                "ended_ns": tick(),
                "public_schedule_uses_exact_retained_objects": True,
            },
            "rounds": [],
        }
        for index, order in enumerate(
            schedule["profiler_pair_schedules"]["0"][objective]
        ):
            pair = {"index": index, "order": order, "measurements": []}
            for role in order:
                label = f"late_{objective}_{index:02d}_{role}"
                tactic = contract["fixed_descriptor"] if role == "winner" else -1
                elapsed = 0.02 if role == "winner" else 0.01
                attempt_index = len(data["diagnostic_profile_attempts"])
                data["diagnostic_profile_attempts"].append(
                    {
                        "index": attempt_index,
                        "stage": label,
                        "objective": objective,
                        "tactic": tactic,
                        "settings": copy.deepcopy(settings),
                        "metadata": copy.deepcopy(metadata),
                        "tensor_object_ids": list(range(100, 106)),
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
                        "output_data_ptr": metadata[4]["data_ptr"],
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
                        "attempt_index": attempt_index,
                        "correctness_check": label,
                    }
                )
            group["rounds"].append(pair)
        group.update(ended_ns=tick(), explicit_over_default_median=2)
        data["paired_profiles"].append(group)
        digest = data["startup"]["cache_state_sha256"]
        data["cache_partition_restorations"].append(
            {
                "objective": objective,
                "before_sha256": digest,
                "after_sha256": digest,
                "restored": True,
                "changed_fields": [],
            }
        )
    last_profile = clock
    for role in ("winner", "default"):
        row = {
            "role": role,
            "tactic": contract["fixed_descriptor"] if role == "winner" else -1,
            "inputs": metadata,
            "started_ns": tick(),
            "used_descriptor": 1 if role == "winner" else None,
            "descriptor_acceptance": "accepted"
            if role == "winner"
            else "not_applicable_default_route",
            "checks": [],
        }
        for index, suffix in enumerate(("", "_changed_input", "_restored_input")):
            label = "profile_accept_" + role + suffix
            data["profile_correctness"].append(
                {
                    "check": label,
                    "passed": True,
                    "output_data_ptr": metadata[4]["data_ptr"],
                    "max_abs_error": 0,
                    "exact_zero_required": index == 1,
                    "checked_ns": tick(),
                }
            )
            row["checks"].append(label)
        row["ended_ns"] = tick()
        data["synthetic_acceptance"].append(row)
    return data, end_graph, last_profile


def run(args):
    a = module(args.analyzer)
    leaf = a.leaf_helpers(args.leaf_helpers)
    original = a.load(args.historical_result)
    contract = a.load(args.historical_contract)
    schedule = a.load(args.schedule)
    passed = []

    def reject(name, value, mutation, validator):
        changed = copy.deepcopy(value)
        mutation(changed)
        try:
            validator(changed)
        except ValueError:
            passed.append(name)
        else:
            raise AssertionError(name + " was accepted")

    baseline = lambda data: a.validate_baseline(data, contract, leaf)
    a.close(
        baseline(original),
        contract["original_profile_ratio"],
        "preserved original ratio",
    )
    passed.append("preserved real81-call baseline accepted")
    reject(
        "nonfinite profile",
        original,
        lambda d: d["startup"]["profiles"][74].update(elapsed_ms=float("nan")),
        baseline,
    )
    reject(
        "dropped original history",
        original,
        lambda d: d["startup"]["profiles"].pop(0),
        baseline,
    )
    reject(
        "candidate substituted",
        original,
        lambda d: d["startup"]["profiles"][74].update(tactic=-1),
        baseline,
    )
    reject(
        "candidate-order changed",
        original,
        lambda d: d["startup"]["candidate_order"]["144"].reverse(),
        baseline,
    )
    reject(
        "profile bucket changed",
        original,
        lambda d: d["startup"]["profiles"][74].update(key="other"),
        baseline,
    )
    reject(
        "profiler objective silently changed",
        original,
        lambda d: d["startup"]["profiles"][74]["settings"]["config"].update(
            use_cuda_graph=True
        ),
        baseline,
    )
    reject(
        "synthetic operand replaced",
        original,
        lambda d: d["startup"]["profiles"][80]["metadata"][0].update(data_ptr=123456),
        baseline,
    )
    reject(
        "timing startup hashed inline",
        original,
        lambda d: d["startup"]["profiles"][74].update(fingerprinted=True),
        baseline,
    )
    reject(
        "natural winner replaced",
        original,
        lambda d: d["natural"].update(tactic=-1),
        baseline,
    )

    graphs = lambda data: a.validate_graphs(
        data, contract, leaf, leaf.correctness_index(data)
    )
    graphs(original)
    passed.append("original graph ownership and changed-input controls accepted")
    reject(
        "fixed graph uses fallback tactic",
        original,
        lambda d: d["graphs"][0].update(tactic=-1),
        graphs,
    )
    reject(
        "default falsely cache-hit",
        original,
        lambda d: d["graphs"][1]["selection"].update(cache_hit=True),
        graphs,
    )
    reject(
        "stale non-owned graph output",
        original,
        lambda d: d["graphs"][0]["output"].update(data_ptr=123456),
        graphs,
    )
    reject(
        "output check failed",
        original,
        lambda d: d["correctness"][0].update(passed=False),
        graphs,
    )
    reject(
        "changed input not exact zero",
        original,
        lambda d: next(r for r in d["correctness"] if r["exact_zero_required"]).update(
            max_abs_error=1
        ),
        graphs,
    )

    historical_schedule = copy.deepcopy(schedule)
    historical_schedule["round_schedules"]["0"]["orders"] = original["round_schedule"]
    timing = lambda data: a.validate_graph_timing(
        data, historical_schedule, graphs(data), leaf.correctness_index(data), leaf
    )
    a.close(
        timing(original)[0]["ratio"],
        contract["original_graph_ratio"],
        "preserved graph ratio",
    )
    passed.append("all48 historical retained graph samples accepted")
    reject("missing graph round", original, lambda d: d["rounds"].pop(), timing)
    reject(
        "substituted measured graph",
        original,
        lambda d: d["rounds"][0]["measurements"][0].update(graph_object_id=0),
        timing,
    )
    reject(
        "profiling during graph timing",
        original,
        lambda d: d["rounds"][0]["measurements"][0].update(profile_attempts_after=82),
        timing,
    )
    reject(
        "incorrect graph normalization",
        original,
        lambda d: d["rounds"][0]["measurements"][0].update(gpu_event_per_replay_ms=1),
        timing,
    )
    reject(
        "false queue evidence",
        original,
        lambda d: d["rounds"][0]["measurements"][0].update(queue_evidence="invented"),
        timing,
    )
    reject(
        "overlapping timing windows",
        original,
        lambda d: d["rounds"][1]["measurements"][0].update(wall_started_ns=1),
        timing,
    )

    a.validate_schedule(schedule)
    passed.append("frozen balanced schedule accepted")
    reject(
        "unbalanced pair order",
        schedule,
        lambda d: d["profiler_pair_schedules"]["0"].update(
            auto=[["winner", "default"]] * 12
        ),
        a.validate_schedule,
    )
    reject(
        "missing process", schedule, lambda d: d["entries"].pop(), a.validate_schedule
    )
    reject(
        "duplicated objective block",
        schedule,
        lambda d: d["entries"][0].update(profiler_objective_order=["auto", "auto"]),
        a.validate_schedule,
    )

    fixture, end_graph, last_profile = late_fixture(original, schedule, contract)
    profiler = lambda data: a.validate_profiler(data, schedule, contract, end_graph)
    result, actual_end, checks = profiler(fixture)
    a.require(
        actual_end == last_profile and all(v["ratio"] == 2 for v in result.values()),
        "CPU profiler fixture",
    )
    passed.append(
        "artificial48-call profiler schema accepted without performance inference"
    )
    reject(
        "retention before final history",
        fixture,
        lambda d: d["retained_profile"].update(retained_after_index=72),
        profiler,
    )
    reject(
        "reference prepared before graph timing",
        fixture,
        lambda d: d["retained_profile"].update(reference_preparation_started_ns=1),
        profiler,
    )
    reject(
        "retained tensor object substituted",
        fixture,
        lambda d: d["diagnostic_profile_attempts"][0]["tensor_object_ids"].__setitem__(
            0, 999
        ),
        profiler,
    )
    reject(
        "fixed late candidate substituted",
        fixture,
        lambda d: d["diagnostic_profile_attempts"][0].update(tactic=[0] * 8),
        profiler,
    )
    reject(
        "late profiler fallback failure",
        fixture,
        lambda d: d["diagnostic_profile_attempts"][0].update(
            error="rejected descriptor"
        ),
        profiler,
    )
    reject(
        "late graph silently measured eager",
        fixture,
        lambda d: next(
            r for r in d["diagnostic_profile_attempts"] if r["objective"] == "graph"
        )["settings"]["config"].update(use_cuda_graph=False),
        profiler,
    )
    reject(
        "missing late call",
        fixture,
        lambda d: d["diagnostic_profile_attempts"].pop(),
        profiler,
    )
    reject(
        "late result bound to wrong invocation",
        fixture,
        lambda d: d["paired_profiles"][0]["rounds"][0]["measurements"][0].update(
            attempt_index=4
        ),
        profiler,
    )
    reject(
        "late nonfinite time",
        fixture,
        lambda d: d["diagnostic_profile_attempts"][0].update(elapsed_ms=float("inf")),
        profiler,
    )
    reject(
        "late no-write numerical failure",
        fixture,
        lambda d: d["profile_correctness"][0].update(passed=False),
        profiler,
    )
    reject(
        "late stale output check owner",
        fixture,
        lambda d: d["profile_correctness"][0].update(output_data_ptr=1),
        profiler,
    )
    reject(
        "late cache not restored",
        fixture,
        lambda d: d["cache_partition_restorations"][0].update(after_sha256="different"),
        profiler,
    )
    reject(
        "late policy changes winner entries",
        fixture,
        lambda d: d["cache_partition_restorations"][0].update(
            changed_fields=["profiling_cache"]
        ),
        profiler,
    )
    synthetic = lambda data: a.validate_synthetic_acceptance(
        data,
        contract,
        {r["check"]: r for r in data["profile_correctness"]},
        last_profile,
    )
    synthetic(fixture)
    passed.append("synthetic native acceptance/zero/restore schema accepted")
    reject(
        "native descriptor silently falls back",
        fixture,
        lambda d: d["synthetic_acceptance"][0].update(used_descriptor=0),
        synthetic,
    )
    reject(
        "native acceptance uses other inputs",
        fixture,
        lambda d: d["synthetic_acceptance"][0].update(inputs=[]),
        synthetic,
    )
    reject(
        "native acceptance before timing ends",
        fixture,
        lambda d: d["synthetic_acceptance"][0].update(started_ns=1),
        synthetic,
    )

    ids = [f"{i:064x}" for i in range(9)]
    a.validate_container_ids(ids)
    passed.append("nine separate container identities accepted despite PID namespaces")
    reject(
        "container reused for multiple workers",
        ids,
        lambda d: d.__setitem__(8, d[0]),
        a.validate_container_ids,
    )
    reject(
        "truncated container identity",
        ids,
        lambda d: d.__setitem__(0, d[0][:12]),
        a.validate_container_ids,
    )

    # Fixed synthetic process-level estimates test statistics without any GPU data.
    processes = [
        {
            "block": b,
            "original_history_profiler_ratio": 1.0,
            "layer_graph": {"ratio": 2.0},
            "profiler": {"auto": {"ratio": 4.0}, "graph": {"ratio": 2.0}},
        }
        for b in range(8)
    ]
    summary = a.summarize(processes, replicates=100)
    for key, expected in {
        "original_history_profiler": 1,
        "layer_graph": 2,
        "late_auto_profiler": 4,
        "late_graph_profiler": 2,
        "late_graph_over_auto_profiler": 0.5,
    }.items():
        a.close(
            summary["metrics"][key]["geometric_mean_ratio"],
            expected,
            "process estimator",
        )
        for endpoint in summary["metrics"][key]["exploratory_process_bootstrap_95pct"]:
            a.close(endpoint, expected, "degenerate process bootstrap")
    passed.append("whole-process paired estimator and degenerate intervals")
    reject(
        "incomplete cohort cannot summarize", processes, lambda d: d.pop(), a.summarize
    )
    reject(
        "duplicate process cannot summarize",
        processes,
        lambda d: d[7].update(block=0),
        a.summarize,
    )
    return {
        "status": "passed",
        "cpu_only": True,
        "historical_timing_pooled": False,
        "checks": passed,
        "count": len(passed),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analyzer", type=Path, default=Path(__file__).with_name("analyzer.py")
    )
    parser.add_argument("--leaf-helpers", type=Path, required=True)
    parser.add_argument("--historical-result", type=Path, required=True)
    parser.add_argument("--historical-contract", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run(arguments)
    text = json.dumps(result, indent=2) + "\n"
    if arguments.output:
        arguments.output.write_text(text)
    print(text, end="")
