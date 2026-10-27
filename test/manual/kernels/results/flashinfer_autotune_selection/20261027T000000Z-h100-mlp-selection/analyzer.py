"""Independent block-level analysis of the prespecified three-arm MLP study."""

import argparse
import collections
import hashlib
import itertools
import json
import math
import random
import statistics
from pathlib import Path

ARMS = ("default", "v1", "v2")
MODES = ("eager", "cuda_graph")
ROLES = ("natural", "default")


def require(condition, label):
    if not condition:
        raise ValueError(label)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def positive(value):
    require(
        type(value) in (int, float) and math.isfinite(value) and value > 0,
        "duration must be finite and strictly positive",
    )
    return value


def quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def estimates(logs, resamples):
    require(
        logs and all(math.isfinite(x) for x in logs), "finite complete block estimates"
    )
    require(
        resamples
        and all(
            len(indices) == len(logs)
            and all(type(i) is int and 0 <= i < len(logs) for i in indices)
            for indices in resamples
        ),
        "whole-block resampling indices",
    )
    means = [
        math.exp(statistics.mean(logs[i] for i in indices)) for indices in resamples
    ]
    return {
        "n_blocks": len(logs),
        "geometric_mean_ratio": math.exp(statistics.mean(logs)),
        "per_block_ratios": [math.exp(x) for x in logs],
        "range": [math.exp(min(logs)), math.exp(max(logs))],
        "exploratory_block_bootstrap_95": [
            quantile(means, 0.025),
            quantile(means, 0.975),
        ],
    }


def validate_schedule(schedule):
    entries = schedule["entries"]
    require(
        len(entries) == 36 and len({x["id"] for x in entries}) == 36,
        "exactly 36 unique fresh-process assignments",
    )
    permutations = list(itertools.permutations(ARMS))
    for trial in range(6):
        rows = [x for x in entries if x["trial"] == trial]
        require(
            [(x["arm"], x["phase"]) for x in rows]
            == [(a, p) for a in permutations[trial] for p in ("cold", "warm")],
            "all arm permutations and cold-then-fresh-warm order",
        )
        for mode in MODES:
            orders = schedule["round_schedules"][str(trial)][mode]
            require(
                len(orders) == 12
                and all(sorted(o) == sorted(ROLES) for o in orders)
                and sum(o[0] == "natural" for o in orders) == 6,
                "twelve balanced paired windows",
            )
    for arm in ARMS:
        rows = [x for x in entries if x["arm"] == arm and x["phase"] == "warm"]
        require(
            collections.Counter(x["capture_order"] for x in rows)
            == {"natural-first": 3, "default-first": 3},
            "balanced graph capture order",
        )
        require(
            collections.Counter(x["mode_order"] for x in rows)
            == {"eager-first": 3, "graph-first": 3},
            "balanced execution mode order",
        )


def paired_logs(pairs, orders, mode):
    require(len(pairs) == len(orders) == 12, "all twelve pairs retained")
    require(mode in MODES, "known measurement mode")
    metric = "wall_per_call_ms" if mode == "eager" else "gpu_event_per_call_ms"
    logs, absolute = [], {r: [] for r in ROLES}
    previous_end = 0
    for i, (pair, order) in enumerate(zip(pairs, orders)):
        require(pair["index"] == i and pair["order"] == order, "frozen pair order")
        rows = pair["measurements"]
        require([r["role"] for r in rows] == order, "actual execution order")
        require(
            len(rows) == 2 and rows[0]["iterations"] == rows[1]["iterations"],
            "same iteration count within each paired window",
        )
        values = {}
        for row in rows:
            n = row["iterations"]
            require(type(n) is int and n > 0, "positive integer batch length")
            for clock in ("wall", "gpu_event"):
                batch = positive(row[clock + "_batch_ms"])
                per_call = positive(row[clock + "_per_call_ms"])
                require(
                    math.isclose(batch / n, per_call, rel_tol=1e-12, abs_tol=1e-15),
                    "per-call timing recomputed from raw batch",
                )
            require(
                row["ended_ns"] > row["started_ns"] >= previous_end,
                "ordered nonoverlapping actual sample timestamps",
            )
            previous_end = row["ended_ns"]
            values[row["role"]] = row[metric]
            absolute[row["role"]].append(row[metric])
        logs.append(math.log(values["natural"] / values["default"]))
    return {
        "primary_metric": metric,
        "paired_log_ratios": logs,
        "median_log_ratio": statistics.median(logs),
        "natural_over_default": math.exp(statistics.median(logs)),
        "median_absolute_ms": {r: statistics.median(v) for r, v in absolute.items()},
        "first_position_strata": {
            role: math.exp(
                statistics.median(x for x, o in zip(logs, orders) if o[0] == role)
            )
            for role in ROLES
        },
    }


def summarize(processes):
    require(
        len(processes) == 36, "only a complete prespecified cohort may be estimated"
    )
    indexed = {(x["trial"], x["arm"], x["phase"]): x for x in processes}
    require(len(indexed) == 36, "no duplicate or missing worker")
    rng = random.Random(202610270000)
    resamples = [[rng.randrange(6) for _ in range(6)] for _ in range(100000)]
    phases = {}
    for phase in ("warm", "cold"):
        phase_results = {}
        for mode in MODES:
            by_arm = {
                arm: [
                    indexed[t, arm, phase]["modes"][mode]["median_log_ratio"]
                    for t in range(6)
                ]
                for arm in ARMS
            }
            normalized = [b - a for a, b in zip(by_arm["v1"], by_arm["v2"])]
            absolute = [
                math.log(
                    indexed[t, "v2", phase]["modes"][mode]["median_absolute_ms"][
                        "natural"
                    ]
                    / indexed[t, "v1", phase]["modes"][mode]["median_absolute_ms"][
                        "natural"
                    ]
                )
                for t in range(6)
            ]
            phase_results[mode] = {
                "natural_over_own_default": {
                    a: estimates(v, resamples) for a, v in by_arm.items()
                },
                "v2_over_v1_normalized_to_own_default": estimates(
                    normalized, resamples
                ),
                "v2_over_v1_absolute_natural_latency_secondary": estimates(
                    absolute, resamples
                ),
            }
        phases[phase] = phase_results
    return {
        "primary_phase": "warm",
        "estimates": phases,
        "limits": [
            "Six blocks on one fixture, GPU and time window; within-process pairs are dependent.",
            "Block bootstrap intervals are exploratory, not equivalence, safety or population guarantees.",
            "Normalizing against own retained default does not eliminate capture/allocation/history effects.",
            "Default-vs-default independent graphs remain measured; their ratios are never forced to one.",
            "Queued graph timing reports start-event evidence; completed queries are inconclusive and never excluded.",
        ],
    }


def trace_signatures(path, helper_path):
    import copy
    import importlib.util

    require(
        sha(helper_path)
        == "cf72c74cf2e9c343e46af1b1b74a232a8b201de06dc2cd01bef6552f0981ce4f",
        "frozen qualified trace parser",
    )
    spec = importlib.util.spec_from_file_location("qualified_trace_parser", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    trace = copy.deepcopy(load(path))
    for event in trace["traceEvents"]:
        if (
            event.get("cat") == "user_annotation"
            and event.get("name") == "retained_natural"
        ):
            event["name"] = "retained_winner"
    result = helper.parse_retained_trace(trace)
    result["natural"] = result.pop("winner")
    result["natural_default_signatures_equal"] = result.pop(
        "winner_default_signatures_equal"
    )
    return result


def validate_worker(args, entry, plan, schedule):
    prefix = "admission-" if args.admission else ""
    name = prefix + entry["id"]
    path = args.raw_root / (name + ".json")
    data, bootstrap = load(path), load(path.with_suffix(".bootstrap.json"))
    require(
        data["status"] == "success"
        and data["schema_version"] == 2
        and data["study"] == "mlp_default_v1_v2",
        "successful complete worker",
    )
    require(
        data["assignment"] == entry
        and all(data[k] == entry[k] for k in ("trial", "arm", "phase")),
        "actual frozen process assignment",
    )
    require(
        data["mode"] == ("admission" if args.admission else "timing"),
        "separate cohort mode",
    )
    require(
        data["worker_sha256"] == sha(args.worker)
        and data["plan_sha256"] == sha(args.plan)
        and data["schedule_sha256"] == sha(args.schedule),
        "exact frozen worker and protocol",
    )
    require(
        bootstrap["status"] == "passed"
        and bootstrap["result_sha256"] == sha(path)
        and bootstrap["worker_sha256"] == sha(args.worker),
        "qualified bootstrap/result binding",
    )
    require(
        data["source"]["critical_sha256"] == plan["source_contract"]["runtime_sha256"]
        and data["source"]["flashinfer_commit"]
        == plan["source_contract"]["flashinfer_commit"]
        and data["source"]["native_identity_passthrough"] is True,
        "qualified source identity",
    )
    fixture = data["fixture"]
    require(
        fixture["manifest_sha256"] == plan["fixture_manifest_sha256"]
        and fixture["tensor_file_sha256"] == plan["fixture_sha256"]
        and fixture["shape"] == [144, 1024, 3072]
        and data["shape"] == fixture["shape"]
        and data["seed"] == 19,
        "actual immutable MLP fixture",
    )
    require(
        fixture["capture"]["activation_byte_match"] is True
        and fixture["capture"]["checkpoint_tensor_byte_match"] is True
        and fixture["capture"]["full_bf16_forward_completed"] is True,
        "real captured activation and checkpoint provenance",
    )
    startup, natural, final, cache = (
        data[k] for k in ("startup", "natural", "final", "cache")
    )
    profiles = startup["profiles"]
    require(
        len(startup["selection"]) == 1
        and startup["selection"][0]["in_v2_context"] == (entry["arm"] == "v2")
        and startup["selection"][0]["is_tuning_mode"] == (entry["arm"] != "default")
        and startup["selection"][0]["measurement_policy"] is None,
        "actual startup context in every cold and warm arm",
    )
    require(
        startup["profile_attempts"] == len(profiles) == final["profile_attempts"]
        and not data["forbidden_profile_attempts"]
        and not data["profile_protocol_errors"],
        "no swallowed or additional profiling",
    )
    require(
        all(
            final[k] is True
            for k in (
                "zero_additional_profiling",
                "cache_unchanged",
                "operands_unchanged",
                "native_provider_restored",
                "on_caller_stream",
            )
        )
        and not any(
            final[k]
            for k in ("active_v2", "is_tuning_mode", "tuning_contexts", "managed_store")
        )
        and final["measurement_policy"] is None,
        "bare serving and restored state",
    )
    require(
        cache["after_startup_sha256"] == cache["final_sha256"]
        and cache["canonical_state_after_startup"] == cache["canonical_state_final"],
        "persistent bytes and natural cache state preserved",
    )
    expected_tier = (
        "miss"
        if entry["arm"] == "default"
        else "memory"
        if entry["phase"] == "cold"
        else "file"
    )
    require(
        natural["lookup_tier"] == expected_tier
        and natural["cache_hit"] == (expected_tier != "miss"),
        "genuine natural cache hit or default miss",
    )
    if entry["phase"] == "warm":
        require(
            cache["before_sha256"] == cache["final_sha256"]
            and not profiles
            and cache["memory_entries"] == 0,
            "fresh warm file reuse without profiling",
        )
    else:
        require(cache["before_sha256"] is None, "genuine empty cold cache")
    if entry["arm"] == "default":
        require(
            not profiles
            and natural["tactic"] == -1
            and cache["memory_entries"] == cache["file_entries"] == 0,
            "untuned control remains a genuine fallback",
        )
    elif entry["phase"] == "cold":
        require(profiles, "real cold search occurred")
        grouped = collections.defaultdict(list)
        for index, profile in enumerate(profiles):
            require(
                profile["index"] == index
                and profile["stage"] == "startup"
                and profile["runner"] == "CublasFp8GemmRunner"
                and profile["is_tuning_mode"] is True
                and profile["in_v2_context"] == (entry["arm"] == "v2"),
                "actual legacy or v2 search context",
            )
            settings = profile["autotuner_settings"]
            require(
                settings["effective_measure_policy"] is None
                and settings["effective_policy_timer"] is None
                and settings["warmup"] == 3
                and settings["repeat"] == 10
                and settings["stream_delay_micro_secs"] == 5000
                and settings["cuda_graph_delay_micro_secs"] == 100
                and settings["use_global_timer"] is False,
                "unchanged unspecified measurement settings",
            )
            require(
                settings["config"]
                == {
                    "use_cuda_graph": False,
                    "use_cold_l2_cache": False,
                    "cuda_graph_profile_replays": 1,
                    "profiling_repeat": None,
                    "effective_profiling_repeat": 10,
                },
                "unchanged operation timing configuration",
            )
            require(
                profile["fingerprinted"] == args.admission,
                "declared admission-only input hashing",
            )
            if profile["profiler_ms"] is not None:
                positive(profile["profiler_ms"])
            grouped[profile["m"]].append(profile)
        require(
            set(grouped) == {1, 2, 4, 8, 16, 32, 64, 128, 144},
            "all original dynamic profiles",
        )
        for rows in grouped.values():
            require(
                sum(r["tactic"] == -1 for r in rows) == int(entry["arm"] == "v2"),
                "default included exactly for v2",
            )
        runtime = [
            r
            for r in profiles
            if r["key"] == natural["key"] and r["profiler_ms"] is not None
        ]
        require(
            runtime
            and min(runtime, key=lambda r: r["profiler_ms"])["tactic"]
            == natural["tactic"],
            "actual runtime winner is first finite search minimum",
        )
    bindings = data["binding_acceptance_checks"]
    require(
        bindings
        and all(
            r["accepted"] is True
            and r["used_descriptor"] == ([] if r["tactic"] == -1 else [1])
            for r in bindings
        ),
        "actual native descriptor acceptance without fallback",
    )
    checks = {r["check"]: r for r in data["correctness"]}
    require(
        len(checks) == len(data["correctness"]) and checks, "unique numerical evidence"
    )
    require(
        all(
            r["quantized_reference"]["passed"] is True
            and math.isfinite(r["quantized_reference"]["max_normalized_error"])
            and r["quantized_reference"]["max_normalized_error"] <= 1
            for r in checks.values()
        ),
        "every output passes mixed numerical tolerance",
    )
    graphs = {g["id"]: g for g in data["graphs"]}
    capture_order = (
        list(ROLES)
        if entry["capture_order"] == "natural-first"
        else list(reversed(ROLES))
    )
    require(
        [g["id"] for g in data["graphs"]] == capture_order, "actual graph capture order"
    )
    first_graph, second_graph = (graphs[role] for role in capture_order)
    require(
        first_graph["capture_started_ns"]
        < first_graph["capture_ended_ns"]
        <= second_graph["capture_started_ns"]
        < second_graph["capture_ended_ns"],
        "ordered independent graph captures",
    )
    require(
        set(graphs) == set(ROLES)
        and len(data["graphs"]) == 2
        and len({g["graph_object_id"] for g in graphs.values()}) == 2
        and len({g["output"]["data_ptr"] for g in graphs.values()}) == 2,
        "independent retained graph objects and outputs",
    )
    for role, graph in graphs.items():
        require(
            graph["tactic"] == (natural["tactic"] if role == "natural" else -1)
            and graph["lookup_tier"]
            == (expected_tier if role == "natural" else "miss"),
            "original natural and default graph choices",
        )
        require(
            graph["output"]["data_ptr"] == graph["inputs"][4]["data_ptr"],
            "graph output allocation binding",
        )
        for suffix in ("changed_input", "restored_input"):
            checked = checks[role + "_" + suffix]
            require(
                checked["output_data_ptr"] == graph["output"]["data_ptr"]
                and checked["exact_zero_required"] == (suffix == "changed_input"),
                "changed-input check binds original graph output",
            )
    require(
        all(
            r["restored"]
            and r["cache_sha256_before"] == r["cache_sha256_after"]
            and r["winner_entries_after_removal"]
            == r["file_entries_after_removal"]
            == 0
            for r in data["miss_scopes"]
        ),
        "default scopes restore genuine natural cache",
    )
    modes = {}
    mode_order = (
        list(MODES) if entry["mode_order"] == "eager-first" else list(reversed(MODES))
    )
    first_mode, second_mode = (data["measurements"][mode] for mode in mode_order)
    require(
        second_graph["capture_ended_ns"]
        <= first_mode["started_ns"]
        < first_mode["ended_ns"]
        <= second_mode["started_ns"]
        < second_mode["ended_ns"],
        "actual execution-mode ordering after graph capture",
    )
    for mode in MODES:
        group = data["measurements"][mode]
        orders = schedule["round_schedules"][str(entry["trial"])][mode]
        modes[mode] = paired_logs(group["pairs"], orders, mode)
        common = group["common_iterations"]
        require(
            len(group["initial_warmups"]) == 2
            and all(r["actual_ms"] >= 500 for r in group["initial_warmups"]),
            "full warmup for both roles",
        )
        if mode == "eager":
            require(
                common
                == max(
                    r["effective_iterations"]
                    for r in group["calibration_by_role"].values()
                ),
                "common batch length from both independent calibrations",
            )
        else:
            require(
                common == 16 and not group["calibration_by_role"],
                "prespecified queued graph batch length",
            )
        for pair in group["pairs"]:
            for row in pair["measurements"]:
                require(
                    row["iterations"] == common and row["rewarm"]["actual_ms"] >= 20,
                    "common batch length and rewarm",
                )
                require(
                    row["profile_attempts_before"]
                    == row["profile_attempts_after"]
                    == len(profiles),
                    "no hidden profiling during timing",
                )
                checked = checks[row["correctness_check"]]
                require(
                    checked["checked_ns"] >= row["ended_ns"],
                    "numerical check follows timed batch",
                )
                if mode == "cuda_graph":
                    require(
                        row["prequeue_cycles"] == 10000000
                        and type(row["start_event_complete_after_enqueue"]) is bool
                        and row["queue_query_is_exclusion_criterion"] is False,
                        "queued replay with all query outcomes retained",
                    )
                    require(
                        row["output_data_ptr"]
                        == graphs[row["role"]]["output"]["data_ptr"],
                        "timed replay checks its retained allocation",
                    )
                for key in ("lookup_before", "lookup_after"):
                    expected = expected_tier if row["role"] == "natural" else "miss"
                    lookup = row[key]
                    require(
                        lookup["lookup_tier"] == expected
                        and lookup["cache_hit"] == (expected != "miss")
                        and lookup["tactic"]
                        == (natural["tactic"] if row["role"] == "natural" else -1),
                        "actual choice and cache source around every timed window",
                    )
    trace_path = args.raw_root / (name + ".trace.json")
    require(
        data["trace"]["sha256"] == sha(trace_path)
        and data["trace"]["started_ns"]
        > max(g["ended_ns"] for g in data["measurements"].values()),
        "trace is the exact posttiming observation",
    )
    signatures = trace_signatures(trace_path, args.trace_helper)
    require(
        {
            r["id"]: (r["graph_object_id"], r["output_data_ptr"])
            for r in data["trace"]["graphs"]
        }
        == {
            r: (g["graph_object_id"], g["output"]["data_ptr"])
            for r, g in graphs.items()
        },
        "same original measured graph objects in trace",
    )
    return {
        "name": name,
        "result_sha256": sha(path),
        "bootstrap_sha256": sha(path.with_suffix(".bootstrap.json")),
        **{
            k: entry[k]
            for k in ("trial", "arm", "phase", "capture_order", "mode_order")
        },
        "natural_tactic": natural["tactic"],
        "profile_groups": {
            str(m): {
                "tactics": [r["tactic"] for r in profiles if r["m"] == m],
                "keys": sorted({r["key"] for r in profiles if r["m"] == m}),
                "input_fingerprints": sorted(
                    {
                        tuple(r["fingerprints"])
                        for r in profiles
                        if r["m"] == m and "fingerprints" in r
                    }
                ),
            }
            for m in sorted({r["m"] for r in profiles})
        },
        "natural_key": natural["key"],
        "cache_final_sha256": cache["final_sha256"],
        "profile_attempts": len(profiles),
        "startup_ms": startup["startup_ms"],
        "correctness_checks": len(checks),
        "modes": modes,
        "trace_signatures": signatures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw-root", "plan", "schedule", "worker", "trace-helper", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--admission", action="store_true")
    args = parser.parse_args()
    require(not args.output.exists(), "refuse to overwrite independent analysis")
    plan, schedule = load(args.plan), load(args.schedule)
    validate_schedule(schedule)
    entries = (
        [e for e in schedule["entries"] if e["trial"] == 0]
        if args.admission
        else schedule["entries"]
    )
    processes = [validate_worker(args, entry, plan, schedule) for entry in entries]
    indexed = {(r["trial"], r["arm"], r["phase"]): r for r in processes}
    for trial in range(1 if args.admission else 6):
        for arm in ARMS:
            cold, warm = (indexed[trial, arm, phase] for phase in ("cold", "warm"))
            require(
                cold["natural_tactic"] == warm["natural_tactic"]
                and cold["natural_key"] == warm["natural_key"]
                and cold["cache_final_sha256"] == warm["cache_final_sha256"]
                and warm["profile_attempts"] == 0,
                "actual cold winner survives fresh-process persistence",
            )
        legacy = indexed[trial, "v1", "cold"]["profile_groups"]
        current = indexed[trial, "v2", "cold"]["profile_groups"]
        require(
            set(legacy) == set(current), "same actual profile inventory across arms"
        )
        for m, old in legacy.items():
            new = current[m]
            require(
                old["tactics"] == [t for t in new["tactics"] if t != -1]
                and old["keys"] == new["keys"],
                "same explicit candidate order and keys; only default is added",
            )
            if args.admission:
                require(
                    len(old["input_fingerprints"]) == 1
                    and old["input_fingerprints"] == new["input_fingerprints"],
                    "admission profiles use identical synthesized operands across arms",
                )
    result = {
        "status": "passed",
        "mode": "admission" if args.admission else "timing",
        "analyzer_sha256": sha(__file__),
        "trace_helper_sha256": sha(args.trace_helper),
        "plan_sha256": sha(args.plan),
        "schedule_sha256": sha(args.schedule),
        "results": processes,
        "correctness_checks": sum(r["correctness_checks"] for r in processes),
    }
    if not args.admission:
        result.update(summarize(processes))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "processes": len(processes),
                "correctness_checks": result["correctness_checks"],
            }
        )
    )


if __name__ == "__main__":
    main()
