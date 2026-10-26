#!/usr/bin/env python3
"""Independently validate the frozen fixed-pair experiment and summarize processes.

This is a research artifact, not a registered test or a runtime dependency.
Earlier results and the disposable admission are never pooled with this cohort.
"""

import argparse
import collections
import copy
import hashlib
import importlib.util
import json
import math
import random
import re
import statistics
from pathlib import Path

LEAF_SHA = "cf72c74cf2e9c343e46af1b1b74a232a8b201de06dc2cd01bef6552f0981ce4f"
ROLES = ("winner", "default")
ENDPOINTS = (1, 2, 4, 8, 16, 32, 64, 128, 144)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positive(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0, label)
    return value


def close(left, right, label):
    require(math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15), label)


def leaf_helpers(path):
    """Reuse only reviewed tensor/key/trace parsing and numerical ownership helpers."""
    require(sha256(path) == LEAF_SHA, "frozen leaf parser source hash")
    spec = importlib.util.spec_from_file_location("fixed_pair_leaf", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_orders(orders, count, label):
    require(len(orders) == count, label + ": complete round count")
    require(all(sorted(order) == sorted(ROLES) for order in orders), label + ": pairs")
    require(
        collections.Counter(order[0] for order in orders)
        == {r: count // 2 for r in ROLES},
        label + ": balanced first positions",
    )


def validate_schedule(schedule):
    entries = schedule["entries"]
    require(
        len(entries) == 8 and [r["block"] for r in entries] == list(range(8)),
        "eight prespecified processes",
    )
    require(
        all(r["cell"] == "auto_synthetic" for r in entries), "one startup treatment"
    )
    require(
        collections.Counter(r["capture_order"] for r in entries)
        == {"winner-first": 4, "default-first": 4},
        "balanced graph capture order",
    )
    require(
        collections.Counter(tuple(r["profiler_objective_order"]) for r in entries)
        == {("auto", "graph"): 4, ("graph", "auto"): 4},
        "balanced objective block order",
    )
    for block in range(8):
        validate_orders(
            schedule["round_schedules"][str(block)]["orders"], 24, "graph schedule"
        )
        for objective in ("auto", "graph"):
            validate_orders(
                schedule["profiler_pair_schedules"][str(block)][objective],
                12,
                "profiler schedule",
            )


def validate_container_ids(identities):
    require(
        all(re.fullmatch(r"[0-9a-f]{64}", value) for value in identities),
        "retained full Docker container identities",
    )
    require(
        len(set(identities)) == len(identities),
        "each worker executed in a distinct container; PID namespaces may all use PID1",
    )


def validate_identity(data, assignment, contract, args):
    require(
        data["schema_version"] == 1 and data["status"] == "success",
        "complete successful worker result",
    )
    require(
        data["mode"] == ("admission" if args.admission else "timing"),
        "assigned run mode",
    )
    require(
        data["block"] == data["trial"] == assignment["block"]
        and data["cell"] == "auto_synthetic"
        and data["objective"] == "auto"
        and data["values"] == "synthetic",
        "one original auto/synthetic startup per fresh process",
    )
    require(
        data["capture_order"] == assignment["capture_order"]
        and data["assignment"] == assignment,
        "actual frozen assignment",
    )
    require(
        data["worker_sha256"] == sha256(args.worker), "exact executed worker source"
    )
    require(
        data["plan_sha256"] == sha256(args.plan)
        and data["schedule_sha256"] == sha256(args.schedule),
        "exact executed plan and schedule",
    )
    require(
        data["source"] == contract["source"]
        and data["fixture"] == contract["fixture"]
        and data["environment"] == contract["environment"],
        "unchanged qualified source, fixture, physical GPU and numerical stack",
    )
    require(
        not data["protocol_errors"]
        and not data["forbidden_profile_attempts"]
        and data.get("pending_sample") is None,
        "no hidden failures or unfinished measurements",
    )
    require(
        data["initializer_config_restored"] is True
        and data["synthetic_input_references_retained"] is True
        and data["persistence_under_test"] is False
        and data["all_samples_retained"] is True
        and data["queue_query_is_exclusion_criterion"] is False,
        "declared retention/restoration and no outcome filtering",
    )
    require(
        data["fixed_descriptor"] == contract["fixed_descriptor"]
        and data["historical_contract_sha256"] == sha256(args.historical_contract),
        "saved descriptor and historical contract",
    )
    require(
        data["graph_role_alias"]
        == {"winner": "fixed_explicit", "default": "untuned_default"},
        "winner graph label is explicitly a fixed-pair alias",
    )
    require(
        data["parent_worker_sha256"]
        == "979e28aeb3248b27cb27df21b98cf274442c0e6b92c4df3444c48f7214ce90ba",
        "unchanged original experiment implementation",
    )
    require(
        data["declared_late_profile_attempts"] == 48
        and data["total_actual_profiler_calls"] == 129,
        "bounded complete original plus diagnostic calls",
    )
    require(
        data["profiler_objective_order"] == assignment["profiler_objective_order"],
        "declared diagnostic objective order",
    )
    require(
        data["profiling_byte_attestation"]
        == (
            "admission_full_hashes"
            if args.admission
            else "metadata_only_startup_postgraph_read_hashes"
        ),
        "explicit postgraph synthetic byte observation scope",
    )
    layer = data["actual_layer"]
    require(
        {k: layer[k] for k in ("function", "custom_op", "quantizer", "backend")}
        == {
            "function": "apply_fp8_linear_bmm_flashinfer",
            "custom_op": "flashinfer_bmm_fp8",
            "quantizer": "static_quant_fp8",
            "backend": "cublas",
        },
        "actual SGLang full-layer public path",
    )
    require(
        layer["inference_mode"] is True
        and layer["custom_op_direct_wrapper"] is True
        and not any(
            layer[k]
            for k in (
                "grad_enabled",
                "dispatch_modes",
                "function_modes",
                "torch_profiler_enabled",
                "compiling",
                "kernel_api_log_level",
            )
        ),
        "normal inference custom-op path",
    )
    require(
        data["delay_helper"]["prime_cycles"] == 10_000_000
        and data["delay_helper"]["primed_before_startup"] is True,
        "original delay helper initialization",
    )


def validate_final(data, graphs, checks, leaf):
    startup = data["startup"]
    require(
        data["pre_capture_cache_state_sha256"] == startup["cache_state_sha256"],
        "startup cache retained before capture",
    )
    require(
        data["pre_capture_cache_state_field_sha256"]
        == startup["cache_state_field_sha256"],
        "all original cache fields retained before capture",
    )
    for name in ("pre_trace_checks", "final"):
        row = data[name]
        leaf.validate_bare(row, name)
        require(
            row["profile_attempts"] == 81
            and not row["forbidden_profile_attempts"]
            and row["cache_sha256"] == startup["cache_sha256"]
            and row["cache_state_sha256"] == startup["cache_state_sha256"]
            and row["cache_unchanged"] is True
            and row["operands_unchanged"] is True,
            "complete original live/disk cache and serving operand restoration",
        )
        for role, graph in graphs.items():
            leaf.check_owner(
                checks,
                ("pre_trace_" if name == "pre_trace_checks" else "final_") + role,
                graph,
            )


def validate_profiler(data, schedule, contract, earliest):
    retained = data["retained_profile"]
    metadata = data["startup"]["profiles"][80]["metadata"]
    require(
        retained["retained_after_index"] == 80
        and retained["metadata"] == metadata
        and retained["after_pairs_metadata"] == metadata,
        "exact original final-profile objects retained and unchanged",
    )
    ids = retained["tensor_object_ids"]
    require(
        len(ids) == 6
        and len(set(ids)) == 6
        and all(type(i) is int and i > 0 for i in ids),
        "six original tensor object identities",
    )
    require(
        data["startup"]["started_ns"]
        < retained["retained_after_ns"]
        < data["startup"]["ended_ns"]
        < earliest
        < retained["reference_preparation_started_ns"]
        < retained["reference_preparation_ended_ns"],
        "retention after original profiling, reference after graph timing",
    )
    require(
        len(retained["read_operand_hashes"]) == 4
        and all(
            type(h) is str and len(h) == 64 for h in retained["read_operand_hashes"]
        ),
        "postgraph exact synthetic read operand hashes",
    )
    if data["mode"] == "admission":
        require(
            retained["read_operand_hashes"]
            == data["startup"]["profiles"][80]["fingerprints"],
            "admission observes retained original synthetic bytes",
        )
    checks = data["profile_correctness"]
    require(
        len(checks) == 54 and len({r["check"] for r in checks}) == 54,
        "48 diagnostic and6 synthetic native numerical checks",
    )
    checks = {r["check"]: r for r in checks}
    for row in checks.values():
        require(
            row["passed"] is True
            and type(row["max_abs_error"]) in (int, float)
            and math.isfinite(row["max_abs_error"])
            and row["max_abs_error"] >= 0
            and row["output_data_ptr"] == metadata[4]["data_ptr"],
            "finite check on original profiling output",
        )
        if row["exact_zero_required"]:
            require(
                row["max_abs_error"] == 0, "exact-zero synthetic changed-input check"
            )
    attempts = data["diagnostic_profile_attempts"]
    require(
        len(attempts) == 48 and [r["index"] for r in attempts] == list(range(48)),
        "all declared direct actual profiler calls",
    )
    groups = data["paired_profiles"]
    require(
        [g["objective"] for g in groups] == data["profiler_objective_order"],
        "actual objective block order",
    )
    previous = retained["reference_preparation_ended_ns"]
    result, cursor = {}, 0
    for group in groups:
        objective = group["objective"]
        orders = schedule["profiler_pair_schedules"][str(data["block"])][objective]
        expected_settings = copy.deepcopy(contract["original_settings"])
        if objective == "graph":
            expected_settings["config"]["use_cuda_graph"] = True
            expected_settings["measurement_policy"] = {
                "execution_mode": "cuda_graph",
                "cold_l2": None,
                "_timer": "auto",
                "graph_timing": None,
            }
            expected_settings["effective_policy_timer"] = "events"
        require(
            group["settings"] == expected_settings,
            "actual effective objective without other timer/L2/replay changes",
        )
        require(
            previous < group["started_ns"] <= group["preparation"]["ended_ns"],
            "public preparation after previous block",
        )
        require(
            group["preparation"]["public_schedule_uses_exact_retained_objects"] is True,
            "public schedule preserves exact retained operands",
        )
        previous = group["preparation"]["ended_ns"]
        require(len(group["rounds"]) == 12, "all twelve diagnostic pairs")
        ratios, strata, durations = [], {r: [] for r in ROLES}, {r: [] for r in ROLES}
        for index, pair in enumerate(group["rounds"]):
            require(
                pair["index"] == index and pair["order"] == orders[index],
                "frozen actual profiler pair order",
            )
            rows = pair["measurements"]
            require(
                [r["role"] for r in rows] == orders[index],
                "both fixed profiler candidates once per pair",
            )
            for row in rows:
                role = row["role"]
                tactic = contract["fixed_descriptor"] if role == "winner" else -1
                require(
                    row["attempt_index"] == cursor,
                    "pair bound to exact original-profiler invocation",
                )
                attempt = attempts[cursor]
                label = f"late_{objective}_{index:02d}_{role}"
                require(
                    row["tactic"] == attempt["tactic"] == tactic
                    and attempt["objective"] == objective
                    and attempt["stage"] == label,
                    "fixed candidate bound to declared objective",
                )
                require(
                    attempt["error"] is None
                    and attempt["settings"] == expected_settings
                    and attempt["metadata"] == metadata
                    and attempt["tensor_object_ids"] == ids,
                    "unchanged exact original profiler inputs and recipe",
                )
                positive(attempt["elapsed_ms"], "finite actual diagnostic duration")
                close(
                    row["elapsed_ms"],
                    attempt["elapsed_ms"],
                    "actual measured duration retained",
                )
                require(
                    row["correctness_check"] == label,
                    "individual refreshed-output numerical check",
                )
                check = checks[label]
                require(
                    check["exact_zero_required"] is False
                    and previous
                    < attempt["started_ns"]
                    < attempt["ended_ns"]
                    < check["checked_ns"],
                    "nonoverlapping calls followed by individual numerical checks",
                )
                previous = check["checked_ns"]
                durations[role].append(row["elapsed_ms"])
                cursor += 1
            pair_values = {r["role"]: r["elapsed_ms"] for r in rows}
            ratio = pair_values["winner"] / pair_values["default"]
            ratios.append(ratio)
            strata[orders[index][0]].append(ratio)
        require(previous < group["ended_ns"], "complete objective block chronology")
        previous = group["ended_ns"]
        median = statistics.median(ratios)
        close(
            group["explicit_over_default_median"],
            median,
            "independently recomputed profiler ratio",
        )
        result[objective] = {
            "ratio": median,
            "paired_ratios": ratios,
            "first_position_medians": {
                r: statistics.median(v) for r, v in strata.items()
            },
            "median_duration_ms": {
                r: statistics.median(v) for r, v in durations.items()
            },
        }
    restorations = data["cache_partition_restorations"]
    require(
        [r["objective"] for r in restorations] == data["profiler_objective_order"],
        "both diagnostic policy partitions audited",
    )
    for row in restorations:
        require(
            row["restored"] is True
            and row["before_sha256"]
            == row["after_sha256"]
            == data["startup"]["cache_state_sha256"]
            and set(row["changed_fields"])
            <= {"_winner_partitions", "_protocol_record_partitions"},
            "only empty identity partitions allowed and fully restored",
        )
    return result, previous, checks


def validate_synthetic_acceptance(data, contract, checks, earliest):
    rows = data["synthetic_acceptance"]
    require([r["role"] for r in rows] == list(ROLES), "both synthetic native controls")
    metadata = data["retained_profile"]["metadata"]
    previous = earliest
    for row in rows:
        role = row["role"]
        require(
            row["inputs"] == metadata
            and row["tactic"]
            == (contract["fixed_descriptor"] if role == "winner" else -1),
            "native control bound to exact original profiling problem",
        )
        require(
            previous < row["started_ns"] < row["ended_ns"],
            "synthetic controls after all timing",
        )
        require(
            row["used_descriptor"] == (1 if role == "winner" else None)
            and row["descriptor_acceptance"]
            == ("accepted" if role == "winner" else "not_applicable_default_route"),
            "accepted explicit descriptor without fallback substitution",
        )
        labels = [
            "profile_accept_" + role + suffix
            for suffix in ("", "_changed_input", "_restored_input")
        ]
        require(
            row["checks"] == labels,
            "poisoned, changed and restored synthetic native controls",
        )
        last = row["started_ns"]
        for index, label in enumerate(labels):
            check = checks[label]
            require(
                check["exact_zero_required"] is (index == 1)
                and last < check["checked_ns"] < row["ended_ns"],
                "actual ordered changed-input control",
            )
            last = check["checked_ns"]
        previous = row["ended_ns"]
    return previous


def validate_baseline(data, contract, leaf):
    startup = data["startup"]
    rows = startup["profiles"]
    require(
        startup["profile_attempts"] == len(rows) == 81, "all81 original profiling calls"
    )
    require([r["index"] for r in rows] == list(range(81)), "original global call order")
    require(
        [r["m"] for r in rows] == [m for m in ENDPOINTS for _ in range(9)],
        "original nine endpoints",
    )
    require(
        startup["candidate_order"] == contract["candidate_order"],
        "historical full candidate domain and order",
    )
    for offset, m in enumerate(ENDPOINTS):
        group = rows[offset * 9 : (offset + 1) * 9]
        require(
            [r["tactic"] for r in group] == contract["candidate_order"][str(m)],
            "actual historical candidate order",
        )
        for row in group:
            require(row["error"] is None, "no hidden startup profile failures")
            positive(row["elapsed_ms"], "finite startup duration")
            require(
                row["key"] == contract["profile_keys"][row["index"]],
                "historical profile/cache identity",
            )
            require(
                row["settings"] == contract["original_settings"],
                "historical effective profiler settings",
            )
            require(
                row["fingerprinted"] is (data["mode"] == "admission"),
                "no inline timing-startup byte hashing",
            )
            require(
                ("fingerprints" in row) is (data["mode"] == "admission"),
                "byte attestation scope",
            )
            inputs = row["metadata"]
            require(len(inputs) == 6, "all actual profiler operands")
            leaf.validate_tensor(
                inputs[0], [1, m, 1024], "torch.float8_e4m3fn", "profile activation"
            )
            leaf.validate_tensor(
                inputs[1], [1, 1024, 4096], "torch.float8_e4m3fn", "profile weight"
            )
            leaf.validate_tensor(
                inputs[4], [1, m, 4096], "torch.bfloat16", "profile output"
            )
    require(
        rows[72]["tactic"] == -1 and rows[74]["tactic"] == contract["fixed_descriptor"],
        "fixed original pair ordinal72/74",
    )
    require(
        all(r["metadata"] == rows[72]["metadata"] for r in rows[72:]),
        "same target operands across original candidates",
    )
    natural = data["natural"]
    require(
        natural["tactic"] == min(rows[72:], key=lambda r: r["elapsed_ms"])["tactic"],
        "natural winner remains original minimum",
    )
    require(
        natural["key"] == rows[72]["key"]
        and natural["cache_hit"] is True
        and natural["lookup_tier"] == "memory",
        "natural winner real lookup",
    )
    calls = startup["initializer_calls"]
    require(
        len(calls) == 9
        and [r["shape"] for r in calls] == [[1, m, 1024] for m in ENDPOINTS],
        "nine original synthetic activation draws",
    )
    for index, row in enumerate(calls):
        require(
            all(row[k] is (index == 8) for k in ("targeted", "cloned", "copied")),
            "matched M144 clone/copy recipe",
        )
        require(
            ("generated_sha256" in row) is (data["mode"] == "admission"),
            "timing initializer does not hash bytes",
        )
    require(startup["incidental_legacy_entries"] == 9, "original legacy persistence")
    ratio = rows[74]["elapsed_ms"] / rows[72]["elapsed_ms"]
    if "original_history" in data:
        history = data["original_history"]
        require(
            history["default_index"] == 72
            and history["explicit_index"] == 74
            and history["actual_m"] == 144
            and history["canonical_cache_m"] == 256
            and history["all_81_candidate_order_keys_settings_match"] is True,
            "declared original history",
        )
        close(history["default_ms"], rows[72]["elapsed_ms"], "original default time")
        close(
            history["explicit_ms"],
            rows[74]["elapsed_ms"],
            "original fixed descriptor time",
        )
        close(history["explicit_over_default"], ratio, "original fixed pair ratio")
    return ratio


def validate_graphs(data, contract, leaf, checks):
    rows = data["graphs"]
    expected = list(
        ROLES if data["capture_order"] == "winner-first" else reversed(ROLES)
    )
    require([r["id"] for r in rows] == expected, "prespecified graph capture order")
    require(
        len({r["graph_object_id"] for r in rows})
        == len({r["output"]["data_ptr"] for r in rows})
        == 2,
        "distinct original graphs and outputs",
    )
    graphs = {r["id"]: r for r in rows}
    for role, graph in graphs.items():
        tactic = contract["fixed_descriptor"] if role == "winner" else -1
        require(
            graph["tactic"] == tactic, "fixed descriptor irrespective of natural winner"
        )
        require(
            graph["lookup_tier"] == ("memory" if role == "winner" else "miss"),
            "real fixed hit/default miss",
        )
        selection = graph["selection"]
        require(
            selection["tactic"] == tactic
            and selection["cache_hit"] is (role == "winner")
            and selection["capturing"] is True,
            "actual capture selection",
        )
        require(
            selection["is_tuning_mode"] is False
            and selection["managed_store"] is False
            and selection["measurement_policy"] is None,
            "capture replay-only auto policy",
        )
        inputs = graph["inputs"]
        require(
            len(inputs) == 6 and inputs[4]["data_ptr"] == graph["output"]["data_ptr"],
            "actual graph-owned output",
        )
        leaf.validate_tensor(
            inputs[0], [1, 144, 1024], "torch.float8_e4m3fn", "capture activation"
        )
        leaf.validate_tensor(
            inputs[1], [1, 1024, 4096], "torch.float8_e4m3fn", "capture weight"
        )
        leaf.validate_tensor(
            graph["output"], [144, 4096], "torch.bfloat16", "layer output"
        )
        require(
            inputs[0]["stride"][-2:] == [1024, 1]
            and inputs[1]["stride"][-2:] == [1, 1024],
            "original layouts",
        )
        require(
            graph["read_operand_hashes"][0]
            == contract["fixture"]["quantized_input_sha256"],
            "actual serving quantized input",
        )
        for suffix in ("poisoned", "restored_input"):
            leaf.check_owner(checks, role + "_" + suffix, graph)
        leaf.check_owner(checks, role + "_changed_input", graph, zero=True)
        require(
            data["startup"]["ended_ns"]
            < graph["capture_started_ns"]
            < graph["capture_ended_ns"],
            "capture after complete startup",
        )
    require(
        graphs["winner"]["read_operand_hashes"]
        == graphs["default"]["read_operand_hashes"],
        "same serving read operands",
    )
    return graphs


def validate_graph_timing(data, schedule, graphs, checks, leaf):
    if data["mode"] == "admission":
        require(data["rounds"] == [], "admission excluded from graph performance")
        return None, max(g["capture_ended_ns"] for g in graphs.values())
    orders = schedule["round_schedules"][str(data["block"])]["orders"]
    require(
        data["round_schedule"] == orders and len(data["rounds"]) == 24,
        "all prespecified graph rounds",
    )
    require(
        [r["graph_id"] for r in data["initial_warmups"]] == list(ROLES),
        "initial graph warmups",
    )
    require(
        all(
            r["actual_ms"] >= 500 and r["calls"] >= 5120
            for r in data["initial_warmups"]
        ),
        "500ms initial warmups",
    )
    previous = max(g["capture_ended_ns"] for g in graphs.values())
    ratios, strata, queries = [], {r: [] for r in ROLES}, collections.Counter()
    for index, group in enumerate(data["rounds"]):
        require(
            group["index"] == index and group["order"] == orders[index],
            "actual scheduled graph round",
        )
        rows = group["measurements"]
        require(
            [r["graph_id"] for r in rows] == orders[index],
            "both original graph measurements",
        )
        for row in rows:
            role, graph = row["graph_id"], graphs[row["graph_id"]]
            require(
                row["graph_object_id"] == graph["graph_object_id"]
                and row["output_data_ptr"] == graph["output"]["data_ptr"],
                "original measured graph/output owner",
            )
            require(
                row["replays"] == 16
                and row["prequeue_cycles"] == 10_000_000
                and row["rewarm"]["actual_ms"] >= 20
                and row["rewarm"]["calls"] >= 5120,
                "unchanged graph timing recipe",
            )
            require(
                row["profile_attempts_before"] == row["profile_attempts_after"] == 81,
                "no profiling in graph windows",
            )
            require(
                previous
                < row["wall_started_ns"]
                <= row["enqueue_started_ns"]
                < row["enqueue_ended_ns"]
                <= row["ended_ns"],
                "nonoverlapping graph chronology",
            )
            previous = row["ended_ns"]
            positive(row["gpu_event_batch_ms"], "finite event duration")
            close(
                row["gpu_event_per_replay_ms"],
                row["gpu_event_batch_ms"] / 16,
                "per replay normalization",
            )
            require(
                type(row["start_event_complete_after_enqueue"]) is bool
                and row["start_query_error"] is None,
                "actual queue observation",
            )
            outcome = (
                "start_complete_inconclusive"
                if row["start_event_complete_after_enqueue"]
                else "start_pending_after_enqueue"
            )
            require(row["queue_evidence"] == outcome, "honest queue evidence")
            queries[outcome] += 1
            require(
                row["correctness_check"] == f"round_{index:02d}_{role}",
                "per sample numerical check",
            )
            leaf.check_owner(checks, row["correctness_check"], graph)
        paired = {r["graph_id"]: r["gpu_event_per_replay_ms"] for r in rows}
        ratio = paired["winner"] / paired["default"]
        ratios.append(ratio)
        strata[orders[index][0]].append(ratio)
    result = {
        "ratio": statistics.median(ratios),
        "paired_ratios": ratios,
        "first_position_medians": {r: statistics.median(v) for r, v in strata.items()},
        "queue_outcomes_all_retained": dict(queries),
    }
    close(
        data["selected_over_default_graph_event_median"],
        result["ratio"],
        "independently recomputed graph estimator",
    )
    return result, previous


def quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize(processes, replicates=100000):
    """Paired process resampling; replay windows never enlarge sample size."""
    require(
        len(processes) == 8 and [p["block"] for p in processes] == list(range(8)),
        "complete fixed eight-process cohort",
    )
    metrics = {
        "original_history_profiler": [
            p["original_history_profiler_ratio"] for p in processes
        ],
        "layer_graph": [p["layer_graph"]["ratio"] for p in processes],
        "late_auto_profiler": [p["profiler"]["auto"]["ratio"] for p in processes],
        "late_graph_profiler": [p["profiler"]["graph"]["ratio"] for p in processes],
        "late_graph_over_auto_profiler": [
            p["profiler"]["graph"]["ratio"] / p["profiler"]["auto"]["ratio"]
            for p in processes
        ],
    }
    logs = {
        k: [math.log(positive(x, "finite process ratio")) for x in v]
        for k, v in metrics.items()
    }
    draws = {k: [] for k in metrics}
    rng = random.Random(202610262200)
    for _ in range(replicates):
        indices = [rng.randrange(8) for _ in range(8)]
        for name, values in logs.items():
            draws[name].append(math.exp(sum(values[i] for i in indices) / 8))
    result = {}
    for name, values in metrics.items():
        result[name] = {
            "processes": 8,
            "geometric_mean_ratio": math.exp(statistics.mean(logs[name])),
            "min": min(values),
            "max": max(values),
            "below_one": sum(x < 1 for x in values),
            "above_one": sum(x > 1 for x in values),
            "equal_one": sum(x == 1 for x in values),
            "exploratory_process_bootstrap_95pct": [
                quantile(draws[name], 0.025),
                quantile(draws[name], 0.975),
            ],
        }
    return {
        "metrics": result,
        "process_rows": processes,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": 202610262200,
        "independent_unit": "fresh process (8), not paired measurements",
        "limits": "Same physical GPU and fixed workload; exploratory descriptive intervals, no population or serving-performance claim. Historical and admission results excluded.",
    }


def validate_scopes(data, leaf):
    stages = {
        "startup",
        "natural_lookup",
        "winner_capture",
        "default_capture",
        "late_auto",
        "late_graph",
    }
    records = data["startup"]["scope_records"]
    require(
        collections.Counter((r["stage"], r["event"]) for r in records)
        == collections.Counter(
            (stage, event)
            for stage in stages
            for event in ("enter", "before_exit", "after_exit")
        ),
        "complete actual policy scope history",
    )
    for row in records:
        if row["event"] == "after_exit":
            require(
                row["active_v2"] is False
                and row["measurement_policy"] is None
                and row["managed_store"] is False,
                "policy restored after actual scope",
            )
            # Late scope exit is recorded while still on the forward stream;
            # complete bare caller-stream restoration is checked after the block.
            if row["stage"] not in ("startup", "late_auto", "late_graph"):
                leaf.validate_bare(row, "capture scope exit")
        else:
            policy = (
                {
                    "execution_mode": "cuda_graph",
                    "cold_l2": None,
                    "_timer": "auto",
                    "graph_timing": None,
                }
                if row["stage"] == "late_graph"
                else None
            )
            require(
                row["active_v2"] is True
                and row["managed_store"] is False
                and row["measurement_policy"] == policy
                and row["is_tuning_mode"]
                is (row["stage"] in ("startup", "late_auto", "late_graph")),
                "only declared startup/direct profiling scopes can tune",
            )
    require(len(data["miss_scopes"]) == 2, "both controlled graph cache scopes")
    for row in data["miss_scopes"]:
        require(
            row["restored"] is True
            and row["winner_entries_before"] == 9
            and row["winner_entries_after_removal"]
            == row["file_entries_before"]
            == row["file_entries_after_removal"]
            == 0
            and row["cache_sha256_before"]
            == row["cache_sha256_after"]
            == data["startup"]["cache_sha256"],
            "reversible real cache scope preserved original winners and disk bytes",
        )


def validate_worker(data, assignment, contract, schedule, args, leaf, raw_path):
    validate_identity(data, assignment, contract, args)
    original = validate_baseline(data, contract, leaf)
    checks = leaf.correctness_index(data)
    graphs = validate_graphs(data, contract, leaf, checks)
    graph_timing, last_graph = validate_graph_timing(
        data, schedule, graphs, checks, leaf
    )
    profiler, last_profile, profile_checks = validate_profiler(
        data, schedule, contract, last_graph
    )
    leaf.validate_acceptance(
        data["post_timing_acceptance"], graphs["winner"], checks, last_profile
    )
    leaf.validate_acceptance(
        data["default_acceptance"],
        graphs["default"],
        checks,
        data["post_timing_acceptance"]["ended_ns"],
    )
    last_native = validate_synthetic_acceptance(
        data, contract, profile_checks, data["default_acceptance"]["ended_ns"]
    )
    require(
        last_native < data["pre_trace_checks"]["ended_ns"],
        "all native checks precede tracing",
    )
    validate_scopes(data, leaf)
    validate_final(data, graphs, checks, leaf)
    trace = leaf.validate_trace(
        data, graphs, checks, args.raw_root, raw_path.stem + ".trace.json"
    )
    return {
        "block": data["block"],
        "pid": data["pid"],
        "result_sha256": sha256(raw_path),
        "capture_order": data["capture_order"],
        "profiler_objective_order": data["profiler_objective_order"],
        "natural_tactic": data["natural"]["tactic"],
        "fixed_descriptor": data["fixed_descriptor"],
        "original_history_profiler_ratio": original,
        "layer_graph": graph_timing,
        "profiler": profiler,
        "trace": trace,
    }


def analyze(args):
    plan, schedule, contract = (
        load(args.plan),
        load(args.schedule),
        load(args.historical_contract),
    )
    require(
        plan["schedule_sha256"] == sha256(args.schedule)
        and plan["historical_contract_sha256"] == sha256(args.historical_contract),
        "frozen plan bindings",
    )
    require(
        plan["protocol"]["fresh_processes"] == 8
        and plan["protocol"]["original_measurements"] == 81
        and plan["protocol"]["fixed_descriptor"] == contract["fixed_descriptor"],
        "bounded fixed pair protocol",
    )
    validate_schedule(schedule)
    leaf = leaf_helpers(args.leaf_helpers)
    entries = schedule["entries"][:1] if args.admission else schedule["entries"]
    processes, containers = [], []
    for entry in entries:
        name = (
            "admission-auto_synthetic.json"
            if args.admission
            else f"block{entry['block']}-auto_synthetic-timing.json"
        )
        path = args.raw_root / name
        processes.append(
            validate_worker(load(path), entry, contract, schedule, args, leaf, path)
        )
        containers.append(path.with_suffix(".container-id").read_text().strip())
    if not args.admission:
        containers.append(
            (args.raw_root / "admission-auto_synthetic.container-id")
            .read_text()
            .strip()
        )
    validate_container_ids(containers)
    result = {
        "status": "passed",
        "mode": "admission" if args.admission else "timing",
        "plan_sha256": sha256(args.plan),
        "schedule_sha256": sha256(args.schedule),
        "historical_contract_sha256": sha256(args.historical_contract),
        "worker_sha256": sha256(args.worker),
        "analyzer_sha256": sha256(__file__),
        "leaf_helpers_sha256": LEAF_SHA,
        "distinct_container_ids": containers,
    }
    if args.admission:
        raw = args.raw_root / "admission-auto_synthetic.json"
        bootstrap_path = (
            args.admission_bootstrap
            or args.raw_root / "admission-auto_synthetic.bootstrap.json"
        )
        bootstrap = load(bootstrap_path)
        require(
            bootstrap["status"] == "passed"
            and bootstrap["result_sha256"] == sha256(raw)
            and bootstrap["worker_sha256"] == sha256(args.worker),
            "qualified admission bootstrap binds exact successful result and worker",
        )
        result.update(
            admission_result_sha256=sha256(raw),
            admission_bootstrap_sha256=sha256(bootstrap_path),
            integrity_only=True,
            performance_inference=False,
            process_integrity=processes,
        )
    else:
        result.update(summarize(processes))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "plan",
        "schedule",
        "historical-contract",
        "raw-root",
        "worker",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--leaf-helpers",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "20261024T180000Z-h100-objective-values"
        / "analyze.py",
    )
    parser.add_argument("--admission", action="store_true")
    parser.add_argument("--admission-bootstrap", type=Path)
    args = parser.parse_args()
    try:
        result = analyze(args)
    except Exception as error:
        result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        raise
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "mode": result["mode"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
