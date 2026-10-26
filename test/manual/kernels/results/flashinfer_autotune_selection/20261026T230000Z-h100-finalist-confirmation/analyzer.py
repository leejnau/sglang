#!/usr/bin/env python3
"""Independent admission and process-level analysis for shadow confirmation.

The original and hypothetical decisions share the same post-confirmation
graphs. This does not compare untouched production startup with a new policy.
"""

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import random
import statistics
from pathlib import Path

PRIOR_ANALYZER_SHA = "b73acfc80a676ad24bb0358a6d66880531947f628783cba6fb1765c5154269ec"
BASE_WORKER_SHA = "979e28aeb3248b27cb27df21b98cf274442c0e6b92c4df3444c48f7214ce90ba"
FIXED_HELPER_SHA = "7c8c743fe08b898035bbdf693123ea7b0c8a2d49a29db801f0db3c81178abf42"
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


def helpers(path):
    """Reuse frozen ownership, trace and native-acceptance validators only."""
    require(sha256(path) == PRIOR_ANALYZER_SHA, "unchanged prior independent validator")
    spec = importlib.util.spec_from_file_location("fixed_pair_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    leaf = module.leaf_helpers(
        path.parent.parent / "20261024T180000Z-h100-objective-values" / "analyze.py"
    )
    return module, leaf


def pair_logs(rounds, role_key):
    logs = []
    for pair in rounds:
        rows = pair["measurements"]
        require(
            len(rows) == 2 and sorted(r[role_key] for r in rows) == sorted(ROLES),
            "exactly one measurement per paired role",
        )
        values = {
            r[role_key]: positive(
                r["elapsed_ms"] if role_key == "role" else r["gpu_event_per_replay_ms"],
                "finite paired duration",
            )
            for r in rows
        }
        logs.append(math.log(values["winner"] / values["default"]))
    return logs


def expected_decision(rounds, explicit, decided_ns):
    require(len(rounds) == 12, "decision uses every prespecified confirmation pair")
    logs = pair_logs(rounds, "role")
    median = statistics.median(logs)
    role = "winner" if median < 0 else "default"
    return {
        "explicit_tactic": explicit,
        "default_tactic": -1,
        "pair_orders": [r["order"] for r in rounds],
        "log_ratios": logs,
        "median_log_ratio": median,
        "decision_role": role,
        "decision_tactic": explicit if role == "winner" else -1,
        "decided_ns": decided_ns,
    }


def digest(payload):
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def choice_effect(original_role, confirmed_role, graph_log_ratio):
    require(
        original_role in ROLES
        and confirmed_role in ROLES
        and math.isfinite(graph_log_ratio),
        "valid frozen choices and held-out graph estimate",
    )
    change = int(confirmed_role == "winner") - int(original_role == "winner")
    value = change * graph_log_ratio
    outcome = (
        "unchanged_choice"
        if change == 0
        else "measured_better"
        if value < 0
        else "measured_worse"
        if value > 0
        else "changed_choice_measured_tie"
    )
    old_regret = int(original_role == "winner") * graph_log_ratio - min(
        0, graph_log_ratio
    )
    new_regret = int(confirmed_role == "winner") * graph_log_ratio - min(
        0, graph_log_ratio
    )
    return {
        "log_hypothetical_over_original": value,
        "hypothetical_over_original": math.exp(value),
        "original_log_regret_two_candidate_only": old_regret,
        "confirmed_log_regret_two_candidate_only": new_regret,
        "outcome": outcome,
        "transition": original_role + "->" + confirmed_role,
    }


def validate_schedule(schedule, prior):
    entries = schedule["entries"]
    require(
        len(entries) == 8
        and [e["block"] for e in entries] == list(range(8))
        and all(e["cell"] == "auto_synthetic" for e in entries),
        "eight frozen processes and one startup recipe",
    )
    require(
        collections.Counter(e["capture_order"] for e in entries)
        == {"winner-first": 4, "default-first": 4},
        "independently balanced graph capture order",
    )
    for block in range(8):
        prior.validate_orders(
            schedule["confirmation_pair_schedules"][str(block)],
            12,
            "confirmation schedule",
        )
        prior.validate_orders(
            schedule["round_schedules"][str(block)]["orders"], 24, "graph schedule"
        )


def validate_identity(data, assignment, contract, args):
    require(
        data["schema_version"] == 1
        and data["status"] == "success"
        and data["study"] == "best_explicit_confirmation",
        "complete declared study result",
    )
    require(
        data["mode"] == ("admission" if args.admission else "timing")
        and data["trial"] == data["block"] == assignment["block"]
        and data["cell"] == "auto_synthetic"
        and data["objective"] == "auto"
        and data["values"] == "synthetic",
        "actual fresh-process assignment",
    )
    require(
        data["assignment"] == assignment
        and data["capture_order"] == assignment["capture_order"],
        "frozen capture order independent of decision",
    )
    require(
        data["worker_sha256"] == sha256(args.worker)
        and data["parent_worker_sha256"] == BASE_WORKER_SHA
        and data["fixed_pair_helper_sha256"] == FIXED_HELPER_SHA,
        "exact worker and helper sources",
    )
    require(
        data["plan_sha256"] == sha256(args.plan)
        and data["schedule_sha256"] == sha256(args.schedule)
        and data["historical_contract_sha256"] == sha256(args.historical_contract),
        "executed frozen protocol",
    )
    require(
        data["source"] == contract["source"]
        and data["fixture"] == contract["fixture"]
        and data["environment"] == contract["environment"],
        "qualified source, physical GPU, stack and fixture",
    )
    require(
        not data["protocol_errors"]
        and not data["forbidden_profile_attempts"]
        and data.get("pending_sample") is None,
        "no omitted failed or unfinished measurement",
    )
    require(
        data["declared_late_profile_attempts"] == 24
        and data["total_actual_profiler_calls"] == 105,
        "complete bounded81+24 profiling calls",
    )
    require(
        data["graph_role_alias"]
        == {"winner": "best_explicit", "default": "untuned_default"},
        "explicit graph is the observed finalist",
    )
    require(
        data["initializer_config_restored"] is True
        and data["synthetic_input_references_retained"] is True
        and data["persistence_under_test"] is False
        and data["all_samples_retained"] is True
        and data["queue_query_is_exclusion_criterion"] is False,
        "restored controls and no outcome filtering",
    )
    require(
        data["profiling_byte_attestation"]
        == (
            "admission_full_hashes"
            if args.admission
            else "metadata_only_startup_poststartup_read_hashes"
        ),
        "declared synthetic byte observation history",
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
        "actual SGLang full-layer path",
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
        "ordinary inference path",
    )
    require(
        data["delay_helper"]["prime_cycles"] == 10_000_000
        and data["delay_helper"]["primed_before_startup"] is True,
        "unchanged delay helper initialization",
    )


def validate_startup(data, contract, leaf):
    startup, rows = data["startup"], data["startup"]["profiles"]
    require(
        startup["profile_attempts"] == len(rows) == 81
        and [r["index"] for r in rows] == list(range(81)),
        "complete original81-call history",
    )
    require(
        [r["m"] for r in rows] == [m for m in ENDPOINTS for _ in range(9)]
        and startup["candidate_order"] == contract["candidate_order"],
        "unchanged nine profile endpoints and candidate pools",
    )
    for offset, m in enumerate(ENDPOINTS):
        group = rows[offset * 9 : (offset + 1) * 9]
        require(
            [r["tactic"] for r in group] == contract["candidate_order"][str(m)],
            "actual original candidate order",
        )
        for row in group:
            require(
                row["error"] is None
                and row["key"] == contract["profile_keys"][row["index"]]
                and row["settings"] == contract["original_settings"],
                "original finite profile identity and effective settings",
            )
            positive(row["elapsed_ms"], "finite original profile")
            require(
                row["fingerprinted"] is (data["mode"] == "admission")
                and ("fingerprints" in row) is (data["mode"] == "admission"),
                "admission-only startup byte observation",
            )
            require(len(row["metadata"]) == 6, "all six actual profiling operands")
            leaf.validate_tensor(
                row["metadata"][0],
                [1, m, 1024],
                "torch.float8_e4m3fn",
                "profile activation",
            )
            leaf.validate_tensor(
                row["metadata"][4], [1, m, 4096], "torch.bfloat16", "profile output"
            )
    require(
        rows[72]["tactic"] == -1
        and all(r["metadata"] == rows[72]["metadata"] for r in rows[72:]),
        "same runtime operands across default and all explicit candidates",
    )
    finalist = min(rows[73:], key=lambda r: r["elapsed_ms"])
    declared = data["finalist"]
    require(
        {k: declared[k] for k in ("tactic", "key", "elapsed_ms")}
        == {k: finalist[k] for k in ("tactic", "key", "elapsed_ms")}
        and declared["profile_index"] == finalist["index"],
        "first minimum of all observed explicit candidates; no fixed descriptor substitution",
    )
    natural = min(rows[72:], key=lambda r: r["elapsed_ms"])
    choices = data["startup_choices"]
    require(
        len(choices) == 1
        and choices[0]["op"] == "fp8_gemm"
        and choices[0]["profile_attempts"] == 81
        and choices[0]["tactic"] == natural["tactic"]
        and choices[0]["key"] == natural["key"],
        "actual original public choose_one result",
    )
    require(
        type(choices[0]["requested_runner_identity"]) is int
        and choices[0]["requested_runner_identity"] > 0
        and choices[0]["requested_runner_identity"]
        == choices[0]["returned_runner_identity"],
        "actual startup preserves requested runner identity",
    )
    saved = data["natural"]
    require(
        saved["tactic"] == natural["tactic"]
        and saved["key"] == natural["key"]
        and saved["first_pass_profile_index"] == natural["index"]
        and saved["cache_hit"] is True
        and saved["lookup_tier"] == "memory"
        and saved["cache_preserved_after_confirmation"] is True,
        "actual natural winner survives hypothetical confirmation",
    )
    require(
        saved["tactic"] in (-1, finalist["tactic"])
        and saved["cache_sha256"] == startup["cache_sha256"]
        and saved["cache_state_sha256"] == startup["cache_state_sha256"],
        "original choice and live/disk cache preserved",
    )
    retained = data["retained_profile"]
    require(
        startup["started_ns"]
        < retained["retained_after_ns"]
        < choices[0]["returned_ns"]
        < startup["ended_ns"]
        < saved["frozen_ns"],
        "original return recorded before confirmation",
    )
    calls = startup["initializer_calls"]
    require(
        len(calls) == 9
        and [r["shape"] for r in calls] == [[1, m, 1024] for m in ENDPOINTS],
        "nine original activation draws",
    )
    for index, row in enumerate(calls):
        require(
            all(row[k] is (index == 8) for k in ("targeted", "cloned", "copied"))
            and ("generated_sha256" in row) is (data["mode"] == "admission"),
            "matched M144 initializer with no timing-startup hash",
        )
    require(
        startup["incidental_legacy_entries"] == 9, "original v1 save remains intact"
    )
    history = data["original_history"]
    require(
        history["default_index"] == 72
        and history["explicit_index"] == finalist["index"]
        and history["actual_m"] == 144
        and history["canonical_cache_m"] == 256
        and history["all_81_candidate_order_keys_settings_match"] is True,
        "dynamic finalist original ordinal",
    )
    close(history["default_ms"], rows[72]["elapsed_ms"], "original default time")
    close(history["explicit_ms"], finalist["elapsed_ms"], "original finalist time")
    close(
        history["explicit_over_default"],
        finalist["elapsed_ms"] / rows[72]["elapsed_ms"],
        "original finalist ratio",
    )
    return finalist


def validate_confirmation(data, schedule, contract, explicit):
    retained, confirmation = data["retained_profile"], data["confirmation"]
    metadata = data["startup"]["profiles"][80]["metadata"]
    require(
        retained["retained_after_index"] == 80
        and retained["metadata"] == metadata
        and retained["after_confirmation_metadata"] == metadata,
        "exact final-profile operands retained without substitution",
    )
    ids = retained["tensor_object_ids"]
    require(
        len(ids) == len(set(ids)) == 6 and all(type(v) is int and v > 0 for v in ids),
        "six retained tensor object identities",
    )
    require(
        data["natural"]["frozen_ns"]
        < retained["reference_preparation_started_ns"]
        < retained["reference_preparation_ended_ns"]
        < confirmation["started_ns"]
        <= confirmation["preparation"]["ended_ns"],
        "reference and confirmation follow untouched original search",
    )
    hashes = retained["read_operand_hashes"]
    require(
        len(hashes) == 4 and all(type(h) is str and len(h) == 64 for h in hashes),
        "actual poststartup synthetic read hashes",
    )
    if data["mode"] == "admission":
        require(
            hashes == data["startup"]["profiles"][80]["fingerprints"],
            "admission retained synthetic bytes unchanged",
        )
    require(
        confirmation["objective"] == "auto"
        and confirmation["settings"] == contract["original_settings"]
        and confirmation["preparation"]["public_schedule_uses_exact_retained_objects"]
        is True,
        "actual auto profiler and exact public input schedule",
    )
    checks = data["profile_correctness"]
    require(
        len(checks) == len({r["check"] for r in checks}) == 30,
        "24 confirmation and6 synthetic native numerical checks",
    )
    checks = {r["check"]: r for r in checks}
    for check in checks.values():
        require(
            check["passed"] is True
            and type(check["max_abs_error"]) in (int, float)
            and math.isfinite(check["max_abs_error"])
            and check["max_abs_error"] >= 0
            and check["output_data_ptr"] == metadata[4]["data_ptr"],
            "finite numerical check on exact retained output",
        )
        if check["exact_zero_required"]:
            require(
                check["max_abs_error"] == 0, "synthetic changed input yields exact zero"
            )
    attempts = data["diagnostic_profile_attempts"]
    require(
        len(attempts) == 24 and [r["index"] for r in attempts] == list(range(24)),
        "every declared original-profiler confirmation call",
    )
    rounds = confirmation["rounds"]
    orders = schedule["confirmation_pair_schedules"][str(data["block"])]
    require(len(rounds) == 12, "all confirmation pairs retained")
    previous, cursor = confirmation["preparation"]["ended_ns"], 0
    for index, pair in enumerate(rounds):
        require(
            pair["index"] == index
            and pair["order"] == orders[index]
            and [r["role"] for r in pair["measurements"]] == orders[index],
            "actual frozen balanced confirmation order",
        )
        for row in pair["measurements"]:
            role, attempt = row["role"], attempts[cursor]
            label = f"confirmation_{index:02d}_{role}"
            require(
                row["attempt_index"] == cursor
                and row["tactic"]
                == attempt["tactic"]
                == (explicit if role == "winner" else -1),
                "pair bound to actual finalist/default invocation",
            )
            require(
                attempt["error"] is None
                and attempt["objective"] == "auto"
                and attempt["stage"] == label
                and attempt["settings"] == contract["original_settings"]
                and attempt["metadata"] == metadata
                and attempt["tensor_object_ids"] == ids,
                "actual original profiler, same operands, unchanged settings",
            )
            positive(attempt["elapsed_ms"], "finite confirmation duration")
            close(
                row["elapsed_ms"],
                attempt["elapsed_ms"],
                "actual profile duration retained",
            )
            require(
                row["correctness_check"] == label,
                "each fresh output numerically checked",
            )
            check = checks[label]
            require(
                check["exact_zero_required"] is False
                and previous
                < attempt["started_ns"]
                < attempt["ended_ns"]
                < check["checked_ns"],
                "nonoverlapping actual calls and output checks",
            )
            previous, cursor = check["checked_ns"], cursor + 1
    require(
        previous
        < confirmation["ended_ns"]
        < confirmation["decided_ns"]
        < data["natural"]["confirmed_after_ns"]
        < min(g["capture_started_ns"] for g in data["graphs"]),
        "authoritative decision precedes both held-out graph captures",
    )
    expected = expected_decision(rounds, explicit, confirmation["decided_ns"])
    require(
        confirmation["decision_payload"] == expected
        and all(confirmation[k] == v for k, v in expected.items()),
        "decision independently reconstructed from all12 raw paired log ratios",
    )
    decision_hash = digest(expected)
    require(
        confirmation["decision_sha256"]
        == confirmation["initial_decision_sha256"]
        == data["final"]["decision_sha256"]
        == decision_hash,
        "immutable confirmation payload through final output",
    )
    for graph in data["graphs"]:
        require(
            graph["confirmation_decision_sha256"] == decision_hash,
            "captured graph bound to the preexisting decision",
        )
    restorations = data["cache_partition_restorations"]
    require(
        len(restorations) == 1 and restorations[0]["objective"] == "auto",
        "single automatic confirmation scope",
    )
    row = restorations[0]
    require(
        row["restored"] is True
        and row["before_sha256"]
        == row["after_sha256"]
        == data["startup"]["cache_state_sha256"]
        and set(row["changed_fields"])
        <= {"_winner_partitions", "_protocol_record_partitions"},
        "original winner/file/protocol state conserved",
    )
    original_role = "default" if data["natural"]["tactic"] == -1 else "winner"
    require(
        data["graph_choices"]
        == {
            "natural_role": original_role,
            "confirmation_role": expected["decision_role"],
            "both_evaluated_on_same_retained_graphs": True,
        },
        "original and hypothetical choices share held-out graphs",
    )
    return checks, expected


def validate_scopes(data, leaf):
    stages = {
        "startup",
        "confirmation",
        "natural_lookup",
        "winner_capture",
        "default_capture",
    }
    records = data["startup"]["scope_records"]
    require(
        collections.Counter((r["stage"], r["event"]) for r in records)
        == collections.Counter(
            (stage, event)
            for stage in stages
            for event in ("enter", "before_exit", "after_exit")
        ),
        "complete declared scope history",
    )
    for row in records:
        require(
            row["measurement_policy"] is None and row["managed_store"] is False,
            "no explicit measurement policy or managed persistence",
        )
        if row["event"] == "after_exit":
            require(row["active_v2"] is False, "v2 scope restored")
            if row["stage"] not in ("startup", "confirmation"):
                leaf.validate_bare(row, "capture scope exit")
        else:
            require(
                row["active_v2"] is True
                and row["is_tuning_mode"]
                is (row["stage"] in ("startup", "confirmation")),
                "only original search and declared confirmation can profile",
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
            "temporary explicit/default graph scopes restore original cache",
        )


def validate_worker(data, assignment, contract, schedule, args, prior, leaf, path):
    validate_identity(data, assignment, contract, args)
    finalist = validate_startup(data, contract, leaf)
    profile_checks, decision = validate_confirmation(
        data, schedule, contract, finalist["tactic"]
    )
    checks = leaf.correctness_index(data)
    # The old graph/native validators accept an expected explicit tactic;
    # supply this process's independently selected finalist, never an old one.
    dynamic_contract = dict(contract, fixed_descriptor=finalist["tactic"])
    graphs = prior.validate_graphs(data, dynamic_contract, leaf, checks)
    timings, last_graph = prior.validate_graph_timing(
        data, schedule, graphs, checks, leaf
    )
    leaf.validate_acceptance(
        data["post_timing_acceptance"], graphs["winner"], checks, last_graph
    )
    leaf.validate_acceptance(
        data["default_acceptance"],
        graphs["default"],
        checks,
        data["post_timing_acceptance"]["ended_ns"],
    )
    last_native = prior.validate_synthetic_acceptance(
        data, dynamic_contract, profile_checks, data["default_acceptance"]["ended_ns"]
    )
    require(
        last_native < data["pre_trace_checks"]["ended_ns"],
        "native controls precede trace",
    )
    validate_scopes(data, leaf)
    prior.validate_final(data, graphs, checks, leaf)
    trace = leaf.validate_trace(
        data, graphs, checks, args.raw_root, path.stem + ".trace.json"
    )
    original_role = "default" if data["natural"]["tactic"] == -1 else "winner"
    result = {
        "block": data["block"],
        "pid_namespace_local": data["pid"],
        "result_sha256": sha256(path),
        "capture_order": data["capture_order"],
        "finalist": finalist["tactic"],
        "finalist_profile_index": finalist["index"],
        "original_role": original_role,
        "confirmed_role": decision["decision_role"],
        "original_profiler_ratio": data["original_history"]["explicit_over_default"],
        "confirmation_log_ratio": decision["median_log_ratio"],
        "confirmation_ratio": math.exp(decision["median_log_ratio"]),
        "decision_sha256": digest(decision),
        "trace": trace,
    }
    result["confirmation_first_position_median_logs"] = {
        role: statistics.median(
            log
            for log, order in zip(
                decision["log_ratios"], decision["pair_orders"], strict=True
            )
            if order[0] == role
        )
        for role in ROLES
    }
    if timings is not None:
        logs = pair_logs(data["rounds"], "graph_id")
        graph_log = statistics.median(logs)
        result.update(
            graph_log_ratio=graph_log,
            graph_ratio=math.exp(graph_log),
            graph_first_position_median_logs={
                role: statistics.median(
                    value
                    for value, pair in zip(logs, data["rounds"], strict=True)
                    if pair["order"][0] == role
                )
                for role in ROLES
            },
            queue_outcomes_all_retained=timings["queue_outcomes_all_retained"],
            graph_log_ratios=logs,
            **choice_effect(original_role, decision["decision_role"], graph_log),
        )
    return result


def quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def summarize(processes, replicates=100000):
    require(
        len(processes) == 8 and [p["block"] for p in processes] == list(range(8)),
        "all eight processes retained for inference",
    )
    log_metrics = {
        "hypothetical_over_original": [
            p["log_hypothetical_over_original"] for p in processes
        ],
        "finalist_graph_over_default": [p["graph_log_ratio"] for p in processes],
        "confirmation_finalist_over_default": [
            p["confirmation_log_ratio"] for p in processes
        ],
        "original_profile_finalist_over_default": [
            math.log(p["original_profiler_ratio"]) for p in processes
        ],
        "original_regret_two_candidate_only": [
            p["original_log_regret_two_candidate_only"] for p in processes
        ],
        "confirmed_regret_two_candidate_only": [
            p["confirmed_log_regret_two_candidate_only"] for p in processes
        ],
    }
    require(
        all(math.isfinite(v) for values in log_metrics.values() for v in values),
        "finite process estimates",
    )
    draws, rng = {k: [] for k in log_metrics}, random.Random(202610262300)
    for _ in range(replicates):
        indices = [rng.randrange(8) for _ in range(8)]
        for key, values in log_metrics.items():
            draws[key].append(math.exp(sum(values[i] for i in indices) / 8))
    metrics = {}
    for key, values in log_metrics.items():
        metrics[key] = {
            "processes": 8,
            "geometric_mean_ratio": math.exp(statistics.mean(values)),
            "min_ratio": math.exp(min(values)),
            "max_ratio": math.exp(max(values)),
            "below_one": sum(v < 0 for v in values),
            "above_one": sum(v > 0 for v in values),
            "equal_one": sum(v == 0 for v in values),
            "exploratory_process_bootstrap_95pct": [
                quantile(draws[key], 0.025),
                quantile(draws[key], 0.975),
            ],
        }
    strata = {}
    for order in ("winner-first", "default-first"):
        rows = [p for p in processes if p["capture_order"] == order]
        require(
            len(rows) == 4, "four independently scheduled capture orders per stratum"
        )
        strata[order] = {
            "blocks": [r["block"] for r in rows],
            "geometric_mean_hypothetical_over_original": math.exp(
                statistics.mean(r["log_hypothetical_over_original"] for r in rows)
            ),
            "geometric_mean_finalist_graph_over_default": math.exp(
                statistics.mean(r["graph_log_ratio"] for r in rows)
            ),
            "outcomes": dict(collections.Counter(r["outcome"] for r in rows)),
            "transitions": dict(collections.Counter(r["transition"] for r in rows)),
        }
    return {
        "metrics": metrics,
        "outcomes": dict(collections.Counter(p["outcome"] for p in processes)),
        "transitions": dict(collections.Counter(p["transition"] for p in processes)),
        "capture_order_strata": strata,
        "process_rows": processes,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": 202610262300,
        "inference_unit": "Fresh process n=8; same GPU and fixed workload; pair counts do not increase n.",
        "limits": [
            "Both choices use common post-confirmation graphs, not untouched versus modified startup.",
            "Capture-order associations remain possible; near-zero measured better/worse counts are descriptive, not proven improvements.",
            "All eight outcomes retained, including unchanged and default-winning processes; historical/admission results excluded.",
            "No global margin, production selection change, universal policy claim or serving-performance inference.",
        ],
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
        "plan binds schedule and search recipe",
    )
    prior, leaf = helpers(args.prior_analyzer)
    validate_schedule(schedule, prior)
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
            validate_worker(
                load(path), entry, contract, schedule, args, prior, leaf, path
            )
        )
        containers.append(path.with_suffix(".container-id").read_text().strip())
    if not args.admission:
        containers.append(
            (args.raw_root / "admission-auto_synthetic.container-id")
            .read_text()
            .strip()
        )
    prior.validate_container_ids(containers)
    result = {
        "status": "passed",
        "mode": "admission" if args.admission else "timing",
        "plan_sha256": sha256(args.plan),
        "schedule_sha256": sha256(args.schedule),
        "historical_contract_sha256": sha256(args.historical_contract),
        "worker_sha256": sha256(args.worker),
        "analyzer_sha256": sha256(__file__),
        "prior_analyzer_sha256": PRIOR_ANALYZER_SHA,
        "leaf_helpers_sha256": prior.LEAF_SHA,
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
            "qualified admission bootstrap binds exact result and source",
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
        "--prior-analyzer",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "20261026T220000Z-h100-fixed-pair"
        / "analyzer.py",
    )
    parser.add_argument("--admission", action="store_true")
    parser.add_argument("--admission-bootstrap", type=Path)
    args = parser.parse_args()
    try:
        result = analyze(args)
    except Exception as error:
        args.output.write_text(
            json.dumps(
                {"status": "failed", "error": f"{type(error).__name__}: {error}"},
                indent=2,
            )
            + "\n"
        )
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
