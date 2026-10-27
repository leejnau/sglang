#!/usr/bin/env python3
"""MLP confirmation analysis using frozen decision, cache, and timing validators.

Only fixture geometry and original initializer validation differ from the prior
confirmation analysis. Both choices share the same post-confirmation graphs.
"""

import argparse
import collections
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT_SHA = "0369f0d952fc9f519c6a0d5511ab280afbe6065224b6a0d34db3f7fab0b74d98"
PARENT_WORKER_SHA = "3ec7d31ea510b29b9e5e2c289e82a3cfe31867b11ba4a28d9d900d76c1fa6413"
FIXTURE_SHA = "0e456c2d371906be235c5fb405fc20c0a969f672ee26722264ae6761b1838809"
MANIFEST_SHA = "e5e39380e83c45c9f4678b0c9b76aa04c1bae6523d434a275f7fc8955c5bd518"
SHAPE = [144, 1024, 3072]


def load_parent():
    import hashlib

    path = HERE.parent / "20261026T230000Z-h100-finalist-confirmation" / "analyzer.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError("frozen confirmation validator changed")
    spec = importlib.util.spec_from_file_location("mlp_parent_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parent = load_parent()
require, load, sha256 = parent.require, parent.load, parent.sha256
positive, close, digest = parent.positive, parent.close, parent.digest
expected_decision, choice_effect = parent.expected_decision, parent.choice_effect
validate_confirmation = parent.validate_confirmation
validate_schedule = parent.validate_schedule
ROLES, ENDPOINTS = parent.ROLES, parent.ENDPOINTS
parent_helpers, parent_identity = parent.helpers, parent.validate_identity
parent_summary = parent.summarize


def helpers(path):
    prior, leaf = parent_helpers(path)
    prior.validate_graphs = validate_graphs
    return prior, leaf


def validate_identity(data, assignment, contract, args):
    parent_identity(data, assignment, contract, args)
    require(
        data["mlp_adapter"]
        == {
            "parent_worker_sha256": PARENT_WORKER_SHA,
            "fixture_kind": "qwen3_mlp_down_proj",
            "shape": SHAPE,
            "initializer_recipe": "original_synthetic_no_clone_copy",
            "fixture_sha256": FIXTURE_SHA,
            "fixture_manifest_sha256": MANIFEST_SHA,
            "original_initializer_returns_same_object": True,
        },
        "actual MLP fixture, inherited worker, and unmodified synthetic draw",
    )
    fixture = data["fixture"]
    require(
        fixture["shape"] == SHAPE
        and fixture["tensor_file_sha256"] == FIXTURE_SHA
        and fixture["manifest_sha256"] == MANIFEST_SHA,
        "fixed captured MLP fixture",
    )
    require(
        "python/sglang/srt/models/qwen2.py" in data["source"]["sglang_source_sha256"],
        "MLP implementation source included",
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
                [1, m, 3072],
                "torch.float8_e4m3fn",
                "profile activation",
            )
            leaf.validate_tensor(
                row["metadata"][4], [1, m, 1024], "torch.bfloat16", "profile output"
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
        and [r["shape"] for r in calls] == [[1, m, 3072] for m in ENDPOINTS],
        "nine original activation draws",
    )
    for index, row in enumerate(calls):
        require(
            row["targeted"] is (index == 8)
            and row["cloned"] is False
            and row["copied"] is False
            and row["returned_same_object"] is True
            and ("generated_sha256" in row) is (data["mode"] == "admission"),
            "original synthetic draw without clone, copy, or timing-startup hash",
        )
        require(
            ("returned_sha256" in row) is (data["mode"] == "admission"),
            "admission-only returned input byte observation",
        )
        if data["mode"] == "admission":
            require(
                row["generated_sha256"] == row["returned_sha256"],
                "original generated bytes returned unchanged",
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
            graph["tactic"] == tactic,
            "observed explicit finalist irrespective of natural winner",
        )
        require(
            graph["lookup_tier"] == ("memory" if role == "winner" else "miss"),
            "real finalist hit/default miss",
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
            inputs[0], [1, 144, 3072], "torch.float8_e4m3fn", "capture activation"
        )
        leaf.validate_tensor(
            inputs[1], [1, 3072, 1024], "torch.float8_e4m3fn", "capture weight"
        )
        leaf.validate_tensor(
            graph["output"], [144, 1024], "torch.bfloat16", "layer output"
        )
        require(
            inputs[0]["stride"][-2:] == [3072, 1]
            and inputs[1]["stride"][-2:] == [1, 3072],
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


def slow_choice_counts(processes):
    """Directional counts against the measured faster of exactly two graphs."""
    counts = collections.Counter()
    for row in processes:
        log_ratio = row["graph_log_ratio"]
        require(math.isfinite(log_ratio), "finite held-out graph estimate")
        old = row["original_log_regret_two_candidate_only"] > 0
        new = row["confirmed_log_regret_two_candidate_only"] > 0
        counts["original_slow"] += old
        counts["confirmed_slow"] += new
        counts["corrected"] += old and not new
        counts["introduced"] += not old and new
        counts["remaining_slow"] += old and new
        counts["neither_slow"] += not old and not new
        counts["graph_point_ties"] += log_ratio == 0
    require(
        sum(
            counts[k]
            for k in ("corrected", "introduced", "remaining_slow", "neither_slow")
        )
        == len(processes),
        "all outcomes counted exactly once",
    )
    return dict(counts)


def disposition(counts, metric):
    upper = metric["exploratory_process_bootstrap_95pct"][1]
    if counts["corrected"] > 0 and counts["introduced"] == 0 and upper < 1:
        result = "supports_focused_review"
    elif counts["introduced"]:
        result = "introduced_slow_choices_stop"
    elif counts["corrected"]:
        result = "promising_but_inconclusive_stop"
    else:
        result = "neutral_or_ineffective_stop"
    return {
        "result": result,
        "fixed_cohort_complete": True,
        "additional_processes": 0,
        "automatic_production_change_or_pr": False,
        "rule": "Review only if corrected>0, introduced=0, and whole-process ratio CI upper<1; otherwise stop this bounded checkpoint.",
        "counts_are_descriptive": True,
    }


def summarize(processes, replicates=100000):
    for row in processes:
        expected = choice_effect(
            row["original_role"], row["confirmed_role"], row["graph_log_ratio"]
        )
        require(
            all(row[k] == v for k, v in expected.items()),
            "choice effects independently match frozen roles and replay score",
        )
    result = parent_summary(processes, replicates)
    counts = slow_choice_counts(processes)
    result["slow_choice_counts"] = counts
    result["checkpoint_disposition"] = disposition(
        counts, result["metrics"]["hypothetical_over_original"]
    )
    result["limits"].append(
        "Slow-choice counts use the measured faster of two graphs without a margin; they are descriptive, not per-process significance or a full candidate oracle."
    )
    return result


def analyze(args):
    protocol_path = HERE / "analysis-protocol.json"
    plan = load(args.plan)
    require(
        plan["analysis_protocol_sha256"] == sha256(protocol_path),
        "frozen analysis and stopping rule",
    )
    protocol = load(protocol_path)
    require(
        protocol["fresh_timing_processes"] == 8
        and protocol["admission_processes"] == 1,
        "fixed cohort bounds",
    )
    parent.helpers = helpers
    parent.validate_identity = validate_identity
    parent.validate_startup = validate_startup
    parent.summarize = summarize
    result = parent.analyze(args)
    result.update(
        analyzer_sha256=sha256(__file__),
        parent_confirmation_analyzer_sha256=PARENT_SHA,
        analysis_protocol_sha256=sha256(protocol_path),
    )
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
        default=HERE.parent / "20261026T220000Z-h100-fixed-pair" / "analyzer.py",
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
