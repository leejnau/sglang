#!/usr/bin/env python3
"""Validate the bounded scoped-cache lifecycle; retain negative findings."""

import argparse
import importlib.util
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE.parent / "20261027T030000Z-h100-replay-coverage/analyze.py"
SPEC = importlib.util.spec_from_file_location("coverage_analysis", BASE_PATH)
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)
require, read, write = base.require, base.read, base.write
WORKER = "test/manual/kernels/audit_flashinfer_scoped_cache.py"
PARENT_WORKER = "test/manual/kernels/audit_flashinfer_replay_coverage.py"
STUDY = "actual_model_scoped_graph_cache"
POLICY = {"measure_execution_mode": "cuda_graph"}
CONTROL_KEYS = ("skip_ops", "buckets", "round_up", "profile_replays")


def finding(findings, condition, kind, **details):
    if not condition:
        findings.append({"kind": kind, **details})


def report_summary(report, identity, findings, label, expected=128):
    """Validate record accounting without equating default tactics with misses."""
    for key in ("hits", "misses", "skipped", "dropped_records", "max_records"):
        base.nonnegative_int(report.get(key), label + ": " + key)
    rows = report.get("records")
    require(isinstance(rows, list), label + ": missing records")
    total = sum(report[k] for k in ("hits", "misses", "skipped"))
    require(total == len(rows) + report["dropped_records"], label + ": count mismatch")
    require(len(rows) <= report["max_records"], label + ": record bound exceeded")
    counts = Counter()
    sources = Counter()
    for row in rows:
        status, source = row.get("status"), row.get("source")
        require(status in ("hit", "miss", "skipped"), label + ": invalid status")
        require(
            source in ("memory", "legacy", "managed", "bundled", "fallback"),
            label + ": invalid source",
        )
        require(isinstance(row.get("op"), str) and row["op"], label + ": missing op")
        require(isinstance(row.get("input_shapes"), list), label + ": missing shape")
        require(isinstance(row.get("runner"), str), label + ": missing runner")
        require(isinstance(row.get("selected_tactic"), str), label + ": missing tactic")
        require(
            not (status == "hit" and source == "fallback"),
            label + ": fallback mislabeled hit",
        )
        require(
            status != "miss" or source == "fallback",
            label + ": miss lost fallback source",
        )
        counts[status] += 1
        sources[source] += 1
        finding(
            findings,
            row.get("cache_identity") == identity,
            "lookup_identity_mismatch",
            stage=label,
        )
        if status == "hit":
            allowed = (
                ("memory", "legacy") if identity is None else ("memory", "managed")
            )
            finding(
                findings,
                source in allowed,
                "lookup_used_unqualified_source",
                stage=label,
                source=source,
            )
            if identity is not None:
                finding(
                    findings,
                    row.get("requested_policy") is not None
                    and row.get("requested_policy") == row.get("stored_policy"),
                    "strict_profiling_provenance_mismatch",
                    stage=label,
                )
    for status, key in (("hit", "hits"), ("miss", "misses"), ("skipped", "skipped")):
        require(counts[status] <= report[key], label + ": status count exceeds total")
        require(
            report["dropped_records"] or counts[status] == report[key],
            label + ": status count mismatch",
        )
    fp8 = sum(row["op"] == "fp8_gemm" for row in rows)
    finding(
        findings,
        fp8 == expected,
        "unexpected_fp8_lookup_count",
        stage=label,
        expected=expected,
        actual=fp8,
    )
    return {
        **{
            key: report[key] for key in ("hits", "misses", "skipped", "dropped_records")
        },
        "lookups": total,
        "fp8_lookups": fp8,
        "sources": dict(sources),
        "cached_default_hits": sum(
            r["status"] == "hit" and r["selected_tactic"] == "-1" for r in rows
        ),
        "default_fallback_misses": sum(
            r["status"] == "miss" and r["selected_tactic"] == "-1" for r in rows
        ),
    }


def decisions(report):
    return [
        (r["op"], r["input_shapes"], r["runner"], r["selected_tactic"])
        for r in report["records"]
    ]


def complete_hits(summary, findings, label):
    finding(
        findings,
        summary["hits"] == 128
        and not any(summary[k] for k in ("misses", "skipped", "dropped_records")),
        "incomplete_hit_coverage",
        stage=label,
        counts=summary,
    )


def scope_state(state, reference, mode, findings, label):
    required = (
        *base.STATE_KEYS,
        "in_v2_context",
        "cache_scope",
        "strict_replay",
        "isolate_legacy",
        "store_root",
        "store_env_hash",
        "policy_fields",
    )
    require(
        all(key in state for key in required), label + ": incomplete scope snapshot"
    )
    identity = state["winner_identity"]
    require(
        isinstance(identity, list)
        and len(identity) == 2
        and all(isinstance(v, str) and v for v in identity),
        label + ": malformed managed identity",
    )
    finding(
        findings,
        identity == [state["store_root"], state["store_env_hash"]],
        "store_identity_mismatch",
        stage=label,
    )
    finding(
        findings,
        state["managed_store"] is True
        and state["ambient_store"] is False
        and state["in_v2_context"] is True
        and state["cache_scope"] == "context"
        and state["isolate_legacy"] is True,
        "scope_ownership_failure",
        stage=label,
    )
    finding(
        findings,
        state["policy_fields"] == POLICY,
        "unexpected_measurement_policy",
        stage=label,
    )
    finding(
        findings,
        state["is_tuning"] is (mode == "tune")
        and state["strict_replay"] is (mode == "replay"),
        "unexpected_scope_mode",
        stage=label,
    )
    finding(
        findings,
        all(state[key] == reference[key] for key in CONTROL_KEYS),
        "inherited_controls_changed",
        stage=label,
    )
    return identity


def profile_delta(row, findings, label, expected_zero=True):
    before = base.nonnegative_int(
        row.get("profile_count_before"), label + ": before profiles"
    )
    after = base.nonnegative_int(
        row.get("profile_count_after"), label + ": after profiles"
    )
    require(after >= before, label + ": profiling counter moved backwards")
    if expected_zero:
        finding(
            findings,
            after == before,
            "unexpected_profiling",
            stage=label,
            count=after - before,
        )
    return after - before


def managed_manifest(files, identity, findings):
    path = f"v2/{identity[1]}/manifest.json"
    require(path in files, "missing managed identity manifest")
    manifest = files[path]["contents"]
    require(
        base.sha(base.canonical(manifest).encode())[:16] == identity[1],
        "managed manifest hash does not match identity",
    )
    finding(
        findings,
        all(manifest.get(k) == v for k, v in POLICY.items()),
        "managed_manifest_policy_mismatch",
    )
    finding(
        findings,
        any(name.startswith(f"v2/{identity[1]}/entries/") for name in files),
        "managed_store_has_no_winners",
    )


def validate_worker(worker, phase):
    require(
        worker.get("schema") == 1 and worker.get("study") == STUDY,
        "wrong worker schema",
    )
    require(
        worker.get("status") == "completed",
        "worker incomplete; retain original error evidence",
    )
    require(
        worker.get("phase") == phase
        and worker.get("model_id") == base.MODEL
        and worker.get("model_revision") == base.REVISION,
        "wrong model or phase",
    )
    require(
        type(worker.get("started_ns")) is int
        and worker["finished_ns"] > worker["started_ns"],
        "invalid chronology",
    )
    findings = []
    startup = worker["startup"]
    lifecycle = worker["scoped_lifecycle"]
    reference = lifecycle["legacy_reference_state"]
    base.legacy_state(reference)
    require(
        reference["in_v2_context"] is False and reference["policy_fields"] == {},
        "legacy reference was already scoped",
    )
    legacy = lifecycle["legacy_reference_cache"]
    before_files = base.file_inventory(worker["cache_before_startup"])
    managed_before = base.file_inventory(worker["managed_before_startup"])
    require(
        bool(before_files) is (phase == "warm")
        and bool(managed_before) is (phase == "warm"),
        "incorrect cold/warm cache setup",
    )
    startup_files = base.file_inventory(startup["files"])
    require(startup_files, "no legacy persistence evidence")
    eligible = startup.get("eligible_linears")
    require(
        isinstance(eligible, list)
        and len(eligible) == startup.get("eligible_linear_layers")
        and len({r["name"] for r in eligible}) == len(eligible),
        "invalid eligible linear inventory",
    )
    finding(
        findings,
        len(eligible) == 128,
        "unexpected_eligible_linear_count",
        actual=len(eligible),
    )
    profiles = worker["profile_calls"]
    require(
        isinstance(profiles, list)
        and all(isinstance(r.get("stage"), str) for r in profiles),
        "invalid profile inventory",
    )
    legacy_profiles = base.nonnegative_int(
        lifecycle["legacy_startup_profile_count"], "legacy profiles"
    )
    require(
        legacy_profiles <= startup["profile_count"] <= len(profiles),
        "invalid startup profile counts",
    )
    require(
        sum(r["stage"] == "startup" for r in profiles) == legacy_profiles,
        "legacy profile count does not match calls",
    )
    finding(
        findings,
        legacy_profiles > 0 if phase == "cold" else legacy_profiles == 0,
        "unexpected_legacy_profile_count",
        count=legacy_profiles,
    )
    protocols = worker["protocol_calls"]
    require(
        isinstance(protocols, list)
        and all(type(r.get("observed")) is bool for r in protocols),
        "invalid protocol inventory",
    )
    finding(
        findings,
        not any(r["observed"] for r in protocols),
        "unqualified_protocol_consumer",
    )
    coverage = []
    for label in ("legacy_before", "legacy_after"):
        summary = report_summary(lifecycle[label], None, findings, label)
        complete_hits(summary, findings, label)
        coverage.append({"stage": label, **summary})
    finding(
        findings,
        decisions(lifecycle["legacy_before"]) == decisions(lifecycle["legacy_after"]),
        "legacy_selection_changed",
    )
    finding(
        findings,
        lifecycle["legacy_exit_state"] == reference
        and lifecycle["legacy_restored"] is True
        and lifecycle["legacy_exit_cache"] == legacy,
        "legacy_restoration_failed",
    )
    finding(
        findings,
        legacy["files"] == startup["files"] == worker["cache_after_numerics"],
        "legacy_file_bytes_changed",
    )
    finding(
        findings,
        legacy["inventory"] == startup["cache"] == worker["inventory_after_numerics"],
        "legacy_live_winners_changed",
    )
    finding(
        findings,
        all(
            startup["state"][k] == worker["final_state"][k] == reference[k]
            for k in base.STATE_KEYS
        ),
        "bare_serving_controls_changed",
    )

    negative = lifecycle["negative"]
    negative_identity = scope_state(
        negative["state_inside"], reference, "replay", findings, "negative"
    )
    negative_summary = report_summary(
        negative["coverage"], negative_identity, findings, "negative", expected=1
    )
    error = negative.get("error", {})
    finding(
        findings,
        error.get("type") == "RuntimeError"
        and "autotune strict replay miss" in error.get("message", ""),
        "strict_negative_did_not_raise_expected_error",
    )
    finding(
        findings,
        negative_summary["misses"] == 1
        and negative_summary["lookups"] == 1
        and negative_summary["default_fallback_misses"] == 1,
        "strict_negative_did_not_expose_one_miss",
    )
    profile_delta(negative, findings, "negative")
    finding(
        findings,
        negative["files_before"]
        == negative["files_after"]
        == worker["negative_after_numerics"]
        == {},
        "strict_negative_published_or_reused_files",
    )
    finding(
        findings,
        negative["state_after"] == reference and negative["legacy_after"] == legacy,
        "strict_negative_failed_restoration",
    )

    tune = lifecycle["tune"]
    identity = scope_state(tune["state_inside"], reference, "tune", findings, "tune")
    finding(findings, identity[0] != negative_identity[0], "negative_root_not_isolated")
    tune_profiles = profile_delta(tune, findings, "tune", expected_zero=False)
    require(
        sum(r["stage"] == "scoped_tune" for r in profiles) == tune_profiles,
        "graph profile count does not match calls",
    )
    finding(
        findings,
        tune_profiles > 0 if phase == "cold" else tune_profiles == 0,
        "unexpected_graph_profile_count",
        count=tune_profiles,
    )
    finding(findings, tune["state_after"] == reference, "tune_failed_restoration")
    require(base.file_inventory(tune["files_after"]), "no managed persistence evidence")
    managed_manifest(tune["files_after"], identity, findings)
    finding(
        findings,
        tune["files_before"] == worker["managed_before_startup"],
        "managed_files_changed_before_tune",
    )
    if phase == "warm":
        finding(
            findings,
            tune["files_after"] == tune["files_before"],
            "warm_graph_files_rewritten",
        )
    strict = lifecycle["strict"]
    strict_identity = scope_state(
        strict["state_inside"], reference, "replay", findings, "strict"
    )
    finding(findings, strict_identity == identity, "tune_capture_identity_changed")
    finding(
        findings,
        strict["state_before"] == strict["state_after"] == reference,
        "capture_failed_restoration",
    )
    finding(
        findings,
        strict["files_before"]
        == strict["files_after"]
        == tune["files_after"]
        == worker["managed_after_numerics"],
        "managed_files_changed_after_tune",
    )
    profile_delta(strict, findings, "strict")
    rows = strict["records"]
    require(
        isinstance(rows, list) and len(rows) == 3,
        "expected two existing warmups and one capture",
    )
    require(
        [r.get("capturing") for r in rows] == [False, False, True]
        and [r.get("stage") for r in rows]
        == ["scoped_warmup", "scoped_warmup", "scoped_capture"],
        "incorrect capture boundaries",
    )
    require(
        [r["stage"] for r in worker["capture_records"]]
        == [
            "legacy_before",
            "strict_negative",
            "scoped_warmup",
            "scoped_warmup",
            "scoped_capture",
            "legacy_after",
        ],
        "unexpected prepared-forward sequence",
    )
    require(
        worker["capture_records"][2:5] == rows, "aggregate capture records disagree"
    )
    for index, key in ((0, "legacy_before"), (1, "negative"), (5, "legacy_after")):
        row = worker["capture_records"][index]
        expected_report = (
            lifecycle[key]["coverage"] if key == "negative" else lifecycle[key]
        )
        require(
            row["coverage"] == expected_report,
            "prepared report disagrees with lifecycle",
        )
        require(row["capturing"] is False, "extra forward entered CUDA capture")
        profile_delta(row, findings, row["stage"])
        finding(
            findings,
            row["state_before"] == row["state_after"],
            "prepared_forward_changed_controls",
            stage=row["stage"],
        )
        if key != "negative":
            finding(
                findings,
                all(
                    row["state_before"][field] == reference[field]
                    for field in base.STATE_KEYS
                ),
                "legacy_forward_changed_effective_controls",
                stage=row["stage"],
            )
    aggregate = strict["coverage"]
    require(
        aggregate["records"]
        == [item for row in rows for item in row["coverage"]["records"]],
        "aggregate strict decisions disagree",
    )
    for field in ("hits", "misses", "skipped", "dropped_records"):
        require(
            aggregate[field] == sum(row["coverage"][field] for row in rows),
            "aggregate strict counts disagree",
        )
    for index, row in enumerate(rows):
        label = f"{row['stage']}_{index}"
        finding(
            findings,
            row["state_before"] == row["state_after"] == strict["state_inside"],
            "forward_changed_scope",
            stage=label,
        )
        profile_delta(row, findings, label)
        summary = report_summary(row["coverage"], identity, findings, label)
        complete_hits(summary, findings, label)
        finding(
            findings,
            decisions(row["coverage"]) == decisions(rows[0]["coverage"]),
            "capture_winner_changed",
            stage=label,
        )
        coverage.append({"stage": label, **summary})
    require(
        startup["profile_count"] == legacy_profiles + tune_profiles,
        "startup counter includes unaccounted profiling",
    )
    finding(
        findings,
        all(r["stage"] in ("startup", "scoped_tune") for r in profiles),
        "serving_profiled",
    )
    finding(
        findings,
        len(profiles) == startup["profile_count"],
        "post_startup_profiled",
        count=len(profiles) - startup["profile_count"],
    )
    if phase == "warm":
        finding(findings, not profiles, "fresh_warm_profiled", count=len(profiles))
    persistence = worker["persistence_calls"]
    require(isinstance(persistence, list), "missing legacy persistence calls")
    loads = [call for call in persistence if call["operation"] == "load_configs"]
    if phase == "warm":
        finding(findings, bool(loads), "warm_legacy_cache_not_loaded")
    for call in loads:
        require(
            call["path"] in before_files
            and call["before_sha256"] == before_files[call["path"]],
            "load did not read supplied legacy file",
        )
        expected = {
            key: value[:2]
            for key, value in worker["cache_before_startup"][call["path"]][
                "contents"
            ].items()
            if not key.startswith("_")
        }
        finding(
            findings,
            all(
                call["cache_after"]["loaded"].get(key) == value
                for key, value in expected.items()
            ),
            "persisted_legacy_winners_not_loaded",
        )
    findings.extend(base.validate_numerics(worker["numerics"]))
    numeric_reports = worker["numerical_reports"]
    require(
        isinstance(numeric_reports, list)
        and [r.get("arm") for r in numeric_reports] == ["eager", "graph"] * 3,
        "missing numerical selection reports",
    )
    for index, row in enumerate(numeric_reports):
        label = f"numerical_{index}_{row['arm']}"
        finding(
            findings,
            row["state_before"] == row["state_after"] == reference,
            "numerical_scope_not_restored",
            stage=label,
        )
        finding(
            findings,
            row["cache_before"] == row["cache_after"] == legacy,
            "numerical_legacy_cache_changed",
            stage=label,
        )
        finding(
            findings,
            row["profile_delta"] == 0,
            "numerical_selection_profiled",
            stage=label,
        )
        require(
            row["replay_delta"] == (1 if row["arm"] == "graph" else 0),
            "numerical selection report disagrees with route",
        )
        summary = report_summary(
            row["coverage"],
            None,
            findings,
            label,
            expected=128 if row["arm"] == "eager" else 0,
        )
        if row["arm"] == "eager":
            complete_hits(summary, findings, label)
            finding(
                findings,
                decisions(row["coverage"]) == decisions(lifecycle["legacy_after"]),
                "numerical_legacy_selection_changed",
                stage=label,
            )
        else:
            finding(
                findings,
                summary["lookups"] == 0,
                "graph_replay_performed_python_lookup",
                stage=label,
            )
        coverage.append({"stage": label, **summary})
    for row in worker["numerics"]["checks"]:
        for arm, item in row["arms"].items():
            finding(
                findings,
                item["profile_delta"] == 0,
                "numerical_execution_profiled",
                case=row["case"],
                arm=arm,
            )
    controls = base.validate_controls(worker["isolated_control"])
    return {
        "phase": phase,
        "legacy_profile_count": legacy_profiles,
        "graph_profile_count": tune_profiles,
        "post_startup_profile_count": len(profiles) - startup["profile_count"],
        "eligible_linear_layers": len(eligible),
        "managed_identity": identity,
        "cache_before_startup": before_files,
        "cache_after_startup": startup_files,
        "managed_before_startup": managed_before,
        "managed_after_numerics": base.file_inventory(worker["managed_after_numerics"]),
        "coverage": coverage,
        "strict_negative": negative_summary,
        "report_controls": controls,
        "findings": findings,
    }


def validate_cross_process(cold, warm):
    require(
        cold["managed_after_numerics"] == warm["managed_before_startup"],
        "fresh warm did not receive exact cold managed bytes",
    )
    left, right = cold["scoped_lifecycle"], warm["scoped_lifecycle"]
    require(
        left["tune"]["state_inside"]["winner_identity"]
        == right["tune"]["state_inside"]["winner_identity"],
        "fresh warm changed managed identity",
    )
    findings = []
    for label in ("legacy_before", "legacy_after"):
        finding(
            findings,
            decisions(left[label]) == decisions(right[label]),
            "fresh_warm_legacy_selection_changed",
            stage=label,
        )
    for index, (a, b) in enumerate(
        zip(left["strict"]["records"], right["strict"]["records"])
    ):
        finding(
            findings,
            decisions(a["coverage"]) == decisions(b["coverage"]),
            "fresh_warm_graph_selection_changed",
            invocation=index,
        )
    return findings


def analyze(args):
    require(
        args.phases in (["cold"], ["cold", "warm"]), "expected cold or cold then warm"
    )
    source, commit = args.source_root.resolve(), args.expected_source_commit
    base_relative = str(BASE_PATH.relative_to(source))
    require(
        base.sha(BASE_PATH.read_bytes())
        == base.sha(base.git_bytes(source, commit, base_relative)),
        "historical analyzer differs from frozen source",
    )
    provenance = read(args.raw_root / "provenance.json")
    for path, digest in provenance["harness_source_sha256"].items():
        require(
            digest == base.sha(base.git_bytes(source, commit, path)),
            "harness dependency differs from frozen commit: " + path,
        )
    require(
        {WORKER, PARENT_WORKER} <= provenance["harness_source_sha256"].keys(),
        "missing harness source bindings",
    )
    for phase in args.phases:
        worker = read(args.raw_root / f"{phase}.json")
        require(
            worker["parent_worker_sha256"]
            == provenance["harness_source_sha256"][PARENT_WORKER],
            "wrong parent worker source",
        )
        require(
            worker["scoped_lifecycle"]["tune"]["state_inside"]["store_root"]
            == provenance["persistent_cache_roots"]["managed"],
            "managed root differs from frozen provenance",
        )
    old_worker, old_validate = base.WORKER, base.validate_worker
    try:
        base.WORKER, base.validate_worker = WORKER, validate_worker
        result = base.analyze(args)
    finally:
        base.WORKER, base.validate_worker = old_worker, old_validate
    if args.phases == ["cold", "warm"]:
        cold, warm = (read(args.raw_root / f"{phase}.json") for phase in args.phases)
        result["findings"].extend(
            {"phase": "warm", **item} for item in validate_cross_process(cold, warm)
        )
    result.update(
        study=STUDY,
        qualification_complete=args.phases == ["cold", "warm"]
        and not result["findings"],
        phase_requirements_met=not result["findings"],
        interpretation="One model's managed graph-cache lifecycle only. Valid negative outcomes are retained; no performance or general policy claim follows.",
    )
    result.pop("coverage_complete")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--plan", type=Path, default=HERE / "plan.json")
    parser.add_argument(
        "--phases", nargs="+", choices=("cold", "warm"), default=["cold", "warm"]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = analyze(args)
    except Exception as error:
        write(
            args.output,
            {
                "status": "invalid_or_incomplete_evidence",
                "error": {"type": type(error).__name__, "message": str(error)},
            },
        )
        raise
    write(args.output, result)


if __name__ == "__main__":
    main()
