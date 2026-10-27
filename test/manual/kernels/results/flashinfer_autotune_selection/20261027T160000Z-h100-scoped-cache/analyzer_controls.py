"""CPU controls for rejecting malformed evidence and retaining genuine gaps."""

import argparse
import copy
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "scoped_analysis", Path(__file__).with_name("analyze.py")
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
MANIFEST = {"cache_schema": "v2", **audit.POLICY}
ENV_HASH = audit.base.sha(audit.base.canonical(MANIFEST).encode())[:16]


def report(identity=None, count=128, miss=False):
    rows = [
        {
            "op": "fp8_gemm",
            "input_shapes": [[1, 1, 4096]],
            "cache_identity": identity,
            "status": "miss" if miss else "hit",
            "source": "fallback" if miss else "memory",
            "runner": "Runner",
            "selected_tactic": "-1" if miss else "7",
            "requested_policy": [
                "cuda_graph_profile_replays",
                None,
                "l2_cache_policy",
                "hot",
            ],
            "stored_policy": None
            if miss
            else ["cuda_graph_profile_replays", None, "l2_cache_policy", "hot"],
            "reasons": ["cache_miss"] if miss else [],
        }
        for _ in range(count)
    ]
    return {
        "hits": 0 if miss else count,
        "misses": count if miss else 0,
        "skipped": 0,
        "dropped_records": 0,
        "max_records": 8192,
        "records": rows,
    }


def state(mode=None, root="/managed"):
    scoped = mode is not None
    return {
        "is_tuning": mode == "tune",
        "active_tuning_contexts": int(mode == "tune"),
        "managed_store": scoped,
        "ambient_store": False,
        "winner_identity": [root, ENV_HASH] if scoped else None,
        "measurement_policy": "graph" if scoped else "None",
        "skip_ops": [],
        "buckets": None,
        "round_up": False,
        "profile_replays": None,
        "in_v2_context": scoped,
        "cache_scope": "context" if scoped else "process",
        "strict_replay": mode == "replay",
        "isolate_legacy": scoped,
        "store_root": root if scoped else None,
        "store_env_hash": ENV_HASH if scoped else None,
        "policy_fields": audit.POLICY if scoped else {},
    }


def numeric_fixture():
    rows = []
    for case, token in (("original", 2), ("changed", 3), ("restored", 2)):
        arms = {}
        for arm in ("eager", "graph"):
            arms[arm] = {
                "finite": True,
                "replay_delta": int(arm == "graph"),
                "profile_delta": 0,
                "poisoned_graph_output": arm == "graph",
                "graph_output_owner_unchanged": True,
                "original_graph_unchanged": True,
                "output_sha256": "a" * 64,
                "output_shape": [1, 128],
                "prompt_input_ids": [4, 5],
                "decode_token": token,
                "prefix_seq_lens": [2],
                "graph_input_ids": [token] if arm == "graph" else None,
            }
        rows.append(
            {
                "case": case,
                "decode_token": token,
                "prompt_input_ids": [4, 5],
                "allclose": True,
                "argmax_equal": True,
                "max_abs_error": 0.0,
                "atol": 0.05,
                "rtol": 0.01,
                "arms": arms,
            }
        )
    return {"checks": rows, "changed_output": True, "restored_output_allclose": True}


def worker_fixture(phase="cold"):
    bare, tuning, replay = state(), state("tune"), state("replay")
    identity = tuning["winner_identity"]
    files = {"cache.json": {"sha256": "a" * 64, "contents": {"_metadata": {}}}}
    managed = {
        f"v2/{ENV_HASH}/entries/entry.json": {
            "sha256": "b" * 64,
            "contents": {"tactic": 7},
        },
        f"v2/{ENV_HASH}/manifest.json": {"sha256": "c" * 64, "contents": MANIFEST},
    }
    inventory = {"memory": [], "loaded": {}}
    legacy = {
        "inventory": inventory,
        "files": files,
        "file_policies": [],
        "memory_policies": [],
        "excluded_file_keys": [],
        "invalidated_file_keys": [],
    }
    count = 1 if phase == "cold" else 0

    def row(label, selected, scope, capturing=False):
        return {
            "stage": label,
            "capturing": capturing,
            "coverage": selected,
            "state_before": scope,
            "state_after": scope,
            "profile_count_before": count * 2,
            "profile_count_after": count * 2,
        }

    before = report()
    after = copy.deepcopy(before)
    negative = {
        "state_inside": state("replay", "/negative"),
        "state_after": bare,
        "files_before": {},
        "files_after": {},
        "legacy_after": legacy,
        "profile_count_before": count,
        "profile_count_after": count,
        "error": {"type": "RuntimeError", "message": "autotune strict replay miss"},
        "coverage": report(["/negative", ENV_HASH], 1, True),
    }
    strict_rows = [
        row("scoped_warmup", report(identity), replay),
        row("scoped_warmup", report(identity), replay),
        row("scoped_capture", report(identity), replay, True),
    ]
    strict = {
        "state_before": bare,
        "state_inside": replay,
        "state_after": bare,
        "files_before": managed,
        "files_after": managed,
        "profile_count_before": count * 2,
        "profile_count_after": count * 2,
        "records": strict_rows,
        "coverage": report(identity, 384),
    }
    control_miss, control_hit = report(count=1, miss=True), report(count=1)
    control_hit["records"][0]["selected_tactic"] = "-1"
    numeric_reports = []
    for _ in range(3):
        for arm in ("eager", "graph"):
            numeric_reports.append(
                {
                    "arm": arm,
                    "coverage": report(count=128 if arm == "eager" else 0),
                    "replay_delta": int(arm == "graph"),
                    "profile_delta": 0,
                    "state_before": bare,
                    "state_after": bare,
                    "cache_before": legacy,
                    "cache_after": legacy,
                }
            )
    result = {
        "schema": 1,
        "study": audit.STUDY,
        "status": "completed",
        "phase": phase,
        "model_id": audit.base.MODEL,
        "model_revision": audit.base.REVISION,
        "started_ns": 1,
        "finished_ns": 2,
        "environment": {"pid": 1},
        "cache_before_startup": files if phase == "warm" else {},
        "managed_before_startup": managed if phase == "warm" else {},
        "cache_after_numerics": files,
        "managed_after_numerics": managed,
        "negative_after_numerics": {},
        "inventory_after_numerics": inventory,
        "final_state": bare,
        "graph_replays": 3,
        "startup": {
            "profile_count": count * 2,
            "cache": inventory,
            "state": bare,
            "files": files,
            "eligible_linear_layers": 128,
            "eligible_linears": [{"name": f"layer{i}"} for i in range(128)],
        },
        "profile_calls": [{"stage": "startup"}, {"stage": "scoped_tune"}]
        if count
        else [],
        "protocol_calls": [],
        "persistence_calls": [
            {
                "operation": "load_configs",
                "path": "cache.json",
                "before_sha256": "a" * 64,
                "after_sha256": "a" * 64,
                "cache_after": inventory,
            }
        ]
        if phase == "warm"
        else [],
        "isolated_control": {
            "miss": control_miss,
            "cached_default": control_hit,
            "exception_restored": True,
            "state_restored": True,
            "singleton_restored": True,
            "model_cache_unchanged": True,
            "profile_count": 0,
        },
        "numerics": numeric_fixture(),
        "numerical_reports": numeric_reports,
        "scoped_lifecycle": {
            "shape_key": {"bs": 1},
            "legacy_entry_state": bare,
            "legacy_entry_cache": legacy,
            "legacy_reference_state": bare,
            "legacy_reference_cache": legacy,
            "legacy_startup_profile_count": count,
            "legacy_before": before,
            "legacy_after": after,
            "legacy_exit_state": bare,
            "legacy_exit_cache": legacy,
            "legacy_restored": True,
            "negative": negative,
            "tune": {
                "state_inside": tuning,
                "state_after": bare,
                "files_before": managed if phase == "warm" else {},
                "files_after": managed,
                "profile_count_before": count,
                "profile_count_after": count * 2,
            },
            "strict": strict,
        },
        "capture_records": [
            row("legacy_before", before, bare),
            row("strict_negative", negative["coverage"], negative["state_inside"]),
            *strict_rows,
            row("legacy_after", after, bare),
        ],
    }
    return copy.deepcopy(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    passed = []

    def check(name, mutate=None, *, phase="cold", reject=False, expected_finding=None):
        worker = worker_fixture(phase)
        if mutate:
            mutate(worker)
        try:
            result = audit.validate_worker(worker, phase)
        except audit.base.EvidenceError:
            audit.require(reject, name + " unexpectedly rejected")
        else:
            audit.require(not reject, name + " accepted malformed evidence")
            if expected_finding:
                audit.require(
                    any(r["kind"] == expected_finding for r in result["findings"]),
                    name + " lost negative outcome",
                )
            else:
                audit.require(not result["findings"], name + " unexpected findings")
        passed.append({"name": name, "passed": True})

    check("complete_cold_phase")
    check("complete_fresh_warm_phase", phase="warm")
    check(
        "count_conservation",
        lambda w: w["scoped_lifecycle"]["legacy_before"].update(hits=127),
        reject=True,
    )
    check(
        "missing_capture",
        lambda w: w["scoped_lifecycle"]["strict"]["records"].pop(),
        reject=True,
    )
    check(
        "capture_flag",
        lambda w: w["scoped_lifecycle"]["strict"]["records"][2].update(capturing=False),
        reject=True,
    )
    check(
        "false_fallback_hit",
        lambda w: w["scoped_lifecycle"]["legacy_before"]["records"][0].update(
            source="fallback"
        ),
        reject=True,
    )
    check(
        "aggregate_disagreement",
        lambda w: w["scoped_lifecycle"]["strict"]["coverage"].update(hits=383),
        reject=True,
    )
    check(
        "missing_scope_snapshot",
        lambda w: w["scoped_lifecycle"]["tune"]["state_inside"].pop("cache_scope"),
        reject=True,
    )
    check(
        "policy_mismatch_is_finding",
        lambda w: w["scoped_lifecycle"]["tune"]["state_inside"].update(
            policy_fields={}
        ),
        expected_finding="unexpected_measurement_policy",
    )
    check(
        "negative_did_not_raise",
        lambda w: w["scoped_lifecycle"]["negative"].update(error={}),
        expected_finding="strict_negative_did_not_raise_expected_error",
    )
    check(
        "negative_profiled",
        lambda w: w["scoped_lifecycle"]["negative"].update(profile_count_after=2),
        expected_finding="unexpected_profiling",
    )
    check(
        "legacy_restoration_failed",
        lambda w: w["scoped_lifecycle"].update(legacy_restored=False),
        expected_finding="legacy_restoration_failed",
    )
    check(
        "managed_file_rewritten",
        lambda w: w["scoped_lifecycle"]["strict"].update(files_after={}),
        expected_finding="managed_files_changed_after_tune",
    )
    check(
        "numerical_difference_retained",
        lambda w: w["numerics"]["checks"][1].update(allclose=False),
        expected_finding="eager_graph_numerical_difference",
    )
    check(
        "unchanged_input_challenge",
        lambda w: w["numerics"].update(changed_output=False),
        expected_finding="changed_input_did_not_change_output",
    )
    check(
        "changed_graph_owner",
        lambda w: w["numerics"]["checks"][0]["arms"]["graph"].update(
            graph_output_owner_unchanged=False
        ),
        reject=True,
    )
    check(
        "missing_numerical_reports", lambda w: w["numerical_reports"].pop(), reject=True
    )
    check(
        "numerical_profiled",
        lambda w: w["numerical_reports"][0].update(profile_delta=1),
        expected_finding="numerical_selection_profiled",
    )
    check(
        "warm_missing_load",
        lambda w: w.update(persistence_calls=[]),
        phase="warm",
        expected_finding="warm_legacy_cache_not_loaded",
    )
    check(
        "wrong_warm_load_bytes",
        lambda w: w["persistence_calls"][0].update(before_sha256="f" * 64),
        phase="warm",
        reject=True,
    )
    check(
        "missing_target_op",
        lambda w: [
            r.update(op="different_op")
            for r in w["scoped_lifecycle"]["legacy_before"]["records"]
        ],
        expected_finding="unexpected_fp8_lookup_count",
    )
    cold, warm = worker_fixture(), worker_fixture("warm")
    audit.require(
        not audit.validate_cross_process(cold, warm), "unchanged pair rejected"
    )
    warm["scoped_lifecycle"]["strict"]["records"][0]["coverage"]["records"][0][
        "selected_tactic"
    ] = "8"
    audit.require(
        any(
            r["kind"] == "fresh_warm_graph_selection_changed"
            for r in audit.validate_cross_process(cold, warm)
        ),
        "changed warm graph winner lost",
    )
    passed.append({"name": "fresh_warm_graph_decisions_bound", "passed": True})
    warm = worker_fixture("warm")
    warm["scoped_lifecycle"]["legacy_after"]["records"][0]["selected_tactic"] = "8"
    audit.require(
        any(
            r["kind"] == "fresh_warm_legacy_selection_changed"
            for r in audit.validate_cross_process(cold, warm)
        ),
        "changed warm legacy winner lost",
    )
    passed.append({"name": "fresh_warm_legacy_decisions_bound", "passed": True})
    warm = worker_fixture("warm")
    warm["managed_before_startup"] = {}
    try:
        audit.validate_cross_process(cold, warm)
    except audit.base.EvidenceError:
        passed.append({"name": "fresh_warm_requires_exact_files", "passed": True})
    else:
        raise AssertionError("missing warm managed files accepted")
    default_hit = report(count=1)
    default_hit["records"][0]["selected_tactic"] = "-1"
    findings = []
    summary = audit.report_summary(
        default_hit, None, findings, "cached_default", expected=1
    )
    audit.require(
        summary["cached_default_hits"] == 1
        and summary["default_fallback_misses"] == 0
        and not findings,
        "cached default confused with miss",
    )
    passed.append({"name": "cached_default_is_genuine_hit", "passed": True})
    audit.write(
        args.output,
        {
            "status": "passed",
            "controls": passed,
            "count": len(passed),
            "analyzer_sha256": audit.base.sha(Path(audit.__file__).read_bytes()),
        },
    )


if __name__ == "__main__":
    main()
