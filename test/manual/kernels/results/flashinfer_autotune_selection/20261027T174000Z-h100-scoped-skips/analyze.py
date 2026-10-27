#!/usr/bin/env python3
"""Assess saved skip-handoff evidence; never import or execute model code."""

import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "20261027T030000Z-h100-replay-coverage/analyze.py"
SPEC = importlib.util.spec_from_file_location("replay_evidence", BASE)
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)
WORKERS = {
    "worker_sha256": "test/manual/kernels/audit_flashinfer_scoped_skip.py",
    "parent_worker_sha256": "test/manual/kernels/audit_flashinfer_replay_coverage.py",
    "scoped_worker_sha256": "test/manual/kernels/audit_flashinfer_scoped_cache.py",
    "adapter_sha256": "test/manual/kernels/audit_flashinfer_scoped_handoff.py",
}
POLICY = {"measure_execution_mode": "cuda_graph"}


def analyze(args):
    checks = []

    def check(condition, label):
        base.require(condition, label)
        checks.append(label)

    source, commit, raw = (
        args.source_root.resolve(),
        args.expected_source_commit,
        args.raw_root,
    )
    for path in (Path(__file__).resolve(), HERE / "plan.json", BASE):
        check(
            path.read_bytes()
            == base.git_bytes(source, commit, str(path.relative_to(source))),
            "frozen analysis dependency: " + path.name,
        )
    plan = base.read(HERE / "plan.json")
    provenance_path, worker_path = raw / "provenance.json", raw / "cold.json"
    provenance, worker = base.read(provenance_path), base.read(worker_path)
    check(
        provenance["execution_commit"] == commit,
        "execution commit matches frozen source",
    )
    check(
        provenance["plan_sha256"] == base.sha((HERE / "plan.json").read_bytes()),
        "frozen plan binding",
    )
    check(
        worker["provenance"] == provenance
        and worker["provenance_sha256"] == base.sha(provenance_path.read_bytes()),
        "worker provenance bytes",
    )
    bindings = provenance["harness_source_sha256"]
    check(
        set(WORKERS.values()) <= bindings.keys(), "all four worker dependencies bound"
    )
    for path, digest in bindings.items():
        check(
            digest == base.sha(base.git_bytes(source, commit, path)),
            "harness source: " + path,
        )
    for field, path in WORKERS.items():
        check(worker[field] == bindings[path], "worker records exact source: " + field)
    check(
        provenance["worker_sha256"] == worker["worker_sha256"],
        "bootstrap targets correct worker",
    )
    imported = worker["sources"]
    check(
        {name: item["sha256"] for name, item in imported.items()}
        == provenance["source_sha256"],
        "actual imported source bindings",
    )
    for name, item in imported.items():
        if name.startswith("sglang."):
            check(
                item["sha256"]
                == base.sha(
                    base.git_bytes(
                        source, commit, "python/" + name.replace(".", "/") + ".py"
                    )
                ),
                "imported SGLang source: " + name,
            )
    for name in ("autotuner.autotuner", "autotune_cache", "gemm.gemm_base"):
        check(
            imported["flashinfer." + name]["sha256"]
            == plan["environment"]["flashinfer_source_sha256"][
                name.replace(".", "/") + ".py"
            ],
            "qualified FlashInfer source: " + name,
        )
    check(
        (raw / "cold.exit-code").read_text().strip() == "0",
        "model process exits successfully",
    )
    container = (raw / "cold.container-id").read_text().strip()
    check(
        len(container) == 64 and all(c in "0123456789abcdef" for c in container),
        "actual container identity",
    )
    bootstrap_path = raw / "cold.bootstrap.json"
    bootstrap = base.read(bootstrap_path)
    check(
        bootstrap["status"] == "passed"
        and bootstrap["worker_sha256"] == worker["worker_sha256"]
        and bootstrap["result_sha256"] == base.sha(worker_path.read_bytes()),
        "source/native bootstrap bound to result",
    )
    check(
        worker["schema"] == 1
        and worker["study"] == "actual_model_scoped_skip_handoff"
        and worker["status"] == "completed"
        and worker["phase"] == "cold",
        "completed single cold skip worker",
    )
    check(
        worker["model_id"] == base.MODEL and worker["model_revision"] == base.REVISION,
        "pinned model identity",
    )
    check(
        type(worker["started_ns"]) is int
        and worker["finished_ns"] > worker["started_ns"],
        "worker chronology",
    )
    check(
        worker["environment"]["torch"] == plan["environment"]["torch"]
        and worker["environment"]["gpu"] == plan["environment"]["gpu"]
        and worker["environment"]["capability"] == [9, 0],
        "qualified framework and GPU architecture",
    )
    config = worker["server_args"]
    check(
        config["flashinfer_autotune_skip_ops"] == ["fp8_gemm"]
        and config["tp_size"] == 1
        and config["quantization"] == "modelopt_fp8",
        "actual requested skip and bounded model configuration",
    )
    check(
        config["cuda_graph_config"]
        == {
            "decode": {"backend": "full", "bs": [1], "max_bs": 1},
            "prefill": {"backend": "disabled"},
        },
        "one full decode graph configuration",
    )
    check(
        worker["profile_calls"] == []
        and worker["protocol_calls"] == []
        and worker["graph_replays"] == 3,
        "zero profiling/protocol calls and three retained replays",
    )
    startup, lifecycle = worker["startup"], worker["scoped_lifecycle"]
    check(
        startup["eligible_linear_layers"] == len(startup["eligible_linears"]) == 128,
        "128 real FP8 linear layers",
    )
    check(
        startup["profile_count"] == 0
        and startup["cache"] == {"memory": [], "loaded": {}},
        "startup has no profiled or loaded winners",
    )
    check(
        worker["cache_before_startup"] == {} and worker["managed_after_numerics"] == {},
        "fresh legacy and unpublished managed store",
    )
    base.file_inventory(startup["files"])
    check(
        bool(startup["files"])
        and all(
            set(item["contents"]) <= {"_metadata", "_generation"}
            for item in startup["files"].values()
        ),
        "legacy save contains metadata only",
    )
    check(
        worker["cache_after_numerics"] == startup["files"]
        and worker["inventory_after_numerics"] == startup["cache"]
        and worker["final_state"] == startup["state"],
        "numerics preserve legacy cache/files/controls",
    )
    reference, legacy = (
        lifecycle["legacy_reference_state"],
        lifecycle["legacy_reference_cache"],
    )
    base.legacy_state(reference)
    check(
        reference["skip_ops"] == []
        and reference["in_v2_context"] is False
        and reference["policy_fields"] == {},
        "legacy scopes have exited",
    )
    check(
        lifecycle["configured_skips"] == lifecycle["resolved_skips"] == ["fp8_gemm"],
        "actual resolved skip handoff",
    )
    check(
        lifecycle["region_root"] == provenance["persistent_cache_roots"]["managed"]
        and lifecycle["region_policy"] == POLICY,
        "one planned managed root and policy",
    )
    tune, strict = lifecycle["tune"], lifecycle["strict"]
    identity = tune["state_inside"]["winner_identity"]
    check(
        isinstance(identity, list)
        and len(identity) == 2
        and identity[0] == lifecycle["region_root"],
        "identified region root",
    )

    def scoped(state, tuning, label):
        check(
            state["winner_identity"] == identity
            and identity == [state["store_root"], state["store_env_hash"]],
            label + ": matching identity",
        )
        check(
            state["is_tuning"] is tuning
            and state["strict_replay"] is (not tuning)
            and state["in_v2_context"] is True
            and state["cache_scope"] == "context"
            and state["managed_store"] is True
            and state["ambient_store"] is False
            and state["isolate_legacy"] is True,
            label + ": scoped ownership and mode",
        )
        check(
            state["skip_ops"] == ["fp8_gemm"]
            and state["policy_fields"] == POLICY
            and all(
                state[k] == reference[k]
                for k in ("buckets", "round_up", "profile_replays")
            ),
            label + ": skip/policy and inherited controls",
        )

    def report(value, status, count, owner, label):
        counters = {"hit": "hits", "miss": "misses", "skipped": "skipped"}
        check(
            all(
                type(value[k]) is int
                and value[k] == (count if k == counters[status] else 0)
                for k in ("hits", "misses", "skipped", "dropped_records")
            ),
            label + ": exact status accounting",
        )
        check(
            len(value["records"]) == count
            and type(value["max_records"]) is int
            and value["max_records"] >= count,
            label + ": complete bounded records",
        )
        expected_source = "skip_ops" if status == "skipped" else "fallback"
        check(
            all(
                row["op"] == "fp8_gemm"
                and row["status"] == status
                and row["source"] == expected_source
                and row["selected_tactic"] == "-1"
                and row["runner"] == "CublasFp8GemmRunner"
                and row["cache_identity"] == owner
                and isinstance(row["input_shapes"], list)
                for row in value["records"]
            ),
            label + ": actual FP8 default selection and identity",
        )

    def decisions(value):
        return [
            (r["op"], r["input_shapes"], r["runner"], r["selected_tactic"])
            for r in value["records"]
        ]

    scoped(tune["state_inside"], True, "preparation")
    check(
        tune["state_inside"] == tune["state_after_forward"]
        and tune["profile_count_before"] == tune["profile_count_after"] == 0
        and tune["files_before"] == tune["files_after"] == {},
        "preparation preserves controls and publishes nothing",
    )
    check(
        len(tune["selections"]) == 128
        and all(
            row["op"] == "fp8_gemm"
            and row["skip_active"] is True
            and row["runner_is_first"] is True
            and row["selected_tactic"] == "-1"
            and row["capturing"] is False
            and row["state"] == tune["state_inside"]
            for row in tune["selections"]
        ),
        "128 actual preparation skips before CUDA recording",
    )
    scoped(lifecycle["replay_state_inside"], False, "replay")
    records = strict["records"]
    check(
        [r["capturing"] for r in records] == [False, False, True],
        "original two warmups then actual capture",
    )
    for index, row in enumerate(records):
        scoped(row["state_before"], False, "capture stage " + str(index))
        check(
            row["state_after"] == row["state_before"]
            and row["profile_count_before"] == row["profile_count_after"] == 0,
            "capture stage controls/profile count " + str(index),
        )
        report(row["coverage"], "skipped", 128, identity, "capture stage " + str(index))
        check(
            decisions(row["coverage"]) == decisions(records[0]["coverage"]),
            "warmup/capture selection agreement " + str(index),
        )
    report(strict["coverage"], "skipped", 384, identity, "aggregate graph coverage")
    check(
        strict["coverage"]["records"]
        == [r for stage in records for r in stage["coverage"]["records"]],
        "aggregate report binds actual invocation records",
    )
    check(
        lifecycle["complete"] is True
        and lifecycle["legacy_restored"] is True
        and len(lifecycle["reset_calls"]) == 3,
        "completed adapter and three supplied extra resets",
    )
    check(
        strict["files_after"] == {}
        and strict["state_after"] == lifecycle["legacy_exit_state"] == reference
        and lifecycle["legacy_exit_cache"] == legacy,
        "graph exit restores legacy ownership/files",
    )
    for label in ("legacy_before", "legacy_after"):
        report(lifecycle[label], "miss", 128, None, label)
    check(
        decisions(lifecycle["legacy_before"]) == decisions(lifecycle["legacy_after"]),
        "legacy defaults unchanged across graph scope",
    )
    numerical = worker["numerical_reports"]
    check(
        [r["arm"] for r in numerical] == ["eager", "graph"] * 3,
        "three eager/retained-graph pairs",
    )
    for index, row in enumerate(numerical):
        check(
            row["state_before"] == row["state_after"] == reference
            and row["cache_before"] == row["cache_after"] == legacy
            and row["profile_delta"] == 0,
            "numerical report preserves scope/cache " + str(index),
        )
        graph = row["arm"] == "graph"
        check(row["replay_delta"] == int(graph), "actual numerical route " + str(index))
        report(
            row["coverage"],
            "miss",
            0 if graph else 128,
            None,
            "numerical " + str(index),
        )
        if not graph:
            check(
                decisions(row["coverage"]) == decisions(lifecycle["legacy_after"]),
                "bare numerical default choices " + str(index),
            )
    calls = worker["selection_calls"]
    check(
        calls
        and all(
            r["op"] == "fp8_gemm"
            and r["runner_is_first"] is True
            and r["selected_tactic"] == "-1"
            for r in calls
        ),
        "all raw operation selections use first runner/default",
    )
    startup_calls = [
        r for r in calls if r["stage"] == "startup" and r["state"]["is_tuning"]
    ]
    check(
        startup_calls
        and all(
            r["skip_active"] is True and r["state"]["skip_ops"] == ["fp8_gemm"]
            for r in startup_calls
        ),
        "actual startup tuning honors configured exclusions",
    )
    check(
        sum(r["capturing"] is True for r in calls) == 128,
        "one actual capture with 128 selections",
    )
    check(
        not base.validate_numerics(worker["numerics"]),
        "frozen numerical/freshness/changed-restored checks pass",
    )
    check(
        all(
            a["profile_delta"] == 0
            for row in worker["numerics"]["checks"]
            for a in row["arms"].values()
        ),
        "numerical arms perform no profiling",
    )
    base.validate_controls(worker["isolated_control"])
    checks.append("isolated real-tuner hit/miss controls pass")
    return {
        "status": "passed",
        "qualification_complete": True,
        "source_commit": commit,
        "result_sha256": base.sha(worker_path.read_bytes()),
        "bootstrap_sha256": base.sha(bootstrap_path.read_bytes()),
        "container_id": container,
        "checks": checks,
        "check_count": len(checks),
        "findings": [],
        "graph_skips": 384,
        "prepared_skips": 128,
        "neutral_decode_misses": 640,
        "startup_skipped_calls": len(startup_calls),
        "profile_calls": 0,
        "retained_graph_replays": 3,
        "numerical_cases": worker["numerics"]["checks"],
        "limits": "One dense TP1 H100 lifecycle/correctness run; no cache-hit/reuse or performance evidence. Exact model files/native providers require the separately bound outer audit.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = analyze(args)
    except Exception as error:
        result = {
            "status": "failed",
            "qualification_complete": False,
            "findings": [{"type": type(error).__name__, "message": str(error)}],
        }
    base.write(args.output, result)
    raise SystemExit(result["status"] != "passed")


if __name__ == "__main__":
    main()
