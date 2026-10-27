#!/usr/bin/env python3
"""Validate report-only replay evidence and preserve coverage gaps as findings."""

import argparse
import copy
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

MODEL = "nvidia/Llama-3.1-8B-Instruct-FP8"
REVISION = "42d9515ebd69eea3a87351d079c671c3c5ff0a31"
WORKER = "test/manual/kernels/audit_flashinfer_replay_coverage.py"
STATE_KEYS = (
    "is_tuning",
    "active_tuning_contexts",
    "managed_store",
    "ambient_store",
    "winner_identity",
    "measurement_policy",
    "skip_ops",
    "buckets",
    "round_up",
    "profile_replays",
)


class EvidenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def nonnegative_int(value, label):
    require(type(value) is int and value >= 0, f"{label}: expected nonnegative integer")
    return value


def normalize_report(report):
    """Keep lookup status distinct from tactic, including a genuine cached -1."""
    for key in ("hits", "misses", "skipped", "dropped_records", "max_records"):
        nonnegative_int(report.get(key), key)
    records = report.get("records")
    require(isinstance(records, list), "coverage records must be a list")
    total = sum(report[key] for key in ("hits", "misses", "skipped"))
    require(total > 0, "empty replay coverage is not evidence")
    require(len(records) <= report["max_records"], "coverage detail exceeds bound")
    require(
        total == len(records) + report["dropped_records"], "coverage count conservation"
    )
    by_status = Counter()
    groups = Counter()
    for record in records:
        status, source = record.get("status"), record.get("source")
        require(status in ("hit", "miss", "skipped"), "unknown replay status")
        require(isinstance(record.get("op"), str) and record["op"], "missing operation")
        require(isinstance(record.get("input_shapes"), list), "missing input shapes")
        require(isinstance(source, str) and source, "missing cache source")
        require(
            isinstance(record.get("selected_tactic"), str),
            "missing tactic representation",
        )
        require(
            record.get("cache_identity") is None,
            "unexpected identified winner partition",
        )
        for field in ("requested_policy", "stored_policy"):
            policy = record.get(field)
            if policy is not None:
                require(
                    isinstance(policy, list) and len(policy) in (4, 6),
                    "malformed legacy profiling provenance",
                )
                require(
                    policy[0] == "cuda_graph_profile_replays"
                    and policy[2] == "l2_cache_policy",
                    "unknown legacy provenance fields",
                )
                require(
                    policy[1] is None or (type(policy[1]) is int and policy[1] >= 1),
                    "invalid replay-count provenance",
                )
                require(isinstance(policy[3], str), "invalid L2 provenance")
                if len(policy) == 6:
                    require(
                        policy[4:] == ["cuda_graph_timing_mode", "individual"],
                        "unknown legacy timing provenance",
                    )
        if status == "miss":
            require(
                source == "fallback", "ordinary lookup miss must retain fallback source"
            )
        if source == "fallback":
            require(status in ("miss", "skipped"), "fallback cannot be relabeled a hit")
        if status == "hit":
            require(source in ("memory", "legacy", "bundled"), "unqualified hit source")
        by_status[status] += 1
        groups[
            (
                record["op"],
                status,
                source,
                record["selected_tactic"],
                canonical(record["input_shapes"]),
            )
        ] += 1
    for status, field in (("hit", "hits"), ("miss", "misses"), ("skipped", "skipped")):
        require(
            by_status[status] <= report[field],
            "detail count exceeds corresponding total",
        )
        if report["dropped_records"] == 0:
            require(
                by_status[status] == report[field],
                "missing or mislabeled report record",
            )
    return {
        "lookups": total,
        **{
            key: report[key] for key in ("hits", "misses", "skipped", "dropped_records")
        },
        "groups": [
            {
                "op": op,
                "status": status,
                "source": source,
                "tactic_repr": tactic,
                "count": count,
            }
            for (op, status, source, tactic, shapes), count in sorted(groups.items())
        ],
        "cached_default_hits": sum(
            v
            for (o, st, so, tactic, shapes), v in groups.items()
            if st == "hit" and tactic == "-1"
        ),
        "fallback_default_misses": sum(
            v
            for (o, st, so, tactic, shapes), v in groups.items()
            if st == "miss" and tactic == "-1"
        ),
        "intended_fp8_gemm_records": sum(
            v for (op, st, so, ta, shapes), v in groups.items() if op == "fp8_gemm"
        ),
    }


def legacy_state(state):
    require(
        all(key in state for key in STATE_KEYS), "incomplete tuner control snapshot"
    )
    require(
        state["is_tuning"] is False and state["active_tuning_contexts"] == 0,
        "tuning still active",
    )
    require(
        state["managed_store"] is False and state["ambient_store"] is False,
        "ambient managed store changes observer semantics",
    )
    require(
        state["winner_identity"] is None and state["measurement_policy"] == "None",
        "observer changed cache or policy identity",
    )
    return {key: state[key] for key in STATE_KEYS}


def file_inventory(files):
    require(isinstance(files, dict), "invalid cache file inventory")
    for name, item in files.items():
        require(
            isinstance(name, str) and name.endswith(".json"), "invalid cache filename"
        )
        require(
            isinstance(item.get("sha256"), str) and len(item["sha256"]) == 64,
            "invalid cache byte hash",
        )
        require(
            isinstance(item.get("contents"), dict), "missing complete cache contents"
        )
    return {name: item["sha256"] for name, item in files.items()}


def validate_controls(control):
    require(isinstance(control, dict), "missing isolated real-tuner control")
    normalized = [normalize_report(control[key]) for key in ("miss", "cached_default")]
    require(
        normalized[0]["fallback_default_misses"] == 1 and normalized[0]["lookups"] == 1,
        "negative control did not expose a real default miss",
    )
    require(
        normalized[1]["cached_default_hits"] == 1 and normalized[1]["lookups"] == 1,
        "positive control did not expose a real cached default",
    )
    for field in (
        "exception_restored",
        "state_restored",
        "singleton_restored",
        "model_cache_unchanged",
    ):
        require(control.get(field) is True, "isolated control failed: " + field)
    require(control.get("profile_count") == 0, "isolated lookup control profiled")
    return normalized


def validate_numerics(numerics):
    checks = numerics.get("checks")
    require(
        isinstance(checks, list)
        and [row.get("case") for row in checks] == ["original", "changed", "restored"],
        "missing changed/restored numerical cases",
    )
    tokens, prompts = [], []
    gaps = []
    for row in checks:
        require(type(row.get("decode_token")) is int, "missing actual decode token")
        tokens.append(row["decode_token"])
        require(
            isinstance(row.get("prompt_input_ids"), list) and row["prompt_input_ids"],
            "missing actual prefill tokens",
        )
        prompts.append(row["prompt_input_ids"])
        require(
            type(row.get("allclose")) is bool and type(row.get("argmax_equal")) is bool,
            "missing numerical comparison",
        )
        require(
            isinstance(row.get("max_abs_error"), (int, float))
            and math.isfinite(row["max_abs_error"])
            and row["max_abs_error"] >= 0,
            "invalid numerical error",
        )
        require(
            row.get("atol") == 0.05 and row.get("rtol") == 0.01,
            "unregistered numerical tolerance",
        )
        arms = row.get("arms", {})
        require(set(arms) == {"eager", "graph"}, "missing eager/graph comparison")
        require(
            arms["eager"]["output_shape"] == arms["graph"]["output_shape"],
            "mismatched logits shapes",
        )
        for arm, replay_count in (("eager", 0), ("graph", 1)):
            item = arms[arm]
            require(item.get("finite") is True, "nonfinite output after execution")
            require(
                item.get("replay_delta") == replay_count,
                "numerical execution did not use requested serving path",
            )
            require(
                item.get("poisoned_graph_output") is (arm == "graph"),
                "missing graph output freshness challenge",
            )
            require(
                item.get("graph_output_owner_unchanged") is True,
                "captured output owner changed",
            )
            require(
                isinstance(item.get("output_sha256"), str)
                and len(item["output_sha256"]) == 64,
                "missing cloned output hash",
            )
            nonnegative_int(item.get("profile_delta"), "numerical profile delta")
            require(
                item["prompt_input_ids"] == row["prompt_input_ids"],
                "eager/graph prefix mismatch",
            )
            require(
                item.get("decode_token") == row["decode_token"],
                "numerical arm changed token",
            )
            require(
                item.get("original_graph_unchanged") is True,
                "numerics recaptured or replaced graph",
            )
            if arm == "graph":
                require(
                    item.get("graph_input_ids") == [row["decode_token"]],
                    "retained graph did not consume changed/restored input",
                )
        require(
            arms["eager"]["prefix_seq_lens"]
            == arms["graph"]["prefix_seq_lens"]
            == [len(row["prompt_input_ids"])],
            "eager/graph KV prefix lengths differ",
        )
        if not row["allclose"] or not row["argmax_equal"]:
            gaps.append(
                {
                    "kind": "eager_graph_numerical_difference",
                    "case": row["case"],
                    "max_abs_error": row["max_abs_error"],
                    "allclose": row["allclose"],
                    "argmax_equal": row["argmax_equal"],
                }
            )
    require(
        tokens[0] == tokens[2] and tokens[0] != tokens[1],
        "vacuous changed/restored token control",
    )
    require(
        prompts[0] == prompts[1] == prompts[2],
        "numerical cases used different prefixes",
    )
    require(
        type(numerics.get("changed_output")) is bool
        and type(numerics.get("restored_output_allclose")) is bool,
        "missing changed/restored result",
    )
    if not numerics["changed_output"]:
        gaps.append({"kind": "changed_input_did_not_change_output"})
    if not numerics["restored_output_allclose"]:
        gaps.append({"kind": "restored_input_changed_output"})
    return gaps


def validate_worker(worker, phase):
    require(
        worker.get("schema") == 1
        and worker.get("study") == "actual_model_decode_replay_coverage",
        "wrong worker schema",
    )
    require(
        worker.get("status") == "completed",
        "worker incomplete; preserve its error as incomplete evidence",
    )
    require(
        worker.get("phase") == phase
        and worker.get("model_id") == MODEL
        and worker.get("model_revision") == REVISION,
        "wrong model or phase",
    )
    require(
        type(worker.get("started_ns")) is int
        and worker["finished_ns"] > worker["started_ns"],
        "invalid worker chronology",
    )
    files_before = file_inventory(worker["cache_before_startup"])
    require(bool(files_before) is (phase == "warm"), "cold/warm cache setup mismatch")
    startup = worker["startup"]
    nonnegative_int(startup.get("profile_count"), "startup profiles")
    require(
        startup.get("eligible_linear_layers", 0) > 0,
        "model did not select intended FlashInfer linear route",
    )
    require(
        isinstance(startup.get("eligible_linears"), list)
        and len(startup["eligible_linears"]) == startup["eligible_linear_layers"],
        "missing actual eligible linear inventory",
    )
    require(
        len({row["name"] for row in startup["eligible_linears"]})
        == startup["eligible_linear_layers"],
        "duplicate eligible linear",
    )
    startup_state = legacy_state(startup["state"])
    startup_files = file_inventory(startup["files"])
    require(startup_files, "startup produced no v1 persistence evidence")
    require(
        startup_files == file_inventory(worker["cache_after_numerics"]),
        "serving overwrote persisted choices",
    )
    require(
        legacy_state(worker["final_state"]) == startup_state,
        "numerics changed serving controls",
    )
    require(
        worker["inventory_after_numerics"] == startup["cache"],
        "numerics changed live winner cache",
    )
    profiles = worker.get("profile_calls")
    require(isinstance(profiles, list), "missing actual profiling counter")
    require(
        all(isinstance(row.get("stage"), str) for row in profiles),
        "missing profiling stage",
    )
    require(
        startup["profile_count"] <= len(profiles), "profiling counter moved backwards"
    )
    protocols = worker.get("protocol_calls")
    require(isinstance(protocols, list), "missing protocol call observations")
    require(
        all(type(row.get("observed")) is bool for row in protocols),
        "missing protocol observation scope",
    )
    require(
        not any(row["observed"] for row in protocols),
        "protocol neutrality was not qualified for this observer",
    )
    records = worker.get("capture_records")
    require(
        isinstance(records, list) and len(records) == 3,
        "expected the two existing warmups and one actual capture",
    )
    require(
        [r.get("ordinal") for r in records] == [0, 1, 2],
        "missing/duplicated capture invocation",
    )
    require(
        [r.get("capturing") for r in records] == [False, False, True],
        "warmup and true capture were conflated",
    )
    require(
        [r.get("stage") for r in records]
        == ["capture_warmup", "capture_warmup", "capture"],
        "incorrect capture stage labels",
    )
    require(
        len({canonical(row["shape_key"]) for row in records}) == 1,
        "multiple decode shapes entered this bounded audit",
    )
    findings, coverage = [], []
    for row in records:
        before = legacy_state(row["state_before"])
        require(
            before
            == startup_state
            == legacy_state(row["state_inside"])
            == legacy_state(row["state_after"]),
            "observer changed effective serving controls",
        )
        require(row.get("context_restored") is True, "observer context leaked")
        require(
            row["cache_before"] == row["cache_after"] == startup["cache"],
            "observer changed live winner/file choices",
        )
        require(
            row["profile_count_before"] == row["profile_count_after"],
            "profiling occurred inside report-only scope",
        )
        report = normalize_report(row["coverage"])
        require(
            report["intended_fp8_gemm_records"] > 0,
            "report has no intended FP8 selection calls",
        )
        coverage.append(
            {
                "ordinal": row["ordinal"],
                "stage": row["stage"],
                "shape_key": row["shape_key"],
                **report,
            }
        )
        if report["intended_fp8_gemm_records"] != startup["eligible_linear_layers"]:
            findings.append(
                {
                    "kind": "fp8_lookup_count_differs_from_eligible_layers",
                    "ordinal": row["ordinal"],
                    "actual": report["intended_fp8_gemm_records"],
                    "expected": startup["eligible_linear_layers"],
                }
            )
        for field in ("misses", "skipped", "dropped_records"):
            if report[field]:
                findings.append(
                    {
                        "kind": "coverage_" + field,
                        "ordinal": row["ordinal"],
                        "count": report[field],
                    }
                )
    persistence = worker.get("persistence_calls")
    require(isinstance(persistence, list), "missing persistence call observations")
    for call in persistence:
        require(
            call.get("operation") in ("load_configs", "save_configs"),
            "unknown persistence operation",
        )
        require(
            isinstance(call.get("path"), str) and call["path"],
            "missing actual cache path",
        )
        require(
            isinstance(call.get("after_sha256"), str)
            and len(call["after_sha256"]) == 64,
            "missing persisted byte hash",
        )
    if phase == "warm":
        loads = [row for row in persistence if row["operation"] == "load_configs"]
        if not loads:
            findings.append({"kind": "warm_cache_not_loaded"})
        for call in loads:
            require(
                call["path"] in files_before
                and call["before_sha256"] == files_before[call["path"]],
                "warm load did not read preserved cold file",
            )
            expected = {
                key: value[:2]
                for key, value in worker["cache_before_startup"][call["path"]][
                    "contents"
                ].items()
                if not key.startswith("_")
            }
            if any(
                call["cache_after"]["loaded"].get(key) != value
                for key, value in expected.items()
            ):
                findings.append(
                    {"kind": "persisted_entries_not_adopted", "path": call["path"]}
                )
        if not any(
            group["source"] == "legacy"
            for report in coverage
            for group in report["groups"]
        ):
            findings.append({"kind": "no_observed_v1_file_hit"})
    if phase == "warm" and startup["profile_count"]:
        findings.append(
            {"kind": "warm_startup_profiled", "count": startup["profile_count"]}
        )
    extra_profiles = len(profiles) - startup["profile_count"]
    if extra_profiles:
        findings.append({"kind": "post_startup_profiled", "count": extra_profiles})
    findings.extend(validate_numerics(worker["numerics"]))
    control = validate_controls(worker.get("isolated_control"))
    return {
        "phase": phase,
        "pid": worker["environment"]["pid"],
        "startup_profile_count": startup["profile_count"],
        "post_startup_profile_count": extra_profiles,
        "eligible_linear_layers": startup["eligible_linear_layers"],
        "unobserved_protocol_calls": len(protocols),
        "cache_before_startup": files_before,
        "cache_after_startup": startup_files,
        "coverage": coverage,
        "findings": findings,
        "report_controls": control,
    }


def git_bytes(root, commit, path):
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=root, check=True, capture_output=True
    )
    return result.stdout


def analyze(args):
    root, source = args.raw_root, args.source_root
    plan = read(args.plan)
    require(
        plan["model"]["repository"] == MODEL and plan["model"]["revision"] == REVISION,
        "wrong frozen plan",
    )
    expected_worker = sha(git_bytes(source, args.expected_source_commit, WORKER))
    provenance_path = root / "provenance.json"
    provenance = read(provenance_path)
    require(
        provenance.get("execution_commit") == args.expected_source_commit,
        "wrong execution commit",
    )
    require(
        provenance.get("plan_sha256") == sha(args.plan.read_bytes()),
        "wrong frozen plan hash",
    )
    require(
        provenance.get("worker_sha256") == expected_worker, "wrong worker provenance"
    )
    workers, container_ids = [], []
    for phase in args.phases:
        path = root / f"{phase}.json"
        worker = read(path)
        require(
            worker.get("worker_sha256") == expected_worker,
            "worker source hash does not match execution commit",
        )
        require(
            {name: item["sha256"] for name, item in worker["sources"].items()}
            == provenance.get("source_sha256"),
            "imported sources differ from frozen provenance",
        )
        for module, item in worker["sources"].items():
            if module.startswith("sglang."):
                module_path = "python/" + module.replace(".", "/") + ".py"
                require(
                    item["sha256"]
                    == sha(git_bytes(source, args.expected_source_commit, module_path)),
                    "imported SGLang module differs from frozen source: " + module,
                )
        require(
            {
                "flashinfer.autotune_cache",
                "flashinfer.autotuner.autotuner",
                "flashinfer.gemm.gemm_base",
            }
            <= worker["sources"].keys(),
            "missing actual FlashInfer source evidence",
        )
        require(
            worker.get("provenance_sha256") == sha(provenance_path.read_bytes())
            and worker["provenance"] == provenance,
            "worker provenance does not match original bytes",
        )
        require(
            (root / f"{phase}.exit-code").read_text().strip() == "0",
            "worker process did not finish normally",
        )
        container_id = (root / f"{phase}.container-id").read_text().strip()
        require(
            len(container_id) == 64
            and all(c in "0123456789abcdef" for c in container_id),
            "missing actual container identity",
        )
        container_ids.append(container_id)
        bootstrap_path = root / f"{phase}.bootstrap.json"
        bootstrap = read(bootstrap_path)
        require(
            bootstrap.get("status") == "passed", "native/source bootstrap did not pass"
        )
        require(
            bootstrap.get("worker_sha256") == expected_worker
            and bootstrap.get("result_sha256") == sha(path.read_bytes()),
            "bootstrap is not bound to actual worker result",
        )
        item = validate_worker(worker, phase)
        item.update(
            result_file=path.name,
            result_sha256=sha(path.read_bytes()),
            bootstrap_sha256=sha(bootstrap_path.read_bytes()),
            container_id=container_id,
        )
        workers.append((worker, item))
    if len(workers) == 2:
        cold, warm = (pair[0] for pair in workers)
        require(
            cold["phase"] == "cold" and warm["phase"] == "warm",
            "invalid process ordering",
        )
        require(
            cold["finished_ns"] < warm["started_ns"],
            "fresh warm process preceded cold completion",
        )
        require(len(set(container_ids)) == 2, "cold/warm containers were reused")
        require(
            cold["server_args"] == warm["server_args"],
            "cold/warm serving configuration changed",
        )
        require(cold["sources"] == warm["sources"], "cold/warm imported source changed")
        require(
            cold["provenance_sha256"] == warm["provenance_sha256"],
            "cold/warm provenance changed",
        )
        require(
            cold["startup"]["files"] == warm["cache_before_startup"],
            "fresh warm process did not receive exact cold cache bytes",
        )
    result = {
        "schema": 1,
        "status": "valid_evidence",
        "study": "actual_model_decode_replay_coverage",
        "source_commit": args.expected_source_commit,
        "plan_sha256": sha(args.plan.read_bytes()),
        "workers": [item for worker, item in workers],
        "coverage_complete": all(not item["findings"] for worker, item in workers),
        "findings": [
            {"phase": item["phase"], **finding}
            for worker, item in workers
            for finding in item["findings"]
        ],
        "limits": plan["limits"],
        "interpretation": "Coverage gaps are preserved audit findings, not excluded samples. Selection hits do not prove every backend's physical kernel identity. Numerical checks establish eager/graph consistency for one model and workload.",
    }
    return result


def self_test():
    record = {
        "op": "fp8_gemm",
        "input_shapes": [[1, 1, 32]],
        "cache_identity": None,
        "status": "miss",
        "source": "fallback",
        "runner": "Dummy",
        "selected_tactic": "-1",
        "requested_policy": None,
        "stored_policy": None,
        "reasons": ["cache_miss"],
    }
    report = {
        "hits": 0,
        "misses": 1,
        "skipped": 0,
        "dropped_records": 0,
        "max_records": 8,
        "records": [record],
    }
    controls = []

    def check(name, action, fails=False):
        try:
            action()
        except EvidenceError:
            require(fails, name + " unexpectedly failed")
        else:
            require(not fails, name + " did not reject malformed evidence")
        controls.append({"name": name, "passed": True})

    check(
        "real_default_miss",
        lambda: require(
            normalize_report(report)["fallback_default_misses"] == 1, "miss lost"
        ),
    )
    hit = copy.deepcopy(report)
    hit.update(hits=1, misses=0)
    hit["records"][0].update(
        status="hit",
        source="legacy",
        requested_policy=["cuda_graph_profile_replays", None, "l2_cache_policy", "hot"],
        stored_policy=["cuda_graph_profile_replays", None, "l2_cache_policy", "hot"],
    )
    check(
        "cached_default_is_a_hit",
        lambda: require(normalize_report(hit)["cached_default_hits"] == 1, "hit lost"),
    )
    control = {
        "miss": report,
        "cached_default": hit,
        "exception_restored": True,
        "state_restored": True,
        "singleton_restored": True,
        "model_cache_unchanged": True,
        "profile_count": 0,
    }
    check("isolated_control_normalization", lambda: validate_controls(control))
    empty = copy.deepcopy(report)
    empty.update(misses=0, records=[])
    check("empty_report_rejected", lambda: normalize_report(empty), True)
    missing = copy.deepcopy(report)
    missing["records"] = []
    check("missing_record_rejected", lambda: normalize_report(missing), True)
    mislabeled = copy.deepcopy(report)
    mislabeled["records"][0]["status"] = "hit"
    check("fallback_relabel_rejected", lambda: normalize_report(mislabeled), True)
    truncated = copy.deepcopy(report)
    truncated.update(misses=2, dropped_records=1)
    check(
        "explicit_truncation_preserves_total",
        lambda: require(normalize_report(truncated)["lookups"] == 2, "count lost"),
    )
    state = dict(
        is_tuning=False,
        active_tuning_contexts=0,
        managed_store=False,
        ambient_store=False,
        winner_identity=None,
        measurement_policy="None",
        skip_ops=[],
        buckets=None,
        round_up=False,
        profile_replays=None,
    )
    inventory = {"memory": [], "loaded": {}}
    files = {"cache.json": {"sha256": "a" * 64, "contents": {"_metadata": {}}}}
    numeric = []
    for case, token in (("original", 2), ("changed", 3), ("restored", 2)):
        arms = {}
        for arm in ("eager", "graph"):
            arms[arm] = dict(
                finite=True,
                replay_delta=int(arm == "graph"),
                profile_delta=0,
                output_shape=[1, 8],
                output_sha256="b" * 64,
                poisoned_graph_output=arm == "graph",
                graph_output_owner_unchanged=True,
                prompt_input_ids=[11, 12],
                prefix_seq_lens=[2],
                decode_token=token,
                original_graph_unchanged=True,
                graph_input_ids=[token] if arm == "graph" else None,
            )
        numeric.append(
            dict(
                case=case,
                decode_token=token,
                prompt_input_ids=[11, 12],
                arms=arms,
                allclose=True,
                argmax_equal=True,
                max_abs_error=0,
                atol=0.05,
                rtol=0.01,
            )
        )
    payload = dict(
        schema=1,
        study="actual_model_decode_replay_coverage",
        status="completed",
        phase="cold",
        model_id=MODEL,
        model_revision=REVISION,
        started_ns=1,
        finished_ns=2,
        cache_before_startup={},
        startup=dict(
            profile_count=1,
            eligible_linear_layers=1,
            eligible_linears=[{"name": "layer"}],
            state=state,
            cache=inventory,
            files=files,
        ),
        cache_after_numerics=files,
        inventory_after_numerics=inventory,
        final_state=state,
        profile_calls=[{"stage": "startup"}],
        protocol_calls=[],
        capture_records=[],
        persistence_calls=[],
        isolated_control=copy.deepcopy(control),
        environment={"pid": 42},
        numerics=dict(
            checks=numeric, changed_output=True, restored_output_allclose=True
        ),
    )
    for i, stage in enumerate(("capture_warmup", "capture_warmup", "capture")):
        payload["capture_records"].append(
            dict(
                ordinal=i,
                stage=stage,
                capturing=i == 2,
                shape_key={"bs": 1},
                state_before=state,
                state_inside=state,
                state_after=state,
                cache_before=inventory,
                cache_after=inventory,
                profile_count_before=1,
                profile_count_after=1,
                context_restored=True,
                coverage=copy.deepcopy(hit),
            )
        )
    check("complete_worker_payload", lambda: validate_worker(payload, "cold"))
    missed = copy.deepcopy(payload)
    missed["capture_records"][2]["coverage"] = copy.deepcopy(report)
    check(
        "valid_capture_miss_is_a_finding",
        lambda: require(
            any(
                row["kind"] == "coverage_misses"
                for row in validate_worker(missed, "cold")["findings"]
            ),
            "miss was filtered",
        ),
    )
    warm = copy.deepcopy(payload)
    warm.update(phase="warm", cache_before_startup=files)
    check(
        "warm_profiling_is_a_finding",
        lambda: require(
            any(
                row["kind"] == "warm_startup_profiled"
                for row in validate_worker(warm, "warm")["findings"]
            ),
            "warm profile was filtered",
        ),
    )
    startup_protocol = copy.deepcopy(payload)
    startup_protocol["protocol_calls"] = [{"stage": "startup", "observed": False}]
    check(
        "startup_only_protocol_is_outside_report_scope",
        lambda: validate_worker(startup_protocol, "cold"),
    )
    observed_protocol = copy.deepcopy(payload)
    observed_protocol["protocol_calls"] = [{"stage": "capture", "observed": True}]
    check(
        "observed_protocol_rejected",
        lambda: validate_worker(observed_protocol, "cold"),
        True,
    )
    absent = copy.deepcopy(payload)
    absent["capture_records"].pop()
    check(
        "missing_actual_capture_rejected", lambda: validate_worker(absent, "cold"), True
    )
    changed = copy.deepcopy(payload)
    changed["capture_records"][0]["state_inside"] = {
        **changed["capture_records"][0]["state_inside"],
        "skip_ops": ["fp8_gemm"],
    }
    check(
        "observer_control_change_rejected",
        lambda: validate_worker(changed, "cold"),
        True,
    )
    wrong_prefix = copy.deepcopy(payload)
    wrong_prefix["numerics"]["checks"][0]["arms"]["eager"]["prefix_seq_lens"] = [3]
    check(
        "different_kv_prefix_rejected",
        lambda: validate_worker(wrong_prefix, "cold"),
        True,
    )
    control["model_cache_unchanged"] = False
    check("control_state_mutation_rejected", lambda: validate_controls(control), True)
    return {"status": "passed", "controls": controls, "count": len(controls)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--expected-source-commit")
    parser.add_argument(
        "--plan", type=Path, default=Path(__file__).with_name("plan.json")
    )
    parser.add_argument(
        "--phases", nargs="+", choices=("cold", "warm"), default=["cold", "warm"]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            result = self_test()
        else:
            require(
                args.raw_root and args.source_root and args.expected_source_commit,
                "analysis paths and source commit are required",
            )
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
