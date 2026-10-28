"""Validate matched v1/v2 startup and summarize whole paired timing blocks.

Actual natural/default graphs remain measured even when both select default.
Admission and fresh-process persistence checks never enter timing estimates.
"""

import argparse
import collections
import copy
import importlib.util
import json
import math
import random
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
ARMS = ("v1", "v2")
ROLES = ("natural", "default")
MODES = ("eager", "cuda_graph")
ENDPOINTS = (1, 2, 4, 8, 16, 32, 64, 128, 144)
BLOCKS, RESAMPLES, SEED = 8, 100_000, 202610280300
PRIOR = HERE.parent / "20261027T000000Z-h100-mlp-selection"
PRIOR_SHA = "96ce2d985ca0777a447102baf54f725d4960d3ec61295ed3946457491a30332e"


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


legacy = module(PRIOR / "analyzer.py", "matched_mlp_leaf_analysis")
require, load, sha, positive = legacy.require, legacy.load, legacy.sha, legacy.positive
require(
    sha(PRIOR / "analyzer.py") == PRIOR_SHA, "unchanged trace and arithmetic helpers"
)


def close(actual, expected, label):
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15), label)


def validate_schedule(schedule):
    require(schedule["blocks"] == BLOCKS, "eight prespecified paired blocks")
    entries = schedule["entries"]
    require(len(entries) == 16, "exactly sixteen cold timing assignments")
    for index, entry in enumerate(entries):
        block, position = divmod(index, 2)
        arms = ARMS if block % 2 == 0 else ARMS[::-1]
        require(
            entry["sequence"] == index
            and entry["block"] == entry["trial"] == block
            and entry["arm"] == arms[position]
            and entry["phase"] == "cold",
            "alternating complete v1/v2 launch pairs",
        )
        if position:
            require(
                all(
                    entry[k] == entries[index - 1][k]
                    for k in ("capture_order", "mode_order")
                ),
                "matched within-pair histories",
            )
    for key, choices in (
        ("capture_order", ("natural-first", "default-first")),
        ("mode_order", ("eager-first", "graph-first")),
    ):
        require(
            collections.Counter(row[key] for row in entries[::2])
            == dict.fromkeys(choices, 4),
            "balanced " + key,
        )
    require(
        len(
            {
                (row["arm"], row["capture_order"], row["mode_order"])
                for row in entries[::2]
            }
        )
        == 8,
        "all eight joint launch/capture/mode assignments",
    )
    require(
        set(schedule["round_schedules"]) == {str(i) for i in range(BLOCKS)},
        "all round schedules",
    )
    for schedules in schedule["round_schedules"].values():
        require(set(schedules) == set(MODES), "only eager and graph schedules")
        for mode, count in (("eager", 12), ("cuda_graph", 24)):
            orders = schedules[mode]
            require(
                len(orders) == count
                and all(sorted(row) == sorted(ROLES) for row in orders),
                "complete paired " + mode + " schedule",
            )
            require(
                collections.Counter(row[0] for row in orders)
                == dict.fromkeys(ROLES, count // 2),
                "balanced " + mode + " first positions",
            )
    return entries


def validate_startup(data):
    arm, phase = data["arm"], data["phase"]
    startup, natural, cache, final = (
        data[key] for key in ("startup", "natural", "cache", "final")
    )
    profiles = startup["profiles"]
    require(
        len(profiles) == startup["profile_attempts"] == final["profile_attempts"],
        "all profile calls accounted for",
    )
    require(
        not data["forbidden_profile_attempts"] and not data["profile_protocol_errors"],
        "no forbidden or swallowed profiling",
    )
    selections = startup["selection"]
    require(
        len(selections) == 1
        and selections[0]["in_v2_context"] == (arm == "v2")
        and selections[0]["is_tuning_mode"] is True
        and selections[0]["measurement_policy"] is None,
        "actual assigned startup API and unspecified policy",
    )
    require(
        cache["after_startup_sha256"] == cache["final_sha256"]
        and cache["canonical_state_after_startup"] == cache["canonical_state_final"],
        "unchanged persistent and in-memory winners",
    )
    require(
        all(
            final[key] is True
            for key in (
                "zero_additional_profiling",
                "cache_unchanged",
                "operands_unchanged",
                "native_provider_restored",
                "on_caller_stream",
            )
        ),
        "restored serving state",
    )
    require(
        not any(
            final[key]
            for key in (
                "active_v2",
                "is_tuning_mode",
                "tuning_contexts",
                "managed_store",
            )
        )
        and final["measurement_policy"] is None,
        "bare serving after context exit",
    )
    tier = "memory" if phase == "cold" else "file"
    require(
        natural["lookup_tier"] == tier
        and natural["cache_hit"] is True
        and natural["tactic"] == selections[0]["tactic"],
        "actual startup winner consumed from cache",
    )
    if phase == "warm":
        require(
            not profiles
            and cache["before_sha256"] == cache["final_sha256"]
            and cache["memory_entries"] == 0,
            "fresh warm file reuse without profiling",
        )
        return {}
    require(phase == "cold" and cache["before_sha256"] is None, "empty cold cache")
    count = 8 + int(arm == "v2")
    require(len(profiles) == len(ENDPOINTS) * count, "complete original search count")
    require(
        [r["m"] for r in profiles] == [m for m in ENDPOINTS for _ in range(count)],
        "ordered profile buckets",
    )
    groups = {}
    for index, row in enumerate(profiles):
        require(
            row["index"] == index
            and row["stage"] == "startup"
            and row["runner"] == "CublasFp8GemmRunner"
            and row["is_tuning_mode"] is True
            and row["in_v2_context"] == (arm == "v2"),
            "actual search context and candidate index",
        )
        require(
            row["status"] == "finite" and not row.get("error"),
            "all candidates measured successfully",
        )
        positive(row["profiler_ms"])
        settings = row["autotuner_settings"]
        require(
            settings["warmup"] == 3
            and settings["repeat"] == 10
            and settings["stream_delay_micro_secs"] == 5000
            and settings["cuda_graph_delay_micro_secs"] == 100
            and settings["use_global_timer"] is False
            and settings["effective_measure_policy"] is None
            and settings["effective_policy_timer"] is None,
            "matched unchanged measurement settings",
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
            "matched operation timing configuration",
        )
        require(
            row["fingerprinted"] == (data["mode"] == "admission")
            and ("fingerprints" in row) == row["fingerprinted"],
            "admission-only input fingerprints",
        )
        groups.setdefault(str(row["m"]), []).append(row)
    for rows in groups.values():
        require(
            sum(row["tactic"] == -1 for row in rows) == int(arm == "v2"),
            "default participates only in v2",
        )
        require(
            len({json.dumps(row["tactic"]) for row in rows}) == count
            and len({row["key"] for row in rows}) == 1,
            "unique candidates share one profile key",
        )
        require(
            all(row["metadata"] == rows[0]["metadata"] for row in rows),
            "same operands across each candidate pool",
        )
        if arm == "v2":
            require(
                rows[0]["tactic"] == -1, "v2 default retains its natural first position"
            )
    runtime = groups["144"]
    require(
        natural["key"] == runtime[0]["key"]
        and min(runtime, key=lambda row: row["profiler_ms"])["tactic"]
        == natural["tactic"],
        "original runtime winner is first profiling minimum",
    )
    return {
        m: {
            "tactics": [r["tactic"] for r in rows],
            "key": rows[0]["key"],
            "fingerprints": sorted(
                {tuple(r["fingerprints"]) for r in rows if "fingerprints" in r}
            ),
        }
        for m, rows in groups.items()
    }


def validate_matched_candidates(old, new, admission):
    require(
        set(old) == set(new) == {str(m) for m in ENDPOINTS}, "matched bucket inventory"
    )
    for key, v1 in old.items():
        v2 = new[key]
        require(
            v1["tactics"] == [t for t in v2["tactics"] if t != -1]
            and v1["key"] == v2["key"],
            "identical explicit pool/order/key across arms",
        )
        if admission:
            require(
                len(v1["fingerprints"]) == 1
                and v1["fingerprints"] == v2["fingerprints"],
                "identical synthesized admission inputs",
            )


def validate_outputs(data, entry):
    natural, tier = (
        data["natural"]["tactic"],
        "memory" if entry["phase"] == "cold" else "file",
    )
    bindings = data["binding_acceptance_checks"]
    require(
        bindings
        and all(
            row["accepted"] is True
            and row["used_descriptor"] == ([] if row["tactic"] == -1 else [1])
            for row in bindings
        ),
        "native descriptor acceptance without silent fallback",
    )
    checks = {row["check"]: row for row in data["correctness"]}
    require(
        checks and len(checks) == len(data["correctness"]), "unique numerical checks"
    )
    for checked in checks.values():
        value = checked["quantized_reference"]
        require(
            value["passed"] is True
            and math.isfinite(value["max_normalized_error"])
            and 0 <= value["max_normalized_error"] <= 1,
            "every output meets numerical tolerance",
        )
    order = list(ROLES if entry["capture_order"] == "natural-first" else ROLES[::-1])
    graphs = {row["id"]: row for row in data["graphs"]}
    require(
        [row["id"] for row in data["graphs"]] == order and len(graphs) == 2,
        "frozen graph capture order",
    )
    require(
        len({g["graph_object_id"] for g in graphs.values()})
        == len({g["output"]["data_ptr"] for g in graphs.values()})
        == 2,
        "independent retained graphs and outputs",
    )
    previous = 0
    for role in order:
        graph = graphs[role]
        require(
            previous <= graph["capture_started_ns"] < graph["capture_ended_ns"],
            "sequential graph captures",
        )
        previous = graph["capture_ended_ns"]
        require(
            graph["tactic"] == (natural if role == "natural" else -1)
            and graph["lookup_tier"] == (tier if role == "natural" else "miss"),
            "actual selected and default graphs",
        )
        require(
            graph["output"]["data_ptr"] == graph["inputs"][4]["data_ptr"],
            "graph output ownership",
        )
        for suffix in ("changed_input", "restored_input"):
            check = checks[role + "_" + suffix]
            require(
                check["output_data_ptr"] == graph["output"]["data_ptr"]
                and check["exact_zero_required"] == (suffix == "changed_input"),
                "changed inputs affect the retained graph output",
            )
    require(
        data["miss_scopes"]
        and all(
            row["restored"]
            and row["cache_sha256_before"] == row["cache_sha256_after"]
            and row["winner_entries_after_removal"]
            == row["file_entries_after_removal"]
            == 0
            for row in data["miss_scopes"]
        ),
        "genuine fallback controls restore original winners",
    )
    return checks, graphs, previous


def paired_logs(pairs, orders, mode):
    count = 12 if mode == "eager" else 24
    require(
        mode in MODES and len(pairs) == len(orders) == count,
        "all prespecified timing pairs retained",
    )
    field = "wall_per_call_ms" if mode == "eager" else "gpu_event_per_call_ms"
    logs, absolute, previous = [], {role: [] for role in ROLES}, 0
    for index, (pair, order) in enumerate(zip(pairs, orders, strict=True)):
        require(
            pair["index"] == index
            and pair["order"] == order
            and [row["role"] for row in pair["measurements"]] == order,
            "actual frozen paired timing order",
        )
        rows = pair["measurements"]
        require(
            len(rows) == 2 and rows[0]["iterations"] == rows[1]["iterations"],
            "matched iterations within pair",
        )
        values = {}
        for row in rows:
            n = row["iterations"]
            require(type(n) is int and n > 0, "positive batch count")
            for clock in ("wall", "gpu_event"):
                close(
                    positive(row[clock + "_batch_ms"]) / n,
                    positive(row[clock + "_per_call_ms"]),
                    "raw batch to per-call timing",
                )
            require(
                previous <= row["started_ns"] < row["ended_ns"],
                "strictly ordered timing windows",
            )
            previous = row["ended_ns"]
            values[row["role"]] = row[field]
            absolute[row["role"]].append(row[field])
        logs.append(math.log(values["natural"] / values["default"]))
    return {
        "metric": field,
        "paired_log_ratios": logs,
        "median_log_ratio": statistics.median(logs),
        "natural_over_default": math.exp(statistics.median(logs)),
        "median_absolute_ms": {
            role: statistics.median(values) for role, values in absolute.items()
        },
    }


def validate_measurements(data, entry, schedule, checks, graphs, previous):
    if data["mode"] == "admission":
        require(
            data["measurements"] == {},
            "admission and warm verification contain no performance windows",
        )
        return {}, previous
    order = MODES if entry["mode_order"] == "eager-first" else MODES[::-1]
    require(set(data["measurements"]) == set(MODES), "both serving modes measured")
    modes = {}
    for mode in order:
        group = data["measurements"][mode]
        require(
            previous <= group["started_ns"] < group["ended_ns"],
            "actual measurement mode order",
        )
        previous = group["ended_ns"]
        modes[mode] = paired_logs(
            group["pairs"], schedule["round_schedules"][str(entry["trial"])][mode], mode
        )
        require(
            [row["role"] for row in group["initial_warmups"]] == list(ROLES)
            and all(row["actual_ms"] >= 500 for row in group["initial_warmups"]),
            "full prescribed initial warmups",
        )
        common = group["common_iterations"]
        calibrations = group["calibration_by_role"]
        if mode == "eager":
            require(
                set(calibrations) == set(ROLES)
                and common
                == max(row["effective_iterations"] for row in calibrations.values()),
                "common batch from both eager calibrations",
            )
            for calibration in calibrations.values():
                batches = calibration["batches"]
                require(
                    batches
                    and batches[0]["iterations"] == 20
                    and batches[-1]["iterations"] == calibration["effective_iterations"]
                    and batches[-1]["wall_batch_ms"] >= 20,
                    "bounded complete eager calibration",
                )
                require(
                    all(row["wall_batch_ms"] < 20 for row in batches[:-1])
                    and all(
                        b["iterations"] == a["iterations"] * 2
                        for a, b in zip(batches, batches[1:])
                    ),
                    "calibration stops at first qualifying batch",
                )
        else:
            require(
                common == 16
                and not calibrations
                and group["batch_policy"] == "fixed16_queued",
                "fixed queued graph sample count",
            )
        batches = [
            row for value in calibrations.values() for row in value["batches"]
        ] + [row for pair in group["pairs"] for row in pair["measurements"]]
        for row in batches:
            require(
                row["rewarm"]["actual_ms"] >= 20
                and row["profile_attempts_before"]
                == row["profile_attempts_after"]
                == data["startup"]["profile_attempts"],
                "rewarm without late profiling",
            )
            require(
                group["started_ns"]
                <= row["started_ns"]
                < row["ended_ns"]
                <= group["ended_ns"],
                "sample inside its declared mode",
            )
            check = checks[row["correctness_check"]]
            require(
                check["checked_ns"] >= row["ended_ns"]
                and check["output_data_ptr"] == row["output_data_ptr"],
                "numerical guard follows and owns the measured output",
            )
            for key in ("lookup_before", "lookup_after"):
                lookup = row[key]
                tier = "memory" if row["role"] == "natural" else "miss"
                require(
                    lookup["lookup_tier"] == tier
                    and lookup["cache_hit"] == (tier != "miss")
                    and lookup["tactic"]
                    == (data["natural"]["tactic"] if row["role"] == "natural" else -1),
                    "actual selected/default lookup around measurement",
                )
            if mode == "cuda_graph":
                require(
                    row["iterations"] == 16
                    and row["prequeue_cycles"] == 10_000_000
                    and row["start_query_error"] is None
                    and type(row["start_event_complete_after_enqueue"]) is bool
                    and row["queue_query_is_exclusion_criterion"] is False,
                    "all valid queue-query outcomes retained",
                )
                require(
                    row["output_data_ptr"] == graphs[row["role"]]["output"]["data_ptr"]
                    and row["wall_diagnostic_only"] is True,
                    "retained graph ownership and diagnostic wall time",
                )
        require(
            all(
                row["iterations"] == common
                for pair in group["pairs"]
                for row in pair["measurements"]
            ),
            "common paired batch count",
        )
    return modes, previous


def validate_protocol(data, plan, contract):
    record, arm = data["startup_protocol"], data["arm"]
    require(
        data["confirmation_performed"] is False
        and not any(
            key in data for key in ("confirmation", "diagnostic_profile_attempts")
        ),
        "no shadow confirmation or extra candidate timing",
    )
    require(
        record["context_intervention"] == "inner_v2_context_only",
        "only the inner startup context differs",
    )
    inside = record["inside"]
    require(
        inside["active_v2"] == (arm == "v2")
        and inside["is_tuning_mode"] is True
        and inside["on_forward_stream"] is True
        and inside["measurement_policy"] is None
        and inside["managed_store"] is False,
        "startup API, stream and policy witness",
    )
    for point in ("before", "after"):
        state = record[point]
        require(
            not any(
                state[key]
                for key in (
                    "active_v2",
                    "is_tuning_mode",
                    "tuning_contexts",
                    "managed_store",
                )
            )
            and state["measurement_policy"] is None
            and state["on_caller_stream"] is True,
            "restored startup scope " + point,
        )
    require(
        record["before_cache_sha256"] == data["cache"]["before_sha256"]
        and record["after_cache_sha256"] == data["cache"]["after_startup_sha256"]
        and record["profile_attempts"] == data["startup"]["profile_attempts"],
        "startup cache and call accounting",
    )
    close(
        positive((record["ended_ns"] - record["started_ns"]) / 1e6),
        positive(data["startup"]["startup_ms"]),
        "actual startup host interval",
    )
    require(
        record["ended_ns"]
        < min(graph["capture_started_ns"] for graph in data["graphs"]),
        "startup ends before retained graph capture",
    )
    require(
        record["wrapper_sha256"]
        == sha(ROOT / "python/sglang/srt/model_executor/runner/flashinfer_autotune.py"),
        "actual SGLang wrapper source",
    )
    expected = (
        {}
        if data["phase"] == "warm"
        else {
            str(m): [
                t for t in contract["candidate_order"][str(m)] if arm == "v2" or t != -1
            ]
            for m in ENDPOINTS
        }
    )
    require(
        record["candidate_order"] == expected,
        "original candidate enumeration unchanged",
    )
    keys = (
        []
        if data["phase"] == "warm"
        else [
            key
            for i, key in enumerate(contract["profile_keys"])
            if arm == "v2" or i % 9
        ]
    )
    require(
        [row["key"] for row in data["startup"]["profiles"]] == keys,
        "exact source-defined candidate cache keys",
    )


def validate_binding_header(path, data, plan, worker, ordering=False):
    boot = load(path.with_suffix(".bootstrap.json"))
    require(
        boot["status"] == "passed"
        and boot["binding_arm"] == "patched"
        and (ordering or boot["tuning_arm"] == data["arm"])
        and boot["pid"] == data["pid"]
        and boot["result_sha256"] == sha(path)
        and boot["worker_sha256"] == sha(worker),
        "same-process bootstrap and actual treatment",
    )
    witness = boot["binding_before"]
    actual = data["attestation_before"] if ordering else data["correction"]
    require(
        witness == boot["binding_after"] == actual
        and witness["arm"] == "patched"
        and witness["wrapper_sha256"] == plan["binding"]["patched_sha256"]
        and witness["upstream_head"] == plan["binding"]["upstream_head"],
        "same corrected helper before and after every worker",
    )
    require(
        all(
            witness[key] is True
            for key in (
                "callable_matches_source",
                "profiler_alias_matches",
                "native_identity_passthrough",
            )
        ),
        "loaded callable, profiler alias and native identity",
    )
    return boot


def validate_bootstrap(path, data, plan, raw_root, worker, ordering=False):
    boot = validate_binding_header(path, data, plan, worker, ordering)
    reference_path = raw_root / "native-reference.bootstrap.json"
    require(
        sha(reference_path)
        == "7c4727aca3d99adcb3fe2f35d06f4408b512baa4fe9a91b975bdf77358fe205d",
        "immutable native reference",
    )
    reference = load(reference_path)
    reuse_path = (
        raw_root
        / ("v1" if ordering else data["arm"])
        / "native-build-cache/reuse-manifest.json"
    )
    reuse = load(reuse_path)
    require(
        reuse["status"] == "passed"
        and reuse["native_reference_sha256"] == sha(reference_path)
        and boot["native_reuse"]["reference_sha256"] == sha(reference_path)
        and boot["native_reuse"]["manifest_sha256"] == sha(reuse_path),
        "exact native reuse identity",
    )
    for point in ("before", "after_build", "after_worker"):
        guard = boot["native_reuse"][point]
        require(
            guard["status"] == "passed"
            and guard["elf_sha256"] == plan["native_reuse"]["required_library_sha256"]
            and guard["files_verified"] == len(reuse["native_files"]),
            "unchanged native files " + point,
        )
    require(
        [row["name"] for row in boot["native_reuse"]["ninja_dry_runs"]]
        == ["gemm", "trtllm_utils"],
        "both native providers checked",
    )
    for row in boot["native_reuse"]["ninja_dry_runs"]:
        directory = (
            "/study/runtime-attempt01/cache/current/.cache/flashinfer/0.0.0+unknown/90a/cached_ops/"
            + row["name"]
        )
        require(
            row["returncode"] == 0
            and row["stdout"]
            == f"ninja: Entering directory `{directory}'\nninja: no work to do.\n",
            "native artifact reused without rebuild",
        )
    for name, native in reference["native"]["providers"].items():
        require(
            all(
                boot["native"]["providers"][name][field] == native[field]
                for field in (
                    "library_path",
                    "library_sha256",
                    "ninja_path",
                    "ninja_sha256",
                    "sources",
                    "compile_commands",
                )
            ),
            "identical native provider " + name,
        )
    require(
        boot["native"]["tvm_ffi_sha256"] == reference["native"]["tvm_ffi_sha256"]
        and boot["native_identity_passthrough"] is True,
        "unmodified native binding",
    )
    expected_import = copy.deepcopy(reference["import"])
    expected_import["source_modules"]["flashinfer.tllm_utils"]["sha256"] = plan[
        "binding"
    ]["patched_sha256"]
    require(
        boot["import"] == expected_import,
        "same qualified imports with only corrected wrapper",
    )
    expected_source = copy.deepcopy(reference["source"])
    expected_source["source_sha256"]["flashinfer/tllm_utils.py"] = plan["binding"][
        "patched_sha256"
    ]
    expected_source["declared_overlay"] = {
        "arm": "patched",
        "path": "flashinfer/tllm_utils.py",
        "baseline_sha256": "5eaaa6773f5d7b5bc00de2183eef488f5b2497d7efc687ebf45f97d7f2fd3593",
        "actual_sha256": plan["binding"]["patched_sha256"],
    }
    require(
        boot["source"] == expected_source
        and boot["helper_sha256"] == reference["helper_sha256"]
        and boot["candidate_manifest_sha256"] == reference["candidate_manifest_sha256"],
        "identical fixed source and bootstrap helper",
    )
    return boot


def validate_worker(path, entry, phase, plan, schedule, contract, raw_root):
    data = load(path)
    worker = ROOT / "test/manual/kernels/bench_flashinfer_v1_v2_corrected.py"
    require(
        data["schema_version"] == 2
        and data["study"] == "mlp_v1_v2_corrected"
        and data["status"] == "success",
        "complete successful matched worker",
    )
    require(
        data["assignment"] == entry
        and data["trial"] == entry["trial"]
        and data["arm"] == entry["arm"]
        and data["phase"] == ("warm" if phase == "warm" else "cold")
        and data["mode"] == ("timing" if phase == "timing" else "admission"),
        "exact treatment and process assignment",
    )
    require(
        data["worker_sha256"] == sha(worker)
        and data["plan_sha256"] == sha(HERE / "plan.json")
        and data["schedule_sha256"] == sha(HERE / "schedule.json"),
        "frozen worker and protocol identity",
    )
    require(
        data["parent_worker_sha256"]
        == sha(ROOT / "test/manual/kernels/bench_flashinfer_mlp_selection.py"),
        "unchanged qualified MLP parent",
    )
    require(
        data["source"]["critical_sha256"] == plan["source_contract"]["runtime_sha256"]
        and data["source"]["flashinfer_commit"]
        == plan["source_contract"]["flashinfer_commit"]
        and data["source"]["native_identity_passthrough"] is True,
        "qualified FlashInfer runtime source",
    )
    for name, expected in data["source"]["sglang_source_sha256"].items():
        require(sha(ROOT / name) == expected, "actual SGLang source " + name)
    helpers = data["source"]["manual_helpers_sha256"]
    require(
        set(helpers)
        == {
            "bench_flashinfer_objective_values.py",
            "flashinfer_autotune_linear_layer.py",
            "flashinfer_autotune_descriptor_selection.py",
        },
        "complete inherited runtime helper inventory",
    )
    for name, expected in helpers.items():
        require(
            sha(ROOT / "test/manual/kernels" / name) == expected,
            "executed helper " + name,
        )
    layer = data["actual_layer"]
    require(
        {key: layer[key] for key in ("function", "custom_op", "quantizer", "backend")}
        == {
            "function": "apply_fp8_linear_bmm_flashinfer",
            "custom_op": "flashinfer_bmm_fp8",
            "quantizer": "static_quant_fp8",
            "backend": "cublas",
        },
        "actual SGLang FP8 facade",
    )
    require(
        layer["inference_mode"] is True
        and layer["custom_op_direct_wrapper"] is True
        and layer["quantization_reference"]["byte_match"] is True
        and not any(
            layer[key]
            for key in (
                "grad_enabled",
                "allow_tf32",
                "initializer_override",
                "dispatch_modes",
                "function_modes",
                "torch_profiler_enabled",
                "compiling",
                "kernel_api_log_level",
            )
        ),
        "unchanged normal inference and quantization path",
    )
    fixture = data["fixture"]
    require(
        fixture["manifest_sha256"] == plan["fixture_manifest_sha256"]
        and fixture["tensor_file_sha256"] == plan["fixture_sha256"]
        and fixture["shape"] == data["shape"] == [144, 1024, 3072]
        and data["seed"] == 19,
        "frozen real MLP fixture",
    )
    require(
        all(
            fixture["capture"][key] is True
            for key in (
                "activation_byte_match",
                "checkpoint_tensor_byte_match",
                "full_bf16_forward_completed",
            )
        ),
        "actual fixture capture provenance",
    )
    environment = data["environment"]
    require(
        environment["torch"] == plan["environment"]["torch"]
        and environment["cuda"] == "13.0"
        and environment["gpu"] == plan["environment"]["gpu"]
        and environment["gpu_uuid"].removeprefix("GPU-") == plan["environment"]["uuid"],
        "same qualified physical GPU and numerical stack",
    )
    require(
        data["all_samples_retained"] is True
        and data["queue_query_is_exclusion_criterion"] is False
        and data["persistence_under_test"] is True
        and data.get("pending_sample") is None,
        "complete unfiltered evidence",
    )
    boot = validate_bootstrap(path, data, plan, raw_root, worker)
    groups = validate_startup(data)
    validate_protocol(data, plan, contract)
    checked_entry = {**entry, "phase": data["phase"]}
    checks, graphs, capture_end = validate_outputs(data, checked_entry)
    modes, final_end = validate_measurements(
        data, checked_entry, schedule, checks, graphs, capture_end
    )
    trace_path = path.with_suffix(".trace.json")
    require(
        data["trace"]["sha256"] == sha(trace_path)
        and data["trace"]["started_ns"] > final_end,
        "post-serving trace identity",
    )
    require(
        {
            row["id"]: (row["graph_object_id"], row["output_data_ptr"])
            for row in data["trace"]["graphs"]
        }
        == {
            role: (graph["graph_object_id"], graph["output"]["data_ptr"])
            for role, graph in graphs.items()
        },
        "trace observes the original retained graphs",
    )
    trace_helper = HERE.parent / "20261024T180000Z-h100-objective-values/analyze.py"
    signatures = legacy.trace_signatures(trace_path, trace_helper)
    ranked = sorted(
        (row for row in data["startup"]["profiles"] if row["m"] == 144),
        key=lambda row: row["profiler_ms"],
    )
    ranking = [
        {
            "rank": rank + 1,
            "profile_index": row["index"],
            "tactic": row["tactic"],
            "profiler_ms": row["profiler_ms"],
            "over_fastest": row["profiler_ms"] / ranked[0]["profiler_ms"],
        }
        for rank, row in enumerate(ranked)
    ]
    return {
        "trial": data["trial"],
        "arm": data["arm"],
        "phase": data["phase"],
        "mode": data["mode"],
        "capture_order": entry["capture_order"],
        "mode_order": entry["mode_order"],
        "natural_tactic": data["natural"]["tactic"],
        "natural_key": data["natural"]["key"],
        "target_candidate_ranking": ranking,
        "target_runner_up_over_winner": ranking[1]["over_fastest"] if ranking else None,
        "profile_groups": groups,
        "profile_attempts": len(data["startup"]["profiles"]),
        "startup_ms": positive(data["startup"]["startup_ms"]),
        "cache_sha256": data["cache"]["final_sha256"],
        "correctness_checks": len(checks),
        "modes": modes,
        "trace_signatures": signatures,
        "correction": boot["binding_before"],
        "result_sha256": sha(path),
        "bootstrap_sha256": sha(path.with_suffix(".bootstrap.json")),
    }


def summarize(processes, resamples=RESAMPLES):
    require(len(processes) == 16, "complete paired cohort required for inference")
    indexed = {(row["trial"], row["arm"]): row for row in processes}
    require(
        set(indexed) == {(block, arm) for block in range(BLOCKS) for arm in ARMS},
        "each paired cell exactly once",
    )
    require(
        all(row["phase"] == "cold" and row["mode"] == "timing" for row in processes),
        "admission and warm results excluded from estimates",
    )
    metrics, pairs = {}, []
    for block in range(BLOCKS):
        v1, v2 = (indexed[block, arm] for arm in ARMS)
        require(
            all(v1[key] == v2[key] for key in ("capture_order", "mode_order")),
            "matched graph and mode histories",
        )
        pairs.append(
            {
                "block": block,
                "first_arm": "v1" if block % 2 == 0 else "v2",
                "capture_order": v1["capture_order"],
                "mode_order": v1["mode_order"],
                "v1_tactic": v1["natural_tactic"],
                "v2_tactic": v2["natural_tactic"],
                "same_tactic": v1["natural_tactic"] == v2["natural_tactic"],
                "same_natural_trace_signature": v1["trace_signatures"]["natural"][
                    "ordered_launch_signature"
                ]
                == v2["trace_signatures"]["natural"]["ordered_launch_signature"],
                "same_default_trace_signature": v1["trace_signatures"]["default"][
                    "ordered_launch_signature"
                ]
                == v2["trace_signatures"]["default"]["ordered_launch_signature"],
            }
        )
        for mode in MODES:
            values = [row["modes"][mode]["median_log_ratio"] for row in (v1, v2)]
            for arm, value in zip(ARMS, values, strict=True):
                metrics.setdefault(
                    arm + "." + mode + ".measured_natural_over_default", []
                ).append(value)
            metrics.setdefault(
                "v2_over_v1." + mode + ".measured_normalized", []
            ).append(values[1] - values[0])
            for role in ROLES:
                absolute = [
                    math.log(positive(row["modes"][mode]["median_absolute_ms"][role]))
                    for row in (v1, v2)
                ]
                for arm, value in zip(ARMS, absolute, strict=True):
                    metrics.setdefault(
                        arm + "." + mode + ".absolute_" + role + "_ms", []
                    ).append(value)
                metrics.setdefault(
                    "v2_over_v1." + mode + ".absolute_" + role, []
                ).append(absolute[1] - absolute[0])
        choice_scores = [
            0.0
            if row["natural_tactic"] == -1
            else row["modes"]["cuda_graph"]["median_log_ratio"]
            for row in (v1, v2)
        ]
        for arm, value in zip(ARMS, choice_scores, strict=True):
            metrics.setdefault(
                arm + ".choice_identity_graph_score_secondary", []
            ).append(value)
        metrics.setdefault(
            "v2_over_v1.choice_identity_graph_score_secondary", []
        ).append(choice_scores[1] - choice_scores[0])
    require(
        resamples > 0
        and all(
            len(values) == BLOCKS and all(math.isfinite(v) for v in values)
            for values in metrics.values()
        ),
        "finite complete paired metrics",
    )
    rng, draws = random.Random(SEED), {key: [] for key in metrics}
    for _ in range(resamples):
        indices = [rng.randrange(BLOCKS) for _ in range(BLOCKS)]
        for key, values in metrics.items():
            draws[key].append(math.exp(sum(values[i] for i in indices) / BLOCKS))
    estimates = {
        key: {
            "paired_blocks": BLOCKS,
            "geometric_mean": math.exp(statistics.mean(values)),
            "values_by_block": [math.exp(v) for v in values],
            "exploratory_paired_block_bootstrap_95pct": [
                legacy.quantile(draws[key], 0.025),
                legacy.quantile(draws[key], 0.975),
            ],
        }
        for key, values in metrics.items()
    }
    primary = "v2_over_v1.cuda_graph.measured_normalized"
    strata = {
        key: {
            value: {
                "blocks": [p["block"] for p in pairs if p[key] == value],
                "geometric_mean_ratio": math.exp(
                    statistics.mean(
                        metrics[primary][p["block"]] for p in pairs if p[key] == value
                    )
                ),
            }
            for value in sorted({p[key] for p in pairs})
        }
        for key in ("first_arm", "capture_order", "mode_order")
    }
    return {
        "primary": primary,
        "metrics": estimates,
        "paired_rows": pairs,
        "same_tactic_pairs": sum(row["same_tactic"] for row in pairs),
        "same_natural_trace_signature_pairs": sum(
            row["same_natural_trace_signature"] for row in pairs
        ),
        "primary_descriptive_strata": strata,
        "counts": {
            arm: {
                "default_winners": sum(
                    indexed[i, arm]["natural_tactic"] == -1 for i in range(BLOCKS)
                ),
                "explicit_winners": sum(
                    indexed[i, arm]["natural_tactic"] != -1 for i in range(BLOCKS)
                ),
            }
            for arm in ARMS
        },
        "startup_host_cost_ms": {
            arm: {
                "values_by_block": [
                    indexed[i, arm]["startup_ms"] for i in range(BLOCKS)
                ],
                "median": statistics.median(
                    indexed[i, arm]["startup_ms"] for i in range(BLOCKS)
                ),
            }
            for arm in ARMS
        },
        "bootstrap_seed": SEED,
        "bootstrap_resamples": resamples,
        "limits": [
            "Eight fresh-process pairs on one fixed GPU and fixture; intervals are exploratory, not equivalence or safety claims.",
            "Primary uses every actual graph ratio, including independent default/default graphs. Choice-identity normalization is a separate secondary diagnostic.",
            "Identical selected descriptors do not establish a candidate-selection benefit from noisy latency differences.",
            "Eager wall time is secondary; graph wall time includes prequeue delay and is diagnostic only.",
            "No confirmation, retuning, historical pooling, admission pooling, performance-sign exclusions or adaptive retries.",
        ],
    }


def validate_plan(plan_path):
    plan, schedule = load(plan_path), load(Path(plan_path).with_name("schedule.json"))
    require(
        plan["study"] == "mlp_v1_v2_corrected"
        and plan["schedule_sha256"] == sha(Path(plan_path).with_name("schedule.json")),
        "plan binds complete corrected comparison schedule",
    )
    protocol = plan["protocol"]
    require(
        protocol["blocks"] == 8
        and protocol["arms"] == list(ARMS)
        and protocol["admission_processes"] == protocol["warm_processes"] == 2
        and protocol["fresh_timing_processes"] == 16
        and protocol["ordering_processes"] == 1
        and protocol["total_processes"] == 21,
        "fixed bounded process counts",
    )
    require(
        plan["analysis"]["uncertainty"]["resamples"] == RESAMPLES
        and plan["analysis"]["uncertainty"]["seed"] == SEED
        and schedule["seed"] == SEED,
        "prespecified uncertainty recipe",
    )
    for relative, expected in plan["source_files_sha256"].items():
        path = ROOT / relative
        require(
            path.resolve().is_relative_to(ROOT) and sha(path) == expected,
            "frozen source " + relative,
        )
    contract_path = ROOT / plan["candidate_contract"]["path"]
    require(
        sha(contract_path) == plan["candidate_contract"]["sha256"],
        "candidate inventory contract unchanged",
    )
    validate_schedule(schedule)
    return plan, schedule, load(contract_path)


def validate_manifest(raw_root, plan_path, plan):
    manifest = load(raw_root / "manifest.json")
    require(
        manifest["master_plan_sha256"] == sha(plan_path)
        and manifest["master_schedule_sha256"]
        == sha(Path(plan_path).with_name("schedule.json"))
        and manifest["fixed_source_sha256"] == sha(raw_root / "fixed-source.json"),
        "execution manifest binds source and protocol",
    )
    require(
        re.fullmatch("[0-9a-f]{40}", manifest["execution_commit"]) is not None
        and load(raw_root / "source-snapshot.json")["source_commit"]
        == manifest["execution_commit"],
        "explicit execution source identity",
    )
    for relative, expected in manifest["source_sha256"].items():
        path = ROOT / relative
        require(
            path.resolve().is_relative_to(ROOT) and sha(path) == expected,
            "actual executed source " + relative,
        )
    for name, expected in manifest["batch_files_sha256"].items():
        require(
            Path(name).name == name and sha(HERE / name) == expected,
            "executed batch artifact " + name,
        )
    fixed = load(raw_root / "fixed-source.json")
    require(
        fixed["status"] == "passed"
        and fixed["binding_arm"] == "patched"
        and fixed["same_input_tree_for_both_arms"] is True
        and fixed["wrapper_sha256"] == plan["binding"]["patched_sha256"],
        "both arms use exactly the same corrected source tree",
    )
    require(
        sha(raw_root / "v1/native-build-cache/reuse-manifest.json")
        == sha(raw_root / "v2/native-build-cache/reuse-manifest.json"),
        "same complete source/native inventory in both arms",
    )
    return manifest


def validate_ledger(ledger, entries, raw_root, plan_path, phase):
    require(
        ledger["schema_version"] == 1
        and ledger["master_plan_sha256"] == sha(plan_path)
        and ledger["master_schedule_sha256"]
        == sha(Path(plan_path).with_name("schedule.json")),
        "launch ledger protocol binding",
    )
    require(
        phase in ("ordering", "admission", "warm", "timing"), "known cumulative phase"
    )
    phases = (
        ()
        if phase == "ordering"
        else (("admission",) if phase == "admission" else ("admission", "warm"))
    )
    expected = [("ordering", "patched", None, None)] + [
        (label, arm, None, None) for label in phases for arm in ARMS
    ]
    if phase == "timing":
        expected += [
            ("timing", row["arm"], row["block"], row["sequence"]) for row in entries
        ]
    require(
        len(ledger["entries"]) == len(expected),
        "all required launches and no extra attempts",
    )
    previous, containers = -1, set()
    for row, identity in zip(ledger["entries"], expected, strict=True):
        require(
            tuple(row[k] for k in ("phase", "arm", "block", "sequence")) == identity,
            "exact phase and paired launch order",
        )
        require(
            type(row["started_ns"]) is type(row["ended_ns"]) is int
            and previous < row["started_ns"] < row["ended_ns"],
            "sequential fresh-process launches",
        )
        previous = row["ended_ns"]
        cid = row["container_id"]
        require(
            isinstance(cid, str)
            and re.fullmatch("[0-9a-f]{64}", cid) is not None
            and cid not in containers,
            "unique full container identity",
        )
        containers.add(cid)
        label, arm, block, _ = identity
        name = (
            f"{arm}/block{block}-auto_synthetic-timing.json"
            if label == "timing"
            else f"{arm}/{label}-auto_synthetic.json"
        )
        if label == "ordering":
            name = "ordering/patched.json"
        path = raw_root / name
        require(
            row["result_relative_path"] == name
            and row["returncode"] == 0
            and row["cleanup_error"] is None
            and path.with_suffix(".exit-code").read_text().strip() == "0"
            and path.with_suffix(".container-id").read_text().strip() == cid,
            "successful canonical worker and cleanup",
        )
        require(
            row["result_sha256"] == sha(path)
            and row["bootstrap_sha256"] == sha(path.with_suffix(".bootstrap.json")),
            "exact result and bootstrap retained",
        )
        if label == "ordering":
            require(
                all(
                    row[key] is None
                    for key in (
                        "cache_relative_path",
                        "cache_sha256",
                        "cache_seed_relative_path",
                        "cache_seed_sha256",
                    )
                ),
                "ordering has no tuner-cache artifact",
            )
            continue
        cache = load(path)["cache"]
        require(
            row["cache_relative_path"] == str(Path(name).with_suffix(".cache.json"))
            and row["cache_sha256"]
            == sha(path.with_suffix(".cache.json"))
            == cache["after_startup_sha256"]
            == cache["final_sha256"],
            "original v1 winner cache bytes retained",
        )
        if label == "warm":
            seed = f"{arm}/admission-auto_synthetic.cache.json"
            require(
                row["cache_seed_relative_path"] == seed
                and row["cache_seed_sha256"]
                == sha(raw_root / seed)
                == cache["before_sha256"],
                "fresh warm process seeded only from its admission arm",
            )
        else:
            require(
                row["cache_seed_relative_path"] is None
                and row["cache_seed_sha256"] is None,
                "cold runs have no seeded cache",
            )
    return ledger["entries"]


def dependency_identity(witness):
    return {
        key: value
        for key, value in witness.items()
        if key not in ("torch_stream", "ambient_ffi_stream")
    }


def validate_fixed_ordering(raw_root, plan, plan_path, manifest):
    path = raw_root / "ordering/patched.json"
    data = load(path)
    require(
        data["schema_version"] == 1
        and data["status"] == "passed"
        and data["arm"] == "patched"
        and data["upstream_head"] == plan["binding"]["upstream_head"],
        "successful fixed-only ordering qualification",
    )
    worker = ROOT / "test/manual/kernels/flashinfer_autotune_delay_binding.py"
    binding = module(worker, "matched_fixed_ordering")
    binding.validate_ordering(data)
    boot = validate_bootstrap(path, data, plan, raw_root, worker, ordering=True)
    return {
        "schema_version": 1,
        "status": "passed",
        "phase": "ordering",
        "processes": 1,
        "correctness_checks": len(data["actual_profiler"]["samples"])
        + len(data["graph_replays"]),
        "result_sha256": sha(path),
        "bootstrap_sha256": sha(path.with_suffix(".bootstrap.json")),
        "plan_sha256": sha(plan_path),
        "schedule_sha256": sha(Path(plan_path).with_name("schedule.json")),
        "execution_commit": manifest["execution_commit"],
        "attestation": boot["binding_before"],
    }


def analyze(args):
    plan, schedule, contract = validate_plan(args.plan)
    manifest = validate_manifest(args.raw_root, args.plan, plan)
    ledger = validate_ledger(
        load(args.raw_root / "launch-ledger.json"),
        schedule["entries"],
        args.raw_root,
        args.plan,
        args.phase,
    )
    ordering = validate_fixed_ordering(args.raw_root, plan, args.plan, manifest)
    if args.phase == "ordering":
        return ordering
    require(
        load(args.raw_root / "fixed-ordering-analysis.json") == ordering,
        "exact rederived ordering gate",
    )
    processes, witnesses = [], [dependency_identity(ordering["attestation"])]
    for row in ledger[1:]:
        trial = row["block"] if row["phase"] == "timing" else 0
        entry = next(
            e
            for e in schedule["entries"]
            if e["trial"] == trial and e["arm"] == row["arm"]
        )
        process = validate_worker(
            args.raw_root / row["result_relative_path"],
            entry,
            row["phase"],
            plan,
            schedule,
            contract,
            args.raw_root,
        )
        process["cohort_phase"] = row["phase"]
        witness = process.pop("correction")
        witnesses.append(dependency_identity(witness))
        processes.append(process)
    require(
        all(row == witnesses[0] for row in witnesses),
        "identical fixed wrapper, FFI and native dependencies across every process",
    )
    indexed = {
        (row["cohort_phase"], row["trial"], row["arm"]): row for row in processes
    }
    for phase, trials in (
        ("admission", [0]),
        ("timing", range(BLOCKS) if args.phase == "timing" else []),
    ):
        for trial in trials:
            v1, v2 = (indexed[phase, trial, arm] for arm in ARMS)
            validate_matched_candidates(
                v1["profile_groups"], v2["profile_groups"], phase == "admission"
            )
    if args.phase != "admission":
        for arm in ARMS:
            cold, warm = indexed["admission", 0, arm], indexed["warm", 0, arm]
            require(
                all(
                    cold[key] == warm[key]
                    for key in ("natural_tactic", "natural_key", "cache_sha256")
                )
                and warm["profile_attempts"] == 0,
                "actual cold admission winner persists into fresh warm serving",
            )
    result = {
        "status": "passed",
        "phase": args.phase,
        "analyzer_sha256": sha(__file__),
        "execution_commit": manifest["execution_commit"],
        "plan_sha256": sha(args.plan),
        "schedule_sha256": sha(Path(args.plan).with_name("schedule.json")),
        "results": processes,
        "correctness_checks": sum(row["correctness_checks"] for row in processes),
        "processes": len(ledger),
        "mlp_processes": len(processes),
        "ordering_qualification": ordering,
        "fresh_warm_persistence_verified": args.phase != "admission",
    }
    if args.phase == "timing":
        result.update(
            summarize([row for row in processes if row["cohort_phase"] == "timing"])
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("ordering", "admission", "warm", "timing"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refuse to overwrite analysis")
    result = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "phase", "processes", "correctness_checks")
            }
        )
    )


if __name__ == "__main__":
    main()
