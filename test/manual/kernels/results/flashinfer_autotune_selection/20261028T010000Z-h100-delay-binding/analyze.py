#!/usr/bin/env python3
"""Validate the fixed delay-binding cohort and compare complete paired blocks.

The unchanged historical analyzer validates each raw MLP process. New inference
uses only this cohort, pairs the two arms by block, and keeps startup host costs
separate from the post-confirmation serving-graph score.
"""

from __future__ import annotations

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
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
ARMS = ("baseline", "patched")
BLOCKS = 8
RESAMPLES = 100_000
SEED = 202610280100
INHERITED = HERE.parent / "20261027T020000Z-h100-mlp-confirmation"
INHERITED_ANALYZER_SHA = (
    "1fa9a5ea7312dc4fb3e5fb5766c95fc9c64b5ac9d91095a59a72655f3847542e"
)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


legacy = module(INHERITED / "analyzer.py", "delay_binding_mlp_analysis")
require, load, sha256 = legacy.require, legacy.load, legacy.sha256
positive = legacy.positive
quantile = legacy.parent.quantile


def validate_schedule(schedule, inherited):
    require(
        schedule["schema_version"] == 1
        and schedule["seed"] == SEED
        and schedule["blocks"] == BLOCKS,
        "fixed paired-cohort schedule",
    )
    expected = []
    for block, original in enumerate(inherited["entries"]):
        for position, arm in enumerate(ARMS if block % 2 == 0 else ARMS[::-1]):
            expected.append(
                {
                    "sequence": len(expected),
                    "block": block,
                    "arm": arm,
                    "position": position,
                    "cell": original["cell"],
                    "capture_order": original["capture_order"],
                }
            )
    require(
        len(expected) == 16 and schedule["entries"] == expected,
        "all eight pairs use alternating launch order and unchanged within-worker schedules",
    )
    return expected


def validate_plan(path):
    plan = load(path)
    inherited = plan["inherited"]
    require(ROOT / inherited["batch"] == INHERITED, "fixed inherited batch")
    require(
        sha256(INHERITED / "analyzer.py") == INHERITED_ANALYZER_SHA,
        "unchanged per-arm integrity analyzer",
    )
    for name, expected in inherited["files_sha256"].items():
        require(
            name == Path(name).name and sha256(INHERITED / name) == expected,
            "inherited artifact unchanged: " + name,
        )
    require(
        set(inherited["files_sha256"])
        == {
            "plan.json",
            "schedule.json",
            "analysis-protocol.json",
            "historical-contract.json",
            "analyzer.py",
            "outer_audit.py",
            "run_current.py",
        },
        "complete inherited artifact inventory",
    )
    worker = ROOT / inherited["worker"]
    require(sha256(worker) == inherited["worker_sha256"], "unchanged MLP worker")
    old_plan = load(INHERITED / "plan.json")
    protocol = plan["protocol"]
    require(
        protocol["blocks"] == BLOCKS
        and protocol["fresh_timing_processes"] == 16
        and protocol["arms"] == list(ARMS)
        and protocol["admission_processes"] == protocol["ordering_processes"] == 2,
        "fixed admission, ordering and timing process counts",
    )
    for key, value in protocol["timing_settings"].items():
        require(value == old_plan["protocol"][key], "inherited setting: " + key)
    require(
        plan["analysis"]["uncertainty"]["resamples"] == RESAMPLES
        and plan["analysis"]["uncertainty"]["seed"] == SEED,
        "frozen paired-block uncertainty recipe",
    )
    schedule_path = Path(path).parent / "schedule.json"
    require(sha256(schedule_path) == plan["schedule_sha256"], "master schedule hash")
    schedule = load(schedule_path)
    validate_schedule(schedule, load(INHERITED / "schedule.json"))
    return plan, schedule, worker


def augment_process(row, raw):
    """Add normalized decisions and host costs without changing raw evidence."""
    result = dict(row)
    original = int(row["original_role"] == "winner")
    confirmed = int(row["confirmed_role"] == "winner")
    graph = row["graph_log_ratio"]
    result.update(
        original_selected_log_over_default=original * graph,
        original_selected_over_default=math.exp(original * graph),
        confirmed_selected_log_over_default=confirmed * graph,
        confirmed_selected_over_default=math.exp(confirmed * graph),
        startup_explicit_faster_confirmation_explicit_slower=(
            row["original_profiler_ratio"] < 1 and row["confirmation_log_ratio"] > 0
        ),
        startup_explicit_slower_confirmation_explicit_faster=(
            row["original_profiler_ratio"] > 1 and row["confirmation_log_ratio"] < 0
        ),
        startup_explicit_faster_graph_explicit_slower=(
            row["original_profiler_ratio"] < 1 and graph > 0
        ),
        startup_profile_point_tie=row["original_profiler_ratio"] == 1,
        confirmation_point_tie=row["confirmation_log_ratio"] == 0,
    )
    for phase in ("startup", "confirmation"):
        interval = raw[phase]
        elapsed = positive(
            (interval["ended_ns"] - interval["started_ns"]) / 1e6,
            "positive " + phase + " host interval",
        )
        result[phase + "_host_ms"] = elapsed
    calls = raw["diagnostic_profile_attempts"]
    result["confirmation_profiler_host_ms"] = sum(
        positive(
            (call["ended_ns"] - call["started_ns"]) / 1e6,
            "positive confirmation profiler host interval",
        )
        for call in calls
    )
    default_logs = [
        math.log(positive(m["gpu_event_per_replay_ms"], "default graph duration"))
        for pair in raw["rounds"]
        for m in pair["measurements"]
        if m["graph_id"] == "default"
    ]
    require(len(default_logs) == 24, "all default graph windows retained")
    result["default_graph_log_ms"] = statistics.median(default_logs)
    result["default_graph_ms"] = math.exp(result["default_graph_log_ms"])
    return result


def paired_metrics(processes, entries):
    require(set(processes) == set(ARMS), "both comparison arms present")
    for arm in ARMS:
        require(
            len(processes[arm]) == BLOCKS
            and [r["block"] for r in processes[arm]] == list(range(BLOCKS)),
            "exactly eight complete processes in " + arm,
        )
        for row in processes[arm]:
            for decision in ("original", "confirmed"):
                role = row[decision + "_role"]
                require(role in ("winner", "default"), "valid selected role")
                expected = int(role == "winner") * row["graph_log_ratio"]
                require(
                    row[decision + "_selected_log_over_default"] == expected
                    and row[decision + "_selected_over_default"] == math.exp(expected),
                    "normalized score follows actual frozen decision",
                )
            effect = legacy.choice_effect(
                row["original_role"], row["confirmed_role"], row["graph_log_ratio"]
            )
            require(
                all(row[key] == value for key, value in effect.items()),
                "choice comparison follows original and hypothetical decisions",
            )
    metrics = {}
    fields = {
        "original_selected_over_default": "original_selected_log_over_default",
        "confirmed_selected_over_default": "confirmed_selected_log_over_default",
        "confirmation_over_original": "log_hypothetical_over_original",
        "default_graph_absolute_ms": "default_graph_log_ms",
    }
    pairs = []
    for block in range(BLOCKS):
        baseline, patched = (processes[arm][block] for arm in ARMS)
        require(
            baseline["capture_order"] == patched["capture_order"],
            "paired workers use same graph capture order",
        )
        pair = {
            "block": block,
            "first_arm": entries[2 * block]["arm"],
            "capture_order": baseline["capture_order"],
            "baseline_finalist": baseline["finalist"],
            "patched_finalist": patched["finalist"],
            "same_finalist": baseline["finalist"] == patched["finalist"],
            "log_patched_over_baseline": {},
        }
        for label, field in fields.items():
            values = [baseline[field], patched[field]]
            require(all(math.isfinite(v) for v in values), "finite paired metric")
            for arm, value in zip(ARMS, values, strict=True):
                metrics.setdefault(arm + "." + label, []).append(value)
            delta = patched[field] - baseline[field]
            metrics.setdefault("patched_over_baseline." + label, []).append(delta)
            pair["log_patched_over_baseline"][label] = delta
        pairs.append(pair)
    return metrics, pairs


def interval_metrics(metrics, resamples=RESAMPLES):
    require(resamples > 0, "positive bootstrap count")
    require(
        all(len(values) == BLOCKS for values in metrics.values()),
        "bootstrap resamples complete paired blocks",
    )
    draws, rng = {key: [] for key in metrics}, random.Random(SEED)
    for _ in range(resamples):
        # Resample a whole pair, using identical indices for every endpoint.
        indices = [rng.randrange(BLOCKS) for _ in range(BLOCKS)]
        for key, values in metrics.items():
            draws[key].append(math.exp(sum(values[i] for i in indices) / BLOCKS))
    return {
        key: {
            "paired_blocks": BLOCKS,
            "unit": "ms"
            if key in {arm + ".default_graph_absolute_ms" for arm in ARMS}
            else "ratio",
            "geometric_mean": math.exp(statistics.mean(values)),
            "values_by_block": [math.exp(value) for value in values],
            "min": math.exp(min(values)),
            "max": math.exp(max(values)),
            **(
                {}
                if key in {arm + ".default_graph_absolute_ms" for arm in ARMS}
                else {
                    "below_one": sum(value < 0 for value in values),
                    "above_one": sum(value > 0 for value in values),
                    "equal_one": sum(value == 0 for value in values),
                }
            ),
            "exploratory_paired_block_bootstrap_95pct": [
                quantile(draws[key], 0.025),
                quantile(draws[key], 0.975),
            ],
        }
        for key, values in metrics.items()
    }


def summarize(processes, entries, resamples=RESAMPLES):
    metrics, pairs = paired_metrics(processes, entries)
    counts, costs, strata = {}, {}, {}
    for arm, rows in processes.items():
        counts[arm] = {
            "startup_default_winners": sum(
                r["original_role"] == "default" for r in rows
            ),
            "confirmed_default_winners": sum(
                r["confirmed_role"] == "default" for r in rows
            ),
            "startup_explicit_faster_confirmation_explicit_slower": sum(
                r["startup_explicit_faster_confirmation_explicit_slower"] for r in rows
            ),
            "startup_explicit_slower_confirmation_explicit_faster": sum(
                r["startup_explicit_slower_confirmation_explicit_faster"] for r in rows
            ),
            "startup_explicit_faster_graph_explicit_slower": sum(
                r["startup_explicit_faster_graph_explicit_slower"] for r in rows
            ),
            "startup_profile_point_ties": sum(
                r["startup_profile_point_tie"] for r in rows
            ),
            "confirmation_point_ties": sum(r["confirmation_point_tie"] for r in rows),
            "transitions": dict(collections.Counter(r["transition"] for r in rows)),
            **legacy.slow_choice_counts(rows),
        }
        costs[arm] = {
            key: {
                "values_by_block": [r[key] for r in rows],
                "median_ms": statistics.median(r[key] for r in rows),
                "min_ms": min(r[key] for r in rows),
                "max_ms": max(r[key] for r in rows),
            }
            for key in (
                "startup_host_ms",
                "confirmation_profiler_host_ms",
                "confirmation_host_ms",
            )
        }
    for field in ("first_arm", "capture_order"):
        strata[field] = {}
        for value in sorted({pair[field] for pair in pairs}):
            group = [pair for pair in pairs if pair[field] == value]
            strata[field][value] = {
                "blocks": [pair["block"] for pair in group],
                "geometric_mean_patched_over_baseline": {
                    key: math.exp(
                        statistics.mean(
                            pair["log_patched_over_baseline"][key] for pair in group
                        )
                    )
                    for key in pairs[0]["log_patched_over_baseline"]
                },
            }
    return {
        "metrics": interval_metrics(metrics, resamples),
        "counts": counts,
        "cold_startup_and_confirmation_host_costs": costs,
        "process_rows": processes,
        "paired_rows": pairs,
        "strata": strata,
        "bootstrap_resamples": resamples,
        "bootstrap_seed": SEED,
        "inference_unit": "Eight complete paired blocks on one GPU/fixture; all within-process repetitions retained.",
        "limits": [
            "Original and hypothetical choices share post-confirmation graphs; this is not untouched startup or full-model serving.",
            "Baseline and patched finalists can differ. Normalized scores compare the declared protocols, not one fixed kernel.",
            "Absolute default graph timing is a context diagnostic, not an isolated candidate-selection effect.",
            "Directional reversals and slower-choice counts use point estimates without significance thresholds or margins.",
            "No historical/admission pooling, performance-sign exclusions, retries or adaptive sample expansion.",
            "Exploratory intervals from eight pairs do not establish equivalence, universal safety or a serving speedup.",
        ],
        "automatic_production_change_or_pr": False,
    }


def validate_ledger(ledger, entries, raw_root, plan_path, admission):
    require(
        ledger["schema_version"] == 1
        and ledger["master_plan_sha256"] == sha256(plan_path)
        and ledger["master_schedule_sha256"]
        == sha256(Path(plan_path).parent / "schedule.json"),
        "launch ledger binds the master protocol",
    )
    expected = [
        (phase, arm, None, None) for phase in ("ordering", "admission") for arm in ARMS
    ]
    if not admission:
        expected += [("timing", r["arm"], r["block"], r["sequence"]) for r in entries]
    require(
        len(ledger["entries"]) == len(expected),
        "every required launch and no extra attempts",
    )
    containers, previous = set(), -1
    for row, assignment in zip(ledger["entries"], expected, strict=True):
        require(
            tuple(row[k] for k in ("phase", "arm", "block", "sequence")) == assignment,
            "actual phase and paired launch order",
        )
        require(
            type(row["started_ns"]) is int
            and type(row["ended_ns"]) is int
            and previous < row["started_ns"] < row["ended_ns"],
            "strictly sequential nonoverlapping container launches",
        )
        previous = row["ended_ns"]
        cid = row["container_id"]
        require(
            isinstance(cid, str)
            and re.fullmatch("[0-9a-f]{64}", cid) is not None
            and cid not in containers,
            "fresh container identity across every phase and arm",
        )
        containers.add(cid)
        phase, arm, block, _ = assignment
        name = (
            f"ordering/{arm}.json"
            if phase == "ordering"
            else f"{arm}/admission-auto_synthetic.json"
            if phase == "admission"
            else f"{arm}/block{block}-auto_synthetic-timing.json"
        )
        require(row["result_relative_path"] == name, "canonical arm/phase result path")
        path = raw_root / name
        require(
            row["returncode"] == 0
            and path.with_suffix(".exit-code").read_text().strip() == "0"
            and path.with_suffix(".container-id").read_text().strip() == cid
            and row["result_sha256"] == sha256(path)
            and row["bootstrap_sha256"] == sha256(path.with_suffix(".bootstrap.json")),
            "successful launch binds exact raw result and bootstrap",
        )
        if phase != "ordering":
            raw = load(path)
            require(
                row["cache_sha256"]
                == sha256(path.with_suffix(".cache.json"))
                == raw["startup"]["cache_sha256"]
                == raw["final"]["cache_sha256"],
                "original v1 winner cache retained unchanged as an artifact",
            )
    return ledger["entries"]


def validate_binding(witness, arm, ordering_witness, binding):
    require(
        witness["arm"] == arm
        and witness["upstream_head"] == binding.UPSTREAM_HEAD
        and witness["wrapper_sha256"] == binding.wrapper_sha(arm)
        and witness["wrapper_path"]
        == "/study/input/current/source/flashinfer/tllm_utils.py"
        and witness["callable_matches_source"] is True
        and witness["profiler_alias_matches"] is True
        and witness["native_identity_passthrough"] is True,
        "actual imported wrapper and profiler alias match the assigned arm",
    )
    require(
        {name: value["library_sha256"] for name, value in witness["native"].items()}
        == binding.NATIVE_HASHES,
        "exact same qualified native providers",
    )
    stable = lambda record: {
        key: value
        for key, value in record.items()
        if key not in ("torch_stream", "ambient_ffi_stream")
    }
    require(
        stable(witness) == stable(ordering_witness),
        "qualified wrapper, FFI, native and dependency identity carried into MLP",
    )


def validate_bootstrap(path, arm, plan, raw_root, ordering_witness, binding):
    raw, boot = load(path), load(path.with_suffix(".bootstrap.json"))
    require(
        boot["status"] == "passed"
        and boot["binding_arm"] == arm
        and boot["pid"] == raw["pid"]
        and boot["result_sha256"] == sha256(path)
        and boot["worker_sha256"] == sha256(binding.__file__),
        "outer bootstrap binds same-process helper and result",
    )
    require(
        boot["binding_before"] == boot["binding_after"], "binding restored after worker"
    )
    validate_binding(boot["binding_before"], arm, ordering_witness, binding)
    reference_path = raw_root / "native-reference.bootstrap.json"
    require(
        sha256(reference_path)
        == "7c4727aca3d99adcb3fe2f35d06f4408b512baa4fe9a91b975bdf77358fe205d",
        "immutable qualified native reference",
    )
    reference = load(reference_path)
    reuse_path = raw_root / arm / "native-build-cache/reuse-manifest.json"
    reuse = load(reuse_path)
    require(
        reuse["status"] == "passed"
        and reuse["native_reference_sha256"] == sha256(reference_path)
        and boot["native_reuse"]["reference_sha256"] == sha256(reference_path)
        and boot["native_reuse"]["manifest_sha256"] == sha256(reuse_path),
        "exact source/native reuse record",
    )
    for point in ("before", "after_build", "after_worker"):
        guard = boot["native_reuse"][point]
        require(
            guard["status"] == "passed"
            and guard["elf_sha256"] == plan["native_reuse"]["required_library_sha256"]
            and guard["files_verified"] == len(reuse["native_files"]),
            "native bytes verified " + point,
        )
    runs = boot["native_reuse"]["ninja_dry_runs"]
    require(
        [r["name"] for r in runs] == ["gemm", "trtllm_utils"],
        "both native builders checked",
    )
    for row in runs:
        directory = (
            "/study/runtime-attempt01/cache/current/.cache/flashinfer/0.0.0+unknown/90a/cached_ops/"
            + row["name"]
        )
        require(
            row["returncode"] == 0
            and row["stdout"]
            == f"ninja: Entering directory `{directory}'\nninja: no work to do.\n",
            "no implicit native rebuild",
        )
    for name, native in reference["native"]["providers"].items():
        actual = boot["native"]["providers"][name]
        for field in (
            "library_path",
            "library_sha256",
            "ninja_path",
            "ninja_sha256",
            "sources",
            "compile_commands",
        ):
            require(
                actual[field] == native[field],
                "native provider unchanged: " + name + "." + field,
            )
    require(
        boot["native"]["tvm_ffi_sha256"] == reference["native"]["tvm_ffi_sha256"]
        and boot["native_identity_passthrough"] is True,
        "unmodified native binding and no proxy",
    )
    expected_import = copy.deepcopy(reference["import"])
    expected_import["source_modules"]["flashinfer.tllm_utils"]["sha256"] = (
        binding.wrapper_sha(arm)
    )
    require(
        boot["import"] == expected_import,
        "complete import inventory differs only in delay wrapper",
    )
    expected_source = copy.deepcopy(reference["source"])
    expected_source["source_sha256"]["flashinfer/tllm_utils.py"] = binding.wrapper_sha(
        arm
    )
    expected_source["declared_overlay"] = {
        "arm": arm,
        "path": "flashinfer/tllm_utils.py",
        "baseline_sha256": binding.wrapper_sha("baseline"),
        "actual_sha256": binding.wrapper_sha(arm),
    }
    require(
        boot["source"] == expected_source
        and boot["helper_sha256"] == reference["helper_sha256"]
        and boot["candidate_manifest_sha256"] == reference["candidate_manifest_sha256"],
        "explicit wrapper-only overlay over qualified source",
    )
    if path.parent.name != "ordering":
        bound = load(path.with_suffix(".binding.json"))
        require(
            bound["status"] == "passed"
            and bound["pid"] == raw["pid"]
            and bound["arm"] == arm
            and bound["upstream_head"] == binding.UPSTREAM_HEAD
            and bound["worker_sha256"] == plan["inherited"]["worker_sha256"]
            and bound["result_sha256"] == sha256(path)
            and bound["attestation_before"]
            == bound["attestation_after"]
            == boot["binding_before"],
            "unchanged inherited worker bracketed by same binding witness",
        )
    return {
        "result_sha256": sha256(path),
        "bootstrap_sha256": sha256(path.with_suffix(".bootstrap.json")),
    }


def validate_manifest(raw_root, plan_path):
    manifest = load(raw_root / "manifest.json")
    require(
        manifest["master_plan_sha256"] == sha256(plan_path)
        and manifest["master_schedule_sha256"]
        == sha256(Path(plan_path).parent / "schedule.json")
        and manifest["source_arms_sha256"] == sha256(raw_root / "source-arms.json"),
        "execution manifest binds current protocol and source arms",
    )
    for relative, expected in manifest["source_sha256"].items():
        path = ROOT / relative
        require(
            path.resolve().is_relative_to(ROOT) and sha256(path) == expected,
            "manual source unchanged: " + relative,
        )
    for name, expected in manifest["batch_files_sha256"].items():
        require(
            name == Path(name).name and sha256(HERE / name) == expected,
            "batch source unchanged: " + name,
        )
    for name in (
        "analyze.py",
        "plan.json",
        "schedule.json",
        "launch.py",
        "run_current.py",
    ):
        require(
            name in manifest["batch_files_sha256"],
            "required analysis/launch source attested",
        )
    require(
        "test/manual/kernels/flashinfer_autotune_delay_binding.py"
        in manifest["source_sha256"],
        "ordering and binding validator attested",
    )
    return manifest


def validate_source_arms(raw_root, binding):
    records = {
        arm: load(raw_root / arm / "native-build-cache/reuse-manifest.json")
        for arm in ARMS
    }
    relative = "source/flashinfer/tllm_utils.py"
    expected = copy.deepcopy(records["baseline"])
    expected["input_files"][relative] = binding.wrapper_sha("patched")
    expected["declared_overlay"].update(
        arm="patched", actual_sha256=binding.wrapper_sha("patched")
    )
    require(
        records["patched"] == expected,
        "entire source and native manifests differ only by the wrapper",
    )
    arms = load(raw_root / "source-arms.json")
    require(
        arms["schema_version"] == 1
        and arms["status"] == "passed"
        and arms["changed_files"] == [relative]
        and arms["native_files_identical"] is True
        and arms["helper_sha256"] == sha256(binding.__file__),
        "frozen staging comparison and helper identity",
    )
    for arm in ARMS:
        record = records[arm]
        require(
            record["status"] == "passed"
            and record["input_files"][relative] == binding.wrapper_sha(arm)
            and record["declared_overlay"]
            == {
                "arm": arm,
                "path": relative,
                "baseline_sha256": binding.wrapper_sha("baseline"),
                "actual_sha256": binding.wrapper_sha(arm),
            }
            and arms["arms"][arm]
            == {
                "wrapper_sha256": binding.wrapper_sha(arm),
                "reuse_manifest_sha256": sha256(
                    raw_root / arm / "native-build-cache/reuse-manifest.json"
                ),
                "input_files": len(record["input_files"]),
            },
            "complete arm inventory and declared source overlay",
        )


def analyze(args):
    plan, schedule, worker = validate_plan(args.plan)
    manifest = validate_manifest(args.raw_root, args.plan)
    binding = module(
        ROOT / "test/manual/kernels/flashinfer_autotune_delay_binding.py",
        "delay_binding_integrity",
    )
    require(
        plan["upstream"]["head"] == binding.UPSTREAM_HEAD
        and all(
            plan["binding"][arm + "_sha256"] == binding.wrapper_sha(arm) for arm in ARMS
        ),
        "plan and loaded binding validation agree",
    )
    validate_source_arms(args.raw_root, binding)
    ordering = binding.check_ordering(args.raw_root)
    require(
        ordering == load(args.raw_root / "ordering-gate.json"),
        "ordering gate independently reproduced",
    )
    ledger = load(args.raw_root / "launch-ledger.json")
    launches = validate_ledger(
        ledger, schedule["entries"], args.raw_root, args.plan, args.admission
    )
    provenance = {}
    for row in launches:
        arm, path = row["arm"], args.raw_root / row["result_relative_path"]
        provenance[row["result_relative_path"]] = validate_bootstrap(
            path,
            arm,
            plan,
            args.raw_root,
            ordering["arms"][arm]["attestation"],
            binding,
        )
    inputs = {
        "master_plan_sha256": sha256(args.plan),
        "master_schedule_sha256": plan["schedule_sha256"],
        "analyzer_sha256": sha256(__file__),
        "manifest_sha256": sha256(args.raw_root / "manifest.json"),
        "ordering_gate_sha256": sha256(args.raw_root / "ordering-gate.json"),
        "admission_launches_sha256": legacy.digest(launches[:4]),
        "admission_provenance": {
            r["result_relative_path"]: provenance[r["result_relative_path"]]
            for r in launches[:4]
        },
    }
    if not args.admission:
        admitted = load(args.raw_root / "admission-analysis.json")
        require(
            admitted["status"] == "passed"
            and admitted["mode"] == "admission"
            and admitted["inputs"] == inputs,
            "independent admission gate binds these exact inputs",
        )
    validated, processes = {}, {}
    for arm in ARMS:
        old_args = SimpleNamespace(
            plan=INHERITED / "plan.json",
            schedule=INHERITED / "schedule.json",
            historical_contract=INHERITED / "historical-contract.json",
            raw_root=args.raw_root / arm,
            worker=worker,
            prior_analyzer=HERE.parent / "20261026T220000Z-h100-fixed-pair/analyzer.py",
            admission=args.admission,
            admission_bootstrap=args.raw_root
            / arm
            / "admission-auto_synthetic.binding.json",
        )
        result = legacy.analyze(old_args)
        validated[arm] = {
            key: value
            for key, value in result.items()
            if key
            in (
                "status",
                "mode",
                "plan_sha256",
                "schedule_sha256",
                "historical_contract_sha256",
                "worker_sha256",
                "analyzer_sha256",
                "parent_confirmation_analyzer_sha256",
                "prior_analyzer_sha256",
                "leaf_helpers_sha256",
                "distinct_container_ids",
            )
        }
        if args.admission:
            processes[arm] = result["process_integrity"]
        else:
            processes[arm] = [
                augment_process(
                    row,
                    load(
                        args.raw_root
                        / arm
                        / f"block{row['block']}-auto_synthetic-timing.json"
                    ),
                )
                for row in result["process_rows"]
            ]
    result = {
        "schema_version": 1,
        "status": "passed",
        "mode": "admission" if args.admission else "timing",
        "inputs": inputs,
        "execution_commit": manifest["execution_commit"],
        "per_arm_integrity": validated,
        "process_integrity": processes if args.admission else None,
        "ordering_gate": ordering,
        "provenance": provenance,
        "performance_inference": not args.admission,
        "primary_metric": "patched_over_baseline.original_selected_over_default"
        if not args.admission
        else None,
    }
    if not args.admission:
        result.update(summarize(processes, schedule["entries"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "raw-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--admission", action="store_true")
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite an analysis")
    try:
        result = analyze(args)
    except Exception as error:
        result = {
            "status": "failed",
            "comparable": False,
            "performance_inference": False,
            "error": f"{type(error).__name__}: {error}",
        }
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        raise
    result["comparable"] = not args.admission
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
