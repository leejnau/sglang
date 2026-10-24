#!/usr/bin/env python3
"""Validate six complete replay-calibration processes without selecting samples."""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import itertools
import json
import math
import random
import re
import statistics
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

BATCH = "test/manual/kernels/results/flashinfer_autotune_selection/20261024T090000Z-h100-replay-calibration"
WORKER = "test/manual/kernels/bench_flashinfer_oracle_calibration.py"
HISTORY_HELPER = "test/manual/kernels/results/flashinfer_autotune_selection/20261024T080000Z-h100-selection-history/analyze.py"
HISTORY_HELPER_SHA256 = (
    "5eb00bfb7a68be662ba628b643149f49e4c7c1d0f283de5f6b3a99284f9e51f5"
)
METHODS = {
    "plain5120": {"replays": 5120, "prequeue_cycles": 0},
    "queued16_10m": {"replays": 16, "prequeue_cycles": 10_000_000},
    "queued16_40m": {"replays": 16, "prequeue_cycles": 40_000_000},
}
GRAPHS = ("graph_a", "graph_b")
TRIALS = tuple(range(6))


def pinned_history():
    path = Path(__file__).resolve().parents[6] / HISTORY_HELPER
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != HISTORY_HELPER_SHA256:
        raise ValueError("frozen history analysis helper changed")
    module = types.ModuleType("frozen_history_analysis")
    module.__file__ = str(path)
    exec(compile(content, str(path), "exec"), module.__dict__)
    return module


x = pinned_history()
o, h, g = x.o, x.h, x.g
require, load, canonical, sha256 = h.require, h.load, h.canonical, h.sha256
o.BATCH, o.WORKER = BATCH, WORKER


def expected_schedule(trial):
    rows = []
    for methods in itertools.permutations(METHODS):
        for occurrence in range(4):
            order = GRAPHS if occurrence % 2 == 0 else tuple(reversed(GRAPHS))
            rows.append({"methods": list(methods), "graph_order": list(order)})
    random.Random(7200 + trial).shuffle(rows)
    return rows


def expected_workers():
    return [
        {
            "trial": trial,
            "capture_order": "a-first" if trial % 2 == 0 else "b-first",
            "raw_name": f"trial{trial}-calibration.json",
            "trace_name": f"trial{trial}-calibration.trace.json",
        }
        for trial in TRIALS
    ]


def validate_plan(plan):
    require(plan["batch_id"] == Path(BATCH).name, "calibration batch")
    protocol = plan["protocol"]
    require(
        protocol["trials"] == list(TRIALS)
        and protocol["timing_workers"] == 6
        and protocol["worker"] == WORKER
        and protocol["worker_schedule"] == expected_workers(),
        "fixed six-process schedule",
    )
    require(
        protocol["methods"] == METHODS
        and protocol["rounds"] == 24
        and protocol["rewarm_ms"] == 20
        and protocol["initial_warmup_ms"] == 500,
        "fixed timing recipes",
    )
    require(
        protocol["round_schedules"]
        == [
            {
                "trial": trial,
                "seed": 7200 + trial,
                "rounds": [
                    {"index": index, **row}
                    for index, row in enumerate(expected_schedule(trial))
                ],
            }
            for trial in TRIALS
        ],
        "frozen balanced complete rounds",
    )
    require(
        plan["flashinfer_source"]["source_manifest_sha256"]
        == "3f4305fce604d3cd51e85047678c6efbf71e35bcf51037e9586d15bc20486895",
        "fixed qualified upstream source",
    )


def validate_outer(args):
    result = o.validate_outer(args)
    validate_plan(result["plan"])
    sources = load(args.raw_root / "manifest.json")["source_sha256"]
    require(
        sources.get(HISTORY_HELPER) == HISTORY_HELPER_SHA256
        and sources.get(x.OBJECTIVE_HELPER) == x.OBJECTIVE_HELPER_SHA256,
        "pinned inherited analysis gates",
    )
    return result


def validate_metadata(data, args, provenance, label):
    helpers = data["environment"]["helper_sha256"]
    parent = "bench_flashinfer_selection_history.py"
    require(
        helpers.get(parent) == h.source_hash(args, "test/manual/kernels/" + parent),
        label + ": actual inherited history worker",
    )
    view = copy.deepcopy(data)
    del view["environment"]["helper_sha256"][parent]
    x.validate_metadata(view, args, provenance, label)


def validate_startup(data, label):
    x.validate_profiles(data, False, label)
    x.validate_rng({"rng_states": data["rng_states"]}, label)
    require(
        data["natural"]["consumed_by_calibration_graphs"] is False,
        label + ": natural winner misrepresented as measured default",
    )
    objective = data["objective"]
    require(
        objective["requested_policy"] is None
        and not objective["protocol_errors"]
        and objective["startup_api_calls"]
        == [{"mode": "tune", "persistent_cache": False, "measurement_policy": None}],
        label + ": actual fixed startup policy",
    )
    require(
        objective["final_state"]
        == {
            "active_v2": False,
            "measurement_policy": None,
            "is_tuning_mode": False,
            "tuning_contexts": 0,
            "managed_store": False,
            "caller_stream_restored": True,
        },
        label + ": startup/capture scope leak",
    )
    records = objective["scope_records"]
    stages = {
        "startup",
        "natural_before",
        *(name + "_capture_scope" for name in GRAPHS),
    }
    require(
        collections.Counter((row["stage"], row["event"]) for row in records)
        == collections.Counter(
            (stage, event)
            for stage in stages
            for event in ("enter", "before_exit", "after_exit")
        ),
        label + ": complete startup/capture scope lifetime",
    )
    for row in records:
        startup, active = row["stage"] == "startup", row["event"] != "after_exit"
        require(
            row["mode"] == ("tune" if startup else "replay")
            and row["measurement_policy"] is None
            and row["winner_identity"] == "None"
            and row["active_v2"] is active
            and row["managed_store"] is False
            and row["is_tuning_mode"] is startup
            and row["tuning_contexts"]
            == (2 if startup and active else 1 if startup else 0)
            and row["on_forward_stream"] is startup
            and row["on_caller_stream"] is (not startup),
            label + ": actual scope/stream restoration",
        )
    o.validate_settings(
        data["autotuner_settings"]["initial"],
        "auto",
        label,
        configured=False,
        active=False,
    )
    o.validate_settings(
        data["autotuner_settings"]["after"], "auto", label, active=False
    )


def validate_graphs(data, label):
    trial = data["trial"]
    order = list(GRAPHS if trial % 2 == 0 else reversed(GRAPHS))
    cal, controls = data["calibration"], data["graph_controls"]
    require(
        controls["blocks"] == controls["sequential"] == controls["schedule"] == [],
        label + ": undeclared legacy timing windows",
    )
    require(
        cal["capture_order"] == controls["capture_order"] == order
        and [row["id"] for row in controls["graphs"]] == order,
        label + ": balanced actual capture order",
    )
    graphs = {row["id"]: row for row in controls["graphs"]}
    require(
        len(graphs) == 2
        and len({row["output"]["data_ptr"] for row in graphs.values()}) == 2
        and len({row["graph_object_id"] for row in graphs.values()}) == 2,
        label + ": distinct retained graph objects/outputs",
    )
    require(len(controls["miss_scope"]) == 2, label + ": two real default misses")
    for scope in controls["miss_scope"]:
        require(
            scope["winner_entries_before"] > 0
            and scope["winner_entries_after_removal"]
            == scope["file_entries_after_removal"]
            == 0
            and scope["restored"] is True
            and scope["all_cache_state_restored"] is True,
            label + ": default capture used real cache removal/restoration",
        )
        require(
            scope["winner_contents_sha256_before"]
            == scope["winner_contents_sha256_after"]
            and scope["file_contents_sha256_before"]
            == scope["file_contents_sha256_after"]
            and scope["cache_sha256_before"]
            == scope["cache_sha256_after"]
            == data["cache"]["after_controls_sha256"],
            label + ": miss changed original tuned caches",
        )
    m, n, k = data["shape"]
    previous = data["graph_controls"]["startup_ended_ns"]
    for name in order:
        graph = graphs[name]
        begin, end = g.interval(graph, label + "/" + name)
        require(begin > previous, label + ": graph capture chronology")
        previous = end
        require(
            graph["tactic"] == -1
            and graph["lookup_tier"] == "miss"
            and graph["policy_identity"] == "None"
            and graph["measurement_policy"] is None,
            label + ": actual default identity",
        )
        selected = graph["selection"]
        require(
            selected["stage"] == name + "_capture"
            and selected["tactic"] == -1
            and selected["cache_hit"] is False
            and selected["lookup_tier"] == "miss"
            and selected["capturing"] is True
            and selected["active_v2"] is True
            and selected["is_tuning_mode"] is False,
            label + ": real default capture selection",
        )
        require(
            graph["output"]["shape"] == [m, n]
            and graph["output"]["dtype"] == "torch.bfloat16"
            and graph["inputs"][4]["data_ptr"] == graph["output"]["data_ptr"]
            and graph["inputs"][0]["shape"] == [1, m, k]
            and graph["inputs"][1]["shape"] == [1, k, n],
            label + ": actual captured allocation/layout",
        )
        fixture = data["fixture"]["tensors"]
        require(
            graph["quantized_input_sha256"]
            == data["actual_layer"]["quantization_reference"]["actual_input_fp8_sha256"]
            and graph["weight_rowmajor_sha256"]
            == fixture["weight_fp8_rowmajor"]["sha256"]
            and graph["input_scale_sha256"] == fixture["input_scale"]["sha256"]
            and graph["weight_scale_sha256"] == fixture["weight_scale"]["sha256"],
            label + ": frozen actual graph inputs",
        )
    return graphs


def validate_lifecycle(data, graphs, label):
    winner = data["natural"]["tactic"]
    expected = {
        "setup": (-1, "miss", False),
        "startup": (winner, "memory", False),
        "natural_before": (winner, "memory", False),
    }
    for name in GRAPHS:
        expected[name + "_before"] = (-1, "miss", False)
        expected[name + "_capture"] = (-1, "miss", True)
    g.validate_stage_coverage(
        data["selections"], expected, label + ": selection coverage"
    )
    for row in data["selections"]:
        tactic, tier, capturing = expected[row["stage"]]
        require(
            row["tactic"] == tactic
            and row["lookup_tier"] == tier
            and row["cache_hit"] is (tier == "memory")
            and row["capturing"] is capturing
            and row["measurement_policy"] is None
            and row["policy_identity"] == "None"
            and row["is_tuning_mode"] is (row["stage"] == "startup")
            and row["active_v2"] is (row["stage"] != "setup"),
            label + ": startup/default selection",
        )
    binding_stages = set(expected) - {"setup", "startup"}
    g.validate_stage_coverage(
        data["binding_acceptance_checks"],
        dict.fromkeys(binding_stages),
        label + ": native coverage",
    )
    for row in data["binding_acceptance_checks"]:
        h.validate_acceptance(
            row, expected[row["stage"]][0], label + ": real native default path"
        )
    layer = data["actual_layer"]
    g.validate_stage_coverage(
        layer["observed_public_inputs"], expected, label + ": real public inputs"
    )
    g.validate_stage_coverage(
        layer["native_output_poison_checks"],
        dict.fromkeys(binding_stages),
        label + ": output poison",
    )
    for row in layer["native_output_poison_checks"]:
        require(
            row["capturing"] is expected[row["stage"]][2]
            and row["poisoned_actual_allocation"] is (not row["capturing"]),
            label + ": poison entered capture",
        )
    require(
        data["correctness"]["passed"] is True
        and (data["correctness"]["rtol"], data["correctness"]["atol"]) == (0.02, 0.125),
        label + ": unchanged numerical policy",
    )
    checks = x.numeric_checks(data, label)
    expected_numeric = {"initial_untuned", "natural_before"}
    expected_numeric.update(name + "_before" for name in GRAPHS)
    for name in GRAPHS:
        expected_numeric.update(
            name + suffix for suffix in ("_changed_input", "_restored_input")
        )
        expected_numeric.update(
            prefix + name for prefix in ("initial_warm_", "final_", "traced_")
        )
    expected_numeric.update(
        f"round_{index:02d}_{method}_{name}"
        for index in range(24)
        for method in METHODS
        for name in GRAPHS
    )
    require(set(checks) == expected_numeric, label + ": complete numerical lifecycle")
    for stage, row in checks.items():
        if stage in {
            "initial_untuned",
            "natural_before",
            *(name + "_before" for name in GRAPHS),
        }:
            require(row["graph_id"] is None, label + ": eager guard replaced by graph")
        else:
            name = next(name for name in GRAPHS if name in stage)
            x.check_allocation(checks, stage, name, graphs[name], label)
    return checks


def validate_delay_helper(cal, graphs, label):
    helper = cal["delay_helper"]
    require(
        helper["implementation"] == "torch.cuda._sleep calling torch._C._cuda_sleep"
        and helper["prime_cycles"] == 10_000_000
        and helper["primed_before_captures"] is True
        and 0
        < helper["prime_started_ns"]
        < helper["prime_ended_ns"]
        < min(row["started_ns"] for row in graphs.values()),
        label + ": delay preparation happened inside timing",
    )
    require(
        "_cuda_sleep" in helper["wrapper_source"]
        and hashlib.sha256(helper["wrapper_source"].encode()).hexdigest()
        == helper["wrapper_source_sha256"]
        and h.valid_sha(helper["wrapper_file_sha256"]),
        label + ": actual delay wrapper source",
    )
    require(
        len(helper["native_binaries"]) == 2
        and all(
            h.valid_sha(row["sha256"]) and row["size_bytes"] > 0
            for row in helper["native_binaries"]
        )
        and any(
            Path(row["path"]).name == "libtorch_cuda.so"
            for row in helper["native_binaries"]
        ),
        label + ": actual delay native implementation",
    )


def validate_samples(data, graphs, checks, label):
    cal = data["calibration"]
    require(
        cal["methods"] == METHODS
        and cal["schedule_seed"] == 7200 + data["trial"]
        and cal["schedule"] == expected_schedule(data["trial"])
        and cal["pending_sample"] is None
        and cal["all_samples_retained"] is True
        and cal["queue_query_is_exclusion_criterion"] is False
        and cal["synthetic_byte_attestation"] == "metadata_only_F0",
        label + ": fixed recipes/all observations retained",
    )
    require(
        [row["graph_id"] for row in cal["initial_warmups"]] == cal["capture_order"],
        label + ": common initial warmup coverage",
    )
    require(
        cal["captures_finished_ns"] > max(row["ended_ns"] for row in graphs.values()),
        label + ": warmup precedes capture",
    )
    for row in cal["initial_warmups"]:
        require(
            row["actual_ms"] >= 500
            and row["calls"] >= 5120
            and row["calls"] % 5120 == 0,
            label + ": initial graph warmup",
        )
    require(len(cal["rounds"]) == 24, label + ": no round exclusion")
    previous = cal["captures_finished_ns"]
    for index, (round_, requested) in enumerate(zip(cal["rounds"], cal["schedule"])):
        require(
            round_["index"] == index
            and round_["methods"] == requested["methods"]
            and round_["graph_order"] == requested["graph_order"],
            label + ": complete balanced round",
        )
        expected = [
            (method, name)
            for method in requested["methods"]
            for name in requested["graph_order"]
        ]
        require(
            [(row["method"], row["graph_id"]) for row in round_["measurements"]]
            == expected,
            label + ": no missing/reordered/excluded sample",
        )
        for row in round_["measurements"]:
            method, name = row["method"], row["graph_id"]
            require(
                row["replays"] == METHODS[method]["replays"]
                and row["prequeue_cycles"] == METHODS[method]["prequeue_cycles"],
                label + ": exact declared recipe",
            )
            require(
                h.positive(row["gpu_event_batch_ms"])
                and h.positive(row["gpu_event_per_replay_ms"])
                and math.isclose(
                    row["gpu_event_batch_ms"] / row["replays"],
                    row["gpu_event_per_replay_ms"],
                    rel_tol=1e-12,
                ),
                label + ": raw event normalization",
            )
            require(
                row["graph_object_id"] == graphs[name]["graph_object_id"]
                and row["output_data_ptr"] == graphs[name]["output"]["data_ptr"],
                label + ": measurement substituted another graph",
            )
            query = row["start_event_complete_after_enqueue"]
            expected_evidence = (
                (
                    "start_pending_after_enqueue"
                    if not query
                    else "start_complete_inconclusive"
                )
                if row["prequeue_cycles"]
                else "not_prequeued"
            )
            require(
                type(query) is bool
                and row["start_query_error"] is None
                and row["queue_evidence"] == expected_evidence,
                label + ": explicit query evidence without sample filtering",
            )
            begin, enqueue, issued, end = (
                row["wall_started_ns"],
                row["enqueue_started_ns"],
                row["enqueue_ended_ns"],
                row["ended_ns"],
            )
            require(
                previous < begin <= enqueue < issued <= end,
                label + ": sample chronology",
            )
            require(
                math.isclose(
                    (issued - enqueue) / 1e6, row["enqueue_cpu_ms"], rel_tol=1e-12
                )
                and math.isclose(
                    (end - begin) / 1e6,
                    row["wall_including_prequeue_ms"],
                    rel_tol=1e-12,
                )
                and row["enqueue_span_includes_start_query"] is True
                and row["wall_diagnostic_only"] is True,
                label + ": diagnostic wall/enqueue scope",
            )
            require(
                row["rewarm"]["actual_ms"] >= 20
                and row["rewarm"]["calls"] >= 5120
                and row["rewarm"]["calls"] % 5120 == 0,
                label + ": common prebatch rewarm",
            )
            require(
                row["profile_attempts_before"]
                == row["profile_attempts_after"]
                == data["profile_attempts"]
                and row["policy_scope_active"] is False
                and row["tuning_mode"] is False,
                label + ": profiling/context entered event sample",
            )
            stage = f"round_{index:02d}_{method}_{name}"
            require(
                row["correctness_check"] == stage,
                label + ": timed numerical check identity",
            )
            x.check_allocation(checks, stage, name, graphs[name], label)
            previous = end


def validate_trace(data, graphs, checks, raw_root, label):
    cal = data["calibration"]
    before, trace, after = (
        cal["pre_trace_checks"],
        cal["post_timing_trace"],
        cal["post_trace_checks"],
    )
    latest = max(
        row["ended_ns"] for group in cal["rounds"] for row in group["measurements"]
    )
    require(
        latest
        < before["ended_ns"]
        < trace["started_ns"]
        < trace["ended_ns"]
        < after["ended_ns"],
        label + ": profiler entered primary measurements",
    )
    for key in (
        "file_sha256",
        "winner_sha256",
        "loaded_file_sha256",
        "profile_attempts",
    ):
        require(
            before[key] == after[key],
            label + ": trace altered startup cache/profile state",
        )
    require(
        before["profile_attempts"] == data["profile_attempts"]
        and before["file_sha256"] == data["cache"]["after_controls_sha256"],
        label + ": trace used changed startup state",
    )
    path = raw_root / Path(trace["path"]).name
    require(
        sha256(path) == trace["sha256"] and trace["calls_per_graph"] == 3,
        label + ": actual retained trace",
    )
    raw = load(path)
    verified = g.validate_trace(raw, graph_ids=GRAPHS)
    require(
        [row["graph_id"] for row in trace["graphs"]] == list(GRAPHS),
        label + ": original graph trace coverage",
    )
    total = 0
    signatures = {}
    for row in trace["graphs"]:
        name, graph = row["graph_id"], graphs[row["graph_id"]]
        require(
            row["graph_object_id"] == graph["graph_object_id"]
            and row["output_data_ptr"] == graph["output"]["data_ptr"],
            label + ": trace recaptured or substituted the original graph",
        )
        scope = next(
            event
            for event in raw["traceEvents"]
            if event.get("cat") == "user_annotation"
            and event.get("name") == "retained_" + name
        )
        kernels = sorted(
            [
                event
                for event in raw["traceEvents"]
                if event.get("cat") == "kernel"
                and scope["ts"] <= event["ts"] <= scope["ts"] + scope["dur"]
            ],
            key=lambda event: event["ts"],
        )
        unique, sequence = [], []
        for event in kernels:
            signature = x.trace_signature(event)
            if signature not in unique:
                unique.append(signature)
            sequence.append(unique.index(signature))
        require(
            verified[name]["graph_launches"] == row["graph_launch_events"] == 3
            and verified[name]["kernel_events"]
            == row["kernel_events"]
            == len(kernels)
            == 6
            and row["signatures"] == unique
            and row["kernel_sequence"] == sequence,
            label + ": trace summary differs from actual CUDA launches",
        )
        x.check_allocation(checks, "traced_" + name, name, graph, label)
        signatures[name] = unique
        total += len(kernels)
    require(
        total
        == trace["kernel_events"]
        == len([event for event in raw["traceEvents"] if event.get("cat") == "kernel"]),
        label + ": unexpected GPU work outside retained ranges",
    )
    return {
        "file": str(path),
        "sha256": trace["sha256"],
        "actual_CUDA_graphs": verified,
        "same_ordered_signatures": signatures["graph_a"] == signatures["graph_b"],
        "scope": "Original retained objects observed after timing, no instrumented-latency inference.",
    }


def validate_payload(data, raw_root, label):
    validate_startup(data, label)
    graphs = validate_graphs(data, label)
    checks = validate_lifecycle(data, graphs, label)
    validate_delay_helper(data["calibration"], graphs, label)
    validate_samples(data, graphs, checks, label)
    return validate_trace(data, graphs, checks, raw_root, label)


def validate_worker(path, args, provenance, scheduled):
    data = load(path)
    require(data["trial"] == scheduled["trial"], path.name + ": trial identity")
    validate_metadata(data, args, provenance, path.name)
    trace = validate_payload(data, args.raw_root, path.name)
    observer = data["observer"]
    require(
        observer["enabled"] is False
        and observer["constructor_guard_suspended"] is False
        and observer["real_interposition_remained_loaded"] is False
        and not observer["preload_files"]
        and not observer["phases"]
        and observer["trace"] == data["calibration"]["post_timing_trace"],
        path.name + ": probe or early profiler path",
    )
    require(
        path.with_suffix(".exit-code").read_text().strip() == "0",
        path.name + ": worker failure",
    )
    cid = path.with_suffix(".container-id").read_text().strip()
    require(
        re.fullmatch("[0-9a-f]{64}", cid) is not None,
        path.name + ": fresh process container",
    )
    data.update(_path=str(path), _container_id=cid, _trace=trace)
    return data


def process_estimates(data):
    measurements = []
    for round_ in data["calibration"]["rounds"]:
        measurements.append(
            {
                (row["method"], row["graph_id"]): row["gpu_event_per_replay_ms"]
                for row in round_["measurements"]
            }
        )
    estimates = []
    for name in GRAPHS:
        for numerator, denominator in (
            ("queued16_10m", "plain5120"),
            ("queued16_40m", "queued16_10m"),
            ("queued16_40m", "plain5120"),
        ):
            pairs = [
                row[numerator, name] / row[denominator, name] for row in measurements
            ]
            estimates.append(
                {
                    "trial": data["trial"],
                    "kind": "same_graph_method_ratio",
                    "graph": name,
                    "numerator": numerator,
                    "denominator": denominator,
                    "ratio": statistics.median(pairs),
                    "round_ratios": pairs,
                }
            )
    first, second = data["calibration"]["capture_order"]
    for method in METHODS:
        pairs = [row[method, first] / row[method, second] for row in measurements]
        estimates.append(
            {
                "trial": data["trial"],
                "kind": "capture_order_ratio",
                "method": method,
                "first_graph": first,
                "second_graph": second,
                "ratio": statistics.median(pairs),
                "round_ratios": pairs,
            }
        )
    return estimates


def summarize(workers):
    estimates = [row for data in workers for row in process_estimates(data)]
    summaries = []
    keys = []
    for row in estimates:
        key = (
            row["kind"],
            row.get("graph"),
            row.get("numerator"),
            row.get("denominator"),
            row.get("method"),
        )
        if key not in keys:
            keys.append(key)
    for key in keys:
        selected = [
            row
            for row in estimates
            if (
                row["kind"],
                row.get("graph"),
                row.get("numerator"),
                row.get("denominator"),
                row.get("method"),
            )
            == key
        ]
        summary = o.summarize_ratios(selected)
        interval = summary["exploratory_cluster_bootstrap_95_percent"]
        summary["interval_width_ratio_units"] = (
            None if interval is None else interval["upper"] - interval["lower"]
        )
        summaries.append(
            {
                "kind": key[0],
                "graph": key[1],
                "numerator": key[2],
                "denominator": key[3],
                "method": key[4],
                **summary,
            }
        )
    query_counts = []
    for data in workers:
        for method in METHODS:
            samples = [
                row
                for round_ in data["calibration"]["rounds"]
                for row in round_["measurements"]
                if row["method"] == method
            ]
            query_counts.append(
                {
                    "trial": data["trial"],
                    "method": method,
                    "samples": len(samples),
                    "start_pending_after_enqueue": sum(
                        not row["start_event_complete_after_enqueue"] for row in samples
                    ),
                    "start_complete_inconclusive": sum(
                        row["start_event_complete_after_enqueue"] for row in samples
                    ),
                    "excluded_samples": 0,
                    "median_event_ms_per_replay": statistics.median(
                        row["gpu_event_per_replay_ms"] for row in samples
                    ),
                    "median_wall_ms_diagnostic": statistics.median(
                        row["wall_including_prequeue_ms"] for row in samples
                    ),
                    "median_enqueue_ms_diagnostic": statistics.median(
                        row["enqueue_cpu_ms"] for row in samples
                    ),
                }
            )
    return {
        "process_estimates": estimates,
        "summaries": summaries,
        "query_outcomes_all_retained": query_counts,
        "contextual_subpercent_scale": 0.0065,
        "resolution_note": "Compare interval widths and10m/40m sensitivity to the contextual scale; this is not an acceptance or equivalence threshold. If comparable, resolution remains unresolved.",
    }


def analyze(args):
    require(
        args.trials
        and sorted(set(args.trials)) == args.trials
        and all(t in TRIALS for t in args.trials),
        "ordered unique processes",
    )
    provenance = validate_outer(args)
    workers = [
        validate_worker(args.raw_root / row["raw_name"], args, provenance, row)
        for row in expected_workers()
        if row["trial"] in args.trials
    ]
    require(
        len({row["_container_id"] for row in workers}) == len(workers),
        "container reused",
    )
    for key in ("library_sha256", "ninja_sha256"):
        require(
            len({row["environment"]["native_artifact"][key] for row in workers}) == 1,
            "native implementation changed",
        )
    require(
        len({canonical(x.profiling_signature(row)) for row in workers}) == 1,
        "startup candidate/input metadata changed",
    )
    require(
        len(
            {
                canonical(row["calibration"]["delay_helper"]["native_binaries"])
                for row in workers
            }
        )
        == 1,
        "delay native implementation changed",
    )
    build = o.validate_build(args, provenance, workers)
    return {
        "status": "success",
        "schema_version": 1,
        "batch_id": Path(BATCH).name,
        "source_commit": args.expected_source_commit,
        "analyzer_sha256": sha256(Path(__file__)),
        "trials": args.trials,
        "independent_processes": len(workers),
        "no_performance_acceptance_threshold": True,
        "no_query_outcome_exclusions": True,
        "actual_native_build": build,
        "validated_workers": [
            {
                "file": row["_path"],
                "sha256": sha256(Path(row["_path"])),
                "trial": row["trial"],
                "container_id": row["_container_id"],
                "numeric_checks": len(row["correctness"]["checks"]),
                "actual_trace": row["_trace"],
            }
            for row in workers
        ],
        "analysis": summarize(workers),
        "limits": [
            "Six fresh processes are inferential units, not24rounds or144dependent batches.",
            "Queued wall spans include deliberate delay and are diagnostic, not serving latency.",
            "5120/noqueue versus16/queued compares complete recipes, not pure host submission cost.",
            "Prequeue delay can change device state;10m/40m sensitivity is not an isolated causal mechanism.",
            "Start.query True is inconclusive and retained; no outcome-based filtering/retries.",
            "Bootstrap intervals are exploratory/unadjusted and crossing1 does not establish equivalence.",
            "A single static-calibrated FP8 fixture/H100 does not establish fullmodel or universal ranking behavior.",
        ],
    }


def control_worker(trial, root):
    """Synthetic raw-schema fixture, never GPU or performance evidence."""
    requested = next(
        row
        for row in x.preregistered_schedule()
        if row["trial"] == trial and row["cell"] == "h0f0"
    )
    with patch.object(x, "GRAPH_IDS", ("winner", "default")):
        data = x.control_worker(requested, root)
    original = copy.deepcopy(data)
    old_trace = data["selection_history"]["post_timing_trace"]
    controls = data["graph_controls"]
    controls["blocks"], controls["sequential"], controls["schedule"] = [], [], []
    controls["miss_scope"].append(copy.deepcopy(controls["miss_scope"][0]))
    order = list(GRAPHS if trial % 2 == 0 else reversed(GRAPHS))
    controls["capture_order"] = order
    template = next(
        row for row in original["graph_controls"]["graphs"] if row["id"] == "default"
    )
    graphs = {}
    for index, name in enumerate(GRAPHS):
        graph = copy.deepcopy(template)
        pointer = 4096 if name == "graph_a" else 8192
        graph.update(id=name, graph_object_id=101 + index)
        graph["selection"]["stage"] = name + "_capture"
        graph["output"]["data_ptr"] = pointer
        graph["inputs"][4]["data_ptr"] = pointer
        graph["inputs"][0]["data_ptr"] = 16384 + index * 4096
        graphs[name] = graph
    controls["graphs"] = [graphs[name] for name in order]
    for index, graph in enumerate(controls["graphs"]):
        graph.update(
            started_ns=1_001_000_000 + index * 2000,
            ended_ns=1_001_001_000 + index * 2000,
        )
    for field in ("selections", "binding_acceptance_checks"):
        data[field] = [
            copy.deepcopy(row)
            for row in original[field]
            if row["stage"] in {"setup", "startup", "natural_before"}
        ]
        for name in GRAPHS:
            for suffix in ("before", "capture"):
                row = copy.deepcopy(
                    next(
                        row
                        for row in original[field]
                        if row["stage"] == "default_" + suffix
                    )
                )
                row["stage"] = name + "_" + suffix
                data[field].append(row)
    for field in ("observed_public_inputs", "native_output_poison_checks"):
        data["actual_layer"][field] = [
            copy.deepcopy(row)
            for row in original["actual_layer"][field]
            if row["stage"] in {"setup", "startup", "natural_before"}
        ]
        for name in GRAPHS:
            for suffix in ("before", "capture"):
                row = copy.deepcopy(
                    next(
                        row
                        for row in original["actual_layer"][field]
                        if row["stage"] == "default_" + suffix
                    )
                )
                row["stage"] = name + "_" + suffix
                data["actual_layer"][field].append(row)
    data["correctness"]["checks"] = [
        copy.deepcopy(row)
        for row in original["correctness"]["checks"]
        if row["check"] in {"initial_untuned", "natural_before"}
    ]

    def add_numeric(stage, name, old="final_default", eager=False):
        row = copy.deepcopy(
            next(
                row for row in original["correctness"]["checks"] if row["check"] == old
            )
        )
        row.update(
            check=stage,
            graph_id=None if eager else name,
            output_data_ptr=65536 if eager else graphs[name]["output"]["data_ptr"],
        )
        data["correctness"]["checks"].append(row)

    for name in GRAPHS:
        add_numeric(name + "_before", name, "default_before", True)
        add_numeric(name + "_changed_input", name, "default_changed_input")
        add_numeric(name + "_restored_input", name, "default_restored_input")
        for prefix in ("initial_warm_", "final_", "traced_"):
            add_numeric(prefix + name, name)
    data["rng_states"] = data.pop("selection_history")["rng_states"]
    data["natural"]["consumed_by_calibration_graphs"] = False
    records = []
    rename_scope = {
        "winner_capture_scope": "graph_a_capture_scope",
        "default_capture_scope": "graph_b_capture_scope",
    }
    for row in data["objective"]["scope_records"]:
        if row["stage"] != "natural_final":
            row["stage"] = rename_scope.get(row["stage"], row["stage"])
            records.append(row)
    data["objective"]["scope_records"] = records
    cal = {
        "methods": copy.deepcopy(METHODS),
        "capture_order": order,
        "schedule_seed": 7200 + trial,
        "schedule": expected_schedule(trial),
        "rounds": [],
        "initial_warmups": [
            {"graph_id": name, "actual_ms": 500, "calls": 5120} for name in order
        ],
        "pending_sample": None,
        "captures_finished_ns": 1_002_000_000,
        "all_samples_retained": True,
        "queue_query_is_exclusion_criterion": False,
        "synthetic_byte_attestation": "metadata_only_F0",
    }
    wrapper = "def _sleep(cycles):\n    return torch._C._cuda_sleep(cycles)\n"
    cal["delay_helper"] = {
        "implementation": "torch.cuda._sleep calling torch._C._cuda_sleep",
        "prime_cycles": 10_000_000,
        "primed_before_captures": True,
        "prime_started_ns": 600_000_000,
        "prime_ended_ns": 601_000_000,
        "wrapper_source": wrapper,
        "wrapper_source_sha256": hashlib.sha256(wrapper.encode()).hexdigest(),
        "wrapper_file_sha256": "a" * 64,
        "native_binaries": [
            {"path": "_C.so", "sha256": "b" * 64, "size_bytes": 1},
            {"path": "libtorch_cuda.so", "sha256": "c" * 64, "size_bytes": 2},
        ],
    }
    moment = 2_000_000_000
    for index, requested in enumerate(cal["schedule"]):
        round_ = {"index": index, **requested, "measurements": []}
        for method in requested["methods"]:
            for name in requested["graph_order"]:
                duration = {
                    "plain5120": 0.004,
                    "queued16_10m": 0.0032,
                    "queued16_40m": 0.003216,
                }[method] * (1 if name == "graph_a" else 1.0065)
                start = moment + 20_000_000
                end = (
                    start
                    + {
                        "plain5120": 50_000_000,
                        "queued16_10m": 10_000_000,
                        "queued16_40m": 30_000_000,
                    }[method]
                )
                complete = index % 2 == 0
                evidence = (
                    (
                        "start_complete_inconclusive"
                        if complete
                        else "start_pending_after_enqueue"
                    )
                    if METHODS[method]["prequeue_cycles"]
                    else "not_prequeued"
                )
                stage = f"round_{index:02d}_{method}_{name}"
                row = {
                    "method": method,
                    "graph_id": name,
                    **METHODS[method],
                    "graph_object_id": graphs[name]["graph_object_id"],
                    "output_data_ptr": graphs[name]["output"]["data_ptr"],
                    "gpu_event_batch_ms": duration * METHODS[method]["replays"],
                    "gpu_event_per_replay_ms": duration,
                    "start_event_complete_after_enqueue": complete,
                    "start_query_error": None,
                    "queue_evidence": evidence,
                    "enqueue_started_ns": start + 1000,
                    "enqueue_ended_ns": start + 6000,
                    "enqueue_cpu_ms": 0.005,
                    "enqueue_span_includes_start_query": True,
                    "wall_started_ns": start,
                    "ended_ns": end,
                    "wall_including_prequeue_ms": (end - start) / 1e6,
                    "wall_diagnostic_only": True,
                    "rewarm": {"actual_ms": 20, "calls": 5120},
                    "profile_attempts_before": data["profile_attempts"],
                    "profile_attempts_after": data["profile_attempts"],
                    "policy_scope_active": False,
                    "tuning_mode": False,
                    "correctness_check": stage,
                }
                round_["measurements"].append(row)
                add_numeric(stage, name)
                moment = end
        cal["rounds"].append(round_)
    events = load(Path(old_trace["path"]))
    rename = {
        "retained_winner": "retained_graph_a",
        "retained_default": "retained_graph_b",
    }
    for event in events["traceEvents"]:
        if event.get("name") in rename:
            event["name"] = rename[event["name"]]
    trace_path = root / f"trial{trial}-calibration.trace.json"
    trace_path.write_text(json.dumps(events))
    trace = copy.deepcopy(old_trace)
    trace.update(
        path=str(trace_path),
        sha256=sha256(trace_path),
        started_ns=moment + 200,
        ended_ns=moment + 300,
    )
    for row, name in zip(trace["graphs"], GRAPHS):
        row.update(
            graph_id=name,
            graph_object_id=graphs[name]["graph_object_id"],
            output_data_ptr=graphs[name]["output"]["data_ptr"],
        )
    snapshot = {
        "ended_ns": moment + 100,
        "file_sha256": data["cache"]["after_controls_sha256"],
        "winner_sha256": "e" * 64,
        "loaded_file_sha256": "f" * 64,
        "profile_attempts": data["profile_attempts"],
    }
    cal.update(
        pre_trace_checks=snapshot,
        post_timing_trace=trace,
        post_trace_checks={**snapshot, "ended_ns": moment + 400},
    )
    data["calibration"] = cal
    return data


def self_test():
    passed = []
    with tempfile.TemporaryDirectory(prefix="replay-calibration-controls-") as temp:
        root = Path(temp)
        validate_plan(load(Path(__file__).with_name("plan.json")))
        for trial in TRIALS:
            schedule = expected_schedule(trial)
            combinations = collections.Counter(
                (tuple(row["methods"]), tuple(row["graph_order"])) for row in schedule
            )
            require(
                len(combinations) == 12 and set(combinations.values()) == {2},
                "method/graph order balance within each permutation",
            )
            validate_payload(control_worker(trial, root), root, "CPU-positive")
        passed.append(
            "all6_processes_all_methods_orders_and_both_query_outcomes_retained"
        )

        def reject(name, mutate):
            data = control_worker(0, root)
            mutate(data)
            try:
                validate_payload(data, root, name)
            except (ValueError, KeyError, TypeError, AttributeError):
                passed.append(name)
            else:
                raise ValueError("fault control unexpectedly accepted: " + name)

        sample = lambda d: d["calibration"]["rounds"][0]["measurements"][0]
        reject(
            "missing_round_is_not_excluded", lambda d: d["calibration"]["rounds"].pop()
        )
        reject(
            "missing_sample_is_not_excluded",
            lambda d: d["calibration"]["rounds"][0]["measurements"].pop(),
        )
        reject(
            "wrong16_vs5120_normalization",
            lambda d: sample(d).update(
                gpu_event_per_replay_ms=sample(d)["gpu_event_per_replay_ms"] * 2
            ),
        )
        reject(
            "query_outcome_exclusion_flag",
            lambda d: d["calibration"].update(queue_query_is_exclusion_criterion=True),
        )
        reject(
            "query_evidence_mislabel",
            lambda d: sample(d).update(queue_evidence="pretend_invalid"),
        )
        reject(
            "query_error_cannot_be_success",
            lambda d: sample(d).update(start_query_error="CUDA error"),
        )
        reject(
            "wall_scope_must_remain_diagnostic",
            lambda d: sample(d).update(wall_diagnostic_only=False),
        )
        reject(
            "cpu_enqueue_scope_must_include_query",
            lambda d: sample(d).update(enqueue_span_includes_start_query=False),
        )
        reject(
            "delay_helper_preparation_inside_capture",
            lambda d: d["calibration"]["delay_helper"].update(
                prime_ended_ns=2_000_000_000
            ),
        )
        reject(
            "timed_graph_substituted", lambda d: sample(d).update(graph_object_id=999)
        )
        reject(
            "timed_output_substituted", lambda d: sample(d).update(output_data_ptr=999)
        )
        reject(
            "extra_profiling_during_measurement",
            lambda d: sample(d).update(
                profile_attempts_after=d["profile_attempts"] + 1
            ),
        )
        reject(
            "actual_default_capture_used_winner",
            lambda d: d["graph_controls"]["graphs"][0]["selection"].update(
                cache_hit=True
            ),
        )
        reject(
            "default_miss_cache_not_restored",
            lambda d: d["graph_controls"]["miss_scope"][0].update(restored=False),
        )
        reject(
            "profile_or_helper_inside_timed_context",
            lambda d: sample(d).update(policy_scope_active=True),
        )
        reject(
            "numerical_guard_checked_other_output",
            lambda d: next(
                row
                for row in d["correctness"]["checks"]
                if row["check"].startswith("round_")
            ).update(output_data_ptr=999),
        )
        reject(
            "trace_before_measurements_finish",
            lambda d: d["calibration"]["post_timing_trace"].update(started_ns=1),
        )
        reject(
            "trace_substituted_original_graph",
            lambda d: d["calibration"]["post_timing_trace"]["graphs"][0].update(
                graph_object_id=999
            ),
        )
        reject("scope_record_missing", lambda d: d["objective"]["scope_records"].pop())

        def corrupt_trace(data):
            trace = data["calibration"]["post_timing_trace"]
            path = Path(trace["path"])
            raw = load(path)
            next(event for event in raw["traceEvents"] if event.get("cat") == "kernel")[
                "args"
            ]["correlation"] = 999
            path.write_text(json.dumps(raw))
            trace["sha256"] = sha256(path)

        reject("trace_invalid_actual_kernel_correlation", corrupt_trace)
        workers = []
        for trial in TRIALS:
            data = control_worker(trial, root)
            data.update(
                _path=str(root / f"trial{trial}.json"), _container_id=f"{trial:064x}"
            )
            workers.append(data)
        summary = summarize(workers)
        require(
            all(row["independent_trials"] == 6 for row in summary["summaries"]),
            "rounds/graphs pooled as processes",
        )
        require(
            all(
                row["samples"] == 48 and row["excluded_samples"] == 0
                for row in summary["query_outcomes_all_retained"]
            ),
            "inconclusive samples were dropped",
        )
        for row in summary["summaries"]:
            if row["kind"] == "same_graph_method_ratio":
                expected = {
                    ("queued16_10m", "plain5120"): 0.8,
                    ("queued16_40m", "queued16_10m"): 1.005,
                    ("queued16_40m", "plain5120"): 0.804,
                }[row["numerator"], row["denominator"]]
                require(
                    math.isclose(row["median_ratio"], expected, rel_tol=1e-12),
                    "known paired-method ratio",
                )
        passed.append(
            "full6_summary_exact_process_units_known_ratios_and_no_query_exclusions"
        )
        require(
            o.summarize_ratios([{"trial": 0, "ratio": 1}])[
                "exploratory_cluster_bootstrap_95_percent"
            ]
            is None,
            "one-process checkpoint CI",
        )
        passed.append("checkpoint_n1_has_no_uncertainty_interval")
    return {
        "status": "success",
        "scope": "stdlib CPU synthetic controls; no performance/GPU evidence",
        "passed_controls": len(passed),
        "controls": passed,
        "analyzer_sha256": sha256(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "raw-root",
        "source-root",
        "source-manifest",
        "fixture-manifest",
        "build-provenance",
        "output",
    ):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--expected-source-commit")
    parser.add_argument(
        "--trials", type=int, nargs="+", choices=TRIALS, default=list(TRIALS)
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        for name in (
            "raw_root",
            "source_root",
            "source_manifest",
            "fixture_manifest",
            "expected_source_commit",
        ):
            if getattr(args, name) is None:
                parser.error("--" + name.replace("_", "-") + " is required")
    try:
        result = self_test() if args.self_test else analyze(args)
        code = 0
    except Exception as error:
        result, code = (
            {"status": "error", "error": f"{type(error).__name__}: {error}"},
            1,
        )
    content = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    print(content, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
