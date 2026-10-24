#!/usr/bin/env python3
"""Gate the fixed history-by-observation study and summarize complete trial blocks."""

from __future__ import annotations

import argparse
import base64
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

BATCH = "test/manual/kernels/results/flashinfer_autotune_selection/20261024T080000Z-h100-selection-history"
WORKER = "test/manual/kernels/bench_flashinfer_selection_history.py"
OBJECTIVE_HELPER = "test/manual/kernels/results/flashinfer_autotune_selection/20261024T031500Z-h100-measurement-objective/analyze.py"
OBJECTIVE_HELPER_SHA256 = (
    "d497a177b66062f87fd422dabd49ca3e598cfde857810765e8394b2f7cf1e5ad"
)
CELLS = ("h0f0", "h0f1", "h1f0", "h1f1")
TRIALS = tuple(range(8))
GRAPH_IDS = ("winner", "default")


def pinned_module(relative, expected):
    path = Path(__file__).resolve().parents[6] / relative
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("frozen analysis helper changed: " + relative)
    module = types.ModuleType("frozen_objective_analysis")
    module.__file__ = str(path)
    exec(compile(content, str(path), "exec"), module.__dict__)
    return module


o = pinned_module(OBJECTIVE_HELPER, OBJECTIVE_HELPER_SHA256)
h, g = o.h, o.g
require, load, canonical, sha256 = h.require, h.load, h.canonical, h.sha256
# Parameterize imported validators' source binding without editing helper files.
o.BATCH, o.WORKER = BATCH, WORKER


def preregistered_schedule():
    cells = dict(zip("ABCD", CELLS))
    orders = ("ABDC", "BCAD", "CDBA", "DACB")
    bits = ((0, 0, 1, 1), (1, 0, 1, 0), (1, 1, 0, 0), (0, 1, 0, 1))
    rows = []
    for trial in TRIALS:
        index = trial if trial < 4 else 7 - trial
        for position, letter in enumerate(orders[index]):
            cell = cells[letter]
            winner_first = bool(bits[index][position] ^ (trial >= 4))
            rows.append(
                {
                    "trial": trial,
                    "williams_block": trial // 4,
                    "launch_position": position,
                    "cell": cell,
                    "history": int(cell[1]),
                    "fingerprint": int(cell[3]),
                    "capture_order": "winner-first"
                    if winner_first
                    else "default-first",
                    "replay_order_seed": 6100 + trial,
                    "raw_name": f"trial{trial}-{cell}-timing.json",
                }
            )
    return rows


def validate_plan(plan):
    require(plan["batch_id"] == Path(BATCH).name, "preregistered batch")
    protocol = plan["protocol"]
    require(
        protocol["worker_schedule"] == preregistered_schedule(),
        "preregistered32worker schedule",
    )
    require(
        protocol["timing_workers"] == 32 and protocol["trials"] == list(TRIALS),
        "fixed cohort",
    )
    require(set(protocol["cells"]) == set(CELLS), "factor cell coverage")
    for cell, factors in protocol["cells"].items():
        require(
            factors == {"history": int(cell[1]), "fingerprint": int(cell[3])},
            "cell factor mapping",
        )
    require(protocol["worker"] == WORKER, "preregistered worker")
    require(
        plan["flashinfer_source"]["base_release"] == "0.7.2"
        and plan["flashinfer_source"]["patch_head"]
        == "14c5c67a1eb9ad3a608b8d3da3bf6d8924037c5c"
        and plan["flashinfer_source"]["source_manifest_sha256"]
        == "3f4305fce604d3cd51e85047678c6efbf71e35bcf51037e9586d15bc20486895",
        "fixed prior upstream source; no current-main shift",
    )
    require(
        protocol["profiling_defaults"]
        == {
            "warmup": 3,
            "repeat": 10,
            "profiling_repeat": None,
            "effective_profiling_repeat": 10,
            "cuda_graph_profile_replays": 1,
            "use_cuda_graph": False,
            "use_cold_l2_cache": False,
            "stream_delay_micro_secs": 5000,
            "cuda_graph_delay_micro_secs": 100,
            "use_global_timer": False,
        },
        "fixed auto profiler defaults",
    )
    require(
        (
            protocol["samples"],
            protocol["requested_iterations"],
            protocol["warmup_ms"],
            protocol["min_sample_ms"],
            protocol["blocks"],
            protocol["rewarm_ms"],
        )
        == (15, 5120, 500, 20, 24, 20),
        "fixed replay protocol",
    )


def expected_replay_schedule(trial):
    rows = [
        list(order) for _ in range(12) for order in itertools.permutations(GRAPH_IDS)
    ]
    random.Random(6100 + trial).shuffle(rows)
    return rows


o.expected_schedule = expected_replay_schedule


def validate_outer(args):
    provenance = o.validate_outer(args)
    validate_plan(provenance["plan"])
    outer = load(args.raw_root / "manifest.json")
    require(
        outer["source_sha256"].get(OBJECTIVE_HELPER) == OBJECTIVE_HELPER_SHA256,
        "pinned objective analysis helper binding",
    )
    return provenance


def validate_metadata(data, args, provenance, label):
    require(
        data["arm"] == "auto" and data["trial"] in args.trials,
        label + ": fixed auto/trial",
    )
    helpers = data["environment"]["helper_sha256"]
    parent = "flashinfer_autotune_measurement_objective.py"
    require(
        parent in helpers
        and helpers[parent] == h.source_hash(args, "test/manual/kernels/" + parent),
        label + ": actual parent worker source",
    )
    # The frozen validator knows the original five helpers. Validate the new
    # inherited helper above, then delegate unchanged production/native gates.
    view = copy.deepcopy(data)
    del view["environment"]["helper_sha256"][parent]
    o.validate_metadata(view, args, provenance, label)


def validate_profiles(data, fingerprint, label):
    profiles, summary = data["profiles"], data["runtime_profile_summary"]
    require(
        data["profile_attempts"] == len(profiles) > 0
        and not data["forbidden_profile_attempts"],
        label + ": profiling accounting",
    )
    groups = data["profiling_input_groups"]
    require(bool(groups), label + ": actual profiler operand metadata")
    by_profile = collections.defaultdict(list)
    for row in profiles:
        require(
            row["stage"] == "startup"
            and row["runner"] == "CublasFp8GemmRunner"
            and h.valid_tactic(row["tactic"]),
            label + ": profiling scope/tactic",
        )
        require(
            row["fingerprinted"] is fingerprint
            and row["use_cuda_graph"] is False
            and row["use_cold_l2_cache"] is False,
            label + ": intervention/objective",
        )
        o.validate_settings(row["autotuner_settings"], "auto", label)
        require(
            row["policy_identity"] == "None"
            and row["policy_manifest_fields"] == {}
            and row["default_race_enabled"] is True,
            label + ": actual v2 default race",
        )
        require(
            row["normalized_profile"] == h.normalized_profile(row["input_shapes"]),
            label + ": actual profile normalization",
        )
        if row.get("profiler_ms") is None:
            require(bool(row.get("error")), label + ": unreported candidate failure")
        else:
            require(
                h.positive(row["profiler_ms"]),
                label + ": finite positive candidate time",
            )
        key = row["profiling_input_group"]
        require(key in groups and h.valid_sha(key), label + ": operand metadata group")
        operands = groups[key]
        require(
            set(operands) == {"activation", "weight", "input_scale", "weight_scale"},
            label + ": operand coverage",
        )
        for index, name in enumerate(
            ("activation", "weight", "input_scale", "weight_scale")
        ):
            tensor = operands[name]
            require(
                set(tensor)
                == (
                    {"shape", "stride", "dtype", "sha256"}
                    if fingerprint
                    else {"shape", "stride", "dtype"}
                ),
                label + ": missing hash or fabricated F0 byte attestation",
            )
            require(
                tensor["shape"] == row["input_shapes"][index]
                and tensor["dtype"]
                == ("torch.float8_e4m3fn" if index < 2 else "torch.float32")
                and len(tensor["stride"]) == len(tensor["shape"])
                and all(
                    type(value) is int and value >= 0 for value in tensor["stride"]
                ),
                label + ": operand layout/dtype",
            )
            if fingerprint:
                require(
                    h.valid_sha(tensor["sha256"]), label + ": actual F1 byte digest"
                )
        require(
            hashlib.sha256(canonical(operands).encode()).hexdigest() == key,
            label + ": metadata-group digest",
        )
        by_profile[canonical(row["normalized_profile"])].append(row)
    require(
        {row["input_shapes"][0][-2] for row in profiles}
        == {1, 2, 4, 8, 16, 32, 64, 128, 144},
        label + ": actual startup bucket coverage",
    )
    require(
        set(groups) == {row["profiling_input_group"] for row in profiles},
        label + ": unreferenced groups",
    )
    for rows in by_profile.values():
        require(
            rows[0]["tactic"] == -1 and rows[0].get("profiler_ms") is not None,
            label + ": default-first finite participation",
        )
        require(
            len({canonical(row["tactic"]) for row in rows}) == len(rows),
            label + ": duplicate candidate",
        )
        require(
            len({row["profiling_input_group"] for row in rows}) == 1,
            label + ": candidate operand metadata mismatch",
        )
    runtime = by_profile[canonical(o.runtime_profile(data["shape"]))]
    finite = [row for row in runtime if row.get("profiler_ms") is not None]
    explicit = [row for row in finite if row["tactic"] != -1]
    require(bool(explicit), label + ": no finite explicit comparator")
    default = next(row["profiler_ms"] for row in finite if row["tactic"] == -1)
    minimum = min(row["profiler_ms"] for row in finite)
    best = min(row["profiler_ms"] for row in explicit)
    expected = {
        "normalized_profile": o.runtime_profile(data["shape"]),
        "profiles": len(runtime),
        "finite_profiles": len(finite),
        "minimum_ms": minimum,
        "minimum_tactics": [
            row["tactic"] for row in finite if row["profiler_ms"] == minimum
        ],
        "default_profiler_ms": default,
        "best_explicit_profiler_ms": best,
        "best_explicit_tactics": [
            row["tactic"] for row in explicit if row["profiler_ms"] == best
        ],
        "best_explicit_over_default": best / default,
    }
    require(summary == expected, label + ": actual continuous runtime profile margin")
    winner = data["natural"]["tactic"]
    require(
        winner == min(finite, key=lambda row: row["profiler_ms"])["tactic"],
        label + ": natural first finite minimum",
    )
    require(
        data["natural"]["cache_hit"] is True
        and data["natural"]["lookup_tier"] == "memory",
        label + ": true natural memory winner",
    )
    h.validate_acceptance(
        data["natural"]["acceptance"], winner, label + ": native descriptor acceptance"
    )
    require(
        data["cache"]["before_sha256"] is None
        and data["cache"]["miss_scopes_restored"] is True,
        label + ": fresh cache and reversible miss",
    )
    return winner


def numeric_checks(data, label):
    checks = {row["check"]: row for row in data["correctness"]["checks"]}
    require(
        len(checks) == len(data["correctness"]["checks"]),
        label + ": duplicate numeric labels",
    )
    for row in checks.values():
        g.validate_number(row, label)
    return checks


def expected_extra_checks(history):
    result = {"traced_winner", "traced_default"}
    if history:
        result |= {
            "pre_default_before",
            "pre_default_changed_input",
            "pre_default_restored_input",
            "history_before_startup",
            "history_final_changed_input",
            "history_final_restored_input",
        }
    return result


def validate_rng(history, label):
    rows = history["rng_states"]
    require(
        [row["stage"] for row in rows]
        == ["before_history", "before_startup", "after_startup"],
        label + ": RNG boundary coverage",
    )
    for row in rows:
        require(row["cuda_device"] == "cuda:0", label + ": active CUDA generator")
        for name in ("cpu", "cuda"):
            record = row[name]
            raw = base64.b64decode(record["base64"], validate=True)
            require(
                len(raw) == record["nbytes"] > 0
                and hashlib.sha256(raw).hexdigest() == record["sha256"],
                label + ": RNG-state bytes",
            )
    require(
        all(rows[0][name] == rows[1][name] for name in ("cpu", "cuda")),
        label + ": history changed the profiling RNG state",
    )


def check_allocation(checks, stage, graph_id, graph, label):
    row = checks[stage]
    require(
        row["graph_id"] == graph_id
        and row["output_data_ptr"] == graph["output"]["data_ptr"],
        label + ": checked another retained allocation at " + stage,
    )


def validate_history(data, scheduled, checks, label):
    evidence = data["selection_history"]
    enabled, fingerprint = bool(scheduled["history"]), bool(scheduled["fingerprint"])
    expected_order = list(
        GRAPH_IDS
        if scheduled["capture_order"] == "winner-first"
        else reversed(GRAPH_IDS)
    )
    require(
        evidence["history_enabled"] is enabled
        and evidence["inline_fingerprinting"] is fingerprint
        and evidence["requested_capture_order"] == expected_order
        and evidence["synthetic_input_references_retained"] is False,
        label + ": preregistered factors/capture/ref ownership",
    )
    require(
        evidence["profiling_byte_attestation"]
        == (
            "inline_full_hashes"
            if fingerprint
            else "metadata_only_no_synthetic_byte_attestation"
        ),
        label + ": honest profiling byte attestation",
    )
    require(
        evidence["runtime_best_explicit_over_default_profile"]
        == data["runtime_profile_summary"]["best_explicit_over_default"],
        label + ": reported profile ratio",
    )
    validate_rng(evidence, label)
    history = evidence["history"]
    require((history is not None) is enabled, label + ": history factor coverage")
    stages = {"pre_default_before", "pre_default_capture"} if enabled else set()
    fields = (
        ("selections", data["selections"]),
        ("native", data["binding_acceptance_checks"]),
        ("public", data["actual_layer"]["observed_public_inputs"]),
        ("poison", data["actual_layer"]["native_output_poison_checks"]),
    )
    for field, rows in fields:
        extras = [row for row in rows if row["stage"].startswith("pre_default")]
        require(
            collections.Counter(row["stage"] for row in extras)
            == collections.Counter(stages),
            label + ": exact history " + field + " coverage",
        )
        for row in extras:
            capture = row["stage"] == "pre_default_capture"
            if field == "selections":
                require(
                    row["tactic"] == -1
                    and row["lookup_tier"] == "miss"
                    and row["cache_hit"] is False
                    and row["capturing"] is capture
                    and row["active_v2"] is False
                    and row["is_tuning_mode"] is False
                    and row["policy_identity"] == "None"
                    and row["measurement_policy"] is None,
                    label + ": actual pre-startup miss",
                )
            elif field == "native":
                h.validate_acceptance(row, -1, label + ": history native path")
            elif field == "public":
                require(
                    row["capturing"] is capture and row["backend"] == "cublas",
                    label + ": history backend/capture",
                )
                if not capture:
                    require(
                        row["quantized_input_sha256"]
                        == data["actual_layer"]["quantization_reference"][
                            "actual_input_fp8_sha256"
                        ],
                        label + ": history actual frozen activation",
                    )
            else:
                require(
                    row["capturing"] is capture
                    and row["poisoned_actual_allocation"] is (not capture),
                    label + ": history poison must stay outside capture",
                )
    if enabled:
        graph, window = history["graph"], history["measurement"]
        require(
            graph["id"] == "pre_default"
            and graph["tactic"] == -1
            and graph["lookup_tier"] == "miss"
            and graph["selection"]["stage"] == "pre_default_capture",
            label + ": actual retained history graph",
        )
        require(
            history["retained_through_end"] is True
            and history["profile_attempts_before"]
            == history["profile_attempts_after"]
            == 0
            and history["cache_before"] == history["cache_after"]
            and history["cache_before"]["file_sha256"] is None,
            label + ": history polluted cache/profiling or was released",
        )
        begin, end = g.interval(graph, label + "/history_graph")
        wbegin, wend = g.interval(window, label + "/history_window")
        g.validate_measurement(window["measurement"], label + "/history_window")
        require(
            begin < end < wbegin < wend < data["graph_controls"]["startup_started_ns"],
            label + ": pre-startup recipe reordered",
        )
        require(
            window["graph_id"] == "pre_default"
            and window["correctness_check"] == "history_before_startup"
            and window["profile_attempts_before"]
            == window["profile_attempts_after"]
            == 0
            and window["policy_scope_active"] is False
            and window["tuning_mode"] is False,
            label + ": history measured while tuning",
        )
        for stage in expected_extra_checks(True) - {
            "pre_default_before",
            "traced_winner",
            "traced_default",
        }:
            check_allocation(checks, stage, "pre_default", graph, label)
        require(
            checks["pre_default_before"]["graph_id"] is None,
            label + ": eager history guard changed ownership",
        )
    return expected_order


def base_view(data, history):
    """Remove only validated new-stage extensions for frozen base checks."""
    result = copy.deepcopy(data)
    if history:
        for name in ("selections", "binding_acceptance_checks"):
            result[name] = [
                row
                for row in result[name]
                if not row["stage"].startswith("pre_default")
            ]
        for name in ("observed_public_inputs", "native_output_poison_checks"):
            result["actual_layer"][name] = [
                row
                for row in result["actual_layer"][name]
                if not row["stage"].startswith("pre_default")
            ]
    extras = expected_extra_checks(history)
    result["correctness"]["checks"] = [
        row for row in result["correctness"]["checks"] if row["check"] not in extras
    ]
    return result


def trace_signature(event):
    args = event["args"]
    return {
        "name": event["name"],
        **{
            key: args[key]
            for key in (
                "device",
                "grid",
                "block",
                "registers per thread",
                "shared memory",
            )
        },
    }


def validate_trace(data, graphs, checks, raw_root, label):
    evidence = data["selection_history"]
    before, trace, after = (
        evidence["pre_trace_checks"],
        evidence["post_timing_trace"],
        evidence["post_trace_checks"],
    )
    start, end = g.interval(trace, label + "/post_timing_trace")
    last_timing = max(
        row["ended_ns"]
        for block in data["graph_controls"]["blocks"]
        for row in block["measurements"]
    )
    require(
        last_timing < before["ended_ns"] < start < end < after["ended_ns"],
        label + ": profiler entered primary timing",
    )
    for key in (
        "file_sha256",
        "winner_sha256",
        "loaded_file_sha256",
        "profile_attempts",
    ):
        require(before[key] == after[key], label + ": trace altered cache/profiles")
    require(
        before["profile_attempts"] == data["profile_attempts"]
        and before["file_sha256"] == data["cache"]["after_controls_sha256"]
        and h.valid_sha(before["winner_sha256"])
        and h.valid_sha(before["loaded_file_sha256"]),
        label + ": trace used changed winner state",
    )
    path = raw_root / Path(trace["path"]).name
    require(
        path.is_file()
        and sha256(path) == trace["sha256"]
        and trace["calls_per_graph"] == 3,
        label + ": exact actual replay trace",
    )
    trace_data = load(path)
    verified_graphs = g.validate_trace(trace_data, graph_ids=GRAPH_IDS)
    events = trace_data["traceEvents"]
    require(
        [row["graph_id"] for row in trace["graphs"]] == list(GRAPH_IDS),
        label + ": trace graph coverage",
    )
    kernel_total, observed = 0, {}
    for graph_id, row in zip(GRAPH_IDS, trace["graphs"]):
        graph = graphs[graph_id]
        require(
            row["output_data_ptr"] == graph["output"]["data_ptr"]
            and row["graph_object_id"] == graph["graph_object_id"],
            label + ": trace substituted another graph",
        )
        markers = [
            event
            for event in events
            if event.get("cat") == "user_annotation"
            and event.get("name") == "retained_" + graph_id
        ]
        require(len(markers) == 1, label + ": actual retained graph label")
        marker = markers[0]
        inside = [
            event
            for event in events
            if marker["ts"] <= event.get("ts", -1) <= marker["ts"] + marker["dur"]
        ]
        launches = sorted(
            [
                event
                for event in inside
                if event.get("cat") == "cuda_runtime"
                and "cudaGraphLaunch" in event.get("name", "")
            ],
            key=lambda event: event["ts"],
        )
        kernels = sorted(
            [event for event in inside if event.get("cat") == "kernel"],
            key=lambda event: event["ts"],
        )
        require(
            len(launches) == 3 and len(kernels) == 6,
            label + ": actual three quantizer-plus-GEMM replays",
        )
        cycles = []
        for launch in launches:
            correlated = [
                event
                for event in kernels
                if event["args"]["correlation"] == launch["args"]["correlation"]
            ]
            require(
                len(correlated) == 2,
                label + ": kernels not correlated to retained launch",
            )
            signatures = [trace_signature(event) for event in correlated]
            require(
                "static_quant_fp8" in signatures[0]["name"],
                label + ": serving quantizer missing",
            )
            for signature in signatures:
                require(
                    signature["device"] == 0
                    and all(
                        type(v) is int and v > 0
                        for key in ("grid", "block")
                        for v in signature[key]
                    )
                    and type(signature["registers per thread"]) is int
                    and type(signature["shared memory"]) is int,
                    label + ": complete CUDA launch signature",
                )
            cycles.append(signatures)
        require(
            all(cycle == cycles[0] for cycle in cycles),
            label + ": retained launch sequence changed across replays",
        )
        signatures, sequence = [], []
        for event in kernels:
            signature = trace_signature(event)
            if signature not in signatures:
                signatures.append(signature)
            sequence.append(signatures.index(signature))
        require(
            row["graph_launch_events"] == 3
            and row["kernel_events"] == 6
            and row["signatures"] == signatures
            and row["kernel_sequence"] == sequence,
            label + ": trace summary differs from actual CUDA events",
        )
        check_allocation(checks, "traced_" + graph_id, graph_id, graph, label)
        observed[graph_id] = cycles[0]
        kernel_total += len(kernels)
    require(
        kernel_total
        == trace["kernel_events"]
        == len([event for event in events if event.get("cat") == "kernel"]),
        label + ": unexpected CUDA work outside retained ranges",
    )
    return {
        "file": str(path),
        "sha256": trace["sha256"],
        "original_retained_objects": True,
        "cuda_graph_identities": verified_graphs,
        "launches": observed,
        "winner_default_signatures_equal": observed["winner"] == observed["default"],
    }


def validate_timing(data, graphs, checker, label):
    controls = data["graph_controls"]
    require(
        controls["common_iterations"] == 5120
        and controls["schedule_seed"] == 6100 + data["trial"]
        and controls["schedule"] == expected_replay_schedule(data["trial"]),
        label + ": balanced replay protocol",
    )
    require(
        [row["graph_id"] for row in controls["sequential"]] == list(GRAPH_IDS),
        label + ": sequential coverage",
    )
    previous = max(row["ended_ns"] for row in graphs.values())
    for row in controls["sequential"]:
        stage = "sequential_" + row["graph_id"]
        require(
            row["correctness_check"] == stage
            and row["policy_scope_active"] is False
            and row["tuning_mode"] is False
            and row["profile_attempts_before"]
            == row["profile_attempts_after"]
            == data["profile_attempts"],
            label + ": sequential profiling/context leak",
        )
        g.validate_measurement(row["measurement"], label + "/" + stage)
        begin, end = g.interval(row, label + "/" + stage)
        require(
            begin > previous
            and (end - begin) / 1e6 >= sum(row["measurement"]["wall_batch_ms"]),
            label + ": sequential chronology",
        )
        previous = end
        checker(stage, row["graph_id"])
    require(len(controls["blocks"]) == 24, label + ": paired block count")
    for index, (block, order) in enumerate(
        zip(controls["blocks"], controls["schedule"])
    ):
        require(
            block["index"] == index
            and block["order"] == order
            and [row["graph_id"] for row in block["measurements"]] == order,
            label + ": actual paired block order",
        )
        for row in block["measurements"]:
            stage = f"block_{index:02d}_{row['graph_id']}"
            require(
                row["correctness_check"] == stage
                and row["policy_scope_active"] is False
                and row["tuning_mode"] is False,
                label + ": block context leak",
            )
            g.validate_batch(row, label + "/" + stage)
            begin, end = g.interval(row, label + "/" + stage)
            require(
                begin > previous
                and math.isclose(
                    (end - begin) / 1e6, row["wall_batch_ms"], rel_tol=1e-12
                ),
                label + ": actual replay clock/chronology",
            )
            previous = end
            checker(stage, row["graph_id"])


def validate_payload(data, scheduled, raw_root, label):
    require(
        (data["trial"], data["arm"]) == (scheduled["trial"], "auto"),
        label + ": cell process identity",
    )
    checks = numeric_checks(data, label)
    order = validate_history(data, scheduled, checks, label)
    winner = validate_profiles(data, bool(scheduled["fingerprint"]), label)
    o.validate_objective(data, label)
    require(
        data["graph_controls"]["capture_order"] == order,
        label + ": actual independently scheduled capture order",
    )
    # Adapt only the old graph validator's parity-based order selector after
    # checking actual trial/cell/order. Measurement/statistical input is untouched.
    graph_view = copy.deepcopy(data)
    graph_view["trial"] = 0 if order == list(GRAPH_IDS) else 1
    graphs = o.validate_graphs(graph_view, winner, label)
    require(
        all(
            type(row["graph_object_id"]) is int and row["graph_object_id"] > 0
            for row in graphs.values()
        )
        and len({row["graph_object_id"] for row in graphs.values()}) == 2,
        label + ": actual distinct Python graph objects",
    )
    startup_begin = data["graph_controls"]["startup_started_ns"]
    startup_end = data["graph_controls"]["startup_ended_ns"]
    require(
        0
        < startup_begin
        < startup_end
        < min(row["started_ns"] for row in graphs.values()),
        label + ": startup/capture lifecycle",
    )
    if scheduled["history"]:
        history = data["selection_history"]["history"]["graph"]
        require(
            history["output"]["data_ptr"]
            not in {row["output"]["data_ptr"] for row in graphs.values()}
            and history["graph_object_id"]
            not in {row["graph_object_id"] for row in graphs.values()},
            label + ": retained history allocation/object reused",
        )
    view = base_view(data, bool(scheduled["history"]))
    checker = o.validate_observed_stages(view, graphs, winner, label)
    validate_timing(view, graphs, checker, label)
    require(
        data["graph_controls"]["schedule_seed"] == 6100 + data["trial"],
        label + ": scheduled replay seed",
    )
    ratios = [
        next(
            row["gpu_event_per_call_ms"]
            for row in block["measurements"]
            if row["graph_id"] == "winner"
        )
        / next(
            row["gpu_event_per_call_ms"]
            for row in block["measurements"]
            if row["graph_id"] == "default"
        )
        for block in data["graph_controls"]["blocks"]
    ]
    require(
        data["selection_history"]["selected_over_default_graph_event_median"]
        == statistics.median(ratios),
        label + ": raw selected/default replay ratio",
    )
    return validate_trace(data, graphs, checks, raw_root, label)


def validate_worker(path, scheduled, args, provenance):
    data, label = load(path), path.name
    validate_metadata(data, args, provenance, label)
    trace = validate_payload(data, scheduled, args.raw_root, label)
    observer = data["observer"]
    require(
        observer["enabled"] is False
        and observer["constructor_guard_suspended"] is False
        and observer["real_interposition_remained_loaded"] is False
        and not observer["preload_files"]
        and not observer["phases"]
        and observer["trace"] == data["selection_history"]["post_timing_trace"]
        and not observer["native_output_path"],
        label + ": probe entered primary latency path",
    )
    require(type(data["pid"]) is int and data["pid"] > 0, label + ": actual process")
    require(
        path.with_suffix(".exit-code").read_text().strip() == "0",
        label + ": container exit",
    )
    cid = path.with_suffix(".container-id").read_text().strip()
    require(re.fullmatch("[0-9a-f]{64}", cid) is not None, label + ": fresh container")
    data.update(
        _path=str(path),
        _container_id=cid,
        _cell=scheduled["cell"],
        _trace_validation=trace,
    )
    return data


def profiling_signature(data, include_bytes=False):
    rows = []
    for row in data["profiles"]:
        operands = copy.deepcopy(
            data["profiling_input_groups"][row["profiling_input_group"]]
        )
        if not include_bytes:
            for tensor in operands.values():
                tensor.pop("sha256", None)
        rows.append(
            {
                key: row[key]
                for key in (
                    "runner",
                    "input_shapes",
                    "normalized_profile",
                    "tactic",
                    "autotuner_settings",
                )
            }
            | {"operands": operands}
        )
    return rows


def validate_cohort(workers):
    require(
        len({row["_container_id"] for row in workers}) == len(workers),
        "container reused across factorial cells",
    )
    for key in ("library_sha256", "ninja_sha256"):
        require(
            len({row["environment"]["native_artifact"][key] for row in workers}) == 1,
            "native build changed across cells",
        )
    require(
        len({canonical(row["environment"]["operand_sha256"]) for row in workers}) == 1,
        "fixture bytes changed",
    )
    require(
        len({canonical(profiling_signature(row)) for row in workers}) == 1,
        "candidate pools/order/layout/settings changed",
    )
    hashed = [
        row for row in workers if row["selection_history"]["inline_fingerprinting"]
    ]
    require(
        len({canonical(profiling_signature(row, include_bytes=True)) for row in hashed})
        == 1,
        "actual F1 profiling operand hashes changed",
    )
    for boundary in range(3):
        for generator in ("cpu", "cuda"):
            require(
                len(
                    {
                        canonical(
                            row["selection_history"]["rng_states"][boundary][generator]
                        )
                        for row in workers
                    }
                )
                == 1,
                "profiling generator state changed across cells",
            )


def trial_summary(values):
    return o.summarize_ratios(
        [{"trial": trial, "ratio": value} for trial, value in values]
    )


def factorial_ratios(cells):
    logs = {key: math.log(value) for key, value in cells.items()}
    return {
        "history_main": math.exp(
            0.5 * (logs["h1f0"] + logs["h1f1"] - logs["h0f0"] - logs["h0f1"])
        ),
        "fingerprint_main": math.exp(
            0.5 * (logs["h0f1"] + logs["h1f1"] - logs["h0f0"] - logs["h1f0"])
        ),
        "interaction": math.exp(
            logs["h1f1"] - logs["h1f0"] - logs["h0f1"] + logs["h0f0"]
        ),
    }


def summarize(workers):
    by_trial = collections.defaultdict(dict)
    records = []
    for row in workers:
        scores = {effect["channel"]: effect for effect in o.worker_effects(row)}
        profile = row["runtime_profile_summary"]
        record = {
            "trial": row["trial"],
            "cell": row["_cell"],
            "winner": row["natural"]["tactic"],
            "best_explicit_over_default_profile": profile["best_explicit_over_default"],
            "profile_explicit_minus_default_percent": 100
            * (profile["best_explicit_over_default"] - 1),
            "default_profiler_ms": profile["default_profiler_ms"],
            "best_explicit_profiler_ms": profile["best_explicit_profiler_ms"],
            "serving": scores,
            "capture_order": row["graph_controls"]["capture_order"],
            "startup_ms_secondary_instrumented": row["startup_ms"],
            "candidate_failures": [
                p for p in row["profiles"] if p.get("profiler_ms") is None
            ],
            "trace": row["_trace_validation"],
        }
        records.append(record)
        by_trial[row["trial"]][row["_cell"]] = record
    cell_summaries, factorial = [], []
    for measure in ("profile_margin", "serving_gpu_event", "serving_wall"):

        def value(record):
            if measure == "profile_margin":
                return record["best_explicit_over_default_profile"]
            return record["serving"][measure.removeprefix("serving_")]["ratio"]

        for cell in CELLS:
            subset = [record for record in records if record["cell"] == cell]
            cell_summaries.append(
                {
                    "cell": cell,
                    "measure": measure,
                    "primary": measure != "serving_wall",
                    **trial_summary(
                        [(record["trial"], value(record)) for record in subset]
                    ),
                    "default_winners": sum(record["winner"] == -1 for record in subset),
                    "explicit_winners": sum(
                        record["winner"] != -1 for record in subset
                    ),
                }
            )
        contrasts = {
            trial: factorial_ratios(
                {cell: value(record) for cell, record in cells.items()}
            )
            for trial, cells in sorted(by_trial.items())
        }
        for contrast in ("history_main", "fingerprint_main", "interaction"):
            factorial.append(
                {
                    "measure": measure,
                    "contrast": contrast,
                    "primary": measure != "serving_wall",
                    "ratio_statistic": "exponentiated within-trial log contrast; median across independent complete repetitions",
                    **trial_summary(
                        [
                            (trial, values[contrast])
                            for trial, values in contrasts.items()
                        ]
                    ),
                }
            )
    return {
        "workers": records,
        "cell_summaries": cell_summaries,
        "factorial_effects": factorial,
        "confidence_unit": "Complete four-cell repetition; exact multinomial whole-trial bootstrap, exploratory/unadjusted.",
    }


def worker_validation_record(row):
    path = Path(row["_path"])
    return {
        "file": str(path),
        "sha256": sha256(path),
        "trial": row["trial"],
        "cell": row["_cell"],
        "container_id": row["_container_id"],
        "profiles": len(row["profiles"]),
        "numeric_checks": len(row["correctness"]["checks"]),
    }


def analyze(args):
    require(
        args.trials
        and sorted(set(args.trials)) == args.trials
        and all(trial in TRIALS for trial in args.trials),
        "ordered unique trials",
    )
    provenance = validate_outer(args)
    workers = [
        validate_worker(args.raw_root / row["raw_name"], row, args, provenance)
        for row in preregistered_schedule()
        if row["trial"] in args.trials
    ]
    validate_cohort(workers)
    build = o.validate_build(args, provenance, workers)
    return {
        "status": "success",
        "schema_version": 1,
        "batch_id": Path(BATCH).name,
        "source_commit": args.expected_source_commit,
        "analyzer_sha256": sha256(Path(__file__)),
        "frozen_objective_helper_sha256": OBJECTIVE_HELPER_SHA256,
        "trials": args.trials,
        "independent_complete_repetitions": len(args.trials),
        "validated_workers": [worker_validation_record(row) for row in workers],
        "no_latency_acceptance_threshold": True,
        "persistence_qualified": False,
        "actual_native_build": build,
        "analysis": summarize(workers),
        "limitations": [
            "History bundles pre-startup capture/retained allocations/GPU work/time; it does not isolate thermal state or tuning causality.",
            "F0 records no synthetic byte hashes and retains no synthetic inputs; equal metadata/RNG state is not direct byte proof.",
            "Four fresh cells form one independent repetition; replay samples and trace calls are dependent subsamples.",
            "Intervals are exploratory/unadjusted whole-trial bootstrap, not equivalence or significance claims. All32workers were planned.",
            "CUDA event replay spans include host submission gaps. Post-timing traces establish launch signatures, not uninstrumented kernel latency or physical memory equivalence.",
            "Single static-calibrated FP8 QKV fixture/H100; no fullmodel accuracy/serving performance, persistence, TP/EP or backend-wide claim.",
            "This fixed-source factorial does not retroactively establish the cause of another campaign's measurements.",
        ],
    }


def control_worker(scheduled, root):
    """Synthetic CPU-only producer fixture; never reported as GPU evidence."""
    data = o.control_worker("auto", scheduled["trial"])
    controls = data["graph_controls"]
    for row in controls["graphs"]:
        row["started_ns"] += 1_000_000_000
        row["ended_ns"] += 1_000_000_000
        row["graph_object_id"] = 101 if row["id"] == "winner" else 102
    for row in controls["sequential"]:
        row["started_ns"] += 1_000_000_000
        row["ended_ns"] += 1_000_000_000
    for block in controls["blocks"]:
        for row in block["measurements"]:
            row["started_ns"] += 1_000_000_000
            row["ended_ns"] += 1_000_000_000
    order = list(
        GRAPH_IDS
        if scheduled["capture_order"] == "winner-first"
        else reversed(GRAPH_IDS)
    )
    controls["capture_order"] = order
    graphs = {row["id"]: row for row in controls["graphs"]}
    controls["graphs"] = [graphs[name] for name in order]
    for index, row in enumerate(controls["graphs"]):
        row["started_ns"] = 1_001_000_000 + index * 2000
        row["ended_ns"] = row["started_ns"] + 1000
    controls["startup_started_ns"], controls["startup_ended_ns"] = (
        500_000_000,
        501_000_000,
    )
    controls["schedule_seed"] = 6100 + scheduled["trial"]
    fingerprint = bool(scheduled["fingerprint"])
    groups = {}
    for row in data["profiles"]:
        operands = copy.deepcopy(
            data["profiling_input_groups"][row["profiling_input_group"]]
        )
        if not fingerprint:
            for tensor in operands.values():
                del tensor["sha256"]
        group = hashlib.sha256(canonical(operands).encode()).hexdigest()
        groups[group] = operands
        row.update(profiling_input_group=group, fingerprinted=fingerprint)
    data["profiling_input_groups"] = groups
    summary = data["runtime_profile_summary"]
    summary.update(
        default_profiler_ms=2.0,
        best_explicit_profiler_ms=1.0,
        best_explicit_tactics=[list(range(8))],
        best_explicit_over_default=0.5,
    )
    rng = lambda raw: {
        "nbytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "base64": base64.b64encode(raw).decode("ascii"),
    }
    states = [
        {
            "stage": stage,
            "cuda_device": "cuda:0",
            "cpu": rng(b"fixed-cpu-state"),
            "cuda": rng(b"fixed-cuda-state"),
        }
        for stage in ("before_history", "before_startup", "after_startup")
    ]
    history = None
    if scheduled["history"]:
        graph = copy.deepcopy(graphs["default"])
        graph.update(
            id="pre_default", graph_object_id=103, started_ns=1000, ended_ns=2000
        )
        graph["output"]["data_ptr"] = 40960
        graph["inputs"][4]["data_ptr"] = 40960
        graph["inputs"][0]["data_ptr"] = 45056
        for field in ("selections", "binding_acceptance_checks"):
            for suffix in ("before", "capture"):
                row = copy.deepcopy(
                    next(
                        row
                        for row in data[field]
                        if row["stage"] == "default_" + suffix
                    )
                )
                row["stage"] = "pre_default_" + suffix
                if field == "selections":
                    row["active_v2"] = False
                    if suffix == "capture":
                        graph["selection"] = copy.deepcopy(row)
                data[field].append(row)
        for field in ("observed_public_inputs", "native_output_poison_checks"):
            for suffix in ("before", "capture"):
                row = copy.deepcopy(
                    next(
                        row
                        for row in data["actual_layer"][field]
                        if row["stage"] == "default_" + suffix
                    )
                )
                row["stage"] = "pre_default_" + suffix
                data["actual_layer"][field].append(row)
        window = copy.deepcopy(controls["sequential"][1])
        window.update(
            graph_id="pre_default",
            correctness_check="history_before_startup",
            started_ns=3000,
            ended_ns=400_003_000,
            profile_attempts_before=0,
            profile_attempts_after=0,
        )
        snapshot = {
            "file_sha256": None,
            "winner_sha256": "d" * 64,
            "loaded_file_sha256": "e" * 64,
        }
        history = {
            "graph": graph,
            "measurement": window,
            "cache_before": snapshot,
            "cache_after": copy.deepcopy(snapshot),
            "profile_attempts_before": 0,
            "profile_attempts_after": 0,
            "retained_through_end": True,
        }
        for stage in expected_extra_checks(True) - {"traced_winner", "traced_default"}:
            source = (
                "default_changed_input"
                if stage.endswith("_changed_input")
                else "default_restored_input"
            )
            if stage == "pre_default_before":
                source = "default_before"
            row = copy.deepcopy(
                next(
                    row
                    for row in data["correctness"]["checks"]
                    if row["check"] == source
                )
            )
            row.update(
                check=stage,
                graph_id=None if stage == "pre_default_before" else "pre_default",
                output_data_ptr=65536 if stage == "pre_default_before" else 40960,
            )
            data["correctness"]["checks"].append(row)
    events, trace_rows = [], []
    for index, graph_id in enumerate(GRAPH_IDS):
        start = index * 200
        events.append(
            {
                "cat": "user_annotation",
                "ph": "X",
                "name": "retained_" + graph_id,
                "ts": start,
                "dur": 100,
            }
        )
        signatures, sequence = [], []
        for cycle in range(3):
            correlation = index * 10 + cycle + 1
            launch = start + cycle * 25 + 5
            events.append(
                {
                    "cat": "cuda_runtime",
                    "ph": "X",
                    "name": "cudaGraphLaunch",
                    "ts": launch,
                    "dur": 1,
                    "args": {"correlation": correlation},
                }
            )
            for offset, name in enumerate(("_static_quant_fp8", "controlled_gemm")):
                event = {
                    "cat": "kernel",
                    "ph": "X",
                    "name": name,
                    "ts": launch + 2 + offset * 3,
                    "dur": 2,
                    "args": {
                        "correlation": correlation,
                        "graph id": index + 1,
                        "device": 0,
                        "grid": [1, 1, 1],
                        "block": [128, 1, 1],
                        "registers per thread": 32,
                        "shared memory": 0,
                    },
                }
                events.append(event)
                signature = trace_signature(event)
                if signature not in signatures:
                    signatures.append(signature)
                sequence.append(signatures.index(signature))
        # Torch can emit a GPU annotation with the same label. It is not the
        # CPU range containing cudaGraphLaunch calls and must not add a scope.
        events.append(
            {
                "cat": "gpu_user_annotation",
                "ph": "X",
                "name": "retained_" + graph_id,
                "ts": start + 1,
                "dur": 98,
            }
        )
        trace_rows.append(
            {
                "graph_id": graph_id,
                "graph_object_id": graphs[graph_id]["graph_object_id"],
                "output_data_ptr": graphs[graph_id]["output"]["data_ptr"],
                "graph_launch_events": 3,
                "kernel_events": 6,
                "signatures": signatures,
                "kernel_sequence": sequence,
            }
        )
        numeric = copy.deepcopy(
            next(
                row
                for row in data["correctness"]["checks"]
                if row["check"] == "final_" + graph_id
            )
        )
        numeric["check"] = "traced_" + graph_id
        data["correctness"]["checks"].append(numeric)
    path = root / (
        "control-" + scheduled["raw_name"].removesuffix(".json") + ".trace.json"
    )
    path.write_text(json.dumps({"traceEvents": events}))
    end = max(
        row["ended_ns"] for block in controls["blocks"] for row in block["measurements"]
    )
    snapshot = {
        "ended_ns": end + 100,
        "file_sha256": data["cache"]["after_controls_sha256"],
        "winner_sha256": "e" * 64,
        "loaded_file_sha256": "f" * 64,
        "profile_attempts": data["profile_attempts"],
    }
    trace = {
        "path": str(path),
        "sha256": sha256(path),
        "started_ns": end + 200,
        "ended_ns": end + 300,
        "calls_per_graph": 3,
        "kernel_events": 12,
        "graphs": trace_rows,
    }
    data["selection_history"] = {
        "history_enabled": bool(scheduled["history"]),
        "inline_fingerprinting": fingerprint,
        "requested_capture_order": order,
        "history": history,
        "rng_states": states,
        "runtime_best_explicit_over_default_profile": 0.5,
        "selected_over_default_graph_event_median": statistics.median(
            next(
                row["gpu_event_per_call_ms"]
                for row in block["measurements"]
                if row["graph_id"] == "winner"
            )
            / next(
                row["gpu_event_per_call_ms"]
                for row in block["measurements"]
                if row["graph_id"] == "default"
            )
            for block in controls["blocks"]
        ),
        "profiling_byte_attestation": "inline_full_hashes"
        if fingerprint
        else "metadata_only_no_synthetic_byte_attestation",
        "synthetic_input_references_retained": False,
        "pre_trace_checks": snapshot,
        "post_timing_trace": trace,
        "post_trace_checks": {**snapshot, "ended_ns": end + 400},
    }
    return data


def self_test():
    passed = []
    with tempfile.TemporaryDirectory(prefix="selection-history-analyzer-") as temp:
        root = Path(temp)
        schedule = preregistered_schedule()
        plan = load(Path(__file__).with_name("plan.json"))
        validate_plan(plan)
        for block in (0, 1):
            subset = [row for row in schedule if row["williams_block"] == block]
            require(
                all(
                    collections.Counter(
                        row["cell"]
                        for row in subset
                        if row["launch_position"] == position
                    )
                    == collections.Counter(CELLS)
                    for position in range(4)
                ),
                "balanced launch slots",
            )
            pairs = collections.Counter()
            for trial in range(block * 4, block * 4 + 4):
                cells = [row["cell"] for row in subset if row["trial"] == trial]
                pairs.update(zip(cells, cells[1:]))
            require(
                len(pairs) == 12 and set(pairs.values()) == {1},
                "Williams predecessor balance",
            )
        for cell in CELLS:
            for position in range(4):
                require(
                    collections.Counter(
                        row["capture_order"]
                        for row in schedule
                        if row["cell"] == cell and row["launch_position"] == position
                    )
                    == collections.Counter(("winner-first", "default-first")),
                    "independent capture balance",
                )
        passed.append("full32_preregistered_launch_and_capture_counterbalance")
        controls = []
        for scheduled in schedule:
            data = control_worker(scheduled, root)
            trace = validate_payload(data, scheduled, root, "CPU-positive")
            path = root / scheduled["raw_name"]
            path.write_text(json.dumps(data))
            data.update(
                _path=str(path),
                _cell=scheduled["cell"],
                _trace_validation=trace,
                _container_id=hashlib.sha256(
                    scheduled["raw_name"].encode()
                ).hexdigest(),
                startup_ms=100.0,
            )
            record = worker_validation_record(data)
            require(
                record["sha256"] == sha256(path) and record["file"] == str(path),
                "final reporting accepts actual str raw paths",
            )
            controls.append(data)
        passed.append("all32_H_F_capture_order_payloads_no_GPU")
        summary = summarize(controls)
        require(
            all(row["independent_trials"] == 8 for row in summary["cell_summaries"])
            and all(
                row["independent_trials"] == 8 for row in summary["factorial_effects"]
            ),
            "full32 output must retain8 complete repetitions",
        )
        require(
            all(row["median_ratio"] == 1 for row in summary["factorial_effects"]),
            "equal cells must have neutral factorial effects",
        )
        passed.append("successful_final_record_hash_and_full32_summary_branch")

        def reject(name, mutate, cell="h1f1"):
            scheduled = next(
                row for row in schedule if row["trial"] == 0 and row["cell"] == cell
            )
            data = control_worker(scheduled, root)
            mutate(data)
            try:
                validate_payload(data, scheduled, root, name)
            except (ValueError, KeyError, TypeError, StopIteration):
                passed.append(name)
            else:
                raise ValueError("fault control unexpectedly accepted: " + name)

        reject(
            "history_enabled_but_absent",
            lambda d: d["selection_history"].update(history=None),
        )
        reject(
            "H0_fabricated_history",
            lambda d: d["selection_history"].update(history={}),
            "h0f0",
        )
        reject(
            "F0_fabricated_byte_hash",
            lambda d: next(iter(d["profiling_input_groups"].values()))[
                "activation"
            ].update(sha256="a" * 64),
            "h0f0",
        )
        reject(
            "F1_missing_byte_hash",
            lambda d: next(iter(d["profiling_input_groups"].values()))[
                "activation"
            ].pop("sha256"),
        )
        reject(
            "profiling_input_references_retained",
            lambda d: d["selection_history"].update(
                synthetic_input_references_retained=True
            ),
        )
        reject("lost_true_winner_hit", lambda d: d["natural"].update(cache_hit=False))
        reject(
            "default_did_not_participate",
            lambda d: d["profiles"][0].update(tactic=list(range(8))),
        )
        reject(
            "wrong_profiler_minimum",
            lambda d: d["runtime_profile_summary"].update(
                best_explicit_over_default=0.9
            ),
        )
        reject(
            "extra_profiling_attempt",
            lambda d: d.update(profile_attempts=d["profile_attempts"] + 1),
        )
        reject(
            "history_advanced_rng",
            lambda d: d["selection_history"]["rng_states"][1].update(
                cpu=d["selection_history"]["rng_states"][1]["cuda"]
            ),
        )
        reject(
            "history_tuned",
            lambda d: d["selection_history"]["history"].update(
                profile_attempts_after=1
            ),
        )
        reject(
            "history_released_before_end",
            lambda d: d["selection_history"]["history"].update(
                retained_through_end=False
            ),
        )
        reject(
            "history_after_startup",
            lambda d: d["selection_history"]["history"]["measurement"].update(
                ended_ns=d["graph_controls"]["startup_started_ns"] + 1
            ),
        )
        reject(
            "wrong_scheduled_capture_order",
            lambda d: d["selection_history"].update(
                requested_capture_order=list(
                    reversed(d["selection_history"]["requested_capture_order"])
                )
            ),
        )
        reject(
            "shared_primary_output",
            lambda d: d["graph_controls"]["graphs"][1]["output"].update(
                data_ptr=d["graph_controls"]["graphs"][0]["output"]["data_ptr"]
            ),
        )
        reject(
            "cache_miss_not_restored",
            lambda d: d["graph_controls"]["miss_scope"][0].update(
                all_cache_state_restored=False
            ),
        )
        reject(
            "unbalanced_actual_block_order",
            lambda d: d["graph_controls"]["schedule"].__setitem__(
                0, ["winner", "winner"]
            ),
        )
        reject(
            "wrong_timed_output_owner",
            lambda d: next(
                row
                for row in d["correctness"]["checks"]
                if row["check"] == "block_00_default"
            ).update(output_data_ptr=777),
        )
        reject(
            "trace_before_latency_ended",
            lambda d: d["selection_history"]["post_timing_trace"].update(started_ns=1),
        )
        reject(
            "trace_replaced_graph",
            lambda d: d["selection_history"]["post_timing_trace"]["graphs"][0].update(
                graph_object_id=999
            ),
        )
        reject(
            "trace_accepted_wrong_output",
            lambda d: d["selection_history"]["post_timing_trace"]["graphs"][0].update(
                output_data_ptr=999
            ),
        )
        reject(
            "trace_tuned_after_latency",
            lambda d: d["selection_history"]["post_trace_checks"].update(
                profile_attempts=d["profile_attempts"] + 1
            ),
        )

        def mutate_trace(data, update):
            trace = data["selection_history"]["post_timing_trace"]
            path = Path(trace["path"])
            raw = load(path)
            update(raw)
            path.write_text(json.dumps(raw))
            trace["sha256"] = sha256(path)

        reject(
            "trace_missing_CPU_range_despite_GPU_label",
            lambda d: mutate_trace(
                d,
                lambda t: t.update(
                    traceEvents=[
                        e
                        for e in t["traceEvents"]
                        if not (
                            e.get("cat") == "user_annotation"
                            and e.get("name") == "retained_winner"
                        )
                    ]
                ),
            ),
        )
        reject(
            "trace_duplicate_CPU_range",
            lambda d: mutate_trace(
                d,
                lambda t: t["traceEvents"].append(
                    copy.deepcopy(
                        next(
                            e
                            for e in t["traceEvents"]
                            if e.get("cat") == "user_annotation"
                            and e.get("name") == "retained_winner"
                        )
                    )
                ),
            ),
        )
        reject(
            "trace_CPU_range_loses_launches",
            lambda d: mutate_trace(
                d,
                lambda t: next(
                    e
                    for e in t["traceEvents"]
                    if e.get("cat") == "user_annotation"
                    and e.get("name") == "retained_winner"
                ).update(dur=1),
            ),
        )
        reject(
            "trace_lost_kernel_correlation",
            lambda d: mutate_trace(
                d,
                lambda t: next(e for e in t["traceEvents"] if e.get("cat") == "kernel")[
                    "args"
                ].update(correlation=999),
            ),
        )
        reject(
            "trace_same_CUDA_graph_both_labels",
            lambda d: mutate_trace(
                d,
                lambda t: [
                    e["args"].update({"graph id": 1})
                    for e in t["traceEvents"]
                    if e.get("cat") == "kernel"
                ],
            ),
        )
        factors = factorial_ratios(
            {
                "h0f0": 1.1,
                "h0f1": 1.1 * 0.8,
                "h1f0": 1.1 * 1.2,
                "h1f1": 1.1 * 1.2 * 0.8 * 1.3,
            }
        )
        require(
            math.isclose(factors["history_main"], 1.2 * math.sqrt(1.3), rel_tol=1e-12)
            and math.isclose(
                factors["fingerprint_main"], 0.8 * math.sqrt(1.3), rel_tol=1e-12
            )
            and math.isclose(factors["interaction"], 1.3, rel_tol=1e-12),
            "known factorial log effects",
        )
        passed.append("known_history_fingerprint_interaction_log_contrasts")
        require(
            len(h.bootstrap_samples(8)) == 6435
            and sum(weight for _, weight in h.bootstrap_samples(8)) == 8**8,
            "exact8trial cluster weights",
        )
        passed.append("exact_compact8trial_bootstrap_weights")
        values = [0.8, 1.0, 1.4]
        exhaustive = sorted(
            statistics.median(values[i] for i in sample)
            for sample in itertools.product(range(3), repeat=3)
        )
        interval = h.interval_for_trial_median(values)

        def percentile(p):
            position = p * (len(exhaustive) - 1)
            lo, hi = math.floor(position), math.ceil(position)
            return exhaustive[lo] + (exhaustive[hi] - exhaustive[lo]) * (position - lo)

        require(
            interval["lower"] == percentile(0.025)
            and interval["upper"] == percentile(0.975),
            "compact versus exhaustive cluster interval",
        )
        require(
            trial_summary([(0, 1.1)])["exploratory_cluster_bootstrap_95_percent"]
            is None,
            "n1 checkpoint must not invent interval",
        )
        passed.append("cluster_interval_bruteforce_and_n1_no_CI")
    return {
        "status": "success",
        "scope": "stdlib CPU synthetic records only; no GPU/performance evidence",
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
        result = {"status": "error", "error": f"{type(error).__name__}: {error}"}
        code = 1
    content = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    print(content, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
