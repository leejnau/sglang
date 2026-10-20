#!/usr/bin/env python3
"""Validate retained-graph controls before estimating paired replay effects.

Latency workers and diagnostic observers are separate cohorts. Trial units,
not repeated replay blocks, determine the exploratory bootstrap uncertainty.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import itertools
import math
import random
import re
import statistics
import subprocess
import types
from pathlib import Path

BATCH = "test/manual/kernels/results/flashinfer_autotune_selection/20261020T143705Z-h100-graph-controls"
WORKER = "test/manual/kernels/flashinfer_autotune_graph_controls.py"
HELPER = "test/manual/kernels/results/flashinfer_autotune_selection/20261020T034000Z-h100-fp8-linear/analyze.py"
HELPER_SHA256 = "e551c7508890045395074d7532c7a2b621cc91fbfe9b4b8c23a7e9e7070027d5"
ARMS = ("v1", "v2")
GRAPH_IDS = ("pre_default", "post_hit", "post_miss")
TRIALS = tuple(range(6))


def load_helper():
    root = Path(__file__).resolve().parents[6]
    path = root / HELPER
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != HELPER_SHA256:
        raise ValueError("frozen analysis helper changed")
    module = types.ModuleType("frozen_linear_analysis")
    module.__file__ = str(path)
    # Compile the pinned pure-stdlib source directly; do not write bytecode to
    # an earlier immutable batch or import the experiment's CUDA worker.
    exec(compile(content, str(path), "exec"), module.__dict__)
    return module


h = load_helper()
require, load, canonical, sha256 = h.require, h.load, h.canonical, h.sha256


def statistic(rows):
    """Retain whole-trial estimates and resample those units only."""
    return h.summarize_ratios(rows)


def median_measurement(measurement, channel):
    key = "samples_ms" if channel == "wall" else "gpu_event_samples_ms"
    return statistics.median(measurement[key])


def validate_measurement(row, label):
    iterations = row["effective_iterations"]
    require(
        type(iterations) is int and 5120 <= iterations <= 65536, label + ": iterations"
    )
    require(
        row["requested_iterations"] == 5120
        and row["iterations_per_sample"] == iterations,
        label + ": normalization metadata",
    )
    require(
        h.positive(row["warmup_actual_ms"])
        and row["warmup_actual_ms"] >= 500
        and row["warmup_calls"] >= 5120,
        label + ": warmup",
    )
    calibration = row["calibration_batches"]
    require(
        calibration
        and calibration[0]["iterations"] == 5120
        and calibration[-1]["iterations"] == iterations
        and calibration[-1]["wall_ms"] >= 20,
        label + ": calibration bounds",
    )
    for before, after in zip(calibration, calibration[1:]):
        require(
            before["wall_ms"] < 20
            and after["iterations"] == min(before["iterations"] * 2, 65536),
            label + ": calibration progression",
        )
    for samples, batches in (
        ("samples_ms", "wall_batch_ms"),
        ("gpu_event_samples_ms", "gpu_event_batch_ms"),
    ):
        require(len(row[samples]) == len(row[batches]) == 15, label + ": sample count")
        for sample, batch in zip(row[samples], row[batches]):
            require(
                h.positive(sample)
                and h.positive(batch)
                and math.isclose(sample * iterations, batch, rel_tol=1e-12),
                label + ": raw timing normalization",
            )
    require(
        math.isclose(
            row["median_ms"], statistics.median(row["samples_ms"]), rel_tol=1e-12
        ),
        label + ": reported median",
    )


def validate_batch(row, label):
    require(row["iterations"] == 5120, label + ": fixed replay count")
    for batch, per_call in (
        ("wall_batch_ms", "wall_per_call_ms"),
        ("gpu_event_batch_ms", "gpu_event_per_call_ms"),
    ):
        require(
            h.positive(row[batch])
            and h.positive(row[per_call])
            and math.isclose(row[batch], row[per_call] * 5120, rel_tol=1e-12),
            label + ": raw batch normalization",
        )
    require(
        h.positive(row["rewarm_actual_ms"])
        and row["rewarm_actual_ms"] >= 20
        and type(row["rewarm_calls"]) is int
        and row["rewarm_calls"] >= 5120
        and row["rewarm_calls"] % 5120 == 0,
        label + ": replay rewarm",
    )


def expected_schedule(trial):
    permutations = list(itertools.permutations(GRAPH_IDS))
    schedule = [list(order) for _ in range(4) for order in permutations]
    random.Random(4000 + trial).shuffle(schedule)
    return schedule


def correlation(xs, ys):
    if len(xs) < 2:
        return None
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    xx = sum((x - mean_x) ** 2 for x in xs)
    yy = sum((y - mean_y) ** 2 for y in ys)
    if xx == 0 or yy == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / math.sqrt(xx * yy)


def clock_mhz(snapshot):
    # Snapshots are outside latency windows; their association with latency is
    # descriptive and does not identify clocks as the cause of a difference.
    fields = snapshot.split(",") if isinstance(snapshot, str) else []
    if len(fields) < 5:
        return None
    match = re.fullmatch(r"\s*([0-9.]+)\s*MHz\s*", fields[4])
    return float(match.group(1)) if match else None


def interval(row, label):
    begin, end = row["started_ns"], row["ended_ns"]
    require(
        type(begin) is int and type(end) is int and 0 < begin < end,
        label + ": chronology",
    )
    return begin, end


def validate_number(row, label):
    quant, dense = row["quantized_reference"], row["bf16_reference"]
    require(
        quant["passed"] is True and dense["diagnostic_only"] is True,
        label + ": reference roles",
    )
    for metrics in (quant, dense):
        require(
            all(
                type(metrics[key]) in (int, float)
                and math.isfinite(metrics[key])
                and metrics[key] >= 0
                for key in (
                    "max_abs_error",
                    "max_normalized_error",
                    "rmse",
                    "relative_l2",
                )
            ),
            label + ": finite numerical metrics",
        )
    require(
        quant["max_normalized_error"] <= 1
        and row["max_abs_error"] == quant["max_abs_error"]
        and h.valid_sha(row["output_sha256"]),
        label + ": numerical evidence",
    )
    zero = row["check"].endswith("_changed_input")
    require(row["exact_zero_required"] is zero, label + ": changed-input zero guard")
    if zero:
        require(
            quant["max_abs_error"] == dense["max_abs_error"] == 0,
            label + ": zero-input replay",
        )


def validate_blocks(controls, trial, check_graph, label):
    require(
        controls["common_iterations"] == 5120
        and controls["schedule_seed"] == 4000 + trial,
        label + ": fixed protocol",
    )
    schedule = expected_schedule(trial)
    require(controls["schedule"] == schedule, label + ": balanced seeded schedule")
    blocks = controls["blocks"]
    require(len(blocks) == 24, label + ": block count")
    previous_end = 0
    for index, (block, order) in enumerate(zip(blocks, schedule)):
        require(
            block["index"] == index
            and block["order"] == order
            and [row["graph_id"] for row in block["measurements"]] == order,
            label + ": block order/coverage",
        )
        for row in block["measurements"]:
            stage = f"block_{index:02d}_{row['graph_id']}"
            require(
                row["correctness_check"] == stage, label + ": timed-output check label"
            )
            validate_batch(row, label + "/" + stage)
            begin, end = interval(row, label + "/" + stage)
            require(
                begin > previous_end, label + ": overlapping/reordered timed batches"
            )
            require(
                math.isclose((end - begin) / 1e6, row["wall_batch_ms"], rel_tol=1e-12),
                label + ": batch clock normalization",
            )
            previous_end = end
            check_graph(stage, row["graph_id"])
    return previous_end


def validate_metadata(data, args, provenance, mode, label):
    fixture = provenance["fixture"]
    require(
        data["status"] == "success"
        and data["schema_version"] == 1
        and data["implementation"] == "candidate"
        and data["mode"] == mode
        and data["arm"] in ARMS
        and data["phase"] == "cold"
        and data["seed"] == 19
        and data["shape"] == fixture["shape"],
        label + ": identity",
    )
    require(
        data["worker_sha256"] == h.source_hash(args, WORKER)
        and data["manifest_sha256"] == sha256(args.source_manifest),
        label + ": source binding",
    )
    env = data["environment"]
    require(
        env["sglang_commit"] == args.expected_source_commit
        and env["sglang_wrapper_sha256"] == h.source_hash(args, h.WRAPPER),
        label + ": checkout",
    )
    require(
        (env["torch"], env["torch_cuda"], env["flashinfer"])
        == ("2.14.1+cu130", "13.0", "0.7.2")
        and env["backend"] == "cublas"
        and env["measurement_policy"] == "unspecified"
        and env["managed_store"] is False,
        label + ": software/policy/store",
    )
    expected_gpu = provenance["plan"]["environment"]
    require(
        env["gpu"] == expected_gpu["gpu"]
        and env["capability"] == [9, 0]
        and env["physical_gpu_uuid"].removeprefix("GPU-")
        == expected_gpu["physical_gpu_uuid"],
        label + ": GPU",
    )
    require(
        all(
            expected_gpu["driver"] in env[key]
            for key in ("gpu_status_before", "gpu_status_after")
        ),
        label + ": driver",
    )
    sources = h.expected_sources(provenance["source_manifest"])
    native = env["native_artifact"]
    require(
        native["source_sha256"] == sources
        and env["flashinfer_tuner_sha256"]
        == sources["flashinfer/autotuner/autotuner.py"]
        and native["cuda_arch_list"] == "9.0"
        and "bmm_fp8_run_with_descriptor" in native["required_native_exports"],
        label + ": qualified native source/build",
    )
    require(
        h.valid_sha(native["library_sha256"]) and h.valid_sha(native["ninja_sha256"]),
        label + ": native artifact hashes",
    )
    helpers = {
        "flashinfer_autotune_linear_layer.py",
        "flashinfer_autotune_descriptor_selection.py",
        "flashinfer_autotune_descriptor_qualification.py",
        "flashinfer_autotune_eager_setup.py",
    }
    require(set(env["helper_sha256"]) == helpers, label + ": helper coverage")
    for name, value in env["helper_sha256"].items():
        require(
            value == h.source_hash(args, "test/manual/kernels/" + name),
            label + ": helper source",
        )
    require(
        env["operand_sha256"]
        == {name: row["sha256"] for name, row in fixture["tensors"].items()},
        label + ": actual fixture operands",
    )
    observed_fixture = data["fixture"]
    require(
        observed_fixture["manifest_sha256"] == sha256(args.fixture_manifest)
        and observed_fixture["tensor_file_sha256"] == fixture["tensor_file"]["sha256"]
        and all(
            observed_fixture[key] == fixture[key]
            for key in ("shape", "model", "capture", "tensors")
        ),
        label + ": immutable loaded fixture",
    )
    layer = data["actual_layer"]
    require(
        layer["function"] == "apply_fp8_linear_bmm_flashinfer"
        and layer["custom_op"] == "flashinfer_bmm_fp8"
        and layer["quantizer"] == "static_quant_fp8"
        and layer["source_sha256"] == fixture["source"]["sglang_source_sha256"],
        label + ": actual production path",
    )
    require(
        layer["inference_mode"] is True
        and layer["grad_enabled"] is False
        and layer["allow_tf32"] is False
        and layer["dispatch_modes"] == 0
        and not layer["function_modes"]
        and not layer["torch_profiler_enabled"]
        and not layer["compiling"]
        and layer["kernel_api_log_level"] == 0
        and layer["custom_op_direct_wrapper"] is True,
        label + ": inference fast path",
    )
    quant = layer["quantization_reference"]
    require(
        quant["byte_match"] is True
        and quant["cpu_and_gpu_independent_references_match"] is True
        and quant["actual_input_fp8_sha256"]
        == quant["independent_input_fp8_sha256"]
        == fixture["quantization"]["independent_input_fp8_sha256"]
        and quant["scale_sha256"] == fixture["tensors"]["input_scale"]["sha256"],
        label + ": real static quantizer bytes",
    )
    require(
        bool(env["LD_PRELOAD"]) is (mode == "observe"),
        label + ": timing instrumentation",
    )
    h.validate_settings(data, label)


def validate_profile_cache(data, args, label):
    profiles, summary = data["profiles"], data["runtime_profile_summary"]
    require(
        data["profile_attempts"] == len(profiles) > 0
        and not data["forbidden_profile_attempts"],
        label + ": profiling attempt accounting",
    )
    m, n, k = data["shape"]
    expected = [
        [1, 1 << (m - 1).bit_length(), k],
        [1, k, n],
        [1],
        [1],
        [1, -1, n],
        [-1],
    ]
    runtime = []
    for row in profiles:
        require(
            row["stage"] == "startup"
            and row["runner"] == "CublasFp8GemmRunner"
            and h.valid_tactic(row["tactic"]),
            label + ": profiler scope",
        )
        require(
            not row["use_cuda_graph"]
            and not row["use_cold_l2_cache"]
            and row["autotuner_settings"] == data["autotuner_settings"]["after"],
            label + ": actual profiling objective",
        )
        require(
            row["normalized_profile"] == h.normalized_profile(row["input_shapes"]),
            label + ": canonical profile",
        )
        require(
            row.get("profiler_ms") is None or h.positive(row["profiler_ms"]),
            label + ": profile outcome",
        )
        if row["normalized_profile"] == expected:
            runtime.append(row)
    require(
        {row["input_shapes"][0][-2] for row in profiles}
        == {1, 2, 4, 8, 16, 32, 64, 128, 144},
        label + ": startup bucket coverage",
    )
    finite = [row for row in runtime if row.get("profiler_ms") is not None]
    require(finite, label + ": runtime profile finite coverage")
    minimum = min(row["profiler_ms"] for row in finite)
    require(
        summary
        == {
            "normalized_profile": expected,
            "profiles": len(runtime),
            "finite_profiles": len(finite),
            "minimum_ms": minimum,
            "minimum_tactics": [
                r["tactic"] for r in finite if r["profiler_ms"] == minimum
            ],
        },
        label + ": profiler minimum summary",
    )
    winner = data["natural"]["tactic"]
    require(
        winner == min(finite, key=lambda row: row["profiler_ms"])["tactic"],
        label + ": first finite winner",
    )
    if data["arm"] == "v2":
        require(
            runtime[0]["tactic"] == -1
            and any(
                row["tactic"] == -1 and row.get("profiler_ms") is not None
                for row in runtime
            ),
            label + ": default candidate participation",
        )
    else:
        require(
            all(row["tactic"] != -1 for row in profiles),
            label + ": v1 default exclusion",
        )
    require(
        data["natural"]["cache_hit"] is True
        and data["natural"]["lookup_tier"] == "memory",
        label + ": natural memory winner",
    )
    h.validate_acceptance(
        data["natural"]["acceptance"], winner, label + ": native natural winner"
    )
    cache = data["cache"]
    require(
        cache["before_sha256"] is None and cache["miss_scopes_restored"] is True,
        label + ": initially cold/reversible cache",
    )
    path = h.locate(args.raw_root, cache["path"])
    require(
        sha256(path) == cache["after_startup_sha256"] == cache["after_controls_sha256"],
        label + ": persisted cache bytes changed",
    )
    payload = load(path)
    h.validate_generation(payload, label)
    key = (
        "('fp8_gemm', 'CublasFp8GemmRunner', "
        + repr(tuple(tuple(s) for s in expected))
        + ", (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.bfloat16))"
    )
    require(
        cache["runtime_file_key"] == key and payload[key][1] == winner,
        label + ": persisted natural tactic",
    )
    return winner


def validate_graphs(data, winner, label):
    controls = data["graph_controls"]
    order = ["pre_default"] + (
        ["post_hit", "post_miss"]
        if data["trial"] % 2 == 0
        else ["post_miss", "post_hit"]
    )
    require(
        controls["capture_order"] == order
        and [row["id"] for row in controls["graphs"]] == order,
        label + ": capture order",
    )
    by_id = {row["id"]: row for row in controls["graphs"]}
    require(
        len({row["output"]["data_ptr"] for row in by_id.values()}) == 3,
        label + ": retained graphs alias output allocations",
    )
    m, n, k = data["shape"]
    require(len(controls["miss_scope"]) == 1, label + ": miss-scope coverage")
    scope = controls["miss_scope"][0]
    require(
        scope["winner_entries_before"] > 0
        and scope["winner_entries_after_removal"]
        == scope["file_entries_after_removal"]
        == 0
        and scope["restored"] is True
        and scope["all_cache_state_restored"] is True
        and scope["cache_sha256_before"]
        == scope["cache_sha256_after"]
        == data["cache"]["after_controls_sha256"]
        and scope["winner_contents_sha256_before"]
        == scope["winner_contents_sha256_after"]
        and scope["file_contents_sha256_before"] == scope["file_contents_sha256_after"],
        label + ": effective cache removal/restoration",
    )
    previous_end = 0
    for graph_id in order:
        row = by_id[graph_id]
        expected_tactic, tier = (
            (winner, "memory") if graph_id == "post_hit" else (-1, "miss")
        )
        require(
            row["tactic"] == expected_tactic and row["lookup_tier"] == tier,
            label + ": graph choice/source",
        )
        selection = row["selection"]
        require(
            selection["stage"] == graph_id + "_capture"
            and selection["tactic"] == expected_tactic
            and selection["lookup_tier"] == tier
            and selection["cache_hit"] is (tier != "miss")
            and selection["capturing"] is True,
            label + ": capture actual lookup",
        )
        begin, end = interval(row, label + "/" + graph_id)
        require(begin > previous_end, label + ": capture chronology")
        previous_end = end
        output, inputs = row["output"], row["inputs"]
        require(
            output["shape"] == [m, n]
            and output["stride"] == [n, 1]
            and output["dtype"] == "torch.bfloat16"
            and len(inputs) == 6
            and inputs[4]["data_ptr"] == output["data_ptr"]
            and inputs[0]["shape"] == [1, m, k]
            and inputs[1]["shape"] == [1, k, n]
            and inputs[0]["dtype"] == inputs[1]["dtype"] == "torch.float8_e4m3fn"
            and inputs[1]["stride"][-2:] == [1, k],
            label + ": actual graph operands/output",
        )
        for tensor in [output, row["linear_input"], *inputs]:
            pointer = tensor["data_ptr"]
            require(
                type(pointer) is int
                and pointer > 0
                and tensor["alignment_mod256"] == pointer % 256
                and tensor["alignment_mod4096"] == pointer % 4096,
                label + ": retained allocation metadata",
            )
        require(
            row["linear_input"]["shape"] == [m, k]
            and row["linear_input"]["dtype"] == "torch.bfloat16",
            label + ": actual BF16 input",
        )
        fixture = data["fixture"]["tensors"]
        require(
            row["quantized_input_sha256"]
            == data["actual_layer"]["quantization_reference"]["actual_input_fp8_sha256"]
            and row["weight_rowmajor_sha256"]
            == fixture["weight_fp8_rowmajor"]["sha256"]
            and row["input_scale_sha256"] == fixture["input_scale"]["sha256"]
            and row["weight_scale_sha256"] == fixture["weight_scale"]["sha256"],
            label + ": capture operand bytes",
        )
    require(
        len({row["linear_input"]["data_ptr"] for row in by_id.values()}) == 1
        and all(
            len({row["inputs"][index]["data_ptr"] for row in by_id.values()}) == 1
            for index in (1, 2, 3)
        ),
        label + ": graphs do not share actual fixture allocations",
    )
    require(
        by_id["pre_default"]["ended_ns"]
        < controls["startup_started_ns"]
        < controls["startup_ended_ns"]
        < min(by_id[name]["started_ns"] for name in ("post_hit", "post_miss")),
        label + ": startup/capture ordering",
    )
    return by_id


def validate_stage_coverage(rows, expected, label):
    # Counter(mapping) interprets mapping values as counts. Stage metadata
    # values are tuples, so coverage must explicitly count the mapping keys.
    require(
        collections.Counter(row["stage"] for row in rows)
        == collections.Counter(expected.keys()),
        label,
    )


def validate_observed_stages(data, graphs, winner, label):
    expected = {
        "setup": (-1, "miss", False),
        "startup": (winner, "memory", False),
        "natural_after_startup": (winner, "memory", False),
        "natural_final": (winner, "memory", False),
    }
    for graph_id in GRAPH_IDS:
        tactic, tier = (winner, "memory") if graph_id == "post_hit" else (-1, "miss")
        expected[graph_id + "_before"] = (tactic, tier, False)
        expected[graph_id + "_capture"] = (tactic, tier, True)
        if data["mode"] == "observe":
            for index in range(3):
                expected[f"{graph_id}_warm_eager_{index}"] = (tactic, tier, False)
    selections = data["selections"]
    validate_stage_coverage(selections, expected, label + ": lookup stage coverage")
    for row in selections:
        tactic, tier, capture = expected[row["stage"]]
        require(
            row["tactic"] == tactic
            and row["lookup_tier"] == tier
            and row["cache_hit"] is (tier != "miss")
            and row["capturing"] is capture,
            label + ": actual selected tactic/source",
        )
        require(
            row["on_forward_stream"] and not row["on_caller_stream"]
            if row["stage"] == "startup"
            else capture or row["on_caller_stream"],
            label + ": stream restoration",
        )
    binding_stages = set(expected) - {"setup", "startup"}
    bindings = data["binding_acceptance_checks"]
    require(
        collections.Counter(row["stage"] for row in bindings)
        == collections.Counter(binding_stages),
        label + ": native acceptance coverage",
    )
    for row in bindings:
        h.validate_acceptance(
            row, expected[row["stage"]][0], label + ": native acceptance"
        )
    layer = data["actual_layer"]
    validate_stage_coverage(
        layer["observed_public_inputs"],
        expected,
        label + ": actual operand observation coverage",
    )
    for row in layer["observed_public_inputs"]:
        capture = expected[row["stage"]][2]
        require(
            row["capturing"] is capture and row["backend"] == "cublas",
            label + ": actual backend",
        )
        if not capture and row["stage"] != "startup":
            require(
                row["quantized_input_sha256"]
                == layer["quantization_reference"]["actual_input_fp8_sha256"]
                and row["weight_rowmajor_sha256"]
                == data["fixture"]["tensors"]["weight_fp8_rowmajor"]["sha256"],
                label + ": observed public bytes",
            )
    poison = layer["native_output_poison_checks"]
    require(
        collections.Counter(row["stage"] for row in poison)
        == collections.Counter(binding_stages),
        label + ": actual allocation poison coverage",
    )
    for row in poison:
        capture = expected[row["stage"]][2]
        require(
            row["capturing"] is capture
            and row["poisoned_actual_allocation"] is not capture,
            label + ": capture contains poison",
        )
    checks = data["correctness"]["checks"]
    require(
        data["correctness"]["passed"] is True
        and (data["correctness"]["rtol"], data["correctness"]["atol"]) == (0.02, 0.125),
        label + ": numerical protocol",
    )
    by_check = {row["check"]: row for row in checks}
    require(len(by_check) == len(checks), label + ": duplicate numerical check")
    for row in checks:
        validate_number(row, label)
        require(
            type(row["output_data_ptr"]) is int and row["output_data_ptr"] > 0,
            label + ": checked allocation pointer",
        )
    expected_checks = {
        "initial_untuned",
        "pre_default_before",
        "natural_after_startup",
        "natural_final",
        "post_hit_before",
        "post_miss_before",
        "history_before_startup",
        "history_after_startup_pre_capture",
    }
    for graph_id in GRAPH_IDS:
        expected_checks.update(
            graph_id + suffix for suffix in ("_changed_input", "_restored_input")
        )
        expected_checks.update(
            prefix + graph_id for prefix in ("sequential_", "final_")
        )
    if data["mode"] == "observe":
        expected_checks.update("traced_" + name for name in GRAPH_IDS)
        expected_checks.update(
            f"{name}_warm_eager_{index}" for name in GRAPH_IDS for index in range(3)
        )
    else:
        expected_checks.update(
            f"block_{index:02d}_{name}" for index in range(24) for name in GRAPH_IDS
        )
    require(set(by_check) == expected_checks, label + ": numerical lifecycle coverage")

    def check_graph(stage, graph_id):
        row = by_check[stage]
        require(
            row["graph_id"] == graph_id
            and row["output_data_ptr"] == graphs[graph_id]["output"]["data_ptr"],
            label + ": checked wrong graph output",
        )

    for graph_id in GRAPH_IDS:
        for stage in (
            graph_id + "_changed_input",
            graph_id + "_restored_input",
            "sequential_" + graph_id,
            "final_" + graph_id,
        ):
            check_graph(stage, graph_id)
        if data["mode"] == "observe":
            check_graph("traced_" + graph_id, graph_id)
    for stage in ("history_before_startup", "history_after_startup_pre_capture"):
        check_graph(stage, "pre_default")
    for stage in set(by_check) & set(expected):
        require(
            by_check[stage]["graph_id"] is None,
            label + ": eager check substitutes retained graph",
        )
    return check_graph


def validate_timing_controls(data, graphs, check_graph, label):
    controls = data["graph_controls"]
    if data["mode"] == "observe":
        require(
            controls["history_before_startup"]
            is controls["history_after_startup_pre_capture"]
            is None
            and controls["sequential"]
            == controls["blocks"]
            == controls["schedule"]
            == [],
            label + ": instrumented timing entered latency cohort",
        )
        return
    windows = [
        controls["history_before_startup"],
        controls["history_after_startup_pre_capture"],
    ]
    require(
        all(row["graph_id"] == "pre_default" for row in windows),
        label + ": history graph replaced",
    )
    require(
        [row["graph_id"] for row in controls["sequential"]] == list(GRAPH_IDS),
        label + ": sequential coverage",
    )
    windows += controls["sequential"]
    for row in windows:
        validate_measurement(row["measurement"], label + "/" + row["correctness_check"])
        begin, end = interval(row, label)
        require(
            (end - begin) / 1e6 >= sum(row["measurement"]["wall_batch_ms"]),
            label + ": measured window chronology",
        )
        check_graph(row["correctness_check"], row["graph_id"])
    before, after = windows[:2]
    require(
        graphs["pre_default"]["ended_ns"]
        < before["started_ns"]
        < before["ended_ns"]
        < controls["startup_started_ns"]
        < controls["startup_ended_ns"]
        < after["started_ns"]
        < after["ended_ns"]
        < min(graphs[name]["started_ns"] for name in ("post_hit", "post_miss")),
        label + ": retained history windows reordered",
    )
    require(
        max(row["ended_ns"] for row in graphs.values()) < windows[2]["started_ns"]
        and all(
            a["ended_ns"] < b["started_ns"] for a, b in zip(windows[2:], windows[3:])
        ),
        label + ": sequential capture/window order",
    )
    validate_blocks(controls, data["trial"], check_graph, label)
    require(
        controls["blocks"][0]["measurements"][0]["started_ns"]
        > windows[-1]["ended_ns"],
        label + ": interleaving precedes sequential windows",
    )


def kernel_signature(event):
    args = event["args"]
    signature = {"name": event["name"]}
    for key in ("grid", "block", "registers per thread", "shared memory"):
        signature[key] = args[key]
    require(
        all(
            len(signature[key]) == 3
            and all(type(v) is int and v > 0 for v in signature[key])
            for key in ("grid", "block")
        ),
        "trace launch dimensions",
    )
    return signature


def validate_trace(trace, graph_ids=GRAPH_IDS):
    events = trace["traceEvents"]
    result, all_graph_ids = {}, []
    for graph_id in graph_ids:
        scopes = [
            e
            for e in events
            if e.get("name") == "retained_" + graph_id
            and e.get("cat") == "user_annotation"
            and e.get("ph") == "X"
        ]
        require(len(scopes) == 1, "trace retained graph scope")
        scope = scopes[0]
        begin, end = scope["ts"], scope["ts"] + scope["dur"]
        launches = sorted(
            [
                e
                for e in events
                if e.get("cat") == "cuda_runtime"
                and "cudaGraphLaunch" in e.get("name", "")
                and begin <= e["ts"] <= end
            ],
            key=lambda e: e["ts"],
        )
        require(len(launches) == 3, "trace three actual retained graph launches")
        correlations = [e["args"]["correlation"] for e in launches]
        require(len(set(correlations)) == 3, "trace launch correlation identity")
        cycles = []
        graph_identity = set()
        for correlation_id in correlations:
            kernels = sorted(
                [
                    e
                    for e in events
                    if e.get("cat") == "kernel"
                    and e.get("args", {}).get("correlation") == correlation_id
                ],
                key=lambda e: e["ts"],
            )
            require(
                len(kernels) >= 2
                and all(
                    begin <= e["ts"] and e["ts"] + e["dur"] <= end + 1 for e in kernels
                ),
                "trace launch kernels/scoped lifetime",
            )
            require(
                any("_static_quant_fp8" in e["name"] for e in kernels)
                and any("_static_quant_fp8" not in e["name"] for e in kernels),
                "trace actual quantizer and matrix operation",
            )
            graph_identity.update(e["args"]["graph id"] for e in kernels)
            cycles.append([kernel_signature(e) for e in kernels])
        require(
            cycles[0] == cycles[1] == cycles[2],
            "trace repeated ordered launch signatures",
        )
        require(
            len(graph_identity) == 1 and next(iter(graph_identity)) > 0,
            "trace retained graph identity",
        )
        all_graph_ids.extend(graph_identity)
        result[graph_id] = {
            "ordered_launch_signature": cycles[0],
            "graph_id": next(iter(graph_identity)),
            "graph_launches": 3,
            "kernel_events": sum(len(cycle) for cycle in cycles),
        }
    require(
        len(set(all_graph_ids)) == len(graph_ids),
        "trace replayed same graph under multiple labels",
    )
    return result


def validate_native_rows(records, graphs):
    summaries = {r["phase"]: r for r in records if r["event"] == "phase_summary"}
    require(
        len(summaries) == sum(r["event"] == "phase_summary" for r in records),
        "duplicate native phase summary",
    )
    require(len({r["pid"] for r in records}) == 1, "native log process identity")
    identities = {}
    attributes = {
        "id",
        "tile_id",
        "split_k",
        "reduction_scheme",
        "cta_swizzling",
        "custom_option",
        "stages_id",
    }
    for graph_id, graph in graphs.items():
        phase = graph_id + "_capture"
        calls = [r for r in records if r["event"] == "matmul" and r["phase"] == phase]
        heuristics = [
            r for r in records if r["event"] == "heuristic" and r["phase"] == phase
        ]
        require(
            len(calls) == 1 and calls[0]["status"] == 0,
            "native capture actual successful matmul",
        )
        call, api = calls[0], summaries[phase]["apis"]
        require(
            api["matmul"]["calls"] == 1
            and api["matmul"]["failures"] == 0
            and api["heuristic"]["calls"] == len(heuristics),
            "native capture counters",
        )
        algo = call["algo"]
        require(
            algo["raw_bytes"] == 64
            and re.fullmatch("[0-9a-f]{128}", algo["raw_hex"]) is not None
            and set(algo["config"]) == attributes,
            "native complete algorithm bytes/attributes",
        )
        require(
            all(
                row["status"] == 0 and row["bytes"] == 4 and type(row["value"]) is int
                for row in algo["config"].values()
            ),
            "native public config query status",
        )
        if graph["tactic"] == -1:
            require(
                len(heuristics) == 1
                and heuristics[0]["status"] == 0
                and heuristics[0]["requested"] == heuristics[0]["returned"] == 1
                and heuristics[0]["algorithms"][0]["state"] == 0
                and heuristics[0]["algorithms"][0]["algo"] == algo,
                "native default heuristic/submitted algorithm",
            )
        else:
            expected_hex = b"".join(
                word.to_bytes(8, "little") for word in graph["tactic"]
            ).hex()
            require(
                not heuristics and algo["raw_hex"] == expected_hex,
                "native accepted descriptor identity",
            )
        identities[graph_id] = {
            "algorithm": algo,
            "workspace_bytes": call["workspace_bytes"],
            "heuristic_calls": len(heuristics),
            "matmul_calls": 1,
        }
    replay_phases = [name + "_numerics" for name in GRAPH_IDS]
    replay_phases += ["history_before_startup", "history_after_startup_pre_capture"]
    replay_phases += [
        prefix + name for prefix in ("sequential_", "trace_") for name in GRAPH_IDS
    ]
    for phase in replay_phases:
        api = summaries[phase]["apis"]
        require(
            api["matmul"]["calls"] == api["heuristic"]["calls"] == 0,
            "graph replay performed a Python/native selection call",
        )
    return identities


def validate_observer(data, args, graphs, label):
    observer = data["observer"]
    enabled = data["mode"] == "observe"
    require(
        observer["enabled"]
        is observer["constructor_guard_suspended"]
        is observer["real_interposition_remained_loaded"]
        is enabled,
        label + ": observer guard scope",
    )
    if not enabled:
        require(
            not observer["preload_files"]
            and observer["trace"] is None
            and not observer["phases"],
            label + ": instrumentation in timing worker",
        )
        return None
    require(observer["preload_files"], label + ": actual process-start probe")
    for row in observer["preload_files"]:
        require(h.valid_sha(row["sha256"]), label + ": probe ELF identity")
        candidates = list(args.raw_root.rglob(Path(row["path"]).name))
        if len(candidates) == 1:
            require(
                sha256(candidates[0]) == row["sha256"],
                label + ": retained probe ELF changed",
            )
    phases = observer["phases"]
    expected = {"history_before_startup", "history_after_startup_pre_capture"} | {
        "sequential_" + name for name in GRAPH_IDS
    }
    require(
        {r["phase"] for r in phases} == expected
        and len(phases) == len(expected)
        and all(r["calls"] == 3 for r in phases),
        label + ": diagnostic replay phases",
    )
    native_path = h.locate(args.raw_root, observer["native_output_path"])
    records = [
        h.json.loads(line)
        for line in native_path.read_text().splitlines()
        if line.strip()
    ]
    native = validate_native_rows(records, graphs)
    trace_info = observer["trace"]
    trace_path = h.locate(args.raw_root, trace_info["path"])
    require(
        sha256(trace_path) == trace_info["sha256"]
        and trace_info["calls_per_graph"] == 3,
        label + ": original retained trace bytes",
    )
    trace = validate_trace(load(trace_path))
    require(
        sum(r["kernel_events"] for r in trace.values()) == trace_info["kernel_events"]
        and sum(r["graph_launches"] for r in trace.values())
        == trace_info["graph_launch_events"],
        label + ": actual trace count coverage",
    )
    return {
        "native_log_sha256": sha256(native_path),
        "trace_sha256": sha256(trace_path),
        "graphs": {
            name: {"native": native[name], "trace": trace[name]} for name in GRAPH_IDS
        },
        "algorithm_raw_equal": len(
            {row["algorithm"]["raw_hex"] for row in native.values()}
        )
        == 1,
        "ordered_launch_signatures_equal": len(
            {canonical(row["ordered_launch_signature"]) for row in trace.values()}
        )
        == 1,
        "scope": "Separate instrumented process; this does not identify the physical launches in latency workers. AlgoCheck is not instrumented by this probe.",
    }


def validate_outer(args):
    plan_bytes = h.git_blob(args, BATCH + "/plan.json")
    plan, outer = h.json.loads(plan_bytes), load(args.raw_root / "manifest.json")
    require(
        outer["plan_sha256"] == hashlib.sha256(plan_bytes).hexdigest(), "frozen plan"
    )
    require(
        outer["source_manifest_sha256"]
        == sha256(args.source_manifest)
        == plan["flashinfer_source"]["source_manifest_sha256"],
        "overlay manifest",
    )
    require(
        outer["fixture_manifest_sha256"]
        == sha256(args.fixture_manifest)
        == plan["fixture"]["manifest_sha256"],
        "fixture manifest",
    )
    fixture = load(args.fixture_manifest)
    h.validate_fixture_manifest(fixture, args)
    require(
        sha256(args.raw_root / "fixture" / "manifest.json")
        == sha256(args.fixture_manifest)
        and outer["fixture_tensor_file_sha256"]
        == fixture["tensor_file"]["sha256"]
        == plan["fixture"]["tensor_file_sha256"],
        "archived fixture binding",
    )
    private = args.raw_root / "fixture" / fixture["tensor_file"]["name"]
    if private.is_file():
        require(
            sha256(private) == fixture["tensor_file"]["sha256"]
            and private.stat().st_size == fixture["tensor_file"]["size_bytes"],
            "private fixture changed",
        )
    require(
        (args.raw_root / "execution-commit.txt").read_text().strip()
        == args.expected_source_commit,
        "execution checkout",
    )
    for path, value in outer["source_sha256"].items():
        require(h.source_hash(args, path) == value, "frozen source: " + path)
    coverage = {WORKER, HELPER, BATCH + "/analyze.py", h.WRAPPER, h.FP8_UTILS}
    if args.observations:
        coverage.add("test/manual/kernels/flashinfer_autotune_eager_setup_native.cpp")
    require(
        coverage <= set(outer["source_sha256"])
        and outer["source_sha256"][HELPER] == HELPER_SHA256,
        "source coverage",
    )
    for trial in args.trials:
        require(
            (args.raw_root / f"trial{trial}.launcher.exit-code").read_text().strip()
            == "0",
            f"incomplete trial{trial}",
        )
    require(
        (args.raw_root / "prepare.exit-code").read_text().strip() == "0"
        and not (args.raw_root / "gpu-processes-before.csv").read_text().strip(),
        "preparation/idle GPU",
    )
    image = load(args.raw_root / "image.json")
    require(
        image["Architecture"] == "amd64"
        and image["Os"] == "linux"
        and any(
            value.split("@")[-1] == plan["container_image"].split("@")[-1]
            for value in image["RepoDigests"]
        ),
        "pinned image",
    )
    gpu = plan["environment"]
    require(
        outer["gpu"]["name"] == gpu["gpu"]
        and outer["gpu"]["uuid"].removeprefix("GPU-") == gpu["physical_gpu_uuid"]
        and outer["gpu"]["driver_version"] == gpu["driver"],
        "physical device",
    )
    return {
        "plan": plan,
        "fixture": fixture,
        "source_manifest": load(args.source_manifest),
        "manifest_sha256": sha256(args.raw_root / "manifest.json"),
        "private_fixture_rehashed": private.is_file(),
        "gpu": outer["gpu"],
    }


def validate_worker(path, args, provenance, mode):
    data, label = load(path), path.name
    validate_metadata(data, args, provenance, mode, label)
    winner = validate_profile_cache(data, args, label)
    graphs = validate_graphs(data, winner, label)
    check_graph = validate_observed_stages(data, graphs, winner, label)
    validate_timing_controls(data, graphs, check_graph, label)
    diagnostic = validate_observer(data, args, graphs, label)
    require(
        path.with_suffix(".exit-code").read_text().strip() == "0",
        label + ": container exit",
    )
    cid = path.with_suffix(".container-id").read_text().strip()
    require(re.fullmatch("[0-9a-f]{64}", cid) is not None, label + ": fresh container")
    data.update(_path=str(path), _container_id=cid, _observation=diagnostic)
    return data


def worker_effects(data):
    controls, rows = data["graph_controls"], []
    for channel, key in (
        ("gpu_event", "gpu_event_per_call_ms"),
        ("wall", "wall_per_call_ms"),
    ):
        maps = [
            {row["graph_id"]: row[key] for row in block["measurements"]}
            for block in controls["blocks"]
        ]
        for numerator, denominator in (
            ("post_miss", "pre_default"),
            ("post_hit", "post_miss"),
            ("post_hit", "pre_default"),
        ):
            ratios = [row[numerator] / row[denominator] for row in maps]
            rows.append(
                {
                    "trial": data["trial"],
                    "arm": data["arm"],
                    "channel": channel,
                    "primary": channel == "gpu_event",
                    "kind": "interleaved_paired_blocks",
                    "comparison": numerator + "_over_" + denominator,
                    "ratio": statistics.median(ratios),
                    "paired_block_ratios": ratios,
                    "same_public_tactic": numerator != "post_hit"
                    or data["natural"]["tactic"] == -1,
                    "block_index_correlation": correlation(list(range(24)), ratios),
                }
            )
        before = controls["history_before_startup"]["measurement"]
        after = controls["history_after_startup_pre_capture"]["measurement"]
        sequential = {
            row["graph_id"]: row["measurement"] for row in controls["sequential"]
        }
        rows.append(
            {
                "trial": data["trial"],
                "arm": data["arm"],
                "channel": channel,
                "primary": channel == "gpu_event",
                "kind": "chronological_retained_graph",
                "comparison": "after_startup_before_new_captures_over_before_startup",
                "ratio": median_measurement(after, channel)
                / median_measurement(before, channel),
            }
        )
        rows.append(
            {
                "trial": data["trial"],
                "arm": data["arm"],
                "channel": channel,
                "primary": channel == "gpu_event",
                "kind": "chronological_retained_graph",
                "comparison": "after_new_captures_over_after_startup_before_new_captures",
                "ratio": median_measurement(sequential["pre_default"], channel)
                / median_measurement(after, channel),
            }
        )
        for numerator, denominator in (
            ("post_miss", "pre_default"),
            ("post_hit", "post_miss"),
            ("post_hit", "pre_default"),
        ):
            rows.append(
                {
                    "trial": data["trial"],
                    "arm": data["arm"],
                    "channel": channel,
                    "primary": channel == "gpu_event",
                    "kind": "separate_sequential_windows",
                    "comparison": numerator + "_over_" + denominator,
                    "ratio": median_measurement(sequential[numerator], channel)
                    / median_measurement(sequential[denominator], channel),
                    "same_public_tactic": numerator != "post_hit"
                    or data["natural"]["tactic"] == -1,
                }
            )
    return rows


def order_context(data):
    contexts = []
    blocks = data["graph_controls"]["blocks"]
    origin = min(row["started_ns"] for block in blocks for row in block["measurements"])
    for graph_id in GRAPH_IDS:
        selected = [
            (
                block["index"],
                block["order"].index(graph_id),
                next(
                    row for row in block["measurements"] if row["graph_id"] == graph_id
                ),
            )
            for block in blocks
        ]
        for channel, key in (
            ("gpu_event", "gpu_event_per_call_ms"),
            ("wall", "wall_per_call_ms"),
        ):
            values = [row[key] for _, _, row in selected]
            clocks = [clock_mhz(row["gpu_before"]) for _, _, row in selected]
            valid_clock = [
                (clock, value)
                for clock, value in zip(clocks, values)
                if clock is not None
            ]
            contexts.append(
                {
                    "trial": data["trial"],
                    "arm": data["arm"],
                    "graph_id": graph_id,
                    "channel": channel,
                    "latency_vs_block_index": correlation(
                        [i for i, _, _ in selected], values
                    ),
                    "latency_vs_position": correlation(
                        [position for _, position, _ in selected], values
                    ),
                    "latency_vs_elapsed_seconds": correlation(
                        [(row["started_ns"] - origin) / 1e9 for _, _, row in selected],
                        values,
                    ),
                    "latency_vs_clock_before": correlation(
                        [p[0] for p in valid_clock], [p[1] for p in valid_clock]
                    ),
                    "scope": "Descriptive within-worker associations; snapshots are outside latency and do not establish a clock, order or tuning cause.",
                }
            )
    return contexts


def summarize(workers):
    effects = [row for worker in workers for row in worker_effects(worker)]
    grouped = collections.defaultdict(list)
    for row in effects:
        grouped[row["arm"], row["channel"], row["kind"], row["comparison"]].append(row)
    summaries = []
    for (arm, channel, kind, comparison), rows in grouped.items():
        summaries.append(
            {
                "arm": arm,
                "channel": channel,
                "primary": channel == "gpu_event",
                "kind": kind,
                "comparison": comparison,
                "all_same_public_default": all(
                    row.get("same_public_tactic", True) for row in rows
                ),
                **statistic(
                    [{"trial": row["trial"], "ratio": row["ratio"]} for row in rows]
                ),
            }
        )
    profiles = []
    for data in workers:
        expected = data["runtime_profile_summary"]["normalized_profile"]
        finite = sorted(
            [
                row
                for row in data["profiles"]
                if row["normalized_profile"] == expected
                and row.get("profiler_ms") is not None
            ],
            key=lambda r: r["profiler_ms"],
        )
        profiles.append(
            {
                "trial": data["trial"],
                "arm": data["arm"],
                "winner": data["natural"]["tactic"],
                "ranked_profiles": [
                    {"tactic": r["tactic"], "profiler_ms": r["profiler_ms"]}
                    for r in finite
                ],
                "runner_up_fractional_margin": None
                if len(finite) < 2
                else finite[1]["profiler_ms"] / finite[0]["profiler_ms"] - 1,
                "failed_profiles": [
                    r for r in data["profiles"] if r.get("profiler_ms") is None
                ],
            }
        )
    return {
        "worker_effects": effects,
        "trial_summaries": summaries,
        "order_and_time_context": [
            row for data in workers for row in order_context(data)
        ],
        "profile_minima": profiles,
    }


def analyze(args):
    require(
        args.trials
        and sorted(set(args.trials)) == args.trials
        and all(trial in TRIALS for trial in args.trials),
        "ordered unique trials",
    )
    provenance = validate_outer(args)
    workers, observers = [], []
    for trial in args.trials:
        for arm in ARMS:
            row = validate_worker(
                args.raw_root / f"trial{trial}-{arm}-timing.json",
                args,
                provenance,
                "timing",
            )
            require(
                (row["trial"], row["arm"]) == (trial, arm), "timing filename identity"
            )
            workers.append(row)
    if args.observations:
        for arm in ARMS:
            row = validate_worker(
                args.raw_root / f"trial0-{arm}-observe.json",
                args,
                provenance,
                "observe",
            )
            require(
                (row["trial"], row["arm"]) == (0, arm), "observer filename identity"
            )
            observers.append(row)
    all_workers = workers + observers
    require(
        len({row["_container_id"] for row in all_workers}) == len(all_workers),
        "container reused",
    )
    for key in ("library_sha256", "ninja_sha256"):
        require(
            len({row["environment"]["native_artifact"][key] for row in all_workers})
            == 1,
            "native artifact changed",
        )
    require(
        len({canonical(row["environment"]["operand_sha256"]) for row in all_workers})
        == 1,
        "fixture operands changed",
    )
    return {
        "status": "success",
        "schema_version": 1,
        "source_commit": args.expected_source_commit,
        "analyzer_sha256": sha256(Path(__file__)),
        "helper_sha256": HELPER_SHA256,
        "trials": args.trials,
        "independent_trials_per_arm": len(args.trials),
        "observations_required": args.observations,
        "outer_provenance": {
            "manifest_sha256": provenance["manifest_sha256"],
            "private_fixture_rehashed": provenance["private_fixture_rehashed"],
            "gpu": provenance["gpu"],
        },
        "validated_workers": [
            {
                "file": row["_path"],
                "sha256": sha256(Path(row["_path"])),
                "mode": row["mode"],
                "arm": row["arm"],
                "trial": row["trial"],
                "container_id": row["_container_id"],
                "numerical_checks": len(row["correctness"]["checks"]),
            }
            for row in all_workers
        ],
        "analysis": summarize(workers),
        "diagnostic_observers": [
            {
                "arm": row["arm"],
                "winner": row["natural"]["tactic"],
                "evidence": row["_observation"],
            }
            for row in observers
        ],
        "limitations": [
            "One fixed actual static-FP8 layer and six fresh trials per arm; replay blocks and15 sequential samples are not independent trials.",
            "Exploratory unadjusted bootstrap resamples whole trial estimates, never individual replay blocks.",
            "Chronological same-graph before/after changes confound process, elapsed-time and workload history; no matched untuned process identifies tuning as their unique cause.",
            "This retained/interleaved protocol differs from the prior separate-process natural-layer protocol. A vanished gap does not retroactively prove the old result was measurement noise.",
            "Public cache source is resolved at capture and does not recur during replay. Same public-1 alone does not prove identical native or physical GPU launches.",
            "Native/trace observations are separate instrumented processes and cannot establish the kernel identity of actual latency-worker graphs.",
            "Snapshots and order/time correlations are descriptive and cannot establish clocks or order as a cause.",
            "No explicit-policy, managed-store, full-model accuracy or serving-throughput claim.",
        ],
    }


def self_test():
    """Focused synthetic controls; no model, Torch, CUDA or experiment data."""
    tests = []

    def test(name, function, reject=False):
        try:
            function()
        except (ValueError, KeyError, TypeError) as error:
            require(reject, f"{name}: unexpected {error}")
            tests.append({"name": name, "passed": True, "rejection": str(error)})
        else:
            require(not reject, name + ": fault was accepted")
            tests.append({"name": name, "passed": True})

    stage_metadata = {"setup": (-1, "miss", False), "capture": (-1, "miss", True)}
    stage_rows = [{"stage": "setup"}, {"stage": "capture"}]
    test(
        "tuple_metadata_stage_coverage_positive",
        lambda: validate_stage_coverage(stage_rows, stage_metadata, "synthetic"),
    )
    test(
        "duplicate_stage_coverage_rejected",
        lambda: validate_stage_coverage(
            [{"stage": "setup"}, {"stage": "setup"}], stage_metadata, "synthetic"
        ),
        True,
    )
    test(
        "missing_stage_coverage_rejected",
        lambda: validate_stage_coverage(stage_rows[:1], stage_metadata, "synthetic"),
        True,
    )

    blocks, clock = [], 1_000_000_000
    for index, order in enumerate(expected_schedule(0)):
        measurements = []
        for graph_id in order:
            factor = {"pre_default": 1, "post_hit": 1.05, "post_miss": 0.98}[graph_id]
            wall, event = 35 * factor, 32 * factor
            measurements.append(
                {
                    "graph_id": graph_id,
                    "iterations": 5120,
                    "wall_batch_ms": wall,
                    "gpu_event_batch_ms": event,
                    "wall_per_call_ms": wall / 5120,
                    "gpu_event_per_call_ms": event / 5120,
                    "started_ns": clock,
                    "ended_ns": clock + round(wall * 1e6),
                    "rewarm_actual_ms": 20,
                    "rewarm_calls": 5120,
                    "correctness_check": f"block_{index:02d}_{graph_id}",
                    "gpu_before": "0,u,d,P0,1800 MHz",
                    "gpu_after": "0,u,d,P0,1800 MHz",
                }
            )
            clock += 100_000_000
        blocks.append({"index": index, "order": order, "measurements": measurements})
    controls = {
        "common_iterations": 5120,
        "schedule_seed": 4000,
        "schedule": expected_schedule(0),
        "blocks": blocks,
    }

    def callback(stage, graph):
        require(stage.endswith(graph), "checked wrong graph")

    test(
        "balanced_positive_blocks",
        lambda: validate_blocks(controls, 0, callback, "synthetic"),
    )

    def block_fault(name, mutate):
        value = copy.deepcopy(controls)
        mutate(value)
        test(name, lambda: validate_blocks(value, 0, callback, "synthetic"), True)

    block_fault(
        "unbalanced_order", lambda d: d["schedule"].__setitem__(1, d["schedule"][0])
    )
    block_fault("missing_graph_batch", lambda d: d["blocks"][0]["measurements"].pop())
    block_fault(
        "wrong_replay_count",
        lambda d: d["blocks"][0]["measurements"][0].update(iterations=2560),
    )
    block_fault(
        "bad_normalization",
        lambda d: d["blocks"][0]["measurements"][0].update(wall_per_call_ms=1),
    )
    block_fault(
        "missing_rewarm",
        lambda d: d["blocks"][0]["measurements"][0].update(rewarm_actual_ms=19),
    )
    block_fault(
        "mislabelled_timed_check",
        lambda d: d["blocks"][0]["measurements"][0].update(correctness_check="other"),
    )
    block_fault(
        "overlapping_batch",
        lambda d: d["blocks"][0]["measurements"][1].update(started_ns=1),
    )
    block_fault(
        "wrong_clock_units",
        lambda d: d["blocks"][0]["measurements"][0].update(ended_ns=1_100_000_000),
    )
    observed = {
        "mode": "observe",
        "graph_controls": {
            "history_before_startup": None,
            "history_after_startup_pre_capture": None,
            "sequential": [],
            "blocks": [],
            "schedule": [],
        },
    }
    test(
        "observation_cohort_has_no_latency",
        lambda: validate_timing_controls(observed, {}, callback, "synthetic"),
    )
    bad = copy.deepcopy(observed)
    bad["graph_controls"]["blocks"] = blocks
    test(
        "instrumented_latency_rejected",
        lambda: validate_timing_controls(bad, {}, callback, "synthetic"),
        True,
    )
    metrics = {
        "max_abs_error": 0,
        "max_normalized_error": 0,
        "rmse": 0,
        "relative_l2": 0,
    }
    numeric = {
        "check": "pre_default_changed_input",
        "quantized_reference": dict(metrics, passed=True),
        "bf16_reference": dict(metrics, diagnostic_only=True),
        "max_abs_error": 0,
        "exact_zero_required": True,
        "output_sha256": "a" * 64,
    }
    test("exact_zero_reference", lambda: validate_number(numeric, "synthetic"))
    bad = copy.deepcopy(numeric)
    bad["quantized_reference"]["max_abs_error"] = bad["max_abs_error"] = 0.01
    test("stale_zero_output_rejected", lambda: validate_number(bad, "synthetic"), True)
    bad = copy.deepcopy(numeric)
    bad["bf16_reference"]["diagnostic_only"] = False
    test(
        "bf16_diagnostic_is_not_accuracy_gate",
        lambda: validate_number(bad, "synthetic"),
        True,
    )

    events = []
    for index, graph_id in enumerate(GRAPH_IDS):
        start = 1000 + index * 2000
        events.append(
            {
                "name": "retained_" + graph_id,
                "cat": "user_annotation",
                "ph": "X",
                "ts": start,
                "dur": 1000,
            }
        )
        for replay in range(3):
            correlation_id, launch = 1 + index * 10 + replay, start + 10 + replay * 100
            events.append(
                {
                    "name": "cudaGraphLaunch",
                    "cat": "cuda_runtime",
                    "ts": launch,
                    "dur": 2,
                    "args": {"correlation": correlation_id},
                }
            )
            for kernel_index, name in enumerate(("_static_quant_fp8", "nvjet_matrix")):
                events.append(
                    {
                        "name": name,
                        "cat": "kernel",
                        "ts": launch + 3 + kernel_index * 10,
                        "dur": 4,
                        "args": {
                            "correlation": correlation_id,
                            "graph id": index + 1,
                            "grid": [1, 1, 1],
                            "block": [128, 1, 1],
                            "registers per thread": 32,
                            "shared memory": 0,
                        },
                    }
                )
    trace = {"traceEvents": events}
    test("actual_scoped_graph_launch_cycles", lambda: validate_trace(trace))
    bad = copy.deepcopy(trace)
    bad["traceEvents"] = [
        event
        for event in bad["traceEvents"]
        if not (event.get("cat") == "kernel" and event["args"]["correlation"] == 1)
    ]
    test("missing_actual_gpu_launches", lambda: validate_trace(bad), True)
    bad = copy.deepcopy(trace)
    next(e for e in bad["traceEvents"] if e.get("cat") == "kernel")["args"][
        "registers per thread"
    ] = 99
    test("changed_launch_signature", lambda: validate_trace(bad), True)
    bad = copy.deepcopy(trace)
    for event in bad["traceEvents"]:
        if event.get("cat") == "kernel":
            event["args"]["graph id"] = 1
    test("same_graph_substituted_under_three_labels", lambda: validate_trace(bad), True)
    bad = copy.deepcopy(trace)
    next(e for e in bad["traceEvents"] if e.get("cat") == "cuda_runtime")["args"][
        "correlation"
    ] = 999
    test("uncorrelated_kernel_not_attributed", lambda: validate_trace(bad), True)

    algo = {
        "raw_bytes": 64,
        "raw_hex": "00" * 64,
        "config": {
            name: {"status": 0, "bytes": 4, "value": 1}
            for name in (
                "id",
                "tile_id",
                "split_k",
                "reduction_scheme",
                "cta_swizzling",
                "custom_option",
                "stages_id",
            )
        },
    }
    records, graphs = [], {name: {"tactic": -1} for name in GRAPH_IDS}
    replay_phases = [name + "_numerics" for name in GRAPH_IDS]
    replay_phases += ["history_before_startup", "history_after_startup_pre_capture"]
    replay_phases += [
        prefix + name for prefix in ("sequential_", "trace_") for name in GRAPH_IDS
    ]
    for phase in [name + "_capture" for name in GRAPH_IDS] + replay_phases:
        capture = phase in {name + "_capture" for name in GRAPH_IDS}
        records.append(
            {
                "event": "phase_summary",
                "phase": phase,
                "pid": 1,
                "apis": {
                    "matmul": {"calls": int(capture), "failures": 0},
                    "heuristic": {"calls": int(capture), "failures": 0},
                },
            }
        )
        if capture:
            records.append(
                {
                    "event": "matmul",
                    "phase": phase,
                    "pid": 1,
                    "status": 0,
                    "workspace_bytes": 32,
                    "algo": copy.deepcopy(algo),
                }
            )
            records.append(
                {
                    "event": "heuristic",
                    "phase": phase,
                    "pid": 1,
                    "status": 0,
                    "requested": 1,
                    "returned": 1,
                    "algorithms": [{"state": 0, "algo": copy.deepcopy(algo)}],
                }
            )
    test(
        "native_submitted_descriptor_and_heuristic",
        lambda: validate_native_rows(records, graphs),
    )
    bad = copy.deepcopy(records)
    next(row for row in bad if row["event"] == "matmul")["algo"]["config"]["id"][
        "status"
    ] = 7
    test(
        "failed_config_query_not_identity",
        lambda: validate_native_rows(bad, graphs),
        True,
    )
    bad = copy.deepcopy(records)
    next(
        row
        for row in bad
        if row["event"] == "phase_summary" and row["phase"] == "trace_pre_default"
    )["apis"]["matmul"]["calls"] = 1
    test(
        "replay_must_not_reenter_native_selection",
        lambda: validate_native_rows(bad, graphs),
        True,
    )
    values = [0.98, 0.99, 1.0, 1.01, 1.02, 1.03]
    interval_value = h.interval_for_trial_median(values)
    literal = [
        statistics.median(values[i] for i in sample)
        for sample in itertools.product(range(6), repeat=6)
    ]
    test(
        "whole_trial_bootstrap_equals46656_ordered_resamples",
        lambda: require(
            interval_value["lower"] == h.quantile(literal, 0.025)
            and interval_value["upper"] == h.quantile(literal, 0.975),
            "bootstrap units/formula",
        ),
    )
    test(
        "one_trial_checkpoint_has_no_interval",
        lambda: require(h.interval_for_trial_median([1]) is None, "n1 inference"),
    )
    test(
        "no_duplicate_trial_units",
        lambda: statistic([{"trial": 0, "ratio": 1}, {"trial": 0, "ratio": 2}]),
        True,
    )
    test(
        "constant_clock_correlation_is_unknown",
        lambda: require(correlation([1, 1], [1, 2]) is None, "spurious correlation"),
    )
    return {
        "status": "success",
        "synthetic_only": True,
        "analyzer_sha256": sha256(Path(__file__)),
        "helper_sha256": HELPER_SHA256,
        "passed": len(tests),
        "controls": tests,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--fixture-manifest", type=Path)
    parser.add_argument(
        "--trials", type=int, nargs="+", choices=TRIALS, default=list(TRIALS)
    )
    parser.add_argument(
        "--observations",
        action="store_true",
        help="Require both separate diagnostic workers and original native/trace evidence.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run focused synthetic stdlib controls, without model/Torch/CUDA.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.self_test and any(
        getattr(args, key) is None
        for key in (
            "raw_root",
            "source_root",
            "expected_source_commit",
            "source_manifest",
            "fixture_manifest",
            "output",
        )
    ):
        parser.error(
            "analysis requires raw/source roots, source commit, both manifests and output"
        )
    try:
        result = self_test() if args.self_test else analyze(args)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        StopIteration,
        subprocess.SubprocessError,
    ) as error:
        result = {
            "status": "error",
            "validation_errors": [f"{type(error).__name__}: {error}"],
            "analysis": {},
            "comparison_policy": "withheld because required validation failed",
        }
    rendered = h.json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(result["status"])
    else:
        print(rendered, end="")
    if result["status"] != "success":
        print(result["validation_errors"][0])
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
