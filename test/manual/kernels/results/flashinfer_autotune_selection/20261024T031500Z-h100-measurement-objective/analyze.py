#!/usr/bin/env python3
"""Gate the two measurement objectives before summarizing retained graph replay.

Workers, rather than replay blocks, are the independent units. A slower or
unchanged winner remains a valid outcome; no latency threshold accepts a run.
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
import types
from pathlib import Path

BATCH = "test/manual/kernels/results/flashinfer_autotune_selection/20261024T031500Z-h100-measurement-objective"
WORKER = "test/manual/kernels/flashinfer_autotune_measurement_objective.py"
HELPER = "test/manual/kernels/results/flashinfer_autotune_selection/20261020T034000Z-h100-fp8-linear/analyze.py"
HELPER_SHA256 = "e551c7508890045395074d7532c7a2b621cc91fbfe9b4b8c23a7e9e7070027d5"
GRAPH_HELPER = "test/manual/kernels/results/flashinfer_autotune_selection/20261020T143705Z-h100-graph-controls/analyze.py"
GRAPH_HELPER_SHA256 = "7955e92c67430bf6bfec0ba2ac509dd9ba1f856c89069923497593553179de95"
ARMS = ("auto", "cuda_graph")
GRAPH_IDS = ("winner", "default")
TRIALS = tuple(range(6))
POLICY = {"execution_mode": "cuda_graph", "cold_l2": None, "_timer": "auto"}
GRAPH_IDENTITY = "('__policy__', ('measure_execution_mode', 'cuda_graph'))"
AUTOTUNE_CACHE_SHA256 = (
    "7be4f83162ce3afada8593f5acd863769356da74102e6762ece133548a4d8bda"
)


def pinned_module(relative, expected, name):
    path = Path(__file__).resolve().parents[6] / relative
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("frozen analysis helper changed: " + relative)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(content, str(path), "exec"), module.__dict__)
    return module


h = pinned_module(HELPER, HELPER_SHA256, "frozen_linear_analysis")
g = pinned_module(GRAPH_HELPER, GRAPH_HELPER_SHA256, "frozen_graph_analysis")
require, load, canonical, sha256 = h.require, h.load, h.canonical, h.sha256


def expected_policy(arm):
    return None if arm == "auto" else POLICY


def expected_identity(arm):
    return "None" if arm == "auto" else GRAPH_IDENTITY


def expected_fields(arm):
    return {} if arm == "auto" else {"measure_execution_mode": "cuda_graph"}


def validate_settings(value, arm, label, configured=True, active=True):
    require(
        value["warmup"] == 3
        and value["repeat"] == 10
        and value["stream_delay_micro_secs"] == 5000
        and value["cuda_graph_delay_micro_secs"] == 100
        and value["use_global_timer"] is False,
        label + ": historical profiler defaults changed",
    )
    require(
        value["effective_measure_policy"] == (expected_policy(arm) if active else None)
        and value["effective_policy_timer"]
        == ("events" if arm == "cuda_graph" and active else None),
        label + ": effective measurement policy",
    )
    config = (
        None
        if not configured
        else {
            "use_cuda_graph": arm == "cuda_graph",
            "use_cold_l2_cache": False,
            "cuda_graph_profile_replays": 1,
            "profiling_repeat": None,
            "effective_profiling_repeat": 10,
        }
    )
    require(value["config"] == config, label + ": actual profiling configuration")


def expected_schedule(trial):
    result = [
        list(order) for _ in range(12) for order in itertools.permutations(GRAPH_IDS)
    ]
    random.Random(5000 + trial).shuffle(result)
    return result


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
    coverage = {
        WORKER,
        HELPER,
        GRAPH_HELPER,
        BATCH + "/analyze.py",
        h.WRAPPER,
        h.FP8_UTILS,
    }
    require(coverage <= set(outer["source_sha256"]), "source coverage")
    for path, value in outer["source_sha256"].items():
        require(h.source_hash(args, path) == value, "frozen source: " + path)
    require(
        outer["source_sha256"][HELPER] == HELPER_SHA256
        and outer["source_sha256"][GRAPH_HELPER] == GRAPH_HELPER_SHA256,
        "pinned analysis helpers",
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


def validate_policy_source(environment, plan, label):
    require(
        environment["flashinfer_autotune_cache_sha256"]
        == plan["flashinfer_source"]["autotune_cache_sha256"]
        == AUTOTUNE_CACHE_SHA256,
        label + ": exact policy/cache lifecycle source",
    )


def validate_metadata(data, args, provenance, label):
    fixture = provenance["fixture"]
    require(
        data["status"] == "success"
        and data["schema_version"] == 1
        and data["implementation"] == "candidate"
        and data["mode"] == "timing"
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
    validate_policy_source(env, provenance["plan"], label)
    require(
        env["sglang_commit"] == args.expected_source_commit
        and env["sglang_wrapper_sha256"] == h.source_hash(args, h.WRAPPER),
        label + ": checkout",
    )
    require(
        (env["torch"], env["torch_cuda"], env["flashinfer"])
        == ("2.14.1+cu130", "13.0", "0.7.2")
        and env["backend"] == "cublas"
        and env["managed_store"] is False
        and not env["LD_PRELOAD"],
        label + ": uninstrumented software/store",
    )
    expected_gpu = provenance["plan"]["environment"]
    require(
        env["gpu"] == expected_gpu["gpu"]
        and env["capability"] == [9, 0]
        and env["physical_gpu_uuid"].removeprefix("GPU-")
        == expected_gpu["physical_gpu_uuid"]
        and all(
            expected_gpu["driver"] in env[key]
            for key in ("gpu_status_before", "gpu_status_after")
        ),
        label + ": GPU/driver",
    )
    sources, native = (
        h.expected_sources(provenance["source_manifest"]),
        env["native_artifact"],
    )
    require(
        native["source_sha256"] == sources
        and env["flashinfer_tuner_sha256"]
        == sources["flashinfer/autotuner/autotuner.py"]
        and native["cuda_arch_list"] == "9.0"
        and "bmm_fp8_run_with_descriptor" in native["required_native_exports"]
        and h.valid_sha(native["library_sha256"])
        and h.valid_sha(native["ninja_sha256"]),
        label + ": qualified native source/build",
    )
    helpers = {
        "flashinfer_autotune_linear_layer.py",
        "flashinfer_autotune_descriptor_selection.py",
        "flashinfer_autotune_descriptor_qualification.py",
        "flashinfer_autotune_eager_setup.py",
        "flashinfer_autotune_graph_controls.py",
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
    observed = data["fixture"]
    require(
        observed["manifest_sha256"] == sha256(args.fixture_manifest)
        and observed["tensor_file_sha256"] == fixture["tensor_file"]["sha256"]
        and all(
            observed[key] == fixture[key]
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
        label + ": static quantizer bytes",
    )


def runtime_profile(shape):
    m, n, k = shape
    return [[1, 1 << (m - 1).bit_length(), k], [1, k, n], [1], [1], [1, -1, n], [-1]]


def validate_profiles(data, label):
    profiles, summary = data["profiles"], data["runtime_profile_summary"]
    require(
        data["profile_attempts"] == len(profiles) > 0
        and not data["forbidden_profile_attempts"],
        label + ": profiling accounting",
    )
    expected, arm = runtime_profile(data["shape"]), data["arm"]
    groups = data["profiling_input_groups"]
    require(groups, label + ": missing actual profiling input fingerprints")
    runtime, by_profile = [], collections.defaultdict(list)
    for row in profiles:
        require(
            row["stage"] == "startup"
            and row["runner"] == "CublasFp8GemmRunner"
            and h.valid_tactic(row["tactic"]),
            label + ": profiling scope/tactic",
        )
        require(
            row["use_cuda_graph"] is (arm == "cuda_graph")
            and row["use_cold_l2_cache"] is False,
            label + ": actual profiling objective",
        )
        validate_settings(row["autotuner_settings"], arm, label)
        require(
            row["policy_identity"] == expected_identity(arm)
            and row["policy_manifest_fields"] == expected_fields(arm)
            and row["default_race_enabled"] is True,
            label + ": profiling policy identity",
        )
        require(
            row["normalized_profile"] == h.normalized_profile(row["input_shapes"]),
            label + ": normalized profile",
        )
        require(
            row.get("profiler_ms") is None or h.positive(row["profiler_ms"]),
            label + ": finite profiling outcome",
        )
        group = row["profiling_input_group"]
        require(
            h.valid_sha(group) and group in groups, label + ": actual profiler operands"
        )
        operands = groups[group]
        require(
            set(operands) == {"activation", "weight", "input_scale", "weight_scale"},
            label + ": profiling operand coverage",
        )
        for index, name in enumerate(
            ("activation", "weight", "input_scale", "weight_scale")
        ):
            tensor = operands[name]
            require(
                tensor["shape"] == row["input_shapes"][index]
                and tensor["dtype"]
                == ("torch.float8_e4m3fn" if index < 2 else "torch.float32")
                and len(tensor["stride"]) == len(tensor["shape"])
                and all(type(v) is int and v >= 0 for v in tensor["stride"])
                and h.valid_sha(tensor["sha256"]),
                label + ": profiling operand metadata",
            )
        by_profile[canonical(row["normalized_profile"])].append(row)
        if row["normalized_profile"] == expected:
            runtime.append(row)
    require(
        {row["input_shapes"][0][-2] for row in profiles}
        == {1, 2, 4, 8, 16, 32, 64, 128, 144},
        label + ": startup bucket coverage",
    )
    for rows in by_profile.values():
        require(
            rows[0]["tactic"] == -1
            and any(
                row["tactic"] == -1 and row.get("profiler_ms") is not None
                for row in rows
            ),
            label + ": real default candidate participation",
        )
        require(
            len({canonical(row["tactic"]) for row in rows}) == len(rows),
            label + ": duplicate profiling candidate",
        )
        require(
            len({row["profiling_input_group"] for row in rows}) == 1,
            label + ": candidates profiled different operands",
        )
    require(
        set(groups) == {row["profiling_input_group"] for row in profiles},
        label + ": unreferenced profiling input group",
    )
    for digest, operands in groups.items():
        require(
            hashlib.sha256(canonical(operands).encode()).hexdigest() == digest,
            label + ": profiling operand group hash",
        )
    finite = [row for row in runtime if row.get("profiler_ms") is not None]
    require(finite, label + ": finite runtime profiles")
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
        label + ": observed profiler minimum",
    )
    winner = data["natural"]["tactic"]
    require(
        winner == min(finite, key=lambda row: row["profiler_ms"])["tactic"],
        label + ": actual minimum winner",
    )
    require(
        data["natural"]["cache_hit"] is True
        and data["natural"]["lookup_tier"] == "memory",
        label + ": natural policy-memory winner",
    )
    h.validate_acceptance(
        data["natural"]["acceptance"], winner, label + ": native acceptance"
    )
    require(
        data["cache"]["before_sha256"] is None
        and data["cache"]["miss_scopes_restored"] is True,
        label + ": fresh memory-only/restored cache",
    )
    return winner


def validate_objective(data, label):
    arm, objective = data["arm"], data["objective"]
    require(
        objective["requested_policy"] == expected_policy(arm)
        and objective["manifest_fields"] == expected_fields(arm)
        and data["environment"]["measurement_policy"] == expected_policy(arm),
        label + ": requested objective",
    )
    require(
        objective["startup_api_calls"]
        == [
            {
                "mode": "tune",
                "persistent_cache": False,
                "measurement_policy": expected_policy(arm),
            }
        ],
        label + ": real public startup policy call",
    )
    require(
        objective["default_race_enabled_both_arms"] is True
        and objective["effective_hot_l2"] is True
        and not objective["protocol_errors"]
        and objective["replay_outside_all_contexts"] is True,
        label + ": objective protocol",
    )
    validate_settings(
        data["autotuner_settings"]["initial"],
        "auto",
        label,
        configured=False,
        active=False,
    )
    validate_settings(data["autotuner_settings"]["after"], "auto", label, active=False)
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
        label + ": final scope leak",
    )
    rows = objective["scope_records"]
    stages = (
        "startup",
        "natural_before",
        "winner_capture_scope",
        "default_capture_scope",
        "natural_final",
    )
    expected = {
        (stage, event)
        for stage in stages
        for event in ("enter", "before_exit", "after_exit")
    }
    require(
        collections.Counter((row["stage"], row["event"]) for row in rows)
        == collections.Counter(expected),
        label + ": policy scope lifecycle coverage",
    )
    for row in rows:
        startup, active = row["stage"] == "startup", row["event"] != "after_exit"
        require(
            row["mode"] == ("tune" if startup else "replay")
            and row["measurement_policy"] == (expected_policy(arm) if active else None)
            and row["manifest_fields"] == (expected_fields(arm) if active else {})
            and row["winner_identity"] == (expected_identity(arm) if active else "None")
            and row["active_v2"] is active
            and row["managed_store"] is False,
            label + ": policy scope identity/store",
        )
        require(
            row["is_tuning_mode"] is startup
            and row["tuning_contexts"]
            == (2 if startup and active else 1 if startup else 0),
            label + ": real nested tuning/replay lifecycle",
        )
        require(
            row["on_forward_stream"] is startup
            and row["on_caller_stream"] is (not startup),
            label + ": policy stream restoration",
        )
    cache = data["cache"]
    require(
        cache["persistence_under_test"] is False
        and cache["after_startup_sha256"] == cache["after_controls_sha256"],
        label + ": incidental file changed during controls",
    )
    # This does not inspect, reload or qualify persisted winner data.


def validate_graphs(data, winner, label):
    arm, controls = data["arm"], data["graph_controls"]
    order = list(GRAPH_IDS if data["trial"] % 2 == 0 else reversed(GRAPH_IDS))
    require(
        controls["capture_order"] == order
        and [row["id"] for row in controls["graphs"]] == order,
        label + ": capture order",
    )
    graphs = {row["id"]: row for row in controls["graphs"]}
    require(
        set(graphs) == set(GRAPH_IDS)
        and len({row["output"]["data_ptr"] for row in graphs.values()}) == 2,
        label + ": distinct retained graph outputs",
    )
    require(len(controls["miss_scope"]) == 1, label + ": genuine-miss coverage")
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
    m, n, k = data["shape"]
    previous_end = 0
    for graph_id in order:
        row, tactic, tier = (
            graphs[graph_id],
            winner if graph_id == "winner" else -1,
            "memory" if graph_id == "winner" else "miss",
        )
        require(
            row["tactic"] == tactic
            and row["lookup_tier"] == tier
            and row["policy_identity"] == expected_identity(arm)
            and row["measurement_policy"] == expected_policy(arm),
            label + ": graph choice/identity",
        )
        selected = row["selection"]
        require(
            selected["stage"] == graph_id + "_capture"
            and selected["tactic"] == tactic
            and selected["lookup_tier"] == tier
            and selected["cache_hit"] is (tier == "memory")
            and selected["capturing"] is True
            and selected["policy_identity"] == expected_identity(arm)
            and selected["measurement_policy"] == expected_policy(arm)
            and selected["active_v2"] is True
            and selected["is_tuning_mode"] is False,
            label + ": actual capture-time cache choice",
        )
        begin, end = g.interval(row, label + "/" + graph_id)
        require(begin > previous_end, label + ": capture chronology")
        previous_end = end
        out, inputs = row["output"], row["inputs"]
        require(
            out["shape"] == [m, n]
            and out["stride"] == [n, 1]
            and out["dtype"] == "torch.bfloat16"
            and len(inputs) == 6
            and inputs[4]["data_ptr"] == out["data_ptr"]
            and inputs[0]["shape"] == [1, m, k]
            and inputs[1]["shape"] == [1, k, n]
            and inputs[0]["dtype"] == inputs[1]["dtype"] == "torch.float8_e4m3fn"
            and inputs[1]["stride"][-2:] == [1, k],
            label + ": retained actual operands/output",
        )
        require(
            row["linear_input"]["shape"] == [m, k]
            and row["linear_input"]["dtype"] == "torch.bfloat16",
            label + ": actual BF16 input",
        )
        for tensor in [out, row["linear_input"], *inputs]:
            pointer = tensor["data_ptr"]
            require(
                type(pointer) is int
                and pointer > 0
                and tensor["alignment_mod256"] == pointer % 256
                and tensor["alignment_mod4096"] == pointer % 4096,
                label + ": retained allocation",
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
        len({row["linear_input"]["data_ptr"] for row in graphs.values()}) == 1
        and all(
            len({row["inputs"][i]["data_ptr"] for row in graphs.values()}) == 1
            for i in (1, 2, 3)
        ),
        label + ": shared frozen input allocations",
    )
    return graphs


def validate_observed_stages(data, graphs, winner, label):
    expected = {
        "setup": (-1, "miss", False, False),
        "startup": (winner, "memory", False, True),
        "natural_before": (winner, "memory", False, True),
        "natural_eager_changed_input": (winner, "memory", False, True),
        "natural_eager_restored_input": (winner, "memory", False, True),
        "natural_final": (winner, "memory", False, True),
        "bare_after_scope": (
            (-1, "miss", False, False)
            if data["arm"] == "cuda_graph"
            else (winner, "memory", False, False)
        ),
    }
    for graph_id in GRAPH_IDS:
        tactic, tier = (winner, "memory") if graph_id == "winner" else (-1, "miss")
        expected[graph_id + "_before"] = (tactic, tier, False, True)
        expected[graph_id + "_capture"] = (tactic, tier, True, True)
    g.validate_stage_coverage(
        data["selections"], expected, label + ": selected stage coverage"
    )
    for row in data["selections"]:
        tactic, tier, capture, active = expected[row["stage"]]
        require(
            row["tactic"] == tactic
            and row["lookup_tier"] == tier
            and row["cache_hit"] is (tier == "memory")
            and row["capturing"] is capture
            and row["active_v2"] is active
            and row["is_tuning_mode"] is (row["stage"] == "startup")
            and row["policy_identity"]
            == (expected_identity(data["arm"]) if active else "None")
            and row["measurement_policy"]
            == (expected_policy(data["arm"]) if active else None),
            label + ": selected tactic/cache/policy",
        )
        require(
            (row["on_forward_stream"] and not row["on_caller_stream"])
            if row["stage"] == "startup"
            else capture or row["on_caller_stream"],
            label + ": actual forward stream",
        )
    bindings = set(expected) - {"setup", "startup"}
    g.validate_stage_coverage(
        data["binding_acceptance_checks"],
        {key: None for key in bindings},
        label + ": native acceptance coverage",
    )
    for row in data["binding_acceptance_checks"]:
        h.validate_acceptance(
            row, expected[row["stage"]][0], label + ": actual descriptor acceptance"
        )
    layer = data["actual_layer"]
    g.validate_stage_coverage(
        layer["observed_public_inputs"], expected, label + ": public operand coverage"
    )
    for row in layer["observed_public_inputs"]:
        capture = expected[row["stage"]][2]
        require(
            row["capturing"] is capture and row["backend"] == "cublas",
            label + ": actual backend",
        )
        # The real zero-input observer records layout only. Its exact-zero
        # numerical guard is separate; restored/nonzero rows retain all hashes.
        if (
            not capture
            and row["stage"] != "startup"
            and not row["stage"].endswith("_changed_input")
        ):
            require(
                row["quantized_input_sha256"]
                == layer["quantization_reference"]["actual_input_fp8_sha256"]
                and row["weight_rowmajor_sha256"]
                == data["fixture"]["tensors"]["weight_fp8_rowmajor"]["sha256"],
                label + ": observed actual operand bytes",
            )
    g.validate_stage_coverage(
        layer["native_output_poison_checks"],
        {key: None for key in bindings},
        label + ": native output poison coverage",
    )
    for row in layer["native_output_poison_checks"]:
        capture = expected[row["stage"]][2]
        require(
            row["capturing"] is capture
            and row["poisoned_actual_allocation"] is (not capture),
            label + ": numerical poison entered capture",
        )
    correct = data["correctness"]
    require(
        correct["passed"] is True
        and (correct["rtol"], correct["atol"]) == (0.02, 0.125),
        label + ": numerical protocol",
    )
    checks = {row["check"]: row for row in correct["checks"]}
    require(
        len(checks) == len(correct["checks"]), label + ": duplicate numerical check"
    )
    expected_checks = (
        set(expected) - {"setup", "startup", "winner_capture", "default_capture"}
    ) | {"initial_untuned"}
    for graph_id in GRAPH_IDS:
        expected_checks.update(
            graph_id + suffix for suffix in ("_changed_input", "_restored_input")
        )
        expected_checks.update(
            prefix + graph_id for prefix in ("sequential_", "final_")
        )
        expected_checks.update(f"block_{index:02d}_{graph_id}" for index in range(24))
    require(set(checks) == expected_checks, label + ": numerical lifecycle coverage")
    for row in checks.values():
        g.validate_number(row, label)
        require(
            type(row["output_data_ptr"]) is int and row["output_data_ptr"] > 0,
            label + ": actual checked allocation",
        )

    def check_graph(stage, graph_id):
        require(
            checks[stage]["graph_id"] == graph_id
            and checks[stage]["output_data_ptr"]
            == graphs[graph_id]["output"]["data_ptr"],
            label + ": numerical guard checked another allocation",
        )

    for graph_id in GRAPH_IDS:
        for stage in (
            graph_id + "_changed_input",
            graph_id + "_restored_input",
            "sequential_" + graph_id,
            "final_" + graph_id,
        ):
            check_graph(stage, graph_id)
    for stage in set(checks) & set(expected):
        require(
            checks[stage]["graph_id"] is None,
            label + ": eager check replaced by retained graph",
        )
    return check_graph


def validate_timing(data, graphs, check_graph, label):
    controls = data["graph_controls"]
    require(
        controls["common_iterations"] == 5120
        and controls["schedule_seed"] == 5000 + data["trial"]
        and controls["schedule"] == expected_schedule(data["trial"]),
        label + ": balanced replay protocol",
    )
    sequential = controls["sequential"]
    require(
        [row["graph_id"] for row in sequential] == list(GRAPH_IDS),
        label + ": sequential coverage",
    )
    previous_end = max(row["ended_ns"] for row in graphs.values())
    for row in sequential:
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
            begin > previous_end
            and (end - begin) / 1e6 >= sum(row["measurement"]["wall_batch_ms"]),
            label + ": sequential chronology",
        )
        previous_end = end
        check_graph(stage, row["graph_id"])
    blocks = controls["blocks"]
    require(len(blocks) == 24, label + ": paired block count")
    for index, (block, order) in enumerate(zip(blocks, controls["schedule"])):
        require(
            block["index"] == index
            and block["order"] == order
            and [row["graph_id"] for row in block["measurements"]] == order,
            label + ": paired block identity/coverage",
        )
        for row in block["measurements"]:
            stage = f"block_{index:02d}_{row['graph_id']}"
            require(
                row["correctness_check"] == stage
                and row["policy_scope_active"] is False
                and row["tuning_mode"] is False,
                label + ": timed replay context leak",
            )
            g.validate_batch(row, label + "/" + stage)
            begin, end = g.interval(row, label + "/" + stage)
            require(
                begin > previous_end
                and math.isclose(
                    (end - begin) / 1e6, row["wall_batch_ms"], rel_tol=1e-12
                ),
                label + ": raw batch clock/chronology",
            )
            previous_end = end
            check_graph(stage, row["graph_id"])


def validate_worker(path, args, provenance):
    data, label = load(path), path.name
    validate_metadata(data, args, provenance, label)
    validate_objective(data, label)
    winner = validate_profiles(data, label)
    graphs = validate_graphs(data, winner, label)
    checker = validate_observed_stages(data, graphs, winner, label)
    validate_timing(data, graphs, checker, label)
    require(
        type(data["pid"]) is int and data["pid"] > 0,
        label + ": actual process identity",
    )
    observer = data["observer"]
    require(
        observer["enabled"] is False
        and observer["constructor_guard_suspended"] is False
        and observer["real_interposition_remained_loaded"] is False
        and not observer["preload_files"]
        and not observer["phases"]
        and observer["trace"] is None
        and not observer["native_output_path"],
        label + ": diagnostic instrumentation in latency worker",
    )
    require(
        path.with_suffix(".exit-code").read_text().strip() == "0",
        label + ": container exit",
    )
    cid = path.with_suffix(".container-id").read_text().strip()
    require(re.fullmatch("[0-9a-f]{64}", cid) is not None, label + ": fresh container")
    data.update(_path=str(path), _container_id=cid)
    return data


def profiling_signature(data):
    """Input bytes, candidate order and all non-objective settings must agree."""
    result = []
    for row in data["profiles"]:
        settings = copy.deepcopy(row["autotuner_settings"])
        settings.pop("effective_measure_policy")
        settings.pop("effective_policy_timer")
        settings["config"].pop("use_cuda_graph")
        result.append(
            {
                key: row[key]
                for key in (
                    "runner",
                    "input_shapes",
                    "normalized_profile",
                    "tactic",
                    "profiling_input_group",
                )
            }
            | {"unchanged_settings": settings}
        )
    return {"profiles": result, "input_groups": data["profiling_input_groups"]}


def validate_cohort(workers):
    require(
        len({row["_container_id"] for row in workers}) == len(workers),
        "container reused",
    )
    # Container PID namespaces may each assign the same numeric PID. Distinct
    # container IDs plus completed process exits establish fresh worker runs.
    for key in ("library_sha256", "ninja_sha256"):
        require(
            len({row["environment"]["native_artifact"][key] for row in workers}) == 1,
            "native artifact changed across objectives",
        )
    require(
        len({canonical(row["environment"]["operand_sha256"]) for row in workers}) == 1,
        "fixture operands changed",
    )
    require(
        len({canonical(profiling_signature(row)) for row in workers}) == 1,
        "profile operands, candidate pool/order or non-objective settings changed",
    )


def validate_build(args, provenance, workers):
    path = args.build_provenance or args.raw_root / "native-build-provenance.json"
    build = load(path)
    require(
        build["status"] == "success"
        and not build["errors"]
        and not build["missing_records"]
        and build["source_manifest_sha256"] == sha256(args.source_manifest)
        and build["trials"] == args.trials
        and build["expected_native_records"]
        == build["validated_native_records"]
        == len(workers)
        and build["fixture_shape"] == provenance["fixture"]["shape"]
        and set(build["arms"]) == {"candidate"},
        "actual native build provenance",
    )
    arm = build["arms"]["candidate"]
    require(
        arm["library"]["sha256"]
        == workers[0]["environment"]["native_artifact"]["library_sha256"]
        and arm["ninja"]["sha256"]
        == workers[0]["environment"]["native_artifact"]["ninja_sha256"]
        and arm["all_runtime_source_sha256"]
        == h.expected_sources(provenance["source_manifest"]),
        "collector/native source and build equality",
    )
    expected = {Path(row["_path"]).name: sha256(Path(row["_path"])) for row in workers}
    observed = {
        row["result"]: row["result_sha256"] for row in arm["validated_worker_results"]
    }
    require(
        len(observed) == len(arm["validated_worker_results"]) and observed == expected,
        "collector/result binding",
    )
    retained = arm["ninja"]["retained_relative_path"]
    require(
        sha256(path.parent / retained) == arm["ninja"]["sha256"],
        "retained actual build instructions",
    )
    return {
        "file": str(path),
        "sha256": sha256(path),
        "validated_native_records": len(workers),
    }


def summarize_ratios(rows):
    require(
        rows
        and all(type(row["trial"]) is int and h.positive(row["ratio"]) for row in rows),
        "invalid trial ratio",
    )
    result = h.summarize_ratios(rows)
    result["ratios_equal_one"] = sum(row["ratio"] == 1 for row in rows)
    return result


def worker_effects(data):
    result = []
    for channel, key in (
        ("gpu_event", "gpu_event_per_call_ms"),
        ("wall", "wall_per_call_ms"),
    ):
        pairs = [
            {row["graph_id"]: row[key] for row in block["measurements"]}
            for block in data["graph_controls"]["blocks"]
        ]
        ratios = [row["winner"] / row["default"] for row in pairs]
        result.append(
            {
                "trial": data["trial"],
                "arm": data["arm"],
                "channel": channel,
                "ratio": statistics.median(ratios),
                "block_ratios": ratios,
                "block_ratio_min": min(ratios),
                "block_ratio_max": max(ratios),
                "graph_median_ms": {
                    name: statistics.median(row[name] for row in pairs)
                    for name in GRAPH_IDS
                },
                "natural_tactic": data["natural"]["tactic"],
            }
        )
    return result


def summarize(workers):
    effects = [effect for row in workers for effect in worker_effects(row)]
    summaries, comparisons = [], []
    for channel in ("gpu_event", "wall"):
        for arm in ARMS:
            rows = [
                row
                for row in effects
                if row["channel"] == channel and row["arm"] == arm
            ]
            summaries.append(
                {
                    "arm": arm,
                    "channel": channel,
                    "primary": channel == "gpu_event",
                    "ratio": "retained_winner / independent_retained_default",
                    **summarize_ratios(rows),
                }
            )
        by_key = {
            (row["trial"], row["arm"]): row
            for row in effects
            if row["channel"] == channel
        }
        for measure in (
            "normalized_winner_default",
            "winner_latency",
            "default_latency",
        ):
            rows = []
            for trial in sorted({row["trial"] for row in workers}):
                auto, graph = by_key[trial, "auto"], by_key[trial, "cuda_graph"]
                if measure == "normalized_winner_default":
                    numerator, denominator = graph["ratio"], auto["ratio"]
                else:
                    name = "winner" if measure == "winner_latency" else "default"
                    numerator, denominator = (
                        graph["graph_median_ms"][name],
                        auto["graph_median_ms"][name],
                    )
                rows.append(
                    {
                        "trial": trial,
                        "ratio": numerator / denominator,
                        "cuda_graph_estimate": numerator,
                        "auto_estimate": denominator,
                    }
                )
            comparisons.append(
                {
                    "channel": channel,
                    "measure": measure,
                    "primary": channel == "gpu_event"
                    and measure == "normalized_winner_default",
                    "ratio": "explicit_graph_objective / current_auto_objective",
                    **summarize_ratios(rows),
                }
            )
    return {
        "within_arm": summaries,
        "paired_cross_arm": comparisons,
        "selections": [
            {
                "arm": arm,
                "default_winners": sum(
                    row["natural"]["tactic"] == -1
                    for row in workers
                    if row["arm"] == arm
                ),
                "explicit_winners": sum(
                    row["natural"]["tactic"] != -1
                    for row in workers
                    if row["arm"] == arm
                ),
                "trial_tactics": [
                    {
                        "trial": row["trial"],
                        "tactic": row["natural"]["tactic"],
                        "runtime_profile_summary": row["runtime_profile_summary"],
                    }
                    for row in workers
                    if row["arm"] == arm
                ],
            }
            for arm in ARMS
        ],
    }


def analyze(args):
    require(
        args.trials
        and sorted(set(args.trials)) == args.trials
        and all(trial in TRIALS for trial in args.trials),
        "ordered unique trials",
    )
    provenance, workers = validate_outer(args), []
    for trial in args.trials:
        for arm in ARMS:
            row = validate_worker(
                args.raw_root / f"trial{trial}-{arm}-timing.json", args, provenance
            )
            require(
                (row["trial"], row["arm"]) == (trial, arm), "worker filename identity"
            )
            workers.append(row)
    validate_cohort(workers)
    build = validate_build(args, provenance, workers)
    return {
        "status": "success",
        "schema_version": 1,
        "source_commit": args.expected_source_commit,
        "analyzer_sha256": sha256(Path(__file__)),
        "helper_sha256": HELPER_SHA256,
        "graph_helper_sha256": GRAPH_HELPER_SHA256,
        "trials": args.trials,
        "independent_trials_per_arm": len(args.trials),
        "complete_six_trial_cohort": args.trials == list(TRIALS),
        "no_latency_acceptance_threshold": True,
        "persistence_qualified": False,
        "outer_provenance": {
            key: provenance[key]
            for key in ("manifest_sha256", "private_fixture_rehashed", "gpu")
        },
        "actual_native_build": build,
        "validated_workers": [
            {
                "file": row["_path"],
                "sha256": sha256(Path(row["_path"])),
                "arm": row["arm"],
                "trial": row["trial"],
                "pid": row["pid"],
                "container_id": row["_container_id"],
                "profiles": len(row["profiles"]),
                "numerical_checks": len(row["correctness"]["checks"]),
            }
            for row in workers
        ],
        "analysis": summarize(workers),
        "limitations": [
            "One fixed checkpoint-derived static-FP8 layer on one H100. No full-model accuracy or serving-throughput inference.",
            "Startup profiles use synthetic activations; actual defined profiling operands and candidate pools are required identical across objectives. Serving graphs use the frozen captured activation.",
            "Twenty-four replay blocks and fifteen sequential samples are repeated observations within a process, not independent trials.",
            "Exploratory unadjusted bootstrap resamples whole trial estimates only. A one-trial checkpoint has no uncertainty interval.",
            "The primary cross-objective comparison is the paired trial ratio of within-process winner/default ratios. Absolute latency comparisons retain separate-process history and elapsed-time confounding.",
            "Raw CUDA graph replay occurs outside policy scopes. Policy/cache identity is established at capture and no Python tactic lookup occurs during replay.",
            "Public tactic IDs and equal source/build hashes do not establish physical kernel identity; this batch has no instrumented launch observation.",
            "Memory-only scopes qualify tuning/capture lifecycle only. Incidental legacy saves are not a policy-aware persistence test.",
            "Profiling operand fingerprints are collected outside each original profiler timing routine in both arms; their surrounding process history may differ from unobserved startup.",
            "No speedup threshold gates acceptance; slower, unchanged and faster outcomes are all retained.",
        ],
    }


def control_worker(arm="cuda_graph", trial=0):
    """CPU-only raw-record fixture; it is never a performance observation."""
    tactic, shape = list(range(8)), [144, 4096, 1024]

    def settings(active=False, configured=True):
        return {
            "warmup": 3,
            "repeat": 10,
            "stream_delay_micro_secs": 5000,
            "cuda_graph_delay_micro_secs": 100,
            "use_global_timer": False,
            "effective_measure_policy": expected_policy(arm) if active else None,
            "effective_policy_timer": "events"
            if active and arm == "cuda_graph"
            else None,
            "config": {
                "use_cuda_graph": active and arm == "cuda_graph",
                "use_cold_l2_cache": False,
                "cuda_graph_profile_replays": 1,
                "profiling_repeat": None,
                "effective_profiling_repeat": 10,
            }
            if configured
            else None,
        }

    data = {
        "arm": arm,
        "trial": trial,
        "shape": shape,
        "profile_attempts": 18,
        "forbidden_profile_attempts": [],
        "profiles": [],
        "profiling_input_groups": {},
        "autotuner_settings": {
            "initial": settings(configured=False),
            "after": settings(),
        },
        "environment": {
            "measurement_policy": expected_policy(arm),
            "operand_sha256": {"x": "a" * 64},
            "native_artifact": {"library_sha256": "b" * 64, "ninja_sha256": "c" * 64},
        },
        "cache": {
            "before_sha256": None,
            "after_startup_sha256": "d" * 64,
            "after_controls_sha256": "d" * 64,
            "miss_scopes_restored": True,
            "persistence_under_test": False,
        },
        "natural": {
            "tactic": tactic,
            "cache_hit": True,
            "lookup_tier": "memory",
            "acceptance": {"tactic": tactic, "accepted": True, "used_descriptor": [1]},
        },
        "objective": {
            "requested_policy": expected_policy(arm),
            "manifest_fields": expected_fields(arm),
            "startup_api_calls": [
                {
                    "mode": "tune",
                    "persistent_cache": False,
                    "measurement_policy": expected_policy(arm),
                }
            ],
            "default_race_enabled_both_arms": True,
            "effective_hot_l2": True,
            "protocol_errors": [],
            "replay_outside_all_contexts": True,
            "final_state": {
                "active_v2": False,
                "measurement_policy": None,
                "is_tuning_mode": False,
                "tuning_contexts": 0,
                "managed_store": False,
                "caller_stream_restored": True,
            },
            "scope_records": [],
        },
        "fixture": {
            "tensors": {
                key: {"sha256": value * 64}
                for key, value in (
                    ("weight_fp8_rowmajor", "e"),
                    ("input_scale", "f"),
                    ("weight_scale", "1"),
                )
            }
        },
        "actual_layer": {
            "quantization_reference": {"actual_input_fp8_sha256": "2" * 64},
            "observed_public_inputs": [],
            "native_output_poison_checks": [],
        },
        "selections": [],
        "binding_acceptance_checks": [],
        "correctness": {"passed": True, "rtol": 0.02, "atol": 0.125, "checks": []},
        "_container_id": arm + str(trial),
    }
    for m in (1, 2, 4, 8, 16, 32, 64, 128, 144):
        shapes = [[1, m, 1024], [1, 1024, 4096], [1], [1], [1, m, 4096], [1024]]
        operands = {
            name: {
                "shape": shapes[index],
                "stride": (
                    [m * 1024, 1024, 1]
                    if index == 0
                    else [1024 * 4096, 1, 1024]
                    if index == 1
                    else [1]
                ),
                "dtype": "torch.float8_e4m3fn" if index < 2 else "torch.float32",
                "sha256": "3" * 64,
            }
            for index, name in enumerate(
                ("activation", "weight", "input_scale", "weight_scale")
            )
        }
        digest = hashlib.sha256(canonical(operands).encode()).hexdigest()
        data["profiling_input_groups"][digest] = operands
        for candidate, cost in ((-1, 2.0), (tactic, 1.0)):
            data["profiles"].append(
                {
                    "stage": "startup",
                    "runner": "CublasFp8GemmRunner",
                    "input_shapes": shapes,
                    "normalized_profile": h.normalized_profile(shapes),
                    "tactic": candidate,
                    "profiler_ms": cost,
                    "profiling_input_group": digest,
                    "use_cuda_graph": arm == "cuda_graph",
                    "use_cold_l2_cache": False,
                    "policy_identity": expected_identity(arm),
                    "policy_manifest_fields": expected_fields(arm),
                    "default_race_enabled": True,
                    "autotuner_settings": settings(active=True),
                }
            )
    data["runtime_profile_summary"] = {
        "normalized_profile": runtime_profile(shape),
        "profiles": 2,
        "finite_profiles": 2,
        "minimum_ms": 1.0,
        "minimum_tactics": [tactic],
    }
    for stage in (
        "startup",
        "natural_before",
        "winner_capture_scope",
        "default_capture_scope",
        "natural_final",
    ):
        startup = stage == "startup"
        for event in ("enter", "before_exit", "after_exit"):
            active = event != "after_exit"
            data["objective"]["scope_records"].append(
                {
                    "stage": stage,
                    "mode": "tune" if startup else "replay",
                    "event": event,
                    "measurement_policy": expected_policy(arm) if active else None,
                    "manifest_fields": expected_fields(arm) if active else {},
                    "winner_identity": expected_identity(arm) if active else "None",
                    "active_v2": active,
                    "is_tuning_mode": startup,
                    "tuning_contexts": 2 if startup and active else 1 if startup else 0,
                    "managed_store": False,
                    "on_forward_stream": startup,
                    "on_caller_stream": not startup,
                }
            )
    stages = {
        "setup": (-1, "miss", False, False),
        "startup": (tactic, "memory", False, True),
        "natural_before": (tactic, "memory", False, True),
        "natural_eager_changed_input": (tactic, "memory", False, True),
        "natural_eager_restored_input": (tactic, "memory", False, True),
        "natural_final": (tactic, "memory", False, True),
        "bare_after_scope": (-1, "miss", False, False)
        if arm == "cuda_graph"
        else (tactic, "memory", False, False),
    }
    for name in GRAPH_IDS:
        value, tier = (tactic, "memory") if name == "winner" else (-1, "miss")
        stages[name + "_before"], stages[name + "_capture"] = (
            (value, tier, False, True),
            (value, tier, True, True),
        )
    for stage, (value, tier, capture, active) in stages.items():
        data["selections"].append(
            {
                "stage": stage,
                "tactic": value,
                "lookup_tier": tier,
                "cache_hit": tier == "memory",
                "capturing": capture,
                "active_v2": active,
                "is_tuning_mode": stage == "startup",
                "policy_identity": expected_identity(arm) if active else "None",
                "measurement_policy": expected_policy(arm) if active else None,
                "on_forward_stream": stage == "startup",
                "on_caller_stream": stage != "startup",
            }
        )
        data["actual_layer"]["observed_public_inputs"].append(
            {
                "stage": stage,
                "capturing": capture,
                "backend": "cublas",
                "quantized_input_sha256": "2" * 64,
                "weight_rowmajor_sha256": "e" * 64,
            }
        )
        if stage.endswith("_changed_input"):
            data["actual_layer"]["observed_public_inputs"][-1].pop(
                "quantized_input_sha256"
            )
            data["actual_layer"]["observed_public_inputs"][-1].pop(
                "weight_rowmajor_sha256"
            )
        if stage not in ("setup", "startup"):
            data["binding_acceptance_checks"].append(
                {
                    "stage": stage,
                    "tactic": value,
                    "accepted": True,
                    "used_descriptor": [] if value == -1 else [1],
                }
            )
            data["actual_layer"]["native_output_poison_checks"].append(
                {
                    "stage": stage,
                    "capturing": capture,
                    "poisoned_actual_allocation": not capture,
                }
            )

    def tensor(pointer, tensor_shape, stride, dtype):
        return {
            "data_ptr": pointer,
            "shape": tensor_shape,
            "stride": stride,
            "dtype": dtype,
            "alignment_mod256": pointer % 256,
            "alignment_mod4096": pointer % 4096,
        }

    order = list(GRAPH_IDS if trial % 2 == 0 else reversed(GRAPH_IDS))
    graphs = []
    for index, name in enumerate(order):
        pointer = 4096 if name == "winner" else 8192
        out = tensor(pointer, [144, 4096], [4096, 1], "torch.bfloat16")
        graphs.append(
            {
                "id": name,
                "tactic": tactic if name == "winner" else -1,
                "lookup_tier": "memory" if name == "winner" else "miss",
                "policy_identity": expected_identity(arm),
                "measurement_policy": expected_policy(arm),
                "selection": next(
                    row
                    for row in data["selections"]
                    if row["stage"] == name + "_capture"
                ),
                "started_ns": 1_000_000 + index * 10_000,
                "ended_ns": 1_001_000 + index * 10_000,
                "output": out,
                "linear_input": tensor(12288, [144, 1024], [1024, 1], "torch.bfloat16"),
                "inputs": [
                    tensor(
                        16384 + index * 4096,
                        [1, 144, 1024],
                        [144 * 1024, 1024, 1],
                        "torch.float8_e4m3fn",
                    ),
                    tensor(
                        24576,
                        [1, 1024, 4096],
                        [1024 * 4096, 1, 1024],
                        "torch.float8_e4m3fn",
                    ),
                    tensor(28672, [1], [1], "torch.float32"),
                    tensor(32768, [1], [1], "torch.float32"),
                    tensor(
                        pointer, [1, 144, 4096], [144 * 4096, 4096, 1], "torch.bfloat16"
                    ),
                    tensor(36864, [1024], [1], "torch.uint8"),
                ],
                "quantized_input_sha256": "2" * 64,
                "weight_rowmajor_sha256": "e" * 64,
                "input_scale_sha256": "f" * 64,
                "weight_scale_sha256": "1" * 64,
            }
        )
    scope = {
        "winner_entries_before": 9,
        "winner_entries_after_removal": 0,
        "file_entries_after_removal": 0,
        "restored": True,
        "all_cache_state_restored": True,
        "cache_sha256_before": "d" * 64,
        "cache_sha256_after": "d" * 64,
        "winner_contents_sha256_before": "4" * 64,
        "winner_contents_sha256_after": "4" * 64,
        "file_contents_sha256_before": "5" * 64,
        "file_contents_sha256_after": "5" * 64,
    }
    controls = {
        "graphs": graphs,
        "capture_order": order,
        "miss_scope": [scope],
        "common_iterations": 5120,
        "schedule_seed": 5000 + trial,
        "schedule": expected_schedule(trial),
        "sequential": [],
        "blocks": [],
    }
    data["graph_controls"] = controls
    measurement = {
        "effective_iterations": 5120,
        "requested_iterations": 5120,
        "iterations_per_sample": 5120,
        "warmup_actual_ms": 500,
        "warmup_calls": 5120,
        "calibration_batches": [{"iterations": 5120, "wall_ms": 20}],
        "samples_ms": [20 / 5120] * 15,
        "wall_batch_ms": [20] * 15,
        "gpu_event_samples_ms": [10 / 5120] * 15,
        "gpu_event_batch_ms": [10] * 15,
        "median_ms": 20 / 5120,
    }
    now = 10_000_000
    for name in GRAPH_IDS:
        controls["sequential"].append(
            {
                "graph_id": name,
                "correctness_check": "sequential_" + name,
                "measurement": copy.deepcopy(measurement),
                "started_ns": now,
                "ended_ns": now + 400_000_000,
                "policy_scope_active": False,
                "tuning_mode": False,
                "profile_attempts_before": 18,
                "profile_attempts_after": 18,
            }
        )
        now += 500_000_000
    for index, order in enumerate(controls["schedule"]):
        block = {"index": index, "order": order, "measurements": []}
        for name in order:
            event = 11 if name == "winner" else 10  # A valid regression must pass.
            block["measurements"].append(
                {
                    "graph_id": name,
                    "iterations": 5120,
                    "wall_batch_ms": 20,
                    "wall_per_call_ms": 20 / 5120,
                    "gpu_event_batch_ms": event,
                    "gpu_event_per_call_ms": event / 5120,
                    "started_ns": now,
                    "ended_ns": now + 20_000_000,
                    "rewarm_actual_ms": 20,
                    "rewarm_calls": 5120,
                    "correctness_check": f"block_{index:02d}_{name}",
                    "policy_scope_active": False,
                    "tuning_mode": False,
                }
            )
            now += 30_000_000
        controls["blocks"].append(block)
    checks = (
        set(stages) - {"setup", "startup", "winner_capture", "default_capture"}
    ) | {"initial_untuned"}
    owners = {}
    for name in GRAPH_IDS:
        for stage in [
            name + "_changed_input",
            name + "_restored_input",
            "sequential_" + name,
            "final_" + name,
            *(f"block_{index:02d}_{name}" for index in range(24)),
        ]:
            checks.add(stage)
            owners[stage] = name
    for stage in sorted(checks):
        owner = owners.get(stage)
        metrics = {
            "max_abs_error": 0,
            "max_normalized_error": 0,
            "rmse": 0,
            "relative_l2": 0,
        }
        data["correctness"]["checks"].append(
            {
                "check": stage,
                "graph_id": owner,
                "output_data_ptr": (4096 if owner == "winner" else 8192)
                if owner
                else 65536,
                "quantized_reference": {**metrics, "passed": True},
                "bf16_reference": {**metrics, "diagnostic_only": True},
                "max_abs_error": 0,
                "output_sha256": "6" * 64,
                "exact_zero_required": stage.endswith("_changed_input"),
            }
        )
    return data


def validate_control(data):
    validate_objective(data, "control")
    winner = validate_profiles(data, "control")
    graphs = validate_graphs(data, winner, "control")
    checker = validate_observed_stages(data, graphs, winner, "control")
    validate_timing(data, graphs, checker, "control")


def self_test():
    passed = []

    def positive(name, callback):
        callback()
        passed.append(name)

    def rejected(name, mutate, validator=validate_control):
        data = control_worker()
        mutate(data)
        try:
            validator(data)
        except (ValueError, KeyError, TypeError):
            passed.append(name)
        else:
            raise ValueError("fault control unexpectedly accepted: " + name)

    for arm in ARMS:
        for trial in (0, 1):
            positive(
                f"complete_{arm}_trial{trial}_regression_retained",
                lambda arm=arm, trial=trial: validate_control(
                    control_worker(arm, trial)
                ),
            )
    source_environment = {"flashinfer_autotune_cache_sha256": AUTOTUNE_CACHE_SHA256}
    source_plan = {
        "flashinfer_source": {"autotune_cache_sha256": AUTOTUNE_CACHE_SHA256}
    }
    positive(
        "exact_policy_cache_source_pin",
        lambda: validate_policy_source(source_environment, source_plan, "control"),
    )
    for name, environment, plan in (
        (
            "wrong_actual_policy_cache_source",
            {"flashinfer_autotune_cache_sha256": "0" * 64},
            source_plan,
        ),
        (
            "wrong_plan_policy_cache_source",
            source_environment,
            {"flashinfer_source": {"autotune_cache_sha256": "0" * 64}},
        ),
    ):
        try:
            validate_policy_source(environment, plan, "control")
        except ValueError:
            passed.append(name)
        else:
            raise ValueError("policy/cache source change accepted")
    positive(
        "zero_input_layout_only_observation", lambda: validate_control(control_worker())
    )
    rejected(
        "restored_input_missing_byte_evidence",
        lambda d: next(
            row
            for row in d["actual_layer"]["observed_public_inputs"]
            if row["stage"] == "natural_eager_restored_input"
        ).pop("weight_rowmajor_sha256"),
    )
    default_winner = control_worker()
    for row in default_winner["profiles"]:
        row["profiler_ms"] = 1.0 if row["tactic"] == -1 else 2.0
    default_winner["natural"].update(
        tactic=-1, acceptance={"tactic": -1, "accepted": True, "used_descriptor": []}
    )
    default_winner["runtime_profile_summary"]["minimum_tactics"] = [-1]
    for row in (
        default_winner["selections"] + default_winner["binding_acceptance_checks"]
    ):
        if row.get("lookup_tier") == "memory" or row["tactic"] != -1:
            row["tactic"] = -1
        if "used_descriptor" in row:
            row["used_descriptor"] = []
    for row in default_winner["graph_controls"]["graphs"]:
        if row["id"] == "winner":
            row["tactic"] = -1
    positive(
        "default_winner_valid_without_speedup", lambda: validate_control(default_winner)
    )
    rejected(
        "wrong_requested_policy", lambda d: d["objective"].update(requested_policy=None)
    )
    rejected("missing_scope_lifecycle", lambda d: d["objective"]["scope_records"].pop())
    rejected(
        "policy_leaks_outside_scope",
        lambda d: d["objective"]["scope_records"][-1].update(active_v2=True),
    )
    rejected(
        "tuning_leaks_after_scope",
        lambda d: d["objective"]["final_state"].update(tuning_contexts=1),
    )
    rejected(
        "changed_profile_counts",
        lambda d: d["profiles"][0]["autotuner_settings"].update(repeat=9),
    )
    rejected(
        "wrong_effective_graph_objective",
        lambda d: d["profiles"][0].update(use_cuda_graph=False),
    )
    rejected(
        "cold_l2_changed", lambda d: d["profiles"][0].update(use_cold_l2_cache=True)
    )
    rejected(
        "profiling_outside_startup",
        lambda d: d["profiles"][0].update(stage="winner_capture"),
    )
    rejected(
        "default_not_profiled", lambda d: d["profiles"][0].update(tactic=list(range(8)))
    )
    rejected(
        "default_profile_not_finite",
        lambda d: d["profiles"][0].update(profiler_ms=None),
    )
    rejected(
        "selected_tactic_not_profile_minimum", lambda d: d["natural"].update(tactic=-1)
    )
    rejected(
        "profile_input_bytes_corrupt",
        lambda d: next(iter(d["profiling_input_groups"].values()))["activation"].update(
            sha256="9" * 64
        ),
    )
    rejected(
        "policy_identity_mismatch",
        lambda d: d["profiles"][0].update(policy_identity="None"),
    )
    rejected(
        "default_capture_hit",
        lambda d: next(
            r for r in d["graph_controls"]["graphs"] if r["id"] == "default"
        )["selection"].update(lookup_tier="memory"),
    )
    rejected(
        "winner_capture_miss",
        lambda d: next(r for r in d["graph_controls"]["graphs"] if r["id"] == "winner")[
            "selection"
        ].update(cache_hit=False),
    )
    rejected(
        "shared_output_allocation",
        lambda d: d["graph_controls"]["graphs"][1]["output"].update(data_ptr=4096),
    )
    rejected(
        "miss_cache_not_removed",
        lambda d: d["graph_controls"]["miss_scope"][0].update(
            winner_entries_after_removal=1
        ),
    )
    rejected(
        "miss_cache_not_restored",
        lambda d: d["graph_controls"]["miss_scope"][0].update(
            all_cache_state_restored=False
        ),
    )
    rejected(
        "numerical_failure",
        lambda d: d["correctness"]["checks"][0]["quantized_reference"].update(
            passed=False
        ),
    )
    rejected(
        "checked_wrong_output",
        lambda d: next(
            r for r in d["correctness"]["checks"] if r["check"] == "final_winner"
        ).update(output_data_ptr=8192),
    )
    rejected(
        "changed_input_not_zero",
        lambda d: next(
            r
            for r in d["correctness"]["checks"]
            if r["check"] == "winner_changed_input"
        )["quantized_reference"].update(max_abs_error=0.1),
    )
    rejected("missing_block", lambda d: d["graph_controls"]["blocks"].pop())
    rejected(
        "unbalanced_block_schedule",
        lambda d: d["graph_controls"]["schedule"][0].reverse(),
    )
    rejected(
        "wrong_replay_count",
        lambda d: d["graph_controls"]["blocks"][0]["measurements"][0].update(
            iterations=20
        ),
    )
    rejected(
        "bad_raw_normalization",
        lambda d: d["graph_controls"]["blocks"][0]["measurements"][0].update(
            wall_per_call_ms=1
        ),
    )
    rejected(
        "timed_policy_scope",
        lambda d: d["graph_controls"]["blocks"][0]["measurements"][0].update(
            policy_scope_active=True
        ),
    )
    rejected(
        "sequential_profiling",
        lambda d: d["graph_controls"]["sequential"][0].update(
            profile_attempts_after=19
        ),
    )
    rejected("numerical_stage_missing", lambda d: d["correctness"]["checks"].pop())
    cohort = [control_worker(arm) for arm in ARMS]
    positive("cohort_only_objective_differs", lambda: validate_cohort(cohort))
    changed = copy.deepcopy(cohort)
    changed[1]["profiles"][1]["tactic"] = list(range(1, 9))
    try:
        validate_cohort(changed)
    except ValueError:
        passed.append("cross_arm_candidate_change_rejected")
    else:
        raise ValueError("candidate change accepted")
    effect = worker_effects(cohort[0])[0]
    positive(
        "median_of_paired_block_ratios",
        lambda: require(math.isclose(effect["ratio"], 1.1), "paired block statistic"),
    )
    summary = summarize(cohort)
    positive(
        "one_trial_checkpoint_no_interval",
        lambda: require(
            all(
                row["exploratory_cluster_bootstrap_95_percent"] is None
                for row in summary["within_arm"]
            ),
            "one-trial interval",
        ),
    )
    positive(
        "cross_arm_normalized_ratio_of_ratios",
        lambda: require(
            all(row["median_ratio"] == 1 for row in summary["paired_cross_arm"]),
            "ratio of ratios",
        ),
    )
    unequal = copy.deepcopy(cohort)
    for block in unequal[1]["graph_controls"]["blocks"]:
        for row in block["measurements"]:
            value = 9 if row["graph_id"] == "winner" else 12
            row["gpu_event_batch_ms"], row["gpu_event_per_call_ms"] = (
                value,
                value / 5120,
            )
    unequal_summary = summarize(unequal)
    normalized = next(
        row for row in unequal_summary["paired_cross_arm"] if row["primary"]
    )
    positive(
        "nontrivial_ratio_of_ratios",
        lambda: require(
            math.isclose(normalized["median_ratio"], 0.75 / 1.1),
            "normalized paired effect",
        ),
    )
    varied = control_worker()
    for index, block in enumerate(varied["graph_controls"]["blocks"]):
        winner, default = ((2, 1), (5, 5), (3, 1))[index % 3]
        for row in block["measurements"]:
            value = winner if row["graph_id"] == "winner" else default
            row["gpu_event_batch_ms"], row["gpu_event_per_call_ms"] = (
                value,
                value / 5120,
            )
    positive(
        "paired_ratio_precedes_trial_median",
        lambda: require(
            worker_effects(varied)[0]["ratio"] == 2, "median of within-block ratios"
        ),
    )
    values = [0.8, 0.9, 1, 1.1, 1.2, 1.4]
    distribution = sorted(
        statistics.median(values[i] for i in indices)
        for indices in itertools.product(range(6), repeat=6)
    )
    interval = h.interval_for_trial_median(values)
    positive(
        "whole_trial_bootstrap_46656_resamples",
        lambda: require(
            interval["ordered_resamples"] == 46656
            and interval["lower"] == h.quantile(distribution, 0.025)
            and interval["upper"] == h.quantile(distribution, 0.975),
            "whole-trial bootstrap",
        ),
    )
    try:
        summarize_ratios([{"trial": 0, "ratio": 1}, {"trial": 0, "ratio": 2}])
    except ValueError:
        passed.append("duplicate_independent_trial_rejected")
    else:
        raise ValueError("duplicate trial accepted")
    return {
        "status": "success",
        "cpu_only": True,
        "controls": len(passed),
        "passed": passed,
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
    if args.self_test:
        result = self_test()
    else:
        for name in (
            "raw_root",
            "source_root",
            "source_manifest",
            "fixture_manifest",
            "expected_source_commit",
        ):
            if getattr(args, name) is None:
                parser.error("--" + name.replace("_", "-") + " is required")
        result = analyze(args)
    content = h.json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
