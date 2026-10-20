#!/usr/bin/env python3
"""Validate a fixed real-activation SGLang FP8 linear-layer comparison.

Six fresh trial pairs are the independent units. Natural layer timings retain
the actual untuned miss or tuned memory/file lookup path for each arm.
"""

from __future__ import annotations

import argparse
import ast
import collections
import functools
import hashlib
import itertools
import json
import math
import re
import statistics
import subprocess
from pathlib import Path

ARMS = ("default", "v1", "v2")
PHASES = ("cold", "warm")
TRIALS = tuple(range(6))
EXECUTIONS = ("eager", "cuda_graph")
WORKER = "test/manual/kernels/flashinfer_autotune_linear_layer.py"
FP8_UTILS = "python/sglang/srt/layers/quantization/fp8_utils.py"
WRAPPER = "python/sglang/srt/model_executor/runner/flashinfer_autotune.py"
BATCH = "test/manual/kernels/results/flashinfer_autotune_selection/20261020T034000Z-h100-fp8-linear"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load(path):
    def invalid(value):
        raise ValueError(f"nonfinite JSON value: {value}")

    def finite_float(value):
        number = float(value)
        require(math.isfinite(number), "nonfinite JSON numeric literal: " + value)
        return number

    return json.loads(
        path.read_text(), parse_constant=invalid, parse_float=finite_float
    )


def positive(value):
    return type(value) in (float, int) and math.isfinite(value) and value > 0


def descriptor(tactic):
    return (
        isinstance(tactic, list)
        and len(tactic) == 8
        and all(type(word) is int and 0 <= word < 1 << 64 for word in tactic)
    )


def valid_tactic(tactic):
    return tactic == -1 or descriptor(tactic)


def git_blob(args, path):
    return cached_git_blob(str(args.source_root), args.expected_source_commit, path)


@functools.lru_cache(maxsize=None)
def cached_git_blob(root, commit, path):
    return subprocess.check_output(
        ["git", "show", commit + ":" + path],
        cwd=root,
    )


def source_hash(args, path):
    return hashlib.sha256(git_blob(args, path)).hexdigest()


def locate(root, stored_path):
    path = Path(stored_path)
    if "caches" in path.parts:
        offset = path.parts.index("caches") + 1
        candidate = root / "caches" / Path(*path.parts[offset:])
        require(candidate.is_file(), "missing cache artifact: " + str(candidate))
        return candidate
    matches = list(root.rglob(path.name))
    require(len(matches) == 1, "missing or ambiguous artifact: " + stored_path)
    return matches[0]


def runtime_bucket(m):
    return 1 << (m - 1).bit_length()


def statistics_for(values):
    require(
        values and all(positive(value) for value in values), "invalid statistic input"
    )
    return {
        "count": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
    }


def validate_timing(row, label):
    iterations = row["effective_iterations"]
    require(
        type(iterations) is int and 20 <= iterations <= 65536, label + ": iterations"
    )
    require(row["requested_iterations"] == 20, label + ": requested iterations")
    require(
        row["iterations_per_sample"] == iterations, label + ": sample normalization"
    )
    require(
        row["warmup_actual_ms"] >= 500 and row["warmup_calls"] >= 20, label + ": warmup"
    )
    calibration = row["calibration_batches"]
    require(
        calibration and calibration[0]["iterations"] == 20,
        label + ": calibration start",
    )
    require(
        calibration[-1]["iterations"] == iterations
        and calibration[-1]["wall_ms"] >= 20,
        label + ": calibration end",
    )
    for previous, current in zip(calibration, calibration[1:]):
        require(previous["wall_ms"] < 20, label + ": unnecessary calibration")
        require(
            current["iterations"] == min(previous["iterations"] * 2, 65536),
            label + ": calibration doubling",
        )
    for samples, batches in (
        ("samples_ms", "wall_batch_ms"),
        ("gpu_event_samples_ms", "gpu_event_batch_ms"),
    ):
        require(len(row[samples]) == len(row[batches]) == 15, label + ": sample count")
        for sample, batch in zip(row[samples], row[batches]):
            require(
                positive(sample) and positive(batch), label + ": nonpositive timing"
            )
            require(
                math.isclose(sample * iterations, batch, rel_tol=1e-12),
                label + ": inconsistent normalization",
            )
    require(
        math.isclose(
            row["median_ms"], statistics.median(row["samples_ms"]), rel_tol=1e-12
        ),
        label + ": incorrect median",
    )


def expected_sources(manifest):
    return {
        row["installed_package_path"]: row["candidate_sha256"]
        for row in manifest["files"]
        if row["runtime_overlay"]
    }


def validate_generation(payload, label):
    content = {key: value for key, value in payload.items() if key != "_generation"}
    require(
        payload["_generation"]
        == hashlib.sha256(canonical(content).encode()).hexdigest(),
        label + ": invalid persisted generation",
    )


def quantile(values, probability):
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@functools.lru_cache(maxsize=8)
def bootstrap_samples(n):
    """Exact multinomial resampling of whole trial units, represented compactly."""
    samples = []
    for indices in itertools.combinations_with_replacement(range(n), n):
        counts = collections.Counter(indices)
        weight = math.factorial(n)
        for count in counts.values():
            weight //= math.factorial(count)
        samples.append((indices, weight))
    require(sum(weight for _, weight in samples) == n**n, "bootstrap weights")
    return samples


def interval_for_trial_median(values):
    if len(values) < 2:
        return None
    distribution = sorted(
        (statistics.median(values[i] for i in indices), weight)
        for indices, weight in bootstrap_samples(len(values))
    )
    total = len(values) ** len(values)

    def at(index):
        cumulative = 0
        for value, weight in distribution:
            cumulative += weight
            if cumulative > index:
                return value
        raise ValueError("bootstrap quantile index")

    def percentile(p):
        position = p * (total - 1)
        lo, hi = math.floor(position), math.ceil(position)
        return at(lo) + (at(hi) - at(lo)) * (position - lo)

    return {
        "level": 0.95,
        "lower": percentile(0.025),
        "upper": percentile(0.975),
        "method": "exhaustive percentile bootstrap of paired trial units; median statistic",
        "ordered_resamples": total,
        "independent_units": len(values),
        "exploratory_unadjusted": True,
    }


def summarize_ratios(rows):
    values = [row["ratio"] for row in rows]
    require(
        len({row["trial"] for row in rows}) == len(rows), "trial pooled or repeated"
    )
    return {
        "trial_estimates": rows,
        "independent_trials": len(rows),
        "median_ratio": statistics.median(values),
        "min_ratio": min(values),
        "max_ratio": max(values),
        "ratios_below_one": sum(v < 1 for v in values),
        "ratios_above_one": sum(v > 1 for v in values),
        "exploratory_cluster_bootstrap_95_percent": interval_for_trial_median(values),
    }


def validate_outer(args, trials):
    plan_bytes = git_blob(args, BATCH + "/plan.json")
    plan = json.loads(plan_bytes)
    outer = load(args.raw_root / "manifest.json")
    require(
        outer["plan_sha256"] == hashlib.sha256(plan_bytes).hexdigest(),
        "launcher plan hash",
    )
    require(
        outer["source_manifest_sha256"]
        == sha256(args.source_manifest)
        == plan["flashinfer_source"]["source_manifest_sha256"],
        "overlay manifest binding",
    )
    require(
        outer["fixture_manifest_sha256"]
        == sha256(args.fixture_manifest)
        == plan["fixture"]["manifest_sha256"],
        "fixture manifest binding",
    )
    require(
        sha256(args.raw_root / "fixture" / "manifest.json")
        == sha256(args.fixture_manifest),
        "archived fixture manifest differs",
    )
    fixture = load(args.fixture_manifest)
    validate_fixture_manifest(fixture, args)
    require(
        outer["fixture_tensor_file_sha256"]
        == fixture["tensor_file"]["sha256"]
        == plan["fixture"]["tensor_file_sha256"],
        "private tensor fixture binding",
    )
    tensor_file = args.raw_root / "fixture" / fixture["tensor_file"]["name"]
    if tensor_file.is_file():
        require(
            sha256(tensor_file) == fixture["tensor_file"]["sha256"]
            and tensor_file.stat().st_size == fixture["tensor_file"]["size_bytes"],
            "private fixture bytes differ",
        )
    require(
        (args.raw_root / "execution-commit.txt").read_text().strip()
        == args.expected_source_commit,
        "execution commit",
    )
    for path, value in outer["source_sha256"].items():
        require(source_hash(args, path) == value, "source differs: " + path)
    require(
        {
            WORKER,
            WRAPPER,
            FP8_UTILS,
            BATCH + "/analyze.py",
            BATCH + "/capture_fixture.py",
        }
        <= set(outer["source_sha256"]),
        "source coverage",
    )
    image = load(args.raw_root / "image.json")
    require(
        image["Architecture"] == "amd64" and image["Os"] == "linux", "image platform"
    )
    expected_image = plan["container_image"].split("@")[-1]
    require(
        any(value.split("@")[-1] == expected_image for value in image["RepoDigests"]),
        "pinned image digest",
    )
    require(
        (args.raw_root / "prepare.exit-code").read_text().strip() == "0",
        "prepare failed",
    )
    require(
        not (args.raw_root / "gpu-processes-before.csv").read_text().strip(),
        "GPU not idle",
    )
    for trial in trials:
        require(
            (args.raw_root / f"trial{trial}.launcher.exit-code").read_text().strip()
            == "0",
            f"trial{trial} incomplete",
        )
    environment = plan["environment"]
    require(
        outer["gpu"]["name"] == environment["gpu"]
        and outer["gpu"]["uuid"].removeprefix("GPU-")
        == environment["physical_gpu_uuid"]
        and outer["gpu"]["driver_version"] == environment["driver"],
        "hardware differs",
    )
    return {
        "plan": plan,
        "fixture_manifest": fixture,
        "image": image,
        "gpu": outer["gpu"],
        "environment": environment,
        "private_fixture_locally_rehashed": tensor_file.is_file(),
        "launcher_manifest_sha256": sha256(args.raw_root / "manifest.json"),
        "plan_sha256": outer["plan_sha256"],
        "fixture_manifest_sha256": sha256(args.fixture_manifest),
    }


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_fixture_manifest(fixture, args):
    require(
        fixture["schema_version"] == 1
        and len(fixture["shape"]) == 3
        and fixture["shape"][0] == 144
        and all(type(d) is int and d > 0 for d in fixture["shape"]),
        "fixture shape/schema",
    )
    m, n, k = fixture["shape"]
    model, capture = fixture["model"], fixture["capture"]
    require(
        model["repo_id"] == "Qwen/Qwen3-0.6B"
        and model["revision"]
        == model["resolved_snapshot_revision"]
        == "c1899de289a04d12100db370d81485cdf75e47ca",
        "checkpoint revision",
    )
    require(
        capture["layer"] == "model.layers.0.self_attn"
        and capture["hook"] == "model.layers.0.self_attn.q_proj:forward_pre_hook"
        and capture["captured_shape"] == [1, m, k]
        and capture["saved_shape"] == [m, k]
        and capture["weight_order"] == ["q_proj", "k_proj", "v_proj"]
        and capture["tensor_parallel_size"] == 1
        and capture["bias"] is None
        and capture["full_bf16_forward_completed"] is True
        and capture["model_training"] is False
        and capture["seed"] == 19,
        "captured model-layer provenance",
    )
    shapes = capture["projection_shapes"]
    require(
        len(shapes) == 3
        and all(len(shape) == 2 and shape[1] == k for shape in shapes)
        and sum(shape[0] for shape in shapes) == n,
        "fused QKV shape/order",
    )
    tensors = fixture["tensors"]
    expected = {
        "input_bf16": ("torch.bfloat16", [m, k], 2),
        "weight_fp8_rowmajor": ("torch.float8_e4m3fn", [n, k], 1),
        "weight_bf16_rowmajor": ("torch.bfloat16", [n, k], 2),
        "input_scale": ("torch.float32", [1], 4),
        "weight_scale": ("torch.float32", [1], 4),
        "input_ids": ("torch.int64", [1, m], 8),
    }
    require(set(tensors) == set(expected), "fixture tensor coverage")
    for name, (dtype, shape, size) in expected.items():
        row = tensors[name]
        require(
            row["dtype"] == dtype
            and row["shape"] == shape
            and row["nbytes"] == math.prod(shape) * size
            and valid_sha(row["sha256"]),
            "fixture tensor metadata: " + name,
        )
    tokenization = fixture["tokenization"]
    require(
        len(tokenization["input_ids"]) == 1
        and len(tokenization["input_ids"][0]) == m
        and tokenization["attention_mask"] == [[1] * m]
        and tokenization["add_special_tokens"] is False
        and tokenization["truncation"] is True
        and tokenization["max_length"] == 144
        and tokenization["padding"] is False,
        "fixed tokenization",
    )
    quant = fixture["quantization"]
    require(
        quant["dtype"] == "torch.float8_e4m3fn"
        and quant["max_finite"] == 448
        and quant["scale_rule"] == "absmax_fp32 / 448"
        and quant["weight_rule"]
        == "clamp(weight.float() / weight_scale,-448,448).to(float8_e4m3fn)"
        and quant["independent_input_rule"]
        == "clamp(input.float() * input_scale.reciprocal(),-448,448).to(float8_e4m3fn)"
        and valid_sha(quant["independent_input_fp8_sha256"])
        and quant["calibration_batches"] == 1,
        "fixture quantization provenance",
    )
    source = fixture["source"]
    require(
        re.fullmatch(r"[0-9a-f]{40}", source["sglang_commit"]) is not None,
        "capture source commit",
    )
    capture_path = BATCH + "/capture_fixture.py"
    capture_bytes = cached_git_blob(
        str(args.source_root), source["sglang_commit"], capture_path
    )
    require(
        hashlib.sha256(capture_bytes).hexdigest()
        == source["capture_script_sha256"]
        == source_hash(args, capture_path),
        "capture script changed after fixture capture",
    )
    require(
        FP8_UTILS in source["sglang_source_sha256"], "capture SGLang source coverage"
    )
    for path, value in source["sglang_source_sha256"].items():
        require(
            hashlib.sha256(
                cached_git_blob(str(args.source_root), source["sglang_commit"], path)
            ).hexdigest()
            == value
            == source_hash(args, path),
            "captured source differs: " + path,
        )
    require(
        fixture["checkpoint_files"]
        and all(
            valid_sha(row["sha256"]) and row["size_bytes"] > 0
            for row in fixture["checkpoint_files"].values()
        ),
        "checkpoint file provenance",
    )
    require(
        fixture["tensor_file"]["name"] == "fixture.safetensors"
        and valid_sha(fixture["tensor_file"]["sha256"])
        and fixture["tensor_file"]["size_bytes"] > 0,
        "private fixture file provenance",
    )


def validate_settings(data, label):
    settings = data["autotuner_settings"]
    initial, after = settings["initial"], settings["after"]
    require(initial["config"] is None, label + ": initial config")
    for value in (initial, after):
        require(
            value["warmup"] == 3
            and value["repeat"] == 10
            and value["stream_delay_micro_secs"] == 5000
            and value["cuda_graph_delay_micro_secs"] == 100,
            label + ": tuner timing defaults changed",
        )
        require(
            type(value["use_global_timer"]) is bool
            and value["effective_measure_policy"] is None
            and value["effective_policy_timer"] is None,
            label + ": unspecified measurement policy changed",
        )
    require(
        {k: v for k, v in initial.items() if k != "config"}
        == {k: v for k, v in after.items() if k != "config"},
        label + ": initial/final timer state changed",
    )
    require(
        after["config"]
        == {
            "use_cuda_graph": False,
            "use_cold_l2_cache": False,
            "cuda_graph_profile_replays": 1,
            "profiling_repeat": None,
            "effective_profiling_repeat": 10,
        },
        label + ": profiling operation defaults",
    )


def validate_fixture_worker(data, args, provenance, label):
    fixture = provenance["fixture_manifest"]
    observed = data["fixture"]
    require(
        observed["manifest_sha256"] == sha256(args.fixture_manifest)
        and observed["tensor_file_sha256"] == fixture["tensor_file"]["sha256"],
        label + ": loaded fixture hashes",
    )
    require(
        all(
            observed[key] == fixture[key]
            for key in ("shape", "model", "capture", "tensors")
        ),
        label + ": loaded fixture metadata",
    )
    require(
        data["environment"]["operand_sha256"]
        == {name: row["sha256"] for name, row in fixture["tensors"].items()},
        label + ": actual operand bytes differ from fixture",
    )
    layer = data["actual_layer"]
    require(
        layer["function"] == "apply_fp8_linear_bmm_flashinfer"
        and layer["custom_op"] == "flashinfer_bmm_fp8"
        and layer["quantizer"] == "static_quant_fp8",
        label + ": actual operation path",
    )
    require(
        layer["inference_mode"] is True
        and layer["grad_enabled"] is False
        and layer["allow_tf32"] is False
        and layer["dispatch_modes"] == 0
        and layer["function_modes"] is False
        and layer["torch_profiler_enabled"] is False
        and layer["compiling"] is False
        and layer["kernel_api_log_level"] == 0
        and layer["custom_op_direct_wrapper"] is True,
        label + ": normal uninstrumented inference path",
    )
    require(
        layer["source_sha256"] == fixture["source"]["sglang_source_sha256"],
        label + ": actual layer source coverage",
    )
    for path, value in layer["source_sha256"].items():
        require(
            source_hash(args, path) == value, label + ": actual layer source: " + path
        )
    source = git_blob(args, FP8_UTILS).decode()
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "apply_fp8_linear_bmm_flashinfer"
    )
    function_bytes = "".join(
        source.splitlines(keepends=True)[function.lineno - 1 : function.end_lineno]
    ).encode()
    require(
        layer["function_source_sha256"] == hashlib.sha256(function_bytes).hexdigest(),
        label + ": actual function source",
    )
    quant = layer["quantization_reference"]
    require(
        quant["byte_match"] is True
        and quant["cpu_and_gpu_independent_references_match"] is True
        and quant["actual_input_fp8_sha256"]
        == quant["independent_input_fp8_sha256"]
        == fixture["quantization"]["independent_input_fp8_sha256"]
        and quant["scale_sha256"] == fixture["tensors"]["input_scale"]["sha256"],
        label + ": actual static quantization bytes/reference",
    )
    require(
        layer["graph_output_binding"]
        == "actual captured allocation; output rebinding occurs outside timing",
        label + ": actual graph output binding",
    )
    m, n, k = fixture["shape"]
    stages = {
        "setup",
        "startup",
        "natural_eager_changed_input",
        "natural_eager_restored_input",
        "natural_before",
        "natural_capture",
        "natural_after",
        "natural_after_oracle",
    }
    observations = layer["observed_public_inputs"]
    require(
        collections.Counter(row["stage"] for row in observations)
        == collections.Counter(stages),
        label + ": actual native operand observation coverage",
    )
    byte_stages = {
        "setup",
        "natural_eager_restored_input",
        "natural_before",
        "natural_after",
        "natural_after_oracle",
    }
    for row in observations:
        stage = row["stage"]
        require(
            row["capturing"] is (stage == "natural_capture")
            and row["backend"] == "cublas",
            label + ": observed capture/backend",
        )
        inputs = row["inputs"]
        shapes = [[1, m, k], [1, k, n], [1], [1], [1, m, n]]
        dtypes = [
            "torch.float8_e4m3fn",
            "torch.float8_e4m3fn",
            "torch.float32",
            "torch.float32",
            "torch.bfloat16",
        ]
        require(
            len(inputs) == 6
            and all(
                inputs[i]["shape"] == shape and inputs[i]["dtype"] == dtype
                for i, (shape, dtype) in enumerate(zip(shapes, dtypes))
            ),
            label + ": observed actual operand shape/dtype",
        )
        require(
            inputs[0]["stride"][-2:] == [k, 1]
            and inputs[1]["stride"][-2:] == [1, k]
            and inputs[2]["stride"] == inputs[3]["stride"] == [1]
            and inputs[4]["stride"][-2:] == [n, 1]
            and len(inputs[5]["shape"]) == 1
            and inputs[5]["shape"][0] > 0
            and inputs[5]["stride"] == [1],
            label + ": actual operand layout/workspace",
        )
        expected_bytes = {
            "quantized_input_sha256": quant["actual_input_fp8_sha256"],
            "weight_rowmajor_sha256": fixture["tensors"]["weight_fp8_rowmajor"][
                "sha256"
            ],
            "input_scale_sha256": fixture["tensors"]["input_scale"]["sha256"],
            "weight_scale_sha256": fixture["tensors"]["weight_scale"]["sha256"],
        }
        if stage in byte_stages:
            require(
                all(row[key] == value for key, value in expected_bytes.items()),
                label + ": native bytes/fixture binding",
            )
        else:
            require(
                not (set(row) & set(expected_bytes)),
                label + ": byte read during capture/zero-input/startup",
            )
    poison = layer["native_output_poison_checks"]
    native_stages = stages - {"setup", "startup"}
    require(
        collections.Counter(row["stage"] for row in poison)
        == collections.Counter(native_stages),
        label + ": actual allocated output poison coverage",
    )
    for row in poison:
        capture = row["stage"] == "natural_capture"
        require(
            row["capturing"] is capture
            and row["poisoned_actual_allocation"] is not capture,
            label + ": wrong allocated output poison/capture boundary",
        )


def validate_numerical_details(data, label):
    zero_checks = {"natural_changed_input", "natural_eager_changed_input"}
    for row in data["correctness"]["checks"]:
        quantized, bf16 = row["quantized_reference"], row["bf16_reference"]
        require(
            quantized["passed"] is True and bf16["diagnostic_only"] is True,
            label + ": quantized correctness/BF16 diagnostic roles",
        )
        for metrics in (quantized, bf16):
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
                label + ": finite reference error metrics",
            )
        require(
            row["max_abs_error"] == quantized["max_abs_error"]
            and quantized["max_normalized_error"] <= 1,
            label + ": quantized numerical gate",
        )
        require(valid_sha(row["output_sha256"]), label + ": checked output digest")
        zero = row["check"] in zero_checks
        require(row["exact_zero_required"] is zero, label + ": changed-input zero gate")
        if zero:
            require(
                all(
                    quantized[key] == 0 and bf16[key] == 0
                    for key in ("max_abs_error", "rmse", "relative_l2")
                ),
                label + ": changed-input replay/eager output is not exactly zero",
            )


def validate_acceptance(value, tactic, label, accepted=True):
    """Check actual native return evidence, not just a schema-valid descriptor."""
    require(value["tactic"] == tactic and value["accepted"] is accepted, label)
    expected = [] if tactic == -1 else ([1] if accepted else [0])
    require(
        value["used_descriptor"] == expected, label + ": native acceptance evidence"
    )
    require(accepted or descriptor(tactic), label + ": default cannot be excluded")


def normalized_profile(shapes):
    require(
        len(shapes) == 6 and all(isinstance(shape, list) and shape for shape in shapes),
        "malformed profiling shapes",
    )
    result = [list(shape) for shape in shapes]
    require(
        len(result[0]) == len(result[1]) == len(result[4]) == 3,
        "malformed GEMM dimensions",
    )
    result[0][-2] = runtime_bucket(result[0][-2])
    result[4][-2] = -1
    result[5][0] = -1
    return result


def validate_worker(path, args, manifest, provenance):
    data = load(path)
    label = path.name
    require(data["status"] == "success" and data["schema_version"] == 1, label)
    require(data["implementation"] == "candidate", label + ": source arm")
    arm, phase, m = data["arm"], data["phase"], data["shape"][0]
    require(arm in ARMS and phase in PHASES and m == 144, label)
    require(
        data["seed"] == 19
        and data["shape"] == provenance["fixture_manifest"]["shape"]
        and data["trial"] in TRIALS,
        label + ": workload",
    )
    require(
        data["worker_sha256"] == source_hash(args, WORKER), label + ": worker source"
    )
    require(
        data["manifest_sha256"] == sha256(args.source_manifest),
        label + ": source manifest",
    )
    validate_fixture_worker(data, args, provenance, label)
    env = data["environment"]
    require(env["sglang_commit"] == args.expected_source_commit, label + ": Git commit")
    require(
        env["sglang_wrapper_sha256"] == source_hash(args, WRAPPER),
        label + ": wrapper source",
    )
    require(
        env["torch"] == "2.14.1+cu130"
        and env["torch_cuda"] == "13.0"
        and env["flashinfer"] == "0.7.2",
        label + ": software",
    )
    require(
        env["backend"] == "cublas"
        and env["measurement_policy"] == "unspecified"
        and env["managed_store"] is False,
        label + ": operation/policy/store",
    )
    require(not env["LD_PRELOAD"], label + ": instrumented runtime timing")
    require(
        env["capability"] == [9, 0] and env["gpu"] == provenance["environment"]["gpu"],
        label + ": GPU",
    )
    require(
        env["physical_gpu_uuid"].removeprefix("GPU-")
        == provenance["environment"]["physical_gpu_uuid"],
        label + ": physical GPU",
    )
    require(
        all(
            provenance["environment"]["driver"] in env[key]
            for key in ("gpu_status_before", "gpu_status_after")
        ),
        label + ": driver",
    )
    hashes = expected_sources(manifest)
    require(
        env["flashinfer_tuner_sha256"] == hashes["flashinfer/autotuner/autotuner.py"],
        label + ": tuner source",
    )
    native = env["native_artifact"]
    require(native["source_sha256"] == hashes, label + ": complete descriptor source")
    require(
        native["cuda_arch_list"] == "9.0"
        and "bmm_fp8_run_with_descriptor" in native["required_native_exports"],
        label + ": native ABI/build",
    )
    require(
        len(native["library_sha256"]) == len(native["ninja_sha256"]) == 64,
        label + ": native provenance",
    )
    require(
        set(env["helper_sha256"])
        == {
            "flashinfer_autotune_descriptor_qualification.py",
            "flashinfer_autotune_eager_setup.py",
            "flashinfer_autotune_descriptor_selection.py",
        },
        label + ": helper coverage",
    )
    for name, value in env["helper_sha256"].items():
        require(
            source_hash(args, "test/manual/kernels/" + name) == value,
            label + ": helper source",
        )
    require(
        set(env["operand_sha256"]) == set(provenance["fixture_manifest"]["tensors"])
        and all(len(value) == 64 for value in env["operand_sha256"].values()),
        label + ": operand hashes",
    )
    protocol = data["timing_protocol"]
    require(
        protocol["warmup_ms"] == 500
        and protocol["min_sample_ms"] == 20
        and protocol["samples"] == 15
        and protocol["requested_iterations"] == 20,
        label + ": measurement protocol",
    )
    require(
        not data["forbidden_profile_attempts"],
        label + ": forbidden profiling attempted",
    )
    profiles = data["profiles"]
    require(
        data["profile_attempts"] == len(profiles),
        label + ": profiling attempt accounting",
    )
    tuning = arm != "default" and phase == "cold"
    require(
        (data["profile_attempts"] > 0) if tuning else (data["profile_attempts"] == 0),
        label + ": profiling participation",
    )
    require(
        positive(data["startup_ms"]) and positive(data["initial_public_call_ms"]),
        label + ": initialization timing",
    )
    natural = data["natural"]
    require(valid_tactic(natural["tactic"]), label + ": malformed natural tactic")
    tier = "miss" if arm == "default" else ("memory" if phase == "cold" else "file")
    require(
        natural["lookup_tier"] == tier and natural["cache_hit"] is (tier != "miss"),
        label + ": natural lookup source",
    )
    if arm == "default":
        require(natural["tactic"] == -1, label + ": untuned default")
    validate_acceptance(
        natural["acceptance"], natural["tactic"], label + ": natural acceptance"
    )
    for execution in EXECUTIONS:
        validate_timing(natural[execution], label + "/natural/" + execution)
    eligible = data["runtime_eligible_tactics"]
    require(
        eligible and all(descriptor(tactic) for tactic in eligible),
        label + ": runtime candidate schema",
    )
    require(
        data["oracle"] == [] and data["timing_protocol"]["oracle_order"] == [],
        label + ": unexpected forced candidate sweep",
    )
    _, n, k = data["shape"]
    expected_profile = [
        [1, runtime_bucket(m), k],
        [1, k, n],
        [1],
        [1],
        [1, -1, n],
        [-1],
    ]
    file_key = (
        "('fp8_gemm', 'CublasFp8GemmRunner', "
        + repr(tuple(tuple(shape) for shape in expected_profile))
        + ", (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.bfloat16))"
    )
    require(
        data["cache"]["runtime_file_key"] == file_key,
        label + ": runtime persistent key",
    )
    natural_path = locate(args.raw_root, data["cache"]["path"])
    natural_payload = load(natural_path)
    validate_generation(natural_payload, label + ": natural cache")
    require(
        sha256(natural_path)
        == data["cache"]["after_startup_sha256"]
        == data["cache"]["after_oracle_sha256"],
        label + ": persisted bytes changed",
    )
    require(
        natural_payload["_metadata"]["flashinfer_version"] == "0.7.2"
        and natural_payload["_metadata"]["cuda_version"] == "13.0"
        and natural_payload["_metadata"]["gpu"] == env["gpu"],
        label + ": persisted environment",
    )
    if arm == "default":
        require(
            file_key not in natural_payload, label + ": untuned cache contains a winner"
        )
    else:
        require(
            natural_payload[file_key][0] == "CublasFp8GemmRunner"
            and natural_payload[file_key][1] == natural["tactic"],
            label + ": persisted natural winner",
        )
    expected_stages = {
        "setup": (-1, "miss", False),
        "startup": (natural["tactic"], tier, False),
    }
    expected_stages.update(
        {
            name: (natural["tactic"], tier, name == "natural_capture")
            for name in (
                "natural_eager_changed_input",
                "natural_eager_restored_input",
                "natural_before",
                "natural_capture",
                "natural_after",
                "natural_after_oracle",
            )
        }
    )
    required_checks = {
        "natural_eager_changed_input",
        "natural_eager_restored_input",
        "initial_untuned",
        "natural_before",
        "natural_after",
        "natural_after_oracle",
        "natural_changed_input",
        "natural_restored_input",
        "natural_measured_eager",
        "natural_measured_cuda_graph",
    }
    selections = data["selections"]
    require(
        collections.Counter(row["stage"] for row in selections)
        == collections.Counter(expected_stages.keys()),
        label + ": selection coverage",
    )
    for row in selections:
        expected_tactic, expected_tier, capturing = expected_stages[row["stage"]]
        require(
            row["tactic"] == expected_tactic
            and row["lookup_tier"] == expected_tier
            and row["cache_hit"] is (expected_tier != "miss")
            and row["capturing"] is capturing,
            label + ": wrong selected tactic/source/capture",
        )
        if row["stage"] == "startup":
            require(
                row["on_forward_stream"] and not row["on_caller_stream"],
                label + ": startup stream",
            )
        elif not capturing:
            require(row["on_caller_stream"], label + ": caller stream not restored")
    expected_bindings = {
        stage for stage in expected_stages if stage not in ("setup", "startup")
    }
    bindings = data["binding_acceptance_checks"]
    require(
        collections.Counter(row["stage"] for row in bindings)
        == collections.Counter(expected_bindings),
        label + ": native guard coverage",
    )
    for row in bindings:
        tactic, _, _ = expected_stages[row["stage"]]
        validate_acceptance(row, tactic, label + ": native guard " + row["stage"])
    correctness = data["correctness"]
    require(
        correctness["passed"]
        and correctness["rtol"] == 0.02
        and correctness["atol"] == 0.125,
        label + ": numerical protocol",
    )
    require(
        collections.Counter(row["check"] for row in correctness["checks"])
        == collections.Counter(required_checks),
        label + ": eager/graph correctness coverage",
    )
    require(
        all(
            type(row["max_abs_error"]) in (float, int)
            and math.isfinite(row["max_abs_error"])
            and row["max_abs_error"] >= 0
            for row in correctness["checks"]
        ),
        label + ": numerical error metadata",
    )
    validate_settings(data, label)
    validate_numerical_details(data, label)
    runtime_profiles = []
    for row in profiles:
        require(
            row["stage"] == "startup"
            and row["runner"] == "CublasFp8GemmRunner"
            and valid_tactic(row["tactic"]),
            label + ": profiling provenance",
        )
        require(
            not row["use_cuda_graph"] and not row["use_cold_l2_cache"],
            label + ": measurement defaults changed",
        )
        require(
            row["autotuner_settings"] == data["autotuner_settings"]["after"],
            label + ": per-profile timing settings",
        )
        calculated = normalized_profile(row["input_shapes"])
        require(
            row["normalized_profile"] == calculated,
            label + ": profile key normalization",
        )
        require(
            row.get("profiler_ms") is None or positive(row["profiler_ms"]),
            label + ": invalid profile time",
        )
        if calculated == expected_profile:
            runtime_profiles.append(row)
    finite = [row for row in runtime_profiles if row.get("profiler_ms") is not None]
    summary = data["runtime_profile_summary"]
    require(
        summary["normalized_profile"] == expected_profile
        and summary["profiles"] == len(runtime_profiles)
        and summary["finite_profiles"] == len(finite),
        label + ": runtime profile summary",
    )
    minimum = min((row["profiler_ms"] for row in finite), default=None)
    require(
        summary["minimum_ms"] == minimum
        and summary["minimum_tactics"]
        == [row["tactic"] for row in finite if row["profiler_ms"] == minimum],
        label + ": reported profiler minimum",
    )
    if tuning:
        require(
            finite
            and natural["tactic"]
            == min(finite, key=lambda row: row["profiler_ms"])["tactic"],
            label + ": natural winner differs from first finite minimum",
        )
        expected_pool = {canonical(tactic) for tactic in eligible}
        if arm == "v2":
            expected_pool.add(canonical(-1))
            require(
                runtime_profiles[0]["tactic"] == -1
                and any(
                    row["tactic"] == -1 and row.get("profiler_ms") is not None
                    for row in runtime_profiles
                ),
                label + ": v2 default participation",
            )
        require(
            {canonical(row["tactic"]) for row in runtime_profiles} == expected_pool
            and len(runtime_profiles) == len(expected_pool),
            label + ": runtime profiling candidate coverage",
        )
    if tuning:
        buckets = {row["input_shapes"][0][-2] for row in profiles}
        require(
            buckets == {1, 2, 4, 8, 16, 32, 64, 128, 144},
            label + ": startup bucket coverage",
        )
        if arm == "v2":
            require(
                all(
                    any(
                        row["input_shapes"][0][-2] == bucket
                        and row["tactic"] == -1
                        and row.get("profiler_ms") is not None
                        for row in profiles
                    )
                    for bucket in buckets
                ),
                label + ": default missing from startup bucket",
            )
    if arm == "v1":
        require(
            all(row["tactic"] != -1 for row in profiles),
            label + ": v1 profiled default",
        )
    require(
        path.with_suffix(".exit-code").read_text().strip() == "0",
        label + ": Docker exit",
    )
    cid = path.with_suffix(".container-id").read_text().strip()
    require(re.fullmatch(r"[0-9a-f]{64}", cid), label + ": container identity")
    data["_container_id"], data["_path"] = cid, str(path)
    return data


def validate_pair(cold, warm, args):
    require(cold["phase"] == "cold" and warm["phase"] == "warm", "phase pair")
    require(cold["cache"]["before_sha256"] is None, "cold cache was populated")
    require(
        cold["cache"]["after_startup_sha256"]
        == cold["cache"]["after_oracle_sha256"]
        == warm["cache"]["before_sha256"]
        == warm["cache"]["after_startup_sha256"]
        == warm["cache"]["after_oracle_sha256"],
        "persisted bytes changed across fresh restart",
    )
    require(
        cold["cache"]["path"] == warm["cache"]["path"], "warm worker used another cache"
    )
    require(
        cold["natural"]["tactic"] == warm["natural"]["tactic"],
        "persisted natural winner lost",
    )
    require(
        cold["environment"]["operand_sha256"] == warm["environment"]["operand_sha256"],
        "cold/warm inputs differ",
    )
    require(
        not warm["profile_attempts"] and not warm["profiles"],
        "warm worker attempted profiling",
    )


def primary_channel(execution):
    return "wall" if execution == "eager" else "gpu_event"


def summarize(workers, trials):
    indexed = {(row["trial"], row["arm"], row["phase"]): row for row in workers}
    comparisons = []
    for phase in PHASES:
        for execution in EXECUTIONS:
            for channel, sample_key in (
                ("wall", "samples_ms"),
                ("gpu_event", "gpu_event_samples_ms"),
            ):
                for numerator, denominator in (
                    ("v2", "v1"),
                    ("v1", "default"),
                    ("v2", "default"),
                ):
                    rows = []
                    for trial in trials:
                        left = indexed[(trial, numerator, phase)]
                        right = indexed[(trial, denominator, phase)]
                        left_ms = statistics.median(
                            left["natural"][execution][sample_key]
                        )
                        right_ms = statistics.median(
                            right["natural"][execution][sample_key]
                        )
                        rows.append(
                            {
                                "trial": trial,
                                "numerator_ms": left_ms,
                                "denominator_ms": right_ms,
                                "ratio": left_ms / right_ms,
                            }
                        )
                    comparisons.append(
                        {
                            "phase": phase,
                            "execution": execution,
                            "channel": channel,
                            "primary": channel == primary_channel(execution),
                            "comparison": numerator + "_over_" + denominator,
                            "numerator_lookup_tier": indexed[
                                (trials[0], numerator, phase)
                            ]["natural"]["lookup_tier"],
                            "denominator_lookup_tier": indexed[
                                (trials[0], denominator, phase)
                            ]["natural"]["lookup_tier"],
                            **summarize_ratios(rows),
                            "scope": "Natural actual linear-layer path including quantization, dispatch and allocation. Default uses a genuine miss; tuned cold memory and warm file hits differ from that baseline. Graph lookup occurs during capture, not replay.",
                        }
                    )
    profile_rows = []
    for row in workers:
        if row["phase"] != "cold" or row["arm"] == "default":
            continue
        shape = row["runtime_profile_summary"]["normalized_profile"]
        runtime = [r for r in row["profiles"] if r["normalized_profile"] == shape]
        finite = sorted(
            (r for r in runtime if r.get("profiler_ms") is not None),
            key=lambda r: r["profiler_ms"],
        )
        margin = (
            None
            if len(finite) < 2
            else finite[1]["profiler_ms"] / finite[0]["profiler_ms"] - 1
        )
        profile_rows.append(
            {
                "trial": row["trial"],
                "arm": row["arm"],
                "selected_tactic": row["natural"]["tactic"],
                "finite_candidates": len(finite),
                "runtime_profiles": len(runtime),
                "runner_up_fractional_margin": margin,
                "ranked_profiles": [
                    {"tactic": r["tactic"], "profiler_ms": r["profiler_ms"]}
                    for r in finite
                ],
                "default_profile": next(
                    (
                        {"profiler_ms": r.get("profiler_ms"), "error": r.get("error")}
                        for r in runtime
                        if r["tactic"] == -1
                    ),
                    None,
                ),
                "actual_input_m_values": sorted(
                    {r["input_shapes"][0][-2] for r in runtime}
                ),
                "normalized_profile": shape,
                "autotuner_settings": row["autotuner_settings"]["after"],
                "interpretation": "Profiler rankings are startup measurements, not a separate serving-candidate oracle or physical-kernel identity proof.",
            }
        )
    frequencies = []
    for arm in ARMS:
        counts = collections.Counter(
            canonical(indexed[(trial, arm, "cold")]["natural"]["tactic"])
            for trial in trials
        )
        frequencies.append(
            {
                "arm": arm,
                "fresh_trials": len(trials),
                "winners": [
                    {"tactic": json.loads(tactic), "count": count}
                    for tactic, count in counts.items()
                ],
            }
        )
    startup = []
    for phase in PHASES:
        for numerator, denominator in (
            ("v2", "v1"),
            ("v1", "default"),
            ("v2", "default"),
        ):
            rows = [
                {
                    "trial": trial,
                    "ratio": indexed[(trial, numerator, phase)]["startup_ms"]
                    / indexed[(trial, denominator, phase)]["startup_ms"],
                }
                for trial in trials
            ]
            startup.append(
                {
                    "phase": phase,
                    "comparison": numerator + "_over_" + denominator,
                    **summarize_ratios(rows),
                    "scope": "Secondary operation-wrapper startup diagnostic; excludes model acquisition/capture, native construction and initial public-call/quantizer preparation. No cold-start or full-server startup performance claim.",
                }
            )
    return {
        "natural_latency_comparisons": comparisons,
        "cold_winner_frequencies": frequencies,
        "startup_profiler_rankings": profile_rows,
        "startup_diagnostics": startup,
        "bf16_approximation_diagnostics": [
            {
                "trial": row["trial"],
                "arm": row["arm"],
                "phase": row["phase"],
                "diagnostics": [
                    {"check": check["check"], "bf16_reference": check["bf16_reference"]}
                    for check in row["correctness"]["checks"]
                ],
            }
            for row in workers
        ],
    }


def analyze(args):
    trials = args.trials
    require(
        trials and sorted(set(trials)) == trials and all(t in TRIALS for t in trials),
        "trials must be unique, ordered and within0..5",
    )
    manifest = load(args.source_manifest)
    require(
        manifest["upstream_pr"]["number"] == 6148
        and manifest["upstream_pr"]["head_commit"]
        == "14c5c67a1eb9ad3a608b8d3da3bf6d8924037c5c"
        and manifest["baseline"]["tag"] == "v0.7.2"
        and manifest["clean_git_apply_check"]
        and not manifest["manual_resolution"],
        "unqualified descriptor source overlay",
    )
    provenance = validate_outer(args, trials)
    workers = []
    for trial in trials:
        for arm in ARMS:
            pair = []
            for phase in PHASES:
                path = args.raw_root / f"trial{trial}-seed19-{arm}-{phase}.json"
                row = validate_worker(path, args, manifest, provenance)
                require(
                    (row["trial"], row["arm"], row["phase"]) == (trial, arm, phase),
                    "filename/workload mismatch",
                )
                workers.append(row)
                pair.append(row)
            validate_pair(*pair, args)
    require(
        len({row["_container_id"] for row in workers}) == len(workers),
        "container reused",
    )
    require(
        len({canonical(row["environment"]["operand_sha256"]) for row in workers}) == 1,
        "fixed layer operands changed",
    )
    require(
        len({canonical(row["runtime_eligible_tactics"]) for row in workers}) == 1,
        "runtime descriptor enumeration changed",
    )
    for key in ("library_sha256", "ninja_sha256"):
        require(
            len({row["environment"]["native_artifact"][key] for row in workers}) == 1,
            "native artifact changed: " + key,
        )
    require(
        len({canonical(row["autotuner_settings"]["after"]) for row in workers}) == 1,
        "profiler settings changed across arms/repetitions",
    )
    return {
        "status": "success",
        "schema_version": 1,
        "source_commit": args.expected_source_commit,
        "analyzer_sha256": sha256(Path(__file__)),
        "trials": trials,
        "input_seed": 19,
        "independent_trial_triplets": len(trials),
        "source_manifest_sha256": sha256(args.source_manifest),
        "fixture_manifest_sha256": sha256(args.fixture_manifest),
        "outer_provenance": provenance,
        "validated_workers": [
            {
                "file": row["_path"],
                "sha256": sha256(Path(row["_path"])),
                "container_id": row["_container_id"],
                "trial": row["trial"],
                "arm": row["arm"],
                "phase": row["phase"],
                "profile_attempts": row["profile_attempts"],
                "numerical_checks": len(row["correctness"]["checks"]),
            }
            for row in workers
        ],
        "analysis": summarize(workers, trials),
        "limitations": [
            "One H100, one captured model layer/144-token input and six fresh trial triplets. Fifteen samples within a worker are dependent subsamples, not independent trials.",
            "Exploratory percentile95% bootstrap resamples whole paired trial units, is unadjusted for multiple contrasts and is absent for a one-trial checkpoint.",
            "The actual SGLang static FP8 linear path includes quantization, custom-op dispatch, allocation and GEMM. Natural default miss versus tuned memory/file lookup is part of the compared configuration; ratios do not isolate GPU-kernel speed.",
            "Model activations come from a pinned dense BF16 checkpoint; weights/scales are calibrated for this single fixture. Quantized-input FP32 is the correctness gate; dense BF16 approximation is diagnostic, not full-model accuracy.",
            "Graph primary latency is a GPU-event replay span including possible host submission gaps; eager primary is wall time. Capture lookup does not repeat during graph replay.",
            "Recorded startup profiler minima do not provide an independent runtime-candidate oracle. No forced shortlist or physical-kernel identity trace was collected in this batch.",
            "Experimental exact descriptor overlay over0.7.2, unspecified policy and retained v1 persistence; no managed-store adoption, distributed/backend generalization or serving-throughput claim.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument(
        "--trials", type=int, nargs="+", choices=TRIALS, default=list(TRIALS)
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = analyze(args)
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["status"])
    if result["status"] != "success":
        print(result["validation_errors"][0])
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
