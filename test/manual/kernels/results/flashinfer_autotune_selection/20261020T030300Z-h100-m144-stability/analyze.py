#!/usr/bin/env python3
"""Validate fixed-input M144 repetitions and balanced interleaved file-cache timings.

Six fresh trial pairs are the independent units. Candidate/default contrasts
use paired blocks; exploratory uncertainty resamples trial pairs, never blocks.
"""

from __future__ import annotations

import argparse
import collections
import functools
import hashlib
import itertools
import json
import math
import random
import re
import statistics
import subprocess
from pathlib import Path

ARMS = ("v1", "v2")
PHASES = ("cold", "warm")
TRIALS = tuple(range(6))
EXECUTIONS = ("eager", "cuda_graph")
WORKER = "test/manual/kernels/flashinfer_autotune_m144_stability.py"
WRAPPER = "python/sglang/srt/model_executor/runner/flashinfer_autotune.py"
BATCH = "test/manual/kernels/results/flashinfer_autotune_selection/20261020T030300Z-h100-m144-stability"


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


def validate_outer(args, trials):
    plan_bytes = git_blob(args, BATCH + "/plan.json")
    plan = json.loads(plan_bytes)
    outer = load(args.raw_root / "manifest.json")
    require(
        outer["plan_sha256"] == hashlib.sha256(plan_bytes).hexdigest(),
        "launcher plan hash",
    )
    require(
        outer["source_manifest_sha256"] == sha256(args.source_manifest),
        "launcher source manifest",
    )
    require(
        outer["source_manifest_sha256"]
        == plan["flashinfer_source"]["source_manifest_sha256"],
        "source manifest differs from preregistration",
    )
    require(
        (args.raw_root / "execution-commit.txt").read_text().strip()
        == args.expected_source_commit,
        "launcher source commit",
    )
    for path, value in outer["source_sha256"].items():
        require(source_hash(args, path) == value, "launcher source differs: " + path)
    require(
        {WORKER, WRAPPER, BATCH + "/analyze.py", BATCH + "/candidates.json"}
        <= set(outer["source_sha256"]),
        "launcher source coverage",
    )
    image = load(args.raw_root / "image.json")
    require(
        image["Architecture"] == "amd64" and image["Os"] == "linux",
        "container architecture",
    )
    expected_image = plan["container_image"].split("@")[-1]
    require(
        any(value.split("@")[-1] == expected_image for value in image["RepoDigests"]),
        "container digest",
    )
    require(
        (args.raw_root / "prepare.exit-code").read_text().strip() == "0",
        "package preparation failed",
    )
    require(
        not (args.raw_root / "gpu-processes-before.csv").read_text().strip(),
        "GPU not idle at allocation start",
    )
    for trial in trials:
        require(
            (args.raw_root / f"trial{trial}.launcher.exit-code").read_text().strip()
            == "0",
            f"trial {trial} incomplete",
        )
    environment = plan["environment"]
    require(outer["gpu"]["name"] == environment["gpu"], "GPU model differs from plan")
    require(
        outer["gpu"]["uuid"].removeprefix("GPU-") == environment["physical_gpu_uuid"],
        "GPU UUID differs from plan",
    )
    require(
        outer["gpu"]["driver_version"] == environment["driver"],
        "GPU driver differs from plan",
    )
    return {
        "plan_sha256": outer["plan_sha256"],
        "launcher_manifest_sha256": sha256(args.raw_root / "manifest.json"),
        "image": image,
        "plan": plan,
        "gpu": outer["gpu"],
        "environment": environment,
    }


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


def validate_batch(row, iterations, label):
    require(
        type(row["iterations"]) is int and row["iterations"] == iterations,
        label + ": common iteration count",
    )
    for batch_key, per_key in (
        ("wall_batch_ms", "wall_per_call_ms"),
        ("gpu_event_batch_ms", "gpu_event_per_call_ms"),
    ):
        require(
            positive(row[batch_key]) and positive(row[per_key]),
            label + ": positive raw timing",
        )
        require(
            math.isclose(row[batch_key], row[per_key] * iterations, rel_tol=1e-12),
            label + ": timing normalization",
        )


def validate_calibration(row, label):
    iterations = row["effective_iterations"]
    require(
        type(iterations) is int
        and 20 <= iterations <= 65536
        and row["requested_iterations"] == 20,
        label + ": calibration iterations",
    )
    require(
        positive(row["warmup_actual_ms"])
        and row["warmup_actual_ms"] >= 500
        and type(row["warmup_calls"]) is int
        and row["warmup_calls"] >= 20
        and row["warmup_calls"] % 20 == 0,
        label + ": candidate warmup",
    )
    batches = row["calibration_batches"]
    require(
        batches
        and batches[0]["iterations"] == 20
        and batches[-1]["iterations"] == iterations
        and batches[-1]["wall_batch_ms"] >= 20,
        label + ": calibration bounds",
    )
    for value in batches:
        validate_batch(value, value["iterations"], label + ": calibration batch")
    for previous, current in zip(batches, batches[1:]):
        require(
            previous["wall_batch_ms"] < 20
            and current["iterations"] == min(previous["iterations"] * 2, 65536),
            label + ": unnecessary/non-doubling calibration",
        )


def expected_pool(manifest, natural_tactic):
    rows = []
    for original in manifest["candidates"]:
        row = {key: original[key] for key in ("id", "tactic", "source")}
        row["roles"] = [row["id"]]
        if row["tactic"] == natural_tactic:
            row["roles"].append("actual_selected")
        rows.append(row)
    if not any(row["tactic"] == natural_tactic for row in rows):
        rows.append(
            {
                "id": "actual_selected",
                "tactic": natural_tactic,
                "roles": ["actual_selected"],
                "source": "Own actual fresh cold winner consumed by this warm worker; added according to the preregistered union rule.",
            }
        )
    return rows


def validate_stability(
    data,
    args,
    provenance,
    natural_payload,
    file_key,
    expected_stages,
    required_checks,
    label,
):
    stability, protocol = data["stability"], data["timing_protocol"]
    plan = provenance["plan"]
    require(
        stability["trial"] == data["trial"]
        and stability["input_seed"] == 19
        and stability["order_seed"] == data["order_seed"]
        and stability["blocks_per_execution"] == 24,
        label + ": stability identity/counts",
    )
    require(
        stability["candidate_manifest_sha256"]
        == plan["protocol"]["candidate_manifest_sha256"],
        label + ": candidate manifest hash",
    )
    require(
        protocol["blocks_per_execution"] == 24
        and protocol["rewarm_ms"] == 20
        and protocol["input_seed"] == 19
        and protocol["order_seed"] == data["order_seed"],
        label + ": stability timing protocol",
    )
    warm = data["phase"] == "warm"
    require(stability["enabled"] is warm, label + ": stability phase")
    if not warm:
        require(
            stability["candidates"] == stability["blocks"] == []
            and stability["common_iterations"] == stability["schedules"] == {}
            and protocol["oracle_order"] == [],
            label + ": cold shortlist timing",
        )
        return
    manifest = provenance["candidate_manifest"]
    pool = expected_pool(manifest, data["natural"]["tactic"])
    candidates = stability["candidates"]
    require(
        len(candidates) == len(pool) and len(pool) in (3, 4),
        label + ": full preregistered pool required",
    )
    require(
        protocol["oracle_order"] == [row["tactic"] for row in pool],
        label + ": shortlist order",
    )
    by_id = {}
    for row, expected in zip(candidates, pool):
        require(
            all(row[key] == value for key, value in expected.items()),
            label + ": shortlist identity/roles/source",
        )
        tactic, cid = row["tactic"], row["id"]
        require(
            tactic == -1 or tactic in data["runtime_eligible_tactics"],
            label + ": shortlist candidate not eligible",
        )
        require(
            row["accepted"] is True
            and row["exclusion_reason"] is None
            and row["lookup_tier"] == "file",
            label + ": shortlist rejection/tier",
        )
        validate_acceptance(row["acceptance"], tactic, label + ": prepared candidate")
        prefix = "stability_prepare_" + cid
        require(
            row["acceptance"]["stage"] == prefix + "_before",
            label + ": prepared acceptance stage",
        )
        control = locate(args.raw_root, row["control_file"])
        require(sha256(control) == row["control_sha256"], label + ": control bytes")
        payload = load(control)
        validate_generation(payload, label + ": candidate generation")
        require(
            payload[file_key][0] == "CublasFp8GemmRunner"
            and payload[file_key][1] == tactic,
            label + ": persisted control tactic",
        )
        require(
            {k: v for k, v in payload.items() if k not in (file_key, "_generation")}
            == {
                k: v
                for k, v in natural_payload.items()
                if k not in (file_key, "_generation")
            },
            label + ": control altered unrelated entries/policies",
        )
        for suffix in ("_before", "_capture", "_after"):
            expected_stages[prefix + suffix] = (tactic, "file", suffix == "_capture")
        required_checks.update(
            prefix + suffix
            for suffix in (
                "_before",
                "_after",
                "_changed_input",
                "_restored_input",
                "_calibrate_eager",
                "_calibrate_cuda_graph",
            )
        )
        for execution in EXECUTIONS:
            validate_calibration(
                row["calibration"][execution], label + "/" + cid + "/" + execution
            )
        by_id[cid] = row
    ids = list(by_id)
    require(
        set(stability["common_iterations"]) == set(EXECUTIONS)
        and set(stability["schedules"]) == set(EXECUTIONS),
        label + ": mode coverage",
    )
    for execution, offset in zip(EXECUTIONS, (0, 1)):
        common = max(
            row["calibration"][execution]["effective_iterations"] for row in candidates
        )
        require(
            stability["common_iterations"][execution] == common,
            label + ": common iterations not candidate maximum",
        )
        permutations = [list(order) for order in itertools.permutations(ids)]
        expected = [
            list(order)
            for _ in range(24 // len(permutations))
            for order in permutations
        ]
        random.Random(data["order_seed"] + offset).shuffle(expected)
        require(
            stability["schedules"][execution] == expected,
            label + ": unbalanced or wrong-seed schedule",
        )
    require(
        stability["schedule_seed_rule"]
        == "eager=order_seed; cuda_graph=order_seed+1; shuffle the full balanced permutation multiset"
        and stability["mode_order_rule"]
        == "eager then cuda_graph on even blocks, reversed on odd blocks",
        label + ": schedule rules",
    )
    expected_groups = [
        (i, mode)
        for i in range(24)
        for mode in (EXECUTIONS if i % 2 == 0 else EXECUTIONS[::-1])
    ]
    require(
        [(g["index"], g["execution"]) for g in stability["blocks"]] == expected_groups,
        label + ": block/mode order or count",
    )
    for group in stability["blocks"]:
        index, execution = group["index"], group["execution"]
        order = stability["schedules"][execution][index]
        require(
            group["order"] == order
            and [m["candidate_id"] for m in group["measurements"]] == order,
            label + ": block candidate order/coverage",
        )
        iterations = stability["common_iterations"][execution]
        for measurement in group["measurements"]:
            cid = measurement["candidate_id"]
            tactic = by_id[cid]["tactic"]
            prefix = f"stability_{execution}_b{index:02d}_{cid}"
            validate_batch(measurement, iterations, label + "/" + prefix)
            require(
                measurement["rewarm_actual_ms"] >= 20
                and type(measurement["rewarm_calls"]) is int
                and measurement["rewarm_calls"] >= iterations
                and measurement["rewarm_calls"] % iterations == 0,
                label + ": missing candidate rewarm",
            )
            for suffix in ("before", "after"):
                stage = prefix + "_" + suffix
                expected_stages[stage] = (tactic, "file", False)
                required_checks.add(stage)
                acceptance = measurement["acceptance_" + suffix]
                require(
                    acceptance["stage"] == stage, label + ": batch acceptance stage"
                )
                validate_acceptance(
                    acceptance, tactic, label + ": batch native acceptance"
                )
            required_checks.add(prefix + "_timed")


def validate_worker(path, args, manifest, provenance):
    data = load(path)
    label = path.name
    require(data["status"] == "success" and data["schema_version"] == 1, label)
    require(data["implementation"] == "candidate", label + ": source arm")
    arm, phase, m = data["arm"], data["phase"], data["shape"][0]
    require(arm in ARMS and phase in PHASES and m == 144, label)
    require(
        data["seed"] == 19
        and data["shape"] == [144, 1024, 1024]
        and data["trial"] in TRIALS
        and data["order_seed"] == 3000 + data["trial"],
        label + ": workload",
    )
    require(
        data["worker_sha256"] == source_hash(args, WORKER), label + ": worker source"
    )
    require(
        data["manifest_sha256"] == sha256(args.source_manifest),
        label + ": source manifest",
    )
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
        set(env["operand_sha256"]) == {"a", "b", "scale_a", "scale_b"}
        and env["operand_sha256"]
        == provenance["plan"]["workload"]["reference_operand_sha256"],
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
    require(data["oracle"] == [], label + ": legacy oracle sweep enabled")
    expected_profile = [
        [1, runtime_bucket(m), 1024],
        [1, 1024, 1024],
        [1],
        [1],
        [1, -1, 1024],
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
                "natural_before",
                "natural_capture",
                "natural_after",
                "natural_after_oracle",
            )
        }
    )
    required_checks = {
        "initial_untuned",
        "natural_before",
        "natural_after",
        "natural_after_oracle",
        "natural_changed_input",
        "natural_restored_input",
        "natural_measured_eager",
        "natural_measured_cuda_graph",
    }
    validate_stability(
        data,
        args,
        provenance,
        natural_payload,
        file_key,
        expected_stages,
        required_checks,
        label,
    )
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
            label + ": profile settings changed",
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
    if phase == "cold":
        require(
            len(profiles) == (72 if arm == "v1" else 81),
            label + ": complete bucket profiling",
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


def channel_key(channel):
    return "wall_per_call_ms" if channel == "wall" else "gpu_event_per_call_ms"


def primary_channel(execution):
    return "wall" if execution == "eager" else "gpu_event"


def worker_contrasts(worker):
    candidates = worker["stability"]["candidates"]
    selected = next(
        row["id"] for row in candidates if "actual_selected" in row["roles"]
    )
    results = []
    for execution in EXECUTIONS:
        groups = [
            group
            for group in worker["stability"]["blocks"]
            if group["execution"] == execution
        ]
        for channel in ("wall", "gpu_event"):
            key = channel_key(channel)
            maps = [
                {m["candidate_id"]: m[key] for m in group["measurements"]}
                for group in groups
            ]
            contrasts = []
            for candidate in candidates:
                cid = candidate["id"]
                ratios = [row[cid] / row["default"] for row in maps]
                contrasts.append(
                    {
                        "candidate_id": cid,
                        "roles": candidate["roles"],
                        "tactic": candidate["tactic"],
                        "same_block_ratios": ratios,
                        "median_paired_ratio": statistics.median(ratios),
                        "candidate_median_ms": statistics.median(
                            row[cid] for row in maps
                        ),
                        "default_median_ms": statistics.median(
                            row["default"] for row in maps
                        ),
                    }
                )
            ranked = sorted(contrasts, key=lambda row: row["median_paired_ratio"])
            best = ranked[0]
            own = next(row for row in contrasts if row["candidate_id"] == selected)
            results.append(
                {
                    "trial": worker["trial"],
                    "arm": worker["arm"],
                    "execution": execution,
                    "channel": channel,
                    "primary": channel == primary_channel(execution),
                    "lookup_tier": "file",
                    "blocks": 24,
                    "candidate_count": len(candidates),
                    "contrasts": contrasts,
                    "ranked_candidate_ids": [r["candidate_id"] for r in ranked],
                    "selected_candidate_id": selected,
                    "best_shortlist_candidate_id": best["candidate_id"],
                    "selected_rank": 1
                    + sum(
                        r["median_paired_ratio"] < own["median_paired_ratio"]
                        for r in contrasts
                    ),
                    "selected_to_best_score_ratio": own["median_paired_ratio"]
                    / best["median_paired_ratio"],
                    "same_block_selected_to_best_median_ratio": statistics.median(
                        row[selected] / row[best["candidate_id"]] for row in maps
                    ),
                    "ranking_scope": "median paired candidate/default ratios within this accepted 3/4-candidate shortlist; selected/best score ratio is not a global optimality bound",
                }
            )
    return results


def summarize(workers, trials):
    indexed = {(row["trial"], row["arm"], row["phase"]): row for row in workers}
    natural = []
    for phase in PHASES:
        for execution in EXECUTIONS:
            for channel, sample_key in (
                ("wall", "samples_ms"),
                ("gpu_event", "gpu_event_samples_ms"),
            ):
                rows = []
                for trial in trials:
                    medians = {
                        arm: statistics.median(
                            indexed[(trial, arm, phase)]["natural"][execution][
                                sample_key
                            ]
                        )
                        for arm in ARMS
                    }
                    rows.append(
                        {
                            "trial": trial,
                            "v1_ms": medians["v1"],
                            "v2_ms": medians["v2"],
                            "ratio": medians["v2"] / medians["v1"],
                        }
                    )
                natural.append(
                    {
                        "phase": phase,
                        "execution": execution,
                        "channel": channel,
                        "primary": channel == primary_channel(execution),
                        "comparison": "v2_over_v1",
                        "lookup_tier": "memory" if phase == "cold" else "file",
                        **summarize_ratios(rows),
                    }
                )
    within = [
        item
        for worker in workers
        if worker["phase"] == "warm"
        for item in worker_contrasts(worker)
    ]
    by_worker = {
        (row["trial"], row["arm"], row["execution"], row["channel"]): row
        for row in within
    }
    candidate_summaries, normalized = [], []
    for execution in EXECUTIONS:
        for channel in ("wall", "gpu_event"):
            for arm in ARMS:
                for role in ("old_selected", "prior_graph_leader", "actual_selected"):
                    rows = []
                    for trial in trials:
                        record = by_worker[(trial, arm, execution, channel)]
                        contrast = next(
                            row for row in record["contrasts"] if role in row["roles"]
                        )
                        rows.append(
                            {
                                "trial": trial,
                                "ratio": contrast["median_paired_ratio"],
                                "candidate_id": contrast["candidate_id"],
                                "tactic": contrast["tactic"],
                            }
                        )
                    candidate_summaries.append(
                        {
                            "arm": arm,
                            "execution": execution,
                            "channel": channel,
                            "primary": channel == primary_channel(execution),
                            "role": role,
                            "comparison": "same_block_candidate_over_cached_default",
                            **summarize_ratios(rows),
                        }
                    )
            rows = []
            for trial in trials:
                values = {}
                for arm in ARMS:
                    record = by_worker[(trial, arm, execution, channel)]
                    own = next(
                        row
                        for row in record["contrasts"]
                        if "actual_selected" in row["roles"]
                    )
                    values[arm] = own["median_paired_ratio"]
                rows.append(
                    {
                        "trial": trial,
                        "v1_selected_over_default": values["v1"],
                        "v2_selected_over_default": values["v2"],
                        "ratio": values["v2"] / values["v1"],
                    }
                )
            normalized.append(
                {
                    "execution": execution,
                    "channel": channel,
                    "primary": channel == primary_channel(execution),
                    "comparison": "v2_selected_over_default_divided_by_v1_selected_over_default",
                    **summarize_ratios(rows),
                }
            )
    profiles = []
    for worker in workers:
        if worker["phase"] != "cold":
            continue
        expected = worker["runtime_profile_summary"]["normalized_profile"]
        rows = [
            row for row in worker["profiles"] if row["normalized_profile"] == expected
        ]
        finite = sorted(
            (row for row in rows if row.get("profiler_ms") is not None),
            key=lambda row: row["profiler_ms"],
        )
        margin = (
            None
            if len(finite) < 2
            else finite[1]["profiler_ms"] / finite[0]["profiler_ms"] - 1
        )
        profiles.append(
            {
                "trial": worker["trial"],
                "arm": worker["arm"],
                "selected_tactic": worker["natural"]["tactic"],
                "finite_candidates": len(finite),
                "profile_order": [row["tactic"] for row in rows],
                "ranked_profiles": [
                    {"tactic": r["tactic"], "profiler_ms": r["profiler_ms"]}
                    for r in finite
                ],
                "runner_up_fractional_margin": margin,
                "default_profile": next(
                    (
                        {"profiler_ms": r.get("profiler_ms"), "error": r.get("error")}
                        for r in rows
                        if r["tactic"] == -1
                    ),
                    None,
                ),
                "actual_input_m_values": sorted(
                    {r["input_shapes"][0][-2] for r in rows}
                ),
                "normalized_m": 256,
                "autotuner_settings": worker["autotuner_settings"]["after"],
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
    startup = [
        {
            "phase": phase,
            "comparison": "v2_over_v1_startup",
            "scope": "wrapper startup excludes explicit binary build/load and initial public API call; secondary diagnostic",
            **summarize_ratios(
                [
                    {
                        "trial": trial,
                        "ratio": indexed[(trial, "v2", phase)]["startup_ms"]
                        / indexed[(trial, "v1", phase)]["startup_ms"],
                    }
                    for trial in trials
                ]
            ),
        }
        for phase in PHASES
    ]
    return {
        "natural_latency_comparisons": natural,
        "within_worker_paired_contrasts": within,
        "candidate_over_default_trial_summaries": candidate_summaries,
        "default_normalized_cross_arm_comparisons": normalized,
        "startup_profiler_rankings": profiles,
        "cold_winner_frequencies": frequencies,
        "startup_diagnostics": startup,
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
    candidates_bytes = git_blob(args, BATCH + "/candidates.json")
    require(
        hashlib.sha256(candidates_bytes).hexdigest()
        == provenance["plan"]["protocol"]["candidate_manifest_sha256"],
        "fixed candidate manifest differs from plan",
    )
    provenance["candidate_manifest"] = json.loads(candidates_bytes)
    require(
        provenance["candidate_manifest"]["reference_operand_sha256"]
        == provenance["plan"]["workload"]["reference_operand_sha256"],
        "fixed reference operands differ",
    )
    workers = []
    for trial in trials:
        for arm in ARMS:
            pair = []
            for phase in PHASES:
                path = args.raw_root / f"m144-trial{trial}-seed19-{arm}-{phase}.json"
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
        "worker container reused",
    )
    require(
        len({canonical(row["environment"]["operand_sha256"]) for row in workers}) == 1,
        "fixed input bytes changed across fresh workers",
    )
    require(
        len({canonical(row["runtime_eligible_tactics"]) for row in workers}) == 1,
        "runtime candidate enumeration changed",
    )
    for key in ("library_sha256", "ninja_sha256"):
        require(
            len({row["environment"]["native_artifact"][key] for row in workers}) == 1,
            "native build changed: " + key,
        )
    require(
        len({canonical(row["autotuner_settings"]["after"]) for row in workers}) == 1,
        "profiler timing objective changed across repetitions",
    )
    return {
        "status": "success",
        "schema_version": 1,
        "source_commit": args.expected_source_commit,
        "source_manifest_sha256": sha256(args.source_manifest),
        "analyzer_sha256": sha256(Path(__file__)),
        "trials": trials,
        "input_seed": 19,
        "independent_trial_pairs": len(trials),
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
            "Six fresh paired tuning/serving repetitions on one H100 and one identical input seed; within-worker blocks are dependent subsamples, not independent trials.",
            "Bootstrap intervals resample whole trial pairs, are exploratory and unadjusted for multiple contrasts; a one-trial checkpoint has no interval.",
            "Natural samples are sequential windows; balanced interleaved candidate/default contrasts reduce order effects but are separate measurements from natural timing.",
            "Eager primary latency is wall time; graph primary is GPU-event replay span including host submission gaps. Both channels remain available.",
            "Ranking and regret scores cover only the three preregistered candidates plus each own selected winner; union pools depend on that selection and do not establish a global optimum.",
            "The historical graph leader was chosen from noisy previous measurements; a measured shortlist rank is exploratory, not a universal tactic-quality claim.",
            "Startup profiler uses actual operation eager/hot-L2 defaults and its own delays/repeats; synthetic input generation and runtime measurement conditions differ without a policy override.",
            "Exact descriptor overlay over0.7.2 is experimental; no physical-kernel causality, model serving-throughput, broader backend, distributed, or managed-store claim.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
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
