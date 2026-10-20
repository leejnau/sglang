#!/usr/bin/env python3
"""Validate natural startup selection on the pinned descriptor implementation.

The independent unit is a fresh-worker trial. Selected-tactic regret compares
two candidates measured within the same oracle sweep and lookup tier. Natural
latency versus the oracle is a separate comparison across measurement windows.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import re
import statistics
import subprocess
from pathlib import Path

ARMS = ("default", "v1", "v2")
PHASES = ("cold", "warm")
SEEDS = (17, 18, 19)
EXECUTIONS = ("eager", "cuda_graph")
WORKER = "test/manual/kernels/flashinfer_autotune_descriptor_selection.py"
WRAPPER = "python/sglang/srt/model_executor/runner/flashinfer_autotune.py"
BATCH = "test/manual/kernels/results/flashinfer_autotune_selection/20261019T235600Z-h100-descriptor-selection"


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

    return json.loads(path.read_text(), parse_constant=invalid)


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
    return subprocess.check_output(
        ["git", "show", args.expected_source_commit + ":" + path],
        cwd=args.source_root,
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


def validate_outer(args, ms):
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
        WORKER in outer["source_sha256"] and WRAPPER in outer["source_sha256"],
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
    for m in ms:
        require(
            (args.raw_root / f"m{m}.launcher.exit-code").read_text().strip() == "0",
            f"M{m} incomplete",
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


def validate_worker(path, args, manifest, provenance):
    data = load(path)
    label = path.name
    require(data["status"] == "success" and data["schema_version"] == 1, label)
    require(data["implementation"] == "candidate", label + ": source arm")
    arm, phase, m = data["arm"], data["phase"], data["shape"][0]
    require(arm in ARMS and phase in PHASES and m in (128, 144), label)
    require(
        data["seed"] in SEEDS and data["shape"] == [m, 1024, 1024], label + ": workload"
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
    candidate_pool = []
    for tactic in [-1, *eligible]:
        if tactic not in candidate_pool:
            candidate_pool.append(tactic)
    random.Random(data["seed"]).shuffle(candidate_pool)
    require(protocol["oracle_order"] == candidate_pool, label + ": oracle order")
    require(
        [candidate["tactic"] for candidate in data["oracle"]] == candidate_pool,
        label + ": complete oracle candidate pool",
    )
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
    oracle_tier = "memory" if phase == "cold" else "file"
    for index, candidate in enumerate(data["oracle"]):
        tactic = candidate["tactic"]
        accepted = candidate["accepted"]
        require(
            type(accepted) is bool and candidate["lookup_tier"] == oracle_tier,
            label + ": oracle lookup source",
        )
        validate_acceptance(
            candidate["acceptance"], tactic, label + f": candidate {index}", accepted
        )
        prefix = f"oracle_{index:02d}"
        expected_stages[prefix + "_before"] = (tactic, oracle_tier, False)
        required_checks.add(prefix + "_before")
        if phase == "cold":
            require(
                candidate["control_file"] is None
                and candidate["control_sha256"] is None,
                label + ": cold oracle unexpectedly persisted",
            )
        else:
            require(
                candidate["control_file"] and candidate["control_sha256"],
                label + ": missing warm file control",
            )
            control = locate(args.raw_root, candidate["control_file"])
            require(
                sha256(control) == candidate["control_sha256"],
                label + ": oracle file bytes",
            )
            payload = load(control)
            validate_generation(payload, label + ": oracle cache")
            require(
                payload[file_key][0] == "CublasFp8GemmRunner"
                and payload[file_key][1] == tactic,
                label + ": wrong persisted oracle tactic",
            )
            require(
                {
                    key: value
                    for key, value in payload.items()
                    if key not in (file_key, "_generation")
                }
                == {
                    key: value
                    for key, value in natural_payload.items()
                    if key not in (file_key, "_generation")
                },
                label + ": oracle changed unrelated persisted entries",
            )
        if accepted:
            require(
                candidate["exclusion_reason"] is None,
                label + ": accepted candidate has exclusion",
            )
            for execution in EXECUTIONS:
                validate_timing(
                    candidate[execution], label + f"/oracle/{index}/" + execution
                )
            expected_stages[prefix + "_capture"] = (tactic, oracle_tier, True)
            expected_stages[prefix + "_after"] = (tactic, oracle_tier, False)
            required_checks |= {
                prefix + suffix
                for suffix in (
                    "_after",
                    "_changed_input",
                    "_restored_input",
                    "_measured_eager",
                    "_measured_cuda_graph",
                )
            }
        else:
            require(
                tactic != -1
                and candidate["exclusion_reason"]
                and "native descriptor rejected" in candidate["exclusion_reason"],
                label + ": unclassified candidate exclusion",
            )
            require(
                all(execution not in candidate for execution in EXECUTIONS),
                label + ": excluded candidate has accepted timings",
            )
    accepted_tactics = [
        candidate["tactic"] for candidate in data["oracle"] if candidate["accepted"]
    ]
    require(-1 in accepted_tactics, label + ": default missing from accepted oracle")
    if arm != "default":
        require(
            natural["tactic"] in accepted_tactics,
            label + ": natural winner excluded from oracle",
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
        rejected = any(
            candidate["acceptance"]["stage"] == row["stage"]
            and not candidate["accepted"]
            for candidate in data["oracle"]
        )
        validate_acceptance(
            row, tactic, label + ": native guard " + row["stage"], not rejected
        )
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
            {canonical(row["tactic"]) for row in runtime_profiles} == expected_pool,
            label + ": runtime profiling candidate coverage",
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


def summarize_case(workers, m):
    indexed = {(row["seed"], row["arm"], row["phase"]): row for row in workers}
    paired = []
    for phase in PHASES:
        for execution in EXECUTIONS:
            for numerator, denominator in (
                ("v2", "v1"),
                ("v2", "default"),
                ("v1", "default"),
            ):
                trials = []
                for seed in SEEDS:
                    left = indexed[(seed, numerator, phase)]["natural"][execution][
                        "median_ms"
                    ]
                    right = indexed[(seed, denominator, phase)]["natural"][execution][
                        "median_ms"
                    ]
                    trials.append(
                        {
                            "seed": seed,
                            "numerator_ms": left,
                            "denominator_ms": right,
                            "ratio": left / right,
                        }
                    )
                paired.append(
                    {
                        "phase": phase,
                        "execution": execution,
                        "comparison": numerator + "_over_" + denominator,
                        "paired_trials": trials,
                        "median_paired_ratio": statistics.median(
                            row["ratio"] for row in trials
                        ),
                        "independent_trials": 3,
                        "scope": "natural public operation; default uses its genuine uncached fallback path",
                    }
                )
    regrets, natural_diagnostics = [], []
    for row in workers:
        if row["arm"] == "default":
            continue
        accepted = [candidate for candidate in row["oracle"] if candidate["accepted"]]
        selected = next(
            candidate
            for candidate in accepted
            if candidate["tactic"] == row["natural"]["tactic"]
        )
        for execution in EXECUTIONS:
            best = min(
                accepted, key=lambda candidate: candidate[execution]["median_ms"]
            )
            winner_ms = selected[execution]["median_ms"]
            best_ms = best[execution]["median_ms"]
            regrets.append(
                {
                    "seed": row["seed"],
                    "arm": row["arm"],
                    "phase": row["phase"],
                    "execution": execution,
                    "lookup_tier": selected["lookup_tier"],
                    "selected_tactic": row["natural"]["tactic"],
                    "best_tactic": best["tactic"],
                    "selected_oracle_median_ms": winner_ms,
                    "best_oracle_median_ms": best_ms,
                    "selected_to_best_ratio": winner_ms / best_ms,
                    "fractional_regret": winner_ms / best_ms - 1,
                    "accepted_candidates": len(accepted),
                    "excluded_candidates": len(row["oracle"]) - len(accepted),
                }
            )
            natural_ms = row["natural"][execution]["median_ms"]
            natural_diagnostics.append(
                {
                    "seed": row["seed"],
                    "arm": row["arm"],
                    "phase": row["phase"],
                    "execution": execution,
                    "natural_median_ms": natural_ms,
                    "selected_oracle_median_ms": winner_ms,
                    "natural_to_selected_oracle_ratio": natural_ms / winner_ms,
                    "natural_to_best_oracle_ratio": natural_ms / best_ms,
                    "scope": "separate measurement windows; ratios may be below one and are not selected-tactic regret",
                }
            )
    startup = []
    for phase in PHASES:
        ratios = [
            indexed[(seed, "v2", phase)]["startup_ms"]
            / indexed[(seed, "v1", phase)]["startup_ms"]
            for seed in SEEDS
        ]
        startup.append(
            {
                "phase": phase,
                "paired_v2_over_v1": ratios,
                "median_paired_ratio": statistics.median(ratios),
                "scope": "instrumented operation wrapper startup; no full-server startup measurement",
            }
        )
    return {
        "m": m,
        "natural_latency_comparisons": paired,
        "selected_tactic_regret": regrets,
        "cross_window_natural_diagnostics": natural_diagnostics,
        "startup_diagnostics": startup,
        "natural_winners": [
            {
                "seed": row["seed"],
                "arm": row["arm"],
                "phase": row["phase"],
                "tactic": row["natural"]["tactic"],
                "lookup_tier": row["natural"]["lookup_tier"],
            }
            for row in workers
        ],
        "candidate_exclusions": [
            {
                "seed": row["seed"],
                "arm": row["arm"],
                "phase": row["phase"],
                "tactic": candidate["tactic"],
                "reason": candidate["exclusion_reason"],
            }
            for row in workers
            for candidate in row["oracle"]
            if not candidate["accepted"]
        ],
    }


def analyze(args):
    ms = args.ms or ([args.m] if args.m is not None else [128])
    require(len(set(ms)) == len(ms), "duplicate case")
    manifest = load(args.source_manifest)
    require(
        manifest["upstream_pr"]["number"] == 6148
        and manifest["upstream_pr"]["head_commit"]
        == "14c5c67a1eb9ad3a608b8d3da3bf6d8924037c5c",
        "unexpected descriptor implementation",
    )
    require(
        manifest["baseline"]["tag"] == "v0.7.2"
        and manifest["clean_git_apply_check"]
        and not manifest["manual_resolution"],
        "unqualified source overlay",
    )
    provenance = validate_outer(args, ms)
    cases, all_workers = [], []
    for m in ms:
        workers = []
        for seed in SEEDS:
            for arm in ARMS:
                pair = []
                for phase in PHASES:
                    path = args.raw_root / f"m{m}-seed{seed}-{arm}-{phase}.json"
                    data = validate_worker(path, args, manifest, provenance)
                    require(
                        data["shape"] == [m, 1024, 1024]
                        and data["seed"] == seed
                        and data["arm"] == arm
                        and data["phase"] == phase,
                        "worker filename/workload mismatch",
                    )
                    workers.append(data)
                    pair.append(data)
                validate_pair(pair[0], pair[1], args)
        require(len(workers) == 18, "worker count")
        for seed in SEEDS:
            require(
                len(
                    {
                        canonical(row["environment"]["operand_sha256"])
                        for row in workers
                        if row["seed"] == seed
                    }
                )
                == 1,
                f"M{m}/seed{seed}: input bytes differ across arms",
            )
        all_workers.extend(workers)
        cases.append(summarize_case(workers, m))
    require(
        len({row["_container_id"] for row in all_workers}) == len(all_workers),
        "worker container reused",
    )
    require(
        len(
            {
                row["environment"]["native_artifact"]["library_sha256"]
                for row in all_workers
            }
        )
        == 1,
        "native artifact changed",
    )
    return {
        "status": "success",
        "schema_version": 1,
        "source_commit": args.expected_source_commit,
        "source_manifest_sha256": sha256(args.source_manifest),
        "analyzer_sha256": sha256(Path(__file__)),
        "outer_provenance": provenance,
        "validated_workers": [
            {
                "file": row["_path"],
                "sha256": sha256(Path(row["_path"])),
                "container_id": row["_container_id"],
                "seed": row["seed"],
                "arm": row["arm"],
                "phase": row["phase"],
                "m": row["shape"][0],
            }
            for row in all_workers
        ],
        "cases": cases,
        "limitations": [
            "One H100 and three paired fresh-worker trials per case; 15 within-worker samples are not 15 independent trials.",
            "Candidate source is the exact descriptor patch over 0.7.2, not a released package or the entire upstream branch.",
            "Selected-tactic regret uses the selected candidate and best accepted candidate within the same oracle sweep and lookup tier.",
            "Natural/oracle cross-window ratios can be below one and are reported separately from regret.",
            "Default natural execution is a genuine cache miss; cached default oracle measurements use memory or file tier according to phase.",
            "Matching lookup tiers does not establish identical cache occupancy, process history or thermal state.",
            "Unsupported candidates retain explicit exclusion reasons; the best accepted sample is not a statistical guarantee of optimality.",
            "No serving-performance, broad backend, BF16, distributed, managed-store or measurement-policy adoption claim.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--m", type=int, choices=(128, 144))
    group.add_argument("--ms", type=int, nargs="+", choices=(128, 144))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = analyze(args)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as error:
        result = {
            "status": "error",
            "validation_errors": [f"{type(error).__name__}: {error}"],
            "cases": [],
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
