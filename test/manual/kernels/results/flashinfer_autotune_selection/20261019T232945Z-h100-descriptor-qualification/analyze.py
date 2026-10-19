#!/usr/bin/env python3
"""Validate paired FP8 source arms, persistence and separate native observations.

Performance summaries use three paired fresh-worker medians. Instrumented native
API durations and traces provide diagnostic evidence only. No comparison is
published unless every required worker, cache and provenance check passes.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import io
import json
import math
import random
import re
import statistics
import struct
import subprocess
import tarfile
from pathlib import Path

WORKER = "test/manual/kernels/flashinfer_autotune_descriptor_qualification.py"
HELPER = "test/manual/kernels/flashinfer_autotune_eager_setup.py"
WRAPPER = "python/sglang/srt/model_executor/runner/flashinfer_autotune.py"
RUNTIME_KEY_PREFIX = "('fp8_gemm', 'CublasFp8GemmRunner', ((1, 128, 1024),"
SEEDS = (17, 18, 19)
ARMS = ("baseline", "candidate")
PHASES = ("first_eager", "warm_eager", "capture", "graph_replay")
BATCH = "test/manual/kernels/results/flashinfer_autotune_selection/20261019T232945Z-h100-descriptor-qualification"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    def invalid(value):
        raise ValueError(f"nonfinite JSON value: {value}")

    return json.loads(path.read_text(), parse_constant=invalid)


def positive(value):
    return type(value) in (float, int) and math.isfinite(value) and value > 0


def valid_tactic(value, arm):
    if value == -1:
        return True
    if arm == "baseline":
        return type(value) is int and value >= 0
    return (
        isinstance(value, list)
        and len(value) == 8
        and all(type(word) is int and 0 <= word < 1 << 64 for word in value)
    )


def git_hash(root, revision, path):
    content = subprocess.check_output(["git", "show", revision + ":" + path], cwd=root)
    return hashlib.sha256(content).hexdigest()


def git_blob(args, path):
    return subprocess.check_output(
        ["git", "show", args.expected_source_commit + ":" + path], cwd=args.source_root
    )


def validate_outer(args):
    plan_bytes = git_blob(args, BATCH + "/plan.json")
    plan = json.loads(plan_bytes)
    preflight = load(args.raw_root / "manifest.json")
    require(
        preflight["plan_sha256"] == hashlib.sha256(plan_bytes).hexdigest(), "plan hash"
    )
    require(
        preflight["source_manifest_sha256"]
        == sha256(args.source_manifest)
        == plan["source_arms"]["source_manifest_sha256"],
        "source manifest differs from preregistration",
    )
    require(
        (args.raw_root / "execution-commit.txt").read_text().strip()
        == args.expected_source_commit,
        "launcher execution commit",
    )
    for path, expected_hash in preflight["source_sha256"].items():
        require(
            git_hash(args.source_root, args.expected_source_commit, path)
            == expected_hash,
            "launcher source hash: " + path,
        )
    gpu, environment = preflight["gpu"], plan["environment"]
    require(gpu["name"] == environment["gpu"], "GPU model differs from plan")
    require(
        gpu["uuid"].removeprefix("GPU-") == environment["physical_gpu_uuid"],
        "GPU UUID differs from plan",
    )
    require(
        gpu["driver_version"] == environment["driver"], "GPU driver differs from plan"
    )
    require(
        not (args.raw_root / "gpu-processes-before.csv").read_text().strip(),
        "GPU was in use",
    )
    image = load(args.raw_root / "image.json")
    require(
        image["Architecture"] == "amd64" and image["Os"] == "linux", "image platform"
    )
    image_digest = plan["container_image"].split("@")[-1]
    require(
        any(value.split("@")[-1] == image_digest for value in image["RepoDigests"]),
        "image digest differs from plan",
    )
    for name in ("prepare", "build", "launcher", "upstream-tests"):
        require(
            (args.raw_root / (name + ".exit-code")).read_text().strip() == "0",
            name + " failed",
        )
    probe = args.raw_root / "probe.so"
    build_records = re.findall(
        r"(?m)^([0-9a-f]{64})\s+(.+)$", (args.raw_root / "build.log").read_text()
    )
    require(
        (sha256(probe), str(probe))
        in [
            (value, str(args.raw_root / Path(path).name))
            for value, path in build_records
        ],
        "probe binary differs from its build log",
    )
    require(
        git_hash(
            args.source_root,
            args.expected_source_commit,
            "test/manual/kernels/flashinfer_autotune_descriptor_native.cpp",
        )
        in {value for value, path in build_records if path.endswith(".cpp")},
        "probe source differs from its build log",
    )
    return {
        "image": image,
        "plan_sha256": preflight["plan_sha256"],
        "launcher_manifest_sha256": sha256(args.raw_root / "manifest.json"),
        "probe_binary_sha256": sha256(probe),
        "gpu": gpu,
    }


def locate(root, stored_path):
    """Archive paths differ from container paths; require one exact suffix match."""
    stored = Path(stored_path)
    candidates = list(root.rglob(stored.name))
    if len(candidates) > 1:
        suffix = stored.parts[-2:]
        candidates = [path for path in candidates if path.parts[-2:] == suffix]
    require(len(candidates) == 1, "missing or ambiguous artifact: " + str(stored_path))
    return candidates[0]


def manifest_hashes(manifest, arm):
    return {
        row["installed_package_path"]: row[arm + "_sha256"]
        for row in manifest["files"]
        if row["runtime_overlay"]
    }


def validate_timing(row, protocol, strict, label):
    iterations = row["effective_iterations"]
    requested = protocol["requested_iterations"]
    require(type(iterations) is int and requested <= iterations <= 65536, label)
    require(row["iterations_per_sample"] == iterations, label + ": normalization")
    require(row["requested_iterations"] == requested, label + ": requested count")
    require(row["warmup_actual_ms"] >= protocol["warmup_ms"], label + ": warmup")
    require(row["warmup_calls"] >= requested, label + ": warmup calls")
    calibration = row["calibration_batches"]
    require(calibration and calibration[0]["iterations"] == requested, label)
    require(calibration[-1]["iterations"] == iterations, label)
    require(calibration[-1]["wall_ms"] >= protocol["min_sample_ms"], label)
    for previous, current in zip(calibration, calibration[1:]):
        require(previous["wall_ms"] < protocol["min_sample_ms"], label)
        require(current["iterations"] == min(previous["iterations"] * 2, 65536), label)
    for samples, batches in (
        ("samples_ms", "wall_batch_ms"),
        ("gpu_event_samples_ms", "gpu_event_batch_ms"),
    ):
        require(len(row[samples]) == len(row[batches]) == protocol["samples"], label)
        for sample, batch in zip(row[samples], row[batches]):
            require(positive(sample) and positive(batch), label)
            require(math.isclose(sample * iterations, batch, rel_tol=1e-12), label)
    require(
        math.isclose(
            row["median_ms"], statistics.median(row["samples_ms"]), rel_tol=1e-12
        ),
        label + ": median",
    )
    if strict:
        require(
            protocol["warmup_ms"] == 500
            and protocol["min_sample_ms"] == 20
            and protocol["samples"] == 15
            and requested == 20,
            label + ": preregistered timing protocol",
        )


def validate_worker(path, args, manifest, expected, strict):
    data = load(path)
    label = path.name
    require(data["status"] == "success" and data["schema_version"] == 1, label)
    arm, mode = data["implementation"], data["mode"]
    require(arm in ARMS, label)
    require(data["shape"] == [128, 1024, 1024], label + ": workload")
    require(data["seed"] in SEEDS, label + ": seed")
    require(data["worker_sha256"] == expected["worker"], label + ": worker source")
    require(data["helper_sha256"] == expected["helper"], label + ": helper source")
    require(data["manifest_sha256"] == sha256(args.source_manifest), label)
    require(not data["forbidden_profile_attempts"], label + ": forbidden profiling")
    require(
        data["profile_attempts"] == len(data["profiles"]), label + ": profile counts"
    )
    if mode != "lifecycle-cold":
        require(
            data["profile_attempts"] == 0, label + ": profiling outside cold tuning"
        )
    env = data["environment"]
    require(env["sglang_commit"] == args.expected_source_commit, label + ": Git commit")
    require(
        env["sglang_wrapper_sha256"] == expected["wrapper"], label + ": wrapper source"
    )
    require(env["flashinfer_source_sha256"] == manifest_hashes(manifest, arm), label)
    require(
        env["flashinfer_tuner_sha256"]
        == manifest_hashes(manifest, arm)["flashinfer/autotuner/autotuner.py"],
        label + ": tuner source",
    )
    require(env["torch"] == "2.14.1+cu130" and env["torch_cuda"] == "13.0", label)
    require(env["flashinfer"] == "0.7.2" and env["backend"] == "cublas", label)
    require(env["capability"] == [9, 0] and "H100" in env["gpu"], label)
    uuid = env["physical_gpu_uuid"].removeprefix("GPU-")
    require(uuid not in ("", "None"), label + ": absent physical GPU UUID")
    if args.expected_gpu_uuid:
        require(
            uuid == args.expected_gpu_uuid.removeprefix("GPU-"), label + ": GPU differs"
        )
    require(env["measurement_policy"] == "unspecified", label + ": measurement policy")
    if mode == "observe":
        require(env["LD_PRELOAD"] and env["native_probe_output"], label + ": no probe")
        require(not data["measurements"] and data["observations"], label)
    else:
        require(not env["LD_PRELOAD"], label + ": instrumented primary worker")
        require(not data["observations"] and not data["traces"], label)
    artifact = env["native_artifact"]
    require(artifact["source_sha256"] == env["flashinfer_source_sha256"], label)
    require(
        len(artifact["library_sha256"]) == len(artifact["ninja_sha256"]) == 64, label
    )
    require(
        (
            "bmm_fp8_run_with_descriptor"
            if arm == "candidate"
            else "bmm_fp8_run_with_algo"
        )
        in artifact["required_native_exports"],
        label + ": native ABI",
    )
    matched = data["matched_algorithm"]
    raw = bytes.fromhex(matched["raw_hex"])
    require(
        len(raw) == 64 and list(struct.unpack("8Q", raw)) == matched["descriptor"],
        label,
    )
    require(
        matched["actual_tactic"] == (0 if arm == "baseline" else matched["descriptor"]),
        label,
    )
    require(matched["actual_tactic"] in matched["eligible_tactics"], label)
    require(
        set(matched["operand_sha256"]) == {"a", "b", "scale_a", "scale_b"}
        and all(len(value) == 64 for value in matched["operand_sha256"].values()),
        label + ": operand hashes",
    )
    match = matched["match_provenance"]
    if "path" in match:
        require(sha256(locate(args.raw_root, match["path"])) == match["sha256"], label)
    correctness = data["correctness"]
    require(correctness["passed"] and correctness["rtol"] == 0.02, label)
    require(correctness["atol"] == 0.125 and correctness["checks"], label)
    for row in correctness["checks"]:
        require(
            type(row["max_abs_error"]) in (float, int)
            and math.isfinite(row["max_abs_error"]),
            label,
        )
    require(data["lookups"], label + ": no observed real lookups")
    require(
        all(
            row["cache_hit"] and valid_tactic(row["tactic"], arm)
            for row in data["lookups"]
        ),
        label,
    )
    for row in data["binding_acceptance_checks"]:
        require(arm == "candidate", label)
        values = row["used_descriptor"]
        if row["expected_tactic"] == -1:
            require(not values, label + ": default entered descriptor binding")
        else:
            require(
                len(values) == (5 if row["stage"].endswith("_warm_eager") else 1)
                and all(value == 1 for value in values),
                label + ": silent fallback",
            )
    if arm == "candidate":
        require(
            any(row["used_descriptor"] for row in data["binding_acceptance_checks"]),
            label,
        )
    require(
        path.with_suffix(".exit-code").read_text().strip() == "0",
        label + ": Docker exit",
    )
    cid = path.with_suffix(".container-id").read_text().strip()
    require(re.fullmatch(r"[0-9a-f]{64}", cid), label + ": container identity")
    data["_container_id"] = cid
    data["_path"] = str(path)
    if mode in ("benchmark", "observe"):
        expected_order = ["default", "explicit"]
        random.Random(data["seed"]).shuffle(expected_order)
        require(data["timing_protocol"]["variant_order"] == expected_order, label)
        for row in data["lookups"]:
            require(
                row["tactic"]
                == (
                    -1
                    if row["stage"].startswith("default_")
                    else matched["actual_tactic"]
                ),
                label + ": lookup selected another tactic",
            )
            require(
                row["capturing"] == row["stage"].endswith("_capture"),
                label + ": incorrect capture state",
            )
        if mode == "benchmark":
            require(
                [row["name"] for row in data["measurements"]] == expected_order, label
            )
            for row in data["measurements"]:
                require(
                    row["tactic"]
                    == (-1 if row["name"] == "default" else matched["actual_tactic"]),
                    label,
                )
                for execution in ("eager", "cuda_graph"):
                    validate_timing(
                        row[execution],
                        data["timing_protocol"],
                        strict,
                        label + "/" + execution,
                    )
            checks = collections.Counter(row["stage"] for row in data["lookups"])
            require(
                checks
                == collections.Counter(
                    name + "_" + stage
                    for name in ("default", "explicit")
                    for stage in ("before", "capture", "after")
                ),
                label,
            )
            for name in ("default", "explicit"):
                labels = {row["check"] for row in correctness["checks"]}
                require(
                    {
                        name + "_changed_input",
                        name + "_restored_input",
                        name + "_measured_eager",
                        name + "_measured_cuda_graph",
                    }
                    <= labels,
                    label + ": numerical boundary coverage",
                )
    return data


def validate_native(observer, raw_root):
    path = locate(raw_root, observer["environment"]["native_probe_output"])
    events = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    phases = collections.defaultdict(list)
    for event in events:
        phases[event["phase"]].append(event)
    expected = {
        name + "_" + phase for name in ("default", "explicit") for phase in PHASES
    }
    require(set(phases) == expected, "native phase coverage")
    cpu_rows = {row["phase"]: row for row in observer["observations"]}
    require(set(cpu_rows) == expected, "CPU observation coverage")
    arm = observer["implementation"]
    result = {}
    for name in ("default", "explicit"):
        for phase in PHASES:
            label = name + "_" + phase
            rows = phases[label]
            require(len([r for r in rows if r["event"] == "phase_begin"]) == 1, label)
            summaries = [r for r in rows if r["event"] == "phase_summary"]
            require(len(summaries) == 1, label)
            require(
                rows[0]["event"] == "phase_begin"
                and rows[-1]["event"] == "phase_summary",
                label,
            )
            require(rows[0]["algo_bytes"] == 64, label)
            summary = summaries[0]["apis"]
            require(
                "algo_check" in summary
                and all(r["failures"] == 0 for r in summary.values()),
                label,
            )
            cpu = cpu_rows[label]
            expected_calls = 0 if phase == "graph_replay" else cpu["calls"]
            require(
                cpu["calls"] == (5 if phase in ("warm_eager", "graph_replay") else 1),
                label,
            )
            require(
                {key: value["calls"] for key, value in cpu["cpu_boundaries"].items()}
                == ({"tuner_choose_one": expected_calls} if expected_calls else {}),
                label,
            )
            matmul = [r for r in rows if r["event"] == "matmul"]
            heuristic = [r for r in rows if r["event"] == "heuristic"]
            checks = [r for r in rows if r["event"] == "algo_check"]
            expected_queries = (
                expected_calls if name == "default" or arm == "baseline" else 0
            )
            expected_checks = (
                1
                if arm == "candidate" and name == "explicit" and phase == "first_eager"
                else 0
            )
            require(len(matmul) == expected_calls, label + ": matmul interception")
            require(len(heuristic) == expected_queries, label + ": heuristic count")
            require(len(checks) == expected_checks, label + ": descriptor-check count")
            for api, records in (
                ("matmul", matmul),
                ("heuristic", heuristic),
                ("algo_check", checks),
            ):
                require(summary[api]["calls"] == len(records), label)
                require(
                    summary[api]["host_ns"] == sum(r["host_ns"] for r in records), label
                )
                require(all(r["status"] == 0 for r in records), label)
            for query in heuristic:
                require(query["requested"] == (100 if name == "explicit" else 1), label)
                require(0 < query["returned"] <= query["requested"], label)
                require(len(query["algorithms"]) == query["returned"], label)
                require(all(r["state"] == 0 for r in query["algorithms"]), label)
            for record in matmul:
                algorithm = record["algo"]
                require(algorithm and algorithm["raw_bytes"] == 64, label)
                require(len(bytes.fromhex(algorithm["raw_hex"])) == 64, label)
                require(
                    all(
                        r["status"] == 0 and r["bytes"] == 4
                        for r in algorithm["config"].values()
                    ),
                    label,
                )
                if name == "explicit":
                    require(
                        algorithm["raw_hex"]
                        == observer["matched_algorithm"]["raw_hex"],
                        label + ": descriptor identity",
                    )
            for check in checks:
                require(check["result"] and check["result"]["state"] == 0, label)
                require(
                    check["algo"]["raw_hex"]
                    == observer["matched_algorithm"]["raw_hex"],
                    label,
                )
                require(
                    check["result"]["workspace_bytes"] <= matmul[0]["workspace_bytes"],
                    label,
                )
            result[label] = {
                "api_summary": summary,
                "heuristic_requests": [r["requested"] for r in heuristic],
                "submitted_algorithms": [r["algo"] for r in matmul],
                "descriptor_checks": checks,
            }
    return {"path": str(path), "sha256": sha256(path), "phases": result}


def validate_traces(observer, raw_root):
    require(
        {row["variant"] for row in observer["traces"]} == {"default", "explicit"},
        "trace coverage",
    )
    result = {}
    for row in observer["traces"]:
        path = locate(raw_root, row["path"])
        require(sha256(path) == row["sha256"], "trace hash mismatch")
        with gzip.open(path, "rt") as source:
            events = json.load(source)["traceEvents"]
        scopes = [event for event in events if event.get("cat") == "user_annotation"]
        kernels = [event for event in events if event.get("cat") == "kernel"]
        stages = {}
        for execution in ("eager", "graph_replay"):
            labels = [
                event
                for event in scopes
                if event.get("name") == row["variant"] + "/" + execution
            ]
            require(len(labels) == 1, "missing or ambiguous trace scope")
            label = labels[0]
            within = [
                event
                for event in kernels
                if label["ts"] <= event["ts"] < label["ts"] + label["dur"]
            ]
            require(
                len(within) == 5
                and all(positive(event.get("dur")) for event in within),
                "kernel trace coverage",
            )
            stages[execution] = dict(
                collections.Counter(event["name"] for event in within)
            )
        result[row["variant"]] = {
            "sha256": sha256(path),
            "kernel_names_by_execution": stages,
        }
    return result


def validate_lifecycle_pair(cold, warm, raw_root):
    arm = cold["implementation"]
    left, right = cold["lifecycle"], warm["lifecycle"]
    require(left["phase"] == "cold" and right["phase"] == "warm", "lifecycle phases")
    require(left["tuning_version"] == right["tuning_version"], "lifecycle version")
    require(
        left["natural_winner"] == right["natural_winner"],
        "natural winner lost on restart",
    )
    require(
        left["explicit_control"] == right["explicit_control"],
        "explicit winner lost on restart",
    )
    require(
        left["profile_calls"] > 0
        and right["profile_calls"] == right["profile_attempts"] == 0,
        "warm profiling",
    )
    require(
        all(value is None for value in left["cache_sha256_before"].values()),
        "cold cache not empty",
    )
    require(
        left["cache_sha256_after"]
        == right["cache_sha256_before"]
        == right["cache_sha256_after"],
        "cache files changed on warm restart",
    )
    for data in (cold, warm):
        labels = {row["check"] for row in data["correctness"]["checks"]}
        require(
            {
                "natural_graph_changed_input",
                "natural_graph_restored_input",
                "explicit_persisted_graph_changed_input",
                "explicit_persisted_graph_restored_input",
            }
            <= labels,
            "lifecycle graph coverage",
        )
        stages = {row["stage"] for row in data["lookups"]}
        require(
            {
                "natural_bare_after_context",
                "natural_capture_after_context",
                "explicit_bare_after_reload",
                "explicit_capture_after_reload",
            }
            <= stages,
            "lifecycle lookup coverage",
        )
        for name, field in (
            ("natural", "natural_cache"),
            ("explicit", "explicit_cache"),
        ):
            path = locate(raw_root, data["lifecycle"][field])
            require(
                sha256(path) == data["lifecycle"]["cache_sha256_after"][name],
                "persisted cache hash",
            )
            payload = load(path)
            require(
                payload["_metadata"]["flashinfer_version"] == "0.7.2", "cache metadata"
            )
            entries = [
                value
                for key, value in payload.items()
                if key.startswith(RUNTIME_KEY_PREFIX)
            ]
            require(len(entries) == 1, "ambiguous runtime cache entry")
            value = entries[0]
            require(value[0] == "CublasFp8GemmRunner", "persisted runner")
            expected = (
                left["natural_winner"]
                if name == "natural"
                else left["explicit_control"]
            )
            require(
                value[1] == expected and valid_tactic(value[1], arm),
                "persisted runtime winner",
            )
            if name == "explicit":
                require(
                    value == data["lifecycle"]["explicit_file_entry"],
                    "explicit payload mismatch",
                )
    profiles = cold["profiles"]
    require(
        all(
            not row["use_cuda_graph"] and not row["use_cold_l2_cache"]
            for row in profiles
        ),
        "operation measurement defaults changed",
    )
    runtime = [
        row
        for row in profiles
        if row["input_shapes"][0][-2] == 128 and row.get("profiler_ms") is not None
    ]
    require(
        runtime and all(positive(row["profiler_ms"]) for row in runtime),
        "no finite runtime profile",
    )
    if left["tuning_version"] == "v2":
        require(any(row["tactic"] == -1 for row in runtime), "v2 default omitted")
    else:
        require(
            all(row["tactic"] != -1 for row in profiles),
            "v1 unexpectedly profiled default",
        )
    selected = [
        row["profiler_ms"] for row in runtime if row["tactic"] == left["natural_winner"]
    ]
    require(
        selected and min(selected) == min(row["profiler_ms"] for row in runtime),
        "winner differs from runtime profiling minimum",
    )
    return {
        "implementation": arm,
        "tuning_version": left["tuning_version"],
        "natural_winner": left["natural_winner"],
        "forced_persisted_control": left["explicit_control"],
        "cold_profile_calls": left["profile_calls"],
        "warm_profile_calls": right["profile_calls"],
        "runtime_profiles": runtime,
        "cache_sha256": left["cache_sha256_after"],
        "cold_startup_ms": left["startup_ms"],
        "warm_startup_ms": right["startup_ms"],
        "scope": "one pair per implementation/version; no startup performance conclusion",
    }


def validate_upstream(args, manifest):
    path = args.raw_root / "upstream-tests.json"
    data = load(path)
    require(
        data["status"] == "success" and data["pytest_exit_code"] == 0,
        "upstream tests failed",
    )
    require(data["passed"] == data["expected_test_count"] == 19, "upstream pass count")
    require(data["failed_or_skipped"] == 0, "upstream failed/skipped coverage")
    reports = data["reports"]
    require(
        len(reports) == len({row["nodeid"] for row in reports}) == 19
        and all(
            row["phase"] == "call" and row["outcome"] == "passed" for row in reports
        ),
        "upstream case report coverage",
    )
    require(
        data["source_manifest_sha256"] == sha256(args.source_manifest),
        "upstream source manifest",
    )
    artifact = data["native_artifact"]
    require(
        artifact["source_sha256"] == manifest_hashes(manifest, "candidate"),
        "upstream implementation",
    )
    support_bytes = git_blob(args, BATCH + "/upstream-test-support.tar.gz")
    with tarfile.open(fileobj=io.BytesIO(support_bytes), mode="r:gz") as archive:
        support = json.load(archive.extractfile("test-support-manifest.json"))
    expected_sources = {
        row["repository_path"]: row["sha256"] for row in support["files"]
    }
    target = "tests/autotuner/test_autotuner_cublaslt_tactics.py"
    expected_sources[target] = next(
        row["candidate_sha256"]
        for row in manifest["files"]
        if row["repository_path"] == target
    )
    require(
        data["test_source_sha256"] == expected_sources, "upstream tests/helpers changed"
    )
    return {
        "sha256": sha256(path),
        "passed": 19,
        "failed_or_skipped": 0,
        "nodeids": [row["nodeid"] for row in reports],
        "native_library_sha256": artifact["library_sha256"],
        "test_source_sha256": expected_sources,
    }


def summarize_timings(primary):
    indexed = {
        (row["implementation"], row["seed"]): {
            r["name"]: r for r in row["measurements"]
        }
        for row in primary
    }
    comparisons = []
    specifications = [
        (
            "candidate_explicit_over_baseline_explicit",
            ("candidate", "explicit"),
            ("baseline", "explicit"),
        ),
        (
            "candidate_default_over_baseline_default",
            ("candidate", "default"),
            ("baseline", "default"),
        ),
        (
            "baseline_explicit_over_default",
            ("baseline", "explicit"),
            ("baseline", "default"),
        ),
        (
            "candidate_explicit_over_default",
            ("candidate", "explicit"),
            ("candidate", "default"),
        ),
    ]
    for name, numerator, denominator in specifications:
        for execution in ("eager", "cuda_graph"):
            trials = []
            for seed in SEEDS:
                num = indexed[(numerator[0], seed)][numerator[1]][execution][
                    "median_ms"
                ]
                den = indexed[(denominator[0], seed)][denominator[1]][execution][
                    "median_ms"
                ]
                trials.append(
                    {
                        "seed": seed,
                        "numerator_median_ms": num,
                        "denominator_median_ms": den,
                        "ratio": num / den,
                    }
                )
            comparisons.append(
                {
                    "comparison": name,
                    "execution": execution,
                    "paired_trials": trials,
                    "median_paired_ratio": statistics.median(
                        row["ratio"] for row in trials
                    ),
                    "independent_trial_count": 3,
                    "ratio_direction": "latency numerator/denominator; smaller is faster",
                }
            )
    return comparisons


def analyze(args):
    manifest = load(args.source_manifest)
    require(
        manifest["clean_git_apply_check"] and not manifest["manual_resolution"],
        "overlay changed manually",
    )
    require(manifest["baseline"]["tag"] == "v0.7.2", "unexpected source baseline")
    require(
        manifest["upstream_pr"]["number"] == 6148
        and manifest["upstream_pr"]["head_commit"]
        == "14c5c67a1eb9ad3a608b8d3da3bf6d8924037c5c",
        "unexpected source candidate",
    )
    outer = validate_outer(args)
    expected = {
        name: git_hash(args.source_root, args.expected_source_commit, path)
        for name, path in (("worker", WORKER), ("helper", HELPER), ("wrapper", WRAPPER))
    }
    primary, observers, lifecycles, qualification = [], [], {}, []
    expected_names = {f"benchmark-{arm}-{seed}.json" for arm in ARMS for seed in SEEDS}
    expected_names |= {f"observation-{arm}.json" for arm in ARMS}
    pairs = (("baseline", "v2"), ("candidate", "v1"), ("candidate", "v2"))
    expected_names |= {
        f"lifecycle-{arm}-{version}-{phase}.json"
        for arm, version in pairs
        for phase in ("cold", "warm")
    }
    found = [
        path for path in args.raw_root.rglob("*.json") if path.name in expected_names
    ]
    require(
        len({path.name for path in found}) == len(found), "duplicate worker filenames"
    )
    paths = {path.name: path for path in found}
    require(
        set(paths) == expected_names,
        "required worker set missing: " + str(sorted(expected_names - set(paths))),
    )
    all_workers = []
    for name in sorted(paths):
        data = validate_worker(
            paths[name], args, manifest, expected, strict=name.startswith("benchmark-")
        )
        all_workers.append(data)
        if name.startswith("benchmark-"):
            primary.append(data)
        elif name.startswith("observation-"):
            observers.append(data)
        else:
            lifecycle = data["lifecycle"]
            lifecycles[
                (
                    data["implementation"],
                    lifecycle["tuning_version"],
                    lifecycle["phase"],
                )
            ] = data
    for path in sorted(args.raw_root.rglob("qualification-*.json")):
        data = validate_worker(path, args, manifest, expected, strict=False)
        qualification.append(data)
        all_workers.append(data)
    require(qualification, "missing separate qualification workers")
    require(
        len({row["_container_id"] for row in all_workers}) == len(all_workers),
        "reused worker container",
    )
    require(
        len({row["environment"]["physical_gpu_uuid"] for row in all_workers}) == 1,
        "workers used different GPUs",
    )
    for arm in ARMS:
        require(
            len(
                {
                    row["environment"]["native_artifact"]["library_sha256"]
                    for row in all_workers
                    if row["implementation"] == arm
                }
            )
            == 1,
            "native library changed within arm",
        )
    require(
        primary[0]["environment"]["native_artifact"]["library_sha256"]
        != next(
            row
            for row in primary
            if row["implementation"] != primary[0]["implementation"]
        )["environment"]["native_artifact"]["library_sha256"],
        "source arms loaded the same native artifact",
    )
    for seed in SEEDS:
        same_seed = [row for row in all_workers if row["seed"] == seed]
        require(
            len(
                {
                    json.dumps(
                        row["matched_algorithm"]["operand_sha256"], sort_keys=True
                    )
                    for row in same_seed
                }
            )
            == 1,
            "paired operand data differs",
        )
        require(
            len({row["matched_algorithm"]["raw_hex"] for row in same_seed}) == 1,
            "paired descriptor differs",
        )
    native = {
        row["implementation"]: validate_native(row, args.raw_root) for row in observers
    }
    traces = {
        row["implementation"]: validate_traces(row, args.raw_root) for row in observers
    }
    lifecycle_results = [
        validate_lifecycle_pair(
            lifecycles[(arm, version, "cold")],
            lifecycles[(arm, version, "warm")],
            args.raw_root,
        )
        for arm, version in pairs
    ]
    upstream = []
    for path in args.upstream_log:
        text = path.read_text()
        require(
            not re.search(r"\b[1-9][0-9]* (failed|errors?)\b", text),
            "upstream test failure",
        )
        matches = re.findall(r"\b([0-9]+) passed\b", text)
        require(matches, "no upstream pass count")
        upstream.append(
            {"path": str(path), "sha256": sha256(path), "passed": int(matches[-1])}
        )
    if args.expected_upstream_tests is not None:
        require(
            sum(row["passed"] for row in upstream) == args.expected_upstream_tests,
            "upstream test count",
        )
    upstream_result = validate_upstream(args, manifest)
    candidate_libraries = {
        row["environment"]["native_artifact"]["library_sha256"]
        for row in all_workers
        if row["implementation"] == "candidate"
    }
    require(
        upstream_result["native_library_sha256"] in candidate_libraries,
        "upstream tests loaded a different native candidate",
    )
    return {
        "status": "success",
        "schema_version": 1,
        "source_commit": args.expected_source_commit,
        "source_manifest_sha256": sha256(args.source_manifest),
        "analyzer_sha256": sha256(Path(__file__)),
        "outer_provenance": outer,
        "validated_workers": [
            {
                "file": row["_path"],
                "sha256": sha256(Path(row["_path"])),
                "implementation": row["implementation"],
                "mode": row["mode"],
                "seed": row["seed"],
                "container_id": row["_container_id"],
            }
            for row in all_workers
        ],
        "comparisons": summarize_timings(primary),
        "native_observations": native,
        "trace_observations": traces,
        "lifecycle": lifecycle_results,
        "upstream_tests": upstream_result,
        "optional_upstream_logs": upstream,
        "limitations": [
            "One FP8 shape on one H100; three paired worker trials, not 45 independent trials.",
            "Forced explicit descriptors are a mechanism control, not a claim that startup always selects them.",
            "Instrumented API durations are nested diagnostic boundaries and are excluded from performance estimates.",
            "Matching descriptor bytes/configuration and kernel names corroborates this workload only; opaque bytes alone do not prove identical physical kernels.",
            "No model-serving, broad backend, BF16, distributed, migration or clock-controlled qualification is established.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument("--upstream-log", type=Path, action="append", default=[])
    parser.add_argument("--expected-upstream-tests", type=int)
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
            "comparisons": [],
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
