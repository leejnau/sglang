#!/usr/bin/env python3
"""Independently validate and summarize the fixed-workload setup experiment.

Benchmark ratios use one median per fresh worker, then paired worker ratios.
Native API timings and traces are separate diagnostic evidence; they never enter
the uninstrumented timing estimates. Algorithm bytes/configuration observations
do not by themselves prove an identical physical GPU kernel.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
import random
import statistics
import subprocess
from pathlib import Path

PATHS = ("public_cached", "retained_runner", "public_seeded_algorithms")
VARIANTS = {
    path + suffix: {"path": path, "tactic": tactic}
    for path in PATHS
    for suffix, tactic in (("_default", -1), ("_ordinal0", 0))
}
TUNER_HASH = "e222f35f8b5a59a99a45fad627a79db8fc45fadc52b667e63b70c0968509109d"
GEMM_HASH = "ac5fa19277c00e4082858ed943882973399fdf2cd5fa83045df8821f95ba6e2a"
WRAPPER_HASH = "64deff581665bd04830535baccfdfbdfa72794d919bc462fc469a4c42bdd68f5"
GPU_UUID = "bc979292-ef6c-5590-af89-44b70faefecc"
WORKER_PATH = "test/manual/kernels/flashinfer_autotune_eager_setup.py"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_positive(value):
    return isinstance(value, (float, int)) and math.isfinite(value) and value > 0


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON number: {value}")

    return json.loads(path.read_text(), parse_constant=reject_constant)


def validate_measurement(row, protocol, label):
    iterations = row["effective_iterations"]
    require(isinstance(iterations, int) and 20 <= iterations <= 65536, label)
    require(row["iterations_per_sample"] == iterations, label)
    require(row["requested_iterations"] == protocol["requested_iterations"], label)
    require(row["warmup_actual_ms"] >= 500, label + ": short warmup")
    require(row["warmup_calls"] >= 20, label + ": no warmup calls")
    calibration = row["calibration_batches"]
    require(calibration and calibration[-1]["iterations"] == iterations, label)
    require(calibration[-1]["wall_ms"] >= 20, label + ": short calibration")
    require(calibration[0]["iterations"] == 20, label + ": initial calibration")
    for previous, current in zip(calibration, calibration[1:]):
        require(previous["wall_ms"] < 20, label + ": unnecessary calibration")
        require(current["iterations"] == min(previous["iterations"] * 2, 65536), label)
    for samples, batches in (
        ("samples_ms", "wall_batch_ms"),
        ("gpu_event_samples_ms", "gpu_event_batch_ms"),
    ):
        require(len(row[samples]) == len(row[batches]) == 15, label + ": sample count")
        for value, batch in zip(row[samples], row[batches]):
            require(finite_positive(value) and finite_positive(batch), label)
            require(math.isclose(value * iterations, batch, rel_tol=1e-12), label)
    require(
        math.isclose(
            row["median_ms"], statistics.median(row["samples_ms"]), rel_tol=1e-12
        ),
        label + ": median mismatch",
    )


def validate_worker(path, expected_commit, expected_worker_hash, mode):
    data = load(path)
    require(data["status"] == "success" and data["schema_version"] == 1, str(path))
    require(data["mode"] == mode, str(path) + ": mode")
    require(data["shape"] == [128, 1024, 1024], str(path) + ": shape")
    require(data["worker_sha256"] == expected_worker_hash, str(path) + ": source hash")
    require(data["profile_attempts"] == 0, str(path) + ": profiling occurred")
    env = data["environment"]
    require(env["sglang_commit"] == expected_commit, str(path) + ": source commit")
    require(
        env["sglang_wrapper_sha256"] == WRAPPER_HASH,
        str(path) + ": production wrapper changed",
    )
    require(env["torch"] == "2.14.1+cu130" and env["torch_cuda"] == "13.0", str(path))
    require(env["flashinfer"] == "0.7.2", str(path))
    require(env["flashinfer_tuner_sha256"] == TUNER_HASH, str(path) + ": tuner hash")
    require(
        env["flashinfer_gemm_source_sha256"] == GEMM_HASH,
        str(path) + ": GEMM source hash",
    )
    require(env["capability"] == [9, 0] and "H100" in env["gpu"], str(path))
    require(env["physical_gpu_uuid"] not in (None, "None", ""), str(path))
    require(
        env["physical_gpu_uuid"].removeprefix("GPU-") == GPU_UUID,
        str(path) + ": physical GPU differs from plan",
    )
    require(
        all(
            "580.159.04" in env[key]
            for key in ("gpu_status_before", "gpu_status_after")
        ),
        str(path) + ": driver version",
    )
    require(env["backend"] == "cublas" and env["bundled_configs"] is False, str(path))
    require(env["measurement_policy"] == "unspecified; no tuning occurs", str(path))
    if mode == "benchmark":
        require(
            not env["LD_PRELOAD"] and env["native_probe_available"] is False, str(path)
        )
        require(not data["observations"] and not data["traces"], str(path))
    else:
        require(env["LD_PRELOAD"] and env["native_probe_available"] is True, str(path))
        require(not data["measurements"], str(path))
    protocol = data["timing_protocol"]
    require(protocol["warmup_ms"] == 500 and protocol["min_sample_ms"] == 20, str(path))
    require(
        protocol["samples"] == 15 and protocol["requested_iterations"] == 20, str(path)
    )
    expected_order = list(VARIANTS)
    random.Random(data["seed"]).shuffle(expected_order)
    require(protocol["variant_order"] == expected_order, str(path) + ": variant order")
    intervention = data["intervention"]
    require(intervention["algorithm_count"] > 0, str(path))
    require(0 in intervention["runtime_eligible_tactics"], str(path))
    require(len(intervention["donor_cpu_buffer_sha256"]) == 64, str(path))
    require(intervention["donor_cpu_buffer_bytes"] > 0, str(path))
    require(len(intervention["input_signature"]) == 6, str(path))
    require(intervention["scales"] == [1.0, 1.0], str(path))
    require(
        set(intervention["operand_sha256"]) == {"a", "b", "scale_a", "scale_b"},
        str(path),
    )
    require(
        all(len(value) == 64 for value in intervention["operand_sha256"].values()),
        str(path),
    )
    checks = data["lookup_checks"]
    require(len(checks) == 18, str(path) + ": lookup check count")
    counts = collections.Counter((row["variant"], row["stage"]) for row in checks)
    require(
        counts
        == collections.Counter(
            (name, stage)
            for name in VARIANTS
            for stage in ("before", "capture", "after")
        ),
        str(path) + ": missing or duplicate lookup checks",
    )
    for row in checks:
        variant = VARIANTS[row["variant"]]
        expected = (
            []
            if variant["path"] == "retained_runner"
            else [{"cache_hit": True, "tactic": variant["tactic"]}]
        )
        require(
            row["lookups"] == expected, str(path) + ": false cache hit or wrong tactic"
        )
    correctness = data["correctness"]
    require(correctness["passed"] is True, str(path))
    require(correctness["rtol"] == 0.02 and correctness["atol"] == 0.125, str(path))
    labels = collections.Counter(row["check"] for row in correctness["checks"])
    required_labels = ["initial_untuned"]
    for name in VARIANTS:
        required_labels += [
            name + suffix
            for suffix in ("_before", "_changed_input", "_restored_input", "_after")
        ]
        required_labels += (
            [name + "_measured_" + execution for execution in ("eager", "cuda_graph")]
            if mode == "benchmark"
            else [name + "_observed"]
        )
        if mode == "observe":
            required_labels += [
                name + "_observation_graph" + suffix
                for suffix in ("_changed_input", "_restored_input")
            ]
    for label in required_labels:
        require(labels[label] == 1, str(path) + ": missing correctness guard " + label)
    for row in correctness["checks"]:
        require(
            isinstance(row["max_abs_error"], (int, float))
            and math.isfinite(row["max_abs_error"])
            and row["max_abs_error"] >= 0,
            str(path),
        )
    if mode == "benchmark":
        require(
            [row["name"] for row in data["measurements"]] == expected_order, str(path)
        )
        for row in data["measurements"]:
            require(
                {key: row[key] for key in ("path", "tactic")} == VARIANTS[row["name"]],
                str(path),
            )
            for execution in ("eager", "cuda_graph"):
                validate_measurement(
                    row[execution],
                    protocol,
                    str(path) + ": " + row["name"] + "/" + execution,
                )
    return data


def validate_native(path, observer):
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    phases = collections.defaultdict(list)
    for row in records:
        require(
            row["event"] in ("phase_begin", "phase_summary", "heuristic", "matmul"),
            "unknown native event",
        )
        phases[row["phase"]].append(row)
    result = {}
    expected_phases = {
        name + "_" + execution
        for name in VARIANTS
        for execution in ("eager", "capture", "graph_replay")
    }
    require(set(phases) == expected_phases, "missing or extra native phases")
    observations = {row["name"]: row for row in observer["observations"]}
    require(set(observations) == set(VARIANTS), "missing observation variants")
    for phase, rows in phases.items():
        require(
            rows[0]["event"] == "phase_begin" and rows[-1]["event"] == "phase_summary",
            phase,
        )
        require(sum(row["event"] == "phase_begin" for row in rows) == 1, phase)
        require(sum(row["event"] == "phase_summary" for row in rows) == 1, phase)
        require(len({row["pid"] for row in rows}) == 1, phase)
        summary = rows[-1]["apis"]
        heuristic = [row for row in rows if row["event"] == "heuristic"]
        matmul = [row for row in rows if row["event"] == "matmul"]
        name, execution = next(
            (name, execution)
            for name in VARIANTS
            for execution in ("eager", "capture", "graph_replay")
            if phase == name + "_" + execution
        )
        variant = VARIANTS[name]
        observation = observations[name]
        phase_observation = [
            row for row in observation["phase_boundaries"] if row["phase"] == phase
        ]
        require(len(phase_observation) == 1, phase + ": Python/native phase mismatch")
        expected_calls = (
            1 if execution == "capture" else observation["calls_per_execution"]
        )
        require(expected_calls == (1 if execution == "capture" else 5), phase)
        require(phase_observation[0]["calls"] == expected_calls, phase)
        boundaries = phase_observation[0]["cpu_boundaries"]
        for boundary in boundaries.values():
            require(boundary["failed_calls"] == 0, phase + ": Python boundary failed")
            require(
                boundary["calls"]
                == len(boundary["inclusive_samples_ns"])
                == len(boundary["exclusive_samples_ns"]),
                phase,
            )
            require(
                boundary["inclusive_total_ns"] == sum(boundary["inclusive_samples_ns"]),
                phase,
            )
            require(
                boundary["exclusive_total_ns"] == sum(boundary["exclusive_samples_ns"]),
                phase,
            )
            require(
                all(
                    0 <= exclusive <= inclusive
                    for exclusive, inclusive in zip(
                        boundary["exclusive_samples_ns"],
                        boundary["inclusive_samples_ns"],
                    )
                ),
                phase,
            )
        expected_python = {}
        if execution != "graph_replay" and variant["path"] != "retained_runner":
            expected_python.update(
                runner_factory_including_class_definition_and_init=expected_calls,
                tuner_choose_one=expected_calls,
            )
            if variant["tactic"] == 0:
                expected_python["runner_get_algorithms"] = expected_calls
                if variant["path"] == "public_cached":
                    expected_python["native_binding_get_algorithms"] = expected_calls
        require(
            {name: record["calls"] for name, record in boundaries.items()}
            == expected_python,
            phase + ": Python boundary coverage",
        )
        expected_matmul = 0 if execution == "graph_replay" else expected_calls
        expected_heuristic = (
            expected_matmul
            if variant["tactic"] == -1 or variant["path"] == "public_cached"
            else 0
        )
        require(
            len(matmul) == expected_matmul,
            phase + ": native matmul interception coverage",
        )
        require(
            len(heuristic) == expected_heuristic,
            phase + ": native heuristic interception coverage",
        )
        requested = 1 if variant["tactic"] == -1 else 100
        require(
            all(row["requested"] == requested for row in heuristic),
            phase + ": wrong heuristic request size",
        )
        for key, values in (("heuristic", heuristic), ("matmul", matmul)):
            require(summary[key]["calls"] == len(values), phase + ": API count")
            require(
                summary[key]["host_ns"] == sum(row["host_ns"] for row in values),
                phase + ": API duration",
            )
        require(
            all(api["failures"] == 0 for api in summary.values()),
            phase + ": native API failures",
        )
        for row in heuristic:
            require(row["status"] == 0 and row["returned"] > 0, phase)
            require(
                len(row["algorithms"]) == row["returned"] <= row["requested"], phase
            )
            require(
                [algo["ordinal"] for algo in row["algorithms"]]
                == list(range(row["returned"])),
                phase,
            )
            require(all(algo["state"] == 0 for algo in row["algorithms"]), phase)
        for row in matmul:
            require(row["status"] == 0 and row["algo"] is not None, phase)
        if heuristic:
            for query, execution_record in zip(heuristic, matmul):
                require(
                    query["algorithms"][0]["algo"] == execution_record["algo"],
                    phase + ": submitted algorithm differs from first heuristic result",
                )
        for algorithm in [row["algo"] for row in matmul] + [
            algo["algo"] for row in heuristic for algo in row["algorithms"]
        ]:
            require(len(algorithm["raw_hex"]) == 2 * algorithm["raw_bytes"], phase)
            bytes.fromhex(algorithm["raw_hex"])
            for attr in algorithm["config"].values():
                require(
                    attr["status"] == 0 and attr["bytes"] == 4,
                    phase + ": unsupported algorithm attribute",
                )
        result[phase] = {
            "pid": rows[0]["pid"],
            "heuristic_requests": [row["requested"] for row in heuristic],
            "heuristic_returned": [row["returned"] for row in heuristic],
            "matmul_calls": len(matmul),
            "submitted_algorithms": [row["algo"] for row in matmul],
            "api_summary": summary,
        }
    return result


def validate_traces(observer, trace_root):
    require(len(observer["traces"]) == 6, "six CPU/CUDA traces required")
    require(
        {row["variant"] for row in observer["traces"]} == set(VARIANTS),
        "trace variant mismatch",
    )
    result = {}
    for record in observer["traces"]:
        name = record["variant"]
        candidates = list(trace_root.rglob(Path(record["path"]).name))
        require(len(candidates) == 1, "missing or duplicate trace: " + name)
        path = candidates[0]
        require(sha256(path) == record["sha256"], "trace hash mismatch: " + name)
        with gzip.open(path, "rt") as source:
            trace = json.load(source)
        events = trace["traceEvents"]
        scopes = [row for row in events if row.get("cat") == "user_annotation"]
        require(
            all(
                any(row.get("name") == name + "/" + execution for row in scopes)
                for execution in ("eager", "graph_replay")
            ),
            "missing trace scopes: " + name,
        )
        kernels = [row for row in events if row.get("cat") == "kernel"]
        require(len(kernels) >= 10, "missing CUDA kernel trace events: " + name)
        require(
            all(row.get("dur", 0) > 0 for row in kernels),
            "invalid CUDA kernel durations",
        )
        result[name] = {
            "sha256": record["sha256"],
            "kernel_events": len(kernels),
            "kernel_names": dict(collections.Counter(row["name"] for row in kernels)),
            "scope": "instrumented diagnostic kernel events; names alone do not establish equal kernel configuration",
        }
    return result


def algorithm_comparisons(native):
    result = []
    for execution in ("eager", "capture"):
        phases = {
            name: native[name + "_" + execution]["submitted_algorithms"]
            for name in VARIANTS
        }
        reference = phases["public_cached_default"][0]
        for name, algorithms in phases.items():
            result.append(
                {
                    "execution": execution,
                    "variant": name,
                    "calls": len(algorithms),
                    "same_observed_configuration_as_public_default": all(
                        algo["config"] == reference["config"] for algo in algorithms
                    ),
                    "same_opaque_bytes_as_public_default": all(
                        algo["raw_hex"] == reference["raw_hex"] for algo in algorithms
                    ),
                    "unique_observed_configurations": len(
                        {
                            json.dumps(algo["config"], sort_keys=True)
                            for algo in algorithms
                        }
                    ),
                    "unique_opaque_byte_strings": len(
                        {algo["raw_hex"] for algo in algorithms}
                    ),
                }
            )
    return result


def validate_containers(path, worker_paths, workers):
    """Docker PIDs are namespace-local; separate container IDs prove freshness."""
    events = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    result = []
    for worker_path, worker in zip(worker_paths, workers):
        expected_stem = (
            "observation"
            if worker["mode"] == "observe"
            else "benchmark-" + str(worker["seed"])
        )
        require(worker_path.stem == expected_stem, "worker filename/mode/seed mismatch")
        matching = [
            event
            for event in events
            if event.get("Type", "container") == "container"
            and event.get("Actor", {})
            .get("Attributes", {})
            .get("name", "")
            .lstrip("/")
            .endswith("-" + expected_stem)
        ]
        ids = {event.get("Actor", {}).get("ID", event.get("id")) for event in matching}
        require(
            len(ids) == 1 and None not in ids,
            "missing or ambiguous container identity: " + expected_stem,
        )
        container_id = next(iter(ids))
        require(len(container_id) == 64, "full Docker container ID required")
        actions = collections.defaultdict(list)
        for event in matching:
            actions[event.get("Action", event.get("status"))].append(event)
        require(
            all(len(actions[action]) == 1 for action in ("create", "start", "die")),
            "container lifecycle missing or repeated: " + expected_stem,
        )
        require(
            str(actions["die"][0]["Actor"]["Attributes"].get("exitCode")) == "0",
            "worker container did not exit successfully",
        )
        timestamps = [
            actions[action][0].get("timeNano", actions[action][0].get("time"))
            for action in ("create", "start", "die")
        ]
        require(
            all(isinstance(value, int) for value in timestamps)
            and timestamps == sorted(timestamps),
            "container lifecycle timestamps invalid",
        )
        result.append(
            {
                "worker": expected_stem,
                "mode": worker["mode"],
                "seed": worker["seed"],
                "container_id": container_id,
                "namespace_pid": worker["pid"],
                "create_start_die_timestamps": timestamps,
                "exit_code": 0,
            }
        )
    require(
        len({row["container_id"] for row in result}) == len(workers) == 4,
        "worker containers are not distinct",
    )
    return result


def paired_comparisons(workers):
    comparisons = (
        ("public_cached_ordinal0", "public_cached_default"),
        ("public_seeded_algorithms_ordinal0", "public_cached_ordinal0"),
        ("public_seeded_algorithms_default", "public_cached_default"),
        ("retained_runner_ordinal0", "retained_runner_default"),
        ("public_seeded_algorithms_ordinal0", "retained_runner_ordinal0"),
    )
    output = []
    for execution in ("eager", "cuda_graph"):
        for numerator, denominator in comparisons:
            rows = []
            for worker in sorted(workers, key=lambda row: row["seed"]):
                variants = {row["name"]: row for row in worker["measurements"]}
                a = statistics.median(variants[numerator][execution]["samples_ms"])
                b = statistics.median(variants[denominator][execution]["samples_ms"])
                rows.append(
                    {
                        "seed": worker["seed"],
                        "numerator_median_ms": a,
                        "denominator_median_ms": b,
                        "ratio": a / b,
                    }
                )
            output.append(
                {
                    "execution": execution,
                    "numerator": numerator,
                    "denominator": denominator,
                    "trial_units": len(rows),
                    "paired_trials": rows,
                    "median_paired_ratio": statistics.median(
                        row["ratio"] for row in rows
                    ),
                    "min_paired_ratio": min(row["ratio"] for row in rows),
                    "max_paired_ratio": max(row["ratio"] for row in rows),
                }
            )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, action="append", required=True)
    parser.add_argument("--observe", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--containers", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "status": "error",
        "errors": [],
        "interpretation_limits": [
            "Three fresh workers are the trial units; within-worker timing samples are not independent trials.",
            "Fixed-workload cache seeding is a diagnostic intervention, not a production cache implementation.",
            "Retained-runner comparisons remove several API/setup layers together.",
            "Instrumented API durations are perturbed and nested; they must not be summed with Python boundary timings or substituted for uninstrumented latency.",
            "Matching observed algorithm bytes/configuration does not by itself establish identical physical GPU kernels.",
            "One small FP8 operation is not a model-serving performance result.",
        ],
    }
    try:
        worker_source = subprocess.check_output(
            ["git", "show", args.expected_source_commit + ":" + WORKER_PATH],
            cwd=args.source_root,
        )
        source_hash = hashlib.sha256(worker_source).hexdigest()
        workers = [
            validate_worker(path, args.expected_source_commit, source_hash, "benchmark")
            for path in args.benchmark
        ]
        require(
            len(workers) == 3
            and {worker["seed"] for worker in workers} == {17, 18, 19},
            "three seeds17–19 required",
        )
        observer = validate_worker(
            args.observe, args.expected_source_commit, source_hash, "observe"
        )
        all_workers = workers + [observer]
        require(observer["seed"] == 17, "observation seed17 required")
        reference_worker = next(worker for worker in workers if worker["seed"] == 17)
        require(
            observer["intervention"]["operand_sha256"]
            == reference_worker["intervention"]["operand_sha256"],
            "observer and benchmark seed17 inputs differ",
        )
        containers = validate_containers(
            args.containers, args.benchmark + [args.observe], all_workers
        )
        for key in (
            "physical_gpu_uuid",
            "flashinfer_gemm_source_sha256",
            "flashinfer_tuner_sha256",
            "torch",
            "torch_cuda",
            "flashinfer",
            "sglang_commit",
        ):
            require(
                len({worker["environment"][key] for worker in all_workers}) == 1,
                "environment mismatch: " + key,
            )
        native = validate_native(args.native, observer)
        require(
            native and {row["pid"] for row in native.values()} == {observer["pid"]},
            "native observer PID mismatch",
        )
        traces = validate_traces(observer, args.trace_root)
        result.update(
            status="success",
            source_commit=args.expected_source_commit,
            worker_sha256=source_hash,
            worker_pids=[worker["pid"] for worker in all_workers],
            worker_containers=containers,
            physical_gpu_uuid=workers[0]["environment"]["physical_gpu_uuid"],
            comparisons=paired_comparisons(workers),
            native_phases=native,
            algorithm_comparisons=algorithm_comparisons(native),
            trace_summary=traces,
            input_sha256={
                str(path): sha256(path)
                for path in args.benchmark
                + [args.observe, args.native, args.containers]
            },
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        OSError,
        subprocess.SubprocessError,
    ) as error:
        result["errors"].append(f"{type(error).__name__}: {error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
