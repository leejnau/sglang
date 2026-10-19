#!/usr/bin/env python3
"""Validate and describe a FlashInfer selection batch without pooled inference.

Only the standard library is used. Inputs are preserved; this script writes only
the requested analysis JSON. Each case is a downloaded coordinator output dir.
"""

import argparse
import collections
import hashlib
import json
import math
import random
import re
import statistics
import subprocess
import sys
from pathlib import Path

ARMS = ("default", "v1", "v2")
PHASES = ("cold", "warm")
EXECUTIONS = ("eager", "cuda_graph")
TUNER_SHA = "e222f35f8b5a59a99a45fad627a79db8fc45fadc52b667e63b70c0968509109d"
SOURCE_FILES = {
    "coordinator": "test/manual/kernels/bench_flashinfer_autotune_selection.py",
    "worker": "test/manual/kernels/flashinfer_autotune_selection_worker.py",
    "wrapper": "python/sglang/srt/model_executor/runner/flashinfer_autotune.py",
}
IDENTITY_FIELDS = (
    "sglang_commit",
    "torch",
    "torch_cuda",
    "flashinfer",
    "gpu",
    "capability",
    "backend",
    "sglang_wrapper_sha256",
    "flashinfer_tuner_sha256",
    "timer_env",
    "global_timer_enabled",
    "startup_timing_config",
    "measurement_policy",
    "CUDA_VISIBLE_DEVICES",
    "selected_device_ordinal",
    "physical_gpu_uuid",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value, positive=False):
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


def load_json(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    def walk(value):
        if isinstance(value, float):
            require(math.isfinite(value), "nonfinite JSON number")
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    value = json.loads(Path(path).read_text(), parse_constant=reject)
    walk(value)
    return value


def stats(values):
    require(bool(values) and all(finite(x) for x in values), "invalid statistic values")
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "sample_stdev": statistics.stdev(values) if len(values) > 1 else None,
    }


def source_identity(args):
    require(
        re.fullmatch(r"[0-9a-f]{40}", args.expected_source_commit) is not None,
        "expected source commit must be a full lowercase SHA",
    )
    blobs = {}
    for name, path in SOURCE_FILES.items():
        blobs[name] = subprocess.check_output(
            [
                "git",
                "-C",
                str(args.source_root),
                "show",
                f"{args.expected_source_commit}:{path}",
            ],
            stderr=subprocess.PIPE,
        )
    require(
        b"if profile_calls != startup_profile_calls:" in blobs["worker"],
        "expected worker lacks the post-startup profiling-count guard",
    )
    require(
        b"disallow_profiling = True" in blobs["worker"],
        "expected worker lacks the post-startup profiling prohibition",
    )
    return {
        "source_commit": args.expected_source_commit,
        **{
            f"{name}_sha256": hashlib.sha256(blob).hexdigest()
            for name, blob in blobs.items()
        },
        "flashinfer_tuner_sha256": TUNER_SHA,
    }


def measurement(value, protocol, samples):
    require(isinstance(value, dict), "missing measurement")
    requested, maximum = protocol["requested_iterations"], protocol["max_iterations"]
    effective = value.get("effective_iterations")
    require(
        type(effective) is int and requested <= effective <= maximum,
        "invalid effective iteration count",
    )
    require(
        value.get("requested_iterations") == requested
        and value.get("iterations_per_sample") == effective,
        "measurement iterations do not match protocol",
    )
    warmup = value.get("warmup_actual_ms")
    calls, batches = value.get("warmup_calls"), value.get("warmup_batches")
    require(finite(warmup) and warmup >= protocol["warmup_ms"], "warmup budget unmet")
    require(
        type(batches) is int
        and batches > 0
        and type(calls) is int
        and calls == requested * batches,
        "warmup counts inconsistent",
    )
    calibration = value.get("calibration_batches")
    require(isinstance(calibration, list) and bool(calibration), "missing calibration")
    expected = requested
    for index, batch in enumerate(calibration):
        require(
            batch.get("iterations") == expected and finite(batch.get("wall_ms")),
            "invalid calibration sequence",
        )
        if index == len(calibration) - 1:
            require(
                expected == effective and batch["wall_ms"] >= protocol["min_sample_ms"],
                "calibration target unmet",
            )
        else:
            require(
                batch["wall_ms"] < protocol["min_sample_ms"] and expected < maximum,
                "calibration continued beyond its stop condition",
            )
            expected = min(expected * 2, maximum)
    for normalized, raw in (
        ("samples_ms", "wall_batch_ms"),
        ("gpu_event_samples_ms", "gpu_event_batch_ms"),
    ):
        a, b = value.get(normalized), value.get(raw)
        require(
            isinstance(a, list)
            and isinstance(b, list)
            and len(a) == len(b) == samples
            and all(finite(x, positive=True) for x in a + b),
            "invalid raw samples",
        )
        require(
            all(
                math.isclose(x, y / effective, rel_tol=1e-9, abs_tol=1e-12)
                for x, y in zip(a, b)
            ),
            "sample normalization mismatch",
        )
    require(
        math.isclose(
            value.get("median_ms", -1),
            statistics.median(value["samples_ms"]),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ),
        "reported median mismatch",
    )
    return {
        "wall_ms": stats(value["samples_ms"]),
        "gpu_event_ms": stats(value["gpu_event_samples_ms"]),
        "effective_iterations": effective,
        "warmup_actual_ms": warmup,
        "calibration_final_batch_ms": calibration[-1]["wall_ms"],
        "measured_wall_batch_ms": stats(value["wall_batch_ms"]),
    }


def ranks(entries, winner):
    """Ranks and regret refer only to the forced-memory diagnostic sweep."""
    output = {}
    for execution in EXECUTIONS:
        values = {
            item["tactic"]: statistics.median(item[execution]["samples_ms"])
            for item in entries
        }
        best = min(values.values())
        rank = lambda tactic: 1 + sum(x < values[tactic] for x in values.values())
        output[execution] = {
            "candidate_medians_ms": {str(t): values[t] for t in sorted(values)},
            "candidate_ranks": {str(t): rank(t) for t in sorted(values)},
            "best_tactics": sorted(t for t, value in values.items() if value == best),
            "selected_tactic": winner,
            "selected_tactic_rank": rank(winner),
            "selected_tactic_to_best_ratio": values[winner] / best,
            "selected_tactic_diagnostic_regret_fraction": values[winner] / best - 1,
            "selected_tactic_diagnostic_regret_percent": (values[winner] / best - 1)
            * 100,
            "selected_tactic_diagnostic_regret_ms": values[winner] - best,
            "default_tactic_rank": rank(-1),
            "default_tactic_to_best_ratio": values[-1] / best,
            "default_tactic_diagnostic_regret_fraction": values[-1] / best - 1,
            "default_tactic_diagnostic_regret_percent": (values[-1] / best - 1) * 100,
        }
    return output


def profile_comparison(result, winner):
    profiles = result["candidate_profiles"]
    runtime_m = result["shape"][0]
    groups = collections.defaultdict(list)
    for item in profiles:
        groups[item["input_shapes"][0][1]].append(item)
    actual = groups.get(runtime_m, [])
    values = {
        item["tactic"]: item["profiler_ms"]
        for item in actual
        if finite(item.get("profiler_ms")) and "error" not in item
    }
    output = {
        "profile_input_m": sorted(groups),
        "runtime_profile_input_m": runtime_m,
        "profile_calls": result["profile_calls"],
        "finite_profiles": sum(
            finite(x.get("profiler_ms")) and "error" not in x for x in profiles
        ),
        "failed_or_nonfinite_profiles": sum(
            "error" in x or not finite(x.get("profiler_ms")) for x in profiles
        ),
        "default_profile_count": sum(x["tactic"] == -1 for x in profiles),
        "default_profiles_by_input_m": [
            {
                "input_m": m,
                "attempts": sum(x["tactic"] == -1 for x in group),
                "finite_successes": sum(
                    x["tactic"] == -1
                    and finite(x.get("profiler_ms"))
                    and "error" not in x
                    for x in group
                ),
            }
            for m, group in sorted(groups.items())
        ],
        "runtime_candidate_profile_ms": {str(t): values[t] for t in sorted(values)},
        "selected_tactic": winner,
        "profile_errors": [
            {
                **{k: x.get(k) for k in ("tactic", "input_shapes", "error")},
                "reason": x.get("error") or "no finite nonnegative profiler_ms",
            }
            for x in profiles
            if "error" in x or not finite(x.get("profiler_ms"))
        ],
    }
    if values:
        best = min(values.values())
        output.update(
            {
                "profile_min_tactics": sorted(
                    t for t, value in values.items() if value == best
                ),
                "selected_tactic_profile_ms": values.get(winner),
                "selected_tactic_matches_profile_min": values.get(winner) == best,
                "default_runtime_profile_ms": values.get(-1),
                "default_runtime_profile_rank": (
                    1 + sum(x < values[-1] for x in values.values())
                    if -1 in values
                    else None
                ),
            }
        )
    return output


def validate_worker(record, base, plan, source, args):
    require(
        record.get("status") == "success" and record.get("returncode") == 0,
        f"coordinator worker failed: {record.get('errors', [])}",
    )
    result = record.get("result")
    require(
        isinstance(result, dict) and result.get("status") == "success",
        f"worker result failed: {result.get('error') if isinstance(result, dict) else result}",
    )
    trial, arm, phase = record["trial"], record["arm"], record["phase"]
    require(
        type(trial) is int
        and 1 <= trial <= args.expected_trials
        and arm in ARMS
        and phase in PHASES,
        "unexpected worker identity",
    )
    expected_seed = args.expected_seed + trial - 1
    for key, expected in (
        ("arm", arm),
        ("phase", phase),
        ("seed", expected_seed),
        ("shape", plan["shape"]),
    ):
        require(
            result.get(key) == record.get(key) == expected, f"worker {key} mismatch"
        )
    require(
        type(result.get("pid")) is int
        and result["pid"] > 0
        and result["pid"] == record.get("process_pid"),
        "fresh worker PID mismatch",
    )
    directory = base / f"trial_{trial:03d}" / arm
    file = directory / f"{phase}.json"
    log = directory / f"{phase}.log"
    require(load_json(file) == result, "raw worker JSON differs from combined record")
    require(log.is_file(), "missing raw worker log")
    require(
        result.get("instrumentation_sha256") == source["worker_sha256"],
        "worker source hash mismatch",
    )
    environment = result["environment"]
    require(
        all(key in environment for key in IDENTITY_FIELDS),
        "missing environment identity",
    )
    require(
        isinstance(environment.get("physical_gpu_uuid"), str)
        and bool(environment["physical_gpu_uuid"].strip()),
        "missing physical GPU UUID",
    )
    for key, expected in (
        ("sglang_commit", source["source_commit"]),
        ("sglang_wrapper_sha256", source["wrapper_sha256"]),
        ("flashinfer_tuner_sha256", source["flashinfer_tuner_sha256"]),
        ("flashinfer", "0.7.2"),
        ("torch_cuda", "13.0"),
        ("backend", "cublas"),
    ):
        require(environment.get(key) == expected, f"environment {key} mismatch")
    require(environment["torch"].split("+")[0] == "2.14.1", "Torch version mismatch")
    require(
        environment["capability"] == [9, 0] and "H100" in environment["gpu"],
        "expected an H100/SM90 GPU",
    )
    require(
        environment.get("sglang_wrapper_source")
        == "/workspace/python/sglang/srt/model_executor/runner/flashinfer_autotune.py",
        "wrapper imported outside the staged source overlay",
    )
    protocol = plan["timing_protocol"]
    require(
        result.get("timing_protocol") == record.get("timing_protocol") == protocol,
        "worker timing protocol mismatch",
    )
    require(finite(result.get("startup_ms"), True), "invalid startup time")
    profiles = result.get("candidate_profiles")
    calls = result.get("profile_calls")
    require(
        type(calls) is int
        and calls >= 0
        and isinstance(profiles, list)
        and calls == len(profiles),
        "profiling counter/records mismatch",
    )
    require(
        (calls == 0 if arm == "default" or phase == "warm" else calls > 0),
        "unexpected default/warm profiling or absent cold profiling",
    )
    if arm == "v1":
        require(
            all(x["tactic"] != -1 for x in profiles), "v1 profiled the default tactic"
        )
    if arm == "v2" and phase == "cold":
        groups = collections.defaultdict(list)
        for profile in profiles:
            groups[profile["input_shapes"][0][1]].append(profile)
        require(plan["shape"][0] in groups, "runtime max shape was not profiled")
        require(
            any(
                x["tactic"] == -1 and finite(x.get("profiler_ms")) and "error" not in x
                for x in groups[plan["shape"][0]]
            ),
            "v2 default lacked a finite profile at the runtime max-profile input",
        )
    eligible = result["runtime_eligible_tactics"]
    require(
        isinstance(eligible, list)
        and eligible
        and all(type(t) is int and t >= 0 for t in eligible)
        and len(eligible) == len(set(eligible)),
        "invalid eligible runtime ordinals",
    )
    selections = result["selections"]
    natural = []
    for stage in ("startup", "bare", "capture"):
        items = [x for x in selections if x.get("stage") == stage]
        require(len(items) == 1, f"expected one {stage} selection")
        item = items[0]
        require(
            item.get("operation") == "fp8_gemm"
            and item.get("runner") == "CublasFp8GemmRunner",
            "unexpected operation/runner",
        )
        require(
            type(item.get("tactic")) is int and item["tactic"] in [-1, *eligible],
            "selected runtime ordinal would silently fall back",
        )
        require(
            item.get("cache_hit") is (arm != "default"), "wrong natural lookup hit/miss"
        )
        require(item.get("capturing") is (stage == "capture"), "wrong capture state")
        if stage == "startup":
            require(
                item.get("on_forward_stream") and not item.get("on_caller_stream"),
                "startup did not execute on the forward stream",
            )
        elif stage == "bare":
            require(
                item.get("on_caller_stream") and not item.get("on_forward_stream"),
                "bare call did not execute on the caller stream",
            )
        natural.append(item["tactic"])
    require(
        len(set(natural)) == 1 and (arm != "default" or natural[0] == -1),
        "natural startup/bare/capture tactics differ",
    )
    winner = natural[0]
    oracle = result.get("oracle")
    require(
        isinstance(oracle, list)
        and sorted(x["tactic"] for x in oracle) == sorted([-1, *eligible]),
        "oracle did not cover every runtime candidate exactly once",
    )
    expected_order = [-1, *eligible]
    random.Random(expected_seed).shuffle(expected_order)
    require(
        [x["tactic"] for x in oracle] == expected_order,
        "oracle order is not the recorded seeded order",
    )
    for stage in ("oracle_check", "oracle_capture"):
        items = [x for x in selections if x.get("stage") == stage]
        require(
            [x["tactic"] for x in items] == expected_order,
            f"{stage} candidate sequence mismatch",
        )
        require(
            all(
                x.get("cache_hit") is True
                and x.get("capturing") is (stage == "oracle_capture")
                for x in items
            ),
            "forced oracle candidate did not use a real cached lookup",
        )
    correctness = result.get("correctness", {})
    require(
        correctness.get("passed") is True
        and finite(correctness.get("max_abs_error"))
        and correctness.get("rtol") == 0.02
        and correctness.get("atol") == 0.125,
        "numerical check failed or changed tolerance",
    )
    require(
        all(
            correctness.get(k) == 1 + len(oracle)
            for k in ("eager_checks_after_measurement", "changed_input_graph_checks")
        ),
        "natural/oracle eager and changed-input graph checks incomplete",
    )
    natural_measurements = {
        key: measurement(result["measurements"][key], protocol, args.expected_samples)
        for key in EXECUTIONS
    }
    timing_sets = {key: [natural_measurements[key]] for key in EXECUTIONS}
    for item in oracle:
        for key in EXECUTIONS:
            timing_sets[key].append(
                measurement(item[key], protocol, args.expected_samples)
            )
    timing_blocks = [item for values in timing_sets.values() for item in values]
    profile = profile_comparison(result, winner)
    return {
        "trial": trial,
        "arm": arm,
        "phase": phase,
        "seed": expected_seed,
        "status": "success",
        "pid": result["pid"],
        "startup_ms": result["startup_ms"],
        "natural_tactic": winner,
        "natural_cache_hit": arm != "default",
        "candidate_count": {
            "qualification_expected": 9,
            "runtime_observed": len(oracle),
            "deviation": len(oracle) != 9,
        },
        "correctness": correctness,
        "measurements": natural_measurements,
        "timing_validation": {
            "measurement_blocks": len(timing_blocks),
            "warmup_actual_ms": stats([x["warmup_actual_ms"] for x in timing_blocks]),
            "calibration_final_batch_ms": stats(
                [x["calibration_final_batch_ms"] for x in timing_blocks]
            ),
            "effective_iterations": {
                key: sorted({x["effective_iterations"] for x in values})
                for key, values in timing_sets.items()
            },
        },
        "profiling": profile,
        "oracle_forced_memory_diagnostic": ranks(oracle, winner),
        "post_startup_profiling_guard": "passed by successful audited worker",
        "artifacts": [
            {"path": str(path.relative_to(base)), "sha256": sha(path)}
            for path in (file, log)
        ],
    }, result


def analyze_case(base, source, args):
    output = {
        "case": base.name,
        "workers": [],
        "cache_pairs": [],
        "errors": [],
        "warnings": [],
    }
    raw = {}
    try:
        path = base / "results.json"
        report = load_json(path)
        plan = report["plan"]
        output.update(
            {
                "raw_results_sha256": sha(path),
                "shape": plan["shape"],
                "observed_worker_records": len(report.get("runs", [])),
                "raw_started_at": report.get("started_at"),
                "raw_finished_at": report.get("finished_at"),
                "timing_protocol": plan.get("timing_protocol"),
                "cache_policy": plan.get("cache_policy"),
            }
        )
        require(
            report.get("status") == "success",
            f"coordinator status: {report.get('status')}",
        )
        require(
            plan["shape"] in [[m, 1024, 1024] for m in args.expected_ms],
            "unexpected shape",
        )
        require(
            plan["trials"] == args.expected_trials
            and plan["samples"] == args.expected_samples
            and plan["seed"] == args.expected_seed
            and plan.get("oracle") is True,
            "case trial/sample/seed/oracle plan mismatch",
        )
        protocol = plan["timing_protocol"]
        require(
            protocol["warmup_ms"] == 500
            and protocol["min_sample_ms"] == 20
            and protocol["requested_iterations"] == plan["iterations"]
            and type(plan["iterations"]) is int
            and plan["iterations"] >= args.min_iterations
            and protocol["max_iterations"] == 65536,
            "case timing protocol mismatch",
        )
        require(
            plan["instrumentation_sha256"]
            == {key: source[f"{key}_sha256"] for key in ("coordinator", "worker")},
            "plan source hashes differ from the expected Git source",
        )
        order = list(ARMS)
        random.Random(args.expected_seed).shuffle(order)
        expected_jobs = []
        for trial in range(1, args.expected_trials + 1):
            offset = (trial - 1) % len(ARMS)
            for arm in order[offset:] + order[:offset]:
                expected_jobs.extend((trial, arm, phase) for phase in PHASES)
        job_key = lambda x: (x.get("trial"), x.get("arm"), x.get("phase"))
        require(
            [job_key(x) for x in plan["jobs"]] == expected_jobs,
            "planned randomized/rotated order mismatch",
        )
        require(
            [job_key(x) for x in report["runs"]] == expected_jobs,
            "actual worker order or completeness mismatch",
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        output["errors"].append(str(error))
        # Keep rejected coordinator/worker error text even when plan validation fails.
        if "report" in locals():
            output["raw_failures"] = [
                {
                    "trial": x.get("trial"),
                    "arm": x.get("arm"),
                    "phase": x.get("phase"),
                    "status": x.get("status"),
                    "returncode": x.get("returncode"),
                    "errors": x.get("errors"),
                    "worker_error": (x.get("result") or {}).get("error"),
                }
                for x in report.get("runs", [])
                if x.get("status") != "success"
            ]
        return output, raw
    identities = []
    for record in report["runs"]:
        try:
            worker, result = validate_worker(record, base, plan, source, args)
            raw[(worker["trial"], worker["arm"], worker["phase"])] = result
            output["workers"].append(worker)
            identities.append(
                {field: result["environment"][field] for field in IDENTITY_FIELDS}
            )
            if worker["profiling"]["profile_errors"]:
                output["warnings"].append(
                    {
                        "trial": worker["trial"],
                        "arm": worker["arm"],
                        "phase": worker["phase"],
                        "message": "candidate profile errors retained",
                    }
                )
            if worker["candidate_count"]["deviation"]:
                output["warnings"].append(
                    {
                        "trial": worker["trial"],
                        "arm": worker["arm"],
                        "phase": worker["phase"],
                        "message": "runtime candidate count differs from nine qualified candidates",
                        "candidate_count": worker["candidate_count"],
                    }
                )
            if (
                worker["arm"] != "default"
                and worker["phase"] == "cold"
                and worker["profiling"].get("selected_tactic_matches_profile_min")
                is not True
            ):
                output["warnings"].append(
                    {
                        "trial": worker["trial"],
                        "arm": worker["arm"],
                        "phase": worker["phase"],
                        "message": "selected tactic differs from minimum finite runtime profile; retained as a selection diagnostic",
                    }
                )
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            AttributeError,
        ) as error:
            failed = {
                "trial": record.get("trial"),
                "arm": record.get("arm"),
                "phase": record.get("phase"),
                "status": "error",
                "error": str(error),
                "returncode": record.get("returncode"),
                "coordinator_errors": record.get("errors"),
                "worker_error": (record.get("result") or {}).get("error"),
            }
            output["workers"].append(failed)
            output["errors"].append(failed)
    if identities:
        require_same = len({canonical(x) for x in identities}) == 1
        if not require_same:
            output["errors"].append("environment identity changed within the case")
        identity = identities[0]
        output["environment"] = {
            k: v
            for k, v in identity.items()
            if k not in ("physical_gpu_uuid", "CUDA_VISIBLE_DEVICES")
        }
        output["environment"]["same_physical_gpu_across_workers"] = (
            all(
                isinstance(x["physical_gpu_uuid"], str)
                and bool(x["physical_gpu_uuid"].strip())
                for x in identities
            )
            and len({x["physical_gpu_uuid"] for x in identities}) == 1
        )
        drivers, clocks = set(), []
        for result in raw.values():
            for key in ("gpu_status_before", "gpu_status_after"):
                for line in result["environment"].get(key, "").splitlines():
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts) >= 5:
                        drivers.add(parts[2])
                        match = re.fullmatch(r"([0-9.]+) MHz", parts[4])
                        if match:
                            clocks.append(float(match[1]))
        output["environment"]["driver_versions"] = sorted(drivers)
        output["environment"]["observed_sm_clocks_mhz"] = (
            stats(clocks) if clocks else None
        )
        if len(drivers) != 1:
            output["errors"].append(
                "one consistent GPU driver version was not recorded"
            )
        output["_identity"] = identity
    pids = [x["pid"] for x in output["workers"] if x["status"] == "success"]
    if len(pids) != len(set(pids)):
        output["errors"].append("worker PIDs repeated within a case")
    for trial in range(1, args.expected_trials + 1):
        for arm in ARMS:
            pair = {phase: raw.get((trial, arm, phase)) for phase in PHASES}
            if not all(pair.values()):
                continue
            cold, warm = pair["cold"], pair["warm"]
            try:
                directory = base / f"trial_{trial:03d}" / arm
                path = directory / "tactics.json"
                actual = sha(path)
                record = next(
                    x for x in report["runs"] if job_key(x) == (trial, arm, "warm")
                )
                require(cold["cache_before_sha256"] is None, "cold worker had a cache")
                require(
                    cold["cache_sha256"]
                    == warm["cache_before_sha256"]
                    == warm["cache_sha256"]
                    == record["cache_sha256_before_worker"]
                    == actual,
                    "saved/warm/oracle/final cache hashes differ",
                )
                require(cold["pid"] != warm["pid"], "cold and warm reused a process")
                bare = lambda x: next(
                    z for z in x["selections"] if z["stage"] == "bare"
                )
                require(
                    bare(cold)["tactic"] == bare(warm)["tactic"],
                    "winner changed on warm load",
                )
                cache = load_json(path)
                records = [(k, v) for k, v in cache.items() if not k.startswith("_")]
                persisted = None
                if arm == "default":
                    require(not records, "untuned baseline saved tactic records")
                else:
                    bucket = 1 << (plan["shape"][0] - 1).bit_length()
                    candidates = [
                        (k, v)
                        for k, v in records
                        if k.startswith("('fp8_gemm', 'CublasFp8GemmRunner',")
                        and f"((1, {bucket}, 1024), (1, 1024, 1024)," in k
                    ]
                    require(
                        len(candidates) == 1
                        and candidates[0][1][1] == bare(cold)["tactic"],
                        "persisted runtime bucket winner mismatch",
                    )
                    persisted = {
                        "lookup_bucket_m": bucket,
                        "tactic": candidates[0][1][1],
                    }
                output["cache_pairs"].append(
                    {
                        "trial": trial,
                        "arm": arm,
                        "sha256": actual,
                        "records": len(records),
                        "persisted_runtime_winner": persisted,
                    }
                )
                for worker in output["workers"]:
                    if (
                        worker.get("trial") == trial
                        and worker.get("arm") == arm
                        and worker.get("phase") == "warm"
                        and worker["status"] == "success"
                    ):
                        worker["selection_profile_from_cold_pair"] = profile_comparison(
                            cold, bare(warm)["tactic"]
                        )
            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                StopIteration,
                IndexError,
            ) as error:
                output["errors"].append(
                    {"trial": trial, "arm": arm, "error": str(error)}
                )
    return output, raw


def compare_case(case):
    workers = {(x["trial"], x["arm"], x["phase"]): x for x in case["workers"]}
    trials = sorted({x[0] for x in workers})
    metrics = {
        "startup_ms": lambda w: w["startup_ms"],
        **{
            f"{execution}_wall_ms": (
                lambda w, e=execution: w["measurements"][e]["wall_ms"]["median"]
            )
            for execution in EXECUTIONS
        },
        **{
            f"{execution}_gpu_event_ms": (
                lambda w, e=execution: w["measurements"][e]["gpu_event_ms"]["median"]
            )
            for execution in EXECUTIONS
        },
    }
    groups, ratios, frequencies = [], [], []
    for phase in PHASES:
        for arm in ARMS:
            selected = [workers[(trial, arm, phase)] for trial in trials]
            for metric, get in metrics.items():
                values = [get(x) for x in selected]
                groups.append(
                    {
                        "arm": arm,
                        "phase": phase,
                        "metric": metric,
                        "trial_values": [
                            {"trial": t, "value": v} for t, v in zip(trials, values)
                        ],
                        "across_trial_distribution": stats(values),
                    }
                )
            counts = collections.Counter(x["natural_tactic"] for x in selected)
            frequencies.append(
                {
                    "arm": arm,
                    "phase": phase,
                    "trials": len(trials),
                    "tactic_counts": {str(t): counts[t] for t in sorted(counts)},
                    "cached_default_winner_count": sum(
                        x["natural_tactic"] == -1 and x["natural_cache_hit"]
                        for x in selected
                    ),
                }
            )
        for numerator, denominator in (
            ("v2", "v1"),
            ("v1", "default"),
            ("v2", "default"),
        ):
            for metric, get in metrics.items():
                values = [
                    get(workers[(t, numerator, phase)])
                    / get(workers[(t, denominator, phase)])
                    for t in trials
                ]
                ratios.append(
                    {
                        "numerator": numerator,
                        "denominator": denominator,
                        "phase": phase,
                        "metric": metric,
                        "paired_trial_ratios": [
                            {"trial": t, "ratio": v} for t, v in zip(trials, values)
                        ],
                        "across_pair_distribution": stats(values),
                    }
                )
    return {
        "natural_distributions": groups,
        "paired_latency_ratios": ratios,
        "natural_winner_frequencies": frequencies,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--image-json", type=Path, required=True)
    parser.add_argument("--expected-image-digest")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-ms", nargs="+", type=int, default=[128, 144])
    parser.add_argument("--expected-trials", type=int, default=3)
    parser.add_argument("--expected-samples", type=int, default=15)
    parser.add_argument("--expected-seed", type=int, default=17)
    parser.add_argument("--min-iterations", type=int, default=20)
    args = parser.parse_args(argv)
    report = {
        "schema_version": 1,
        "status": "error",
        "analyzer_sha256": sha(__file__),
        "expected_workers": len(args.expected_ms)
        * args.expected_trials
        * len(ARMS)
        * len(PHASES),
        "observed_worker_records": 0,
        "validated_worker_payloads": 0,
        "validated_workers": 0,
        "expectations": {
            "ms": args.expected_ms,
            "n": 1024,
            "k": 1024,
            "trials_per_case": args.expected_trials,
            "samples_per_worker": args.expected_samples,
            "seed": args.expected_seed,
            "minimum_requested_iterations": args.min_iterations,
            "qualified_runtime_candidate_count": 9,
            "warmup_ms": 500,
            "calibration_target_ms": 20,
        },
        "method": {
            "unit": "one fresh process per arm/phase/trial; cold/warm are paired by trial",
            "aggregation": "within-worker sample median, then descriptive distribution across trial medians; no pooled inference",
            "ratios": "paired within shape, trial and phase; numerator latency divided by denominator latency",
            "oracle": "forced-memory-cache public-API diagnostic; selected-to-best is a latency ratio, diagnostic regret is that ratio minus one, all within the same worker",
            "oracle_rank": "ascending within-worker wall medians; exact ties share competition rank",
            "profiling": "actual max-profile input M; warm selection interpretation uses its cold pair",
        },
        "limits": [
            f"{args.expected_trials} trial pairs provide descriptive evidence only; no significance or full-model serving claim.",
            "Natural FILE-loaded warm lookup and forced-memory oracle lookup have different dispatch paths; oracle regret is diagnostic, not exact warm-path optimality.",
            "Ordinal identities are shape-specific; -1 may execute the same underlying algorithm as an explicit ordinal.",
            "M144 in this batch tunes and serves M144; it is not a cross-shape transfer test.",
            "Compilation caches may be shared; warmup budgets do not establish clock stability.",
            "Image metadata accompanies the runs; worker payloads do not independently attest the container digest.",
        ],
        "cases": [],
        "errors": [],
    }
    try:
        require(
            len(args.case) == len(args.expected_ms)
            and len(set(args.expected_ms)) == len(args.expected_ms),
            "case count must match unique expected M values",
        )
        source = source_identity(args)
        report["source_identity"] = source
        image = load_json(args.image_json)
        image = image[0] if isinstance(image, list) else image
        require(
            image.get("Architecture") == "amd64" and image.get("Os") == "linux",
            "image is not native linux/amd64",
        )
        digests = image.get("RepoDigests", [])
        require(bool(digests), "missing immutable image digest")
        if args.expected_image_digest:
            require(
                any(
                    x == args.expected_image_digest
                    or x.endswith("@" + args.expected_image_digest)
                    for x in digests
                ),
                "image digest differs from expected",
            )
        report["image"] = {
            "repo_digests": digests,
            "architecture": image["Architecture"],
            "os": image["Os"],
            "created": image.get("Created"),
            "metadata_sha256": sha(args.image_json),
        }
        for base in args.case:
            case, _ = analyze_case(base, source, args)
            report["cases"].append(case)
        actual_ms = sorted(
            c.get("shape", [None])[0] for c in report["cases"] if "shape" in c
        )
        require(actual_ms == sorted(args.expected_ms), "missing or repeated M case")
        identities = [c.pop("_identity", None) for c in report["cases"]]
        require(
            all(x is not None for x in identities)
            and len({canonical(x) for x in identities}) == 1,
            "environment/source identity differs across cases",
        )
        require(
            not any(c["errors"] for c in report["cases"]),
            "one or more case/worker validations failed",
        )
        expected_workers = (
            len(args.expected_ms) * args.expected_trials * len(ARMS) * len(PHASES)
        )
        require(
            sum(len(c["workers"]) for c in report["cases"]) == expected_workers,
            "worker count mismatch",
        )
        for case in report["cases"]:
            case["comparisons"] = compare_case(case)
        report["status"] = "success"
        report["validated_workers"] = expected_workers
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        subprocess.SubprocessError,
    ) as error:
        report["errors"].append(str(error))
    for case in report["cases"]:
        case.pop("_identity", None)
        if report["status"] != "success":
            case["comparisons"] = None
    report["observed_worker_records"] = sum(
        c.get("observed_worker_records", 0) for c in report["cases"]
    )
    report["validated_worker_payloads"] = sum(
        w.get("status") == "success" for c in report["cases"] for w in c["workers"]
    )
    report["comparison_status"] = (
        "complete" if report["status"] == "success" else "withheld"
    )
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        sys.stdout.write(rendered)
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
