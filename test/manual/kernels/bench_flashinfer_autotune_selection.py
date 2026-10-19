"""Plan or run isolated FlashInfer candidate-selection experiments.

This coordinator uses only the standard library. Without --run it prints the
plan and neither starts workers nor creates files. Each cold/warm pair uses
fresh processes and a private tactic cache; compilation caches may be shared.
The sibling worker measures one bmm_fp8/cublas workload through SGLang's startup
wrapper, rather than a full serving engine.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

ARMS = ("default", "v1", "v2")
ENVIRONMENT_IDENTITY = (
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
OPTIONAL_ENVIRONMENT_VALUES = {"timer_env", "CUDA_VISIBLE_DEVICES", "physical_gpu_uuid"}
CHECKOUT = Path(__file__).resolve().parents[3]
WORKER = Path(__file__).with_name("flashinfer_autotune_selection_worker.py").resolve()


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def build_plan(args):
    output_dir = Path(args.output_dir).expanduser().resolve()
    # Rotate a seeded shuffle so each arm occupies each position every 3 trials.
    initial_order = list(ARMS)
    random.Random(args.seed).shuffle(initial_order)
    jobs = []
    for trial in range(args.trials):
        offset = trial % len(ARMS)
        order = initial_order[offset:] + initial_order[:offset]
        for arm in order:
            directory = output_dir / f"trial_{trial + 1:03d}" / arm
            cache = directory / "tactics.json"
            for phase in ("cold", "warm"):
                result_path = directory / f"{phase}.json"
                command = [
                    sys.executable,
                    str(WORKER),
                    "--arm",
                    arm,
                    "--phase",
                    phase,
                    "--cache",
                    str(cache),
                    "--output",
                    str(result_path),
                    "--m",
                    str(args.m),
                    "--n",
                    str(args.n),
                    "--k",
                    str(args.k),
                    "--samples",
                    str(args.samples),
                    "--iterations",
                    str(args.iterations),
                    "--seed",
                    str(args.seed + trial),
                ]
                if args.oracle:
                    command.append("--oracle")
                jobs.append(
                    {
                        "trial": trial + 1,
                        "arm": arm,
                        "phase": phase,
                        "seed": args.seed + trial,
                        "shape": [args.m, args.n, args.k],
                        "cache": str(cache),
                        "result_path": str(result_path),
                        "log_path": str(directory / f"{phase}.log"),
                        "sglang_cache_dir": str(directory / "sglang_cache"),
                        "command": command,
                    }
                )
    return {
        "schema_version": 1,
        "output_dir": str(output_dir),
        "checkout": str(CHECKOUT),
        "python": sys.executable,
        "worker": str(WORKER),
        "instrumentation_sha256": {
            "coordinator": _sha256(Path(__file__)),
            "worker": _sha256(WORKER) if WORKER.is_file() else None,
        },
        "shape": [args.m, args.n, args.k],
        "trials": args.trials,
        "seed": args.seed,
        "samples": args.samples,
        "iterations": args.iterations,
        "timeout_seconds": args.timeout,
        "oracle": args.oracle,
        "order": "seeded shuffle, rotated between trials; cold then warm per arm",
        "cache_policy": {
            "tactics": "private per arm/trial; cold and warm share only that cache",
            "SGLANG_CACHE_DIR": "private per arm/trial",
            "compilation_caches": {
                name: os.environ.get(name)
                for name in (
                    "FLASHINFER_CACHE_DIR",
                    "XDG_CACHE_HOME",
                    "CUDA_CACHE_PATH",
                    "TORCHINDUCTOR_CACHE_DIR",
                    "TRITON_CACHE_DIR",
                )
            },
            "compilation_note": "Inherited defaults may share compilation caches; no JIT isolation claimed.",
        },
        "jobs": jobs,
    }


def _finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def validate_result(result, job, process_pid, samples):
    """Reject incomplete or incompatible payloads rather than summarize them."""
    if not isinstance(result, dict):
        raise ValueError("worker JSON must be an object")
    if result.get("schema_version", 1) != 1:
        raise ValueError("unsupported worker schema_version")
    if result.get("status") != "success":
        raise ValueError(
            f"worker status is not success: {result.get('error', result.get('status'))}"
        )
    for name in ("arm", "phase", "shape", "seed"):
        if result.get(name) != job[name]:
            raise ValueError(f"worker {name} does not match its command")
    if type(result.get("pid")) is not int or result["pid"] != process_pid:
        raise ValueError("worker pid must match its fresh subprocess")
    if not isinstance(result.get("environment"), dict) or not result["environment"]:
        raise ValueError("worker environment must be a nonempty object")
    for field in ENVIRONMENT_IDENTITY:
        if field not in result["environment"] or (
            result["environment"][field] is None
            and field not in OPTIONAL_ENVIRONMENT_VALUES
        ):
            raise ValueError(f"missing worker environment identity: {field}")
    if result.get("instrumentation_sha256") != _sha256(WORKER):
        raise ValueError("worker instrumentation SHA256 does not match its source")
    if not _finite_number(result.get("startup_ms")):
        raise ValueError("worker startup_ms must be finite and nonnegative")
    calls = result.get("profile_calls")
    if type(calls) is not int or calls < 0:
        raise ValueError("worker profile_calls must be a nonnegative integer")
    if job["arm"] == "default" and calls != 0:
        raise ValueError("untuned default arm unexpectedly profiled")
    if job["arm"] != "default":
        if job["phase"] == "cold" and calls == 0:
            raise ValueError("cold tuned arm did not profile")
        if job["phase"] == "warm" and calls != 0:
            raise ValueError("warm tuned arm unexpectedly profiled")
    if (
        not isinstance(result.get("correctness"), dict)
        or result["correctness"].get("passed") is not True
    ):
        raise ValueError("worker correctness.passed must be true")
    oracle = result.get("oracle", [])
    if not isinstance(oracle, list):
        raise ValueError("worker oracle must be a list")
    for field in ("eager_checks_after_measurement", "changed_input_graph_checks"):
        count = result["correctness"].get(field)
        if type(count) is not int or count != 1 + len(oracle):
            raise ValueError(
                f"worker correctness.{field} must cover natural/oracle paths"
            )
    if not isinstance(result.get("candidate_profiles"), list):
        raise ValueError("worker candidate_profiles must be a list")
    selections = result.get("selections")
    if not isinstance(selections, list) or not selections:
        raise ValueError("worker selections must be a nonempty list")
    for selection in selections:
        if (
            not isinstance(selection, dict)
            or not isinstance(selection.get("operation"), str)
            or not isinstance(selection.get("runner"), str)
            or "tactic" not in selection
            or type(selection.get("cache_hit")) is not bool
        ):
            raise ValueError("invalid worker selection record")
    bare = [selection for selection in selections if selection.get("stage") == "bare"]
    if not bare:
        raise ValueError("worker must record a bare-serving selection")
    for selection in bare:
        if job["arm"] == "default":
            if selection["cache_hit"] or selection["tactic"] != -1:
                raise ValueError("default serving must use the uncached -1 tactic")
        elif not selection["cache_hit"]:
            raise ValueError("tuned bare serving must consume a cached winner")
    measurements = result.get("measurements")
    if not isinstance(measurements, dict):
        raise ValueError("worker measurements must be an object")
    for execution in ("eager", "cuda_graph"):
        measurement = measurements.get(execution)
        values = (
            measurement.get("samples_ms") if isinstance(measurement, dict) else None
        )
        if (
            not isinstance(values, list)
            or len(values) != samples
            or not all(map(_finite_number, values))
        ):
            raise ValueError(
                f"worker {execution} samples_ms must contain {samples} finite nonnegative samples"
            )


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reject_json_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def _run_worker(job, args, expected_worker_sha256):
    record = {**job, "status": "error"}
    Path(job["result_path"]).parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(CHECKOUT / "python"), environment.get("PYTHONPATH")])
    )
    environment["SGLANG_CACHE_DIR"] = job["sglang_cache_dir"]
    environment["PYTHONUNBUFFERED"] = "1"
    errors = []
    started = time.perf_counter()
    try:
        with open(job["log_path"], "x") as log:
            process = subprocess.Popen(
                job["command"],
                cwd=CHECKOUT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=(os.name == "posix"),
            )
            record["process_pid"] = process.pid
            try:
                process.wait(timeout=args.timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                process.wait()
                if isinstance(error, KeyboardInterrupt):
                    record["interrupted"] = True
                    errors.append("interrupted by user")
                else:
                    errors.append(f"worker exceeded {args.timeout:g}s timeout")
            record["returncode"] = process.returncode
            if process.returncode:
                errors.append(f"worker exited with code {process.returncode}")
    except OSError as error:
        errors.append(f"could not execute worker: {error}")
    record["process_wall_ms"] = (time.perf_counter() - started) * 1000
    try:
        # Keep even rejected payloads in the combined result, alongside raw files.
        result = json.loads(
            Path(job["result_path"]).read_text(),
            parse_constant=_reject_json_constant,
        )
        record["result"] = result
        validate_result(result, job, record.get("process_pid"), args.samples)
        if result["instrumentation_sha256"] != expected_worker_sha256:
            raise ValueError("worker source changed after the plan was created")
        if result.get("cache_sha256") != _sha256(job["cache"]):
            raise ValueError("worker cache_sha256 does not match its saved file")
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"invalid worker result: {error}")
    if not errors:
        record["status"] = "success"
    else:
        record["errors"] = errors
    return record


def _stats(values):
    median = statistics.median(values)
    return {
        "count": len(values),
        "median": median,
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "samples_ms": values,
    }


def summarize(records):
    groups = []
    for arm in ARMS:
        for phase in ("cold", "warm"):
            results = [
                r["result"]
                for r in records
                if r["arm"] == arm and r["phase"] == phase and r["status"] == "success"
            ]
            if not results:
                continue
            group = {
                "arm": arm,
                "phase": phase,
                "trials": len(results),
                "startup_tuning_warmup_ms": _stats([r["startup_ms"] for r in results]),
            }
            for execution in ("eager", "cuda_graph"):
                per_trial = [
                    r["measurements"][execution]["samples_ms"] for r in results
                ]
                group[f"{execution}_ms"] = _stats(
                    [v for values in per_trial for v in values]
                )
                group[f"{execution}_trial_medians_ms"] = [
                    statistics.median(values) for values in per_trial
                ]
            groups.append(group)
    return groups


def render_summary(report):
    lines = [
        f"Study status: **{report['status']}**. Full results: `{Path(report['plan']['output_dir']) / 'results.json'}`.",
        "",
        "Startup is instrumented wrapper tuning/warmup; initial FP8 module preparation is excluded. It is not full server startup.",
        "Eager/graph values below are independently measured per-invocation/replay latencies (pooled samples, milliseconds).",
        "",
        "| Arm | Phase | Trials | Startup median | Eager median | Eager SD | Graph median | Graph SD |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["summary"]:
        eager, graph = row["eager_ms"], row["cuda_graph_ms"]
        lines.append(
            f"| {row['arm']} | {row['phase']} | {row['trials']} | {row['startup_tuning_warmup_ms']['median']:.6f} | {eager['median']:.6f} | {eager['stdev']:.6f} | {graph['median']:.6f} | {graph['stdev']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Raw samples, ranges, per-trial medians, candidate profiles, correctness, environment and process-wall times are retained in results.json and worker files.",
            "Compilation caches may be shared. These kernel measurements do not establish full-model serving performance or statistical significance.",
        ]
    )
    if report["status"] != "success":
        lines.append(
            "Comparison summary withheld because the study is incomplete or contains errors."
        )
    failures = [r for r in report["runs"] if r["status"] != "success"]
    for run in failures:
        lines.append(
            f"- Trial {run['trial']} {run['arm']} {run['phase']}: {'; '.join(run.get('errors', []))}. Log: `{run['log_path']}`"
        )
    return "\n".join(lines)


def _write_report(path, report):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_plan(plan, args):
    output_dir = Path(plan["output_dir"])
    if not WORKER.is_file():
        raise ValueError(f"worker not found: {WORKER}")
    # Exclusive creation makes accidental reuse of a cold cache impossible.
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "running",
        "plan": plan,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "runs": [],
        "summary": [],
    }
    result_path = output_dir / "results.json"
    _write_report(result_path, report)
    cold_runs = {}
    study_environment = None
    for job in plan["jobs"]:
        pair = (job["trial"], job["arm"])
        cold = cold_runs.get(pair)
        if job["phase"] == "warm" and cold["status"] != "success":
            record = {
                **job,
                "status": "skipped",
                "errors": ["cold worker failed; warm reuse cannot be evaluated"],
            }
        else:
            if job["phase"] == "warm":
                try:
                    before = _sha256(job["cache"])
                except OSError as error:
                    record = {
                        **job,
                        "status": "error",
                        "errors": [f"warm cache unavailable: {error}"],
                    }
                else:
                    if before != cold["result"]["cache_sha256"]:
                        record = {
                            **job,
                            "status": "error",
                            "errors": [
                                "warm cache differs from cold worker's saved cache"
                            ],
                        }
                    else:
                        record = _run_worker(
                            job, args, plan["instrumentation_sha256"]["worker"]
                        )
                    record["cache_sha256_before_worker"] = before
                    if (
                        record["status"] == "success"
                        and record["result"].get("cache_before_sha256") != before
                    ):
                        record["status"] = "error"
                        record["errors"] = [
                            "warm worker did not observe the expected cold cache"
                        ]
            else:
                record = _run_worker(
                    job, args, plan["instrumentation_sha256"]["worker"]
                )
            if (
                job["phase"] == "warm"
                and record["status"] == "success"
                and record["process_pid"] == cold["process_pid"]
            ):
                record["status"] = "error"
                record["errors"] = [
                    "cold and warm workers must have distinct process IDs"
                ]
            if job["phase"] == "warm" and record["status"] == "success":

                def bare_winner(run):
                    selection = [
                        s
                        for s in run["result"]["selections"]
                        if s.get("stage") == "bare"
                    ][-1]
                    return (
                        selection["operation"],
                        selection["runner"],
                        selection["tactic"],
                    )

                if bare_winner(record) != bare_winner(cold):
                    record["status"] = "error"
                    record["errors"] = [
                        "warm bare-serving winner differs from cold winner"
                    ]
            if record["status"] == "success":
                identity = {
                    field: record["result"]["environment"][field]
                    for field in ENVIRONMENT_IDENTITY
                }
                if study_environment is None:
                    study_environment = identity
                elif identity != study_environment:
                    record["status"] = "error"
                    record["errors"] = [
                        "worker static environment differs within the study"
                    ]
        report["runs"].append(record)
        if job["phase"] == "cold":
            cold_runs[pair] = record
        _write_report(result_path, report)
        if record.get("interrupted"):
            break
    report["status"] = (
        "success"
        if len(report["runs"]) == len(plan["jobs"])
        and all(r["status"] == "success" for r in report["runs"])
        else "error"
    )
    report["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report["summary"] = (
        summarize(report["runs"]) if report["status"] == "success" else []
    )
    report["summary_markdown"] = render_summary(report)
    _write_report(result_path, report)
    print(report["summary_markdown"])
    return 0 if report["status"] == "success" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new directory; --run refuses an existing path",
    )
    parser.add_argument("--m", type=positive_int, default=128)
    parser.add_argument("--n", type=positive_int, default=1024)
    parser.add_argument("--k", type=positive_int, default=1024)
    parser.add_argument("--trials", type=positive_int, default=3)
    parser.add_argument("--samples", type=positive_int, default=15)
    parser.add_argument("--iterations", type=positive_int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--timeout", type=positive_float, default=600)
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="request independent candidate evaluation from each worker",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="execute workers; otherwise print the plan only",
    )
    parser.add_argument("-f", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if any(size % 16 for size in (args.m, args.n, args.k)):
        parser.error("--m, --n and --k must be multiples of 16")
    plan = build_plan(args)
    if not args.run:
        print(json.dumps(plan, indent=2))
        return 0
    try:
        return run_plan(plan, args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
