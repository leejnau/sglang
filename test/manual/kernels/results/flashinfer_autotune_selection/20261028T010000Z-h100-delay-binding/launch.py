"""Run frozen, sequential container phases with explicit evidence and deadlines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path

BATCH = "test/manual/kernels/results/flashinfer_autotune_selection/20261028T010000Z-h100-delay-binding"
CONTAINER_RAW = "/study/results/delay-binding/attempt01"
CONTAINER_RUNTIME = "/study/runtime-attempt01"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write_new(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def replace_json(path, value):
    temporary = path.with_name(path.name + ".next")
    write_new(temporary, value)
    temporary.replace(path)


def query(argv):
    return subprocess.run(
        argv, text=True, capture_output=True, check=True, timeout=30
    ).stdout


def worker_budget(deadline, now):
    budget = min(300, int(deadline - now - 90))
    if budget <= 0:
        raise TimeoutError("allocation preservation reserve reached")
    return budget


def timing_rows(plan, schedule, inherited):
    assert schedule["blocks"] == plan["protocol"]["blocks"] == 8
    rows = schedule["entries"]
    assert len(rows) == 16
    for sequence, row in enumerate(rows):
        block, position = divmod(sequence, 2)
        arms = ("baseline", "patched") if block % 2 == 0 else ("patched", "baseline")
        assert row["sequence"] == sequence and row["block"] == block
        assert row["position"] == position and row["arm"] == arms[position]
        assert row["cell"] == "auto_synthetic"
        assert row["capture_order"] == inherited["entries"][block]["capture_order"]
    return rows


def check_input(study, arm):
    base = study / "input" / ("current" if arm == "baseline" else "patched")
    record = read(study / ("native-reuse-" + arm + ".json"))
    actual = {
        str(p.relative_to(base))
        for p in base.rglob("*")
        if p.is_file() and not p.is_symlink()
    }
    assert actual == set(record["input_files"])
    for relative, expected in record["input_files"].items():
        assert sha(base / relative) == expected, relative
    links = {
        str(p.relative_to(base)): os.readlink(p)
        for p in base.rglob("*")
        if p.is_symlink()
    }
    assert links == record["input_links"]
    return {
        "arm": arm,
        "input_files_verified": len(actual),
        "reuse_manifest_sha256": sha(study / ("native-reuse-" + arm + ".json")),
    }


def verify_result(path):
    bootstrap = path.with_suffix(".bootstrap.json")
    assert read(path)["status"] in ("success", "passed")
    report = read(bootstrap)
    assert report["status"] == "passed" and report["result_sha256"] == sha(path)
    assert report["binding_before"] == report["binding_after"]


def gate(raw, name, phases):
    path = raw / name
    report = read(path)
    assert report["status"] == "passed", name
    ledger = read(raw / "launch-ledger.json")
    entries = [row for row in ledger["entries"] if row["phase"] in phases]
    assert len(entries) == 2 * len(phases)
    for row in entries:
        assert row["returncode"] == 0
        result = raw / row["result_relative_path"]
        assert sha(result) == row["result_sha256"]
        assert sha(result.with_suffix(".bootstrap.json")) == row["bootstrap_sha256"]
        verify_result(result)
    return sha(path)


def remove_own_container(cid_path, name):
    if not cid_path.exists():
        return
    cid = cid_path.read_text().strip()
    if len(cid) != 64 or any(c not in "0123456789abcdef" for c in cid):
        raise RuntimeError("invalid container identity")
    probe = subprocess.run(
        ["docker", "inspect", "--format", "{{.Name}}", cid],
        text=True,
        capture_output=True,
        timeout=15,
    )
    if probe.returncode == 0:
        if probe.stdout.strip() != "/" + name:
            raise RuntimeError("container name does not match owned launch")
        subprocess.run(
            ["docker", "rm", "-f", cid], check=True, timeout=20, capture_output=True
        )


def preserve_tuner(runtime, raw, stem):
    shutil.copytree(runtime / "tuner", raw / (stem + ".tuner"))
    source = runtime / "tuner/cache.json"
    if not source.is_file():
        return {"cache_relative_path": None, "cache_sha256": None}
    destination = raw / (stem + ".cache.json")
    with destination.open("xb") as handle:
        handle.write(source.read_bytes())
    assert sha(destination) == sha(source)
    return {
        "cache_relative_path": str(destination.relative_to(raw)),
        "cache_sha256": sha(destination),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument(
        "--phase", choices=("ordering", "admission", "timing"), required=True
    )
    parser.add_argument("--deadline-unix", type=float, required=True)
    args = parser.parse_args()

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    study = args.study.resolve()
    source = study / "source"
    batch = source / BATCH
    raw = study / "results/delay-binding/attempt01"
    runtime = study / "workers"
    assert query(["git", "-C", str(source), "rev-parse", "HEAD"]).strip() == args.commit
    assert not query(["git", "-C", str(source), "status", "--porcelain"]).strip()
    plan = read(batch / "plan.json")
    arms = read(study / "source-arms.json")
    assert arms["status"] == "passed" and arms["changed_files"] == [
        "source/flashinfer/tllm_utils.py"
    ]
    for arm in ("baseline", "patched"):
        assert arms["arms"][arm]["wrapper_sha256"] == plan["binding"][arm + "_sha256"]
    assert sha(batch / "schedule.json") == plan["schedule_sha256"]
    inherited = source / plan["inherited"]["batch"]
    for name, expected in plan["inherited"]["files_sha256"].items():
        assert sha(inherited / name) == expected
    assert (
        sha(source / plan["inherited"]["worker"]) == plan["inherited"]["worker_sha256"]
    )
    rows = timing_rows(
        plan, read(batch / "schedule.json"), read(inherited / "schedule.json")
    )
    if args.phase == "ordering":
        raw.mkdir(parents=True)
        runtime.mkdir()
        (raw / "ordering").mkdir()
        (raw / "native-build-cache").mkdir()
        shutil.copy2(study / "native-reference.bootstrap.json", raw)
        shutil.copy2(
            study / "native-restoration.json",
            raw / "native-build-cache/restoration.json",
        )
        shutil.copy2(
            study / "native-preservation.json",
            raw / "native-build-cache/prior-preservation.json",
        )
        shutil.copytree(
            study / "runtime-attempt01/cache",
            raw / "native-build-cache/seed",
            symlinks=True,
        )
        shutil.copy2(study / "source-snapshot.json", raw)
        shutil.copy2(study / "source-arms.json", raw)
        for arm in ("baseline", "patched"):
            folder = raw / arm
            (folder / "native-build-cache").mkdir(parents=True)
            shutil.copytree(study / "fixture", folder / "fixture")
            shutil.copy2(study / "native-reference.bootstrap.json", folder)
            shutil.copy2(
                study / ("native-reuse-" + arm + ".json"),
                folder / "native-build-cache/reuse-manifest.json",
            )
        write_new(
            raw / "launch-ledger.json",
            {
                "schema_version": 1,
                "master_plan_sha256": sha(batch / "plan.json"),
                "master_schedule_sha256": sha(batch / "schedule.json"),
                "entries": [],
            },
        )
        write_new(
            raw / "manifest.json",
            {
                "execution_commit": args.commit,
                "master_plan_sha256": sha(batch / "plan.json"),
                "master_schedule_sha256": sha(batch / "schedule.json"),
                "source_sha256": {
                    str(p.relative_to(source)): sha(p)
                    for p in sorted((source / "test/manual/kernels").glob("*.py"))
                },
                "batch_files_sha256": {
                    p.name: sha(p) for p in sorted(batch.iterdir()) if p.is_file()
                },
                "source_arms_sha256": sha(raw / "source-arms.json"),
            },
        )
    else:
        assert raw.is_dir()
        gate(raw, "ordering-gate.json", ("ordering",))
        if args.phase == "timing":
            gate(raw, "admission-analysis.json", ("ordering", "admission"))
    ledger_path = raw / "launch-ledger.json"
    ledger = read(ledger_path)
    assert ledger["master_plan_sha256"] == sha(batch / "plan.json")
    assert ledger["master_schedule_sha256"] == sha(batch / "schedule.json")
    assert not any(row["phase"] == args.phase for row in ledger["entries"]), (
        "phase already attempted"
    )
    write_new(
        raw / (args.phase + ".started.json"),
        {"unix": time.time(), "deadline_unix": args.deadline_unix},
    )
    for arm in ("baseline", "patched"):
        write_new(
            raw / ("source-before-" + args.phase + "-" + arm + ".json"),
            check_input(study, arm),
        )
    gpu_args = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,power.limit,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    gpu = query(gpu_args)
    telemetry_args = [
        "nvidia-smi",
        "--query-gpu=pstate,clocks.sm,clocks.mem,power.limit,temperature.gpu,memory.used",
        "--format=csv",
    ]
    (raw / ("gpu-telemetry-before-" + args.phase + ".csv")).write_text(
        query(telemetry_args)
    )
    (raw / ("gpu-before-" + args.phase + ".csv")).write_text(gpu)
    fields = list(csv.reader(gpu.splitlines()))
    assert len(fields) == 1
    _, name, gpu_uuid, driver, power, memory, _ = [x.strip() for x in fields[0]]
    environment = plan["environment"]
    assert name == environment["gpu"] and driver == environment["driver"]
    if "uuid" in environment:
        assert gpu_uuid.removeprefix("GPU-") == environment["uuid"].removeprefix("GPU-")
    assert float(power) == environment["power_limit_watts"] and int(memory) == 0
    process_args = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader",
    ]
    assert not query(process_args).strip()
    image = json.loads(query(["docker", "image", "inspect", environment["image"]]))[0]
    assert (
        image["Id"] == environment["image_id"]
        and image["Architecture"] == "amd64"
        and image["Os"] == "linux"
    )
    write_new(
        raw / ("image-" + args.phase + ".json"),
        {
            key: image[key]
            for key in ("Id", "RepoDigests", "Architecture", "Os", "Created")
        },
    )
    tasks = (
        rows
        if args.phase == "timing"
        else [
            {
                "arm": arm,
                "block": None,
                "sequence": None,
                "capture_order": rows[0]["capture_order"],
            }
            for arm in ("baseline", "patched")
        ]
    )
    phase_code = 0
    try:
        for task in tasks:
            arm = task["arm"]
            block = task["block"]
            stem = (
                "ordering/" + arm
                if args.phase == "ordering"
                else arm
                + "/"
                + (
                    "admission-auto_synthetic"
                    if args.phase == "admission"
                    else f"block{block}-auto_synthetic-timing"
                )
            )
            result_path = raw / (stem + ".json")
            cid_path = raw / (stem + ".container-id")
            name = "fi-delay-binding-" + uuid.uuid4().hex[:16]
            worker_runtime = runtime / (args.phase + "-" + arm + "-" + str(block))
            worker_runtime.mkdir()
            shutil.copytree(
                study / "runtime-attempt01/cache",
                worker_runtime / "cache",
                symlinks=True,
            )
            (worker_runtime / "home").mkdir()
            (worker_runtime / "tuner").mkdir()
            input_dir = (
                study / "input" / ("current" if arm == "baseline" else "patched")
            )
            cr = CONTAINER_RAW
            ct = CONTAINER_RUNTIME
            result_container = cr + "/" + stem + ".json"
            target = (
                "/workspace/test/manual/kernels/flashinfer_autotune_delay_binding.py"
            )
            tail = [
                "ordering" if args.phase == "ordering" else "mlp",
                "--arm",
                arm,
                "--output",
                result_container,
            ]
            if args.phase != "ordering":
                old = "/workspace/" + plan["inherited"]["batch"]
                tail += [
                    "--",
                    "--mode",
                    args.phase,
                    "--trial",
                    str(block or 0),
                    "--capture-order",
                    task["capture_order"],
                    "--plan",
                    old + "/plan.json",
                    "--schedule",
                    old + "/schedule.json",
                    "--historical-contract",
                    old + "/historical-contract.json",
                    "--fixture",
                    cr + "/" + arm + "/fixture/fixture.safetensors",
                    "--fixture-manifest",
                    cr + "/" + arm + "/fixture/manifest.json",
                    "--cache",
                    ct + "/tuner/cache.json",
                    "--trace",
                    cr + "/" + stem + ".trace.json",
                ]
            command = [
                "docker",
                "run",
                "--rm",
                "--name",
                name,
                "--cidfile",
                str(cid_path),
                "--gpus",
                "device=0",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--network",
                "none",
            ]
            for host, destination, readonly in (
                (source, "/workspace", True),
                (input_dir, "/study/input/current", True),
                (raw, cr, False),
                (worker_runtime, ct, False),
            ):
                command += [
                    "--mount",
                    f"type=bind,source={host},target={destination}"
                    + (",readonly" if readonly else ""),
                ]
            env = {
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "SGLANG_RUST_BUILD_MODE": "never",
                "CUDA_VISIBLE_DEVICES": "0",
                "FLASHINFER_AUTOTUNER_LOAD_FROM_FILE": "0",
                "FLASHINFER_CUDA_ARCH_LIST": "9.0",
                "MAX_JOBS": "4",
                "LD_PRELOAD": "",
                "HOME": ct + "/home",
                "PYTHONPATH": "/study/input/current/tools:/workspace/python:/workspace/test/manual/kernels",
                "FLASHINFER_WORKSPACE_BASE": ct + "/cache/current",
                "XDG_CACHE_HOME": ct + "/cache/xdg",
                "CUDA_CACHE_PATH": ct + "/cache/cuda",
                "TRITON_CACHE_DIR": ct + "/cache/triton",
                "TORCHINDUCTOR_CACHE_DIR": ct + "/cache/inductor",
            }
            for key, value in env.items():
                command += ["--env", key + "=" + value]
            command += [
                "--workdir",
                "/workspace",
                "--entrypoint",
                "/opt/sglang/bin/python",
                environment["image"],
                "/workspace/" + BATCH + "/run_current.py",
                "--arm",
                arm,
                "--source",
                "/study/input/current/source",
                "--manifest",
                "/study/input/current/manifests/candidate-manifest.json",
                "--target",
                target,
                "--output",
                result_container,
                "--native-reference",
                cr + "/native-reference.bootstrap.json",
                "--reuse-manifest",
                cr + "/" + arm + "/native-build-cache/reuse-manifest.json",
                "--",
                *tail,
            ]
            budget = worker_budget(args.deadline_unix, time.time())
            write_new(
                raw / (stem + ".launch.json"),
                {"argv": command, "timeout_seconds": budget},
            )
            started = time.monotonic_ns()
            code = 1
            try:
                with (raw / (stem + ".log")).open("xb") as log:
                    code = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, timeout=budget
                    ).returncode
            except subprocess.TimeoutExpired:
                code = 124
            finally:
                cleanup_error = None
                cache_record = {"cache_relative_path": None, "cache_sha256": None}
                try:
                    remove_own_container(cid_path, name)
                    cache_record = preserve_tuner(worker_runtime, raw, stem)
                except Exception as error:
                    cleanup_error = repr(error)
                    code = 1
                ended = time.monotonic_ns()
                (raw / (stem + ".exit-code")).write_text(str(code) + "\n")
                ledger["entries"].append(
                    {
                        "phase": args.phase,
                        "arm": arm,
                        "block": block,
                        "sequence": task["sequence"],
                        "started_ns": started,
                        "ended_ns": ended,
                        "returncode": code,
                        "container_id": cid_path.read_text().strip()
                        if cid_path.exists()
                        else None,
                        "result_relative_path": stem + ".json",
                        "result_sha256": sha(result_path)
                        if result_path.exists()
                        else None,
                        "bootstrap_sha256": sha(
                            result_path.with_suffix(".bootstrap.json")
                        )
                        if result_path.with_suffix(".bootstrap.json").exists()
                        else None,
                        "runtime_relative_path": str(worker_runtime.relative_to(study)),
                        "cleanup_error": cleanup_error,
                        **cache_record,
                    }
                )
                replace_json(ledger_path, ledger)
            if code:
                raise RuntimeError("worker failed: " + stem)
            verify_result(result_path)
            if args.phase != "ordering":
                result = read(result_path)
                assert (
                    cache_record["cache_sha256"]
                    == result["startup"]["cache_sha256"]
                    == result["final"]["cache_sha256"]
                )
            assert not query(process_args).strip()
            print("Completed " + stem, flush=True)
    except BaseException:
        phase_code = 1
        raise
    finally:
        (raw / ("launcher-" + args.phase + ".exit-code")).write_text("1\n")
        (raw / ("gpu-after-" + args.phase + ".csv")).write_text(query(gpu_args))
        (raw / ("gpu-telemetry-after-" + args.phase + ".csv")).write_text(
            query(telemetry_args)
        )
        processes = query(process_args)
        (raw / ("gpu-processes-after-" + args.phase + ".csv")).write_text(processes)
        assert not processes.strip()
        assert not query(["git", "-C", str(source), "status", "--porcelain"]).strip()
        for arm in ("baseline", "patched"):
            write_new(
                raw / ("source-after-" + args.phase + "-" + arm + ".json"),
                check_input(study, arm),
            )
        (raw / ("launcher-" + args.phase + ".exit-code")).write_text(
            str(phase_code) + "\n"
        )


if __name__ == "__main__":
    main()
