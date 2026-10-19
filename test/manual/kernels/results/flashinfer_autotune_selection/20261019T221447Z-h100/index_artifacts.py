#!/usr/bin/env python3
"""Build the fixed H100 batch index after verifying final analysis and raw bytes.

Prints JSON unless --output is supplied. Never extracts archives or changes raw
inputs. Run after the readable artifacts have received their final formatting.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

BATCH_ID = "20261019T221447Z-h100"
BATCH_REL = f"test/manual/kernels/results/flashinfer_autotune_selection/{BATCH_ID}"
RUNTIME_SHA = "b2e486f842e2c56d4f2f2577f490201eb625a50f"
ANALYZER_SHA = "82c31f030b6636e288732e1c1b866a550c899625066de3c211c7132210420982"
PLAN_SHA = "730093ea29009c45e714d3bd4cb271fdee9ea655868a08ea0ea5a1b3d7da2111"
IMAGE = "lmsysorg/sglang@sha256:c1f4d9fc6fcf50c2269adbc757ce7b1778883af04155edb888fa9eea3b6648c0"
SOURCE = {
    "coordinator": (
        "test/manual/kernels/bench_flashinfer_autotune_selection.py",
        "b27a09d12ae2888f83a616368ff92bd5493888cf748ea6b07068f81d1e8d3c3b",
    ),
    "worker": (
        "test/manual/kernels/flashinfer_autotune_selection_worker.py",
        "7f8932844496f873ba3d7044a05662deab57dba139b01b8cc6b5d528c1d4ec69",
    ),
    "wrapper": (
        "python/sglang/srt/model_executor/runner/flashinfer_autotune.py",
        "64deff581665bd04830535baccfdfbdfa72794d919bc462fc469a4c42bdd68f5",
    ),
}
TUNER_SHA = "e222f35f8b5a59a99a45fad627a79db8fc45fadc52b667e63b70c0968509109d"
CASES = ("m128", "m144")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return digest(Path(path).read_bytes())


def read_json(path):
    return json.loads(Path(path).read_text())


def git_blob(source_root, relative):
    return subprocess.check_output(
        ["git", "-C", str(source_root), "show", f"{RUNTIME_SHA}:{relative}"],
        stderr=subprocess.PIPE,
    )


def analyzer_argv(cases, source_root, raw_root, analyzer, output=None):
    command = ["python3", str(analyzer)]
    for case in cases:
        command.extend(["--case", str(Path(raw_root) / case)])
    command.extend(
        [
            "--source-root",
            str(source_root),
            "--expected-source-commit",
            RUNTIME_SHA,
            "--image-json",
            str(Path(raw_root) / "image.json"),
            "--expected-image-digest",
            IMAGE,
        ]
    )
    if tuple(cases) != CASES:
        command.extend(["--expected-ms", *[case.removeprefix("m") for case in cases]])
    if output is not None:
        command.extend(["--output", str(output)])
    return command


def verify_analysis(path, cases, args):
    saved = read_json(path)
    require(
        saved.get("status") == "success"
        and saved.get("comparison_status") == "complete",
        f"{path.name} is not a completed successful analysis",
    )
    require(
        saved.get("analyzer_sha256") == ANALYZER_SHA, "analysis producer hash mismatch"
    )
    expected = {
        "source_commit": RUNTIME_SHA,
        "flashinfer_tuner_sha256": TUNER_SHA,
        **{f"{name}_sha256": value for name, (_, value) in SOURCE.items()},
    }
    require(
        saved.get("source_identity") == expected, "analysis source identity mismatch"
    )
    require(saved.get("errors") == [], "analysis contains errors")
    require(
        saved.get("image", {}).get("repo_digests") == [IMAGE], "analysis image mismatch"
    )
    command = analyzer_argv(
        cases, args.source_root, args.raw_root, args.batch_dir / "analyze.py"
    )
    command[0] = sys.executable
    process = subprocess.run(command, capture_output=True, text=True, timeout=180)
    require(
        process.returncode == 0,
        f"analyzer verification failed: {process.stderr.strip()}",
    )
    require(
        json.loads(process.stdout) == saved,
        f"{path.name} differs from recomputed analysis",
    )
    expected_workers = len(cases) * 18
    require(
        all(
            saved.get(field) == expected_workers
            for field in (
                "expected_workers",
                "observed_worker_records",
                "validated_worker_payloads",
                "validated_workers",
            )
        ),
        f"{path.name} worker counts differ from {expected_workers}",
    )
    return saved


def archive_manifest(path, raw_root, shared_members):
    members, seen = [], set()
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            require(
                not name.is_absolute() and ".." not in name.parts,
                f"unsafe archive member: {member.name}",
            )
            if member.isdir():
                continue
            require(member.isfile(), f"unsupported archive member: {member.name}")
            require(
                member.name not in seen,
                f"duplicate member within {path.name}: {member.name}",
            )
            seen.add(member.name)
            content = archive.extractfile(member).read()
            item = {
                "path": member.name,
                "bytes": len(content),
                "sha256": digest(content),
            }
            require(
                item["bytes"] == member.size, f"tar member size mismatch: {member.name}"
            )
            local = raw_root.joinpath(*name.parts)
            require(
                local.is_file() and local.read_bytes() == content,
                f"archived bytes differ from analyzed raw file: {member.name}",
            )
            previous = shared_members.setdefault(member.name, item)
            require(
                previous == item, f"conflicting shared archive metadata: {member.name}"
            )
            members.append(item)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha(path),
        "file_count": len(members),
        "members": sorted(members, key=lambda x: x["path"]),
    }


def build(args):
    batch, raw = args.batch_dir, args.raw_root
    require(
        sha(batch / "plan.json") == PLAN_SHA,
        "plan differs from its preregistered bytes",
    )
    require(
        digest(git_blob(args.source_root, f"{BATCH_REL}/plan.json")) == PLAN_SHA,
        "runtime commit does not contain the preregistered plan",
    )
    require(
        sha(batch / "analyze.py") == ANALYZER_SHA,
        "analyzer differs from fixed audited bytes",
    )
    for name, (relative, expected) in SOURCE.items():
        require(
            digest(git_blob(args.source_root, relative)) == expected,
            f"runtime Git source hash mismatch: {name}",
        )
    plan = read_json(batch / "plan.json")
    require(
        plan["batch_id"] == BATCH_ID and plan["status"] == "preregistered",
        "unexpected plan",
    )
    preflight = read_json(raw / "preflight.json")
    require(
        preflight["status"] == "success"
        and preflight["source_commit"] == RUNTIME_SHA
        and preflight["plan_sha256"] == PLAN_SHA
        and preflight["fp8_numerical_passed"] is True,
        "preflight does not bind the successful runtime to the preregistered plan",
    )
    require(
        preflight["source_sha256"] == {path: value for path, value in SOURCE.values()}
        and preflight["flashinfer_tuner_sha256"] == TUNER_SHA,
        "preflight source hashes mismatch",
    )
    image = read_json(raw / "image.json")
    require(
        image.get("RepoDigests") == [IMAGE]
        and image.get("Architecture") == "amd64"
        and image.get("Os") == "linux",
        "unexpected runtime image",
    )
    final = verify_analysis(batch / "analysis.json", CASES, args)
    verify_analysis(batch / "analysis-m128.json", ("m128",), args)
    require(
        {case["case"] for case in final["cases"]} == set(CASES),
        "final analysis lacks a case",
    )
    pairs, tuned, empty = 0, 0, 0
    for case in final["cases"]:
        require(not case["errors"], "case has validation errors")
        for pair in case["cache_pairs"]:
            pairs += 1
            if pair["arm"] == "default":
                require(
                    pair["persisted_runtime_winner"] is None and pair["records"] == 0,
                    "default cache unexpectedly contains a winner",
                )
                empty += 1
            else:
                require(
                    pair["persisted_runtime_winner"] is not None
                    and pair["records"] > 0,
                    "tuned cache lacks a persisted winner",
                )
                tuned += 1
    require((pairs, tuned, empty) == (18, 12, 6), "final cache-pair counts mismatch")
    shared_members = {}
    archives = [
        archive_manifest(batch / f"{case}-raw.tar.gz", raw, shared_members)
        for case in CASES
    ]
    for case in CASES:
        require(
            (raw / f"{case}.exit-code").read_text().strip() == "0",
            f"{case} case exit was nonzero",
        )
        required = {f"{case}/results.json", f"{case}.exit-code", f"{case}.log"}
        for trial in range(1, 4):
            for arm in ("default", "v1", "v2"):
                prefix = f"{case}/trial_{trial:03d}/{arm}"
                required.update(
                    f"{prefix}/{name}"
                    for name in (
                        "cold.json",
                        "cold.log",
                        "warm.json",
                        "warm.log",
                        "tactics.json",
                    )
                )
        require(
            required <= set(shared_members),
            f"{case} raw archive coverage is incomplete",
        )
    require(
        {
            "image.json",
            "preflight.json",
            "preflight.log",
            "gpu-before.csv",
            "gpu-processes-before.csv",
            "host_cpu.json",
            "gpu-after.csv",
        }
        <= set(shared_members),
        "final archives lack complete shared runtime provenance",
    )
    files = []
    for path in sorted(batch.rglob("*")):
        if path.is_file() and path != batch / "artifact-index.json":
            require(
                not path.is_symlink(),
                f"unexpected symlink in artifact bundle: {path.name}",
            )
            files.append(
                {
                    "path": str(path.relative_to(batch)),
                    "bytes": path.stat().st_size,
                    "sha256": sha(path),
                }
            )
    scheduler = next(
        (
            path
            for path in (batch / "scheduler.json", raw / "scheduler.json")
            if path.is_file()
        ),
        None,
    )
    if scheduler is not None:
        require(
            scheduler.is_relative_to(batch) or "scheduler.json" in shared_members,
            "scheduler record exists only outside retained artifacts",
        )
    commands = {}
    for cases, output in ((CASES, "analysis.json"), (("m128",), "analysis-m128.json")):
        commands[output] = analyzer_argv(
            cases,
            "${SOURCE_ROOT}",
            "${RAW_ROOT}",
            "${BATCH_DIR}/analyze.py",
            f"${{OUTPUT_DIR}}/{output}",
        )
    return {
        "schema_version": 1,
        "status": "success",
        "batch_id": BATCH_ID,
        "execution_commit": RUNTIME_SHA,
        "preregistration_commit": RUNTIME_SHA,
        "indexer_sha256": sha(__file__),
        "provenance": {
            "plan_sha256": PLAN_SHA,
            "preflight_sha256": sha(raw / "preflight.json"),
            "plan_preflight_source_binding_verified": True,
            "analyzer_sha256": ANALYZER_SHA,
            "runtime_image": IMAGE,
            "runtime_image_metadata_sha256": sha(raw / "image.json"),
            "analysis_recomputed_and_equal": True,
            "shared_archive_metadata_conflicts": 0,
        },
        "validated_counts": {
            "workers": 36,
            "cache_file_pairs": pairs,
            "tuned_winner_persistence_pairs": tuned,
            "empty_default_cache_pairs": empty,
        },
        "cases": [
            {
                "case": case["case"],
                "raw_started_at": case["raw_started_at"],
                "raw_finished_at": case["raw_finished_at"],
                "exit_code": 0,
                "raw_results_sha256": case["raw_results_sha256"],
                "warnings": case["warnings"],
            }
            for case in final["cases"]
        ],
        "allocation": {
            "job_id": "2674847",
            "scheduler_record_retained": scheduler is not None,
            "scheduler_record_sha256": sha(scheduler) if scheduler else None,
            "scheduler_state_interpretation": "see retained record; not inferred by index builder",
        },
        "artifacts": files,
        "raw_archives": archives,
        "reproduction": {
            "variables": {
                "BATCH_DIR": "this committed batch directory",
                "RAW_ROOT": "a fresh scratch directory for byte-exact extraction",
                "SOURCE_ROOT": f"a local SGLang Git checkout containing {RUNTIME_SHA}",
                "OUTPUT_DIR": "a scratch directory for regenerated analysis",
            },
            "prepare_argv": [["mkdir", "-p", "${RAW_ROOT}", "${OUTPUT_DIR}"]],
            "extract_argv": [
                [
                    "tar",
                    "-xzf",
                    f"${{BATCH_DIR}}/{case}-raw.tar.gz",
                    "-C",
                    "${RAW_ROOT}",
                ]
                for case in CASES
            ],
            "analyzer_argv": commands,
        },
        "retention": "Archives preserve authoritative raw bytes including cache EOFs. The index excludes its own checksum; the Git commit anchors the index. Rebuild after the last readable artifact is formatted.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = build(args)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        subprocess.SubprocessError,
        tarfile.TarError,
    ) as error:
        report = {
            "schema_version": 1,
            "status": "error",
            "batch_id": BATCH_ID,
            "error": str(error),
        }
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        sys.stdout.write(rendered)
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
