#!/usr/bin/env python3
"""Collect and verify node-local FP8 build provenance without importing CUDA."""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

BATCH = "20261020T034000Z-h100-fp8-linear"
GROUP_GEMM_RELEASE_SHA = (
    "727c326e0f2270748f2c9036dba98fc1b64a6c70e2f8891d23633567cc808ecc"
)
TRANSLATION_UNITS = ("bmm_fp8.cu", "group_gemm.cu", "flashinfer_gemm_binding.cu")


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--attempt", default="attempt01")
    parser.add_argument("--batch", default=BATCH)
    parser.add_argument("--trials", nargs="+", type=int, default=list(range(6)))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy-ninja", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    study = args.study.resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.attempt):
        parser.error("invalid attempt identifier")
    runtime = study / ("runtime-" + args.attempt)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.batch):
        parser.error("invalid batch identifier")
    results = study / "results" / args.batch / args.attempt
    manifest_path = runtime / "source-overlay/source-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = {
        row["installed_package_path"]: row
        for row in manifest["files"]
        if row["runtime_overlay"]
    }
    trials = args.trials
    if (
        not trials
        or any(t not in range(6) for t in trials)
        or len(set(trials)) != len(trials)
    ):
        parser.error("expected unique trial IDs in0..5")
    expected = [
        f"trial{trial}-seed19-{arm}-{phase}"
        for trial in trials
        for arm in ("default", "v1", "v2")
        for phase in ("cold", "warm")
    ]
    output = {
        "schema_version": 1,
        "status": "error",
        "source_manifest_sha256": sha(manifest_path),
        "batch_id": args.batch,
        "fixture_shape": json.loads((study / "fixture/manifest.json").read_text())[
            "shape"
        ],
        "trials": trials,
        "expected_native_records": len(expected),
        "arms": {},
        "missing_records": [],
        "errors": [],
    }

    def host_path(container_path):
        path = Path(container_path)
        try:
            suffix = path.relative_to("/study")
        except ValueError:
            raise RuntimeError(
                "native build path is outside the study mount: " + str(path)
            )
        actual = (study / suffix).resolve()
        if not actual.is_relative_to(runtime.resolve()):
            raise RuntimeError(
                "native build path is outside this attempt: " + str(actual)
            )
        return actual

    workers = []
    for name in expected:
        path = results / (name + ".json")
        if not path.is_file():
            output["missing_records"].append(name)
            continue
        record = json.loads(path.read_text())
        if record.get("status") != "success":
            output["errors"].append(name + " is not successful")
            continue
        native = record.get("environment", {}).get("native_artifact")
        if not native:
            output["errors"].append(name + " lacks native provenance")
            continue
        arm = record.get("implementation")
        if arm != "candidate":
            output["errors"].append(name + " has invalid arm")
            continue
        workers.append((name, path, arm, native))
    for arm in ("candidate",):
        records = [row for row in workers if row[2] == arm]
        if not records:
            output["errors"].append("no native records for " + arm)
            continue
        native = records[0][3]
        try:
            library, ninja = (
                host_path(native[key]) for key in ("library_path", "ninja_path")
            )
            if (
                library.parent != ninja.parent
                or library.name != "gemm.so"
                or ninja.name != "build.ninja"
            ):
                raise RuntimeError("unexpected native build filenames or directories")
            library_sha, ninja_sha = sha(library), sha(ninja)
            text = ninja.read_text()
            if "-gencode=arch=compute_90a,code=sm_90a" not in text:
                raise RuntimeError("the FP8 native build did not target SM90a")
            sources = []
            for line in text.splitlines():
                match = re.fullmatch(r"build .+: (?:cuda_compile|compile) (.+)", line)
                if match:
                    sources.append(match.group(1))
            if len(sources) != 3 or {Path(path).name for path in sources} != set(
                TRANSLATION_UNITS
            ):
                raise RuntimeError(
                    "ninja does not contain exactly the three audited FP8 translation units"
                )
            inputs = []
            package = runtime / "packages" / arm
            for source in sources:
                actual = host_path(source)
                expected_source = package / "flashinfer/data/csrc" / actual.name
                if actual != expected_source.resolve():
                    raise RuntimeError(
                        "ninja source is not from the isolated package: " + source
                    )
                actual_sha = sha(actual)
                source_key = "flashinfer/data/csrc/" + actual.name
                expected_sha = (
                    GROUP_GEMM_RELEASE_SHA
                    if actual.name == "group_gemm.cu"
                    else rows[source_key][arm + "_sha256"]
                )
                if actual_sha != expected_sha:
                    raise RuntimeError("translation-unit hash mismatch: " + actual.name)
                inputs.append(
                    {
                        "repository_path": "csrc/" + actual.name,
                        "container_path": source,
                        "sha256": actual_sha,
                        "size": actual.stat().st_size,
                    }
                )
            all_runtime = {}
            for relative, row in rows.items():
                actual = package / relative
                actual_sha = sha(actual)
                if actual_sha != row[arm + "_sha256"]:
                    raise RuntimeError(
                        "collected source differs from frozen manifest: " + relative
                    )
                all_runtime[relative] = actual_sha
            include_path = "/study/" + str(
                (package / "flashinfer/data/include").relative_to(study)
            )
            if include_path not in text:
                raise RuntimeError(
                    "ninja does not use the isolated package include directory"
                )
            references = []
            for name, path, _, recorded in records:
                if (
                    recorded.get("library_path") != native["library_path"]
                    or recorded.get("ninja_path") != native["ninja_path"]
                ):
                    raise RuntimeError("worker switched native build paths: " + name)
                if (
                    recorded.get("library_sha256") != library_sha
                    or recorded.get("ninja_sha256") != ninja_sha
                ):
                    raise RuntimeError(
                        "worker build hash differs from final artifact: " + name
                    )
                if recorded.get("source_sha256") != all_runtime:
                    raise RuntimeError(
                        "worker source hashes differ from final source inputs: " + name
                    )
                if recorded.get("cuda_arch_list") != "9.0":
                    raise RuntimeError(
                        "worker architecture differs from the plan: " + name
                    )
                references.append({"result": path.name, "result_sha256": sha(path)})
            arm_output = {
                "library": {
                    "container_path": native["library_path"],
                    "sha256": library_sha,
                    "size": library.stat().st_size,
                },
                "ninja": {
                    "container_path": native["ninja_path"],
                    "sha256": ninja_sha,
                    "size": ninja.stat().st_size,
                },
                "translation_units": sorted(
                    inputs, key=lambda row: row["repository_path"]
                ),
                "all_runtime_source_sha256": all_runtime,
                "validated_worker_results": references,
                "include_directory": include_path,
            }
            if args.copy_ninja:
                destination = args.output.parent / "native-builds" / arm / "build.ninja"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and sha(destination) != ninja_sha:
                    raise RuntimeError(
                        "refusing to replace a different retained ninja file"
                    )
                shutil.copyfile(ninja, destination)
                arm_output["ninja"]["retained_relative_path"] = str(
                    destination.relative_to(args.output.parent)
                )
            output["arms"][arm] = arm_output
        except Exception as error:
            output["errors"].append(
                arm + ": " + type(error).__name__ + ": " + str(error)
            )
    if output["missing_records"] and not args.allow_partial:
        output["errors"].append("required native records are missing")
    output["validated_native_records"] = sum(
        len(arm["validated_worker_results"]) for arm in output["arms"].values()
    )
    output["scope"] = (
        "FP8 gemm module only; explicit rebuilt library hashes, retained ninja files, all three actual translation-unit hashes and all seven frozen runtime files. No GPU APIs are called by this collector."
    )
    output["status"] = (
        "error"
        if output["errors"]
        else "partial"
        if output["missing_records"]
        else "success"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": output["status"],
                "validated_native_records": output["validated_native_records"],
                "missing": len(output["missing_records"]),
                "errors": output["errors"],
            }
        )
    )
    return 1 if output["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
