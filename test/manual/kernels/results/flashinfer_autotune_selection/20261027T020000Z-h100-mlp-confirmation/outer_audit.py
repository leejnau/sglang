"""Verify mlp-confirmation source, environment and native evidence offline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import shlex
import subprocess
from pathlib import Path

BATCH = "test/manual/kernels/results/flashinfer_autotune_selection/20261027T020000Z-h100-mlp-confirmation"
PRIOR_BATCH = "test/manual/kernels/results/flashinfer_autotune_selection/20261024T180000Z-h100-objective-values"
CHECKS = []


def require(value, label):
    if not value:
        raise AssertionError(label)
    CHECKS.append(label)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def normalize_command(row):
    return {
        "file": row["file"],
        "argv": [
            re.sub(r"/study/runtime-[^/]+", "/study/runtime-ATTEMPT", token)
            for token in shlex.split(row["command"])
        ],
        "directory": re.sub(
            r"/study/runtime-[^/]+", "/study/runtime-ATTEMPT", row["directory"]
        ),
    }


def check_gpu(path, expected):
    rows = [
        {key.strip(): value.strip() for key, value in row.items()}
        for row in csv.DictReader(path.open())
    ]
    require(len(rows) == 1, path.name + ": one physical GPU")
    row = rows[0]
    require(row["name"] == expected["gpu"], path.name + ": GPU model")
    require(
        row["uuid"].removeprefix("GPU-") == expected["uuid"],
        path.name + ": exact GPU UUID",
    )
    require(row["driver_version"] == expected["driver"], path.name + ": driver")
    require(
        float(row["power.limit [W]"].split()[0]) == expected["power_limit_watts"],
        path.name + ": power limit",
    )
    require(
        int(row["memory.used [MiB]"].split()[0]) == 0,
        path.name + ": no GPU memory in use",
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--prior-bootstrap", type=Path, required=True)
    parser.add_argument("--allocation-evidence", type=Path, required=True)
    parser.add_argument("--phase", choices=("admission", "timing"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    FROZEN = args.execution_commit
    require(re.fullmatch(r"[0-9a-f]{40}", FROZEN) is not None, "explicit frozen commit")
    require(not args.output.exists(), "refuse report overwrite")
    raw, source = args.raw_root, args.source_root
    batch = source / BATCH
    plan, schedule, historical = (
        read(batch / name)
        for name in ("plan.json", "schedule.json", "historical-contract.json")
    )
    require(
        subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        == FROZEN,
        "exact frozen research checkout",
    )
    for name in (
        "plan.json",
        "schedule.json",
        "historical-contract.json",
        "launch.sh",
        "run_current.py",
    ):
        committed = subprocess.check_output(
            ["git", "-C", str(source), "show", FROZEN + ":" + BATCH + "/" + name]
        )
        require(
            hashlib.sha256(committed).hexdigest() == sha(batch / name),
            "committed source: " + name,
        )
    require(
        plan["schedule_sha256"] == sha(batch / "schedule.json"),
        "schedule bound by plan",
    )
    require(
        plan["historical_contract_sha256"] == sha(batch / "historical-contract.json"),
        "history bound by plan",
    )
    allocation = args.allocation_evidence.read_text()
    require("1u1g-gen-0100" in allocation, "allocation targets exact node")
    require(
        re.search(r"UserId=lnau(?:\(|\s)", allocation) is not None,
        "allocation belongs to lnau",
    )
    require(
        re.search(r"(?:OverSubscribe=NO|Shared=0)\b", allocation) is not None,
        "Slurm whole-node exclusivity recorded",
    )
    require(re.search(r"NumNodes=1\b", allocation) is not None, "one allocated node")

    manifests = []
    reuse_path = raw / "native-build-cache/reuse-manifest.json"
    reuse = read(reuse_path)
    reference_sha = "7c4727aca3d99adcb3fe2f35d06f4408b512baa4fe9a91b975bdf77358fe205d"
    require(
        sha(args.prior_bootstrap)
        == sha(raw / "native-reference.bootstrap.json")
        == reference_sha,
        "exact previous bootstrap retained",
    )
    require(
        reuse["status"] == "passed"
        and reuse["native_reference_sha256"] == reference_sha,
        "reuse manifest bound to qualified bootstrap",
    )
    restoration_path = raw / "native-build-cache/restoration.json"
    restoration = read(restoration_path)
    require(
        reuse["restoration_method"]
        == "archive-exact-bytes-with-recorded-timestamp-restoration"
        and not reuse["copy_preserves_mtime_ns"],
        "explicit archive restoration amendment",
    )
    require(
        reuse["native_restoration_sha256"] == sha(restoration_path),
        "restoration evidence bound to reuse manifest",
    )
    require(
        restoration["status"] == "passed"
        and restoration["archive_sha256"]
        == plan["native_reuse"]["native_archive_sha256"],
        "exact archived native input",
    )
    require(
        restoration["preservation_manifest_sha256"]
        == sha(raw / "native-build-cache/prior-preservation.json")
        == "40e684976202ae6eca47e462e5da92efa35ccd2fa4cf1ccaaa8db0716936b6df",
        "original preservation manifest retained",
    )
    require(
        not restoration["native_bytes_modified"]
        and not restoration["source_mtimes_modified"]
        and len(restoration["changes"]) == 10,
        "only ten native filesystem timestamps reconstructed",
    )
    require(
        reuse["source_files_verified"] > 0 and reuse["input_files"],
        "complete restored source verification recorded",
    )
    cache_prefix = "current/.cache/flashinfer/0.0.0+unknown/90a/cached_ops/"
    for provider, digest in plan["native_reuse"]["required_library_sha256"].items():
        require(
            reuse["native_files"][f"{cache_prefix}{provider}/{provider}.so"] == digest,
            "reuse manifest ELF matches plan " + provider,
        )
    phases = ["admission"] if args.phase == "admission" else ["admission", "timing"]
    for phase in phases:
        manifest = read(
            raw / ("manifest.json" if phase == "admission" else "manifest-timing.json")
        )
        require(
            manifest["execution_commit"] == FROZEN, phase + ": exact execution commit"
        )
        require(
            manifest["plan_sha256"] == sha(batch / "plan.json"), phase + ": exact plan"
        )
        require(
            manifest["schedule_sha256"] == sha(batch / "schedule.json"),
            phase + ": exact schedule",
        )
        require(
            manifest["historical_contract_sha256"]
            == sha(batch / "historical-contract.json"),
            phase + ": exact history contract",
        )
        require(
            manifest["native_reference_sha256"] == reference_sha
            and manifest["native_reuse_sha256"] == sha(reuse_path),
            phase + ": frozen native reuse bindings",
        )
        retained = raw / "native-build-cache" / phase
        for relative, digest in reuse["cache_files"].items():
            if relative.endswith("/flashinfer_jit.log"):
                continue
            require(
                sha(retained / relative) == digest,
                phase + ": actual copied cache artifact " + relative,
            )
        for provider in plan["native_reuse"]["required_library_sha256"]:
            actual = {
                str(path.relative_to(retained))
                for path in (retained / cache_prefix / provider).rglob("*")
                if path.is_file()
            }
            expected = {
                path
                for path in reuse["native_files"]
                if path.startswith(cache_prefix + provider + "/")
            }
            require(
                actual == expected,
                phase + ": unchanged native file inventory " + provider,
            )
        for path, digest in manifest["source_sha256"].items():
            require(
                sha(source / path) == digest, phase + ": actual source bytes " + path
            )
        for name in ("candidate_manifest_sha256", "helper_sha256"):
            wanted = plan["qualified_source"][
                "container_preflight_sha256" if name == "helper_sha256" else name
            ]
            require(manifest[name] == wanted, phase + ": provider manifest " + name)
        image = read(raw / f"image-{phase}.json")
        require(
            image == manifest["image"], phase + ": image metadata bound to manifest"
        )
        require(
            image["Id"] == plan["environment"]["image_id"]
            and plan["environment"]["image"] in image["RepoDigests"]
            and image["Architecture"] == "amd64"
            and image["Os"] == "linux",
            phase + ": exact pinned container",
        )
        for when in ("before", "after"):
            reuse_check = read(raw / f"reuse-{when}-{phase}.json")
            require(
                reuse_check["status"] == "passed"
                and reuse_check["native_reuse_sha256"] == sha(reuse_path),
                phase + ": source/native reuse verified " + when,
            )
            require(
                reuse_check["input_files_verified"] == len(reuse["input_files"]),
                phase + ": full copied source hash count " + when,
            )
            require(
                reuse_check["cache_files_verified"]
                == sum(
                    not path.endswith("/flashinfer_jit.log")
                    for path in reuse["cache_files"]
                ),
                phase + ": full copied cache hash count " + when,
            )
            gpu = check_gpu(raw / f"gpu-{when}-{phase}.csv", plan["environment"])
            require(
                not (raw / f"gpu-processes-{when}-{phase}.csv").read_text().strip(),
                phase + ": no visible competing compute processes " + when,
            )
            if when == "before":
                require(
                    gpu == manifest["gpu"],
                    phase + ": actual GPU metadata bound to manifest",
                )
        require(
            (raw / f"launcher-{phase}.exit-code").read_text().strip() == "0",
            phase + ": launcher completed",
        )
        require(
            (raw / f"sglang-commit-before-{phase}.txt").read_text().strip()
            == FROZEN
            == (raw / f"sglang-commit-after-{phase}.txt").read_text().strip(),
            phase + ": source commit invariant",
        )
        require(
            not (raw / f"sglang-status-after-{phase}.txt").read_text().strip(),
            phase + ": clean source after execution",
        )
        manifests.append(manifest)
    if len(manifests) == 2:
        for key in (
            "source_sha256",
            "plan_sha256",
            "schedule_sha256",
            "historical_contract_sha256",
            "fixture_manifest_sha256",
            "fixture_tensor_file_sha256",
        ):
            require(
                manifests[0][key] == manifests[1][key],
                "admission/cohort identity invariant: " + key,
            )
    require(
        sha(raw / "fixture/manifest.json") == historical["fixture"]["manifest_sha256"],
        "immutable fixture manifest",
    )
    require(
        sha(raw / "fixture/fixture.safetensors")
        == historical["fixture"]["tensor_file_sha256"],
        "immutable full fixture bytes",
    )

    legacy_path = source / PRIOR_BATCH / "analyze.py"
    spec = importlib.util.spec_from_file_location(
        "qualified_native_checks", legacy_path
    )
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    legacy.require = require
    prior = read(args.prior_bootstrap)
    prior_identity = legacy.normalize_provider_identity(prior)
    for provider, digest in plan["native_reuse"]["required_library_sha256"].items():
        require(
            prior_identity[provider]["library_sha256"] == digest,
            "reference native ELF matches plan " + provider,
        )
    names = [("admission-auto_synthetic", "admission", schedule["entries"][0])]
    if args.phase == "timing":
        names += [
            (f"block{row['block']}-auto_synthetic-timing", "timing", row)
            for row in schedule["entries"]
        ]
    containers, identity_rows, summaries = [], [], []
    for name, phase, assignment in names:
        result_path = raw / (name + ".json")
        data, boot = read(result_path), read(result_path.with_suffix(".bootstrap.json"))
        prefix = name + ": "
        require(
            data["status"] == "success" and boot["status"] == "passed",
            prefix + "successful worker/bootstrap",
        )
        require(
            boot["result_sha256"] == sha(result_path) and boot["pid"] == data["pid"],
            prefix + "same-process exact result binding",
        )
        native_reuse = boot["native_reuse"]
        require(
            native_reuse["reference_sha256"] == reference_sha
            and native_reuse["manifest_sha256"] == sha(reuse_path),
            prefix + "exact native reuse inputs",
        )
        for point in ("before", "after_build", "after_worker"):
            check = native_reuse[point]
            require(
                check["status"] == "passed"
                and check["elf_sha256"]
                == plan["native_reuse"]["required_library_sha256"]
                and check["files_verified"] == len(reuse["native_files"]),
                prefix + "native hash guard " + point,
            )
        dry_runs = native_reuse["ninja_dry_runs"]
        require(
            [entry["name"] for entry in dry_runs] == ["gemm", "trtllm_utils"],
            prefix + "both native builders checked before loading",
        )
        for entry in dry_runs:
            path = "/study/runtime-attempt01/cache/" + cache_prefix + entry["name"]
            require(
                entry["returncode"] == 0
                and entry["stdout"]
                == f"ninja: Entering directory `{path}'\nninja: no work to do.\n",
                prefix + "Ninja required no rebuild " + entry["name"],
            )
        worker_sha = sha(
            source / "test/manual/kernels/bench_flashinfer_mlp_confirmation.py"
        )
        require(
            boot["worker_sha256"] == data["worker_sha256"] == worker_sha,
            prefix + "frozen worker bytes",
        )
        require(
            data["plan_sha256"] == sha(batch / "plan.json")
            and data["schedule_sha256"] == sha(batch / "schedule.json")
            and data["historical_contract_sha256"]
            == sha(batch / "historical-contract.json"),
            prefix + "worker bound to frozen protocol",
        )
        require(
            data["assignment"] == assignment and data["mode"] == phase,
            prefix + "exact scheduled treatment",
        )
        require(
            boot["helper_sha256"] == prior["helper_sha256"]
            and boot["candidate_manifest_sha256"] == prior["candidate_manifest_sha256"]
            and boot["source"] == prior["source"],
            prefix + "unchanged qualified native/source manifest",
        )
        require(
            boot["native_identity_passthrough"]
            and data["source"]["native_identity_passthrough"],
            prefix + "unproxied native provider",
        )
        require(
            boot["import"] == prior["import"],
            prefix + "same exact pre-worker Python/provider import inventory",
        )
        for module, row in boot["import"]["source_modules"].items():
            require(
                row["path"].startswith("/study/input/current/source/"),
                prefix + "current-source import root " + module,
            )
        require(
            data["source"]["critical_sha256"]
            == plan["source_contract"]["runtime_sha256"],
            prefix + "actual core source hashes",
        )
        require(
            data["environment"] == historical["environment"],
            prefix + "actual Torch/CUDA/GPU/inference environment",
        )
        argv = boot["worker_args"]
        for flag, expected in (
            ("--mode", phase),
            ("--trial", str(assignment["block"])),
            ("--capture-order", assignment["capture_order"]),
        ):
            require(
                argv.count(flag) == 1 and argv[argv.index(flag) + 1] == expected,
                prefix + "bootstrap CLI " + flag,
            )
        for flag, expected in (
            ("--plan", batch / "plan.json"),
            ("--schedule", batch / "schedule.json"),
            ("--historical-contract", batch / "historical-contract.json"),
            ("--fixture", raw / "fixture/fixture.safetensors"),
            ("--fixture-manifest", raw / "fixture/manifest.json"),
            ("--output", result_path),
        ):
            require(
                argv.count(flag) == 1
                and Path(argv[argv.index(flag) + 1]).name == expected.name,
                prefix + "bootstrap artifact path " + flag,
            )
        cid = (raw / (name + ".container-id")).read_text().strip()
        require(
            re.fullmatch(r"[0-9a-f]{64}", cid) is not None,
            prefix + "actual Docker container ID",
        )
        containers.append(cid)
        require(
            (raw / (name + ".exit-code")).read_text().strip() == "0",
            prefix + "container exit0",
        )
        identity = legacy.normalize_provider_identity(boot)
        require(
            identity == prior_identity,
            prefix + "exact reused ELF/Ninja/compiler/source identity",
        )
        for provider, row in identity.items():
            prior_row = prior_identity[provider]
            require(
                row["sources"] == prior_row["sources"],
                prefix + "same full native translation units " + provider,
            )
            require(
                [normalize_command(command) for command in row["compile_commands"]]
                == [
                    normalize_command(command)
                    for command in prior_row["compile_commands"]
                ],
                prefix
                + "same full compiler flags/source after runtime-path normalization "
                + provider,
            )
            original = boot["native"]["providers"][provider]
            for path_key, digest_key in (
                ("library_path", "library_sha256"),
                ("ninja_path", "ninja_sha256"),
            ):
                suffix = original[path_key].split("/cache/", 1)[1]
                artifact = raw / "native-build-cache" / phase / suffix
                require(
                    sha(artifact) == original[digest_key],
                    prefix
                    + "actual retained native artifact "
                    + provider
                    + " "
                    + path_key,
                )
        require(
            boot["native"]["tvm_ffi_sha256"] == prior["native"]["tvm_ffi_sha256"],
            prefix + "same native binding package",
        )
        identity_rows.append(identity)
        accepted = data["post_timing_acceptance"]
        require(
            accepted["used_descriptor"] == 1
            and accepted["descriptor_acceptance"] == "accepted"
            and accepted["pointers_unchanged"]
            and accepted["original_graph_replayed"],
            prefix + "explicit descriptor accepted on exact serving problem",
        )
        require(
            data["default_acceptance"]["used_descriptor"] is None
            and data["default_acceptance"]["route"] == "native_default",
            prefix + "genuine untuned default route",
        )
        synthetic = {row["role"]: row for row in data["synthetic_acceptance"]}
        require(
            synthetic["winner"]["used_descriptor"] == 1
            and synthetic["winner"]["descriptor_acceptance"] == "accepted",
            prefix + "explicit descriptor accepted on retained synthetic problem",
        )
        require(
            synthetic["default"]["used_descriptor"] is None,
            prefix + "synthetic default has no invented descriptor flag",
        )
        summaries.append(
            {
                "name": name,
                "raw_sha256": sha(result_path),
                "bootstrap_sha256": sha(result_path.with_suffix(".bootstrap.json")),
                "container_id": cid,
                "worker_sha256": worker_sha,
                "native_artifacts": {
                    p: {k: v[k] for k in ("library_sha256", "ninja_sha256")}
                    for p, v in identity.items()
                },
            }
        )
    require(
        len(set(containers)) == len(containers),
        "distinct containers across all admitted/cohort workers",
    )
    require(
        all(identity == identity_rows[0] for identity in identity_rows),
        "same actual ELF/Ninja/compiler/source identities throughout this attempt",
    )
    report = {
        "status": "passed",
        "phase": args.phase,
        "checks": len(CHECKS),
        "check_labels": CHECKS,
        "audit_sha256": sha(__file__),
        "native_validator_sha256": sha(legacy_path),
        "frozen_source_commit": FROZEN,
        "allocation_evidence_sha256": sha(args.allocation_evidence),
        "prior_bootstrap_sha256": sha(args.prior_bootstrap),
        "workers": summaries,
        "native_matches_historical_ELF": {
            name: identity_rows[0][name]["library_sha256"]
            == prior_identity[name]["library_sha256"]
            for name in prior_identity
        },
        "limits": [
            "Offline audit of recorded source/import/build/driver/resource evidence, not an experiment rerun.",
            "Exact prior ELF/Ninja/compiler/source identity is required; this does not establish identical clocks or device history.",
            "Exclusive Slurm allocation and empty visible compute-process lists do not assert identical clocks or absence of all system services.",
            "Native acceptance is observed after graph timing, on exact retained operands; original-graph trace evidence is separately audited.",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "phase": args.phase,
                "checks": len(CHECKS),
                "workers": len(summaries),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
