#!/usr/bin/env python3
"""CPU controls for pairing, provenance rejection and independent score math.

Historical records supply schema only. Artificial values in these controls are
never GPU qualification or observations in the new performance cohort.
"""

import argparse
import copy
import importlib.util
import json
import math
import tarfile
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "delay_analysis_controls", here / "analyze.py"
    )
    a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a)
    plan, schedule, _ = a.validate_plan(here / "plan.json")
    old_schedule = a.load(a.INHERITED / "schedule.json")
    passed = ["unchanged inherited worker, schedule and analysis sources"]

    def reject(label, value, mutate, validate):
        changed = copy.deepcopy(value)
        mutate(changed)
        try:
            validate(changed)
        except (ValueError, KeyError, IndexError):
            passed.append(label)
        else:
            raise AssertionError(label + " incorrectly accepted")

    check_schedule = lambda value: a.validate_schedule(value, old_schedule)
    check_schedule(schedule)
    reject(
        "missing paired arm rejected",
        schedule,
        lambda d: d["entries"].pop(),
        check_schedule,
    )
    reject(
        "reordered launch rejected",
        schedule,
        lambda d: d["entries"].reverse(),
        check_schedule,
    )
    reject(
        "capture-order mismatch rejected",
        schedule,
        lambda d: d["entries"][0].update(capture_order="winner-first"),
        check_schedule,
    )

    historical = a.load(a.INHERITED / "attempt01/timing-analysis.json")["process_rows"]
    with tarfile.open(a.INHERITED / "attempt01/raw-public.tar.gz") as archive:
        raw = json.load(archive.extractfile("raw/block0-auto_synthetic-timing.json"))
    observed = a.augment_process(historical[0], raw)
    assert (
        observed["startup_host_ms"]
        == (raw["startup"]["ended_ns"] - raw["startup"]["started_ns"]) / 1e6
    )
    assert "startup_profiler_host_ms" not in observed
    assert observed["confirmation_profiler_host_ms"] > 0
    passed.append("real historical schema yields only actually recorded host intervals")
    reject(
        "nonpositive startup host interval rejected",
        raw,
        lambda d: d["startup"].update(ended_ns=d["startup"]["started_ns"]),
        lambda d: a.augment_process(historical[0], d),
    )

    def process(block, original, confirmed, graph_ratio, profile_ratio, confirm_ratio):
        row = copy.deepcopy(historical[block])
        row.update(
            original_role=original,
            confirmed_role=confirmed,
            graph_log_ratio=math.log(graph_ratio),
            graph_ratio=graph_ratio,
            original_profiler_ratio=profile_ratio,
            confirmation_log_ratio=math.log(confirm_ratio),
            confirmation_ratio=confirm_ratio,
        )
        row.update(a.legacy.choice_effect(original, confirmed, row["graph_log_ratio"]))
        return a.augment_process(row, raw)

    def cohort(original="winner", patched="default", confirmed="default"):
        return {
            arm: [
                process(
                    block, role, confirmed, 2.0, 0.5 if role == "winner" else 1.5, 2.0
                )
                for block in range(8)
            ]
            for arm, role in (("baseline", original), ("patched", patched))
        }

    primary = "patched_over_baseline.original_selected_over_default"
    data = cohort()
    summary = a.summarize(data, schedule["entries"], resamples=256)
    assert summary["metrics"][primary]["geometric_mean"] == 0.5
    assert summary["metrics"][primary]["exploratory_paired_block_bootstrap_95pct"] == [
        0.5,
        0.5,
    ]
    assert (
        summary["counts"]["baseline"][
            "startup_explicit_faster_confirmation_explicit_slower"
        ]
        == 8
    )
    assert summary["counts"]["baseline"]["corrected"] == 8
    assert summary["counts"]["patched"]["startup_default_winners"] == 8
    passed.append(
        "known improvement, startup reversals and default counts computed independently"
    )
    adverse = a.summarize(
        cohort(original="default", patched="winner"), schedule["entries"], resamples=256
    )
    assert adverse["metrics"][primary]["geometric_mean"] == 2.0
    assert adverse["metrics"][primary]["above_one"] == 8
    passed.append("adverse choices remain in the comparison")
    neutral = cohort(original="default", patched="default")
    for block in range(8):
        neutral["patched"][block] = process(
            block, "default", "default", 4 + block, 1.5, 2.0
        )
    neutral_summary = a.summarize(neutral, schedule["entries"], resamples=256)
    assert neutral_summary["metrics"][primary]["values_by_block"] == [1.0] * 8
    assert neutral_summary["metrics"][primary][
        "exploratory_paired_block_bootstrap_95pct"
    ] == [1.0, 1.0]
    passed.append("default winner scores exactly one despite different explicit graphs")

    paired = cohort(original="winner", patched="winner", confirmed="winner")
    for block in range(8):
        paired["baseline"][block] = process(
            block, "winner", "winner", 2 ** (block + 2), 0.5, 0.5
        )
        paired["patched"][block] = process(
            block, "winner", "winner", 2 ** (block + 1), 0.5, 0.5
        )
    metric = a.summarize(paired, schedule["entries"], resamples=512)["metrics"][primary]
    assert math.isclose(metric["geometric_mean"], 0.5, abs_tol=1e-14)
    assert all(
        math.isclose(x, 0.5, abs_tol=1e-14)
        for x in metric["exploratory_paired_block_bootstrap_95pct"]
    )
    passed.append("whole paired-block resampling preserves cross-arm covariance")
    check_cohort = lambda value: a.paired_metrics(value, schedule["entries"])
    reject("missing process rejected", data, lambda d: d["patched"].pop(), check_cohort)
    reject(
        "duplicate block rejected",
        data,
        lambda d: d["patched"][7].update(block=6),
        check_cohort,
    )
    reject(
        "nonfinite score rejected",
        data,
        lambda d: d["patched"][0].update(graph_log_ratio=float("nan")),
        check_cohort,
    )
    reject(
        "invented default score rejected",
        data,
        lambda d: d["patched"][0].update(original_selected_over_default=0.9),
        check_cohort,
    )
    reject(
        "invented confirmation effect rejected",
        data,
        lambda d: d["patched"][0].update(log_hypothetical_over_original=-1),
        check_cohort,
    )
    tied = process(0, "default", "default", 1.0, 1.0, 1.0)
    assert tied["startup_profile_point_tie"] and tied["confirmation_point_tie"]
    assert not tied["startup_explicit_faster_confirmation_explicit_slower"]
    passed.append("ties are retained without adding a threshold")

    binding = a.module(
        a.ROOT / "test/manual/kernels/flashinfer_autotune_delay_binding.py",
        "delay_binding_controls_source",
    )
    witness = dict(
        arm="baseline",
        upstream_head=binding.UPSTREAM_HEAD,
        wrapper_path="/study/input/current/source/flashinfer/tllm_utils.py",
        wrapper_sha256=binding.wrapper_sha("baseline"),
        callable_matches_source=True,
        profiler_alias_matches=True,
        native_identity_passthrough=True,
        native={
            name: {"library_sha256": value}
            for name, value in binding.NATIVE_HASHES.items()
        },
        tvm_ffi_version="controlled-fixture",
        torch_stream=0,
        ambient_ffi_stream=0,
    )
    check_binding = lambda value: a.validate_binding(
        value, "baseline", witness, binding
    )
    check_binding(witness)
    reject(
        "wrong wrapper bytes rejected",
        witness,
        lambda d: d.update(wrapper_sha256="0" * 64),
        check_binding,
    )
    reject(
        "stale profiler alias rejected",
        witness,
        lambda d: d.update(profiler_alias_matches=False),
        check_binding,
    )
    reject(
        "native substitution rejected",
        witness,
        lambda d: d["native"]["trtllm_utils"].update(library_sha256="0" * 64),
        check_binding,
    )
    reject(
        "FFI dependency drift rejected",
        witness,
        lambda d: d.update(tvm_ffi_version="changed"),
        check_binding,
    )
    shifted = copy.deepcopy(witness)
    shifted.update(torch_stream=12, ambient_ffi_stream=34)
    check_binding(shifted)
    passed.append("process-local stream handles do not masquerade as dependency drift")

    with tempfile.TemporaryDirectory(prefix="delay-binding-analyzer-controls-") as temp:
        root = Path(temp)
        rows = []
        expected = [
            (phase, arm, None, None)
            for phase in ("ordering", "admission")
            for arm in a.ARMS
        ]
        expected += [
            ("timing", row["arm"], row["block"], row["sequence"])
            for row in schedule["entries"]
        ]
        for index, (phase, arm, block, sequence) in enumerate(expected):
            relative = (
                f"ordering/{arm}.json"
                if phase == "ordering"
                else f"{arm}/admission-auto_synthetic.json"
                if phase == "admission"
                else f"{arm}/block{block}-auto_synthetic-timing.json"
            )
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            cache_hash = None
            result = {}
            if phase != "ordering":
                path.with_suffix(".cache.json").write_text("{}\n")
                cache_hash = a.sha256(path.with_suffix(".cache.json"))
                result = {
                    "startup": {"cache_sha256": cache_hash},
                    "final": {"cache_sha256": cache_hash},
                }
            path.write_text(json.dumps(result) + "\n")
            path.with_suffix(".bootstrap.json").write_text("{}\n")
            path.with_suffix(".exit-code").write_text("0\n")
            cid = f"{index + 1:064x}"
            path.with_suffix(".container-id").write_text(cid + "\n")
            rows.append(
                dict(
                    phase=phase,
                    arm=arm,
                    block=block,
                    sequence=sequence,
                    started_ns=10 * index + 1,
                    ended_ns=10 * index + 2,
                    returncode=0,
                    container_id=cid,
                    result_relative_path=relative,
                    cache_sha256=cache_hash,
                    result_sha256=a.sha256(path),
                    bootstrap_sha256=a.sha256(path.with_suffix(".bootstrap.json")),
                )
            )
        ledger = dict(
            schema_version=1,
            master_plan_sha256=a.sha256(here / "plan.json"),
            master_schedule_sha256=a.sha256(here / "schedule.json"),
            entries=rows,
        )
        check = lambda value: a.validate_ledger(
            value, schedule["entries"], root, here / "plan.json", False
        )
        check(ledger)
        passed.append(
            "separate containers across ordering, admission and all paired workers"
        )
        reject(
            "reused container rejected",
            ledger,
            lambda d: d["entries"][1].update(
                container_id=d["entries"][0]["container_id"]
            ),
            check,
        )
        reject(
            "overlapping workers rejected",
            ledger,
            lambda d: d["entries"][1].update(started_ns=1),
            check,
        )
        reject(
            "unrecorded extra attempt rejected",
            ledger,
            lambda d: d["entries"].append(d["entries"][-1]),
            check,
        )
        reject(
            "reordered phase rejected",
            ledger,
            lambda d: d["entries"][0].update(phase="timing"),
            check,
        )
        reject(
            "raw artifact substitution rejected",
            ledger,
            lambda d: d["entries"][0].update(result_sha256="0" * 64),
            check,
        )
        reject(
            "changed original winner cache rejected",
            ledger,
            lambda d: d["entries"][2].update(cache_sha256="0" * 64),
            check,
        )

    result = {
        "status": "passed",
        "count": len(passed),
        "checks": passed,
        "analyzer_sha256": a.sha256(here / "analyze.py"),
        "historical_schema_only": True,
        "gpu_qualification": False,
        "performance_inference": False,
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
