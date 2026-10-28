"""CPU controls for paired estimates and rejection of corrupted evidence.

Old raw files supply schema-only fixtures. All statistical inputs here are
artificial and never enter the GPU cohort or qualify a performance result.
"""

import argparse
import copy
import importlib.util
import io
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
        "matched_analysis", here / "analyze.py"
    )
    a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a)
    passed = []

    def check(label, condition):
        assert condition, label
        passed.append(label)

    def reject(label, function):
        try:
            function()
        except (ValueError, KeyError, IndexError, RuntimeError):
            passed.append(label)
        else:
            raise AssertionError(label)

    def altered(value, mutate):
        value = copy.deepcopy(value)
        mutate(value)
        return value

    schedule = a.load(here / "schedule.json")
    a.validate_schedule(schedule)
    passed.append("actual balanced eight-pair schedule")
    for label, mutate in (
        ("missing paired arm", lambda d: d["entries"].pop()),
        ("duplicate assignment", lambda d: d["entries"][1].update(arm="v1")),
        (
            "unmatched capture history",
            lambda d: d["entries"][1].update(capture_order="natural-first"),
        ),
        (
            "missing graph window",
            lambda d: d["round_schedules"]["0"]["cuda_graph"].pop(),
        ),
    ):
        reject(
            label, lambda mutate=mutate: a.validate_schedule(altered(schedule, mutate))
        )

    def pairs(mode, ratio):
        result = []
        orders = schedule["round_schedules"]["0"][mode]
        for index, order in enumerate(orders):
            rows = []
            for position, role in enumerate(order):
                value = ratio if role == "natural" else 1.0
                rows.append(
                    {
                        "role": role,
                        "iterations": 16,
                        "wall_batch_ms": value * 32,
                        "wall_per_call_ms": value * 2,
                        "gpu_event_batch_ms": value * 16,
                        "gpu_event_per_call_ms": value,
                        "started_ns": 100 * index + 10 * position,
                        "ended_ns": 100 * index + 10 * position + 5,
                    }
                )
            result.append({"index": index, "order": order, "measurements": rows})
        return result, orders

    for mode in a.MODES:
        rows, orders = pairs(mode, 1.25)
        result = a.paired_logs(rows, orders, mode)
        check(
            mode + " correct clock and paired ratio",
            math.isclose(result["natural_over_default"], 1.25),
        )
        check(
            mode + " absolute correct clock",
            result["median_absolute_ms"]["default"]
            == (2.0 if mode == "eager" else 1.0),
        )
        reject(mode + " missing sample", lambda: a.paired_logs(rows[:-1], orders, mode))
        reject(
            mode + " incorrect batch arithmetic",
            lambda: a.paired_logs(
                altered(
                    rows,
                    lambda d: d[0]["measurements"][0].update(gpu_event_batch_ms=99),
                ),
                orders,
                mode,
            ),
        )
        reject(
            mode + " nonfinite duration",
            lambda: a.paired_logs(
                altered(
                    rows,
                    lambda d: d[0]["measurements"][0].update(
                        wall_batch_ms=float("inf")
                    ),
                ),
                orders,
                mode,
            ),
        )
        reject(
            mode + " execution order mismatch",
            lambda: a.paired_logs(
                altered(rows, lambda d: d[0]["measurements"].reverse()), orders, mode
            ),
        )
    reject("unknown timing mode", lambda: a.paired_logs([], [], "other"))

    def cohort(v1=1.0, v2=1.0, tactic=-1):
        result = []
        for entry in schedule["entries"]:
            ratio = v1 if entry["arm"] == "v1" else v2
            result.append(
                {
                    **entry,
                    "mode": "timing",
                    "natural_tactic": tactic,
                    "startup_ms": 1.0,
                    "trace_signatures": {
                        role: {"ordered_launch_signature": ["same_kernel"]}
                        for role in a.ROLES
                    },
                    "modes": {
                        mode: {
                            "median_log_ratio": math.log(ratio),
                            "median_absolute_ms": {"natural": ratio, "default": 1.0},
                        }
                        for mode in a.MODES
                    },
                }
            )
        return result

    primary = "v2_over_v1.cuda_graph.measured_normalized"
    neutral = a.summarize(cohort(1.1, 1.1), resamples=128)
    check(
        "matched neutral result is retained",
        neutral["metrics"][primary]["geometric_mean"] == 1
        and neutral["metrics"][primary]["exploratory_paired_block_bootstrap_95pct"]
        == [1, 1],
    )
    check(
        "same descriptors and traces explicitly reported",
        neutral["same_tactic_pairs"]
        == neutral["same_natural_trace_signature_pairs"]
        == 8,
    )
    measured = a.summarize(cohort(1, 1.1), resamples=128)
    check(
        "default graphs retain primary measured variance",
        math.isclose(measured["metrics"][primary]["geometric_mean"], 1.1),
    )
    check(
        "choice identity is separate secondary only",
        measured["metrics"]["v2_over_v1.choice_identity_graph_score_secondary"][
            "geometric_mean"
        ]
        == 1,
    )
    for label, ratio in (("improvement", 0.8), ("regression", 1.2)):
        estimate = a.summarize(cohort(1, ratio, tactic=[1]), resamples=128)
        check(
            label + " retained without filtering",
            math.isclose(estimate["metrics"][primary]["geometric_mean"], ratio),
        )
    variable = cohort()
    for row in variable:
        for mode in a.MODES:
            row["modes"][mode]["median_log_ratio"] = row["trial"] - 4
    paired = a.summarize(variable, resamples=128)
    check(
        "whole-pair covariance retained in uncertainty",
        paired["metrics"][primary]["exploratory_paired_block_bootstrap_95pct"]
        == [1, 1],
    )
    reject(
        "missing block cannot estimate", lambda: a.summarize(cohort()[:-1], resamples=2)
    )
    reject(
        "duplicate block cannot estimate",
        lambda: a.summarize(
            altered(cohort(), lambda d: d.__setitem__(1, d[0])), resamples=2
        ),
    )
    reject(
        "warm observation cannot enter timing",
        lambda: a.summarize(
            altered(cohort(), lambda d: d[0].update(phase="warm")), resamples=2
        ),
    )
    reject(
        "admission observation cannot enter timing",
        lambda: a.summarize(
            altered(cohort(), lambda d: d[0].update(mode="admission")), resamples=2
        ),
    )

    # Real old records test the stable raw schema, not any current GPU outcome.
    source = a.PRIOR / "attempt01"
    packed = b"".join(
        (source / f"raw-public.tar.gz.part-{i:02d}").read_bytes() for i in range(3)
    )
    with tarfile.open(fileobj=io.BytesIO(packed), mode="r:gz") as archive:
        fixtures = {
            (arm, phase): json.load(
                archive.extractfile(f"raw/admission-trial0-{arm}-{phase}.json")
            )
            for arm in a.ARMS
            for phase in ("cold", "warm")
        }
    groups = {key: a.validate_startup(value) for key, value in fixtures.items()}
    a.validate_matched_candidates(groups["v1", "cold"], groups["v2", "cold"], True)
    passed.append("historical schema only verifies both API paths and fresh warm reuse")
    cold, warm = fixtures["v2", "cold"], fixtures["v2", "warm"]
    for label, value, mutate in (
        ("late profiler call", cold, lambda d: d["final"].update(profile_attempts=82)),
        (
            "incorrect selected tactic",
            cold,
            lambda d: d["natural"].update(tactic=[999]),
        ),
        (
            "wrong startup context",
            cold,
            lambda d: d["startup"]["selection"][0].update(in_v2_context=False),
        ),
        (
            "changed winner file",
            cold,
            lambda d: d["cache"].update(final_sha256="wrong"),
        ),
        (
            "hidden profiling failure",
            cold,
            lambda d: d["startup"]["profiles"][0].update(error="bad"),
        ),
        (
            "nonfinite candidate",
            cold,
            lambda d: d["startup"]["profiles"][0].update(profiler_ms=float("nan")),
        ),
        (
            "warm profiling",
            warm,
            lambda d: d["startup"].update(
                profiles=[cold["startup"]["profiles"][0]], profile_attempts=1
            ),
        ),
        (
            "warm changed cache",
            warm,
            lambda d: d["cache"].update(before_sha256="wrong"),
        ),
    ):
        reject(
            label,
            lambda value=value, mutate=mutate: a.validate_startup(
                altered(value, mutate)
            ),
        )
    reject(
        "different explicit candidate order",
        lambda: a.validate_matched_candidates(
            groups["v1", "cold"],
            altered(groups["v2", "cold"], lambda d: d["144"]["tactics"].reverse()),
            True,
        ),
    )
    reject(
        "different admission operand bytes",
        lambda: a.validate_matched_candidates(
            groups["v1", "cold"],
            altered(
                groups["v2", "cold"],
                lambda d: d["144"].update(fingerprints=[("wrong",)]),
            ),
            True,
        ),
    )
    entry = {
        **schedule["entries"][1],
        "capture_order": cold["assignment"]["capture_order"],
    }
    checks, graphs, ended = a.validate_outputs(cold, entry)
    passed.append("actual retained graph and numerical schema accepted")
    reject(
        "wrong graph output allocation",
        lambda: a.validate_outputs(
            altered(cold, lambda d: d["graphs"][0]["output"].update(data_ptr=1)), entry
        ),
    )
    reject(
        "failed numerical result",
        lambda: a.validate_outputs(
            altered(
                cold,
                lambda d: d["correctness"][0]["quantized_reference"].update(
                    passed=False
                ),
            ),
            entry,
        ),
    )
    reject(
        "admission performance windows forbidden",
        lambda: a.validate_measurements(cold, entry, schedule, checks, graphs, ended),
    )

    prior_binding = here.parent / "20261028T010000Z-h100-delay-binding/attempt02"
    packed = b"".join(
        (prior_binding / f"raw-public.tar.gz.part-{i:02d}").read_bytes()
        for i in range(2)
    )
    with tarfile.open(fileobj=io.BytesIO(packed), mode="r:gz") as archive:
        ordering = json.load(archive.extractfile("raw/ordering/patched.json"))
        ordering_bootstrap = json.load(
            archive.extractfile("raw/ordering/patched.bootstrap.json")
        )
    binding = a.module(
        a.ROOT / "test/manual/kernels/flashinfer_autotune_delay_binding.py",
        "ordering_controls",
    )
    binding.validate_ordering(ordering)
    passed.append("old fixed-ordering schema only accepted")
    for label, mutate in (
        (
            "failed natural ordering",
            lambda d: [
                r.update(event_ms=0.01) for r in d["natural_streams"][1]["samples"]
            ],
        ),
        (
            "failed conflicting-stream ordering",
            lambda d: [
                r.update(event_ms=0.01) for r in d["controlled_streams"][1]["samples"]
            ],
        ),
        (
            "residual delayed completion",
            lambda d: [
                r.update(residual_sync_ms=10) for r in d["actual_profiler"]["samples"]
            ],
        ),
        (
            "wrong ordering graph output",
            lambda d: d["graph_replays"][0].update(output=999),
        ),
        (
            "changed ordering dependency",
            lambda d: d["attestation_after"].update(wrapper_sha256="wrong"),
        ),
    ):
        reject(
            label,
            lambda mutate=mutate: binding.validate_ordering(altered(ordering, mutate)),
        )

    with tempfile.TemporaryDirectory() as temporary:
        raw = Path(temporary)
        rows = []
        opath = raw / "ordering/patched.json"
        opath.parent.mkdir()
        # The source JSON is preserved byte-for-byte, including its bootstrap hash.
        with tarfile.open(fileobj=io.BytesIO(packed), mode="r:gz") as archive:
            opath.write_bytes(archive.extractfile("raw/ordering/patched.json").read())
            opath.with_suffix(".bootstrap.json").write_bytes(
                archive.extractfile("raw/ordering/patched.bootstrap.json").read()
            )
        plan = a.load(here / "plan.json")
        worker = a.ROOT / "test/manual/kernels/flashinfer_autotune_delay_binding.py"
        a.validate_binding_header(opath, ordering, plan, worker, ordering=True)
        passed.append("ordering result bound to same-process wrapper bootstrap")
        for label, mutate in (
            (
                "ordering bootstrap result substitution",
                lambda d: d.update(result_sha256="wrong"),
            ),
            ("ordering bootstrap PID mismatch", lambda d: d.update(pid=-1)),
            (
                "ordering bootstrap worker mismatch",
                lambda d: d.update(worker_sha256="wrong"),
            ),
        ):
            opath.with_suffix(".bootstrap.json").write_text(
                json.dumps(altered(ordering_bootstrap, mutate))
            )
            reject(
                label,
                lambda: a.validate_binding_header(
                    opath, ordering, plan, worker, ordering=True
                ),
            )
        opath.with_suffix(".bootstrap.json").write_text(json.dumps(ordering_bootstrap))
        cid = format(1, "064x")
        opath.with_suffix(".exit-code").write_text("0\n")
        opath.with_suffix(".container-id").write_text(cid)
        rows.append(
            {
                "phase": "ordering",
                "arm": "patched",
                "block": None,
                "sequence": None,
                "started_ns": 1,
                "ended_ns": 5,
                "returncode": 0,
                "cleanup_error": None,
                "container_id": cid,
                "result_relative_path": "ordering/patched.json",
                "result_sha256": a.sha(opath),
                "bootstrap_sha256": a.sha(opath.with_suffix(".bootstrap.json")),
                "cache_relative_path": None,
                "cache_sha256": None,
                "cache_seed_relative_path": None,
                "cache_seed_sha256": None,
            }
        )
        for index, arm in enumerate(a.ARMS, 1):
            path = raw / arm / "admission-auto_synthetic.json"
            path.parent.mkdir()
            cache = path.with_suffix(".cache.json")
            cache.write_text("{}\n")
            digest = a.sha(cache)
            path.write_text(
                json.dumps(
                    {"cache": {"after_startup_sha256": digest, "final_sha256": digest}}
                )
            )
            path.with_suffix(".bootstrap.json").write_text("{}\n")
            path.with_suffix(".exit-code").write_text("0\n")
            cid = format(index + 1, "064x")
            path.with_suffix(".container-id").write_text(cid)
            rows.append(
                {
                    "phase": "admission",
                    "arm": arm,
                    "block": None,
                    "sequence": None,
                    "started_ns": index * 10 + 1,
                    "ended_ns": index * 10 + 5,
                    "returncode": 0,
                    "cleanup_error": None,
                    "container_id": cid,
                    "result_relative_path": str(path.relative_to(raw)),
                    "result_sha256": a.sha(path),
                    "bootstrap_sha256": a.sha(path.with_suffix(".bootstrap.json")),
                    "cache_relative_path": str(cache.relative_to(raw)),
                    "cache_sha256": digest,
                    "cache_seed_relative_path": None,
                    "cache_seed_sha256": None,
                }
            )
        ledger = {
            "schema_version": 1,
            "master_plan_sha256": a.sha(here / "plan.json"),
            "master_schedule_sha256": a.sha(here / "schedule.json"),
            "entries": rows,
        }

        def validate(value):
            return a.validate_ledger(
                value, schedule["entries"], raw, here / "plan.json", "admission"
            )

        validate(ledger)
        passed.append("complete synthetic launch ledger accepted")
        a.validate_ledger(
            {**ledger, "entries": rows[:1]},
            schedule["entries"],
            raw,
            here / "plan.json",
            "ordering",
        )
        passed.append("fixed-only ordering phase ledger accepted")
        reject(
            "ordering after admission",
            lambda: validate(altered(ledger, lambda d: d["entries"].reverse())),
        )
        reject(
            "ordering tuner cache forbidden",
            lambda: validate(
                altered(
                    ledger,
                    lambda d: d["entries"][0].update(cache_relative_path="wrong"),
                )
            ),
        )
        reject(
            "container reuse",
            lambda: validate(
                altered(
                    ledger,
                    lambda d: d["entries"][1].update(
                        container_id=rows[0]["container_id"]
                    ),
                )
            ),
        )
        reject(
            "overlapping launches",
            lambda: validate(
                altered(ledger, lambda d: d["entries"][1].update(started_ns=3))
            ),
        )
        reject(
            "cache artifact substitution",
            lambda: validate(
                altered(ledger, lambda d: d["entries"][1].update(cache_sha256="wrong"))
            ),
        )
        reject(
            "unexpected seeded cold cache",
            lambda: validate(
                altered(
                    ledger, lambda d: d["entries"][1].update(cache_seed_sha256="wrong")
                )
            ),
        )
    result = {
        "status": "passed",
        "cpu_only": True,
        "historical_records_used_for_schema_only": True,
        "analyzer_sha256": a.sha(here / "analyze.py"),
        "control_sha256": a.sha(__file__),
        "checks": passed,
        "count": len(passed),
    }
    if args.output:
        require = a.require
        require(not args.output.exists(), "refuse to overwrite control result")
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
