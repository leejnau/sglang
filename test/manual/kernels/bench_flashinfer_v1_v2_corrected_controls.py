"""CPU controls for the manual comparison's context and evidence boundaries."""

from __future__ import annotations

import contextlib
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bench_flashinfer_v1_v2_corrected as worker


def main():
    root = Path(__file__).resolve().parents[3]
    batch = root / (
        "test/manual/kernels/results/flashinfer_autotune_selection/"
        "20261028T030000Z-h100-v1-v2-corrected"
    )
    plan = json.loads((batch / "plan.json").read_text())
    schedule = json.loads((batch / "schedule.json").read_text())
    contract_path = root / plan["candidate_contract"]["path"]
    assert worker.sha(contract_path) == plan["candidate_contract"]["sha256"]
    contract = json.loads(contract_path.read_text())
    passed = []

    def reject(label, operation):
        try:
            operation()
        except (RuntimeError, ValueError, TypeError, KeyError):
            passed.append(label)
        else:
            raise AssertionError("invalid boundary accepted: " + label)

    for arm in ("v1", "v2"):
        calls = []

        @contextlib.contextmanager
        def original(**options):
            calls.append(("enter", options))
            try:
                yield
            finally:
                calls.append(("exit", options))

        options = {"mode": "tune", "persistent_cache": False}
        for fail in (False, True):
            calls.clear()
            try:
                with worker.startup_context(arm, original, **options):
                    if fail:
                        raise LookupError("body failed")
            except LookupError:
                assert fail
            assert calls == (
                [("enter", options), ("exit", options)] if arm == "v2" else []
            )
            passed.append(f"actual_context_{arm}_body_failure_{fail}")
        for label, wrong in (
            ("replay", {**options, "mode": "replay"}),
            ("managed_store", {**options, "persistent_cache": True}),
            ("policy", {**options, "measurement_policy": None}),
            ("missing_cache_toggle", {"mode": "tune"}),
        ):
            calls.clear()
            reject(
                f"context_rejects_{arm}_{label}",
                lambda wrong=wrong: worker.startup_context(arm, original, **wrong),
            )
            assert not calls
    reject(
        "unknown_context_arm",
        lambda: worker.startup_context("other", original, **options),
    )

    complete = []
    for m in worker.PROFILE_MS:
        for tactic in contract["candidate_order"][str(m)]:
            complete.append(
                {
                    "m": m,
                    "tactic": tactic,
                    "key": contract["profile_keys"][len(complete)],
                    "status": "finite",
                    "profiler_ms": 0.01,
                }
            )
    for arm in ("v1", "v2"):
        rows = copy.deepcopy(
            complete if arm == "v2" else [r for r in complete if r["tactic"] != -1]
        )
        worker.validate_profiles(rows, arm, "cold", contract)
        worker.validate_profiles([], arm, "warm", contract)
        passed.extend([f"qualified_cold_pool_{arm}", f"empty_warm_profiles_{arm}"])
        mutations = [
            ("missing_candidate", lambda r: r.pop(0)),
            ("wrong_key", lambda r: r[0].update(key="different cache key")),
            ("wrong_shape", lambda r: r[0].update(m=999)),
            ("reversed_order", lambda r: r.reverse()),
            ("failed_candidate", lambda r: r[0].update(status="exception")),
            ("nonfinite_time", lambda r: r[0].update(profiler_ms=float("nan"))),
            ("zero_time", lambda r: r[0].update(profiler_ms=0.0)),
        ]
        for label, mutate in mutations:
            broken = copy.deepcopy(rows)
            mutate(broken)
            reject(
                f"reject_{arm}_{label}",
                lambda broken=broken: worker.validate_profiles(
                    broken, arm, "cold", contract
                ),
            )
        reject(
            f"reject_warm_profiling_{arm}",
            lambda rows=rows: worker.validate_profiles(rows, arm, "warm", contract),
        )
    reject(
        "v1_cannot_inherit_v2_default",
        lambda: worker.validate_profiles(complete, "v1", "cold", contract),
    )

    for row in schedule["entries"]:
        args = SimpleNamespace(**row)
        selected, orders = worker.assignment(schedule, args)
        assert selected == row and len(orders["cuda_graph"]) == 24
        passed.append(f"frozen_assignment_{row['trial']}_{row['arm']}")
    args = SimpleNamespace(**schedule["entries"][0])
    for field, value in (
        ("capture_order", "other"),
        ("mode_order", "other"),
        ("arm", "other"),
    ):
        wrong = copy.copy(args)
        setattr(wrong, field, value)
        reject(
            f"wrong_assignment_{field}",
            lambda wrong=wrong: worker.assignment(schedule, wrong),
        )
    for mode in ("eager", "cuda_graph"):
        for label, alter in (
            (
                "unbalanced",
                lambda pairs: pairs.__setitem__(
                    slice(None), [["natural", "default"]] * len(pairs)
                ),
            ),
            ("missing_pair", lambda pairs: pairs.pop()),
            (
                "duplicate_role",
                lambda pairs: pairs.__setitem__(0, ["natural", "natural"]),
            ),
        ):
            broken = copy.deepcopy(schedule)
            alter(broken["round_schedules"][str(args.trial)][mode])
            reject(
                f"invalid_{mode}_{label}",
                lambda broken=broken: worker.assignment(broken, args),
            )
    broken = copy.deepcopy(schedule)
    broken["entries"].append(copy.deepcopy(broken["entries"][0]))
    reject("duplicate_process_assignment", lambda: worker.assignment(broken, args))

    # Exercise the inherited denial before the real profiler can run. A denied
    # attempt must remain counted even if an upstream caller catches the error.
    experiment = worker.Experiment.__new__(worker.Experiment)
    experiment.profile_attempts = 0
    experiment.allow_profiling = False
    experiment.stage = "startup"
    experiment.profiles = []
    experiment.forbidden_profile_attempts = []

    def forbidden_native(*args, **kwargs):
        raise AssertionError("real profiler invoked during warm reuse")

    experiment.original_profile = forbidden_native
    reject(
        "warm_denial_before_real_profiler",
        lambda: experiment.profile_hook(None, [], -1, None),
    )
    assert experiment.profile_attempts == 1
    assert experiment.forbidden_profile_attempts == ["startup"]
    passed.append("denied_attempt_not_lost")
    for mode in ("admission", "timing"):
        experiment.args = SimpleNamespace(mode=mode)
        with patch.object(worker.parent.Experiment, "measure_modes") as measurement:
            experiment.measure_modes()
            assert measurement.call_count == (1 if mode == "timing" else 0)
        passed.append(f"measurement_dispatch_{mode}")

    print(
        json.dumps(
            {
                "status": "passed",
                "cases": len(passed),
                "controls": passed,
                "limits": "CPU protocol boundaries only; CUDA source, cache reuse, numerics and measurement behavior require separate admission.",
            }
        )
    )


if __name__ == "__main__":
    main()
