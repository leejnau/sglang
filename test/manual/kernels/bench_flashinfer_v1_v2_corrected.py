"""Matched legacy/v2 startup with the corrected delay binding in both arms.

Reuse the qualified MLP facade, numerical checks and independent default graph.
Only the inner startup context differs. No finalist confirmation is performed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bench_flashinfer_mlp_selection as parent
import flashinfer_autotune_delay_binding as binding
from bench_flashinfer_objective_values import require, sha

PARENT_SHA = "ffd68fd070fc330d46d876f1892a236748aed781c5f8d2097542465e910a6ae4"
PROFILE_MS = (1, 2, 4, 8, 16, 32, 64, 128, 144)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("v1", "v2"), required=True)
    parser.add_argument("--phase", choices=("cold", "warm"), required=True)
    parser.add_argument("--trial", type=int, choices=range(8), required=True)
    parser.add_argument("--mode", choices=("admission", "timing"), required=True)
    parser.add_argument(
        "--capture-order", choices=("natural-first", "default-first"), required=True
    )
    parser.add_argument(
        "--mode-order", choices=("eager-first", "graph-first"), required=True
    )
    for name in (
        "plan",
        "schedule",
        "manifest",
        "fixture",
        "fixture-manifest",
        "cache",
        "output",
        "trace",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    require(args.phase == "cold" or args.mode == "admission", "warm timing is excluded")
    args.blocks, args.iterations, args.warmup_ms, args.min_sample_ms = 12, 20, 500, 20
    args.seed, args.oracle = 19, False
    return args


def assignment(schedule, args):
    matches = [
        row
        for row in schedule["entries"]
        if (row["trial"], row["arm"]) == (args.trial, args.arm)
    ]
    require(len(matches) == 1, "schedule must identify exactly one paired arm")
    entry = matches[0]
    require(
        entry["capture_order"] == args.capture_order
        and entry["mode_order"] == args.mode_order,
        "requested capture/mode order differs from frozen schedule",
    )
    orders = schedule["round_schedules"][str(args.trial)]
    for mode, count in (("eager", 12), ("cuda_graph", 24)):
        pairs = orders[mode]
        require(
            len(pairs) == count
            and all(sorted(pair) == sorted(parent.ROLES) for pair in pairs)
            and sum(pair[0] == "natural" for pair in pairs) == count // 2,
            "within-process pairs must be complete and balanced",
        )
    return entry, orders


def startup_context(arm, original, **options):
    require(arm in ("v1", "v2"), "unknown startup arm")
    require(
        options == {"mode": "tune", "persistent_cache": False},
        "actual SGLang startup v2 boundary changed",
    )
    return contextlib.nullcontext() if arm == "v1" else original(**options)


def validate_profiles(rows, arm, phase, contract):
    if phase == "warm":
        require(not rows, "warm startup profiled")
        return {}
    expected = {
        str(m): [
            t for t in contract["candidate_order"][str(m)] if arm == "v2" or t != -1
        ]
        for m in PROFILE_MS
    }
    actual = {str(m): [r["tactic"] for r in rows if r["m"] == m] for m in PROFILE_MS}
    count = 81 if arm == "v2" else 72
    require(len(rows) == count and actual == expected, "candidate enumeration differs")
    keys = [
        key
        for index, key in enumerate(contract["profile_keys"])
        if arm == "v2" or index % 9 != 0
    ]
    require([r["key"] for r in rows] == keys, "profile cache keys differ")
    require(
        all(
            r["status"] == "finite"
            and math.isfinite(r["profiler_ms"])
            and r["profiler_ms"] > 0
            for r in rows
        ),
        "a candidate failed or returned an invalid measurement",
    )
    return actual


class Experiment(parent.Experiment):
    def __init__(self, args):
        require(sha(parent.__file__) == PARENT_SHA, "qualified MLP worker changed")
        # The old worker has a six-block schedule; all runtime methods are reused.
        with patch.object(parent, "assignment", assignment):
            super().__init__(args)
        root = Path(__file__).resolve().parents[3]
        contract = self.plan["candidate_contract"]
        path = root / contract["path"]
        require(sha(path) == contract["sha256"], "candidate contract changed")
        self.contract = json.loads(path.read_text())
        self.result.update(
            study="mlp_v1_v2_corrected",
            worker_sha256=sha(__file__),
            parent_worker_sha256=PARENT_SHA,
            correction=binding.attest("patched"),
            startup_protocol={},
            confirmation_performed=False,
            limits=self.plan["limits"],
        )

    def startup(self):
        from sglang.srt.model_executor.runner import flashinfer_autotune as startup
        from sglang.srt.runtime_context import get_context, get_parallel

        self.startup_source = Path(startup.__file__).resolve()
        require(
            self.startup_source.is_relative_to(
                Path(__file__).resolve().parents[3] / "python"
            ),
            "SGLang startup came from another checkout",
        )
        original_v2 = self.flashinfer.autotune_v2

        def selected_context(**options):
            return startup_context(self.args.arm, original_v2, **options)

        record = self.result["startup_protocol"]
        record.update(
            wrapper_sha256=sha(self.startup_source),
            context_intervention="inner_v2_context_only",
            before=self.scope_state(),
            before_cache_sha256=self.cache_before,
        )
        self.stage = "startup"
        self.allow_profiling = self.args.phase == "cold"
        try:
            with (
                get_context().override_server_args(
                    moe_runner_backend="triton", flashinfer_autotune_skip_ops=[]
                ),
                get_parallel().override(tp_group=SimpleNamespace(world_size=1)),
                patch.object(
                    startup, "flashinfer_autotune_cache_path", lambda _: self.args.cache
                ),
                patch.object(self.flashinfer, "autotune_v2", selected_context),
            ):
                self.torch.cuda.synchronize()
                record["started_ns"] = time.perf_counter_ns()
                with (
                    self.trace_lookup(),
                    startup.flashinfer_autotune_context(
                        self.model_runner, run_lm_head=False
                    ),
                ):
                    record["inside"] = self.scope_state()
                    self.operation()
                self.torch.cuda.synchronize()
                record["ended_ns"] = time.perf_counter_ns()
                self.startup_ms = (record["ended_ns"] - record["started_ns"]) / 1e6
        finally:
            self.allow_profiling = False
        self.startup_profile_attempts = self.profile_attempts
        self.assert_bare()
        self.cache_after_startup = sha(self.args.cache)
        require(
            self.args.phase == "cold" or self.cache_before == self.cache_after_startup,
            "warm startup changed the saved cache",
        )
        record.update(
            after=self.scope_state(),
            candidate_order=validate_profiles(
                self.profiles, self.args.arm, self.args.phase, self.contract
            ),
            after_cache_sha256=self.cache_after_startup,
            profile_attempts=self.profile_attempts,
        )
        inside = record["inside"]
        require(
            inside["active_v2"] == (self.args.arm == "v2")
            and inside["is_tuning_mode"]
            and inside["on_forward_stream"]
            and inside["measurement_policy"] is None
            and not inside["managed_store"],
            "actual startup context or stream differs",
        )

    def measure_modes(self):
        # Admission/restart processes verify real numerics and capture only.
        if self.args.mode == "timing":
            super().measure_modes()


def main(argv=None):
    args = parse_args(argv)
    require(not args.output.exists(), "refusing to overwrite an attempted result")
    experiment = None
    try:
        import torch

        with torch.inference_mode():
            experiment = Experiment.__new__(Experiment)
            experiment.__init__(args)
            result = experiment.run()
    except Exception as error:
        traceback.print_exc()
        result = getattr(experiment, "result", {"pid": os.getpid(), "arm": args.arm})
        result.update(status="error", error=f"{type(error).__name__}: {error}")
        if experiment is not None:
            result["partial_evidence"] = {
                key: getattr(experiment, key)
                for key in (
                    "profiles",
                    "selections",
                    "correctness",
                    "profile_attempts",
                    "profile_protocol_errors",
                    "forbidden_profile_attempts",
                )
                if hasattr(experiment, key)
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
