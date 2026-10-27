"""Qualify configured FP8 skips through the research graph-region handoff.

One cold TP1 dense model run; no winner is expected in either cache. The real
startup wrapper, region adapter, backend capture and numerical helpers run with
the configured skip. This checks lifecycle/correctness, not performance or warm
cache reuse. Selection records distinguish intentional skips from bare misses.
"""

import argparse
import contextlib
import dataclasses
import importlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import audit_flashinfer_replay_coverage as base
import audit_flashinfer_scoped_cache as scoped
from audit_flashinfer_scoped_handoff import ScopedGraphRegion

OP = "fp8_gemm"
CALLS_PER_FORWARD = 128


def require_report(report, status):
    counter = "skipped" if status == "skipped" else "misses"
    if (
        report[counter] != CALLS_PER_FORWARD
        or report["hits"]
        or report["dropped_records"]
        or report["skipped" if counter == "misses" else "misses"]
        or len(report["records"]) != CALLS_PER_FORWARD
        or any(
            row["op"] != OP
            or row["status"] != status
            or row["selected_tactic"] != "-1"
            or row["source"] != ("skip_ops" if status == "skipped" else "fallback")
            for row in report["records"]
        )
    ):
        raise RuntimeError(f"Expected {CALLS_PER_FORWARD} {status} default selections")


def require_defaults(report):
    require_report(report, "miss")
    if any(row["cache_identity"] is not None for row in report["records"]):
        raise RuntimeError("Bare default selection used an identified cache")


class Observation(scoped.Observation):
    def __init__(self, args, result):
        super().__init__(args, result)
        self.selections = result["selection_calls"] = []

    def legacy_forward(self, label, forward_fn, reset):
        before = scoped.scoped_state(self.tuner)
        base.assert_observable(before)
        report = self.coverage(max_records=8192)
        try:
            with self.autotune_v2(
                mode="replay",
                persistent_cache=False,
                cache_scope="context",
                replay_report=report,
            ):
                self.forward_record(label, forward_fn, report)
        finally:
            reset()
        if scoped.scoped_state(self.tuner) != before:
            raise RuntimeError("Neutral default lookup did not restore its controls")
        value = dataclasses.asdict(report)
        require_defaults(value)
        return value

    def capture(self, original, backend, shape_key, forward_fn, *args, **kwargs):
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )
        from sglang.srt.model_executor.runner.flashinfer_autotune import (
            get_flashinfer_autotune_skip_ops,
        )
        from sglang.srt.runtime_context import get_exec

        runner = backend._cuda_graph_runner
        if type(runner) is not DecodeCudaGraphRunner or self.capture_count:
            raise RuntimeError("Expected one ordinary batch-one decode capture")
        self.capture_count += 1
        configured = get_exec().kernel.flashinfer_autotune_skip_ops
        resolved = get_flashinfer_autotune_skip_ops(runner.model_runner)
        if configured != [OP] or resolved != {OP}:
            raise RuntimeError("Actual resolved startup skips differ from the fixture")
        region = ScopedGraphRegion.create(runner.model_runner, self.args.managed_root)
        if region is None or region.spec.skip_ops != frozenset({OP}):
            raise RuntimeError("Research region did not retain the configured skip")
        hook = args[1] if len(args) > 1 else kwargs.get("post_warmup_hook")
        resets = []

        def reset():
            self.torch.cuda.synchronize()
            if hook is not None:
                hook()
            self.torch.cuda.synchronize()
            resets.append({"hook_present": hook is not None})

        lifecycle = self.result["scoped_lifecycle"] = {
            "shape_key": dataclasses.asdict(shape_key),
            "configured_skips": configured,
            "resolved_skips": sorted(resolved),
            "region_root": str(region.spec.root),
            "region_policy": region.spec.policy.manifest_fields(),
            "legacy_entry_state": scoped.scoped_state(self.tuner),
            "legacy_entry_cache": scoped.legacy_snapshot(
                self.tuner, self.args.cache_dir
            ),
            "reset_calls": resets,
        }
        lifecycle["legacy_before"] = self.legacy_forward(
            "legacy_before", forward_fn, reset
        )
        state = scoped.scoped_state(self.tuner)
        legacy = scoped.legacy_snapshot(self.tuner, self.args.cache_dir)
        managed = scoped.json_files(self.args.managed_root)
        if managed or legacy["inventory"]["memory"] or legacy["inventory"]["loaded"]:
            raise RuntimeError("Skip qualification must begin without cached winners")
        lifecycle["legacy_reference_state"] = state
        lifecycle["legacy_reference_cache"] = legacy
        first_record = len(self.result["capture_records"])

        def prepare():
            previous = self.stage, self.observing
            self.stage, self.observing = "scoped_tune", True
            first = len(self.selections)
            tune = lifecycle["tune"] = {
                "state_inside": scoped.scoped_state(self.tuner),
                "profile_count_before": len(self.profiles),
                "files_before": scoped.json_files(self.args.managed_root),
            }
            try:
                forward_fn()
            finally:
                self.stage, self.observing = previous
                tune["selections"] = self.selections[first:]
                tune["state_after_forward"] = scoped.scoped_state(self.tuner)
                tune["profile_count_after"] = len(self.profiles)
                tune["files_after"] = scoped.json_files(self.args.managed_root)
            if (
                len(tune["selections"]) != CALLS_PER_FORWARD
                or any(
                    not row["skip_active"]
                    or not row["runner_is_first"]
                    or row["selected_tactic"] != "-1"
                    or row["op"] != OP
                    for row in tune["selections"]
                )
                or tune["state_inside"] != tune["state_after_forward"]
                or tune["profile_count_after"]
                or tune["files_after"]
            ):
                raise RuntimeError("Actual graph preparation did not preserve skips")

        def capture():
            lifecycle["replay_state_inside"] = scoped.scoped_state(self.tuner)
            if (
                lifecycle["replay_state_inside"]["winner_identity"]
                != lifecycle["tune"]["state_inside"]["winner_identity"]
            ):
                raise RuntimeError("Preparation and capture used different identities")

            def forward():
                stage = (
                    "scoped_capture"
                    if self.torch.cuda.is_current_stream_capturing()
                    else "scoped_warmup"
                )
                return self.forward_record(stage, forward_fn, region.report)

            return original(backend, shape_key, forward, *args, **kwargs)

        returned = region.run(prepare, capture, reset)
        lifecycle["complete"] = region.complete
        records = self.result["capture_records"][first_record:]
        if [row["capturing"] for row in records] != [False, False, True]:
            raise RuntimeError("Expected original two warmups then CUDA capture")
        for row in records:
            require_report(row["coverage"], "skipped")
            if scoped.decisions(row["coverage"]) != scoped.decisions(
                records[0]["coverage"]
            ):
                raise RuntimeError("Warmups and capture selected different defaults")
        lifecycle["strict"] = {
            "coverage": dataclasses.asdict(region.report),
            "records": records,
            "files_after": scoped.json_files(self.args.managed_root),
            "state_after": scoped.scoped_state(self.tuner),
        }
        lifecycle["legacy_after"] = self.legacy_forward(
            "legacy_after", forward_fn, reset
        )
        lifecycle["legacy_exit_state"] = scoped.scoped_state(self.tuner)
        lifecycle["legacy_exit_cache"] = scoped.legacy_snapshot(
            self.tuner, self.args.cache_dir
        )
        lifecycle["legacy_restored"] = (
            lifecycle["legacy_exit_state"] == state
            and lifecycle["strict"]["state_after"] == state
            and lifecycle["legacy_exit_cache"] == legacy
            and scoped.decisions(lifecycle["legacy_before"])
            == scoped.decisions(lifecycle["legacy_after"])
        )
        if (
            not region.complete
            or not lifecycle["legacy_restored"]
            or len(resets) != 3
            or self.profiles
            or scoped.json_files(self.args.managed_root)
        ):
            raise RuntimeError(
                "Skipped graph lifecycle changed cache or ambient controls"
            )
        return returned

    @contextlib.contextmanager
    def install(self):
        original = self.tuner_class.choose_one

        def choose(tuner, custom_op, runners, tuning_config, inputs, **kwargs):
            row = {
                "stage": self.stage,
                "op": custom_op,
                "input_shapes": tuner._get_input_sizes(inputs),
                "capturing": self.torch.cuda.is_current_stream_capturing(),
                "state": scoped.scoped_state(tuner),
                "skip_active": custom_op in tuner._effective_skip_ops,
            }
            selected, tactic = original(
                tuner, custom_op, runners, tuning_config, inputs, **kwargs
            )
            row.update(
                runner=type(selected).__name__,
                runner_is_first=selected is runners[0],
                selected_tactic=str(tactic),
            )
            self.selections.append(row)
            return selected, tactic

        with super().install(), patch.object(self.tuner_class, "choose_one", choose):
            yield


def run(args, result):
    from sglang.srt import server_args as server_args_module
    from sglang.srt.environ import envs

    constructor = server_args_module.ServerArgs

    def configured_args(**fields):
        # Restore the class symbol before real resolution/publication performs
        # introspection. The base worker already holds this one local factory.
        server_args_module.ServerArgs = constructor
        fields["flashinfer_autotune_skip_ops"] = [OP]
        result["server_args"] = fields.copy()
        return constructor(**fields)

    constructor_patch = patch.object(server_args_module, "ServerArgs", configured_args)
    original_numerics = base.numerical_checks
    with (
        envs.SGLANG_CACHE_DIR.override(str(args.cache_dir)),
        envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.override(True),
    ):
        # Bind transitive imports to the real class before the one-call factory.
        importlib.import_module("sglang.benchmark.one_batch")
        with (
            constructor_patch,
            patch.object(base, "Observation", Observation),
            patch.object(scoped, "require_hits", require_defaults),
            patch.object(
                base,
                "numerical_checks",
                lambda *values: scoped.numerical_checks(original_numerics, *values),
            ),
        ):
            base.run(args, result)
    result["managed_after_numerics"] = scoped.json_files(args.managed_root)
    startup_tuning = [
        row
        for row in result["selection_calls"]
        if row["stage"] == "startup" and row["state"]["is_tuning"]
    ]
    numerics = result["numerics"]
    if (
        not startup_tuning
        or any(not row["skip_active"] for row in startup_tuning)
        or any(
            row["op"] != OP
            or not row["runner_is_first"]
            or row["selected_tactic"] != "-1"
            for row in result["selection_calls"]
        )
        or result["startup"]["eligible_linear_layers"] != CALLS_PER_FORWARD
        or result["profile_calls"]
        or result["protocol_calls"]
        or result["managed_after_numerics"]
        or result["graph_replays"] != 3
        or result["cache_after_numerics"] != result["startup"]["files"]
        or result["inventory_after_numerics"] != result["startup"]["cache"]
        or result["final_state"] != result["startup"]["state"]
        or not all(
            row["allclose"] and row["argmax_equal"] for row in numerics["checks"]
        )
        or not numerics["changed_output"]
        or not numerics["restored_output_allclose"]
    ):
        raise RuntimeError("Configured-skip end-to-end qualification failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("cold",), default="cold")
    for name in ("model-path", "cache-dir", "managed-root", "provenance", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    legacy, managed = args.cache_dir.resolve(), args.managed_root.resolve()
    if legacy == managed or legacy in managed.parents or managed in legacy.parents:
        parser.error("Legacy and graph roots must be disjoint")
    if args.output.exists() or scoped.json_files(args.managed_root):
        parser.error("Output and managed root must be fresh")
    result = {
        "schema": 1,
        "study": "actual_model_scoped_skip_handoff",
        "phase": "cold",
        "status": "started",
        "started_ns": time.time_ns(),
        "model_id": base.MODEL_ID,
        "model_revision": base.MODEL_REVISION,
        "provenance": json.loads(args.provenance.read_text()),
        "provenance_sha256": base.digest(args.provenance.read_bytes()),
        "worker_sha256": base.digest(Path(__file__).read_bytes()),
        "parent_worker_sha256": base.digest(Path(base.__file__).read_bytes()),
        "scoped_worker_sha256": base.digest(Path(scoped.__file__).read_bytes()),
        "adapter_sha256": base.digest(
            Path(__file__).with_name("audit_flashinfer_scoped_handoff.py").read_bytes()
        ),
        "capture_records": [],
        "limits": "Single cold TP1 full-decode lifecycle and numerical comparison; no warm reuse, performance, model-quality, or production-adapter claim",
    }
    try:
        run(args, result)
    except BaseException as error:
        result["status"] = "error"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        result["finished_ns"] = time.time_ns()
        base.write_json(args.output, result)


if __name__ == "__main__":
    main()
