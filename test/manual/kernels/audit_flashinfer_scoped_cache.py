"""Qualify one context-owned graph cache alongside actual legacy model startup.

Run cold, then fresh-process warm with identical legacy and managed roots.
This manual lifecycle qualification makes no performance or quality claim.
"""

import argparse
import contextlib
import dataclasses
import json
import time
from pathlib import Path
from unittest.mock import patch

import audit_flashinfer_replay_coverage as base


def json_files(root):
    return {
        str(path.relative_to(root)): {
            "sha256": base.digest(path.read_bytes()),
            "contents": json.loads(path.read_text()),
        }
        for path in sorted(root.rglob("*.json"))
    }


def legacy_snapshot(tuner, cache_dir):
    """Legacy winners/policies only; identified partitions intentionally coexist."""
    return {
        "inventory": base.cache_inventory(tuner),
        "files": base.cache_files(cache_dir),
        "file_policies": sorted(tuner._file_config_policies.items()),
        "memory_policies": sorted(
            (key.file_key, policy)
            for (identity, key), policy in tuner._profiling_cache_policies.items()
            if identity is None
        ),
        "excluded_file_keys": sorted(tuner._excluded_file_keys.get(None, ())),
        "invalidated_file_keys": sorted(tuner._invalidated_file_keys),
    }


def scoped_state(tuner):
    state = base.tuner_state(tuner)
    store = tuner._active_managed_store
    state.update(
        in_v2_context=tuner._in_v2_context,
        cache_scope=tuner._v2_local.cache_scope,
        strict_replay=tuner._v2_local.strict_replay,
        isolate_legacy=tuner._isolate_legacy_configs,
        store_root=str(store.root) if store else None,
        store_env_hash=store.env_hash if store else None,
        policy_fields=(
            tuner._effective_measure_policy.manifest_fields()
            if tuner._effective_measure_policy
            else {}
        ),
    )
    return state


def coverage_delta(report, before):
    after = dataclasses.asdict(report)
    return {
        "max_records": after["max_records"],
        **{
            name: after[name] - before[name]
            for name in ("hits", "misses", "skipped", "dropped_records")
        },
        "records": after["records"][len(before["records"]) :],
    }


def require_hits(report):
    if (
        report["hits"] <= 0
        or report["misses"]
        or report["skipped"]
        or report["dropped_records"]
        or report["hits"] != len(report["records"])
        or any(row["status"] != "hit" for row in report["records"])
    ):
        raise RuntimeError("Prepared forward did not consume complete cached coverage")


def decisions(report):
    return [
        (row["op"], row["input_shapes"], row["runner"], row["selected_tactic"])
        for row in report["records"]
    ]


def numerical_checks(original, bench, tokenizer, observer):
    """Report the inherited actual decode calls without changing their routing."""
    original_decode = bench.decode
    rows = observer.result["numerical_reports"] = []

    def decode(*args, **kwargs):
        before_state = scoped_state(observer.tuner)
        before_cache = legacy_snapshot(observer.tuner, observer.args.cache_dir)
        before_profiles, before_replays = len(observer.profiles), observer.replays
        report = observer.coverage(max_records=8192)
        previous_observing = observer.observing
        observer.observing = True
        try:
            with observer.autotune_v2(
                mode="replay",
                persistent_cache=False,
                cache_scope="context",
                replay_report=report,
            ):
                returned = original_decode(*args, **kwargs)
        finally:
            observer.observing = previous_observing
            row = {
                "coverage": dataclasses.asdict(report),
                "replay_delta": observer.replays - before_replays,
                "profile_delta": len(observer.profiles) - before_profiles,
                "state_before": before_state,
                "state_after": scoped_state(observer.tuner),
                "cache_before": before_cache,
                "cache_after": legacy_snapshot(observer.tuner, observer.args.cache_dir),
            }
            rows.append(row)
        if (
            row["state_before"] != row["state_after"]
            or row["cache_before"] != row["cache_after"]
            or row["profile_delta"]
        ):
            raise RuntimeError("Numerical replay changed bare legacy state")
        if row["replay_delta"] == 0:
            row["arm"] = "eager"
            require_hits(row["coverage"])
            if decisions(row["coverage"]) != decisions(
                observer.result["scoped_lifecycle"]["legacy_after"]
            ):
                raise RuntimeError(
                    "Bare numerical eager calls did not reuse legacy winners"
                )
        elif row["replay_delta"] == 1:
            row["arm"] = "graph"
            if report.hits or report.misses or report.skipped or report.records:
                raise RuntimeError(
                    "Retained graph replay unexpectedly selected Python tactics"
                )
        else:
            raise RuntimeError("Unexpected numerical decode route")
        return returned

    with patch.object(bench, "decode", decode):
        return original(bench, tokenizer, observer)


class Observation(base.Observation):
    def __init__(self, args, result):
        super().__init__(args, result)
        from flashinfer.autotune_cache import MeasurementPolicy

        self.policy = MeasurementPolicy(execution_mode="cuda_graph")
        self.capture_count = 0

    def context(self, root, mode, report=None, strict=False):
        return self.autotune_v2(
            mode=mode,
            cache_root=root,
            cache_scope="context",
            measurement_policy=self.policy,
            replay_report=report,
            strict_replay=strict,
        )

    def forward_record(self, label, forward_fn, report):
        before_report = dataclasses.asdict(report)
        row = {
            "stage": label,
            "capturing": self.torch.cuda.is_current_stream_capturing(),
            "state_before": scoped_state(self.tuner),
            "profile_count_before": len(self.profiles),
        }
        self.result["capture_records"].append(row)
        previous = self.stage, self.observing
        self.stage, self.observing = label, True
        try:
            return forward_fn()
        finally:
            self.stage, self.observing = previous
            row.update(
                coverage=coverage_delta(report, before_report),
                state_after=scoped_state(self.tuner),
                profile_count_after=len(self.profiles),
            )
            if row["profile_count_after"] != row["profile_count_before"]:
                raise RuntimeError("Replay unexpectedly profiled")
            if row["state_after"] != row["state_before"]:
                raise RuntimeError("Forward changed effective scope controls")

    def legacy_forward(self, label, forward_fn, reset):
        report = self.coverage(max_records=8192)
        before = scoped_state(self.tuner)
        base.assert_observable(before)
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
        if scoped_state(self.tuner) != before:
            raise RuntimeError("Legacy report scope did not restore controls")
        value = dataclasses.asdict(report)
        require_hits(value)
        if any(row["cache_identity"] is not None for row in value["records"]):
            raise RuntimeError("Legacy proof consumed an identified cache")
        return value

    def capture(self, original, backend, shape_key, forward_fn, *args, **kwargs):
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        runner = backend._cuda_graph_runner
        if (
            type(runner) is not DecodeCudaGraphRunner
            or runner.model_runner.spec_algorithm.is_speculative()
            or self.capture_count
        ):
            raise RuntimeError("Expected one ordinary batch-one decode capture")
        self.capture_count += 1
        hook = kwargs.get("post_warmup_hook")
        if len(args) > 1:
            hook = args[1]

        def reset():
            self.torch.cuda.synchronize()
            if hook is not None:
                hook()
            self.torch.cuda.synchronize()

        lifecycle = self.result["scoped_lifecycle"] = {
            "shape_key": dataclasses.asdict(shape_key),
            "legacy_entry_state": scoped_state(self.tuner),
            "legacy_entry_cache": legacy_snapshot(self.tuner, self.args.cache_dir),
            "legacy_startup_profile_count": len(self.profiles),
        }
        lifecycle["legacy_before"] = self.legacy_forward(
            "legacy_before", forward_fn, reset
        )
        legacy_state = scoped_state(self.tuner)
        legacy_cache = legacy_snapshot(self.tuner, self.args.cache_dir)
        lifecycle["legacy_reference_state"] = legacy_state
        lifecycle["legacy_reference_cache"] = legacy_cache

        negative_report = self.coverage(max_records=8192)
        negative = lifecycle["negative"] = {
            "profile_count_before": len(self.profiles),
            "files_before": json_files(self.args.negative_root),
        }
        try:
            with self.context(self.args.negative_root, "replay", negative_report, True):
                negative["state_inside"] = scoped_state(self.tuner)
                self.forward_record("strict_negative", forward_fn, negative_report)
        except RuntimeError as error:
            if "autotune strict replay miss" not in str(error):
                raise
            negative["error"] = {"type": type(error).__name__, "message": str(error)}
        else:
            raise RuntimeError(
                "Empty managed identity unexpectedly reused legacy winners"
            )
        finally:
            reset()
            negative.update(
                coverage=dataclasses.asdict(negative_report),
                profile_count_after=len(self.profiles),
                files_after=json_files(self.args.negative_root),
                state_after=scoped_state(self.tuner),
                legacy_after=legacy_snapshot(self.tuner, self.args.cache_dir),
            )
        if (
            negative["profile_count_after"] != negative["profile_count_before"]
            or negative["files_after"] != negative["files_before"]
            or negative["state_after"] != legacy_state
            or negative["legacy_after"] != legacy_cache
            or negative_report.hits
            or negative_report.misses != 1
        ):
            raise RuntimeError(
                "Strict negative changed legacy state or published a winner"
            )

        tune = lifecycle["tune"] = {
            "profile_count_before": len(self.profiles),
            "files_before": json_files(self.args.managed_root),
        }
        previous_stage = self.stage
        self.stage = "scoped_tune"
        try:
            with self.context(self.args.managed_root, "tune"):
                tune["state_inside"] = scoped_state(self.tuner)
                forward_fn()
        finally:
            self.stage = previous_stage
            reset()
            tune.update(
                profile_count_after=len(self.profiles),
                files_after=json_files(self.args.managed_root),
                state_after=scoped_state(self.tuner),
            )
        if tune["state_after"] != legacy_state:
            raise RuntimeError("Managed tuning left an ambient scope attached")
        if (
            self.args.phase == "cold"
            and tune["profile_count_after"] <= tune["profile_count_before"]
        ):
            raise RuntimeError(
                "Cold managed preparation did not profile its own identity"
            )
        if self.args.phase == "warm" and (
            tune["profile_count_after"] != tune["profile_count_before"]
            or tune["files_after"] != tune["files_before"]
        ):
            raise RuntimeError(
                "Warm managed preparation did not reuse preserved entries"
            )
        if legacy_snapshot(self.tuner, self.args.cache_dir) != legacy_cache:
            raise RuntimeError("Managed tuning changed legacy winners or policies")

        report = self.coverage(max_records=8192)
        strict = lifecycle["strict"] = {
            "state_before": scoped_state(self.tuner),
            "files_before": json_files(self.args.managed_root),
            "profile_count_before": len(self.profiles),
        }
        first_record = len(self.result["capture_records"])

        def observed_forward():
            label = (
                "scoped_capture"
                if self.torch.cuda.is_current_stream_capturing()
                else "scoped_warmup"
            )
            return self.forward_record(label, forward_fn, report)

        try:
            with self.context(self.args.managed_root, "replay", report, True):
                strict["state_inside"] = scoped_state(self.tuner)
                if (
                    strict["state_inside"]["winner_identity"]
                    != tune["state_inside"]["winner_identity"]
                ):
                    raise RuntimeError(
                        "Tune and capture selected different managed identities"
                    )
                original(backend, shape_key, observed_forward, *args, **kwargs)
        finally:
            strict.update(
                coverage=dataclasses.asdict(report),
                records=self.result["capture_records"][first_record:],
                state_after=scoped_state(self.tuner),
                files_after=json_files(self.args.managed_root),
                profile_count_after=len(self.profiles),
            )
        require_hits(strict["coverage"])
        if (
            [row["capturing"] for row in strict["records"]] != [False, False, True]
            or strict["profile_count_before"] != strict["profile_count_after"]
            or strict["files_before"] != strict["files_after"]
            or strict["state_after"] != legacy_state
        ):
            raise RuntimeError("Strict capture lifecycle or restoration failed")
        for row in strict["records"]:
            require_hits(row["coverage"])
            if decisions(row["coverage"]) != decisions(
                strict["records"][0]["coverage"]
            ):
                raise RuntimeError("Warmup and capture consumed different winners")
        lifecycle["legacy_after"] = self.legacy_forward(
            "legacy_after", forward_fn, reset
        )
        lifecycle["legacy_exit_state"] = scoped_state(self.tuner)
        lifecycle["legacy_exit_cache"] = legacy_snapshot(
            self.tuner, self.args.cache_dir
        )
        lifecycle["legacy_restored"] = (
            lifecycle["legacy_exit_state"] == legacy_state
            and lifecycle["legacy_exit_cache"] == legacy_cache
            and decisions(lifecycle["legacy_before"])
            == decisions(lifecycle["legacy_after"])
        )
        if not lifecycle["legacy_restored"]:
            raise RuntimeError("Bare legacy lookup changed after scoped graph capture")

    @contextlib.contextmanager
    def install(self):
        from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
            FullCudaGraphBackend,
        )

        original = FullCudaGraphBackend.capture_one
        with (
            super().install(),
            patch.object(
                FullCudaGraphBackend,
                "capture_one",
                lambda backend, key, forward, *args, **kwargs: self.capture(
                    original, backend, key, forward, *args, **kwargs
                ),
            ),
        ):
            yield


def run(args, result):
    result["managed_before_startup"] = json_files(args.managed_root)
    if bool(result["managed_before_startup"]) != (args.phase == "warm"):
        raise RuntimeError("Managed root must be empty cold and populated warm")
    if json_files(args.negative_root):
        raise RuntimeError("Strict negative needs a fresh empty root")
    original_numerics = base.numerical_checks
    with (
        patch.object(base, "Observation", Observation),
        patch.object(
            base, "numerical_checks", lambda *a: numerical_checks(original_numerics, *a)
        ),
    ):
        base.run(args, result)
    result["managed_after_numerics"] = json_files(args.managed_root)
    result["negative_after_numerics"] = json_files(args.negative_root)
    if (
        result["managed_after_numerics"]
        != result["scoped_lifecycle"]["strict"]["files_after"]
    ):
        raise RuntimeError("Bare numerical execution changed the managed store")
    if args.phase == "warm" and result["profile_calls"]:
        raise RuntimeError("Fresh warm process unexpectedly profiled")
    if any(
        row["stage"] not in ("startup", "scoped_tune")
        for row in result["profile_calls"]
    ):
        raise RuntimeError("Profiling occurred outside prepared startup/tuning")
    numerics = result["numerics"]
    if (
        not all(row["allclose"] and row["argmax_equal"] for row in numerics["checks"])
        or not numerics["changed_output"]
        or not numerics["restored_output_allclose"]
    ):
        raise RuntimeError("Retained scoped graph failed eager execution consistency")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("cold", "warm"), required=True)
    for name in (
        "model-path",
        "cache-dir",
        "managed-root",
        "negative-root",
        "provenance",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    roots = [
        args.cache_dir.resolve(),
        args.managed_root.resolve(),
        args.negative_root.resolve(),
    ]
    if any(
        a == b or a in b.parents or b in a.parents
        for i, a in enumerate(roots)
        for b in roots[i + 1 :]
    ):
        parser.error("Legacy, managed and negative roots must be disjoint")
    if args.output.exists():
        parser.error("Output already exists")
    result = {
        "schema": 1,
        "study": "actual_model_scoped_graph_cache",
        "phase": args.phase,
        "status": "started",
        "started_ns": time.time_ns(),
        "model_id": base.MODEL_ID,
        "model_revision": base.MODEL_REVISION,
        "provenance": json.loads(args.provenance.read_text()),
        "provenance_sha256": base.digest(args.provenance.read_bytes()),
        "worker_sha256": base.digest(Path(__file__).read_bytes()),
        "parent_worker_sha256": base.digest(Path(base.__file__).read_bytes()),
        "capture_records": [],
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
