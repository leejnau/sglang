"""Compare only the profiling objective on one frozen actual FP8 linear layer.

Both arms race the default candidate. The current objective leaves policy
unspecified; the graph objective changes only execution_mode. Real memory-only
policy scopes cover tuning and capture. Retained graph replay runs outside them.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import itertools
import json
import os
import random
import time
import traceback
from pathlib import Path
from unittest.mock import patch

from flashinfer_autotune_descriptor_qualification import tactic_json
from flashinfer_autotune_eager_setup import digest, positive_float
from flashinfer_autotune_graph_controls import (
    GraphControlExperiment,
    batch,
    device_status,
)
from flashinfer_autotune_linear_layer import tensor_sha

GRAPH_IDS = ("winner", "default")
POLICY_SOURCE_SHA256 = (
    "7be4f83162ce3afada8593f5acd863769356da74102e6762ece133548a4d8bda"
)


def schedule(trial, blocks=24):
    if trial not in range(6) or blocks != 24:
        raise ValueError("the protocol requires six trials and 24 paired blocks")
    rows = [
        list(order) for _ in range(12) for order in itertools.permutations(GRAPH_IDS)
    ]
    random.Random(5000 + trial).shuffle(rows)
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("auto", "cuda_graph"), required=True)
    parser.add_argument("--trial", type=int, choices=range(6), required=True)
    parser.add_argument("--seed", type=int, choices=(19,), default=19)
    for name in ("cache", "output", "manifest", "fixture", "fixture-manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--blocks", type=int, choices=(24,), default=24)
    parser.add_argument("--iterations", type=int, choices=(5120,), default=5120)
    parser.add_argument("--samples", type=int, choices=(15,), default=15)
    parser.add_argument("--warmup-ms", type=positive_float, default=500)
    parser.add_argument("--min-sample-ms", type=positive_float, default=20)
    parser.add_argument("--rewarm-ms", type=positive_float, default=20)
    args = parser.parse_args(argv)
    if (args.warmup_ms, args.min_sample_ms, args.rewarm_ms) != (500, 20, 20):
        parser.error("the protocol requires 500/20/20 ms warmup/calibration/rewarm")
    args.phase, args.oracle, args.mode = "cold", False, "timing"
    return args


class ObjectiveExperiment(GraphControlExperiment):
    def __init__(self, args):
        super().__init__(args)
        if [args.m, args.n, args.k] != [144, 4096, 1024]:
            raise RuntimeError("the frozen fused QKV fixture geometry changed")
        from flashinfer import MeasurementPolicy
        from flashinfer import autotune_cache as policy_module

        self.policy_source_sha256 = digest(Path(policy_module.__file__))
        if (
            self.policy_source_sha256 != POLICY_SOURCE_SHA256
            or MeasurementPolicy is not policy_module.MeasurementPolicy
            or self.flashinfer.autotune_v2 is not policy_module.autotune_v2
        ):
            raise RuntimeError(
                "the qualified measurement policy/context source changed"
            )

        self.policy = (
            None
            if args.arm == "auto"
            else MeasurementPolicy(execution_mode="cuda_graph")
        )
        self.original_v2 = self.flashinfer.autotune_v2
        self.scope_records, self.startup_api_calls = [], []
        self.profiling_input_groups, self.protocol_errors = {}, []
        self.trace = None
        self.controls = {
            "capture_order": [],
            "graphs": self.graph_rows,
            "sequential": [],
            "schedule": [],
            "blocks": [],
            "miss_scope": [],
            "common_iterations": args.iterations,
            "schedule_seed": 5000 + args.trial,
        }

    def policy_json(self):
        return dataclasses.asdict(self.policy) if self.policy is not None else None

    def scope_record(self, label, mode, event):
        self.scope_records.append(
            {
                "stage": label,
                "mode": mode,
                "event": event,
                "measurement_policy": (
                    dataclasses.asdict(self.tuner._effective_measure_policy)
                    if self.tuner._effective_measure_policy is not None
                    else None
                ),
                "manifest_fields": (
                    self.tuner._effective_measure_policy.manifest_fields()
                    if self.tuner._effective_measure_policy is not None
                    else {}
                ),
                "winner_identity": repr(self.tuner._winner_cache_identity()),
                "active_v2": self.tuner._in_v2_context,
                "is_tuning_mode": self.tuner.is_tuning_mode,
                "tuning_contexts": self.tuner._active_tuning_contexts,
                "managed_store": self.tuner._active_managed_store is not None,
                "on_forward_stream": (
                    self.torch.cuda.current_stream() == self.model_runner.forward_stream
                ),
                "on_caller_stream": (
                    self.torch.cuda.current_stream() == self.caller_stream
                ),
            }
        )

    @contextlib.contextmanager
    def policy_scope(self, mode, label):
        options = {"mode": mode, "persistent_cache": False}
        if self.policy is not None:
            options["measurement_policy"] = self.policy
        try:
            with self.original_v2(**options):
                self.scope_record(label, mode, "enter")
                try:
                    yield
                finally:
                    self.scope_record(label, mode, "before_exit")
        finally:
            self.scope_record(label, mode, "after_exit")

    @contextlib.contextmanager
    def startup_policy(self, **options):
        if options != {"mode": "tune", "persistent_cache": False}:
            raise RuntimeError("the actual wrapper changed its v2 invocation")
        self.startup_api_calls.append(
            {**options, "measurement_policy": self.policy_json()}
        )
        with self.policy_scope("tune", "startup"):
            yield

    def startup(self):
        # Change the public policy argument at the actual wrapper boundary,
        # retaining its v1 context, streams, skip controls and save behavior.
        with patch.object(self.flashinfer, "autotune_v2", self.startup_policy):
            super().startup()
        if self.protocol_errors:
            raise RuntimeError(
                "profiling protocol changed: " + repr(self.protocol_errors)
            )
        if len(self.startup_api_calls) != 1:
            raise RuntimeError("startup did not enter exactly one real v2 context")
        rows = json.loads(self.args.cache.read_text())
        entries = {key: value for key, value in rows.items() if key.startswith("(")}
        if self.tuner._file_configs:
            raise RuntimeError("fresh objective study unexpectedly loaded v1 entries")
        if self.args.arm == "cuda_graph" and entries:
            raise RuntimeError(
                "explicit graph winners leaked into legacy serialization"
            )
        self.incidental_legacy_entries = len(entries)

    def profile_hook(self, runner, inputs, tactic, config, *more, **kwargs):
        if not self.allow_profiling:
            # Preserve the inherited monotonic attempt guard before touching
            # tensors; capture-time hashing could otherwise mask this failure.
            return super().profile_hook(runner, inputs, tactic, config, *more, **kwargs)
        # Read defined profiling operands outside the original timing routine.
        # Output and workspace bytes are intentionally excluded.
        values = {}
        try:
            for name, tensor in zip(
                ("activation", "weight", "input_scale", "weight_scale"), inputs[:4]
            ):
                values[name] = {
                    "shape": list(tensor.shape),
                    "stride": list(tensor.stride()),
                    "dtype": str(tensor.dtype),
                    "sha256": tensor_sha(self.torch, tensor),
                }
        except Exception as error:
            self.protocol_errors.append({"input_fingerprint_error": str(error)})
            raise
        group = hashlib.sha256(
            json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.profiling_input_groups[group] = values
        settings = self.tuner_settings(config)
        expected_graph = self.args.arm == "cuda_graph"
        expected_settings = (3, 10, 5000, 100, False)
        actual_settings = tuple(
            settings[key]
            for key in (
                "warmup",
                "repeat",
                "stream_delay_micro_secs",
                "cuda_graph_delay_micro_secs",
                "use_global_timer",
            )
        )
        if (
            config.use_cuda_graph != expected_graph
            or config.use_cold_l2_cache
            or not self.tuner._in_v2_context
            or settings["effective_measure_policy"] != self.policy_json()
            or actual_settings != expected_settings
            or config.profiling_repeat is not None
            or config.cuda_graph_profile_replays != 1
        ):
            # The tuner can swallow candidate errors, so retain a fatal flag.
            self.protocol_errors.append({"stage": self.stage, "settings": settings})
        index = len(self.profiles)
        try:
            return super().profile_hook(runner, inputs, tactic, config, *more, **kwargs)
        finally:
            if len(self.profiles) > index:
                self.profiles[index].update(
                    profiling_input_group=group,
                    policy_identity=repr(self.tuner._winner_cache_identity()),
                    policy_manifest_fields=(
                        self.policy.manifest_fields() if self.policy is not None else {}
                    ),
                    default_race_enabled=self.tuner._in_v2_context,
                )

    def choose_hook(self, op, runners, config, inputs, **kwargs):
        result = super().choose_hook(op, runners, config, inputs, **kwargs)
        self.selections[-1].update(
            policy_identity=repr(self.tuner._winner_cache_identity()),
            measurement_policy=(
                dataclasses.asdict(self.tuner._effective_measure_policy)
                if self.tuner._effective_measure_policy is not None
                else None
            ),
            active_v2=self.tuner._in_v2_context,
            is_tuning_mode=self.tuner.is_tuning_mode,
        )
        return result

    def runtime_summary(self):
        super().runtime_summary()
        rows = [
            row
            for row in self.profiles
            if tuple(tuple(shape) for shape in row["normalized_profile"])
            == self.cache_key.nearest_profile
        ]
        if not any(
            row["tactic"] == -1 and row.get("profiler_ms") is not None for row in rows
        ):
            raise RuntimeError("the default candidate did not participate finitely")
        if not all(row["default_race_enabled"] for row in self.profiles):
            raise RuntimeError("a profile ran outside the v2 default race")

    def assert_bare_replay_state(self):
        if (
            self.tuner._in_v2_context
            or self.tuner._effective_measure_policy is not None
            or self.tuner.is_tuning_mode
            or self.tuner._active_tuning_contexts
            or self.tuner._active_managed_store is not None
            or self.native_cell.cell_contents is not self.native_module
            or self.torch.cuda.current_stream() != self.caller_stream
        ):
            raise RuntimeError(
                "replay timing inherited a policy, hook, store or stream"
            )

    def graph_window(self, graph_id, label):
        self.assert_bare_replay_state()
        attempts = self.profile_attempts
        result = super().graph_window(graph_id, label)
        self.assert_bare_replay_state()
        result.update(
            policy_scope_active=self.tuner._in_v2_context,
            tuning_mode=self.tuner.is_tuning_mode,
            profile_attempts_before=attempts,
            profile_attempts_after=self.profile_attempts,
        )
        return result

    def interleave(self):
        self.controls["schedule"] = schedule(self.args.trial, self.args.blocks)
        for index, order in enumerate(self.controls["schedule"]):
            group = {"index": index, "order": order, "measurements": []}
            for graph_id in order:
                self.assert_bare_replay_state()
                graph = self.graphs[graph_id]
                label = f"block_{index:02d}_{graph_id}"
                self.stage, self.out = label, graph.output
                before = device_status()
                started, calls = time.perf_counter_ns(), 0
                while True:
                    for _ in range(self.args.iterations):
                        graph.graph.replay()
                    self.torch.cuda.synchronize()
                    calls += self.args.iterations
                    elapsed = (time.perf_counter_ns() - started) / 1e6
                    if elapsed >= self.args.rewarm_ms:
                        break
                self.out.fill_(float("nan"))
                measured = batch(self.torch, graph.graph.replay, self.args.iterations)
                self.out = graph.output
                self.check_retained_output(graph_id, label)
                group["measurements"].append(
                    {
                        "graph_id": graph_id,
                        **measured,
                        "rewarm_actual_ms": elapsed,
                        "rewarm_calls": calls,
                        "gpu_before": before,
                        "gpu_after": device_status(),
                        "correctness_check": label,
                        "policy_scope_active": self.tuner._in_v2_context,
                        "tuning_mode": self.tuner.is_tuning_mode,
                    }
                )
            self.controls["blocks"].append(group)

    def run(self):
        with patch.object(self.tuner, "_profile_single_kernel", self.profile_hook):
            self.setup()
            self.startup()
            with self.policy_scope("replay", "natural_before"):
                winner, _selection, acceptance = self.observed_call(
                    "natural_before", tier="memory"
                )
                self.natural_tactic = winner
                self.natural = {
                    "tactic": tactic_json(winner),
                    "cache_hit": True,
                    "lookup_tier": "memory",
                    "acceptance": acceptance,
                }
                self.runtime_summary()
                with self.zero_input():
                    self.observed_call(
                        "natural_eager_changed_input", expected=winner, tier="memory"
                    )
                self.observed_call(
                    "natural_eager_restored_input", expected=winner, tier="memory"
                )
            order = (
                GRAPH_IDS if self.args.trial % 2 == 0 else tuple(reversed(GRAPH_IDS))
            )
            for graph_id in order:
                with self.policy_scope("replay", graph_id + "_capture_scope"):
                    with (
                        self.miss_scope()
                        if graph_id == "default"
                        else contextlib.nullcontext()
                    ):
                        expected, tier = (
                            (-1, "miss")
                            if graph_id == "default"
                            else (winner, "memory")
                        )
                        self.observed_call(
                            graph_id + "_before", expected=expected, tier=tier
                        )
                        self.retain_capture(graph_id, expected, tier)
                        self.graph_rows[-1].update(
                            policy_identity=repr(self.tuner._winner_cache_identity()),
                            measurement_policy=self.policy_json(),
                        )
            self.assert_bare_replay_state()
            self.controls["sequential"] = [
                self.graph_window(graph_id, "sequential_" + graph_id)
                for graph_id in GRAPH_IDS
            ]
            self.interleave()
            for graph_id in GRAPH_IDS:
                self.out = self.graphs[graph_id].output
                self.check_retained_output(graph_id, "final_" + graph_id)
            with self.policy_scope("replay", "natural_final"):
                self.observed_call("natural_final", expected=winner, tier="memory")
            self.observed_call(
                "bare_after_scope",
                expected=-1 if self.policy is not None else winner,
                tier="miss" if self.policy is not None else "memory",
            )
        self.assert_bare_replay_state()
        if (
            self.protocol_errors
            or self.forbidden_profile_attempts
            or self.profile_attempts != self.startup_profile_attempts
            or self.profile_attempts != len(self.profiles)
            or not self.cache_scopes_restored
            or self.operands != self.operand_digests()
            or digest(self.args.cache) != self.cache_after_startup
            or len({graph.output.data_ptr() for graph in self.graphs.values()}) != 2
        ):
            raise RuntimeError(
                "objective protocol, inputs, profiles or cache state changed"
            )
        result = super().result()
        result["worker_sha256"] = digest(__file__)
        result["pid"] = os.getpid()
        result["environment"]["helper_sha256"][
            "flashinfer_autotune_graph_controls.py"
        ] = digest(Path(__file__).with_name("flashinfer_autotune_graph_controls.py"))
        result["environment"]["measurement_policy"] = self.policy_json()
        result["environment"]["flashinfer_autotune_cache_sha256"] = (
            self.policy_source_sha256
        )
        result["profiling_input_groups"] = self.profiling_input_groups
        result["objective"] = {
            "requested_policy": self.policy_json(),
            "manifest_fields": self.policy.manifest_fields() if self.policy else {},
            "startup_api_calls": self.startup_api_calls,
            "scope_records": self.scope_records,
            "default_race_enabled_both_arms": True,
            "effective_hot_l2": True,
            "protocol_errors": self.protocol_errors,
            "replay_outside_all_contexts": True,
            "final_state": {
                "active_v2": self.tuner._in_v2_context,
                "measurement_policy": self.tuner._effective_measure_policy,
                "is_tuning_mode": self.tuner.is_tuning_mode,
                "tuning_contexts": self.tuner._active_tuning_contexts,
                "managed_store": self.tuner._active_managed_store is not None,
                "caller_stream_restored": (
                    self.torch.cuda.current_stream() == self.caller_stream
                ),
            },
        }
        result["cache"]["incidental_legacy_entries"] = self.incidental_legacy_entries
        result["cache"]["persistence_under_test"] = False
        result["scope_notes"] = [
            "Only execution_mode changes: auto omits measurement_policy, while cuda_graph passes MeasurementPolicy(execution_mode='cuda_graph'); hot L2 and all tuning counts remain unchanged.",
            "Both arms use the existing v2 default-candidate race and actual SGLang startup wrapper with the same fixed fixture, synthetic profiling-input generation, descriptor source and buckets.",
            "Defined profiling operands are fingerprinted outside the original profiler timing routine; output/workspace bytes are excluded. This observation is common to both arms.",
            "The real memory-only policy scope covers tuning and capture-time forwards. Timing replays retained graph objects outside all tuning/policy scopes with no lookup, native observer or output-copy hook.",
            "Winner capture consumes an actual memory hit; independent default capture reversibly removes effective winners/files and consumes a genuine miss.",
            "The wrapper retains its incidental v1 save behavior. No graph partition is projected into v1 JSON, no saved winner is reloaded, and no persistence behavior is qualified.",
            "This is one checkpoint-derived FP8 linear layer, not a full-model accuracy or serving-throughput benchmark.",
        ]
        return result


def main():
    args = parse_args()
    experiment = None
    try:
        import torch

        with torch.inference_mode():
            experiment = ObjectiveExperiment.__new__(ObjectiveExperiment)
            experiment.__init__(args)
            result = experiment.run()
    except Exception as error:
        traceback.print_exc()
        result = {
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "arm": args.arm,
            "trial": args.trial,
            "pid": os.getpid(),
            "partial_evidence": {
                name: getattr(experiment, name)
                for name in (
                    "profiles",
                    "profile_attempts",
                    "forbidden_profile_attempts",
                    "profiling_input_groups",
                    "protocol_errors",
                    "correctness",
                    "controls",
                    "scope_records",
                    "quantization_evidence",
                )
                if experiment is not None and hasattr(experiment, name)
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
