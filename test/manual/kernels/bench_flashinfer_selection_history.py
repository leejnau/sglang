"""Isolate retained startup history and inline profiling-operand fingerprints.

Every cell keeps the current unspecified policy and v2 default race. Synthetic
profiling inputs are never retained by this worker. The replay trace is collected
only after all latency samples, numerical controls and final cache checks.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import itertools
import json
import os
import random
import statistics
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import flashinfer_autotune_measurement_objective as objective_module
from flashinfer_autotune_descriptor_qualification import tactic_json
from flashinfer_autotune_eager_setup import digest, positive_float
from flashinfer_autotune_graph_controls import GraphControlExperiment
from flashinfer_autotune_linear_layer import LinearLayerExperiment
from flashinfer_autotune_measurement_objective import GRAPH_IDS, ObjectiveExperiment


def schedule(trial, blocks=24):
    if trial not in range(8) or blocks != 24:
        raise ValueError("the protocol requires eight trials and 24 paired blocks")
    rows = [
        list(order) for _ in range(12) for order in itertools.permutations(GRAPH_IDS)
    ]
    random.Random(6100 + trial).shuffle(rows)
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=int, choices=range(8), required=True)
    parser.add_argument("--history", type=int, choices=(0, 1), required=True)
    parser.add_argument("--fingerprint", type=int, choices=(0, 1), required=True)
    parser.add_argument(
        "--capture-order", choices=("winner-first", "default-first"), required=True
    )
    parser.add_argument("--seed", type=int, choices=(19,), default=19)
    for name in ("cache", "output", "manifest", "fixture", "fixture-manifest", "trace"):
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
    args.arm, args.phase, args.oracle, args.mode = "auto", "cold", False, "timing"
    return args


def kernel_signature(event):
    args = event.get("args", {})
    return {
        "name": event["name"],
        **{
            key: args.get(key)
            for key in (
                "device",
                "grid",
                "block",
                "registers per thread",
                "shared memory",
            )
        },
    }


class HistoryExperiment(ObjectiveExperiment):
    def __init__(self, args):
        super().__init__(args)
        self.history_graph, self.history_evidence = None, None
        self.controls["schedule_seed"] = 6100 + args.trial
        self.rng_states = []
        self.requested_order = (
            GRAPH_IDS
            if args.capture_order == "winner-first"
            else tuple(reversed(GRAPH_IDS))
        )

    def check_output(self, label):
        super().check_output(label)
        history = self.history_graph
        if history is not None and self.out.data_ptr() == history.output.data_ptr():
            if self.correctness[-1]["graph_id"] is not None:
                raise RuntimeError(
                    "history and primary graphs alias an output allocation"
                )
            self.correctness[-1]["graph_id"] = "pre_default"

    def retain_capture(self, graph_id, expected, tier):
        graph = super().retain_capture(graph_id, expected, tier)
        self.graph_rows[-1]["graph_object_id"] = id(graph.graph)
        return graph

    def rng_snapshot(self, stage):
        # CUDA is already initialized by real fixture setup. Reading its generator
        # state returns a CPU tensor; no random generation or GPU copy is requested.
        if not self.torch.cuda.is_initialized():
            raise RuntimeError("RNG observation would initialize CUDA")
        row = {"stage": stage, "cuda_device": str(self.a.device)}
        for name, state in (
            ("cpu", self.torch.get_rng_state()),
            ("cuda", self.torch.cuda.get_rng_state(self.a.device)),
        ):
            if state.device.type != "cpu":
                raise RuntimeError("RNG observation unexpectedly returned a GPU tensor")
            raw = state.numpy().tobytes()
            row[name] = {
                "nbytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "base64": base64.b64encode(raw).decode("ascii"),
            }
        self.rng_states.append(row)
        return row

    def cache_snapshot(self):
        return {
            "file_sha256": digest(self.args.cache)
            if self.args.cache.is_file()
            else None,
            "winner_sha256": self.map_sha(self.tuner._winner_cache()),
            "loaded_file_sha256": self.map_sha(self.tuner._file_configs),
        }

    def setup(self):
        super().setup()
        self.rng_snapshot("before_history")
        if self.args.history:
            before, attempts = self.cache_snapshot(), self.profile_attempts
            self.observed_call("pre_default_before", expected=-1, tier="miss")
            self.retain_capture("pre_default", -1, "miss")
            measurement = self.graph_window("pre_default", "history_before_startup")
            self.history_graph = self.graphs.pop("pre_default")
            graph_row = self.graph_rows.pop()
            if graph_row["id"] != "pre_default":
                raise RuntimeError("the history capture was not the retained default")
            self.controls["capture_order"].remove("pre_default")
            after = self.cache_snapshot()
            if before != after or self.profile_attempts != attempts:
                raise RuntimeError("history capture altered tuning or cache state")
            self.history_evidence = {
                "graph": graph_row,
                "measurement": measurement,
                "cache_before": before,
                "cache_after": after,
                "profile_attempts_before": attempts,
                "profile_attempts_after": self.profile_attempts,
            }
        before = self.rng_snapshot("before_startup")
        previous = self.rng_states[0]
        if any(before[key] != previous[key] for key in ("cpu", "cuda")):
            raise RuntimeError("history advanced a profiling RNG generator")

    def startup(self):
        self.controls["startup_started_ns"] = time.perf_counter_ns()
        super().startup()
        self.controls["startup_ended_ns"] = time.perf_counter_ns()
        self.rng_snapshot("after_startup")

    def profile_hook(self, runner, inputs, tactic, config, *more, **kwargs):
        index = len(self.profiles)
        if self.args.fingerprint:
            # Preserve the exact original full fingerprint operation and timing.
            try:
                return super().profile_hook(
                    runner, inputs, tactic, config, *more, **kwargs
                )
            finally:
                if len(self.profiles) > index:
                    self.profiles[index]["fingerprinted"] = True
        if not self.allow_profiling:
            return LinearLayerExperiment.profile_hook(
                self, runner, inputs, tactic, config, *more, **kwargs
            )
        # Metadata only: no tensor copy, hash, synchronization or retained input
        # references. This arm does not attest actual synthetic operand bytes.
        values = {
            name: {
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype),
            }
            for name, tensor in zip(
                ("activation", "weight", "input_scale", "weight_scale"), inputs[:4]
            )
        }
        group = hashlib.sha256(
            json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.profiling_input_groups[group] = values
        settings = self.tuner_settings(config)
        if (
            config.use_cuda_graph
            or config.use_cold_l2_cache
            or not self.tuner._in_v2_context
            or settings["effective_measure_policy"] is not None
            or tuple(
                settings[key]
                for key in (
                    "warmup",
                    "repeat",
                    "stream_delay_micro_secs",
                    "cuda_graph_delay_micro_secs",
                    "use_global_timer",
                )
            )
            != (3, 10, 5000, 100, False)
            or config.profiling_repeat is not None
            or config.cuda_graph_profile_replays != 1
        ):
            self.protocol_errors.append({"stage": self.stage, "settings": settings})
        try:
            return LinearLayerExperiment.profile_hook(
                self, runner, inputs, tactic, config, *more, **kwargs
            )
        finally:
            if len(self.profiles) > index:
                self.profiles[index].update(
                    profiling_input_group=group,
                    policy_identity=repr(self.tuner._winner_cache_identity()),
                    policy_manifest_fields={},
                    default_race_enabled=self.tuner._in_v2_context,
                    fingerprinted=False,
                )

    def runtime_summary(self):
        super().runtime_summary()
        rows = [
            row
            for row in self.profiles
            if row["normalized_profile"]
            == self.runtime_profile_summary["normalized_profile"]
            and row.get("profiler_ms") is not None
        ]
        default = next(row["profiler_ms"] for row in rows if row["tactic"] == -1)
        explicit = [row for row in rows if row["tactic"] != -1]
        best = min(row["profiler_ms"] for row in explicit)
        self.runtime_profile_summary.update(
            default_profiler_ms=default,
            best_explicit_profiler_ms=best,
            best_explicit_tactics=[
                row["tactic"] for row in explicit if row["profiler_ms"] == best
            ],
            best_explicit_over_default=best / default,
        )

    def interleave(self):
        # Only this pure schedule helper is adapted for eight trials. No GPU
        # operation, lookup or native binding is replaced in measurement windows.
        with patch.object(objective_module, "schedule", schedule):
            super().interleave()

    def final_checks(self):
        self.assert_bare_replay_state()
        graphs = list(self.graphs.values())
        if self.history_graph is not None:
            graphs.append(self.history_graph)
        if (
            self.protocol_errors
            or self.forbidden_profile_attempts
            or self.profile_attempts != self.startup_profile_attempts
            or self.profile_attempts != len(self.profiles)
            or not self.cache_scopes_restored
            or self.operands != self.operand_digests()
            or digest(self.args.cache) != self.cache_after_startup
            or len({graph.output.data_ptr() for graph in graphs}) != len(graphs)
            or tuple(self.controls["capture_order"]) != self.requested_order
        ):
            raise RuntimeError(
                "history study changed profiles, inputs, caches or graph ownership"
            )
        return {
            "ended_ns": time.perf_counter_ns(),
            **self.cache_snapshot(),
            "profile_attempts": self.profile_attempts,
        }

    def trace_after_timing(self):
        torch = self.torch
        self.assert_bare_replay_state()
        self.args.trace.parent.mkdir(parents=True, exist_ok=True)
        if self.args.trace.exists():
            raise RuntimeError("refusing to overwrite a retained replay trace")
        started = time.perf_counter_ns()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as profiler:
            for graph_id in GRAPH_IDS:
                with torch.profiler.record_function("retained_" + graph_id):
                    for _ in range(3):
                        self.graphs[graph_id].graph.replay()
                    torch.cuda.synchronize()
        profiler.export_chrome_trace(str(self.args.trace))
        ended = time.perf_counter_ns()
        events = json.loads(self.args.trace.read_text()).get("traceEvents", [])
        kernels = [event for event in events if event.get("cat") == "kernel"]
        rows = []
        for graph_id in GRAPH_IDS:
            markers = [
                event
                for event in events
                if event.get("cat") == "user_annotation"
                and event.get("name") == "retained_" + graph_id
            ]
            if len(markers) != 1:
                raise RuntimeError("trace lacks an unambiguous retained graph range")
            marker = markers[0]
            inside = [
                event
                for event in events
                if marker["ts"] <= event.get("ts", -1) <= marker["ts"] + marker["dur"]
            ]
            launches = [
                event for event in inside if "cudaGraphLaunch" in event.get("name", "")
            ]
            selected = sorted(
                (event for event in inside if event.get("cat") == "kernel"),
                key=lambda event: event["ts"],
            )
            if len(launches) != 3 or not selected:
                raise RuntimeError(
                    "trace lacks actual retained graph launches or CUDA kernels"
                )
            signatures, sequence = [], []
            for event in selected:
                signature = kernel_signature(event)
                if signature not in signatures:
                    signatures.append(signature)
                sequence.append(signatures.index(signature))
            rows.append(
                {
                    "graph_id": graph_id,
                    "graph_object_id": id(self.graphs[graph_id].graph),
                    "output_data_ptr": self.graphs[graph_id].output.data_ptr(),
                    "graph_launch_events": len(launches),
                    "kernel_events": len(selected),
                    "signatures": signatures,
                    "kernel_sequence": sequence,
                }
            )
            self.out = self.graphs[graph_id].output
            self.check_retained_output(graph_id, "traced_" + graph_id)
        if sum(row["kernel_events"] for row in rows) != len(kernels):
            raise RuntimeError(
                "trace contains CUDA kernels outside retained graph ranges"
            )
        return {
            "path": str(self.args.trace),
            "sha256": digest(self.args.trace),
            "started_ns": started,
            "ended_ns": ended,
            "calls_per_graph": 3,
            "kernel_events": len(kernels),
            "graphs": rows,
            "scope": "Post-timing replay of the original retained graph objects; no recapture and no latency inference.",
        }

    def run(self):
        # Preserve the original objective worker's sequence, with an explicit
        # capture order independent of trial identity and cell launch order.
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
            for graph_id in self.requested_order:
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
                            measurement_policy=None,
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
            if self.history_graph is not None:
                self.check_graph(self.history_graph, "history_final")
                self.history_evidence["retained_through_end"] = True
            with self.policy_scope("replay", "natural_final"):
                self.observed_call("natural_final", expected=winner, tier="memory")
            self.observed_call("bare_after_scope", expected=winner, tier="memory")
            self.pre_trace_checks = self.final_checks()
            self.trace = self.trace_after_timing()
            self.post_trace_checks = self.final_checks()
        return self.result()

    def result(self):
        result = GraphControlExperiment.result(self)
        result.update(worker_sha256=digest(__file__), pid=os.getpid())
        result["environment"]["helper_sha256"].update(
            {
                name: digest(Path(__file__).with_name(name))
                for name in (
                    "flashinfer_autotune_graph_controls.py",
                    "flashinfer_autotune_measurement_objective.py",
                )
            }
        )
        result["environment"].update(
            measurement_policy=None,
            flashinfer_autotune_cache_sha256=self.policy_source_sha256,
        )
        result["profiling_input_groups"] = self.profiling_input_groups
        result["objective"] = {
            "requested_policy": None,
            "manifest_fields": {},
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
                "caller_stream_restored": self.torch.cuda.current_stream()
                == self.caller_stream,
            },
        }
        result["cache"].update(
            incidental_legacy_entries=self.incidental_legacy_entries,
            persistence_under_test=False,
        )
        ratios = []
        for group in self.controls["blocks"]:
            rows = {row["graph_id"]: row for row in group["measurements"]}
            ratios.append(
                rows["winner"]["gpu_event_per_call_ms"]
                / rows["default"]["gpu_event_per_call_ms"]
            )
        result["selection_history"] = {
            "history_enabled": bool(self.args.history),
            "inline_fingerprinting": bool(self.args.fingerprint),
            "requested_capture_order": list(self.requested_order),
            "history": self.history_evidence,
            "rng_states": self.rng_states,
            "runtime_best_explicit_over_default_profile": self.runtime_profile_summary[
                "best_explicit_over_default"
            ],
            "selected_over_default_graph_event_median": statistics.median(ratios),
            "profiling_byte_attestation": "inline_full_hashes"
            if self.args.fingerprint
            else "metadata_only_no_synthetic_byte_attestation",
            "synthetic_input_references_retained": False,
            "pre_trace_checks": self.pre_trace_checks,
            "post_timing_trace": self.trace,
            "post_trace_checks": self.post_trace_checks,
        }
        result["scope_notes"] = [
            "Policy remains unspecified auto, with the actual v2 default race and unchanged SGLang wrapper, legacy incidental save, streams and synchronization.",
            "History is the retained pre-startup actual default capture, numerical guards and500ms/15sample measurement from the prior history recipe; it stays separate from the two primary graphs.",
            "Fingerprint-on invokes the original full profiling-input hash hook. Fingerprint-off records metadata only, does not retain synthetic tensors and cannot attest their actual bytes.",
            "CPU/CUDA generator state is read at startup boundaries after CUDA initialization; this common host observation does not request random generation or GPU copies.",
            "All timing uses the raw retained graph replay; the profiler trace starts only after samples, numerical controls and final cache checks and replays the same graph objects without recapture.",
            "The profiler margin and selected/default serving-graph ratio describe different execution contracts. This fixed layer does not establish model throughput or general selection quality.",
        ]
        return result


def main():
    args = parse_args()
    experiment = None
    try:
        import torch

        with torch.inference_mode():
            experiment = HistoryExperiment.__new__(HistoryExperiment)
            experiment.__init__(args)
            result = experiment.run()
    except Exception as error:
        traceback.print_exc()
        result = {
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "trial": args.trial,
            "history": args.history,
            "fingerprint": args.fingerprint,
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
                    "history_evidence",
                    "rng_states",
                )
                if experiment is not None and hasattr(experiment, name)
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
