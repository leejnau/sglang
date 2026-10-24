"""Calibrate three replay timing methods on the same two real default graphs.

Startup is the actual unspecified-policy H0/F0 path. Each graph is captured from
an independent genuine default miss. No tuning or replacement occurs afterward.
Queued wall spans include the deliberate delay and are diagnostic only.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import os
import random
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import bench_flashinfer_selection_history as history_module
from bench_flashinfer_selection_history import HistoryExperiment
from flashinfer_autotune_descriptor_qualification import tactic_json
from flashinfer_autotune_eager_setup import digest, positive_float
from flashinfer_autotune_graph_controls import GraphControlExperiment, device_status

GRAPH_IDS = ("graph_a", "graph_b")
METHODS = {
    "plain5120": {"replays": 5120, "prequeue_cycles": 0},
    "queued16_10m": {"replays": 16, "prequeue_cycles": 10_000_000},
    "queued16_40m": {"replays": 16, "prequeue_cycles": 40_000_000},
}


def schedule(trial):
    if trial not in range(6):
        raise ValueError("the calibration requires six fresh processes")
    rows = []
    for methods in itertools.permutations(METHODS):
        for occurrence in range(4):
            graphs = GRAPH_IDS if occurrence % 2 == 0 else tuple(reversed(GRAPH_IDS))
            rows.append({"methods": list(methods), "graph_order": list(graphs)})
    random.Random(7200 + trial).shuffle(rows)
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=int, choices=range(6), required=True)
    parser.add_argument(
        "--capture-order", choices=("a-first", "b-first"), required=True
    )
    for name in ("cache", "output", "manifest", "fixture", "fixture-manifest", "trace"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--blocks", type=int, choices=(24,), default=24)
    parser.add_argument("--rewarm-ms", type=positive_float, default=20)
    args = parser.parse_args(argv)
    if args.rewarm_ms != 20:
        parser.error("the calibration requires 20 ms rewarm")
    args.arm, args.phase, args.mode, args.oracle = "auto", "cold", "timing", False
    args.seed, args.history, args.fingerprint = 19, 0, 0
    args.samples, args.iterations = 15, 5120
    args.warmup_ms, args.min_sample_ms = 500, 20
    return args


def file_sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class CalibrationExperiment(HistoryExperiment):
    def __init__(self, args):
        super().__init__(args)
        self.requested_order = (
            GRAPH_IDS if args.capture_order == "a-first" else tuple(reversed(GRAPH_IDS))
        )
        self.controls["schedule_seed"] = 7200 + args.trial
        self.calibration = {
            "methods": METHODS,
            "capture_order": list(self.requested_order),
            "schedule_seed": 7200 + args.trial,
            "schedule": schedule(args.trial),
            "rounds": [],
            "initial_warmups": [],
            "pending_sample": None,
        }

    def prime_delay(self):
        torch = self.torch
        wrapper = torch.cuda._sleep
        source = inspect.getsource(wrapper)
        wrapper_path = Path(inspect.getsourcefile(wrapper)).resolve()
        native_paths = [
            Path(torch._C.__file__).resolve(),
            Path(torch.__file__).resolve().parent / "lib/libtorch_cuda.so",
        ]
        if any(not path.is_file() for path in native_paths):
            raise RuntimeError("the actual CUDA delay implementation is not preserved")
        native = [
            {
                "path": str(path),
                "sha256": file_sha(path),
                "size_bytes": path.stat().st_size,
            }
            for path in native_paths
        ]
        started = time.perf_counter_ns()
        wrapper(10_000_000)
        torch.cuda.synchronize()
        self.calibration["delay_helper"] = {
            "implementation": "torch.cuda._sleep calling torch._C._cuda_sleep",
            "wrapper_path": str(wrapper_path),
            "wrapper_file_sha256": file_sha(wrapper_path),
            "wrapper_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "wrapper_source": source,
            "native_binaries": native,
            "prime_cycles": 10_000_000,
            "prime_started_ns": started,
            "prime_ended_ns": time.perf_counter_ns(),
            "primed_before_captures": True,
        }

    def warm_graph(self, graph_id, milliseconds):
        graph = self.graphs[graph_id]
        started, calls = time.perf_counter_ns(), 0
        while True:
            for _ in range(5120):
                graph.graph.replay()
            self.torch.cuda.synchronize()
            calls += 5120
            elapsed = (time.perf_counter_ns() - started) / 1e6
            if elapsed >= milliseconds:
                break
        return {"actual_ms": elapsed, "calls": calls}

    def sample(self, graph_id, method, index):
        torch = self.torch
        self.assert_bare_replay_state()
        graph = self.graphs[graph_id]
        self.stage = f"round_{index:02d}_{method}_{graph_id}"
        self.out = graph.output
        row = {
            "method": method,
            "graph_id": graph_id,
            "graph_object_id": id(graph.graph),
            "output_data_ptr": graph.output.data_ptr(),
            **METHODS[method],
            "gpu_before": device_status(),
            "rewarm": self.warm_graph(graph_id, self.args.rewarm_ms),
            "profile_attempts_before": self.profile_attempts,
            "start_query_error": None,
        }
        self.calibration["pending_sample"] = row
        self.out.fill_(float("nan"))
        first, last = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        # Initialize lazy event state and drain the actual output poison before
        # any prequeue or timing command. No previous correct output can survive.
        first.record()
        last.record()
        last.synchronize()
        wall_started = time.perf_counter_ns()
        row["wall_started_ns"] = wall_started
        if row["prequeue_cycles"]:
            torch.cuda._sleep(row["prequeue_cycles"])
        enqueue_started = time.perf_counter_ns()
        row["enqueue_started_ns"] = enqueue_started
        first.record()
        for _ in range(row["replays"]):
            graph.graph.replay()
        last.record()
        try:
            # Deliberately the first call after the final event is enqueued.
            start_complete = first.query()
        except Exception as error:
            row["start_query_error"] = f"{type(error).__name__}: {error}"
            row["query_error_observed_ns"] = time.perf_counter_ns()
            raise
        enqueue_ended = time.perf_counter_ns()
        last.synchronize()
        ended = time.perf_counter_ns()
        elapsed = first.elapsed_time(last)
        if not isinstance(start_complete, bool) or not (0 < elapsed < float("inf")):
            raise RuntimeError("invalid replay duration or event-query result")
        row.update(
            gpu_event_batch_ms=elapsed,
            gpu_event_per_replay_ms=elapsed / row["replays"],
            start_event_complete_after_enqueue=start_complete,
            queue_evidence=(
                "start_pending_after_enqueue"
                if not start_complete
                else "start_complete_inconclusive"
            )
            if row["prequeue_cycles"]
            else "not_prequeued",
            enqueue_started_ns=enqueue_started,
            enqueue_ended_ns=enqueue_ended,
            enqueue_cpu_ms=(enqueue_ended - enqueue_started) / 1e6,
            enqueue_span_includes_start_query=True,
            wall_started_ns=wall_started,
            ended_ns=ended,
            wall_including_prequeue_ms=(ended - wall_started) / 1e6,
            wall_diagnostic_only=True,
            policy_scope_active=self.tuner._in_v2_context,
            tuning_mode=self.tuner.is_tuning_mode,
            profile_attempts_after=self.profile_attempts,
        )
        # Inspect this exact allocation before another replay, lookup or public
        # call could overwrite the timed result.
        self.out = graph.output
        self.check_retained_output(graph_id, self.stage)
        row["correctness_check"] = self.stage
        row["gpu_after"] = device_status()
        self.assert_bare_replay_state()
        self.calibration["pending_sample"] = None
        return row

    def trace_calibration(self):
        # The existing reader validates CPU ranges, graph launches and kernels;
        # recorded identities let the analyzer bind these to original captures.
        with patch.object(history_module, "GRAPH_IDS", GRAPH_IDS):
            return self.trace_after_timing()

    def run(self):
        with patch.object(self.tuner, "_profile_single_kernel", self.profile_hook):
            self.setup()
            self.startup()
            with self.policy_scope("replay", "natural_before"):
                winner, _selected, acceptance = self.observed_call(
                    "natural_before", tier="memory"
                )
                self.natural_tactic = winner
                self.natural = {
                    "tactic": tactic_json(winner),
                    "cache_hit": True,
                    "lookup_tier": "memory",
                    "acceptance": acceptance,
                    "consumed_by_calibration_graphs": False,
                }
                self.runtime_summary()
            self.prime_delay()
            for graph_id in self.requested_order:
                with (
                    self.policy_scope("replay", graph_id + "_capture_scope"),
                    self.miss_scope(),
                ):
                    self.observed_call(graph_id + "_before", expected=-1, tier="miss")
                    self.retain_capture(graph_id, -1, "miss")
                    self.graph_rows[-1].update(
                        policy_identity=repr(self.tuner._winner_cache_identity()),
                        measurement_policy=None,
                        graph_object_id=id(self.graphs[graph_id].graph),
                    )
            self.assert_bare_replay_state()
            self.calibration["captures_finished_ns"] = time.perf_counter_ns()
            for graph_id in self.requested_order:
                warm = self.warm_graph(graph_id, self.args.warmup_ms)
                self.out = self.graphs[graph_id].output
                self.check_retained_output(graph_id, "initial_warm_" + graph_id)
                self.calibration["initial_warmups"].append(
                    {"graph_id": graph_id, **warm}
                )
            for index, requested in enumerate(self.calibration["schedule"]):
                group = {"index": index, **requested, "measurements": []}
                self.calibration["rounds"].append(group)
                for method in requested["methods"]:
                    for graph_id in requested["graph_order"]:
                        group["measurements"].append(
                            self.sample(graph_id, method, index)
                        )
            for graph_id in GRAPH_IDS:
                self.out = self.graphs[graph_id].output
                self.check_retained_output(graph_id, "final_" + graph_id)
            self.pre_trace_checks = self.final_checks()
            self.trace = self.trace_calibration()
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
                    "bench_flashinfer_selection_history.py",
                )
            }
        )
        result["environment"].update(
            measurement_policy=None,
            flashinfer_autotune_cache_sha256=self.policy_source_sha256,
        )
        result["profiling_input_groups"] = self.profiling_input_groups
        result["rng_states"] = self.rng_states
        result["cache"].update(
            incidental_legacy_entries=self.incidental_legacy_entries,
            persistence_under_test=False,
        )
        result["objective"] = {
            "requested_policy": None,
            "startup_api_calls": self.startup_api_calls,
            "scope_records": self.scope_records,
            "protocol_errors": self.protocol_errors,
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
        result["calibration"] = {
            **self.calibration,
            "pre_trace_checks": self.pre_trace_checks,
            "post_timing_trace": self.trace,
            "post_trace_checks": self.post_trace_checks,
            "all_samples_retained": True,
            "queue_query_is_exclusion_criterion": False,
            "synthetic_byte_attestation": "metadata_only_F0",
        }
        result["scope_notes"] = [
            "Every process runs identical H0/F0 unspecified-policy startup before two independently captured genuine-default misses. Natural startup winner is recorded but neither measured graph consumes it.",
            "All three timing methods replay the same original graph objects. Method order uses all six permutations four times; graph order is balanced within each permutation and capture order is explicit.",
            "The Torch CUDA sleep helper is loaded, hashed and primed before capture; each GPU event interval excludes its prequeue delay.",
            "A pending start event after enqueue is evidence of queue ahead; a completed start is inconclusive and never removes or retries a sample. Query errors preserve the partial sample and abort rather than silently excluding data.",
            "CPU enqueue spans include the immediate start-event query. Wall spans include deliberate prequeue and synchronization and are diagnostic only, not serving latency.",
            "Poisoning and event initialization/synchronization occur before every timed interval; numerical checks inspect each graph's own timed output before another operation can mask it.",
            "No Python lookup, autotuner measurement or graph replacement occurs after capture. The diagnostic Torch profiler pass begins after all timing and numerical/cache checks and replays the original objects without recapture.",
        ]
        return result


def main():
    args = parse_args()
    experiment = None
    try:
        import torch

        with torch.inference_mode():
            experiment = CalibrationExperiment.__new__(CalibrationExperiment)
            experiment.__init__(args)
            result = experiment.run()
    except Exception as error:
        traceback.print_exc()
        result = {
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "trial": args.trial,
            "pid": os.getpid(),
            "partial_evidence": {
                name: getattr(experiment, name)
                for name in (
                    "profiles",
                    "profile_attempts",
                    "forbidden_profile_attempts",
                    "protocol_errors",
                    "correctness",
                    "controls",
                    "calibration",
                    "scope_records",
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
