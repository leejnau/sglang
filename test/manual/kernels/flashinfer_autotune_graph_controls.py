"""Separate graph history from capture-time cache selection on one actual layer.

This bounded manual experiment retains the same pre-tuning graph across startup,
then captures a natural winner and a genuine miss. Timing and instrumented
observation are separate processes. Replay measurements never perform lookup.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import ctypes
import hashlib
import inspect
import itertools
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from unittest.mock import patch

from flashinfer_autotune_descriptor_qualification import tactic_json
from flashinfer_autotune_eager_setup import digest, gpu_status, measure, positive_float
from flashinfer_autotune_linear_layer import LinearLayerExperiment, tensor_sha

GRAPH_IDS = ("pre_default", "post_hit", "post_miss")


def schedule(trial, blocks):
    if blocks != 24 or trial not in range(6):
        raise ValueError("the frozen protocol requires24 blocks and trial0..5")
    permutations = list(itertools.permutations(GRAPH_IDS))
    result = [list(row) for _ in range(4) for row in permutations]
    random.Random(4000 + trial).shuffle(result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("v1", "v2"), required=True)
    parser.add_argument("--mode", choices=("timing", "observe"), required=True)
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
    parser.add_argument("--observation-calls", type=int, choices=(3,), default=3)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    if (args.warmup_ms, args.min_sample_ms, args.rewarm_ms) != (500, 20, 20):
        parser.error(
            "the frozen protocol requires500/20/20ms warmup/calibration/rewarm"
        )
    if args.mode == "timing" and args.trace:
        parser.error("traces are permitted only in the separate observe process")
    if args.mode == "observe" and not args.trace:
        parser.error("the observe process requires --trace")
    args.seed, args.phase, args.oracle = 19, "cold", False
    return args


def device_status():
    """Read clocks/power outside measurement, preserving failures explicitly."""
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,driver_version,pstate,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return "unavailable: " + str(error)


def batch(torch, replay, iterations):
    first, last = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    # Initialize lazy event state and drain prior untimed output poisoning.
    first.record()
    last.record()
    last.synchronize()
    started = time.perf_counter_ns()
    first.record()
    for _ in range(iterations):
        replay()
    last.record()
    last.synchronize()
    ended = time.perf_counter_ns()
    wall, event = (ended - started) / 1e6, first.elapsed_time(last)
    if not all(math.isfinite(v) and v > 0 for v in (wall, event)):
        raise RuntimeError("invalid replay batch timing")
    return {
        "iterations": iterations,
        "wall_batch_ms": wall,
        "gpu_event_batch_ms": event,
        "wall_per_call_ms": wall / iterations,
        "gpu_event_per_call_ms": event / iterations,
        "started_ns": started,
        "ended_ns": ended,
    }


class GraphControlExperiment(LinearLayerExperiment):
    def __init__(self, args):
        self.probe = getattr(ctypes.CDLL(None), "fi_probe_set_phase", None)
        self.preload = os.environ.get("LD_PRELOAD")
        self.observations, self.graphs, self.graph_rows = [], {}, []
        self.controls = {
            "capture_order": [],
            "graphs": self.graph_rows,
            "history_before_startup": None,
            "history_after_startup_pre_capture": None,
            "sequential": [],
            "schedule": [],
            "blocks": [],
            "miss_scope": [],
            "common_iterations": args.iterations,
            "schedule_seed": 4000 + args.trial,
        }
        if args.mode == "timing":
            if self.probe or self.preload:
                raise RuntimeError(
                    "timing forbids native preload/probe instrumentation"
                )
            super().__init__(args)
        else:
            if not self.probe or not self.preload:
                raise RuntimeError(
                    "observe requires a real process-start native preload/probe"
                )
            self.probe.argtypes, self.probe.restype = [ctypes.c_char_p], None
            self.probe(None)
            original_cdll = ctypes.CDLL

            class ProcessSymbols:
                def __init__(self, original):
                    self.original = original

                def __getattr__(self, name):
                    if name == "fi_probe_set_phase":
                        raise AttributeError(name)
                    return getattr(self.original, name)

            def constructor_cdll(name, *more, **kwargs):
                original = original_cdll(name, *more, **kwargs)
                return ProcessSymbols(original) if name is None else original

            # The loader interposition is already active in this process. Only
            # the inherited latency-worker guard is suspended during construction.
            try:
                os.environ.pop("LD_PRELOAD", None)
                with patch.object(ctypes, "CDLL", constructor_cdll):
                    super().__init__(args)
            finally:
                os.environ["LD_PRELOAD"] = self.preload
            if (
                ctypes.CDLL is not original_cdll
                or getattr(ctypes.CDLL(None), "fi_probe_set_phase", None) is None
            ):
                raise RuntimeError("observer constructor did not restore symbol lookup")
        self.cache_scopes_restored = True

    @contextlib.contextmanager
    def phase(self, name):
        if self.probe:
            self.probe(name.encode())
        try:
            yield
        finally:
            if self.probe:
                self.probe(None)

    @contextlib.contextmanager
    def miss_scope(self):
        winners, files = self.tuner._winner_cache(), self.tuner._file_configs
        old_winners, old_files = copy.deepcopy(winners), copy.deepcopy(files)
        all_state = copy.deepcopy(self.snapshot_state())
        old_hash = digest(self.args.cache)
        self.cache_scopes_restored = False
        record = {
            "winner_entries_before": len(winners),
            "file_entries_before": len(files),
            "cache_sha256_before": old_hash,
            "restored": False,
            "winner_contents_sha256_before": self.map_sha(winners),
            "file_contents_sha256_before": self.map_sha(files),
        }
        self.controls["miss_scope"].append(record)
        try:
            winners.clear()
            files.clear()
            if self.tuner._winner_cache() or self.tuner._file_configs:
                raise RuntimeError("genuine-miss scope did not remove effective caches")
            record["winner_entries_after_removal"] = len(winners)
            record["file_entries_after_removal"] = len(files)
            yield
        finally:
            winners.clear()
            winners.update(old_winners)
            files.clear()
            files.update(old_files)
            self.restore_state(all_state)
            record["all_cache_state_restored"] = True
            record["cache_sha256_after"] = digest(self.args.cache)
            record["winner_contents_sha256_after"] = self.map_sha(winners)
            record["file_contents_sha256_after"] = self.map_sha(files)
            record["restored"] = (
                winners == old_winners
                and files == old_files
                and record["cache_sha256_after"] == old_hash
            )
            self.cache_scopes_restored = record["restored"]
            if not record["restored"]:
                raise RuntimeError(
                    "genuine-miss scope did not restore actual cache contents"
                )

    @staticmethod
    def map_sha(mapping):
        rows = sorted((repr(key), repr(value)) for key, value in mapping.items())
        return hashlib.sha256(
            json.dumps(rows, separators=(",", ":")).encode()
        ).hexdigest()

    def check_output(self, label):
        super().check_output(label)
        pointer = self.out.data_ptr()
        owners = [
            name
            for name, graph in self.graphs.items()
            if graph.output.data_ptr() == pointer
        ]
        if len(owners) > 1:
            raise RuntimeError("multiple retained graphs alias an output allocation")
        self.correctness[-1].update(
            graph_id=owners[0] if owners else None, output_data_ptr=pointer
        )

    def check_retained_output(self, graph_id, label):
        if self.out.data_ptr() != self.graphs[graph_id].output.data_ptr():
            raise RuntimeError("numerical guard selected another graph's output")
        self.check_output(label)

    def graph_metadata(self, tensor):
        pointer = tensor.data_ptr()
        return {
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype),
            "data_ptr": pointer,
            "alignment_mod256": pointer % 256,
            "alignment_mod4096": pointer % 4096,
        }

    def retain_capture(self, graph_id, expected, tier):
        started = time.perf_counter_ns()
        with self.phase(graph_id + "_capture"):
            graph = self.capture(expected, tier, graph_id + "_capture")
        ended = time.perf_counter_ns()
        inputs = tuple(self.captured["inputs"])
        self.graphs[graph_id] = graph
        self.controls["capture_order"].append(graph_id)
        with self.phase(graph_id + "_numerics"):
            self.check_graph(graph, graph_id)
        actual_sha = tensor_sha(self.torch, inputs[0].squeeze(0))
        if (
            actual_sha != self.quantized_input_sha
            or tensor_sha(self.torch, inputs[1].squeeze(0).t())
            != self.operands["weight_fp8_rowmajor"]
        ):
            raise RuntimeError("retained graph operands do not match the real fixture")
        row = {
            "id": graph_id,
            "tactic": tactic_json(expected),
            "lookup_tier": tier,
            "started_ns": started,
            "ended_ns": ended,
            "selection": dict(self.selections[-1]),
            "inputs": [self.graph_metadata(t) for t in inputs],
            "linear_input": self.graph_metadata(self.a),
            "output": self.graph_metadata(graph.output),
            "quantized_input_sha256": actual_sha,
            "weight_rowmajor_sha256": self.operands["weight_fp8_rowmajor"],
            "input_scale_sha256": tensor_sha(self.torch, inputs[2]),
            "weight_scale_sha256": tensor_sha(self.torch, inputs[3]),
        }
        # Keep original quantization/output/workspace views alive, independently
        # of self.captured and later graph allocations.
        graph.inputs = inputs
        self.graph_rows.append(row)
        return graph

    def observe_replays(self, graph_id, label):
        graph = self.graphs[graph_id]
        self.stage = label
        self.out = graph.output
        self.out.fill_(float("nan"))
        with self.phase(label):
            for _ in range(self.args.observation_calls):
                graph.graph.replay()
            self.torch.cuda.synchronize()
        self.out = graph.output
        self.check_retained_output(graph_id, label)
        self.observations.append(
            {
                "phase": label,
                "graph_id": graph_id,
                "calls": self.args.observation_calls,
                "scope": "instrumented observation, not a latency estimate",
            }
        )

    def graph_window(self, graph_id, label):
        graph = self.graphs[graph_id]
        self.stage, self.out = label, graph.output
        self.out.fill_(float("nan"))
        before, started = device_status(), time.perf_counter_ns()
        measured = measure(self.torch, graph.graph.replay, self.args)
        ended = time.perf_counter_ns()
        self.out = graph.output
        self.check_retained_output(graph_id, label)
        after = device_status()
        return {
            "graph_id": graph_id,
            "measurement": measured,
            "gpu_before": before,
            "gpu_after": after,
            "started_ns": started,
            "ended_ns": ended,
            "correctness_check": label,
        }

    def runtime_summary(self):
        self.runner, self.config, self.inputs = (
            self.captured[key] for key in ("runner", "config", "inputs")
        )
        self.cache_key = self.tuner._get_cache_key(
            "fp8_gemm",
            self.runner,
            tuple(tuple(t.shape) for t in self.inputs),
            self.config,
            self.runner.get_cache_key_extras(self.inputs),
        )
        rows = [
            row
            for row in self.profiles
            if tuple(tuple(s) for s in row["normalized_profile"])
            == self.cache_key.nearest_profile
        ]
        finite = [row for row in rows if row.get("profiler_ms") is not None]
        minimum = min((row["profiler_ms"] for row in finite), default=None)
        minima = [row["tactic"] for row in finite if row["profiler_ms"] == minimum]
        if not finite or tactic_json(self.natural_tactic) not in minima:
            raise RuntimeError(
                "natural winner is not the actual finite startup minimum"
            )
        if self.args.arm == "v2" and not any(
            row["tactic"] == -1 and row.get("profiler_ms") is not None for row in rows
        ):
            raise RuntimeError("v2 failed to profile the default at the runtime bucket")
        if self.args.arm == "v1" and any(row["tactic"] == -1 for row in self.profiles):
            raise RuntimeError("v1 unexpectedly profiled the default")
        self.runtime_profile_summary = {
            "normalized_profile": [list(s) for s in self.cache_key.nearest_profile],
            "profiles": len(rows),
            "finite_profiles": len(finite),
            "minimum_ms": minimum,
            "minimum_tactics": minima,
        }

    def interleave(self):
        self.controls["schedule"] = schedule(self.args.trial, self.args.blocks)
        for index, order in enumerate(self.controls["schedule"]):
            group = {"index": index, "order": order, "measurements": []}
            for graph_id in order:
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
                # Check the timed allocation before any later public call or
                # replay could overwrite/mask its contents.
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
                    }
                )
            self.controls["blocks"].append(group)

    def trace_retained(self):
        torch = self.torch
        self.args.trace.parent.mkdir(parents=True, exist_ok=True)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as profiler:
            for graph_id in GRAPH_IDS:
                with torch.profiler.record_function("retained_" + graph_id):
                    with self.phase("trace_" + graph_id):
                        for _ in range(self.args.observation_calls):
                            self.graphs[graph_id].graph.replay()
                    torch.cuda.synchronize()
        profiler.export_chrome_trace(str(self.args.trace))
        for graph_id in GRAPH_IDS:
            self.out = self.graphs[graph_id].output
            self.check_retained_output(graph_id, "traced_" + graph_id)
        trace = json.loads(self.args.trace.read_text())
        events = trace.get("traceEvents", [])
        kernels = [event for event in events if event.get("cat") == "kernel"]
        launches = [
            event for event in events if "cudaGraphLaunch" in str(event.get("name", ""))
        ]
        if not kernels or not launches:
            raise RuntimeError(
                "observer trace lacks actual CUDA kernels or graph launches"
            )
        return {
            "path": str(self.args.trace),
            "sha256": digest(self.args.trace),
            "kernel_events": len(kernels),
            "graph_launch_events": len(launches),
            "calls_per_graph": self.args.observation_calls,
            "scope": "replay-only trace of retained original graph objects; no recapture",
        }

    def run(self):
        with patch.object(self.tuner, "_profile_single_kernel", self.profile_hook):
            with self.phase("setup"):
                self.setup()
            with self.phase("pre_default_before"):
                self.observed_call("pre_default_before", expected=-1, tier="miss")
            if self.args.mode == "observe":
                for index in range(self.args.observation_calls):
                    with self.phase(f"pre_default_warm_eager_{index}"):
                        self.observed_call(
                            f"pre_default_warm_eager_{index}", expected=-1, tier="miss"
                        )
            self.retain_capture("pre_default", -1, "miss")
            if self.args.mode == "timing":
                self.controls["history_before_startup"] = self.graph_window(
                    "pre_default", "history_before_startup"
                )
            else:
                self.observe_replays("pre_default", "history_before_startup")
            with self.phase("startup"):
                self.controls["startup_started_ns"] = time.perf_counter_ns()
                self.startup()
                self.controls["startup_ended_ns"] = time.perf_counter_ns()
            with self.phase("natural_after_startup"):
                winner, selected, accepted = self.observed_call(
                    "natural_after_startup", tier="memory"
                )
            self.natural_tactic = winner
            self.natural = {
                "tactic": tactic_json(winner),
                "cache_hit": True,
                "lookup_tier": "memory",
                "acceptance": accepted,
            }
            self.runtime_summary()
            if self.args.mode == "timing":
                self.controls["history_after_startup_pre_capture"] = self.graph_window(
                    "pre_default", "history_after_startup_pre_capture"
                )
            else:
                self.observe_replays("pre_default", "history_after_startup_pre_capture")
            order = (
                ("post_hit", "post_miss")
                if self.args.trial % 2 == 0
                else ("post_miss", "post_hit")
            )
            for graph_id in order:
                with (
                    self.miss_scope()
                    if graph_id == "post_miss"
                    else contextlib.nullcontext()
                ):
                    expected, tier = (
                        (-1, "miss") if graph_id == "post_miss" else (winner, "memory")
                    )
                    with self.phase(graph_id + "_before"):
                        self.observed_call(
                            graph_id + "_before", expected=expected, tier=tier
                        )
                    if self.args.mode == "observe":
                        for index in range(self.args.observation_calls):
                            with self.phase(f"{graph_id}_warm_eager_{index}"):
                                self.observed_call(
                                    f"{graph_id}_warm_eager_{index}",
                                    expected=expected,
                                    tier=tier,
                                )
                    self.retain_capture(graph_id, expected, tier)
            if self.args.mode == "timing":
                self.controls["sequential"] = [
                    self.graph_window(graph_id, "sequential_" + graph_id)
                    for graph_id in GRAPH_IDS
                ]
                self.interleave()
                self.trace = None
            else:
                for graph_id in GRAPH_IDS:
                    self.observe_replays(graph_id, "sequential_" + graph_id)
                self.trace = self.trace_retained()
            for graph_id in GRAPH_IDS:
                self.out = self.graphs[graph_id].output
                self.check_retained_output(graph_id, "final_" + graph_id)
            with self.phase("natural_final"):
                self.observed_call("natural_final", expected=winner, tier="memory")
        if (
            self.profile_attempts != self.startup_profile_attempts
            or self.forbidden_profile_attempts
            or self.tuner.is_tuning_mode
            or self.tuner._active_managed_store
            or self.torch.cuda.current_stream() != self.caller_stream
            or not self.cache_scopes_restored
        ):
            raise RuntimeError(
                "profiling, stream, cache, or tuner state leaked after startup"
            )
        if (
            self.operands != self.operand_digests()
            or digest(self.args.cache) != self.cache_after_startup
        ):
            raise RuntimeError("graph controls changed fixture or persisted cache")
        if len({graph.output.data_ptr() for graph in self.graphs.values()}) != 3:
            raise RuntimeError("retained graphs do not own distinct output allocations")
        return self.result()

    def result(self):
        from flashinfer.autotuner import AutoTuner

        torch = self.torch
        source_root = Path(__file__).resolve().parents[3]
        preload_files = []
        if self.preload:
            for name in self.preload.replace(":", " ").split():
                path = Path(name)
                if not path.is_file():
                    raise RuntimeError("preload path is not a preserved actual file")
                preload_files.append({"path": name, "sha256": digest(path)})
        return {
            "schema_version": 1,
            "status": "success",
            "implementation": "candidate",
            "mode": self.args.mode,
            "arm": self.args.arm,
            "phase": "cold",
            "trial": self.args.trial,
            "seed": 19,
            "shape": [self.args.m, self.args.n, self.args.k],
            "worker_sha256": digest(__file__),
            "manifest_sha256": digest(self.args.manifest),
            "environment": {
                "sglang_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
                ).strip(),
                "sglang_wrapper_sha256": digest(self.startup_source),
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "flashinfer": self.flashinfer.__version__,
                "flashinfer_tuner_sha256": digest(inspect.getfile(AutoTuner)),
                "native_artifact": self.native_artifact,
                "helper_sha256": {
                    name: digest(Path(__file__).with_name(name))
                    for name in (
                        "flashinfer_autotune_linear_layer.py",
                        "flashinfer_autotune_descriptor_selection.py",
                        "flashinfer_autotune_descriptor_qualification.py",
                        "flashinfer_autotune_eager_setup.py",
                    )
                },
                "gpu": torch.cuda.get_device_properties(0).name,
                "physical_gpu_uuid": str(
                    getattr(torch.cuda.get_device_properties(0), "uuid", None)
                ),
                "capability": list(torch.cuda.get_device_capability()),
                "gpu_status_before": self.initial_gpu_status,
                "gpu_status_after": gpu_status(),
                "LD_PRELOAD": os.environ.get("LD_PRELOAD"),
                "backend": "cublas",
                "measurement_policy": "unspecified",
                "managed_store": False,
                "operand_sha256": self.operands,
            },
            "fixture": {
                "manifest_sha256": digest(self.args.fixture_manifest),
                "tensor_file_sha256": digest(self.args.fixture),
                **{
                    key: self.fixture_manifest[key]
                    for key in ("shape", "model", "capture", "tensors")
                },
            },
            "actual_layer": {
                **self.actual_layer,
                "observed_public_inputs": self.observed_public_inputs,
                "native_output_poison_checks": self.native_output_poison_checks,
            },
            "autotuner_settings": {
                "initial": self.tuner_settings_initial,
                "after": self.tuner_settings(self.config),
            },
            "initial_public_call_ms": self.setup_ms,
            "startup_ms": self.startup_ms,
            "profile_attempts": self.profile_attempts,
            "forbidden_profile_attempts": self.forbidden_profile_attempts,
            "profiles": self.profiles,
            "selections": self.selections,
            "binding_acceptance_checks": self.binding_checks,
            "natural": self.natural,
            "runtime_profile_summary": self.runtime_profile_summary,
            "cache": {
                "path": str(self.args.cache),
                "runtime_file_key": self.cache_key.file_key,
                "before_sha256": self.cache_before,
                "after_startup_sha256": self.cache_after_startup,
                "after_controls_sha256": digest(self.args.cache),
                "miss_scopes_restored": self.cache_scopes_restored,
            },
            "correctness": {
                "passed": True,
                "rtol": 0.02,
                "atol": 0.125,
                "checks": self.correctness,
            },
            "graph_controls": self.controls,
            "observer": {
                "enabled": self.args.mode == "observe",
                "constructor_guard_suspended": self.args.mode == "observe",
                "real_interposition_remained_loaded": self.args.mode == "observe",
                "preload_files": preload_files,
                "native_output_path": os.environ.get("FI_NATIVE_PROBE_OUTPUT"),
                "phases": self.observations,
                "trace": self.trace,
            },
            "scope_notes": [
                "Graph identities refer to cache source at capture; replay performs no Python tactic lookup.",
                "The exact pre-tuning graph is retained and measured after tuning and after later captures; no recapture substitutes for this history control.",
                "Post-miss capture removes effective winner/file maps reversibly, without replacing search or forcing a manual tactic.",
                "All captured outputs and operand views remain alive and numerical checks inspect each graph's actual allocation.",
                "Timing is uninstrumented; observation uses a separate native preload and replay-only trace and supports no latency inference.",
                "This is one fixed actual FP8 linear layer; no full-model or serving-throughput conclusion follows.",
            ],
        }


def main():
    args = parse_args()
    experiment = None
    try:
        import torch

        with torch.inference_mode():
            experiment = GraphControlExperiment.__new__(GraphControlExperiment)
            experiment.__init__(args)
            result = experiment.run()
    except Exception as error:
        traceback.print_exc()
        result = {
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "mode": args.mode,
            "arm": args.arm,
            "trial": args.trial,
            "partial_evidence": {
                name: getattr(experiment, name)
                for name in (
                    "profiles",
                    "profile_attempts",
                    "forbidden_profile_attempts",
                    "correctness",
                    "controls",
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
