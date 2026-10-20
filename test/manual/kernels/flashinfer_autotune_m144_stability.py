"""Repeat M144 selection and interleave cached candidate measurements.

The frozen selection worker supplies the real startup, persistence, native
acceptance and numerical guards. This study changes the runtime measurement
schedule; it leaves the tuner's unspecified measurement policy unchanged.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import random
import time
import traceback
from pathlib import Path

from flashinfer_autotune_descriptor_qualification import tactic_json
from flashinfer_autotune_descriptor_selection import SelectionExperiment
from flashinfer_autotune_eager_setup import MAX_ITERATIONS, digest, positive_float


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("v1", "v2"), required=True)
    parser.add_argument("--phase", choices=("cold", "warm"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--m", type=int, choices=(144,), default=144)
    parser.add_argument("--n", type=int, choices=(1024,), default=1024)
    parser.add_argument("--k", type=int, choices=(1024,), default=1024)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--trial", type=int, choices=range(6), required=True)
    parser.add_argument("--order-seed", type=int, required=True)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup-ms", type=positive_float, default=500)
    parser.add_argument("--min-sample-ms", type=positive_float, default=20)
    parser.add_argument("--blocks", type=int, default=24)
    parser.add_argument("--rewarm-ms", type=positive_float, default=20)
    args = parser.parse_args()
    if (
        min(args.samples, args.iterations, args.blocks) < 1
        or args.iterations > MAX_ITERATIONS
    ):
        parser.error("sample, iteration and block counts must be positive and bounded")
    if args.seed != 19 or args.order_seed != 3000 + args.trial:
        parser.error("input seed must be19; schedule seed must be3000+trial")
    if args.blocks != 24 or args.samples != 15 or args.iterations != 20:
        parser.error(
            "the preregistered protocol requires24blocks,15natural samples and20initial iterations"
        )
    if args.warmup_ms != 500 or args.min_sample_ms != 20 or args.rewarm_ms != 20:
        parser.error(
            "the preregistered protocol requires500ms warmup and20ms calibration/rewarm"
        )
    args.oracle = False
    return args


def decode_tactic(value):
    tactic = tuple(value) if isinstance(value, list) else value
    tactic_json(tactic)
    if tactic != -1 and not isinstance(tactic, tuple):
        raise RuntimeError("descriptor study tactics must be -1 or eight uint64 words")
    return tactic


def batch(torch, call, iterations, events):
    """One synchronized public-call/replay batch without lookup instrumentation."""
    start_event, end_event = events
    started = time.perf_counter()
    start_event.record()
    for _ in range(iterations):
        call()
    end_event.record()
    end_event.synchronize()
    wall_ms = (time.perf_counter() - started) * 1000
    event_ms = start_event.elapsed_time(end_event)
    if not all(math.isfinite(v) and v > 0 for v in (wall_ms, event_ms)):
        raise RuntimeError("nonpositive or nonfinite interleaved timing")
    return {
        "iterations": iterations,
        "wall_batch_ms": wall_ms,
        "gpu_event_batch_ms": event_ms,
        "wall_per_call_ms": wall_ms / iterations,
        "gpu_event_per_call_ms": event_ms / iterations,
    }


def calibrate(torch, call, args, events):
    started = time.perf_counter()
    calls = 0
    while True:
        for _ in range(args.iterations):
            call()
        torch.cuda.synchronize()
        calls += args.iterations
        elapsed = (time.perf_counter() - started) * 1000
        if elapsed >= args.warmup_ms:
            break
    iterations = args.iterations
    rows = []
    while True:
        timing = batch(torch, call, iterations, events)
        rows.append(timing)
        if timing["wall_batch_ms"] >= args.min_sample_ms:
            break
        if iterations == MAX_ITERATIONS:
            raise RuntimeError("calibration did not reach the minimum duration")
        iterations = min(iterations * 2, MAX_ITERATIONS)
    return {
        "requested_iterations": args.iterations,
        "effective_iterations": iterations,
        "warmup_actual_ms": elapsed,
        "warmup_calls": calls,
        "calibration_batches": rows,
    }


class StabilityExperiment(SelectionExperiment):
    def __init__(self, args):
        self.candidate_manifest = json.loads(args.candidate_manifest.read_text())
        rows = self.candidate_manifest["candidates"]
        if len(rows) != 3 or len({r["id"] for r in rows}) != 3:
            raise RuntimeError(
                "the preregistered manifest must contain three distinct candidate IDs"
            )
        if any(not r["id"].isidentifier() for r in rows):
            raise RuntimeError("candidate IDs must be Python identifiers")
        decoded = [decode_tactic(r["tactic"]) for r in rows]
        if len(set(decoded)) != 3 or decoded.count(-1) != 1:
            raise RuntimeError(
                "the fixed pool must contain the default and two distinct descriptors"
            )
        super().__init__(args)
        if self.operands != self.candidate_manifest["reference_operand_sha256"]:
            raise RuntimeError(
                "actual operands differ from the frozen seed19 reference"
            )
        self.tuner_settings_initial = self.tuner_settings()
        self.stability = {
            "enabled": args.phase == "warm",
            "trial": args.trial,
            "input_seed": args.seed,
            "order_seed": args.order_seed,
            "blocks_per_execution": args.blocks,
            "candidate_manifest_sha256": digest(args.candidate_manifest),
            "candidates": [],
            "common_iterations": {},
            "schedules": {},
            "blocks": [],
        }

    def tuner_settings(self, config=None):
        policy = self.tuner._effective_measure_policy
        return {
            "warmup": self.tuner.warmup,
            "repeat": self.tuner.repeat,
            "stream_delay_micro_secs": self.tuner.stream_delay_micro_secs,
            "cuda_graph_delay_micro_secs": self.tuner._CUDA_GRAPH_DELAY_MICRO_SECS,
            "use_global_timer": self.tuner._use_global_timer,
            "effective_measure_policy": dataclasses.asdict(policy)
            if dataclasses.is_dataclass(policy)
            else None,
            "effective_policy_timer": None if policy is None else policy.timer,
            "config": None
            if config is None
            else {
                "use_cuda_graph": config.use_cuda_graph,
                "use_cold_l2_cache": config.use_cold_l2_cache,
                "cuda_graph_profile_replays": config.cuda_graph_profile_replays,
                "profiling_repeat": config.profiling_repeat,
                "effective_profiling_repeat": self.tuner._get_profiling_repeat(config),
            },
        }

    def profile_hook(self, runner, inputs, tactic, config, *more, **kwargs):
        index = len(self.profiles)
        try:
            return super().profile_hook(runner, inputs, tactic, config, *more, **kwargs)
        finally:
            if len(self.profiles) > index:
                self.profiles[index]["autotuner_settings"] = self.tuner_settings(config)

    def candidate_pool(self):
        pool = []
        for row in self.candidate_manifest["candidates"]:
            tactic = decode_tactic(row["tactic"])
            pool.append(
                {
                    "id": row["id"],
                    "tactic": tactic_json(tactic),
                    "roles": [row["id"]],
                    "source": row["source"],
                }
            )
        selected = [r for r in pool if r["tactic"] == tactic_json(self.natural_tactic)]
        if selected:
            selected[0]["roles"].append("actual_selected")
        else:
            pool.append(
                {
                    "id": "actual_selected",
                    "tactic": tactic_json(self.natural_tactic),
                    "roles": ["actual_selected"],
                    "source": "Own actual fresh cold winner consumed by this warm worker; added according to the preregistered union rule.",
                }
            )
        if len(pool) not in (3, 4):
            raise RuntimeError(
                "the actual shortlist must have three or four candidates"
            )
        return pool

    def load_control(self, row):
        path = Path(row["control_file"])
        if digest(path) != row["control_sha256"]:
            raise RuntimeError("prepared candidate control file changed")
        self.tuner.clear_cache()
        if not self.tuner.load_configs(str(path)) or self.tuner._winner_cache():
            raise RuntimeError("prepared control did not load into file-only state")
        entry = self.tuner._file_configs.get(self.cache_key.file_key)
        if entry is None or entry[1] != decode_tactic(row["tactic"]):
            raise RuntimeError("loaded file has the wrong runtime candidate")

    def rewarm(self, call, iterations):
        started = time.perf_counter()
        calls = 0
        while True:
            for _ in range(iterations):
                call()
            self.torch.cuda.synchronize()
            calls += iterations
            elapsed = (time.perf_counter() - started) * 1000
            if elapsed >= self.args.rewarm_ms:
                return {"rewarm_actual_ms": elapsed, "rewarm_calls": calls}

    def prepare_candidates(self, events):
        graphs = {}
        self.stability["candidates"] = self.candidate_pool()
        for index, row in enumerate(self.stability["candidates"]):
            tactic = decode_tactic(row["tactic"])
            label = "stability_prepare_" + row["id"]
            with self.oracle_scope(tactic, index) as provenance:
                row.update(provenance)
                _, _, acceptance = self.observed_call(
                    label + "_before",
                    expected=tactic,
                    tier="file",
                    allow_rejection=True,
                )
                row.update(
                    acceptance=acceptance,
                    accepted=acceptance["accepted"],
                    exclusion_reason=None,
                    calibration={},
                )
                if not acceptance["accepted"]:
                    row["exclusion_reason"] = (
                        "Actual native descriptor rejection at M144; the full preregistered shortlist is unavailable and no stability contrasts are accepted."
                    )
                    raise RuntimeError(
                        "shortlisted descriptor was rejected; rejection retained in partial study output"
                    )
                graph = self.capture(tactic, "file", label + "_capture")
                self.check_graph(graph, label)
                graphs[row["id"]] = graph
                for execution, call in (
                    ("eager", self.operation),
                    ("cuda_graph", graph.replay),
                ):
                    self.stage = label + "_calibrate_" + execution
                    self.out.fill_(float("nan"))
                    row["calibration"][execution] = calibrate(
                        self.torch, call, self.args, events
                    )
                    self.check_output(self.stage)
                self.observed_call(label + "_after", expected=tactic, tier="file")
        self.stability["common_iterations"] = {
            execution: max(
                r["calibration"][execution]["effective_iterations"]
                for r in self.stability["candidates"]
            )
            for execution in ("eager", "cuda_graph")
        }
        return graphs

    def balanced_schedules(self):
        ids = [r["id"] for r in self.stability["candidates"]]
        permutations = [list(p) for p in itertools.permutations(ids)]
        repetitions = self.args.blocks // len(permutations)
        if repetitions * len(permutations) != self.args.blocks:
            raise RuntimeError("block count does not balance the actual candidate pool")
        for execution, offset in (("eager", 0), ("cuda_graph", 1)):
            schedule = [list(p) for _ in range(repetitions) for p in permutations]
            random.Random(self.args.order_seed + offset).shuffle(schedule)
            self.stability["schedules"][execution] = schedule
        self.stability["schedule_seed_rule"] = (
            "eager=order_seed; cuda_graph=order_seed+1; shuffle the full balanced permutation multiset"
        )
        self.stability["mode_order_rule"] = (
            "eager then cuda_graph on even blocks, reversed on odd blocks"
        )

    def interleaved_blocks(self, graphs, events):
        by_id = {r["id"]: r for r in self.stability["candidates"]}
        for block_index in range(self.args.blocks):
            executions = (
                ("eager", "cuda_graph")
                if block_index % 2 == 0
                else ("cuda_graph", "eager")
            )
            for execution in executions:
                order = self.stability["schedules"][execution][block_index]
                group = {
                    "index": block_index,
                    "execution": execution,
                    "order": order,
                    "measurements": [],
                }
                self.stability["blocks"].append(group)
                iterations = self.stability["common_iterations"][execution]
                for candidate_id in order:
                    row = by_id[candidate_id]
                    tactic = decode_tactic(row["tactic"])
                    label = f"stability_{execution}_b{block_index:02d}_{candidate_id}"
                    self.load_control(row)
                    _, _, before = self.observed_call(
                        label + "_before", expected=tactic, tier="file"
                    )
                    call = (
                        self.operation
                        if execution == "eager"
                        else graphs[candidate_id].replay
                    )
                    self.stage = label + "_rewarm"
                    rewarm = self.rewarm(call, iterations)
                    self.stage = label + "_timed"
                    # A later observed public call must not mask bad timed output.
                    self.out.fill_(float("nan"))
                    timing = batch(self.torch, call, iterations, events)
                    self.check_output(label + "_timed")
                    _, _, after = self.observed_call(
                        label + "_after", expected=tactic, tier="file"
                    )
                    group["measurements"].append(
                        {
                            "candidate_id": candidate_id,
                            **timing,
                            **rewarm,
                            "acceptance_before": before,
                            "acceptance_after": after,
                        }
                    )

    def oracle_measurements(self):
        # Enumerate/report all eligible descriptors without the old ordered sweep.
        super().oracle_measurements()
        self.oracle_order = []
        if self.args.phase == "cold":
            self.observed_call(
                "natural_after_oracle",
                expected=self.natural_tactic,
                tier=self.natural["lookup_tier"],
            )
            return
        saved = self.snapshot_state()
        start_event = self.torch.cuda.Event(enable_timing=True)
        end_event = self.torch.cuda.Event(enable_timing=True)
        start_event.record()
        end_event.record()
        end_event.synchronize()
        try:
            graphs = self.prepare_candidates((start_event, end_event))
            self.balanced_schedules()
            self.oracle_order = [r["tactic"] for r in self.stability["candidates"]]
            self.interleaved_blocks(graphs, (start_event, end_event))
        finally:
            self.restore_state(saved)
            if digest(self.args.cache) != self.cache_after_startup:
                raise RuntimeError(
                    "stability study changed the original natural cache bytes"
                )
        self.observed_call(
            "natural_after_oracle",
            expected=self.natural_tactic,
            tier=self.natural["lookup_tier"],
        )

    def run(self):
        result = super().run()
        inherited_hash = result["worker_sha256"]
        result["worker_sha256"] = digest(__file__)
        result["environment"]["helper_sha256"][
            "flashinfer_autotune_descriptor_selection.py"
        ] = inherited_hash
        result.update(
            trial=self.args.trial,
            order_seed=self.args.order_seed,
            stability=self.stability,
        )
        result["autotuner_settings"] = {
            "initial": self.tuner_settings_initial,
            "after": self.tuner_settings(self.config),
        }
        result["timing_protocol"].update(
            blocks_per_execution=self.args.blocks,
            rewarm_ms=self.args.rewarm_ms,
            input_seed=self.args.seed,
            order_seed=self.args.order_seed,
            scope="Natural timing retains15samples; warm shortlist timing uses24 interleaved blocks with common iterations and actual file-only lookup. Cold workers have no candidate timing sweep.",
        )
        result["scope_notes"] = [
            "This is a preregistered shortlist plus the own natural winner, not a global candidate optimum.",
            "All24workers use the exact original seed19 input bytes; trial ID controls fresh selection repetition and order_seed controls only the block schedule.",
            "Warm controls use actual existing v1 save/load files, with rewarming and observations outside timing; the original natural file and cache state are restored.",
            "Graph event spans include any replay submission gaps; these timings do not identify physical GPU kernels or prove causality.",
            "Actual autotuner timer/config attributes are observed without changing the unspecified policy.",
            "M144 tunes and serves144 while descriptor enumeration uses bucket256; no separate tune256-to-serve144 transfer test.",
        ]
        return result


def main():
    args = parse_args()
    experiment = None
    try:
        experiment = StabilityExperiment(args)
        result = experiment.run()
    except Exception as error:
        traceback.print_exc()
        result = {
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "arm": args.arm,
            "phase": args.phase,
            "seed": args.seed,
            "trial": args.trial,
            "order_seed": args.order_seed,
            "worker_sha256": digest(__file__),
            "stability": getattr(experiment, "stability", None),
            "profiles": getattr(experiment, "profiles", []),
            "profile_attempts": getattr(experiment, "profile_attempts", None),
            "forbidden_profile_attempts": getattr(
                experiment, "forbidden_profile_attempts", []
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
