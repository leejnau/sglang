"""Observe actual server paths without changing selection or evaluation inputs."""

import dataclasses
import functools
import hashlib
import importlib.machinery
import json
import os
import sys
import time
from pathlib import Path

OUT = os.environ.get("SCOPED_QUAL_OUTPUT")
TARGETS = {
    "flashinfer.autotuner.autotuner",
    "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend",
    "sglang.test.run_eval",
}
COUNTS = {"profile": 0, "replay": 0}
SEEN = set()


def emit(kind, **values):
    path = Path(OUT) / f"observer-{os.getpid()}.jsonl"
    row = {"event": kind, "pid": os.getpid(), "time": time.time(), **values}
    with path.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def instrument(module):
    emit(
        "loaded",
        module=module.__name__,
        path=module.__file__,
        sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
    )
    if module.__name__ == "flashinfer.autotuner.autotuner":
        cls = module.AutoTuner
        profile, choose = cls._profile_single_kernel, cls.choose_one

        @functools.wraps(profile)
        def observed_profile(self, *args, **kwargs):
            value = profile(self, *args, **kwargs)
            COUNTS["profile"] += 1
            if COUNTS["profile"] == 1:
                emit(
                    "first_profile",
                    identity=repr(self._winner_cache_identity()),
                    policy=repr(self._effective_measure_policy),
                )
            return value

        @functools.wraps(choose)
        def observed_choose(self, *args, **kwargs):
            winner = choose(self, *args, **kwargs)
            inputs = args[3] if len(args) > 3 else kwargs["inputs"]
            row = dict(
                op=args[0] if args else kwargs.get("op_name"),
                runner=type(winner[0]).__name__,
                tactic=repr(winner[1]),
                identity=repr(self._winner_cache_identity()),
                strict=self._v2_local.strict_replay,
                shapes=[list(x.shape) for x in inputs if hasattr(x, "shape")],
            )
            key = json.dumps(row, sort_keys=True)
            if key not in SEEN:
                SEEN.add(key)
                emit("selection", **row)
            return winner

        cls._profile_single_kernel = observed_profile
        cls.choose_one = observed_choose
    elif module.__name__.endswith("full_cuda_graph_backend"):
        cls = module.FullCudaGraphBackend
        capture, replay = cls.capture_one, cls.replay

        @functools.wraps(capture)
        def observed_capture(self, shape_key, *args, **kwargs):
            from flashinfer.autotuner import AutoTuner

            tuner = AutoTuner.get()
            before = COUNTS["profile"]
            result = capture(self, shape_key, *args, **kwargs)
            report = tuner._v2_local.replay_report
            emit(
                "capture",
                shape_key=repr(shape_key),
                profile_delta=COUNTS["profile"] - before,
                profiles_total=COUNTS["profile"],
                identity=repr(tuner._winner_cache_identity()),
                strict=tuner._v2_local.strict_replay,
                report=dataclasses.asdict(report) if report is not None else None,
            )
            return result

        @functools.wraps(replay)
        def observed_replay(self, shape_key, *args, **kwargs):
            result = replay(self, shape_key, *args, **kwargs)
            COUNTS["replay"] += 1
            if COUNTS["replay"] in (1, 64, 256, 1024, 4096):
                emit(
                    "graph_replay",
                    count=COUNTS["replay"],
                    shape_key=repr(shape_key),
                    profiles_total=COUNTS["profile"],
                )
            return result

        cls.capture_one = observed_capture
        cls.replay = observed_replay
    else:
        original = module.run_eval_once

        @functools.wraps(original)
        def observed_eval(args, *a, **k):
            result, latency, sampler = original(args, *a, **k)
            out = {
                "args": vars(args),
                "score": result.score,
                "metrics": result.metrics,
                "convos": result.convos,
                "latency_seconds": latency,
            }
            (Path(OUT) / "evaluation.json").write_text(json.dumps(out, indent=2) + "\n")
            return result, latency, sampler

        module.run_eval_once = observed_eval


class Loader:
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def create_module(self, spec):
        return self.wrapped.create_module(spec)

    def exec_module(self, module):
        self.wrapped.exec_module(module)
        instrument(module)

    def __getattr__(self, name):
        return getattr(self.wrapped, name)


class Finder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in TARGETS:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None and spec.loader is not None:
                spec.loader = Loader(spec.loader)
            return spec
        return None


if OUT:
    Path(OUT).mkdir(parents=True, exist_ok=True)
    sys.meta_path.insert(0, Finder())
