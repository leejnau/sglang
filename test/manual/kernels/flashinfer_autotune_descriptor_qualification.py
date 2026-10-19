"""Qualify a pinned FP8 descriptor implementation against the original wheel.

This manual operation experiment keeps startup tuning, persistence and runtime
measurement separate. Source hashes and the matched algorithm are required in
the preregistered manifest. It changes no production code or tuning policy.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import inspect
import json
import math
import os
import platform
import random
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from flashinfer_autotune_eager_setup import (
    MAX_ITERATIONS,
    BoundaryRecorder,
    digest,
    gpu_status,
    measure,
    positive_float,
    trace_variant,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("benchmark", "observe", "lifecycle-cold", "lifecycle-warm"),
        required=True,
    )
    parser.add_argument(
        "--implementation", choices=("baseline", "candidate"), required=True
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--match",
        type=Path,
        help="paired baseline worker JSON; takes precedence over the manifest's historical descriptor",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--tuning-version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup-ms", type=positive_float, default=500)
    parser.add_argument("--min-sample-ms", type=positive_float, default=20)
    parser.add_argument("--observation-calls", type=int, default=5)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    if (args.m, args.n, args.k) != (128, 1024, 1024):
        parser.error("this qualification is preregistered for M128,N=K1024")
    if min(args.samples, args.iterations, args.observation_calls) < 1:
        parser.error("counts must be positive")
    if args.iterations > MAX_ITERATIONS or args.observation_calls > 1000:
        parser.error("iteration or observation count exceeds the safety bound")
    if args.trace and args.mode != "observe":
        parser.error("traces are only collected by separate observation workers")
    if args.mode.startswith("lifecycle-") and args.cache is None:
        parser.error("lifecycle workers require --cache")
    return args


def tactic_json(tactic):
    if isinstance(tactic, tuple):
        if len(tactic) != 8 or not all(
            type(v) is int and 0 <= v < 1 << 64 for v in tactic
        ):
            raise RuntimeError("malformed algorithm descriptor tactic")
        return list(tactic)
    if type(tactic) is not int:
        raise RuntimeError(f"unexpected tactic type: {type(tactic).__name__}")
    return tactic


def tensor_digest(torch, tensor):
    value = tensor.detach().contiguous().view(torch.uint8).cpu()
    return hashlib.sha256(bytes(value.flatten().tolist())).hexdigest()


def validate_source(manifest, implementation, flashinfer):
    """Every overlay file, including native sources, must match its frozen arm."""
    if "implementations" in manifest:
        expected_files = manifest["implementations"][implementation]["files"]
    else:
        expected_files = {
            row["installed_package_path"]: row[implementation + "_sha256"]
            for row in manifest["files"]
            if row["runtime_overlay"]
        }
    if not expected_files or "flashinfer/gemm/gemm_base.py" not in expected_files:
        raise RuntimeError("manifest must pin the FP8 Python and native implementation")
    root = Path(flashinfer.__file__).resolve().parent.parent
    observed = {}
    for relative, expected in expected_files.items():
        path = root / relative
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise RuntimeError(
                f"manifest source is absent or outside package: {relative}"
            )
        actual = digest(path)
        expected = expected["sha256"] if isinstance(expected, dict) else expected
        if actual != expected:
            raise RuntimeError(f"source hash mismatch for {implementation}/{relative}")
        observed[relative] = actual
    return observed


def prepare_native(implementation, manifest):
    """Load an explicitly built JIT artifact, bypassing installed AOT providers.

    The same one-time module adapter is used by both implementations. This can
    also initialize the exact module before running the upstream qualification
    tests in a separate process. Nothing is patched around timed public calls.
    """
    import flashinfer
    from flashinfer.gemm import gemm_base

    observed = validate_source(manifest, implementation, flashinfer)
    if gemm_base.get_gemm_module.cache_info().currsize:
        raise RuntimeError("native preparation must precede every FP8 operation")
    spec = gemm_base.gen_gemm_module()
    spec.build()
    native = spec.load()
    library = Path(spec.jit_library_path).resolve()
    ninja = Path(spec.ninja_path).resolve()
    if not library.is_file() or not ninja.is_file():
        raise RuntimeError("explicit JIT build artifacts are absent")
    required = (
        "bmm_fp8",
        "bmm_fp8_get_algos",
        "bmm_fp8_run_with_descriptor"
        if implementation == "candidate"
        else "bmm_fp8_run_with_algo",
    )
    if any(not hasattr(native, name) for name in required):
        raise RuntimeError(
            "loaded native artifact lacks the required implementation API"
        )
    gemm_base.gen_gemm_module = lambda: SimpleNamespace(build_and_load=lambda: native)
    return {
        "source_sha256": observed,
        "library_path": str(library),
        "library_sha256": digest(library),
        "ninja_path": str(ninja),
        "ninja_sha256": digest(ninja),
        "required_native_exports": list(required),
        "bootstrap": "JitSpec.build then JitSpec.load; one-time identical public module adapter; AOT provider deliberately bypassed",
        "cuda_arch_list": os.environ.get("FLASHINFER_CUDA_ARCH_LIST"),
    }


class Experiment:
    def __init__(self, args):
        self.args = args
        self.manifest = json.loads(args.manifest.read_text())
        if args.match:
            paired = json.loads(args.match.read_text())
            if (
                paired.get("status") != "success"
                or paired.get("implementation") != "baseline"
                or paired.get("seed") != args.seed
                or paired.get("shape") != [args.m, args.n, args.k]
            ):
                raise RuntimeError(
                    "--match must name the successful paired baseline workload"
                )
            raw = bytes.fromhex(paired["matched_algorithm"]["raw_hex"])
            self.match_provenance = {
                "path": str(args.match),
                "sha256": digest(args.match),
                "source": "paired baseline ordinal0",
            }
        elif "matched_descriptor" in self.manifest:
            raw = bytes.fromhex(self.manifest["matched_descriptor"]["raw_hex"])
            self.match_provenance = {"source": "manifest-pinned historical descriptor"}
        elif args.implementation == "baseline":
            raw = None
            self.match_provenance = {
                "source": "actual baseline ordinal0, emitted for the paired candidate"
            }
        else:
            raise RuntimeError(
                "candidate requires --match or a manifest-pinned descriptor"
            )
        if raw is not None and len(raw) != 64:
            raise RuntimeError("matched descriptor must contain exactly 64 bytes")
        self.matched_descriptor = struct.unpack("8Q", raw) if raw is not None else None
        self.matched_raw_hex = raw.hex() if raw is not None else None
        self.native_probe = getattr(ctypes.CDLL(None), "fi_probe_set_phase", None)
        if args.mode == "observe" and self.native_probe is None:
            raise RuntimeError("observation workers require the native probe")
        if args.mode != "observe" and (
            self.native_probe or os.environ.get("LD_PRELOAD")
        ):
            raise RuntimeError(
                "timing and lifecycle workers must have no native preload"
            )
        if self.native_probe:
            self.native_probe.argtypes = [ctypes.c_char_p]
            self.native_probe.restype = None
            self.native_probe(None)

        os.environ["FLASHINFER_AUTOTUNER_LOAD_FROM_FILE"] = "0"
        os.environ["SGLANG_FLASHINFER_AUTOTUNE_CACHE"] = "1"
        import flashinfer
        import torch
        from flashinfer.autotuner import AutoTuner
        from flashinfer.gemm import gemm_base

        self.torch, self.flashinfer, self.gemm_base = torch, flashinfer, gemm_base
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            9,
            0,
        ):
            raise RuntimeError("the preregistered device is an SM90 H100")
        if torch.__version__ != "2.14.1+cu130" or torch.version.cuda != "13.0":
            raise RuntimeError(
                "the pinned Torch 2.14.1+cu130/CUDA13.0 stack is required"
            )
        if flashinfer.__version__ != "0.7.2":
            raise RuntimeError("both source arms must report the base version 0.7.2")
        self.source_hashes = validate_source(
            self.manifest, args.implementation, flashinfer
        )
        self.tuner_source = Path(inspect.getfile(AutoTuner))
        if self.source_hashes.get("flashinfer/autotuner/autotuner.py") != digest(
            self.tuner_source
        ):
            raise RuntimeError(
                "source manifest must pin the active tuner, including its docstring-only overlay"
            )
        self.native_artifact = prepare_native(args.implementation, self.manifest)
        self.tuner = AutoTuner.get()
        if (
            self.tuner._winner_cache()
            or self.tuner._file_configs
            or self.tuner._active_managed_store
        ):
            raise RuntimeError("fresh worker unexpectedly has tuner state")
        if self.tuner.is_tuning_mode:
            raise RuntimeError("fresh worker unexpectedly has tuning enabled")
        self.initial_gpu_status = gpu_status()
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.a = torch.randn((1, args.m, args.k), device="cuda").to(torch.float8_e4m3fn)
        self.b = (
            torch.randn((1, args.n, args.k), device="cuda")
            .to(torch.float8_e4m3fn)
            .transpose(1, 2)
        )
        self.scale_a, self.scale_b = (
            torch.ones(1, device="cuda"),
            torch.ones(1, device="cuda"),
        )
        self.out = torch.empty((1, args.m, args.n), device="cuda", dtype=torch.bfloat16)
        self.reference = (
            torch.bmm(self.a.float(), self.b.float()) * self.scale_a * self.scale_b
        )
        self.operands = self.operand_digests()
        self.correctness, self.lookups, self.profiles, self.binding_checks = (
            [],
            [],
            [],
            [],
        )
        self.profile_attempts = 0
        self.forbidden_profile_attempts = []
        self.stage = "setup"
        self.allow_profiling = False
        self.original_choose = self.tuner.choose_one
        self.original_search = self.tuner.search_cache
        self.original_profile = self.tuner._profile_single_kernel
        self.captured = {}
        self.observation_summaries, self.measurements, self.traces = [], [], []

    def operand_digests(self):
        return {
            name: tensor_digest(self.torch, getattr(self, name))
            for name in ("a", "b", "scale_a", "scale_b")
        }

    def operation(self):
        return self.flashinfer.bmm_fp8(
            self.a,
            self.b,
            self.scale_a,
            self.scale_b,
            self.torch.bfloat16,
            out=self.out,
            backend="cublas",
        )

    def check_output(self, label):
        self.torch.cuda.synchronize()
        self.torch.testing.assert_close(
            self.out.float(), self.reference, rtol=0.02, atol=0.125
        )
        self.correctness.append(
            {
                "check": label,
                "max_abs_error": float(
                    (self.out.float() - self.reference).abs().max().item()
                ),
            }
        )

    def profile_hook(self, runner, inputs, tactic, config, *more, **kwargs):
        # The tuner catches profiling errors. Count an attempt before raising
        # so forbidden warm/capture profiling cannot hide as disqualification.
        self.profile_attempts += 1
        if not self.allow_profiling:
            self.forbidden_profile_attempts.append(self.stage)
            raise RuntimeError(f"unexpected profiling during {self.stage}")
        row = {
            "stage": self.stage,
            "tactic": tactic_json(tactic),
            "input_shapes": [list(t.shape) if t is not None else None for t in inputs],
            "use_cuda_graph": config.use_cuda_graph,
            "use_cold_l2_cache": config.use_cold_l2_cache,
        }
        self.profiles.append(row)
        try:
            measured = self.original_profile(
                runner, inputs, tactic, config, *more, **kwargs
            )
            row["profiler_ms"] = measured if math.isfinite(measured) else None
            return measured
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            raise

    def capture_inputs(self):
        def capture(op, runners, config, inputs, **kwargs):
            if op != "fp8_gemm" or len(runners) != 1:
                raise RuntimeError("unexpected operation or candidate runner pool")
            self.captured.update(op=op, runner=runners[0], config=config, inputs=inputs)
            return self.original_choose(op, runners, config, inputs, **kwargs)

        with patch.object(self.tuner, "choose_one", capture):
            self.operation()
        self.check_output("initial_untuned")
        if self.tuner._winner_cache() or self.tuner._file_configs:
            raise RuntimeError("untuned module preparation populated winners")
        from flashinfer.autotuner.autotuner import OptimizationProfile, StaticDim

        self.inputs = self.captured["inputs"]
        self.runner = self.captured["runner"]
        self.config = self.captured["config"]
        self.module = self.gemm_base.get_gemm_module()
        factory = self.module.cublas_fp8_gemm_runner
        closure = dict(zip(factory.__code__.co_freevars, factory.__closure__))
        if "module" not in closure:
            raise RuntimeError(
                "the audited runner factory lost its native-module closure"
            )
        self.native_cell = closure["module"]
        self.native_module = self.native_cell.cell_contents
        profile = OptimizationProfile(
            [[StaticDim(d) for d in t.shape] for t in self.inputs],
            [None] * len(self.inputs),
        )
        self.eligible = self.tuner._blocklist.filter(
            "fp8_gemm", self.runner, self.runner.get_valid_tactics(self.inputs, profile)
        )
        if self.args.implementation == "baseline":
            if 0 not in self.eligible:
                raise RuntimeError("baseline ordinal0 is not eligible")
            buffer, count = self.runner._get_algos(self.inputs)
            actual = bytes(buffer[:64].tolist()).hex()
            if count < 1 or (
                self.matched_raw_hex is not None and actual != self.matched_raw_hex
            ):
                raise RuntimeError(
                    "baseline ordinal0 no longer names the pinned descriptor"
                )
            self.matched_raw_hex = actual
            self.matched_descriptor = struct.unpack("8Q", bytes.fromhex(actual))
            self.explicit = 0
        else:
            if self.matched_descriptor not in self.eligible:
                raise RuntimeError(
                    "pinned descriptor is absent from candidate enumeration"
                )
            self.explicit = self.matched_descriptor
            if not self.runner.validate_tactic(self.inputs, self.explicit):
                raise RuntimeError("candidate schema validator rejected the descriptor")
            if any(
                self.runner.validate_tactic(self.inputs, bad)
                for bad in (0, (), (1,) * 7, (-1,) * 8)
            ):
                raise RuntimeError(
                    "candidate schema validator accepted an ordinal or malformed descriptor"
                )
        self.cache_key = self.tuner._get_cache_key(
            "fp8_gemm",
            self.runner,
            tuple(tuple(t.shape) for t in self.inputs),
            self.config,
            self.runner.get_cache_key_extras(self.inputs),
        )

    @contextlib.contextmanager
    def binding_guard(self, expected, expected_calls=1):
        """Observe native acceptance outside latency windows, without replacing it."""
        if self.args.implementation != "candidate":
            yield
            return
        experiment = self
        native = self.native_module
        observed = []

        class CheckedNative:
            def bmm_fp8_run_with_descriptor(self, *args, **kwargs):
                used = native.bmm_fp8_run_with_descriptor(*args, **kwargs)
                observed.append(int(used))
                if int(used) != 1:
                    raise RuntimeError("descriptor fell back to the heuristic default")
                return used

            def __getattr__(self, name):
                return getattr(native, name)

        if self.native_cell.cell_contents is not native:
            raise RuntimeError("nested native module patch")
        self.native_cell.cell_contents = CheckedNative()
        try:
            yield
        finally:
            self.native_cell.cell_contents = native
            experiment.binding_checks.append(
                {
                    "stage": self.stage,
                    "expected_tactic": tactic_json(expected),
                    "used_descriptor": observed,
                }
            )
        if expected != -1 and len(observed) != expected_calls:
            raise RuntimeError(
                "explicit descriptor did not reach its native entry point the expected number of times"
            )
        if expected == -1 and observed:
            raise RuntimeError(
                "default tactic unexpectedly used the descriptor entry point"
            )

    @contextlib.contextmanager
    def lookup_guard(self, expected, require_hit=True):
        last_hit = False
        observations = []

        def search(*args, **kwargs):
            nonlocal last_hit
            result = self.original_search(*args, **kwargs)
            last_hit = bool(result[0])
            return result

        def choose(op, runners, config, inputs, **kwargs):
            nonlocal last_hit
            last_hit = False
            runner, tactic = self.original_choose(op, runners, config, inputs, **kwargs)
            if op != "fp8_gemm" or tactic != expected or (require_hit and not last_hit):
                raise RuntimeError(
                    "public operation did not consume its expected real cache winner"
                )
            if self.args.implementation == "candidate" and not runner.validate_tactic(
                inputs, tactic
            ):
                raise RuntimeError("candidate returned an invalid cached tactic")
            observations.append(
                {
                    "stage": self.stage,
                    "cache_hit": last_hit,
                    "tactic": tactic_json(tactic),
                    "capturing": self.torch.cuda.is_current_stream_capturing(),
                }
            )
            return runner, tactic

        with (
            patch.object(self.tuner, "search_cache", search),
            patch.object(self.tuner, "choose_one", choose),
        ):
            yield
        if not observations:
            raise RuntimeError("missing public tuner lookup observation")
        self.lookups.extend(observations)

    @contextlib.contextmanager
    def forced(self, tactic):
        winners = self.tuner._winner_cache()
        saved = winners.copy()
        winners.clear()
        winners[self.cache_key] = (tactic, None)
        try:
            yield
        finally:
            winners.clear()
            winners.update(saved)

    def checked_call(self, expected, label):
        self.stage = label
        self.out.fill_(float("nan"))
        with self.lookup_guard(expected), self.binding_guard(expected):
            self.operation()
        self.check_output(label)

    def capture(self, expected, label):
        self.stage = label
        graph = self.torch.cuda.CUDAGraph()
        with (
            self.lookup_guard(expected),
            self.binding_guard(expected),
            self.torch.cuda.graph(graph),
        ):
            self.operation()
        return graph

    def check_graph(self, graph, label):
        saved_a, saved_reference = self.a.clone(), self.reference
        try:
            self.a.zero_()
            self.reference = self.torch.zeros_like(saved_reference)
            self.out.fill_(float("nan"))
            graph.replay()
            self.check_output(label + "_changed_input")
        finally:
            self.a.copy_(saved_a)
            self.reference = saved_reference
        self.out.fill_(float("nan"))
        graph.replay()
        self.check_output(label + "_restored_input")

    def runtime(self):
        variants = [
            {"name": "default", "tactic": -1},
            {"name": "explicit", "tactic": self.explicit},
        ]
        random.Random(self.args.seed).shuffle(variants)
        self.variant_order = [v["name"] for v in variants]
        for variant in variants:
            name, tactic = variant["name"], variant["tactic"]
            with self.forced(tactic):
                if self.args.mode == "observe":
                    graph = self.observe_variant(variant)
                else:
                    self.checked_call(tactic, name + "_before")
                    graph = self.capture(tactic, name + "_capture")
                self.check_graph(graph, name)
                if self.args.mode == "benchmark":
                    row = {"name": name, "tactic": tactic_json(tactic)}
                    for execution, call in (
                        ("eager", self.operation),
                        ("cuda_graph", graph.replay),
                    ):
                        self.out.fill_(float("nan"))
                        row[execution] = measure(self.torch, call, self.args)
                        self.check_output(name + "_measured_" + execution)
                    self.measurements.append(row)
                elif self.args.trace:
                    self.traces.append(
                        trace_variant(
                            self.torch, self.operation, graph, variant, self.args
                        )
                    )
                    self.check_output(name + "_traced")
                self.checked_call(tactic, name + "_after")

    def observe_variant(self, variant):
        """Enable the native probe before the first descriptor execution."""
        name, tactic = variant["name"], variant["tactic"]
        graph = self.torch.cuda.CUDAGraph()
        for phase, calls in (
            ("first_eager", 1),
            ("warm_eager", self.args.observation_calls),
            ("capture", 1),
            ("graph_replay", self.args.observation_calls),
        ):
            self.stage = name + "_" + phase
            recorder = BoundaryRecorder()
            self.native_probe(self.stage.encode())
            try:
                with contextlib.ExitStack() as stack:
                    if phase != "graph_replay":
                        stack.enter_context(self.lookup_guard(tactic))
                    stack.enter_context(
                        patch.object(
                            self.tuner,
                            "choose_one",
                            recorder.wrap("tuner_choose_one", self.tuner.choose_one),
                        )
                    )
                    if phase == "capture":
                        with self.binding_guard(tactic), self.torch.cuda.graph(graph):
                            self.operation()
                    elif phase == "graph_replay":
                        for _ in range(calls):
                            graph.replay()
                    else:
                        with self.binding_guard(tactic, expected_calls=calls):
                            for _ in range(calls):
                                self.operation()
                    self.torch.cuda.synchronize()
            finally:
                self.native_probe(None)
            self.observation_summaries.append(
                {
                    "phase": self.stage,
                    "calls": calls,
                    "cpu_boundaries": recorder.summary(),
                    "scope": "instrumented observation only; not a latency estimate",
                }
            )
            self.check_output(self.stage)
        return graph

    def lifecycle(self):
        from sglang.srt.model_executor.runner import flashinfer_autotune as startup
        from sglang.srt.runtime_context import get_context, get_parallel

        args = self.args
        cold = args.mode == "lifecycle-cold"
        natural_path = args.cache
        explicit_path = natural_path.with_name(natural_path.stem + ".explicit.json")
        for path in (natural_path, explicit_path):
            if cold and path.exists():
                raise RuntimeError("cold lifecycle refuses to overwrite existing cache")
            if not cold and not path.is_file():
                raise RuntimeError(
                    "warm lifecycle requires both preceding cold cache files"
                )
        natural_path.parent.mkdir(parents=True, exist_ok=True)
        before = {
            name: digest(path) if path.is_file() else None
            for name, path in (("natural", natural_path), ("explicit", explicit_path))
        }
        model_runner = SimpleNamespace(
            device="cuda",
            forward_stream=self.torch.cuda.Stream(),
            is_draft_worker=False,
        )
        caller_stream = self.torch.cuda.current_stream()
        startup_selections = []

        def startup_choose(op, runners, config, inputs, **kwargs):
            if (
                op != "fp8_gemm"
                or self.torch.cuda.current_stream() != model_runner.forward_stream
                or self.torch.cuda.is_current_stream_capturing()
            ):
                raise RuntimeError(
                    "startup operation ran outside the wrapper forward stream"
                )
            selected, tactic = self.original_choose(
                op, runners, config, inputs, **kwargs
            )
            startup_selections.append(tactic_json(tactic))
            return selected, tactic

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                get_context().override_server_args(
                    moe_runner_backend="triton", flashinfer_autotune_skip_ops=[]
                )
            )
            stack.enter_context(
                get_parallel().override(tp_group=SimpleNamespace(world_size=1))
            )
            stack.enter_context(
                patch.object(
                    startup, "flashinfer_autotune_cache_path", lambda _: natural_path
                )
            )
            if args.tuning_version == "v1":
                stack.enter_context(
                    patch.object(
                        self.flashinfer,
                        "autotune_v2",
                        lambda **_: contextlib.nullcontext(),
                    )
                )
            stack.enter_context(patch.object(self.tuner, "choose_one", startup_choose))
            self.stage = "startup"
            self.allow_profiling = cold
            started = time.perf_counter()
            try:
                with startup.flashinfer_autotune_context(
                    model_runner, run_lm_head=False
                ):
                    self.operation()
                self.torch.cuda.synchronize()
            finally:
                self.allow_profiling = False
            startup_ms = (time.perf_counter() - started) * 1000
        if (
            self.torch.cuda.current_stream() != caller_stream
            or self.tuner.is_tuning_mode
            or self.tuner._active_managed_store
        ):
            raise RuntimeError("startup failed to restore stream/tuning/store state")
        startup_profile_attempts = self.profile_attempts
        if (
            not startup_selections
            or (cold and not self.profiles)
            or (not cold and self.profiles)
        ):
            raise RuntimeError(
                "cold/warm lifecycle profiling participation is incorrect"
            )
        # Obtain the actual winner from production lookup, without mistaking -1 fallback for a hit.
        found = {}

        def actual_choice(op, runners, config, inputs, **kwargs):
            selected, tactic = self.original_choose(
                op, runners, config, inputs, **kwargs
            )
            found["tactic"] = tactic
            return selected, tactic

        with patch.object(self.tuner, "choose_one", actual_choice):
            self.operation()
        natural = found["tactic"]
        self.checked_call(natural, "natural_bare_after_context")
        graph = self.capture(natural, "natural_capture_after_context")
        self.check_graph(graph, "natural_graph")
        natural_hash = digest(natural_path)
        self.tuner.clear_cache()
        if cold:
            self.tuner._winner_cache()[self.cache_key] = (self.explicit, None)
            self.tuner.save_configs(str(explicit_path))
            self.checked_call(self.explicit, "explicit_before_save_reload")
            self.tuner.clear_cache()
        if not self.tuner.load_configs(str(explicit_path)):
            raise RuntimeError("explicit descriptor cache failed real load_configs")
        self.checked_call(self.explicit, "explicit_bare_after_reload")
        explicit_graph = self.capture(self.explicit, "explicit_capture_after_reload")
        self.check_graph(explicit_graph, "explicit_persisted_graph")
        if digest(natural_path) != natural_hash:
            raise RuntimeError("explicit control changed the natural cache file")
        if not cold and digest(explicit_path) != before["explicit"]:
            raise RuntimeError("warm control modified the cold explicit cache")
        if not cold and digest(natural_path) != before["natural"]:
            raise RuntimeError("warm startup modified the cold natural cache")
        if (
            self.profile_attempts != startup_profile_attempts
            or self.forbidden_profile_attempts
        ):
            raise RuntimeError(
                "profiling was attempted after startup or during warm reuse"
            )
        payload = json.loads(explicit_path.read_text())
        value = payload.get(self.cache_key.file_key)
        if not value or value[1] != tactic_json(self.explicit):
            raise RuntimeError(
                "persisted explicit tactic differs from the matched descriptor"
            )
        if self.args.tuning_version == "v2" and cold:
            runtime = self.cache_key.nearest_profile
            finite_default = [
                p
                for p in self.profiles
                if p["tactic"] == -1
                and p.get("profiler_ms") is not None
                and self.tuner._find_nearest_profile(
                    tuple(
                        tuple(s) if s is not None else (0,) for s in p["input_shapes"]
                    ),
                    self.config,
                )
                == runtime
            ]
            if not finite_default:
                raise RuntimeError("v2 default had no finite runtime-bucket profile")
        self.lifecycle_result = {
            "tuning_version": args.tuning_version,
            "phase": "cold" if cold else "warm",
            "natural_winner": tactic_json(natural),
            "explicit_control": tactic_json(self.explicit),
            "startup_ms": startup_ms,
            "profile_calls": len(self.profiles),
            "profile_attempts": self.profile_attempts,
            "natural_cache": str(natural_path),
            "explicit_cache": str(explicit_path),
            "cache_sha256_before": before,
            "cache_sha256_after": {
                "natural": digest(natural_path),
                "explicit": digest(explicit_path),
            },
            "explicit_file_entry": value,
            "scope": "actual startup wrapper and existing v1 JSON persistence; forced explicit file is a separate persistence control, not a profiled winner",
        }

    def run(self):
        from flashinfer.autotuner import autotune

        with patch.object(self.tuner, "_profile_single_kernel", self.profile_hook):
            with autotune(False):
                self.capture_inputs()
            if self.args.mode.startswith("lifecycle-"):
                self.lifecycle()
            else:
                self.runtime()
        if self.native_probe:
            self.native_probe(None)
        if self.tuner.is_tuning_mode or self.tuner._active_managed_store:
            raise RuntimeError("worker leaked tuning or managed-store state")
        if self.forbidden_profile_attempts:
            raise RuntimeError("a forbidden profile attempt was swallowed by the tuner")
        if self.operand_digests() != self.operands:
            raise RuntimeError("fixed-workload operands changed")
        torch = self.torch
        root = Path(__file__).resolve().parents[3]
        return {
            "status": "success",
            "schema_version": 1,
            "mode": self.args.mode,
            "implementation": self.args.implementation,
            "pid": os.getpid(),
            "seed": self.args.seed,
            "shape": [self.args.m, self.args.n, self.args.k],
            "worker_sha256": digest(__file__),
            "manifest_sha256": digest(self.args.manifest),
            "helper_sha256": digest(
                Path(__file__).with_name("flashinfer_autotune_eager_setup.py")
            ),
            "environment": {
                "sglang_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=root, text=True
                ).strip(),
                "sglang_wrapper_sha256": digest(
                    root
                    / "python/sglang/srt/model_executor/runner/flashinfer_autotune.py"
                ),
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "flashinfer": self.flashinfer.__version__,
                "flashinfer_source_sha256": self.source_hashes,
                "flashinfer_tuner_sha256": digest(self.tuner_source),
                "native_artifact": self.native_artifact,
                "gpu": torch.cuda.get_device_properties(0).name,
                "physical_gpu_uuid": str(
                    getattr(torch.cuda.get_device_properties(0), "uuid", None)
                ),
                "capability": list(torch.cuda.get_device_capability()),
                "gpu_status_before": self.initial_gpu_status,
                "gpu_status_after": gpu_status(),
                "LD_PRELOAD": os.environ.get("LD_PRELOAD"),
                "native_probe_output": os.environ.get("FI_NATIVE_PROBE_OUTPUT"),
                "measurement_policy": "unspecified",
                "backend": "cublas",
                "scope": "one fixed FP8 operation; no model/server",
            },
            "matched_algorithm": {
                "descriptor": list(self.matched_descriptor),
                "raw_hex": self.matched_raw_hex,
                "match_provenance": self.match_provenance,
                "actual_tactic": tactic_json(self.explicit),
                "eligible_tactics": [tactic_json(t) for t in self.eligible],
                "operand_sha256": self.operands,
                "identity_scope": "exact descriptor matched to baseline ordinal0 at fixed M128; native observer separately confirms submitted algorithm and launch",
            },
            "timing_protocol": {
                "warmup_ms": self.args.warmup_ms,
                "min_sample_ms": self.args.min_sample_ms,
                "samples": self.args.samples,
                "requested_iterations": self.args.iterations,
                "variant_order": getattr(self, "variant_order", []),
                "instrumentation": "none inside latency windows; guards and acceptance checks occur separately",
            },
            "measurements": self.measurements,
            "observations": self.observation_summaries,
            "traces": self.traces,
            "lookups": self.lookups,
            "binding_acceptance_checks": self.binding_checks,
            "profiles": self.profiles,
            "profile_attempts": self.profile_attempts,
            "forbidden_profile_attempts": self.forbidden_profile_attempts,
            "lifecycle": getattr(self, "lifecycle_result", None),
            "correctness": {
                "passed": True,
                "rtol": 0.02,
                "atol": 0.125,
                "checks": self.correctness,
            },
        }


def main():
    args = parse_args()
    try:
        result = Experiment(args).run()
    except Exception as error:
        traceback.print_exc()
        result = {
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "pid": os.getpid(),
            "mode": args.mode,
            "implementation": args.implementation,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
