"""Actual default/v1/v2 startup and fresh-cache reuse on one MLP projection.

This manual study keeps the normal SGLang FP8 facade and legacy persistence.
Independent natural/default graphs remain measured even when both use -1.
No confirmation heuristic, candidate oracle or production change is installed.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import dataclasses
import hashlib
import inspect
import json
import math
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bench_flashinfer_objective_values import (
    HELPER_SHA,
    MANIFEST_SHA,
    PIN,
    SOURCE_SHA,
)
from bench_flashinfer_objective_values import Experiment as CurrentExperiment
from bench_flashinfer_objective_values import (
    require,
    sha,
    tactic_json,
)
from flashinfer_autotune_descriptor_selection import SelectionExperiment
from flashinfer_autotune_eager_setup import MAX_ITERATIONS
from flashinfer_autotune_linear_layer import (
    FIXTURE_KEYS,
    MODEL_ID,
    MODEL_REVISION,
    LinearLayerExperiment,
    tensor_metadata,
    tensor_sha,
)

ROLES = ("natural", "default")
MODES = ("eager", "cuda_graph")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("default", "v1", "v2"), required=True)
    parser.add_argument("--phase", choices=("cold", "warm"), required=True)
    parser.add_argument("--trial", type=int, choices=range(6), required=True)
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
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup-ms", type=float, default=500)
    parser.add_argument("--min-sample-ms", type=float, default=20)
    args = parser.parse_args(argv)
    require(
        (args.blocks, args.iterations, args.warmup_ms, args.min_sample_ms)
        == (12, 20, 500, 20),
        "timing arguments differ from the frozen contract",
    )
    args.seed, args.oracle = 19, False
    return args


def assignment(schedule, args):
    matches = [
        r
        for r in schedule["entries"]
        if (r["trial"], r["arm"], r["phase"]) == (args.trial, args.arm, args.phase)
    ]
    require(len(matches) == 1, "schedule must identify one process")
    row = matches[0]
    require(
        row["capture_order"] == args.capture_order
        and row["mode_order"] == args.mode_order,
        "requested order differs from frozen schedule",
    )
    schedules = schedule["round_schedules"][str(args.trial)]
    for mode in MODES:
        orders = schedules[mode]
        require(
            len(orders) == 12
            and all(sorted(r) == sorted(ROLES) for r in orders)
            and sum(r[0] == "natural" for r in orders) == 6,
            "paired schedule is not balanced",
        )
    return row, schedules


def pure_batch(torch, call, iterations):
    """Only the actual operation is repeated inside the event/wall window."""
    require(
        type(iterations) is int and 1 <= iterations <= MAX_ITERATIONS,
        "invalid batch iteration count",
    )
    first, last = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    first.record()
    last.record()
    last.synchronize()
    started = time.perf_counter_ns()
    first.record()
    for _ in range(iterations):
        call()
    last.record()
    last.synchronize()
    ended = time.perf_counter_ns()
    event_ms, wall_ms = first.elapsed_time(last), (ended - started) / 1e6
    require(
        math.isfinite(event_ms)
        and event_ms > 0
        and math.isfinite(wall_ms)
        and wall_ms > 0,
        "invalid paired batch duration",
    )
    return {
        "iterations": iterations,
        "started_ns": started,
        "ended_ns": ended,
        "wall_batch_ms": wall_ms,
        "gpu_event_batch_ms": event_ms,
        "wall_per_call_ms": wall_ms / iterations,
        "gpu_event_per_call_ms": event_ms / iterations,
        "events_primed_before_window": True,
    }


def queued_graph_batch(torch, call, progress):
    """Measure sixteen retained replays after an uncharged queueing delay."""
    first, last = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    first.record()
    last.record()
    last.synchronize()
    progress.update(
        iterations=16,
        prequeue_cycles=10_000_000,
        start_query_error=None,
        events_primed_before_window=True,
        queue_query_is_exclusion_criterion=False,
    )
    progress["started_ns"] = time.perf_counter_ns()
    torch.cuda._sleep(10_000_000)
    progress["enqueue_started_ns"] = time.perf_counter_ns()
    first.record()
    for _ in range(16):
        call()
    last.record()
    try:
        complete = first.query()
    except Exception as error:
        progress["start_query_error"] = f"{type(error).__name__}: {error}"
        progress["query_error_observed_ns"] = time.perf_counter_ns()
        raise
    progress["enqueue_ended_ns"] = time.perf_counter_ns()
    last.synchronize()
    progress["ended_ns"] = time.perf_counter_ns()
    elapsed = first.elapsed_time(last)
    require(
        type(complete) is bool and math.isfinite(elapsed) and elapsed > 0,
        "invalid graph queue outcome or event duration",
    )
    wall = (progress["ended_ns"] - progress["started_ns"]) / 1e6
    progress.update(
        wall_batch_ms=wall,
        wall_per_call_ms=wall / 16,
        gpu_event_batch_ms=elapsed,
        gpu_event_per_call_ms=elapsed / 16,
        wall_diagnostic_only=True,
        wall_includes_prequeue=True,
        enqueue_cpu_ms=(progress["enqueue_ended_ns"] - progress["enqueue_started_ns"])
        / 1e6,
        start_event_complete_after_enqueue=complete,
        queue_evidence="start_complete_inconclusive"
        if complete
        else "start_pending_after_enqueue",
    )
    return dict(progress)


class Experiment(LinearLayerExperiment):
    # These current-source helpers have no fixed shape or zero-argument super.
    cache_state_sha = CurrentExperiment.cache_state_sha
    miss_scope = CurrentExperiment.miss_scope
    assert_bare = CurrentExperiment.assert_bare
    scope_state = CurrentExperiment.scope_state
    metadata = staticmethod(CurrentExperiment.metadata)

    def __init__(self, args):
        self.args = args
        self.plan, self.schedule = (
            json.loads(args.plan.read_text()),
            json.loads(args.schedule.read_text()),
        )
        self.entry, self.orders = assignment(self.schedule, args)
        require(
            self.plan["schedule_sha256"] == sha(args.schedule),
            "plan does not bind schedule",
        )
        require(
            self.plan["fixture_sha256"] is not None
            and self.plan["fixture_manifest_sha256"] is not None
            and sha(args.fixture) == self.plan["fixture_sha256"]
            and sha(args.fixture_manifest) == self.plan["fixture_manifest_sha256"],
            "fixture has not been frozen or differs",
        )
        require(
            not os.environ.get("LD_PRELOAD")
            and not getattr(ctypes.CDLL(None), "fi_probe_set_phase", None),
            "native preload is forbidden",
        )
        os.environ["FLASHINFER_AUTOTUNER_LOAD_FROM_FILE"] = "0"
        os.environ["SGLANG_FLASHINFER_AUTOTUNE_CACHE"] = "1"
        require(
            (args.phase == "cold" and not args.cache.exists())
            or (args.phase == "warm" and args.cache.is_file()),
            "cold/warm cache precondition failed",
        )
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        self.cache_before = sha(args.cache) if args.cache.is_file() else None

        import flashinfer
        import objective_values_bootstrap as bootstrap
        import torch
        from flashinfer.autotuner import AutoTuner
        from flashinfer.gemm import gemm_base
        from safetensors.torch import load_file

        from sglang.kernels.ops.quantization import fp8_kernel
        from sglang.srt.layers.quantization import fp8_utils
        from sglang.srt.utils import custom_op

        self.torch, self.flashinfer, self.gemm_base = torch, flashinfer, gemm_base
        require(
            bootstrap.helper_sha256 == HELPER_SHA
            and bootstrap.candidate_manifest_sha256 == MANIFEST_SHA
            and sha(args.manifest) == MANIFEST_SHA,
            "qualified current bootstrap differs",
        )
        current = Path(bootstrap.source_root).resolve()
        require(
            self.plan["source_contract"]["runtime_sha256"] == SOURCE_SHA
            and all(
                sha(current / name) == expected for name, expected in SOURCE_SHA.items()
            )
            and Path(flashinfer.__file__).resolve().is_relative_to(current),
            "mixed or different current source",
        )
        require(
            torch.__version__ == "2.14.1+cu130"
            and torch.version.cuda == "13.0"
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability() == (9, 0),
            "pinned H100 stack required",
        )
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        require(
            torch.is_inference_mode_enabled() and not torch.is_grad_enabled(),
            "inference_mode required",
        )
        require(
            flashinfer.bmm_fp8 is gemm_base.bmm_fp8
            and hasattr(flashinfer.bmm_fp8, "__wrapped__")
            and fp8_utils._raw_flashinfer_bmm_fp8 is flashinfer.bmm_fp8
            and fp8_utils.static_quant_fp8 is fp8_kernel.static_quant_fp8
            and fp8_utils.flashinfer_per_tensor_fp8_supported(),
            "original decorated facade or quantizer differs",
        )
        source_root = Path(__file__).resolve().parents[3]
        require(
            all(
                Path(module.__file__).resolve().is_relative_to(source_root / "python")
                for module in (fp8_utils, fp8_kernel, custom_op)
            ),
            "mixed SGLang import",
        )
        manifest = json.loads(args.fixture_manifest.read_text())
        self.fixture_manifest = manifest
        args.m, args.n, args.k = manifest["shape"]
        require(
            manifest["schema_version"] == 2
            and manifest["fixture_kind"] == "qwen3_mlp_down_proj"
            and manifest["model"]["repo_id"] == MODEL_ID
            and manifest["model"]["revision"] == MODEL_REVISION
            and [args.m, args.n, args.k] == [144, 1024, 3072],
            "MLP fixture model or shape differs",
        )
        capture = manifest["capture"]
        require(
            capture["weight_order"] == ["down_proj"]
            and capture["bias"] is None
            and capture["saved_shape"] == [args.m, args.k]
            and capture["captured_shape"] == [1, args.m, args.k]
            and capture["seed"] == 19,
            "MLP activation/weight capture contract differs",
        )
        require(
            manifest["tensor_file"]["sha256"] == sha(args.fixture)
            and manifest["tensor_file"]["size_bytes"] == args.fixture.stat().st_size,
            "fixture bytes differ",
        )
        self.layer_sources = manifest["source"]["sglang_source_sha256"]
        require(
            all(
                sha(source_root / name) == expected
                for name, expected in self.layer_sources.items()
            ),
            "fixture/runtime source differs",
        )
        cpu = load_file(str(args.fixture), device="cpu")
        require(
            set(cpu) == set(manifest["tensors"]) == FIXTURE_KEYS,
            "fixture tensor keys differ",
        )
        expected = {
            "input_bf16": (torch.bfloat16, [args.m, args.k]),
            "weight_fp8_rowmajor": (torch.float8_e4m3fn, [args.n, args.k]),
            "weight_bf16_rowmajor": (torch.bfloat16, [args.n, args.k]),
            "input_scale": (torch.float32, [1]),
            "weight_scale": (torch.float32, [1]),
            "input_ids": (torch.int64, [1, args.m]),
        }
        for name, tensor in cpu.items():
            dtype, shape = expected[name]
            require(
                tensor.dtype == dtype
                and list(tensor.shape) == shape
                and tensor.is_contiguous()
                and tensor_metadata(torch, tensor) == manifest["tensors"][name]
                and bool(torch.isfinite(tensor.float()).all()),
                "fixture tensor differs: " + name,
            )
        require(
            all(
                bool(cpu[name].gt(0).all()) for name in ("input_scale", "weight_scale")
            ),
            "invalid scalar scale",
        )
        self.a, self.weight_fp8_rowmajor, self.weight_bf16_rowmajor = (
            cpu[name].cuda()
            for name in ("input_bf16", "weight_fp8_rowmajor", "weight_bf16_rowmajor")
        )
        self.b = self.weight_fp8_rowmajor.t()
        self.scale_a, self.scale_b = (
            cpu[name].cuda() for name in ("input_scale", "weight_scale")
        )
        self.input_ids = cpu["input_ids"]
        self.apply_layer = fp8_utils.apply_fp8_linear_bmm_flashinfer
        self.out = torch.empty((args.m, args.n), dtype=torch.bfloat16, device="cuda")
        qinput, scale = fp8_kernel.static_quant_fp8(
            self.a, self.scale_a, repeat_scale=False
        )
        independent = (
            (self.a.float() * self.scale_a.reciprocal())
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
        )
        expected_q = manifest["quantization"]["independent_input_fp8_sha256"]
        self.quantization_evidence = {
            "actual_sha256": tensor_sha(torch, qinput),
            "independent_gpu_sha256": tensor_sha(torch, independent),
            "expected_sha256": expected_q,
            "scale_pointer_unchanged": scale.data_ptr() == self.scale_a.data_ptr(),
        }
        require(
            self.quantization_evidence["actual_sha256"]
            == self.quantization_evidence["independent_gpu_sha256"]
            == expected_q
            and self.quantization_evidence["scale_pointer_unchanged"],
            "actual static quantizer bytes differ: "
            + json.dumps(self.quantization_evidence, sort_keys=True),
        )
        self.quantized_input_sha = expected_q
        self.reference = (qinput.float() @ self.b.float()) * self.scale_a * self.scale_b
        self.bf16_reference = self.a.float() @ self.weight_bf16_rowmajor.float().t()
        self.zero_expected = False
        self.tuner = AutoTuner.get()
        require(
            not self.tuner._winner_cache()
            and not self.tuner._file_configs
            and not self.tuner._winner_partitions
            and not self.tuner._active_managed_store
            and not self.tuner.is_tuning_mode
            and not self.tuner._in_v2_context,
            "process has preexisting tuner state",
        )
        self.native_module = bootstrap.native_gemm
        self.qualified_native_module = bootstrap.native_gemm
        self.native_artifact = bootstrap.evidence["native"]
        factory = gemm_base.get_gemm_module().cublas_fp8_gemm_runner
        closure = dict(zip(factory.__code__.co_freevars, factory.__closure__))
        self.native_cell = closure["module"]
        require(
            self.native_cell.cell_contents is self.native_module,
            "native provider was replaced",
        )
        self.model_runner = SimpleNamespace(
            device="cuda", forward_stream=torch.cuda.Stream(), is_draft_worker=False
        )
        self.caller_stream = torch.cuda.current_stream()
        self.original_choose, self.original_search, self.original_profile = (
            self.tuner.choose_one,
            self.tuner.search_cache,
            self.tuner._profile_single_kernel,
        )
        self.profile_attempts = 0
        self.profile_protocol_errors = []
        (
            self.forbidden_profile_attempts,
            self.profiles,
            self.selections,
            self.correctness,
            self.binding_checks,
        ) = [], [], [], [], []
        self.observed_public_inputs, self.native_output_poison_checks = [], []
        self.stage, self.allow_profiling, self.captured = "setup", False, {}
        self.last_hit, self.last_tier, self.last_tactic = False, "miss", None
        self.graphs = {}
        self.actual_layer = {
            "function": "apply_fp8_linear_bmm_flashinfer",
            "custom_op": "flashinfer_bmm_fp8",
            "quantizer": "static_quant_fp8",
            "backend": "cublas",
            "source_sha256": self.layer_sources,
            "function_source_sha256": hashlib.sha256(
                inspect.getsource(self.apply_layer).encode()
            ).hexdigest(),
            "inference_mode": True,
            "grad_enabled": False,
            "allow_tf32": False,
            "initializer_override": False,
        }
        self.actual_layer.update(
            dispatch_modes=torch._C._len_torch_dispatch_stack(),
            function_modes=torch._C._is_torch_function_mode_enabled(),
            torch_profiler_enabled=torch._C._autograd._profiler_enabled(),
            compiling=torch.compiler.is_compiling(),
            kernel_api_log_level=custom_op._KERNEL_API_LOG_LEVEL,
            custom_op_direct_wrapper=hasattr(
                fp8_utils.flashinfer_bmm_fp8, "__wrapped__"
            ),
            quantization_reference={
                "byte_match": True,
                "actual_input_fp8_sha256": tensor_sha(torch, qinput),
                "independent_input_fp8_sha256": expected_q,
            },
        )
        require(
            not any(
                self.actual_layer[name]
                for name in (
                    "dispatch_modes",
                    "function_modes",
                    "torch_profiler_enabled",
                    "compiling",
                    "kernel_api_log_level",
                )
            )
            and self.actual_layer["custom_op_direct_wrapper"],
            "normal inference fast path is not active",
        )
        self.result = {
            "schema_version": 2,
            "study": "mlp_default_v1_v2",
            "status": "running",
            "arm": args.arm,
            "phase": args.phase,
            "trial": args.trial,
            "mode": args.mode,
            "pid": os.getpid(),
            "assignment": self.entry,
            "worker_sha256": sha(__file__),
            "plan_sha256": sha(args.plan),
            "schedule_sha256": sha(args.schedule),
            "manifest_sha256": sha(args.manifest),
            "miss_scopes": [],
            "graphs": [],
            "measurements": {},
            "fixture": {
                "manifest_sha256": sha(args.fixture_manifest),
                "tensor_file_sha256": sha(args.fixture),
                **manifest,
            },
            "source": {
                "flashinfer_commit": PIN,
                "critical_sha256": SOURCE_SHA,
                "sglang_source_sha256": self.layer_sources,
                "native_identity_passthrough": True,
            },
            "environment": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
                "flashinfer": flashinfer.__version__,
                "native_artifact": self.native_artifact,
            },
            "actual_layer": self.actual_layer,
        }
        self.operands = self.operand_digests()
        self.tuner_settings_initial = self.tuner_settings()
        require(
            self.tuner_settings_initial["effective_measure_policy"] is None
            and not self.tuner_settings_initial["use_global_timer"]
            and gemm_base._FP8_GEMM_SM100_TUNING_CONFIG.tensor_initializers == ()
            and gemm_base._FP8_GEMM_SM100_TUNING_CONFIG.inputs_pre_hook is None,
            "initial measurement policy or initializer was overridden",
        )

    def profile_hook(self, runner, inputs, tactic, config, *more, **kwargs):
        index = len(self.profiles)
        try:
            return super().profile_hook(runner, inputs, tactic, config, *more, **kwargs)
        finally:
            if len(self.profiles) > index:
                row = self.profiles[index]
                key = self.tuner._get_cache_key(
                    "fp8_gemm",
                    runner,
                    tuple(tuple(t.shape) for t in inputs),
                    config,
                    runner.get_cache_key_extras(inputs),
                )
                row.update(
                    index=index,
                    m=inputs[0].shape[-2],
                    key=key.file_key,
                    metadata=[self.metadata(t) for t in inputs],
                    in_v2_context=self.tuner._in_v2_context,
                    is_tuning_mode=self.tuner.is_tuning_mode,
                    fingerprinted=self.args.mode == "admission",
                )
                if self.args.mode == "admission":
                    row["fingerprints"] = [
                        tensor_sha(self.torch, t) for t in inputs[:4]
                    ]
                row["status"] = (
                    "finite"
                    if row.get("profiler_ms") is not None
                    else "exception"
                    if row.get("error")
                    else "nonfinite_disqualified"
                )
                try:
                    settings = row["autotuner_settings"]
                    require(
                        row["in_v2_context"] == (self.args.arm == "v2")
                        and row["is_tuning_mode"]
                        and settings["warmup"] == 3
                        and settings["repeat"] == 10
                        and settings["stream_delay_micro_secs"] == 5000
                        and settings["cuda_graph_delay_micro_secs"] == 100
                        and not settings["use_global_timer"]
                        and settings["effective_measure_policy"] is None
                        and settings["effective_policy_timer"] is None
                        and not config.use_cuda_graph
                        and not config.use_cold_l2_cache
                        and config.cuda_graph_profile_replays == 1
                        and config.profiling_repeat is None
                        and self.torch.cuda.current_stream()
                        == self.model_runner.forward_stream,
                        "actual profiling context or timing policy differs",
                    )
                except Exception as error:
                    self.profile_protocol_errors.append(str(error))
                    raise

    def choose_hook(self, op, runners, config, inputs, **kwargs):
        returned = super().choose_hook(op, runners, config, inputs, **kwargs)
        policy = self.tuner._effective_measure_policy
        self.selections[-1].update(
            in_v2_context=self.tuner._in_v2_context,
            is_tuning_mode=self.tuner.is_tuning_mode,
            measurement_policy=dataclasses.asdict(policy)
            if policy is not None
            else None,
            active_tuning_contexts=self.tuner._active_tuning_contexts,
        )
        if self.stage == "startup":
            require(
                self.tuner._in_v2_context == (self.args.arm == "v2")
                and self.tuner.is_tuning_mode == (self.args.arm != "default")
                and policy is None
                and self.tuner._active_managed_store is None,
                "actual startup arm context differs",
            )
        return returned

    def check_output(self, label):
        role = getattr(self, "checking_graph_role", None)
        if role is not None:
            self.check_owner(role)
        super().check_output(label)
        self.correctness[-1].update(
            graph_id=role,
            output_data_ptr=self.out.data_ptr(),
            checked_ns=time.perf_counter_ns(),
        )

    def check_owner(self, role):
        saved = self.graphs[role]
        require(
            self.out.data_ptr()
            == saved["binding"].output.data_ptr()
            == saved["inputs"][4].data_ptr(),
            "numerical guard selected another graph output allocation",
        )

    @contextlib.contextmanager
    def graph_owner(self, role):
        previous = getattr(self, "checking_graph_role", None)
        self.checking_graph_role = role
        try:
            yield
        finally:
            self.checking_graph_role = previous

    def expected_tier(self):
        return (
            "miss"
            if self.args.arm == "default"
            else "memory"
            if self.args.phase == "cold"
            else "file"
        )

    def freeze_natural(self):
        tactic, selected, accepted = self.observed_call(
            "natural_before", tier=self.expected_tier()
        )
        self.natural_tactic = tactic
        self.runner, self.config, self.inputs = (
            self.captured[k] for k in ("runner", "config", "inputs")
        )
        self.cache_key = self.tuner._get_cache_key(
            "fp8_gemm",
            self.runner,
            tuple(tuple(t.shape) for t in self.inputs),
            self.config,
            self.runner.get_cache_key_extras(self.inputs),
        )
        runtime = [
            r
            for r in self.profiles
            if tuple(tuple(s) for s in r["normalized_profile"])
            == self.cache_key.nearest_profile
        ]
        finite = [r for r in runtime if r.get("profiler_ms") is not None]
        if self.args.phase == "cold" and self.args.arm != "default":
            require(
                finite
                and tactic_json(tactic)
                == min(finite, key=lambda r: r["profiler_ms"])["tactic"],
                "natural runtime tactic is not first finite profiling minimum",
            )
        require(
            self.args.arm != "default" or tactic == -1, "untuned control is not default"
        )
        if self.args.arm == "v1":
            require(
                not any(r["tactic"] == -1 for r in self.profiles),
                "v1 unexpectedly raced default",
            )
        if self.args.arm == "v2" and self.args.phase == "cold":
            grouped = {
                m: [r for r in self.profiles if r["m"] == m]
                for m in dict.fromkeys(r["m"] for r in self.profiles)
            }
            require(
                grouped
                and all(
                    sum(r["tactic"] == -1 for r in group) == 1
                    for group in grouped.values()
                ),
                "v2 default participation differs",
            )
        self.natural = {
            "tactic": tactic_json(tactic),
            "key": self.cache_key.file_key,
            "cache_hit": selected["cache_hit"],
            "lookup_tier": selected["lookup_tier"],
            "acceptance": accepted,
            "runtime_profiles": len(runtime),
            "runtime_finite_profiles": len(finite),
            "minimum_ms": min((r["profiler_ms"] for r in finite), default=None),
        }
        with self.zero_input():
            self.observed_call(
                "natural_eager_changed_input",
                expected=tactic,
                tier=self.expected_tier(),
            )
        self.observed_call(
            "natural_eager_restored_input", expected=tactic, tier=self.expected_tier()
        )
        self.memory_sha = self.cache_state_sha()
        self.memory_fields = dict(self.cache_state_field_sha256)

    def role_scope(self, role):
        require(role in ROLES, "unknown measurement role")
        return self.miss_scope() if role == "default" else contextlib.nullcontext()

    def role_tactic(self, role):
        return -1 if role == "default" else self.natural_tactic

    def role_tier(self, role):
        return "miss" if role == "default" else self.expected_tier()

    def capture_graphs(self):
        order = (
            ROLES
            if self.args.capture_order == "natural-first"
            else tuple(reversed(ROLES))
        )
        for role in order:
            with self.role_scope(role):
                self.observed_call(
                    role + "_capture_before",
                    expected=self.role_tactic(role),
                    tier=self.role_tier(role),
                )
                capture_started = time.perf_counter_ns()
                graph = self.capture(
                    self.role_tactic(role), self.role_tier(role), role + "_capture"
                )
                capture_ended = time.perf_counter_ns()
                saved = {
                    "binding": graph,
                    "inputs": tuple(self.captured["inputs"]),
                    "runner": self.captured["runner"],
                    "config": self.captured["config"],
                }
                self.graphs[role] = saved
                self.out = graph.output
                with self.graph_owner(role):
                    graph.output.fill_(float("nan"))
                    graph.replay()
                    self.check_output(role + "_poisoned")
                    self.check_graph(graph, role)
                self.result["graphs"].append(
                    {
                        "id": role,
                        "tactic": tactic_json(self.role_tactic(role)),
                        "lookup_tier": self.role_tier(role),
                        "selection": self.selections[-1],
                        "graph_object_id": id(graph.graph),
                        "inputs": [self.metadata(t) for t in saved["inputs"]],
                        "linear_input": self.metadata(self.a),
                        "output": self.metadata(graph.output),
                        "capture_started_ns": capture_started,
                        "capture_ended_ns": capture_ended,
                    }
                )
        require(
            len({g["binding"].output.data_ptr() for g in self.graphs.values()}) == 2,
            "independent graphs share output",
        )
        require(
            self.cache_state_sha() == self.memory_sha,
            "graph scopes changed natural cache state",
        )

    def callable(self, mode, role):
        return (
            self.operation
            if mode == "eager"
            else self.graphs[role]["binding"].graph.replay
        )

    def warm(self, call, milliseconds):
        started, calls = time.perf_counter_ns(), 0
        while True:
            for _ in range(256):
                call()
            self.torch.cuda.synchronize()
            calls += 256
            elapsed = (time.perf_counter_ns() - started) / 1e6
            if elapsed >= milliseconds:
                return {"actual_ms": elapsed, "calls": calls}

    def sample(self, mode, role, iterations, label):
        self.assert_bare()
        require(
            self.native_cell.cell_contents
            is self.native_module
            is self.qualified_native_module,
            "timing native provider is observed or replaced",
        )
        with self.role_scope(role):
            _, before, acceptance_before = self.observed_call(
                label + "_before",
                expected=self.role_tactic(role),
                tier=self.role_tier(role),
            )
            rewarm = self.warm(self.callable(mode, role), 20)
            graph = self.graphs[role]["binding"]
            if mode == "cuda_graph":
                require(
                    graph.output.data_ptr()
                    == self.graphs[role]["inputs"][4].data_ptr(),
                    "wrong retained graph output",
                )
                graph.output.fill_(float("nan"))
            self.stage = label + "_timed"
            count = self.profile_attempts
            progress = {"mode": mode, "role": role, "label": label}
            self.result["pending_sample"] = progress
            measured = (
                queued_graph_batch(self.torch, self.callable(mode, role), progress)
                if mode == "cuda_graph"
                else pure_batch(self.torch, self.callable(mode, role), iterations)
            )
            if mode == "cuda_graph":
                self.out = graph.output
            with (
                self.graph_owner(role)
                if mode == "cuda_graph"
                else contextlib.nullcontext()
            ):
                self.check_output(label + "_timed")
            check = self.correctness[-1]
            _, after, acceptance_after = self.observed_call(
                label + "_after",
                expected=self.role_tactic(role),
                tier=self.role_tier(role),
            )
            require(
                count == self.profile_attempts == self.startup_profile_attempts,
                "profiling inside serving measurement",
            )
            require(
                self.native_cell.cell_contents
                is self.native_module
                is self.qualified_native_module,
                "native observer leaked into timing",
            )
            self.result["pending_sample"] = None
            return {
                "role": role,
                "rewarm": rewarm,
                **measured,
                "lookup_before": before,
                "lookup_after": after,
                "acceptance_before": acceptance_before,
                "acceptance_after": acceptance_after,
                "correctness_check": check["check"],
                "output_data_ptr": check["output_data_ptr"],
                "profile_attempts_before": count,
                "profile_attempts_after": self.profile_attempts,
            }

    def measure_modes(self):
        modes = (
            MODES if self.args.mode_order == "eager-first" else tuple(reversed(MODES))
        )
        for mode in modes:
            group = {
                "started_ns": time.perf_counter_ns(),
                "initial_warmups": [],
                "calibration_by_role": {},
                "pairs": [],
                "batch_policy": "fixed16_queued"
                if mode == "cuda_graph"
                else "calibrated_wall",
            }
            self.result["measurements"][mode] = group
            for role in ROLES:
                with self.role_scope(role):
                    group["initial_warmups"].append(
                        {"role": role, **self.warm(self.callable(mode, role), 500)}
                    )
                if mode == "cuda_graph":
                    continue
                iterations, batches = self.args.iterations, []
                while True:
                    batch = self.sample(
                        mode,
                        role,
                        iterations,
                        f"calibrate_{mode}_{role}_{len(batches):02d}",
                    )
                    batches.append(batch)
                    if (
                        batch[
                            "wall_batch_ms" if mode == "eager" else "gpu_event_batch_ms"
                        ]
                        >= self.args.min_sample_ms
                    ):
                        break
                    require(
                        iterations < MAX_ITERATIONS,
                        "calibration exhausted iteration bound",
                    )
                    iterations = min(iterations * 2, MAX_ITERATIONS)
                group["calibration_by_role"][role] = {
                    "effective_iterations": iterations,
                    "batches": batches,
                }
            common = (
                16
                if mode == "cuda_graph"
                else max(
                    r["effective_iterations"]
                    for r in group["calibration_by_role"].values()
                )
            )
            group["common_iterations"] = common
            for index, order in enumerate(self.orders[mode]):
                pair = {"index": index, "order": order, "measurements": []}
                group["pairs"].append(pair)
                for role in order:
                    pair["measurements"].append(
                        self.sample(
                            mode, role, common, f"pair_{mode}_{index:02d}_{role}"
                        )
                    )
            group["ended_ns"] = time.perf_counter_ns()

    def trace(self):
        torch = self.torch
        require(not self.args.trace.exists(), "refusing to overwrite trace")
        started = time.perf_counter_ns()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            for role in ROLES:
                with torch.profiler.record_function("retained_" + role):
                    for _ in range(3):
                        self.graphs[role]["binding"].graph.replay()
                    torch.cuda.synchronize()
        self.args.trace.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(self.args.trace))
        for role in ROLES:
            self.out = self.graphs[role]["binding"].output
            with self.graph_owner(role):
                self.check_output("traced_" + role)
        return {
            "path": str(self.args.trace),
            "sha256": sha(self.args.trace),
            "started_ns": started,
            "ended_ns": time.perf_counter_ns(),
            "calls_per_graph": 3,
            "graphs": [
                {
                    "id": role,
                    "graph_object_id": id(self.graphs[role]["binding"].graph),
                    "output_data_ptr": self.graphs[role]["binding"].output.data_ptr(),
                }
                for role in ROLES
            ],
        }

    def run(self):
        with patch.object(self.tuner, "_profile_single_kernel", self.profile_hook):
            self.setup()
            require(
                self.native_module
                is self.native_cell.cell_contents
                is self.qualified_native_module,
                "setup replaced qualified native",
            )
            self.torch.cuda._sleep(10_000_000)
            self.torch.cuda.synchronize()
            self.result["delay_helper"] = {
                "function": "torch.cuda._sleep",
                "prime_cycles": 10_000_000,
                "primed_before_startup": True,
                "python_source_sha256": hashlib.sha256(
                    inspect.getsource(self.torch.cuda._sleep).encode()
                ).hexdigest(),
                "torch_extension_sha256": sha(self.torch._C.__file__),
            }
            self.torch.manual_seed(19)
            self.torch.cuda.manual_seed_all(19)
            self.startup()
            self.freeze_natural()
            self.capture_graphs()
            self.measure_modes()
            self.observed_call(
                "natural_final", expected=self.natural_tactic, tier=self.expected_tier()
            )
            self.result["trace"] = self.trace()
        self.assert_bare()
        require(
            self.profile_attempts == self.startup_profile_attempts
            and not self.forbidden_profile_attempts
            and not self.profile_protocol_errors,
            "unexpected profiling or invalid profile protocol",
        )
        require(
            self.operands == self.operand_digests()
            and sha(self.args.cache) == self.cache_after_startup
            and self.cache_state_sha() == self.memory_sha,
            "natural operands or cache changed",
        )
        self.result.update(
            status="success",
            shape=[self.args.m, self.args.n, self.args.k],
            seed=19,
            natural=self.natural,
            startup={
                "profile_attempts": self.startup_profile_attempts,
                "profiles": self.profiles,
                "startup_ms": self.startup_ms,
                "initial_public_call_ms": self.setup_ms,
                "selection": [r for r in self.selections if r["stage"] == "startup"],
            },
            cache={
                "before_sha256": self.cache_before,
                "after_startup_sha256": self.cache_after_startup,
                "final_sha256": sha(self.args.cache),
                "canonical_state_after_startup": self.memory_sha,
                "canonical_state_final": self.cache_state_sha(),
                "memory_entries": len(self.tuner._winner_cache()),
                "file_entries": len(self.tuner._file_configs),
            },
            selections=self.selections,
            correctness=self.correctness,
            binding_acceptance_checks=self.binding_checks,
            observed_public_inputs=self.observed_public_inputs,
            native_output_poison_checks=self.native_output_poison_checks,
            forbidden_profile_attempts=self.forbidden_profile_attempts,
            final={
                "profile_attempts": self.profile_attempts,
                "zero_additional_profiling": True,
                "cache_unchanged": True,
                "operands_unchanged": True,
                "native_provider_restored": self.native_cell.cell_contents
                is self.native_module,
                **self.scope_state(),
            },
        )
        self.result["autotuner_settings"] = {
            "initial": self.tuner_settings_initial,
            "final": self.tuner_settings(self.config),
        }
        self.result["profile_protocol_errors"] = self.profile_protocol_errors
        self.result.update(
            all_samples_retained=True,
            queue_query_is_exclusion_criterion=False,
            persistence_under_test=True,
        )
        self.result["source"]["manual_helpers_sha256"] = {
            Path(inspect.getfile(cls)).name: sha(inspect.getfile(cls))
            for cls in (CurrentExperiment, LinearLayerExperiment, SelectionExperiment)
        }
        return self.result


def main(argv=None):
    args = parse_args(argv)
    require(not args.output.exists(), "refusing to overwrite result")
    experiment = None
    try:
        import torch

        with torch.inference_mode():
            experiment = Experiment.__new__(Experiment)
            experiment.__init__(args)
            result = experiment.run()
    except Exception as error:
        traceback.print_exc()
        result = getattr(
            experiment,
            "result",
            {
                "arm": args.arm,
                "phase": args.phase,
                "trial": args.trial,
                "mode": args.mode,
                "pid": os.getpid(),
            },
        )
        result.update(status="error", error=f"{type(error).__name__}: {error}")
        if experiment is not None:
            result["partial_evidence"] = {
                name: getattr(experiment, name)
                for name in (
                    "quantization_evidence",
                    "profile_attempts",
                    "forbidden_profile_attempts",
                    "profile_protocol_errors",
                    "profiles",
                    "selections",
                    "correctness",
                    "binding_checks",
                )
                if hasattr(experiment, name)
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
