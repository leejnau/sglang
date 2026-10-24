"""Fixed objective-by-activation-values study through the actual FP8 layer.

Run with the qualified external current-source bootstrap. Admission records
profiling bytes; timing records metadata only. Neither mode changes the tuner,
timer, public kernel facade, candidate enumeration or persistent-store policy.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import hashlib
import inspect
import json
import math
import os
import statistics
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PIN = "27c62fdf0ab1cd4014df8a15b66a975541fda83c"
HELPER_SHA = "b8d0f6bbc85eaca0e3ef1419ab99a97d5340c96b3da382cba81d7896768287ba"
MANIFEST_SHA = "9a4c51562d7410d785cc544c004bc4f92486f40311cb5c7a7328c2c1213160d5"
FIXTURE_SHA = "c30aa761eba827bb0b4eb6de0e99d9b8dda8243dd16d00146f9718d2abb05e8b"
FIXTURE_MANIFEST_SHA = (
    "e13a86fab5db5796b22d8dab0f84080b8057b036ca7aef2c7be8832e76505bf1"
)
SOURCE_SHA = {
    "flashinfer/autotuner/autotuner.py": "0db42ad90d1a3fec6f81509a2c520c1892cea0a433f5dbd7f43468a2a10243ca",
    "flashinfer/autotune_cache.py": "1ab2ed7208ed1bf1007b1f74de29fc818c93a5ab091beea0833474177fbb29ad",
    "flashinfer/autotuner/_graph_timing.py": "5f3be31e17501ba11ee95327176ac4a4b95eeaee7be92c2bc7795bcbd30cefe3",
    "flashinfer/gemm/gemm_base.py": "0bba9240029a5e09f331e7a2267f7067e126d57a7a341ca535f7c9557599adca",
}
PROFILE_MS = (1, 2, 4, 8, 16, 32, 64, 128, 144)
GRAPH_IDS = ("winner", "default")
CELLS = ("auto_synthetic", "graph_synthetic", "auto_captured", "graph_captured")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def tactic_json(tactic):
    if tactic == -1:
        return -1
    require(
        isinstance(tactic, tuple)
        and len(tactic) == 8
        and all(type(word) is int and 0 <= word < 2**64 for word in tactic),
        "malformed descriptor tactic",
    )
    return list(tactic)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("admission", "timing"), required=True)
    parser.add_argument("--block", type=int, choices=range(8), required=True)
    parser.add_argument("--cell", choices=CELLS, required=True)
    parser.add_argument(
        "--capture-order", choices=("winner-first", "default-first"), required=True
    )
    for name in (
        "plan",
        "schedule",
        "fixture",
        "fixture-manifest",
        "cache",
        "output",
        "trace",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    return parser.parse_args(argv)


def select_schedule(document, block, cell, capture_order):
    matches = [
        row
        for row in document["entries"]
        if row["block"] == block and row["cell"] == cell
    ]
    require(len(matches) == 1, "schedule must identify one worker")
    selected = matches[0]
    require(
        selected["capture_order"] == capture_order,
        "capture order differs from frozen schedule",
    )
    schedules = document["round_schedules"]
    rounds = schedules[str(block)] if isinstance(schedules, dict) else schedules[block]
    orders = rounds["orders"]
    require(
        len(orders) == 24
        and all(sorted(order) == sorted(GRAPH_IDS) for order in orders),
        "invalid paired-round schedule",
    )
    require(
        sum(order[0] == "winner" for order in orders) == 12,
        "paired-round order is unbalanced",
    )
    return selected, orders


def make_initializer(original, q, values, records, tensor_hash=None):
    """Consume the original draw; match allocation/copy work at exact M144."""

    def initialize(shapes, dtype, device):
        generated = original(shapes, dtype, device)
        targeted = tuple(shapes) == tuple(q.shape)
        row = {
            "shape": list(shapes),
            "dtype": str(dtype),
            "device": str(device),
            "targeted": targeted,
            "cloned": False,
            "copied": False,
        }
        if tensor_hash is not None:
            row["generated_sha256"] = tensor_hash(generated)
        if targeted:
            require(
                dtype == q.dtype and device == q.device and generated.is_contiguous(),
                "captured activation contract changed",
            )
            shadow = generated.clone()
            generated.copy_(q if values == "captured" else shadow)
            row.update(cloned=True, copied=True)
        if tensor_hash is not None:
            row["returned_sha256"] = tensor_hash(generated)
        records.append(row)
        return generated

    return initialize


class Experiment:
    def __init__(self, args):
        self.args = args
        objective, values = args.cell.split("_")
        self.result = {
            "schema_version": 1,
            "status": "running",
            "mode": args.mode,
            "trial": args.block,
            "block": args.block,
            "cell": args.cell,
            "objective": objective,
            "values": values,
            "capture_order": args.capture_order,
            "pid": os.getpid(),
            "worker_sha256": sha(__file__),
            "plan_sha256": sha(args.plan),
            "schedule_sha256": sha(args.schedule),
            "startup": {"profiles": [], "initializer_calls": [], "scope_records": []},
            "graphs": [],
            "selections": [],
            "rounds": [],
            "correctness": [],
            "miss_scopes": [],
            "protocol_errors": [],
            "forbidden_profile_attempts": [],
        }
        self.plan = json.loads(args.plan.read_text())
        self.schedule = json.loads(args.schedule.read_text())
        require(
            self.plan["schedule_sha256"] == self.result["schedule_sha256"]
            and self.plan["source_contract"]["runtime_sha256"] == SOURCE_SHA,
            "plan does not bind this schedule/current source",
        )
        self.assignment, self.orders = select_schedule(
            self.schedule, args.block, args.cell, args.capture_order
        )
        self.result["assignment"] = self.assignment
        self.result["round_schedule"] = self.orders
        self.stage, self.allow_profiling, self.profile_attempts = "setup", False, 0
        self.graphs = {}
        self.policy = None

    def tensor_hash(self, tensor):
        return hashlib.sha256(
            tensor.detach().contiguous().view(self.torch.uint8).cpu().numpy().tobytes()
        ).hexdigest()

    @staticmethod
    def metadata(tensor):
        pointer = tensor.data_ptr()
        return {
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "data_ptr": pointer,
            "alignment_mod256": pointer % 256,
            "alignment_mod4096": pointer % 4096,
        }

    def settings(self, config=None):
        policy = self.tuner._effective_measure_policy
        return {
            "warmup": self.tuner.warmup,
            "repeat": self.tuner.repeat,
            "stream_delay_micro_secs": self.tuner.stream_delay_micro_secs,
            "cuda_graph_delay_micro_secs": self.tuner._CUDA_GRAPH_DELAY_MICRO_SECS,
            "use_global_timer": self.tuner._use_global_timer,
            "measurement_policy": dataclasses.asdict(policy)
            if policy is not None
            else None,
            "effective_policy_timer": None if policy is None else policy.timer,
            "config": None
            if config is None
            else {
                "use_cuda_graph": config.use_cuda_graph,
                "use_cold_l2_cache": config.use_cold_l2_cache,
                "use_cuda_graph_individual_timing": config.use_cuda_graph_individual_timing,
                "cuda_graph_profile_replays": config.cuda_graph_profile_replays,
                "profiling_repeat": config.profiling_repeat,
                "effective_profiling_repeat": self.tuner._get_profiling_repeat(config),
            },
        }

    def rng(self):
        return {
            "cpu_sha256": self.tensor_hash(self.torch.get_rng_state()),
            "cuda_sha256": self.tensor_hash(self.torch.cuda.get_rng_state()),
        }

    def setup(self):
        import flashinfer
        import objective_values_bootstrap as bootstrap
        import torch
        from flashinfer.autotuner import AutoTuner
        from flashinfer.gemm import gemm_base
        from safetensors.torch import load_file

        from sglang.kernels.ops.quantization import fp8_kernel
        from sglang.srt.layers.quantization import fp8_utils
        from sglang.srt.utils import custom_op

        self.torch, self.flashinfer, self.gb = torch, flashinfer, gemm_base
        self.bootstrap, self.native = bootstrap, bootstrap.native_gemm
        require(
            bootstrap.helper_sha256 == HELPER_SHA
            and bootstrap.candidate_manifest_sha256 == MANIFEST_SHA,
            "qualified bootstrap identity differs",
        )
        source = Path(bootstrap.source_root).resolve()
        require(
            all(
                sha(source / name) == expected for name, expected in SOURCE_SHA.items()
            ),
            "current source hash differs",
        )
        require(
            Path(flashinfer.__file__).resolve().is_relative_to(source),
            "mixed current-source import",
        )
        require(
            torch.__version__ == "2.14.1+cu130"
            and torch.version.cuda == "13.0"
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability() == (9, 0),
            "requires pinned H100 stack",
        )
        require(
            not os.environ.get("LD_PRELOAD")
            and os.environ.get("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE") == "0",
            "timing cannot use preload or bundled configs",
        )
        require(
            flashinfer.bmm_fp8 is gemm_base.bmm_fp8
            and hasattr(flashinfer.bmm_fp8, "__wrapped__")
            and fp8_utils._raw_flashinfer_bmm_fp8 is flashinfer.bmm_fp8
            and fp8_utils.static_quant_fp8 is fp8_kernel.static_quant_fp8,
            "original decorated public facade differs",
        )
        factory = gemm_base.get_gemm_module().cublas_fp8_gemm_runner
        closure = dict(zip(factory.__code__.co_freevars, factory.__closure__))
        require(
            closure["module"].cell_contents is self.native,
            "native provider was replaced or proxied",
        )
        require(
            sha(self.args.fixture) == FIXTURE_SHA
            and sha(self.args.fixture_manifest) == FIXTURE_MANIFEST_SHA,
            "fixture identity differs",
        )
        manifest = json.loads(self.args.fixture_manifest.read_text())
        source_root = Path(__file__).resolve().parents[3]
        require(
            all(
                Path(module.__file__).resolve().is_relative_to(source_root / "python")
                for module in (fp8_utils, fp8_kernel, custom_op)
            ),
            "mixed SGLang facade/quantizer import",
        )
        require(
            fp8_utils.flashinfer_per_tensor_fp8_supported(),
            "actual static FP8 support guard failed",
        )
        for name, expected in manifest["source"]["sglang_source_sha256"].items():
            require(
                sha(source_root / name) == expected,
                "actual SGLang source differs: " + name,
            )
        cpu = load_file(str(self.args.fixture), device="cpu")
        require(set(cpu) == set(manifest["tensors"]), "fixture tensor names differ")
        for name, tensor in cpu.items():
            observed = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": self.tensor_hash(tensor),
                "nbytes": tensor.numel() * tensor.element_size(),
            }
            require(
                observed == manifest["tensors"][name], "fixture tensor differs: " + name
            )
        self.input = cpu["input_bf16"].cuda()
        self.saved_input = self.input.clone()
        self.weight = cpu["weight_fp8_rowmajor"].cuda()
        self.scale_a, self.scale_b = (
            cpu["input_scale"].cuda(),
            cpu["weight_scale"].cuda(),
        )
        self.apply = fp8_utils.apply_fp8_linear_bmm_flashinfer
        torch.backends.cuda.matmul.allow_tf32 = False
        self.q, scale = fp8_kernel.static_quant_fp8(
            self.input, self.scale_a, repeat_scale=False
        )
        require(
            self.tensor_hash(self.q)
            == manifest["quantization"]["independent_input_fp8_sha256"]
            and scale.data_ptr() == self.scale_a.data_ptr(),
            "actual static quantizer differs",
        )
        self.reference = (
            self.q.float() @ self.weight.float().t() * self.scale_a * self.scale_b
        )
        self.read_hashes = self.operand_hashes()
        self.tuner = AutoTuner.get()
        require(
            not self.tuner._winner_cache()
            and not self.tuner._file_configs
            and not self.tuner._winner_partitions
            and not self.tuner._active_managed_store
            and not self.tuner.is_tuning_mode
            and not self.tuner._in_v2_context,
            "fresh process has preexisting tuner state",
        )
        require(not self.args.cache.exists(), "fresh worker cache already exists")
        self.original_profile = self.tuner._profile_single_kernel
        self.original_choose, self.original_search = (
            self.tuner.choose_one,
            self.tuner.search_cache,
        )
        self.original_v2 = flashinfer.autotune_v2
        if self.result["objective"] == "graph":
            self.policy = flashinfer.MeasurementPolicy(execution_mode="cuda_graph")
        self.original_config = gemm_base._FP8_GEMM_SM100_TUNING_CONFIG
        require(
            self.original_config.tensor_initializers == ()
            and self.original_config.inputs_pre_hook is None,
            "original FP8 initializer contract changed",
        )
        from flashinfer.autotuner.initializers import autotuner_initializer_rand_scaled

        initialize = make_initializer(
            autotuner_initializer_rand_scaled,
            self.q.unsqueeze(0),
            self.result["values"],
            self.result["startup"]["initializer_calls"],
            self.tensor_hash if self.args.mode == "admission" else None,
        )
        self.config = dataclasses.replace(
            self.original_config, tensor_initializers=((0, initialize),)
        )
        require(
            all(
                getattr(self.config, field.name)
                is getattr(self.original_config, field.name)
                for field in dataclasses.fields(self.config)
                if field.name != "tensor_initializers"
            ),
            "initializer replacement changed another config field",
        )
        self.caller_stream = torch.cuda.current_stream()
        self.model_runner = SimpleNamespace(
            device="cuda", forward_stream=torch.cuda.Stream(), is_draft_worker=False
        )
        props = torch.cuda.get_device_properties(0)
        self.result["source"] = {
            "commit": PIN,
            "critical_sha256": SOURCE_SHA,
            "sglang_source_sha256": manifest["source"]["sglang_source_sha256"],
            "bootstrap_helper_sha256": bootstrap.helper_sha256,
            "candidate_manifest_sha256": bootstrap.candidate_manifest_sha256,
            "facade_source_sha256": hashlib.sha256(
                inspect.getsource(self.apply).encode()
            ).hexdigest(),
            "native_identity_passthrough": True,
        }
        self.result["fixture"] = {
            "manifest_sha256": FIXTURE_MANIFEST_SHA,
            "tensor_file_sha256": FIXTURE_SHA,
            "shape": manifest["shape"],
            "quantized_input_sha256": self.tensor_hash(self.q),
            "tensors": manifest["tensors"],
        }
        self.result["environment"] = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": props.name,
            "gpu_uuid": str(props.uuid),
            "inference_mode": torch.is_inference_mode_enabled(),
            "grad_enabled": torch.is_grad_enabled(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "flashinfer_version": flashinfer.__version__,
        }
        self.result["actual_layer"] = {
            "function": "apply_fp8_linear_bmm_flashinfer",
            "custom_op": "flashinfer_bmm_fp8",
            "quantizer": "static_quant_fp8",
            "backend": "cublas",
            "inference_mode": torch.is_inference_mode_enabled(),
            "grad_enabled": torch.is_grad_enabled(),
            "dispatch_modes": torch._C._len_torch_dispatch_stack(),
            "function_modes": torch._C._is_torch_function_mode_enabled(),
            "torch_profiler_enabled": torch._C._autograd._profiler_enabled(),
            "compiling": torch.compiler.is_compiling(),
            "kernel_api_log_level": custom_op._KERNEL_API_LOG_LEVEL,
            "custom_op_direct_wrapper": hasattr(
                fp8_utils.flashinfer_bmm_fp8, "__wrapped__"
            ),
        }
        layer = self.result["actual_layer"]
        require(
            layer["inference_mode"]
            and not any(
                layer[key]
                for key in (
                    "grad_enabled",
                    "dispatch_modes",
                    "function_modes",
                    "torch_profiler_enabled",
                    "compiling",
                    "kernel_api_log_level",
                )
            )
            and layer["custom_op_direct_wrapper"],
            "normal inference custom-op fast path differs",
        )
        require(
            str(props.uuid).removeprefix("GPU-").lower()
            == self.plan["environment"]["uuid"].removeprefix("GPU-").lower(),
            "physical GPU differs from frozen plan",
        )
        # Prime the same real Torch delay helper in every process before startup.
        torch.cuda._sleep(10_000_000)
        torch.cuda.synchronize()
        self.result["delay_helper"] = {
            "wrapper_source_sha256": hashlib.sha256(
                inspect.getsource(torch.cuda._sleep).encode()
            ).hexdigest(),
            "prime_cycles": 10_000_000,
            "primed_before_startup": True,
        }
        torch.manual_seed(19)
        torch.cuda.manual_seed_all(19)
        self.result["startup"]["rng_before"] = self.rng()
        self.result["startup"]["initial_settings"] = self.settings()

    def operand_hashes(self):
        return {
            name: self.tensor_hash(value)
            for name, value in (
                ("input_bf16", self.input),
                ("weight_fp8_rowmajor", self.weight),
                ("input_scale", self.scale_a),
                ("weight_scale", self.scale_b),
            )
        }

    def operation(self):
        return self.apply(self.input, self.weight.t(), self.scale_b, self.scale_a)

    def scope_state(self):
        policy = self.tuner._effective_measure_policy
        return {
            "active_v2": self.tuner._in_v2_context,
            "is_tuning_mode": self.tuner.is_tuning_mode,
            "tuning_contexts": self.tuner._active_tuning_contexts,
            "measurement_policy": dataclasses.asdict(policy)
            if policy is not None
            else None,
            "winner_identity": repr(self.tuner._winner_cache_identity()),
            "managed_store": self.tuner._active_managed_store is not None,
            "on_caller_stream": self.torch.cuda.current_stream() == self.caller_stream,
            "on_forward_stream": self.torch.cuda.current_stream()
            == self.model_runner.forward_stream,
        }

    @contextlib.contextmanager
    def policy_scope(self, mode, label):
        options = {"mode": mode, "persistent_cache": False}
        if self.policy is not None:
            options["measurement_policy"] = self.policy
        with self.original_v2(**options):
            self.result["startup"]["scope_records"].append(
                {"stage": label, "event": "enter", **self.scope_state()}
            )
            try:
                yield
            finally:
                self.result["startup"]["scope_records"].append(
                    {"stage": label, "event": "before_exit", **self.scope_state()}
                )
        self.result["startup"]["scope_records"].append(
            {"stage": label, "event": "after_exit", **self.scope_state()}
        )

    @contextlib.contextmanager
    def startup_policy(self, **options):
        require(
            options == {"mode": "tune", "persistent_cache": False},
            "actual startup wrapper v2 boundary changed",
        )
        with self.policy_scope("tune", "startup"):
            yield

    def profile(self, runner, inputs, tactic, config, *more, **kwargs):
        self.profile_attempts += 1
        if not self.allow_profiling:
            self.result["forbidden_profile_attempts"].append(self.stage)
            raise RuntimeError("profiling outside actual startup is forbidden")
        row = {
            "index": len(self.result["startup"]["profiles"]),
            "m": inputs[0].shape[-2],
            "tactic": tactic_json(tactic),
            "metadata": [self.metadata(t) for t in inputs],
            "settings": self.settings(config),
            "elapsed_ms": None,
            "error": None,
            "fingerprinted": self.args.mode == "admission",
        }
        self.result["startup"]["profiles"].append(row)
        shapes = tuple(tuple(t.shape) for t in inputs)
        key = self.tuner._get_cache_key(
            "fp8_gemm", runner, shapes, config, runner.get_cache_key_extras(inputs)
        )
        row["key"] = key.file_key
        if self.args.mode == "admission":
            row["fingerprints"] = [self.tensor_hash(t) for t in inputs[:4]]
        try:
            settings = row["settings"]
            expected_policy = (
                dataclasses.asdict(self.policy) if self.policy is not None else None
            )
            require(
                type(runner).__name__ == "CublasFp8GemmRunner"
                and settings["warmup"] == 3
                and settings["repeat"] == 10
                and settings["stream_delay_micro_secs"] == 5000
                and settings["cuda_graph_delay_micro_secs"] == 100
                and not settings["use_global_timer"]
                and settings["measurement_policy"] == expected_policy
                and config.use_cuda_graph == (self.result["objective"] == "graph")
                and not config.use_cold_l2_cache
                and not config.use_cuda_graph_individual_timing
                and config.cuda_graph_profile_replays == 1
                and config.profiling_repeat is None
                and self.tuner._in_v2_context
                and self.torch.cuda.current_stream()
                == self.model_runner.forward_stream,
                "actual profiling contract differs",
            )
            elapsed = self.original_profile(
                runner, inputs, tactic, config, *more, **kwargs
            )
            row["raw_elapsed_repr"] = repr(elapsed)
            if isinstance(elapsed, (float, int)) and math.isfinite(elapsed):
                row["elapsed_ms"] = float(elapsed)
            require(
                isinstance(elapsed, (float, int))
                and math.isfinite(elapsed)
                and elapsed > 0,
                "nonpositive or nonfinite profiling result",
            )
            return elapsed
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            self.result["protocol_errors"].append(
                {"profile_index": row["index"], "error": row["error"]}
            )
            raise

    def startup(self):
        from sglang.srt.model_executor.runner import flashinfer_autotune as startup
        from sglang.srt.runtime_context import get_context, get_parallel

        require(
            Path(startup.__file__)
            .resolve()
            .is_relative_to(Path(__file__).resolve().parents[3] / "python"),
            "actual startup wrapper imported from another checkout",
        )
        self.result["source"]["startup_wrapper_sha256"] = sha(startup.__file__)
        self.args.cache.parent.mkdir(parents=True, exist_ok=True)
        self.stage, self.allow_profiling = "startup", True
        self.result["startup"]["started_ns"] = time.perf_counter_ns()
        try:
            with (
                get_context().override_server_args(
                    moe_runner_backend="triton", flashinfer_autotune_skip_ops=[]
                ),
                get_parallel().override(tp_group=SimpleNamespace(world_size=1)),
                patch.object(
                    startup, "flashinfer_autotune_cache_path", lambda _: self.args.cache
                ),
                patch.object(self.flashinfer, "autotune_v2", self.startup_policy),
                startup.flashinfer_autotune_context(
                    self.model_runner, run_lm_head=False
                ),
            ):
                self.operation()
            self.torch.cuda.synchronize()
        finally:
            self.allow_profiling = False
            self.result["startup"]["ended_ns"] = time.perf_counter_ns()
        self.result["startup"].update(
            profile_attempts=self.profile_attempts,
            rng_after=self.rng(),
            cache_sha256=sha(self.args.cache),
            after_settings=self.settings(),
        )
        require(
            not self.result["protocol_errors"], "startup retained a failed measurement"
        )
        calls = self.result["startup"]["initializer_calls"]
        require(
            [row["shape"][1] for row in calls] == list(PROFILE_MS)
            and sum(row["cloned"] and row["copied"] for row in calls) == 1
            and calls[-1]["targeted"],
            "initializer changed nine-profile domain or matched copy count",
        )
        profiles = self.result["startup"]["profiles"]
        groups = [[row for row in profiles if row["m"] == m] for m in PROFILE_MS]
        require(
            all(
                group
                and group[0]["tactic"] == -1
                and sum(row["tactic"] == -1 for row in group) == 1
                for group in groups
            ),
            "default participation or profile domain changed",
        )
        require(
            [row["m"] for group in groups for row in group]
            == [row["m"] for row in profiles],
            "profiling endpoint order changed",
        )
        require(
            len(profiles) == 81 and all(len(group) == 9 for group in groups),
            "qualified nine-by-nine candidate domain changed",
        )
        self.result["startup"]["candidate_order"] = {
            str(m): [row["tactic"] for row in group]
            for m, group in zip(PROFILE_MS, groups, strict=True)
        }
        data = json.loads(self.args.cache.read_text())
        legacy = {k: v for k, v in data.items() if k.startswith("(")}
        self.result["startup"]["incidental_legacy_entries"] = len(legacy)
        require(
            not self.tuner._file_configs and not self.tuner._active_managed_store,
            "startup loaded files or attached a store",
        )
        require(
            self.result["objective"] != "graph" or not legacy,
            "graph partition leaked into legacy save",
        )
        self.cache_sha = sha(self.args.cache)
        self.memory_sha = self.cache_state_sha()
        self.memory_fields = dict(self.cache_state_field_sha256)
        self.result["startup"]["cache_state_sha256"] = self.memory_sha
        self.result["startup"]["cache_state_field_sha256"] = self.memory_fields
        self.startup_attempts = self.profile_attempts
        self.assert_bare()

    def assert_bare(self):
        require(
            not self.tuner.is_tuning_mode
            and not self.tuner._in_v2_context
            and self.tuner._effective_measure_policy is None
            and not self.tuner._active_managed_store
            and self.tuner._active_tuning_contexts == 0
            and self.torch.cuda.current_stream() == self.caller_stream,
            "raw replay leaked tuning/policy/store/stream state",
        )

    def cache_state_sha(self):
        # Canonicalize every nested container; deepcopy/restore may change set
        # or dictionary iteration order while preserving exactly equal values.
        # Leaf representations/identities are meaningful only in this process.
        names = (
            "profiling_cache",
            "_winner_partitions",
            "_profiling_cache_policies",
            "_file_configs",
            "_file_config_policies",
            "_invalidated_file_keys",
            "_excluded_file_keys",
            "_logged_file_hits",
            "_logged_cache_miss_oor",
            "_logged_invalid_tactic",
            "_namespaced_records",
            "_protocol_record_partitions",
            "_managed_decoded",
            "_dirty",
            "_dirty_seq",
        )

        def canonical(value):
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                return [
                    "dataclass",
                    type(value).__module__ + "." + type(value).__qualname__,
                    [
                        [field.name, canonical(getattr(value, field.name))]
                        for field in dataclasses.fields(value)
                    ],
                ]
            if isinstance(value, dict):
                rows = [
                    [canonical(key), canonical(item)] for key, item in value.items()
                ]
                return [
                    "dict",
                    sorted(rows, key=lambda row: json.dumps(row[0], sort_keys=True)),
                ]
            if isinstance(value, (set, frozenset)):
                return [
                    type(value).__name__,
                    sorted(
                        (canonical(item) for item in value),
                        key=lambda item: json.dumps(item, sort_keys=True),
                    ),
                ]
            if isinstance(value, (tuple, list)):
                return [type(value).__name__, [canonical(item) for item in value]]
            return [
                "atom",
                type(value).__module__ + "." + type(value).__qualname__,
                repr(value),
            ]

        fields = {name: canonical(getattr(self.tuner, name)) for name in names}
        self.cache_state_field_sha256 = {
            name: hashlib.sha256(
                json.dumps(value, separators=(",", ":")).encode()
            ).hexdigest()
            for name, value in fields.items()
        }
        payload = canonical({name: getattr(self.tuner, name) for name in names})
        return hashlib.sha256(
            json.dumps(payload, separators=(",", ":")).encode()
        ).hexdigest()

    def choose_recorder(self, expected, tier, label, saved):
        def choose(op, runners, config, inputs, **kwargs):
            require(
                op == "fp8_gemm"
                and len(runners) == 1
                and type(runners[0]).__name__ == "CublasFp8GemmRunner"
                and not kwargs,
                "actual public runner pool differs",
            )
            result = self.original_choose(op, runners, config, inputs, **kwargs)
            hit, index, cached, _ = self.original_search(
                op,
                runners,
                tuple(tuple(t.shape) for t in inputs),
                config,
                inputs=inputs,
            )
            require(
                bool(hit) == (tier == "memory")
                and result[0] is runners[index]
                and (expected is None or result[1] == expected)
                and (not hit or cached == result[1]),
                "actual public lookup did not consume expected hit/miss winner",
            )
            a, b, sa, sb, out, workspace = inputs
            require(
                list(a.shape) == [1, 144, 1024]
                and a.is_contiguous()
                and a.dtype == self.torch.float8_e4m3fn
                and list(b.shape) == [1, 1024, 4096]
                and list(b.stride())[-2:] == [1, 1024]
                and b.data_ptr() == self.weight.data_ptr()
                and sa.data_ptr() == self.scale_a.data_ptr()
                and sb.data_ptr() == self.scale_b.data_ptr()
                and list(out.shape) == [1, 144, 4096]
                and out.dtype == self.torch.bfloat16
                and workspace.dtype == self.torch.uint8,
                "actual full-layer graph operands differ",
            )
            key = self.tuner._get_cache_key(
                op,
                runners[0],
                tuple(tuple(t.shape) for t in inputs),
                config,
                runners[0].get_cache_key_extras(inputs),
            )
            row = {
                "stage": label,
                "tactic": tactic_json(result[1]),
                "cache_hit": bool(hit),
                "lookup_tier": tier,
                "capturing": self.torch.cuda.is_current_stream_capturing(),
                "key": key.file_key,
                **self.scope_state(),
            }
            self.result["selections"].append(row)
            saved.update(
                inputs=tuple(inputs),
                runner=runners[0],
                tactic=result[1],
                key=key,
                config=config,
                selection=row,
            )
            return result

        return choose

    @contextlib.contextmanager
    def miss_scope(self):
        names = (
            "profiling_cache",
            "_winner_partitions",
            "_profiling_cache_policies",
            "_file_configs",
            "_file_config_policies",
            "_invalidated_file_keys",
            "_excluded_file_keys",
            "_logged_file_hits",
            "_logged_cache_miss_oor",
            "_logged_invalid_tactic",
            "_dirty",
            "_dirty_seq",
        )
        snapshots = {name: copy.deepcopy(getattr(self.tuner, name)) for name in names}
        row = {
            "winner_entries_before": len(self.tuner._winner_cache()),
            "file_entries_before": len(self.tuner._file_configs),
            "cache_sha256_before": sha(self.args.cache),
            "restored": False,
        }
        self.result["miss_scopes"].append(row)
        try:
            self.tuner._winner_cache().clear()
            self.tuner._file_configs.clear()
            row.update(
                winner_entries_after_removal=len(self.tuner._winner_cache()),
                file_entries_after_removal=len(self.tuner._file_configs),
            )
            yield
        finally:
            for name, old in snapshots.items():
                current = getattr(self.tuner, name)
                if isinstance(current, (dict, set)):
                    current.clear()
                    current.update(old)
                else:
                    setattr(self.tuner, name, old)
            row.update(
                cache_sha256_after=sha(self.args.cache),
                restored=all(
                    getattr(self.tuner, name) == old for name, old in snapshots.items()
                ),
            )
            require(
                row["restored"]
                and row["cache_sha256_before"] == row["cache_sha256_after"],
                "miss scope did not restore tuner state",
            )

    def check(self, output, label, graph_id=None, zero=False):
        if graph_id is not None:
            saved = self.graphs[graph_id]
            require(
                output.data_ptr()
                == saved["output"].data_ptr()
                == saved["inputs"][4].data_ptr(),
                "numerical check selected another retained graph allocation",
            )
        self.torch.cuda.synchronize()
        actual = output.float()
        require(
            bool(self.torch.isfinite(actual).all()), "nonfinite actual layer output"
        )
        target = self.torch.zeros_like(self.reference) if zero else self.reference
        difference = (actual - target).abs()
        maximum = float(difference.max())
        require(
            bool(self.torch.count_nonzero(actual) == 0)
            if zero
            else bool(self.torch.allclose(actual, target, rtol=0.02, atol=0.125)),
            "actual FP8 layer numerical check failed: " + label,
        )
        row = {
            "check": label,
            "graph_id": graph_id,
            "output_data_ptr": output.data_ptr(),
            "passed": True,
            "max_abs_error": maximum,
            "exact_zero_required": zero,
        }
        self.result["correctness"].append(row)
        return label

    def graph_checks(self, graph_id, prefix):
        saved = self.graphs[graph_id]
        graph, out = saved["graph"], saved["output"]
        out.fill_(float("nan"))
        graph.replay()
        self.check(out, prefix + "_poisoned", graph_id)
        self.input.zero_()
        out.fill_(float("nan"))
        graph.replay()
        self.check(out, prefix + "_changed_input", graph_id, zero=True)
        self.input.copy_(self.saved_input)
        out.fill_(float("nan"))
        graph.replay()
        self.check(out, prefix + "_restored_input", graph_id)

    def capture(self, graph_id, expected, tier):
        torch, saved = self.torch, {}
        self.stage = graph_id + "_before"
        with patch.object(
            self.tuner,
            "choose_one",
            self.choose_recorder(expected, tier, self.stage, saved),
        ):
            eager = self.operation()
        self.check(eager, self.stage)
        self.stage = graph_id + "_capture"
        graph = torch.cuda.CUDAGraph()
        started = time.perf_counter_ns()
        with patch.object(
            self.tuner,
            "choose_one",
            self.choose_recorder(expected, tier, self.stage, saved),
        ):
            with torch.cuda.graph(graph):
                output = self.operation()
        ended = time.perf_counter_ns()
        require(
            len(
                [row for row in self.result["selections"] if row["stage"] == self.stage]
            )
            == 1,
            "capture did not make one real public selection",
        )
        saved.update(id=graph_id, graph=graph, output=output)
        self.graphs[graph_id] = saved
        self.graph_checks(graph_id, graph_id)
        inputs = saved["inputs"]
        hashes = [self.tensor_hash(t) for t in inputs[:4]]
        require(
            hashes
            == [
                self.tensor_hash(self.q.unsqueeze(0)),
                self.tensor_hash(self.weight.t().unsqueeze(0)),
                self.tensor_hash(self.scale_a),
                self.tensor_hash(self.scale_b),
            ],
            "captured public operand bytes differ",
        )
        row = {
            "id": graph_id,
            "tactic": tactic_json(saved["tactic"]),
            "lookup_tier": tier,
            "selection": saved["selection"],
            "graph_object_id": id(graph),
            "inputs": [self.metadata(t) for t in inputs],
            "linear_input": self.metadata(self.input),
            "output": self.metadata(output),
            "read_operand_hashes": hashes,
            "capture_started_ns": started,
            "capture_ended_ns": ended,
        }
        self.result["graphs"].append(row)
        return saved

    def capture_controls(self):
        self.result["pre_capture_cache_state_sha256"] = self.cache_state_sha()
        self.result["pre_capture_cache_state_field_sha256"] = dict(
            self.cache_state_field_sha256
        )
        require(
            self.result["pre_capture_cache_state_sha256"] == self.memory_sha,
            "winner state changed before capture",
        )
        with self.policy_scope("replay", "natural_lookup"):
            saved = {}
            with patch.object(
                self.tuner,
                "choose_one",
                self.choose_recorder(None, "memory", "natural_lookup", saved),
            ):
                out = self.operation()
            self.check(out, "natural_lookup")
            self.winner = saved["tactic"]
            runtime = [
                row for row in self.result["startup"]["profiles"] if row["m"] == 144
            ]
            require(
                tactic_json(self.winner)
                == min(runtime, key=lambda row: row["elapsed_ms"])["tactic"],
                "runtime winner differs from first finite profiling minimum",
            )
            self.result["natural"] = {
                "tactic": tactic_json(self.winner),
                "key": saved["key"].file_key,
                "cache_hit": True,
                "lookup_tier": "memory",
            }
        order = (
            GRAPH_IDS
            if self.args.capture_order == "winner-first"
            else tuple(reversed(GRAPH_IDS))
        )
        for graph_id in order:
            with self.policy_scope("replay", graph_id + "_capture"):
                with (
                    self.miss_scope()
                    if graph_id == "default"
                    else contextlib.nullcontext()
                ):
                    self.capture(
                        graph_id,
                        -1 if graph_id == "default" else self.winner,
                        "miss" if graph_id == "default" else "memory",
                    )
        require(
            len({saved["output"].data_ptr() for saved in self.graphs.values()}) == 2,
            "independent graphs share an output",
        )
        self.assert_bare()

    def warm(self, graph_id, milliseconds):
        started, calls = time.perf_counter_ns(), 0
        while True:
            for _ in range(5120):
                self.graphs[graph_id]["graph"].replay()
            self.torch.cuda.synchronize()
            calls += 5120
            elapsed = (time.perf_counter_ns() - started) / 1e6
            if elapsed >= milliseconds:
                return {"actual_ms": elapsed, "calls": calls}

    def sample(self, graph_id, index):
        self.assert_bare()
        torch, saved = self.torch, self.graphs[graph_id]
        graph, out = saved["graph"], saved["output"]
        row = {
            "graph_id": graph_id,
            "graph_object_id": id(graph),
            "output_data_ptr": out.data_ptr(),
            "replays": 16,
            "prequeue_cycles": 10_000_000,
            "rewarm": self.warm(graph_id, 20),
            "profile_attempts_before": self.profile_attempts,
            "start_query_error": None,
        }
        self.result["pending_sample"] = row
        out.fill_(float("nan"))
        first, last = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        first.record()
        last.record()
        last.synchronize()
        row["wall_started_ns"] = time.perf_counter_ns()
        torch.cuda._sleep(10_000_000)
        row["enqueue_started_ns"] = time.perf_counter_ns()
        first.record()
        for _ in range(16):
            graph.replay()
        last.record()
        try:
            complete = first.query()
        except Exception as error:
            row["start_query_error"] = f"{type(error).__name__}: {error}"
            row["query_error_observed_ns"] = time.perf_counter_ns()
            raise
        row["enqueue_ended_ns"] = time.perf_counter_ns()
        last.synchronize()
        row["ended_ns"] = time.perf_counter_ns()
        elapsed = first.elapsed_time(last)
        require(
            type(complete) is bool and math.isfinite(elapsed) and elapsed > 0,
            "invalid event duration or start-query outcome",
        )
        row.update(
            gpu_event_batch_ms=elapsed,
            gpu_event_per_replay_ms=elapsed / 16,
            start_event_complete_after_enqueue=complete,
            queue_evidence="start_complete_inconclusive"
            if complete
            else "start_pending_after_enqueue",
            enqueue_cpu_ms=(row["enqueue_ended_ns"] - row["enqueue_started_ns"]) / 1e6,
            enqueue_span_includes_start_query=True,
            wall_including_prequeue_ms=(row["ended_ns"] - row["wall_started_ns"]) / 1e6,
            wall_diagnostic_only=True,
            profile_attempts_after=self.profile_attempts,
        )
        row["correctness_check"] = self.check(
            out, f"round_{index:02d}_{graph_id}", graph_id
        )
        self.result["pending_sample"] = None
        return row

    def time_graphs(self):
        self.result["initial_warmups"] = [
            {"graph_id": name, **self.warm(name, 500)} for name in GRAPH_IDS
        ]
        for index, order in enumerate(self.orders):
            group = {"index": index, "order": order, "measurements": []}
            self.result["rounds"].append(group)
            for graph_id in order:
                group["measurements"].append(self.sample(graph_id, index))
        ratios = []
        for group in self.result["rounds"]:
            rows = {row["graph_id"]: row for row in group["measurements"]}
            ratios.append(
                rows["winner"]["gpu_event_per_replay_ms"]
                / rows["default"]["gpu_event_per_replay_ms"]
            )
        self.result["selected_over_default_graph_event_median"] = statistics.median(
            ratios
        )

    def accept_actual(self, saved, label):
        a, b, sa, sb, out, workspace = saved["inputs"]
        before = [self.metadata(t) for t in saved["inputs"]]
        self.input.copy_(self.saved_input)
        saved["graph"].replay()
        self.torch.cuda.synchronize()
        out.fill_(float("nan"))
        self.torch.cuda.synchronize()
        row = {
            "graph_id": saved["id"],
            "graph_object_id": id(saved["graph"]),
            "output_data_ptr": saved["output"].data_ptr(),
            "tactic": tactic_json(saved["tactic"]),
            "captured_inputs": before,
            "started_ns": time.perf_counter_ns(),
            "used_descriptor": None,
        }
        self.result.setdefault("native_acceptance_attempts", []).append(row)
        if saved["tactic"] == -1:
            self.native.bmm_fp8(a, b, out, sa, sb, workspace)
            row["route"] = "native_default"
            row["descriptor_acceptance"] = "not_applicable_default_route"
        else:
            used = int(
                self.native.bmm_fp8_run_with_descriptor(
                    a,
                    b,
                    out,
                    sa,
                    sb,
                    workspace,
                    self.gb._cublaslt_algo_arg(saved["tactic"]),
                )
            )
            row.update(
                route="native_descriptor",
                used_descriptor=used,
                descriptor_acceptance="accepted" if used == 1 else "rejected_fallback",
            )
            require(
                used == 1,
                "actual selected descriptor fell back at the exact captured problem",
            )
        self.torch.cuda.synchronize()
        row["raw_numerical_check"] = self.check(
            out.squeeze(0), label + "_raw", saved["id"]
        )
        require(
            before == [self.metadata(t) for t in saved["inputs"]],
            "exact captured native operands changed",
        )
        saved["output"].fill_(float("nan"))
        saved["graph"].replay()
        row["original_graph_check"] = self.check(
            saved["output"], label + "_original_graph", saved["id"]
        )
        row.update(
            pointers_unchanged=True,
            original_graph_replayed=True,
            ended_ns=time.perf_counter_ns(),
        )
        return row

    def admission_explicit(self):
        accepted = None
        saved = self.graphs["winner"]
        self.result["known_explicit_control"] = {"attempts": [], "accepted": False}
        control = self.result["known_explicit_control"]
        a, b, sa, sb, out, workspace = saved["inputs"]
        for row in self.result["startup"]["profiles"]:
            if row["m"] != 144 or row["tactic"] == -1:
                continue
            tactic = tuple(row["tactic"])
            out.fill_(float("nan"))
            self.torch.cuda.synchronize()
            used = int(
                self.native.bmm_fp8_run_with_descriptor(
                    a, b, out, sa, sb, workspace, self.gb._cublaslt_algo_arg(tactic)
                )
            )
            self.torch.cuda.synchronize()
            control["attempts"].append(
                {"tactic": row["tactic"], "used_descriptor": used}
            )
            if used == 1:
                self.check(out.squeeze(0), "admission_known_explicit_raw")
                accepted = tactic
                break
        require(accepted is not None, "admission found no accepted explicit descriptor")
        with self.policy_scope("replay", "admission_explicit"):
            with self.miss_scope():
                self.tuner._cache_profile_result(
                    saved["key"], None, saved["config"], accepted
                )
                explicit = self.capture("admission_explicit", accepted, "memory")
                control.update(
                    accepted=True,
                    tactic=tactic_json(accepted),
                    actual_acceptance=self.accept_actual(
                        explicit, "admission_explicit"
                    ),
                )
        self.graphs["winner"]["graph"].replay()
        self.check(
            self.graphs["winner"]["output"], "admission_natural_restored", "winner"
        )

    def final_checks(self, prefix):
        self.assert_bare()
        require(
            self.profile_attempts == self.startup_attempts
            and not self.result["forbidden_profile_attempts"]
            and not self.result["protocol_errors"],
            "profiling escaped startup or a measurement failed",
        )
        require(
            self.operand_hashes() == self.read_hashes
            and sha(self.args.cache) == self.cache_sha,
            "serving operands or incidental cache bytes changed",
        )
        memory_sha = self.cache_state_sha()
        self.result.setdefault("cache_state_guard_checks", []).append(
            {
                "stage": prefix,
                "expected_sha256": self.memory_sha,
                "actual_sha256": memory_sha,
                "expected_field_sha256": self.memory_fields,
                "actual_field_sha256": dict(self.cache_state_field_sha256),
                "changed_fields": [
                    name
                    for name in self.memory_fields
                    if self.memory_fields[name] != self.cache_state_field_sha256[name]
                ],
            }
        )
        require(
            memory_sha == self.memory_sha,
            "live selection/cache state changed after startup",
        )
        for name, saved in self.graphs.items():
            require(
                saved["inputs"][4].data_ptr() == saved["output"].data_ptr(),
                "graph retained another output allocation",
            )
            self.check(saved["output"], prefix + "_" + name, name)
        return {
            "ended_ns": time.perf_counter_ns(),
            "profile_attempts": self.profile_attempts,
            "forbidden_profile_attempts": self.result["forbidden_profile_attempts"],
            "cache_sha256": sha(self.args.cache),
            "cache_state_sha256": memory_sha,
            "cache_unchanged": True,
            "operands_unchanged": True,
            **self.scope_state(),
        }

    def trace(self):
        self.assert_bare()
        torch = self.torch
        require(not self.args.trace.exists(), "refusing to overwrite trace")
        self.args.trace.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter_ns()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as profiler:
            for name in GRAPH_IDS:
                with torch.profiler.record_function("retained_" + name):
                    for _ in range(3):
                        self.graphs[name]["graph"].replay()
                    torch.cuda.synchronize()
        profiler.export_chrome_trace(str(self.args.trace))
        ended = time.perf_counter_ns()
        events = json.loads(self.args.trace.read_text())["traceEvents"]
        rows = []
        for name in GRAPH_IDS:
            markers = [
                event
                for event in events
                if event.get("cat") == "user_annotation"
                and event.get("name") == "retained_" + name
            ]
            require(len(markers) == 1, "trace lacks one CPU retained-graph range")
            marker = markers[0]
            inside = [
                event
                for event in events
                if marker["ts"] <= event.get("ts", -1) <= marker["ts"] + marker["dur"]
            ]
            launches = [
                event for event in inside if "cudaGraphLaunch" in event.get("name", "")
            ]
            kernels = sorted(
                [event for event in inside if event.get("cat") == "kernel"],
                key=lambda event: event["ts"],
            )
            require(
                len(launches) == 3 and len(kernels) >= 6,
                "trace lacks retained launches/quantizer and GEMM kernels",
            )
            signatures, sequence = [], []
            for event in kernels:
                fields = (
                    "device",
                    "grid",
                    "block",
                    "registers per thread",
                    "shared memory",
                )
                signature = {
                    "name": event["name"],
                    **{field: event.get("args", {}).get(field) for field in fields},
                }
                if signature not in signatures:
                    signatures.append(signature)
                sequence.append(signatures.index(signature))
            rows.append(
                {
                    "graph_id": name,
                    "graph_object_id": id(self.graphs[name]["graph"]),
                    "output_data_ptr": self.graphs[name]["output"].data_ptr(),
                    "graph_launch_events": len(launches),
                    "kernel_events": len(kernels),
                    "signatures": signatures,
                    "kernel_sequence": sequence,
                }
            )
            self.check(self.graphs[name]["output"], "traced_" + name, name)
        return {
            "path": str(self.args.trace),
            "sha256": sha(self.args.trace),
            "started_ns": started,
            "ended_ns": ended,
            "calls_per_graph": 3,
            "graphs": rows,
            "scope": "Same retained objects after all primary timings; no latency inference.",
        }

    def run(self):
        self.setup()
        try:
            with (
                patch.object(self.gb, "_FP8_GEMM_SM100_TUNING_CONFIG", self.config),
                patch.object(self.tuner, "_profile_single_kernel", self.profile),
            ):
                self.startup()
                self.capture_controls()
                if self.args.mode == "timing":
                    self.time_graphs()
                self.result["post_timing_acceptance"] = self.accept_actual(
                    self.graphs["winner"], "selected_acceptance"
                )
                if self.args.mode == "admission":
                    self.admission_explicit()
                self.result["pre_trace_checks"] = self.final_checks("pre_trace")
                self.result["trace"] = self.trace()
                self.result["final"] = self.final_checks("final")
        finally:
            require(
                self.gb._FP8_GEMM_SM100_TUNING_CONFIG is self.original_config,
                "temporary initializer config leaked",
            )
        self.result.update(
            status="success",
            initializer_config_restored=True,
            profiling_byte_attestation="admission_full_hashes"
            if self.args.mode == "admission"
            else "metadata_only_no_synthetic_byte_attestation",
            synthetic_input_references_retained=False,
            persistence_under_test=False,
            all_samples_retained=True,
            queue_query_is_exclusion_criterion=False,
        )
        return self.result


def main(argv=None):
    args = parse_args(argv)
    require(not args.output.exists(), "refusing to overwrite a worker result")
    experiment = None
    try:
        import torch

        with torch.inference_mode():
            experiment = Experiment(args)
            result = experiment.run()
    except Exception as error:
        traceback.print_exc()
        result = (
            experiment.result
            if experiment is not None
            else {
                "mode": args.mode,
                "block": args.block,
                "cell": args.cell,
                "pid": os.getpid(),
            }
        )
        result.update(status="error", error=f"{type(error).__name__}: {error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
