"""Compare natural default/v1/v2 choices through the actual FP8 linear facade.

The immutable fixture contains one captured BF16 activation and fused QKV weight.
Each process uses real startup tuning, existing JSON persistence, and the normal
static quantizer/custom-op path. This is a manual layer experiment, not a server
benchmark or a full-model quantization accuracy test.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import inspect
import json
import math
import os
import traceback
from pathlib import Path

from flashinfer_autotune_descriptor_selection import SelectionExperiment
from flashinfer_autotune_eager_setup import (
    MAX_ITERATIONS,
    digest,
    measure,
    positive_float,
)

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
FIXTURE_KEYS = {
    "input_bf16",
    "weight_fp8_rowmajor",
    "weight_bf16_rowmajor",
    "input_scale",
    "weight_scale",
    "input_ids",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("default", "v1", "v2"), required=True)
    parser.add_argument("--phase", choices=("cold", "warm"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--trial", type=int, choices=range(6), required=True)
    parser.add_argument("--seed", type=int, choices=(19,), default=19)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup-ms", type=positive_float, default=500)
    parser.add_argument("--min-sample-ms", type=positive_float, default=20)
    args = parser.parse_args()
    if min(args.samples, args.iterations) < 1 or args.iterations > MAX_ITERATIONS:
        parser.error("sample/iteration counts must be positive and within the bound")
    args.oracle = False
    return args


def tensor_sha(torch, tensor):
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def tensor_metadata(torch, tensor):
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": tensor_sha(torch, tensor),
        "nbytes": tensor.numel() * tensor.element_size(),
    }


class GraphBinding:
    """Keep the actual captured output independent of later eager allocations."""

    def __init__(self, owner, graph, output):
        self.owner, self.graph, self.output = owner, graph, output

    def replay(self):
        self.graph.replay()
        self.owner.out = self.output


class LinearLayerExperiment(SelectionExperiment):
    def __init__(self, args):
        import torch
        from safetensors.torch import load_file

        self.fixture_manifest = json.loads(args.fixture_manifest.read_text())
        manifest = self.fixture_manifest
        if manifest.get("schema_version") != 1:
            raise RuntimeError("unsupported fixture schema")
        if (
            manifest["model"]["repo_id"] != MODEL_ID
            or manifest["model"]["revision"] != MODEL_REVISION
        ):
            raise RuntimeError("fixture checkpoint differs from the pinned revision")
        args.m, args.n, args.k = manifest["shape"]
        if args.m != 144 or min(args.n, args.k) < 1 or args.n % 16 or args.k % 16:
            raise RuntimeError("fixture dimensions cannot use the qualified FP8 path")
        if (
            digest(args.fixture) != manifest["tensor_file"]["sha256"]
            or args.fixture.stat().st_size != manifest["tensor_file"]["size_bytes"]
        ):
            raise RuntimeError("fixture file digest or size changed")
        cpu = load_file(str(args.fixture), device="cpu")
        if set(cpu) != FIXTURE_KEYS or set(manifest["tensors"]) != FIXTURE_KEYS:
            raise RuntimeError("fixture tensor keys changed")
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
            if (
                tensor.dtype != dtype
                or list(tensor.shape) != shape
                or not tensor.is_contiguous()
                or tensor_metadata(torch, tensor) != manifest["tensors"][name]
            ):
                raise RuntimeError(f"fixture tensor changed: {name}")
            if not bool(torch.isfinite(tensor.float()).all()):
                raise RuntimeError(f"fixture tensor is nonfinite: {name}")
        if any(
            not bool(cpu[name].gt(0).all()) for name in ("input_scale", "weight_scale")
        ):
            raise RuntimeError("fixture scales must be strictly positive")
        capture = manifest["capture"]
        if (
            capture["weight_order"] != ["q_proj", "k_proj", "v_proj"]
            or capture["bias"] is not None
            or capture["saved_shape"] != [args.m, args.k]
            or capture["captured_shape"] != [1, args.m, args.k]
            or capture["seed"] != args.seed
        ):
            raise RuntimeError("fixture capture geometry or QKV order changed")
        source_root = Path(__file__).resolve().parents[3]
        self.layer_sources = manifest["source"]["sglang_source_sha256"]
        for name, sha in self.layer_sources.items():
            if digest(source_root / name) != sha:
                raise RuntimeError(f"fixture and runtime source differ: {name}")

        # Import-time platform guards must see the actual CUDA device.
        torch.cuda.set_device(0)
        super().__init__(args)
        if not torch.is_inference_mode_enabled() or torch.is_grad_enabled():
            raise RuntimeError("the actual layer must run under inference_mode")
        from sglang.kernels.ops.quantization import fp8_kernel
        from sglang.srt.layers.quantization import fp8_utils
        from sglang.srt.utils import custom_op

        for module in (fp8_utils, fp8_kernel, custom_op):
            if (
                not Path(module.__file__)
                .resolve()
                .is_relative_to(source_root / "python")
            ):
                raise RuntimeError("actual layer imported from another checkout")
        if (
            not fp8_utils.flashinfer_per_tensor_fp8_supported()
            or fp8_utils.static_quant_fp8 is not fp8_kernel.static_quant_fp8
            or fp8_utils._raw_flashinfer_bmm_fp8 is not self.flashinfer.bmm_fp8
        ):
            raise RuntimeError("the actual FP8 layer/quantizer/public API path changed")
        self.apply_layer = fp8_utils.apply_fp8_linear_bmm_flashinfer
        self.a = cpu["input_bf16"].to("cuda")
        self.weight_fp8_rowmajor = cpu["weight_fp8_rowmajor"].to("cuda")
        self.b = self.weight_fp8_rowmajor.t()
        self.weight_bf16_rowmajor = cpu["weight_bf16_rowmajor"].to("cuda")
        self.scale_a, self.scale_b = (
            cpu[name].to("cuda") for name in ("input_scale", "weight_scale")
        )
        self.input_ids = cpu["input_ids"]
        self.out = torch.empty((args.m, args.n), dtype=torch.bfloat16, device="cuda")
        qinput, scale = fp8_kernel.static_quant_fp8(
            self.a, self.scale_a, repeat_scale=False
        )
        independent_cpu = (
            (cpu["input_bf16"].float() * cpu["input_scale"].reciprocal())
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
        )
        independent_gpu = (
            (self.a.float() * self.scale_a.reciprocal())
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
        )
        expected_sha = manifest["quantization"]["independent_input_fp8_sha256"]
        actual_sha = tensor_sha(torch, qinput)
        self.quantization_evidence = {
            "actual_input_fp8_sha256": actual_sha,
            "independent_input_fp8_sha256": expected_sha,
            "independent_cpu_sha256": tensor_sha(torch, independent_cpu),
            "independent_gpu_sha256": tensor_sha(torch, independent_gpu),
            "scale_pointer_unchanged": scale.data_ptr() == self.scale_a.data_ptr(),
        }
        if (
            any(
                self.quantization_evidence[name] != expected_sha
                for name in (
                    "actual_input_fp8_sha256",
                    "independent_cpu_sha256",
                    "independent_gpu_sha256",
                )
            )
            or not self.quantization_evidence["scale_pointer_unchanged"]
        ):
            raise RuntimeError(
                "actual static quantization differs from independent fixture bytes: "
                + json.dumps(self.quantization_evidence, sort_keys=True)
            )
        self.quantized_input_sha = actual_sha
        self.reference = (qinput.float() @ self.b.float()) * self.scale_a * self.scale_b
        self.bf16_reference = self.a.float() @ self.weight_bf16_rowmajor.float().t()
        self.zero_expected = False
        self.observed_public_inputs = []
        self.native_output_poison_checks = []
        self.actual_layer = {
            "function": "apply_fp8_linear_bmm_flashinfer",
            "custom_op": "flashinfer_bmm_fp8",
            "quantizer": "static_quant_fp8",
            "source_sha256": self.layer_sources,
            "function_source_sha256": hashlib.sha256(
                inspect.getsource(self.apply_layer).encode()
            ).hexdigest(),
            "inference_mode": torch.is_inference_mode_enabled(),
            "grad_enabled": torch.is_grad_enabled(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "dispatch_modes": torch._C._len_torch_dispatch_stack(),
            "function_modes": torch._C._is_torch_function_mode_enabled(),
            "torch_profiler_enabled": torch._C._autograd._profiler_enabled(),
            "compiling": torch.compiler.is_compiling(),
            "kernel_api_log_level": custom_op._KERNEL_API_LOG_LEVEL,
            "custom_op_direct_wrapper": hasattr(
                fp8_utils.flashinfer_bmm_fp8, "__wrapped__"
            ),
            "quantization_reference": {
                "byte_match": True,
                "actual_input_fp8_sha256": actual_sha,
                "independent_input_fp8_sha256": expected_sha,
                "scale_sha256": tensor_sha(torch, self.scale_a),
                "cpu_and_gpu_independent_references_match": True,
            },
            "graph_output_binding": "actual captured allocation; output rebinding occurs outside timing",
            "profiling_input_scope": "FlashInfer synthesizes dynamic FP8 profiling activations; static weights/scales are reused. Observed public calls quantize the actual captured BF16 activation.",
        }
        if (
            self.actual_layer["allow_tf32"]
            or self.actual_layer["dispatch_modes"]
            or self.actual_layer["function_modes"]
            or self.actual_layer["torch_profiler_enabled"]
            or self.actual_layer["compiling"]
            or self.actual_layer["kernel_api_log_level"]
            or not self.actual_layer["custom_op_direct_wrapper"]
        ):
            raise RuntimeError("the normal inference fast path is not active")
        self.operands = self.operand_digests()
        self.tuner_settings_initial = self.tuner_settings()
        # Fixture setup is not part of the fresh profiling RNG sequence.
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    def operand_digests(self):
        if not hasattr(self, "weight_bf16_rowmajor"):
            return super().operand_digests()
        values = {
            "input_bf16": self.a,
            "weight_fp8_rowmajor": self.weight_fp8_rowmajor,
            "weight_bf16_rowmajor": self.weight_bf16_rowmajor,
            "input_scale": self.scale_a,
            "weight_scale": self.scale_b,
            "input_ids": self.input_ids,
        }
        return {name: tensor_sha(self.torch, value) for name, value in values.items()}

    def operation(self):
        self.out = self.apply_layer(
            self.a, self.b, self.scale_b, self.scale_a, bias=None
        )
        return self.out

    def error_metrics(self, reference):
        error = self.out.float() - reference
        absolute = error.abs()
        return {
            "max_abs_error": float(absolute.max().item()),
            "max_normalized_error": float(
                (absolute / (0.125 + 0.02 * reference.abs())).max().item()
            ),
            "rmse": float(error.square().mean().sqrt().item()),
            "relative_l2": float(
                error.norm().item() / max(reference.norm().item(), 1e-30)
            ),
        }

    def check_output(self, label):
        torch = self.torch
        torch.cuda.synchronize()
        if (
            list(self.out.shape) != [self.args.m, self.args.n]
            or self.out.dtype != torch.bfloat16
        ):
            raise RuntimeError("actual layer returned the wrong output shape or dtype")
        torch.testing.assert_close(
            self.out.float(), self.reference, rtol=0.02, atol=0.125
        )
        if self.zero_expected and bool(self.out.ne(0).any()):
            raise RuntimeError("changed zero input did not produce exact zero output")
        quantized = self.error_metrics(self.reference)
        dense = self.error_metrics(self.bf16_reference)
        if any(
            not math.isfinite(value) for value in (*quantized.values(), *dense.values())
        ):
            raise RuntimeError("nonfinite numerical error metric")
        self.correctness.append(
            {
                "check": label,
                "max_abs_error": quantized["max_abs_error"],
                "quantized_reference": {"passed": True, **quantized},
                "bf16_reference": {"diagnostic_only": True, **dense},
                "exact_zero_required": self.zero_expected,
                "output_sha256": tensor_sha(torch, self.out),
            }
        )

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

    def choose_hook(self, op, runners, config, inputs, **kwargs):
        result = super().choose_hook(op, runners, config, inputs, **kwargs)
        a, b, sa, sb, out, _workspace = inputs
        if (
            list(a.shape) != [1, self.args.m, self.args.k]
            or a.dtype != self.torch.float8_e4m3fn
            or not a.is_contiguous()
            or list(b.shape) != [1, self.args.k, self.args.n]
            or b.dtype != self.torch.float8_e4m3fn
            or list(b.stride())[-2:] != [1, self.args.k]
            or b.data_ptr() != self.b.data_ptr()
            or sa.dtype != self.torch.float32
            or sb.dtype != self.torch.float32
            or list(sa.shape) != [1]
            or list(sb.shape) != [1]
            or sa.data_ptr() != self.scale_a.data_ptr()
            or sb.data_ptr() != self.scale_b.data_ptr()
            or list(out.shape) != [1, self.args.m, self.args.n]
            or out.dtype != self.torch.bfloat16
        ):
            raise RuntimeError(
                "the actual linear facade supplied unexpected public operands"
            )
        self.observed_public_inputs.append(
            {
                "stage": self.stage,
                "backend": "cublas",
                "capturing": self.torch.cuda.is_current_stream_capturing(),
                "inputs": [
                    {
                        "shape": list(t.shape),
                        "stride": list(t.stride()),
                        "dtype": str(t.dtype),
                    }
                    for t in inputs
                ],
            }
        )
        return result

    def verify_public_bytes(self):
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError("operand digest observation inside capture")
        inputs = self.captured["inputs"]
        actual = {
            "quantized_input_sha256": tensor_sha(self.torch, inputs[0].squeeze(0)),
            "weight_rowmajor_sha256": tensor_sha(self.torch, inputs[1].squeeze(0).t()),
            "input_scale_sha256": tensor_sha(self.torch, inputs[2]),
            "weight_scale_sha256": tensor_sha(self.torch, inputs[3]),
        }
        expected = {
            "quantized_input_sha256": self.quantized_input_sha,
            "weight_rowmajor_sha256": self.operands["weight_fp8_rowmajor"],
            "input_scale_sha256": self.operands["input_scale"],
            "weight_scale_sha256": self.operands["weight_scale"],
        }
        if actual != expected:
            raise RuntimeError("actual public operands differ from the frozen fixture")
        self.observed_public_inputs[-1].update(actual)

    @contextlib.contextmanager
    def native_observation(self):
        original, calls = self.native_module, []
        owner = self

        class ObservedNative:
            def poison(self, args):
                capturing = owner.torch.cuda.is_current_stream_capturing()
                if not capturing:
                    args[2].fill_(float("nan"))
                owner.native_output_poison_checks.append(
                    {
                        "stage": owner.stage,
                        "capturing": capturing,
                        "poisoned_actual_allocation": not capturing,
                    }
                )

            def bmm_fp8(self, *args, **kwargs):
                self.poison(args)
                return original.bmm_fp8(*args, **kwargs)

            def bmm_fp8_run_with_descriptor(self, *args, **kwargs):
                self.poison(args)
                used = original.bmm_fp8_run_with_descriptor(*args, **kwargs)
                calls.append(int(used))
                return used

            def __getattr__(self, name):
                return getattr(original, name)

        if self.native_cell.cell_contents is not original:
            raise RuntimeError("nested native observation")
        self.native_cell.cell_contents = ObservedNative()
        try:
            yield calls
        finally:
            self.native_cell.cell_contents = original

    def observed_call(self, label, expected=None, tier=None, allow_rejection=False):
        result = super().observed_call(label, expected, tier, allow_rejection)
        if not self.zero_expected:
            self.verify_public_bytes()
        return result

    def setup(self):
        super().setup()
        self.verify_public_bytes()

    def capture(self, tactic, tier, label):
        graph = super().capture(tactic, tier, label)
        return GraphBinding(self, graph, self.out)

    @contextlib.contextmanager
    def zero_input(self):
        saved_a = self.a.clone()
        saved_reference, saved_bf16 = self.reference, self.bf16_reference
        try:
            self.a.zero_()
            self.reference = self.torch.zeros_like(saved_reference)
            self.bf16_reference = self.torch.zeros_like(saved_bf16)
            self.zero_expected = True
            yield
        finally:
            self.a.copy_(saved_a)
            self.reference, self.bf16_reference = saved_reference, saved_bf16
            self.zero_expected = False

    def check_graph(self, graph, label):
        with self.zero_input():
            graph.output.fill_(float("nan"))
            graph.replay()
            self.check_output(label + "_changed_input")
        graph.output.fill_(float("nan"))
        graph.replay()
        self.check_output(label + "_restored_input")

    def measurements(self, graph, label):
        result = {}
        self.stage = label + "_measure_eager"
        result["eager"] = measure(self.torch, self.operation, self.args)
        self.check_output(label + "_measured_eager")
        self.stage = label + "_measure_cuda_graph"
        self.out = graph.output
        self.out.fill_(float("nan"))
        # Replay only the captured graph in timed batches, without a copy or a
        # Python output-binding hook. Rebind its allocation after measurement.
        result["cuda_graph"] = measure(self.torch, graph.graph.replay, self.args)
        self.out = graph.output
        self.check_output(label + "_measured_cuda_graph")
        return result

    def natural_measurements(self):
        with self.zero_input():
            self.observed_call("natural_eager_changed_input")
        self.observed_call("natural_eager_restored_input")
        super().natural_measurements()

    def oracle_measurements(self):
        # Enumerate descriptors for provenance only; no forced candidate sweep.
        super().oracle_measurements()
        self.oracle_order = []
        self.observed_call(
            "natural_after_oracle",
            expected=self.natural_tactic,
            tier=self.natural["lookup_tier"],
        )

    def run(self):
        result = super().run()
        helper = Path(__file__).with_name("flashinfer_autotune_descriptor_selection.py")
        result["worker_sha256"] = digest(__file__)
        result["environment"]["helper_sha256"][helper.name] = digest(helper)
        result["environment"]["scope"] = (
            "actual static FP8 linear facade on a checkpoint-derived activation and fused QKV weight; no server"
        )
        result["trial"] = self.args.trial
        result["fixture"] = {
            "manifest_sha256": digest(self.args.fixture_manifest),
            "tensor_file_sha256": digest(self.args.fixture),
            "shape": self.fixture_manifest["shape"],
            "model": self.fixture_manifest["model"],
            "capture": self.fixture_manifest["capture"],
            "tensors": self.fixture_manifest["tensors"],
        }
        result["actual_layer"] = {
            **self.actual_layer,
            "observed_public_inputs": self.observed_public_inputs,
            "native_output_poison_checks": self.native_output_poison_checks,
        }
        result["autotuner_settings"] = {
            "initial": self.tuner_settings_initial,
            "after": self.tuner_settings(self.config),
        }
        result["correctness"]["reference"] = (
            "FP32 GEMM of actual SGLang-quantized activation bytes and frozen FP8 weight, scaled by the actual scalar scales"
        )
        result["correctness"]["bf16_reference"] = (
            "FP32 GEMM of captured BF16 activation and original BF16 weight; diagnostic only"
        )
        result["scope_notes"] = [
            "The actual layer performs static FP8 quantization and allocates its normal output on every eager call; no prequantized substitution or output copy is timed.",
            "The captured graph retains its own actual output allocation. Changed/restored BF16 input and poisoned-output checks inspect that allocation after replay.",
            "The natural default is an uncached miss; cold tuned calls hit memory and fresh warm calls hit existing v1 files. No forced candidate oracle is run.",
            "The unspecified profiling policy is retained. FlashInfer may synthesize dynamic FP8 activations for profiling; the recorded public-input bytes are the actual captured activation quantized by SGLang.",
            "Quantized-operand FP32 correctness is a gate; BF16-reference errors describe this single-batch static quantization and are diagnostic, not full-model accuracy.",
            "Explicit native build/load, fixture setup, and the initial public call are excluded from startup timing. Candidate enumeration and profiling inside the wrapper are included.",
        ]
        return result


def main():
    args = parse_args()
    experiment = None
    try:
        import torch

        with torch.inference_mode():
            experiment = LinearLayerExperiment.__new__(LinearLayerExperiment)
            experiment.__init__(args)
            result = experiment.run()
    except Exception as error:
        traceback.print_exc()
        result = {
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "pid": os.getpid(),
            "arm": args.arm,
            "phase": args.phase,
            "seed": args.seed,
            "trial": args.trial,
            "fixture_manifest_sha256": digest(args.fixture_manifest)
            if args.fixture_manifest.is_file()
            else None,
            "partial_evidence": {
                name: getattr(experiment, name)
                for name in (
                    "quantization_evidence",
                    "profile_attempts",
                    "forbidden_profile_attempts",
                    "profiles",
                    "selections",
                    "correctness",
                    "native_output_poison_checks",
                )
                if experiment is not None and hasattr(experiment, name)
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
