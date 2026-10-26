"""Capture one fixed BF16 MLP down projection for manual FP8 selection research.

This uses the same pinned checkpoint and tokens as the prior QKV fixture. The
input is the actual post-SiLU-times-up activation from a complete BF16 forward;
the measured worker must quantize it through the real SGLang linear facade.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import json
import math
import subprocess
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
SHAPE = (144, 1024, 3072)
SEED = 19
FIXTURE_KIND = "qwen3_mlp_down_proj"
CHECKPOINT_KEY = "model.layers.0.mlp.down_proj.weight"
PREVIOUS_MANIFEST = (
    "test/manual/kernels/results/flashinfer_autotune_selection/"
    "20261020T034000Z-h100-fp8-linear/fixture-manifest.json"
)
PREVIOUS_MANIFEST_SHA256 = (
    "e13a86fab5db5796b22d8dab0f84080b8057b036ca7aef2c7be8832e76505bf1"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def file_sha(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def tensor_sha(torch, value):
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def tensor_metadata(torch, value):
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": tensor_sha(torch, value),
        "nbytes": value.numel() * value.element_size(),
    }


def previous_evidence(source_root):
    path = source_root / PREVIOUS_MANIFEST
    if file_sha(path) != PREVIOUS_MANIFEST_SHA256:
        raise RuntimeError("the pinned prior fixture manifest changed")
    evidence = json.loads(path.read_text())
    if (
        evidence["model"]["repo_id"] != MODEL_ID
        or evidence["model"]["revision"] != MODEL_REVISION
        or evidence["capture"]["seed"] != SEED
        or evidence["shape"][0] != SHAPE[0]
    ):
        raise RuntimeError("the prior checkpoint or input contract changed")
    return evidence


def source_evidence(source_root):
    """Check the actual TP1 projection's dimensions and activation wiring."""
    model_path = source_root / "python/sglang/srt/models/qwen3.py"
    model_tree = ast.parse(model_path.read_text())
    if not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "sglang.srt.models.qwen2"
        and any(
            alias.name == "Qwen2MLP" and alias.asname == "Qwen3MLP"
            for alias in node.names
        )
        for node in model_tree.body
    ):
        raise RuntimeError("the SGLang Qwen3 MLP implementation changed")
    mlp_path = source_root / "python/sglang/srt/models/qwen2.py"
    tree = ast.parse(mlp_path.read_text())
    mlp = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen2MLP"
    )
    constructors = {}
    for node in ast.walk(mlp):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    constructors[target.attr] = node.value
    down = constructors["down_proj"]
    if (
        not isinstance(down.func, ast.Name)
        or down.func.id != "RowParallelLinear"
        or [ast.unparse(arg) for arg in down.args[:2]]
        != ["intermediate_size", "hidden_size"]
        or not any(
            keyword.arg == "bias"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
            for keyword in down.keywords
        )
        or ast.unparse(constructors["act_fn"].func) != "SiluAndMul"
    ):
        raise RuntimeError("the SGLang down projection or activation contract changed")
    files = (
        model_path,
        mlp_path,
        source_root / "python/sglang/srt/layers/linear.py",
        source_root / "python/sglang/srt/layers/quantization/fp8_utils.py",
        source_root / "python/sglang/kernels/ops/quantization/fp8_kernel.py",
        source_root / "python/sglang/srt/utils/custom_op.py",
        source_root / "python/sglang/srt/runtime_context.py",
    )
    return {str(path.relative_to(source_root)): file_sha(path) for path in files}


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise RuntimeError("capture refuses to overwrite an existing output directory")
    source_root = Path(__file__).resolve().parents[3]
    previous = previous_evidence(source_root)
    source_hashes = source_evidence(source_root)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()

    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA capture requested without an accessible device")
        torch.cuda.set_device(0)
        if torch.cuda.get_device_capability() != (9, 0):
            raise RuntimeError("this fixture capture is qualified for the SM90 host")
    torch.manual_seed(SEED)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=str(args.model_cache.resolve()),
            local_files_only=True,
        )
    )
    if snapshot.name != MODEL_REVISION:
        raise RuntimeError(
            "resolved checkpoint snapshot differs from the pinned revision"
        )
    checkpoint_files = {
        name: {
            "sha256": file_sha(snapshot / name),
            "size_bytes": (snapshot / name).stat().st_size,
        }
        for name in previous["checkpoint_files"]
    }
    if checkpoint_files != previous["checkpoint_files"]:
        raise RuntimeError(
            "checkpoint or tokenizer files differ from the previous fixture"
        )
    config = json.loads((snapshot / "config.json").read_text())
    if (config["hidden_size"], config["intermediate_size"], config["hidden_act"]) != (
        SHAPE[1],
        SHAPE[2],
        "silu",
    ):
        raise RuntimeError(
            "pinned checkpoint geometry or activation differs from the prespecified MLP"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    encoded = tokenizer(
        previous["tokenization"]["prompt"],
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=SHAPE[0],
        padding=False,
    )
    if (
        encoded["input_ids"].tolist() != previous["tokenization"]["input_ids"]
        or encoded["attention_mask"].tolist()
        != previous["tokenization"]["attention_mask"]
        or tensor_sha(torch, encoded["input_ids"])
        != previous["tensors"]["input_ids"]["sha256"]
    ):
        raise RuntimeError(
            "the captured workload must use the exact prior 144 token IDs"
        )
    model = (
        AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        .to(args.device)
        .eval()
    )
    layer = model.model.layers[0]
    mlp = layer.mlp
    if any(
        getattr(mlp, name).bias is not None
        for name in ("gate_proj", "up_proj", "down_proj")
    ):
        raise RuntimeError("this fixture requires unbiased MLP projections")
    weight_bf16 = mlp.down_proj.weight.detach().cpu().contiguous()
    if weight_bf16.dtype != torch.bfloat16 or tuple(weight_bf16.shape) != SHAPE[1:]:
        raise RuntimeError(
            "actual down projection differs from the prespecified BF16 weight"
        )
    with safe_open(
        snapshot / "model.safetensors", framework="pt", device="cpu"
    ) as checkpoint:
        checkpoint_weight = checkpoint.get_tensor(CHECKPOINT_KEY)
    if (
        not torch.equal(checkpoint_weight, weight_bf16)
        or checkpoint_weight.dtype != weight_bf16.dtype
    ):
        raise RuntimeError(
            "captured down weight differs from the raw checkpoint tensor"
        )
    captured = {name: [] for name in ("input", "gate", "up", "output")}

    def capture_input(_module, positional):
        if len(positional) != 1 or positional[0].dtype != torch.bfloat16:
            raise RuntimeError(
                "the actual down projection did not receive BF16 activations"
            )
        captured["input"].append(positional[0].detach().clone())

    def capture_output(name):
        def hook(_module, _positional, output):
            captured[name].append(output.detach().clone())

        return hook

    handles = [
        mlp.down_proj.register_forward_pre_hook(capture_input),
        mlp.down_proj.register_forward_hook(capture_output("output")),
        mlp.gate_proj.register_forward_hook(capture_output("gate")),
        mlp.up_proj.register_forward_hook(capture_output("up")),
    ]
    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=encoded["input_ids"].to(args.device),
                attention_mask=encoded["attention_mask"].to(args.device),
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            if not bool(torch.isfinite(outputs.logits).all()):
                raise RuntimeError("BF16 checkpoint forward produced nonfinite logits")
            logits_shape = list(outputs.logits.shape)
    finally:
        for handle in handles:
            handle.remove()
    if any(len(values) != 1 for values in captured.values()):
        raise RuntimeError("each first-layer MLP hook must run exactly once")
    if tuple(captured["input"][0].shape) != (1, SHAPE[0], SHAPE[2]):
        raise RuntimeError("actual down-projection input has unexpected geometry")
    with torch.inference_mode():
        independent_mlp_input = (
            torch.nn.functional.silu(captured["gate"][0]) * captured["up"][0]
        )
    if not torch.equal(independent_mlp_input, captured["input"][0]):
        raise RuntimeError(
            "captured input differs from the actual gate SiLU times up values"
        )
    input_bf16 = captured["input"][0].cpu().view(SHAPE[0], SHAPE[2]).contiguous()
    fp8_dtype = torch.float8_e4m3fn
    limit = float(torch.finfo(fp8_dtype).max)
    input_scale = (input_bf16.float().abs().max() / limit).reshape(1)
    weight_scale = (weight_bf16.float().abs().max() / limit).reshape(1)
    if not all(
        math.isfinite(float(scale.item())) and scale.item() > 0
        for scale in (input_scale, weight_scale)
    ):
        raise RuntimeError("capture calibration produced invalid scales")
    weight_fp8 = (weight_bf16.float() / weight_scale).clamp(-limit, limit).to(fp8_dtype)
    input_reference_fp8 = (
        (input_bf16.float() * input_scale.reciprocal())
        .clamp(-limit, limit)
        .to(fp8_dtype)
    )
    tensors = {
        "input_bf16": input_bf16,
        "weight_fp8_rowmajor": weight_fp8.contiguous(),
        "weight_bf16_rowmajor": weight_bf16,
        "input_scale": input_scale,
        "weight_scale": weight_scale,
        "input_ids": encoded["input_ids"].contiguous(),
    }
    if any(not bool(torch.isfinite(value.float()).all()) for value in tensors.values()):
        raise RuntimeError("fixture contains nonfinite values")
    upstream_sources = {}
    for cls in (type(model), type(layer), type(mlp), type(tokenizer)):
        path = inspect.getsourcefile(cls)
        if path is not None:
            upstream_sources[cls.__module__] = file_sha(Path(path))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tensor_file = args.output_dir / "fixture.safetensors"
    save_file(
        tensors,
        tensor_file,
        metadata={
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "fixture_kind": FIXTURE_KIND,
            "calibration": "single captured BF16 batch",
        },
    )
    manifest = {
        "schema_version": 2,
        "fixture_kind": FIXTURE_KIND,
        "shape": list(SHAPE),
        "model": {
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "resolved_snapshot_revision": snapshot.name,
            "class": type(model).__name__,
            "checkpoint_dtype": "torch.bfloat16",
        },
        "capture": {
            "layer": "model.layers.0.mlp",
            "hook": "model.layers.0.mlp.down_proj:forward_pre_hook",
            "captured_shape": list(captured["input"][0].shape),
            "saved_shape": list(input_bf16.shape),
            "weight_order": ["down_proj"],
            "projection_shapes": [list(weight_bf16.shape)],
            "checkpoint_tensor": CHECKPOINT_KEY,
            "checkpoint_tensor_byte_match": True,
            "tensor_parallel_size": 1,
            "bias": None,
            "activation": "silu(gate_proj(x)) * up_proj(x)",
            "activation_byte_match": True,
            "hook_values": {
                name: tensor_metadata(torch, values[0])
                for name, values in captured.items()
            },
            "full_bf16_forward_completed": True,
            "logits_shape": logits_shape,
            "attention_implementation": "eager",
            "model_training": model.training,
            "seed": SEED,
            "device": args.device,
        },
        "tensor_file": {
            "name": tensor_file.name,
            "sha256": file_sha(tensor_file),
            "size_bytes": tensor_file.stat().st_size,
        },
        "tensors": {
            name: tensor_metadata(torch, value) for name, value in tensors.items()
        },
        "quantization": {
            "dtype": str(fp8_dtype),
            "max_finite": limit,
            "scale_rule": "absmax_fp32 / 448",
            "weight_rule": "clamp(weight.float() / weight_scale,-448,448).to(float8_e4m3fn)",
            "independent_input_rule": "clamp(input.float() * input_scale.reciprocal(),-448,448).to(float8_e4m3fn)",
            "independent_input_fp8_sha256": tensor_sha(torch, input_reference_fp8),
            "calibration_batches": 1,
        },
        "tokenization": previous["tokenization"],
        "source": {
            "sglang_commit": commit,
            "capture_script_sha256": file_sha(Path(__file__)),
            "sglang_source_sha256": source_hashes,
            "upstream_source_sha256": upstream_sources,
        },
        "previous_fixture": {
            "manifest_path": PREVIOUS_MANIFEST,
            "manifest_sha256": PREVIOUS_MANIFEST_SHA256,
            "input_ids_byte_match": True,
            "checkpoint_files_byte_match": True,
        },
        "software": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "safetensors",
                "huggingface_hub",
                "tokenizers",
            )
        },
        "checkpoint_files": checkpoint_files,
        "scope_notes": [
            "Geometry and first-layer down projection were chosen before timing. Activations come from a complete eager BF16 checkpoint forward, not a quantized full-model forward.",
            "FP8 weights and static scales are calibrated from one captured batch. The original BF16 weight is diagnostic only; no full-model quantization accuracy claim follows.",
            "The measured worker must call actual SGLang static_quant_fp8 and apply_fp8_linear_bmm_flashinfer. This capture performs no autotuning or timing.",
            "Independent activation bytes are a quantizer diagnostic. GEMM references must consume the actual SGLang-quantized input.",
        ],
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": "success",
                "shape": list(SHAPE),
                "fixture_sha256": file_sha(tensor_file),
                "manifest_sha256": file_sha(manifest_path),
            },
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
