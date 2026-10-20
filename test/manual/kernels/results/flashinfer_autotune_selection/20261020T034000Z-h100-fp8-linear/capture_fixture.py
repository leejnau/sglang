"""Freeze one checkpoint-derived BF16 activation and static FP8 QKV weight.

This is preparation for a manual linear-layer experiment. It does not launch an
SGLang server, run autotuning, or claim accuracy for a quantized full model.
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
TOKENS = 144
SEED = 19
CHECKPOINT_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "model.safetensors",
)
PROMPT = (
    "The research team compares repeatable measurements from the same model "
    "inputs, records the numerical results, and checks each cached choice. "
) * 32


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


def source_evidence(source_root):
    """Check the production loader's Q/K/V order without importing its kernels."""
    model_path = source_root / "python/sglang/srt/models/qwen3.py"
    linear_path = source_root / "python/sglang/srt/layers/linear.py"
    model_tree = ast.parse(model_path.read_text())
    mappings = []
    for node in ast.walk(model_tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "stacked_params_mapping"
            for target in node.targets
        ):
            mappings.append(ast.literal_eval(node.value))
    expected = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
    ]
    if not any(mapping[:3] == expected for mapping in mappings):
        raise RuntimeError("production checkpoint-to-QKV loader mapping changed")
    linear_tree = ast.parse(linear_path.read_text())
    qkv_class = next(
        node
        for node in linear_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "QKVParallelLinear"
    )
    offset_function = next(
        node
        for node in qkv_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_get_shard_offset_mapping"
    )
    offsets = next(
        node.value
        for node in offset_function.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "shard_offset_mapping"
            for target in node.targets
        )
    )
    actual = {ast.literal_eval(k): v for k, v in zip(offsets.keys, offsets.values)}
    expected_offsets = {
        "q": "0",
        "k": "self.num_heads * self.head_size",
        "v": "(self.num_heads + self.num_kv_heads) * self.head_size",
    }
    for shard, expression in expected_offsets.items():
        if ast.dump(actual[shard]) != ast.dump(ast.parse(expression, mode="eval").body):
            raise RuntimeError("production QKV output offsets changed")
    files = (
        model_path,
        linear_path,
        source_root / "python/sglang/srt/layers/quantization/fp8_utils.py",
        source_root / "python/sglang/kernels/ops/quantization/fp8_kernel.py",
        source_root / "python/sglang/srt/utils/custom_op.py",
        source_root / "python/sglang/srt/runtime_context.py",
    )
    return {str(path.relative_to(source_root)): file_sha(path) for path in files}


def main():
    args = parse_args()
    # Heavy dependencies and device access follow parsing so --help is CPU-only.
    if args.output_dir.exists():
        raise RuntimeError("capture refuses to overwrite any existing output directory")
    source_root = Path(__file__).resolve().parents[6]
    source_hashes = source_evidence(source_root)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()

    import torch
    from huggingface_hub import snapshot_download
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
            allow_patterns=list(CHECKPOINT_FILES),
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
        for name in CHECKPOINT_FILES
        if (snapshot / name).is_file()
    }
    if (
        not {"config.json", "model.safetensors", "tokenizer_config.json"}
        <= checkpoint_files.keys()
    ):
        raise RuntimeError(
            "the pinned single-file checkpoint or tokenizer is incomplete"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    encoded = tokenizer(
        PROMPT,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=TOKENS,
        padding=False,
    )
    if tuple(encoded["input_ids"].shape) != (1, TOKENS):
        raise RuntimeError("fixed prompt did not yield exactly144 unpadded tokens")
    if not bool(encoded["attention_mask"].all()):
        raise RuntimeError("the capture input contains padding")
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
    projections = [
        getattr(layer.self_attn, name) for name in ("q_proj", "k_proj", "v_proj")
    ]
    if any(projection.bias is not None for projection in projections):
        raise RuntimeError("this bounded fixture expects an unbiased QKV projection")
    weights = [
        projection.weight.detach().cpu().contiguous() for projection in projections
    ]
    if any(weight.dtype != torch.bfloat16 or weight.ndim != 2 for weight in weights):
        raise RuntimeError("checkpoint projections are not rank-two BF16 weights")
    if len({weight.shape[1] for weight in weights}) != 1:
        raise RuntimeError("Q/K/V projection input widths differ")
    captured = []

    def capture_input(_module, positional):
        if len(positional) != 1 or positional[0].dtype != torch.bfloat16:
            raise RuntimeError(
                "the actual Q projection did not receive BF16 activations"
            )
        captured.append(positional[0].detach().cpu().contiguous())

    handle = projections[0].register_forward_pre_hook(capture_input)
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
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("the first-layer Q projection hook did not run exactly once")
    k = weights[0].shape[1]
    if tuple(captured[0].shape) != (1, TOKENS, k):
        raise RuntimeError(
            "actual activation shape differs from the expected linear input"
        )
    input_bf16 = captured[0].view(TOKENS, k).contiguous()
    weight_bf16 = torch.cat(weights, dim=0).contiguous()
    n = weight_bf16.shape[0]
    if k % 16 or n % 16:
        raise RuntimeError(
            "captured projection cannot use the static per-tensor FP8 path"
        )
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
    # This independent quantizer is a diagnostic; the measured layer must call
    # its actual SGLang Triton quantizer, with no prequantized substitution.
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
    tensor_provenance = {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": tensor_sha(torch, value),
            "nbytes": value.numel() * value.element_size(),
        }
        for name, value in tensors.items()
    }
    upstream_sources = {}
    for cls in (type(model), type(layer), type(tokenizer)):
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
            "calibration": "single captured BF16 batch",
        },
    )
    manifest = {
        "schema_version": 1,
        "shape": [TOKENS, n, k],
        "model": {
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "resolved_snapshot_revision": snapshot.name,
            "class": type(model).__name__,
            "checkpoint_dtype": "torch.bfloat16",
        },
        "capture": {
            "layer": "model.layers.0.self_attn",
            "hook": "model.layers.0.self_attn.q_proj:forward_pre_hook",
            "captured_shape": list(captured[0].shape),
            "saved_shape": list(input_bf16.shape),
            "weight_order": ["q_proj", "k_proj", "v_proj"],
            "projection_shapes": [list(weight.shape) for weight in weights],
            "tensor_parallel_size": 1,
            "bias": None,
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
        "tensors": tensor_provenance,
        "quantization": {
            "dtype": str(fp8_dtype),
            "max_finite": limit,
            "scale_rule": "absmax_fp32 / 448",
            "weight_rule": "clamp(weight.float() / weight_scale,-448,448).to(float8_e4m3fn)",
            "independent_input_rule": "clamp(input.float() * input_scale.reciprocal(),-448,448).to(float8_e4m3fn)",
            "independent_input_fp8_sha256": tensor_sha(torch, input_reference_fp8),
            "calibration_batches": 1,
        },
        "tokenization": {
            "prompt": PROMPT,
            "input_ids": encoded["input_ids"].tolist(),
            "attention_mask": encoded["attention_mask"].tolist(),
            "add_special_tokens": False,
            "truncation": True,
            "max_length": TOKENS,
            "padding": False,
            "tokenizer_class": type(tokenizer).__name__,
        },
        "source": {
            "sglang_commit": commit,
            "capture_script_sha256": file_sha(Path(__file__)),
            "sglang_source_sha256": source_hashes,
            "upstream_source_sha256": upstream_sources,
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
            "Original checkpoint weights and captured activations are BF16; FP8 weights and static scales are prepared from this single captured batch, not loaded from a precalibrated FP8 checkpoint.",
            "The measured operation must use the actual SGLang static quantizer and linear facade; this capture performs no autotuning.",
            "Fused Q/K/V output order matches the audited production loader at TP1. This fixture supports one linear-layer experiment and does not establish full-model quantized accuracy or serving performance.",
            "The independent FP8 activation digest is a quantizer diagnostic; an independent GEMM reference should consume the actual SGLang-quantized activation bytes.",
        ],
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": "success",
                "shape": manifest["shape"],
                "fixture_sha256": manifest["tensor_file"]["sha256"],
                "manifest_sha256": file_sha(manifest_path),
            },
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
