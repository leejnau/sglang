"""Greedy multi-token request consistency through SGLang's public Engine."""

import argparse
import json
import os
import time
import traceback
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["legacy", "scoped", "disabled"], required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    result = {"mode": args.mode, "status": "started", "started": time.time()}
    os.environ["SGLANG_ENABLE_FLASHINFER_GRAPH_AUTOTUNE"] = (
        "0" if args.mode == "legacy" else "1"
    )
    os.environ["SGLANG_CACHE_DIR"] = (
        "/study/pilot-scoped-cache"
        if args.mode == "scoped"
        else "/study/pilot-legacy-cache"
    )
    os.environ["SGLANG_FLASHINFER_AUTOTUNE_CACHE"] = "1"
    engine = None
    try:
        from transformers import AutoTokenizer

        from sglang import Engine

        tokenizer = AutoTokenizer.from_pretrained("/study/model", local_files_only=True)
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for text in [
                "Name the capital of France.",
                "What is two plus three?",
                "Name two primary colors.",
            ]
        ]
        cfg = dict(
            model_path="/study/model",
            tokenizer_path="/study/model",
            quantization="modelopt_fp8",
            dtype="bfloat16",
            device="cuda",
            tp_size=1,
            max_running_requests=16,
            max_total_tokens=8192,
            context_length=512,
            mem_fraction_static=0.7,
            disable_overlap_schedule=True,
            disable_radix_cache=True,
            attention_backend="triton",
            kv_cache_dtype="bfloat16",
            cuda_graph_config={
                "decode": {"backend": "disabled"}
                if args.mode == "disabled"
                else {"backend": "full", "bs": [1, 4, 16], "max_bs": 16},
                "prefill": {"backend": "disabled"},
            },
        )
        result["config"] = cfg
        start = time.perf_counter()
        engine = Engine(**cfg)
        result["startup_seconds"] = time.perf_counter() - start
        result["requests"] = []
        for bs in (1, 3):
            output = engine.generate(
                prompt=prompts[:bs],
                sampling_params={"temperature": 0, "max_new_tokens": 32},
                return_logprob=True,
            )
            if isinstance(output, dict):
                output = [output]
            assert len(output) == bs
            for i, item in enumerate(output):
                assert item["text"] and item["meta_info"]["completion_tokens"] > 0
                result["requests"].append({"batch": bs, "prompt": i, "response": item})
        if args.mode == "disabled":
            root = Path(os.environ["SGLANG_CACHE_DIR"]) / "flashinfer" / "autotune"
            assert not list(root.rglob("*.decode_graph")), (
                "disabled graph created a scoped store"
            )
        result["status"] = "passed"
    except BaseException:
        result["status"] = "failed"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        if engine is not None:
            engine.shutdown()
        result["finished"] = time.time()
        args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
