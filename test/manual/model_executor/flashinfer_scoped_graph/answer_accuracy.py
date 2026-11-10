"""Paired answer accuracy with the preserved numerical-mismatch cache.

This is a bounded diagnostic, not the repository CI benchmark. Requests run
one at a time to exercise the implicated batch-one tactic throughout decode.
"""

import argparse
import hashlib
import json
import os
import re
import time
import traceback
from decimal import Decimal
from pathlib import Path


def answer(text):
    tail = text.rsplit("####", 1)[-1]
    values = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", tail)
    return str(Decimal(values[-1].replace(",", "")).normalize()) if values else None


def cache_hashes(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*.json")
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["legacy", "scoped"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    os.environ["SGLANG_ENABLE_FLASHINFER_GRAPH_AUTOTUNE"] = str(
        int(args.arm == "scoped")
    )
    os.environ["SGLANG_FLASHINFER_AUTOTUNE_CACHE"] = "1"
    os.environ["SGLANG_CACHE_DIR"] = str(args.cache)
    from transformers import AutoTokenizer

    from sglang import Engine

    data = Path("/study/gsm8k")
    train = [
        json.loads(line) for line in (data / "train.jsonl").read_text().splitlines()
    ]
    test = [json.loads(line) for line in (data / "test.jsonl").read_text().splitlines()]
    prefix = (
        "\n\n".join(
            "Question: " + r["question"] + "\nAnswer: " + r["answer"] for r in train[:4]
        )
        + "\n\n"
    )
    tokenizer = AutoTokenizer.from_pretrained("/study/model", local_files_only=True)
    cfg = dict(
        model_path="/study/model",
        tokenizer_path="/study/model",
        quantization="modelopt_fp8",
        dtype="bfloat16",
        device="cuda",
        tp_size=1,
        max_running_requests=16,
        max_total_tokens=8192,
        context_length=4096,
        mem_fraction_static=0.7,
        disable_overlap_schedule=True,
        disable_radix_cache=True,
        attention_backend="triton",
        kv_cache_dtype="bfloat16",
        cuda_graph_config={
            "decode": {"backend": "full", "bs": [1, 4, 16], "max_bs": 16},
            "prefill": {"backend": "disabled"},
        },
    )
    result = {
        "status": "started",
        "arm": args.arm,
        "dataset": json.loads((data / "manifest.json").read_text()),
        "configuration": cfg,
        "test_indices": list(range(200)),
        "train_demonstrations": list(range(4)),
        "max_new_tokens": 512,
        "temperature": 0,
        "batch": 1,
        "started": time.time(),
        "cache_before": cache_hashes(args.cache),
        "requests": [],
    }
    engine = None
    try:
        engine = Engine(**cfg)
        for i, row in enumerate(test[:200]):
            prompt = prefix + "Question: " + row["question"] + "\nAnswer:"
            count = len(tokenizer.encode(prompt))
            assert count + 512 <= 4096
            output = engine.generate(
                prompt=prompt,
                sampling_params={
                    "temperature": 0,
                    "max_new_tokens": 512,
                    "stop": ["Question", "Assistant:", "<|separator|>"],
                },
            )
            expected = answer(row["answer"])
            predicted = answer(output["text"])
            result["requests"].append(
                {
                    "index": i,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "input_tokens": count,
                    "expected": expected,
                    "predicted": predicted,
                    "correct": expected == predicted,
                    "text": output["text"],
                    "text_sha256": hashlib.sha256(output["text"].encode()).hexdigest(),
                    "meta_info": output["meta_info"],
                }
            )
            if (i + 1) % 20 == 0:
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(f"{args.arm} completed {i + 1}/200", flush=True)
        result["correct"] = sum(r["correct"] for r in result["requests"])
        result["score"] = result["correct"] / 200
        result["cache_after"] = cache_hashes(args.cache)
        assert result["cache_before"] == result["cache_after"], (
            "accuracy run changed the preserved caches"
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
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
