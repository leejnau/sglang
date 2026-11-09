"""Compare saved logits and greedy generations across independent processes."""

import argparse
import json
from pathlib import Path

import numpy as np


def compare_logits(root):
    paths = {p.name.split("-")[1]: p for p in root.glob("matrix-*.npz")}
    pairs = [
        ("01", "02"),
        ("04", "03"),
        ("06", "05"),
        ("07", "08"),
        ("01", "04"),
        ("02", "03"),
        ("06", "07"),
        ("05", "08"),
    ]
    comparisons = []
    for left, right in pairs:
        with np.load(paths[left]) as a, np.load(paths[right]) as b:
            assert set(a.files) == set(b.files) and len(a.files) == 60
            for key in a.files:
                x, y = a[key], b[key]
                row = {
                    "left": left,
                    "right": right,
                    "key": key,
                    "max_abs_error": float(np.max(np.abs(x - y))),
                    "allclose": bool(np.allclose(x, y, atol=0.05, rtol=0.01)),
                    "argmax_equal": bool(np.array_equal(x.argmax(-1), y.argmax(-1))),
                }
                assert np.isfinite(x).all() and np.isfinite(y).all()
                assert row["allclose"] and row["argmax_equal"], row
                comparisons.append(row)
    return {
        "status": "passed",
        "comparisons": comparisons,
        "count": len(comparisons),
        "max_abs_error": max(r["max_abs_error"] for r in comparisons),
    }


def compare_engine(root):
    runs = {
        mode: json.loads((root / f"engine-{mode}.json").read_text())
        for mode in ("legacy", "scoped", "disabled")
    }
    assert all(r["status"] == "passed" for r in runs.values())
    rows = []
    for mode in ("scoped", "disabled"):
        a, b = (runs[m]["requests"] for m in ("legacy", mode))
        assert len(a) == len(b) == 4
        for x, y in zip(a, b):
            assert (x["batch"], x["prompt"]) == (y["batch"], y["prompt"])
            x, y = x["response"], y["response"]
            lx, ly = (r["meta_info"]["output_token_logprobs"] for r in (x, y))
            assert lx and ly and [v[1] for v in lx] == [v[1] for v in ly]
            px, py = (np.array([v[0] for v in values]) for values in (lx, ly))
            assert np.isfinite(px).all() and np.isfinite(py).all()
            assert x["text"] == y["text"]
            assert np.allclose(px, py, atol=0.05, rtol=0.01)
            rows.append(
                {
                    "mode": mode,
                    "tokens": len(lx),
                    "text_equal": True,
                    "token_ids_equal": True,
                    "max_logprob_abs_error": float(np.abs(px - py).max()),
                }
            )
    return {"status": "passed", "comparisons": rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--kind", choices=("logits", "engine"), required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = (compare_logits if args.kind == "logits" else compare_engine)(args.root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], "comparisons", len(result["comparisons"]))


if __name__ == "__main__":
    main()
