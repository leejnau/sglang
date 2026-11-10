"""Summarize fixed-input logits without changing the original acceptance rule."""

import json
from pathlib import Path

import numpy as np


def compare(left, right):
    rows = []
    with np.load(left) as a, np.load(right) as b:
        for key in sorted(set(a.files) & set(b.files)):
            x, y = a[key].astype(np.float64), b[key].astype(np.float64)
            assert x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all()
            logp = x - x.max(-1, keepdims=True)
            logp -= np.log(np.exp(logp).sum(-1, keepdims=True))
            logq = y - y.max(-1, keepdims=True)
            logq -= np.log(np.exp(logq).sum(-1, keepdims=True))
            kl = (np.exp(logp) * (logp - logq)).sum(-1)
            error = np.abs(x - y)
            rows.append(
                {
                    "key": key,
                    "max_abs_error": float(error.max()),
                    "p99_abs_error": float(np.quantile(error, 0.99)),
                    "allclose_005_001": bool(np.allclose(x, y, atol=0.05, rtol=0.01)),
                    "top1_equal_rows": int((x.argmax(-1) == y.argmax(-1)).sum()),
                    "row_count": len(x),
                    "max_kl_baseline_to_candidate": float(kl.max()),
                    "mean_kl_baseline_to_candidate": float(kl.mean()),
                }
            )
    return {
        "left": left.name,
        "right": right.name,
        "comparisons": rows,
        "max_abs_error": max(r["max_abs_error"] for r in rows),
        "top1_equal_rows": sum(r["top1_equal_rows"] for r in rows),
        "row_count": sum(r["row_count"] for r in rows),
        "max_kl_baseline_to_candidate": max(
            r["max_kl_baseline_to_candidate"] for r in rows
        ),
    }


def main():
    root = Path("/study/reviewed-study")
    baseline = root / "numerical-legacy-warm.npz"
    result = {
        "purpose": "Diagnostic effect sizes; original pointwise criterion remains unchanged.",
        "comparisons": [
            compare(baseline, root / "numerical-scoped-warm.npz"),
            compare(baseline, root / "numerical-scoped-default.npz"),
            compare(baseline, root / "numerical-legacy-explicit-attempt02.npz"),
            compare(
                root / "numerical-scoped-warm.npz",
                root / "numerical-legacy-explicit-attempt02.npz",
            ),
        ],
    }
    (root / "numerical-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            [
                {k: v for k, v in r.items() if k != "comparisons"}
                for r in result["comparisons"]
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
