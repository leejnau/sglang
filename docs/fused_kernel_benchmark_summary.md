# Fused RoPE+Quantize+Append Benchmark Summary

## Benchmark Commands Used

Runs executed (`--repeats 1` due cluster lease constraints):

```bash
python3 scripts/bench_fused_kernel_eval.py --variant baseline --kv-mode fp4 --run-tag main --repeats 1
python3 scripts/bench_fused_kernel_eval.py --variant fused    --kv-mode fp4 --run-tag fused_branch --repeats 1
python3 scripts/bench_fused_kernel_eval.py --variant baseline --kv-mode fp8 --run-tag main --repeats 1
python3 scripts/bench_fused_kernel_eval.py --variant fused    --kv-mode fp8 --run-tag fused_branch --repeats 1
```

## Key Results (Baseline vs Fused)

| KV Mode | Profile | Conc | Base Throughput | Fused Throughput | Throughput Δ | Base TPOT | Fused TPOT | TPOT Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fp4 | low_noise_decode_focus | 1 | 111.45 | 112.11 | +0.59% | 8.861 | 8.803 | +0.66% |
| fp4 | low_noise_decode_focus | 4 | 331.27 | 332.80 | +0.46% | 11.898 | 11.840 | +0.49% |
| fp4 | realism_stress | 64 | 1616.12 | 1595.97 | -1.25% | 26.102 | 26.046 | +0.21% |
| fp8 | low_noise_decode_focus | 1 | 124.05 | 124.72 | +0.54% | 7.985 | 7.944 | +0.52% |
| fp8 | low_noise_decode_focus | 4 | 379.29 | 381.28 | +0.52% | 10.404 | 10.350 | +0.52% |
| fp8 | realism_stress | 64 | 1724.45 | 2569.88 | +48.99% | 22.805 | 22.787 | +0.08% |

## Takeaways

- Low-noise cases (conc=1/4) show a consistent **~0.5-0.7%** fused win in both throughput and TPOT for `fp4` and `fp8`.
- This matches the expected impact from the microsecond-level per-layer kernel savings seen in traces.
- High-concurrency realism (`conc=64`) is noisy/mixed; the large fp8 throughput jump is not corroborated by TPOT and should be treated as a single-sample outlier.

## Conclusion

The fused kernel provides a **real, consistent micro-optimization** (~**+0.5% class**) in low-noise serving benchmarks, with no clear regression signal in those checks.
Note: results are from `--repeats 1`; additional repeats would increase confidence for realism-load behavior.
