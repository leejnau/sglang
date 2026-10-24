# Measurement objective and activation values

This research batch tests two explanations for a previously observed mismatch
between startup profiling and independent replay of a real SGLang FP8 layer.
It does not change production behavior or establish serving performance.

The frozen plan is `plan.json`; `schedule.json` assigns eight complete four-cell
blocks before any GPU results are observed. Each fresh process uses either the
unspecified automatic measurement objective or explicit CUDA Graph objective,
and either matched synthetic activations or the captured activation values at
the exact M144 profiling endpoint. All nine existing profile shapes remain.
Both initializer arms perform the same allocation/copy operations; physical
copy history can still differ. Synthetic mode is a controlled initializer,
not an untouched production baseline.

The worker uses the actual SGLang layer and startup wrapper, the qualified
current-source FlashInfer tuner, and explicitly built native providers. It
retains the natural winner and genuine cache-miss default graphs, measures
their paired replay times, checks their own outputs, tests native acceptance
on their actual operands after timing, and traces those same graph objects.
Every process has a separate empty tuner cache. No managed-store migration or
scoped-cache prototype is included.

`stage.sh` installs a verified sparse snapshot on the reserved host.
`launch.sh` runs four disposable admission processes first, validating shapes,
keys, candidates, RNG preservation, initializer bytes, and explicit native
acceptance. A successful cross-cell admission permits the fixed 32-process
timing cohort. A first-block integrity check precedes the remaining blocks;
winners and timing signs never determine whether the cohort continues.
Failures are preserved and stop the attempt, without outcome-based retries.

`run_current.py` verifies the qualified source and builds/binds the original
native module once per process. `analyze.py` independently checks artifacts
and estimates contrasts using whole blocks, rather than treating the 24
rounds within a process as independent observations. The two primary factors
use prespecified 97.5% bootstrap intervals. With eight blocks these remain
exploratory estimates, not equivalence or safety tests.

The complete raw archive, native binaries, fixture, and source inventory are
preserved separately. Any public result package must retain exact file hashes
and state which source-inventory metadata was omitted. Historical batches are
context only and are not pooled with this current-source cohort.

At source freeze, GPU execution and performance results are pending.

Attempt01 stopped at the first admission on a live-cache fingerprint guard; no timing workers ran. Its complete evidence is preserved in `attempt01/`. A CPU reproduction shows that the checker can hash equal nested sets differently after deep-copy restoration. A corrected checker will be frozen as a new source revision before a fresh admission sequence. The factor definitions, schedule, sample count and estimators remain unchanged. No failed performance trial is being replaced or excluded.

Attempt02 completed all four GPU admission workers (324 profiles, 108 numerical guards), then stopped on an analyzer assertion that incorrectly demanded CUDA flags from host C++ compilation. All actual records passed a separate 2,665-check audit and the corrected analyzer offline. The original failed gate remains in `attempt02/raw-public.tar.gz`; corrected offline analysis is explicitly separate. No timing workers ran. The next source revision changes only compiler-command validation and its CPU fixtures, then starts a fresh complete attempt.


Attempt03 completed on frozen source `e2f0716b5b841360dd5fbeaed9e41f38c6e42071`:
all four admissions and all 32 timing processes passed. The cohort retained
768 paired rounds / 1,536 samples, with 2,080 timing-process numerical checks.
The independent raw/statistical audit passed 28,650 checks and reproduced the
prespecified whole-block bootstrap. A separate audit passed 2,665 actual
admission checks and 192 bindings between the cohort and those admissions.
Earlier failed admissions remain separate.

| Objective / activation recipe | Default selections | Explicit selections | Replay ratio above 1.10 |
| --- | ---: | ---: | ---: |
| Automatic / matched synthetic | 7 / 8 | 1 / 8 | 1 / 8 |
| Graph / matched synthetic | 8 / 8 | 0 / 8 | 0 / 8 |
| Automatic / captured values | 8 / 8 | 0 / 8 | 0 / 8 |
| Graph / captured values | 8 / 8 | 0 / 8 | 0 / 8 |

The explicit case, `block0-auto_synthetic-timing.json`, appeared 0.6311%
faster in profiling (10.0768 versus 10.1408 microseconds). Its retained full
layer graph was **16.0214% slower** than the default graph: paired median ratio
1.160214, with winner-first/default-first strata 1.160709 / 1.159723. Native
acceptance returned 1 for the exact retained operands, all 65 numerical guards
passed, and the original graphs used different GEMM signatures. The common
quantizer was unchanged. This is a selection-quality counterexample under
this qualified source/configuration, with no numerical failure observed.
Post-timing native acceptance and capture/trace evidence remain distinct.

Each factor compares the selected/default replay ratio across recipes; it is
not a direct serving latency ratio. The graph/automatic factor was 0.990918,
exploratory 97.5% interval [0.967842, 1.006265]. The captured/synthetic
factor ratio was 0.990767, interval [0.969008, 1.003062]. Both include 1.
The secondary interaction was 1.014288, 95% interval [0.981947, 1.061109].
Both factor points share the same lone automatic/synthetic outcome; these
are not two independent demonstrations of prevention. These contrasts do not
establish that either change prevents bad selections. The other 31 cases
selected default and had matching ordered kernel signatures; small timing
differences between those graphs are not a selection benefit. A posthoc
diagnostic found the first-captured graph slower in 29/31 default pairs
(median first/second 1.007611); this is a descriptive association, not an
identified cause. Eight workers per cell cannot establish rare-regression
safety. No historical results are pooled.

Retain SGLang's current production policy and v1 persistence. The next bounded
measurement proposal is a fixed-descriptor/default mechanism replication on
the same problem and qualified source, comparing profiling and independent
graph replay under both objectives and initializer recipes. Freeze that
protocol separately before execution; this batch launches no extra trial.
The independent owned-output benchmark validation fix is ready for upstream
review preparation, subject to a fresh duplicate/rebase check and target-main
checks. No PR was opened and the scoped-cache prototype remains parked.

`attempt03/result.json` contains the full counts, factors and counterexample.
The original raw public archive is split into three byte segments to respect
the repository's file-size check. Reconstruct it in a separate directory:

```bash
cat raw-public.tar.gz.part* > raw-public.tar.gz
sha256sum raw-public.tar.gz
# 55af84c2272e7f6b0a98fa10ced3ccd0b1f9b068f058c01840f3087a65aa6706
tar -xzf raw-public.tar.gz
```

`raw-public-parts.json` records every segment hash and the joined archive hash.
The public archive contains 208 byte-exact original files, 36 explicitly
derived bootstrap inventories and its manifest. Only six disallowed provider
inventory labels per bootstrap were omitted; timing, errors, source-critical
and native-provider evidence remain intact. Full originals, native binaries
and fixture tensors are preserved in the private archive with SHA256
`e972e9fa9abc02f841867b6e2921e11825c926ff3a337d5efc210ce54c0d7ddb`.
The H100 allocation was released after preservation and download verification.

This is one M144/N4096/K1024 FP8 layer on H100 NVL, using pinned FlashInfer
27c62 source with the FP8 descriptor port and validation callback. It is not
a released-package qualification, a v1/v2 comparison or an end-to-end serving
performance result. The matched-copy synthetic initializer is a controlled
research recipe; activation contents and device history are not fully separable.

`attempt03/independent-audits.tar.gz` and its index preserve independent
reader/statistical code and full reports; the second independent review passed
9,817 checks. `counterexamples.json` and `interpretation-review.json` provide
readable entry points without extracting the larger raw archive.
