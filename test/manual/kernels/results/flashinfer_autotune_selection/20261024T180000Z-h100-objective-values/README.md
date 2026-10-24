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
