# Owned-output validation candidate

This is a separate FlashInfer benchmark/test candidate. It does not change the qualified experiment runtime or open a PR. `candidate.patch` applies to the recorded 27c62 baseline and to the reviewed current head because both touched files are unchanged. The archive contains exactly those two candidate files plus the gate wrapper; it contains no FlashInfer provider package.

Extract `candidate.tar.gz` into a new directory. After the existing qualified current-source bootstrap, execute the loose `run_cuda_tests.py` in the same Python process with `--candidate <new-directory> --output <new-result.json>`. Keep the candidate directory out of `sys.path`; the real runtime continues to come from the qualified provider. `run_current.py --target` can perform this handoff. Use a separate `.container.log` for container stdout because the gate reserves its own result-adjacent `.log`. Expected real-GPU result: 28 passed, 0 skipped.

The CPU record is 16 passed and 12 CUDA-only skipped. The separate 28-pass graph/event boundary admission is simulated CUDA and is explicitly not a device result. Two red controls demonstrate the missing-callback and inference-tensor regressions. The retained CPU script uses the earlier disclosed boundary loader and pinned 27c62 source; that environment is required to replay it. Standard upstream pytest on CUDA uses the real library and no boundary loader.

The tar and patch are source artifacts only. `cpu-validation.json` records scope, hashes, controls and pending CUDA evidence. The output refresh occurs after timing and before final independent replays, so this fix makes no performance claim.

Actual H100 validation now passes: **28 passed, 0 skipped**, using source freeze `c58cd3e0c149b6dcd12b2bf9bc0cca1172547664`. The separate fresh container used real CUDA allocation, capture and replay with controlled profiling durations. Results, test outcomes and native-provider provenance are retained in `attempt01/`; no selection or performance claim follows.


The 28 passing nodes comprise 12 real CUDA tuner/cache/graph cases and 16 CPU
ownership/view/plumbing controls. Profiling durations are controlled and the
runners use `torch.add`; this is validation coverage, not FP8 performance or
all-backend qualification. Before an upstream PR, refresh the duplicate/rebase
check, settle the optional callback and declared-output contract, and run the
applicable checks against the actual target main. No PR has been opened.
