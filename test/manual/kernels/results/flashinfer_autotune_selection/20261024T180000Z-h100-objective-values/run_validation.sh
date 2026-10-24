#!/usr/bin/env bash
# Separate correctness-only process after the fixed timing cohort has stopped.
set -euo pipefail
study=$(realpath "${1:?study directory required}")
commit=${2:?frozen commit required}
attempt=${3:?existing cohort attempt required}
[[ "$commit" =~ ^[0-9a-f]{40}$ && "$attempt" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 1
batch=test/manual/kernels/results/flashinfer_autotune_selection/20261024T180000Z-h100-objective-values
candidate=test/manual/kernels/results/flashinfer_autotune_selection/20261024T180000Z-harness-validation-fix
source_dir="$study/source"
input_dir="$study/input/current"
runtime="$study/runtime-$attempt"
results="$study/results/validation-fix/$attempt"
cr="/study/results/validation-fix/$attempt"
ct="/study/runtime-$attempt"
name="fi-validation-$attempt"
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$commit" && -z "$(git -C "$source_dir" status --porcelain)" ]] || exit 1
[[ -f "$study/results/objective-values/$attempt/launcher.exit-code" && -d "$input_dir/source/flashinfer/data" && ! -e "$results" ]] || exit 1
mkdir -p "$results/candidate" "$runtime/home/validation"
printf '%s  %s\n' 03860ff06986607d30b8ccfc2daf47e650cf8519a7ffc0ee8ef3cd062f5f16f7 "$source_dir/$candidate/candidate.tar.gz" | sha256sum -c -
tar -xzf "$source_dir/$candidate/candidate.tar.gz" -C "$results/candidate"
cleanup() {
    local code=$? cid actual
    trap - EXIT
    if [[ -f "$results/gate.container-id" ]]; then
        cid=$(cat "$results/gate.container-id")
        if [[ "$cid" =~ ^[0-9a-f]{64}$ ]]; then
            actual=$(docker inspect --format '{{.Name}}' "$cid" 2>/dev/null || true)
            if [[ "$actual" == "/$name" ]]; then docker rm -f "$cid" >/dev/null 2>&1 || true; fi
        fi
    fi
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader > "$results/gpu-processes-after.csv" || code=1
    [[ ! -s "$results/gpu-processes-after.csv" ]] || code=1
    printf '%s\n' "$code" > "$results/launcher.exit-code"
    exit "$code"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
if timeout --kill-after=15s 300 docker run --rm --name "$name" --cidfile "$results/gate.container-id" \
    --gpus all --user "$(id -u):$(id -g)" --network none \
    --mount "type=bind,source=$source_dir,target=/workspace,readonly" \
    --mount "type=bind,source=$input_dir,target=/study/input/current,readonly" \
    --mount "type=bind,source=$results,target=$cr" --mount "type=bind,source=$runtime,target=$ct" \
    --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 --env SGLANG_RUST_BUILD_MODE=never \
    --env CUDA_VISIBLE_DEVICES=0 --env FLASHINFER_AUTOTUNER_LOAD_FROM_FILE=0 \
    --env FLASHINFER_CUDA_ARCH_LIST=9.0 --env MAX_JOBS=4 --env LD_PRELOAD= \
    --env "HOME=$ct/home/validation" --env "PYTHONPATH=/study/input/current/tools:/workspace/python" \
    --env "FLASHINFER_WORKSPACE_BASE=$ct/cache/current" --env "XDG_CACHE_HOME=$ct/cache/xdg" \
    --env "CUDA_CACHE_PATH=$ct/cache/cuda" --env "TRITON_CACHE_DIR=$ct/cache/triton" \
    --workdir /workspace --entrypoint /opt/sglang/bin/python \
    lmsysorg/sglang@sha256:c1f4d9fc6fcf50c2269adbc757ce7b1778883af04155edb888fa9eea3b6648c0 \
    "/workspace/$batch/run_current.py" --source /study/input/current/source \
    --manifest /study/input/current/manifests/candidate-manifest.json \
    --target "/workspace/$candidate/run_cuda_tests.py" --output "$cr/gate.json" -- \
    --candidate "$cr/candidate" --output "$cr/gate.json" > "$results/gate.container.log" 2>&1; then
    printf '0\n' > "$results/gate.exit-code"
else
    code=$?; printf '%s\n' "$code" > "$results/gate.exit-code"
    tail -60 "$results/gate.container.log"
    exit "$code"
fi
python3 - "$results" <<'PY'
import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]);g=json.loads((r/'gate.json').read_text());b=json.loads((r/'gate.bootstrap.json').read_text())
assert g['status']=='passed' and g['counts']=={'passed':28,'failed':0,'skipped':0}
assert g['tuner_and_inference_state_restored'] and b['status']=='passed' and b['native_identity_passthrough']
assert b['result_sha256']==hashlib.sha256((r/'gate.json').read_bytes()).hexdigest()
print(json.dumps({'status':'passed','counts':g['counts']}))
PY
