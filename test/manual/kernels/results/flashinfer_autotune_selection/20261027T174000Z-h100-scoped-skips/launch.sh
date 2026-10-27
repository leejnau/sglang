#!/usr/bin/env bash
# One import preflight, then one actual skipped startup and graph capture.
set -euo pipefail
case "$(hostname -s)" in 1u1g-gen-0100|a1u1g-mil-0836|2u2g-gen-0664) ;; *) exit 1 ;; esac
study=$(realpath "${1:?study directory required}")
commit=${2:?frozen commit required}
phase=${3:?preflight or cold required}
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] || exit 1
[[ "$phase" == preflight || "$phase" == cold ]] || exit 1
batch=test/manual/kernels/results/flashinfer_autotune_selection/20261027T174000Z-h100-scoped-skips
prior=test/manual/kernels/results/flashinfer_autotune_selection/20261027T020000Z-h100-mlp-confirmation
source_dir="$study/source"
results="$study/results/scoped-cache/attempt01"
runtime="$study/runtime-attempt01"
cr=/study/results/scoped-cache/attempt01
ct=/study/runtime-attempt01
image=lmsysorg/sglang@sha256:c1f4d9fc6fcf50c2269adbc757ce7b1778883af04155edb888fa9eea3b6648c0
active_name="fi-scoped-cache-$(basename "$study")-$phase"
active_cid="$results/$phase.container-id"
created=0
started=$SECONDS
cleanup() {
    local code=$? cid actual
    trap - EXIT
    if [[ -f "$active_cid" ]]; then
        cid=$(cat "$active_cid")
        if [[ "$cid" =~ ^[0-9a-f]{64}$ ]]; then
            actual=$(docker inspect --format '{{.Name}}' "$cid" 2>/dev/null || true)
            if [[ "$actual" == "/$active_name" ]]; then docker rm -f "$cid" >/dev/null 2>&1 || true; fi
        fi
    fi
    if [[ "$created" == 1 ]]; then
        mkdir -p "$results/native-build-cache" "$results/tuner-cache" "$results/managed-cache" "$results/negative-cache"
        # Preserve exactly the two controlled providers; other JIT artifacts have
        # a complete hash inventory, without copying unrelated cache binaries.
        python3 "$source_dir/$batch/outer_audit.py" --snapshot "$study" --phase "$phase" || code=1
        if [[ -d "$runtime/sglang-cache" ]]; then cp -a "$runtime/sglang-cache" "$results/tuner-cache/$phase" || code=1; fi
        if [[ -d "$runtime/managed-cache" ]]; then cp -a "$runtime/managed-cache" "$results/managed-cache/$phase" || code=1; fi
        if [[ -d "$runtime/negative-$phase" ]]; then cp -a "$runtime/negative-$phase" "$results/negative-cache/$phase" || code=1; fi
        git -C "$source_dir" status --porcelain > "$results/sglang-status-after-$phase.txt" || code=1
        [[ -z "$(cat "$results/sglang-status-after-$phase.txt")" ]] || code=1
        nvidia-smi -i 0 --query-gpu=name,uuid,driver_version,power.limit,memory.used --format=csv > "$results/gpu-after-$phase.csv" || code=1
        nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader > "$results/gpu-processes-after-$phase.csv" || code=1
        [[ ! -s "$results/gpu-processes-after-$phase.csv" ]] || code=1
        printf '%s\n' "$((SECONDS-started))" > "$results/elapsed-seconds-$phase.txt"
        printf '%s\n' "$code" > "$results/launcher-$phase.exit-code"
    fi
    exit "$code"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$commit" && -z "$(git -C "$source_dir" status --porcelain)" ]] || exit 1
if [[ "$phase" == preflight ]]; then
    [[ ! -e "$results" && ! -e "$runtime/sglang-cache" && ! -e "$runtime/managed-cache" ]] || exit 1
    mkdir -p "$results/native-build-cache"
    cp "$study/native-reference.bootstrap.json" "$study/source-snapshot.json" "$study/model-manifest.json" "$results/"
    cp "$study/native-reuse.json" "$results/native-build-cache/reuse-manifest.json"
    cp "$study/native-restoration.json" "$results/native-build-cache/restoration.json"
    cp "$study/native-preservation.json" "$results/native-build-cache/prior-preservation.json"
    cp "$study/model-stream-plan.json" "$results/native-build-cache/model-stream-plan.json"
elif [[ "$phase" == cold ]]; then
    [[ "$(cat "$results/launcher-preflight.exit-code")" == 0 && ! -e "$runtime/sglang-cache" && ! -e "$runtime/managed-cache" ]] || exit 1

fi
[[ ! -e "$results/$phase.json" && ! -e "$active_cid" && ! -e "$runtime/negative-$phase" ]] || exit 1
mkdir -p "$runtime/home/$phase"
created=1
printf '%s\n' "$commit" > "$results/execution-commit.txt"
docker image inspect "$image" | python3 -c 'import json,sys;x=json.load(sys.stdin)[0];print(json.dumps({k:x[k] for k in ("Id","RepoDigests","Created","Architecture","Os","Size")},indent=2))' > "$results/image-$phase.json"
nvidia-smi -i 0 --query-gpu=name,uuid,driver_version,power.limit,memory.used --format=csv > "$results/gpu-before-$phase.csv"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader > "$results/gpu-processes-before-$phase.csv"
[[ ! -s "$results/gpu-processes-before-$phase.csv" ]] || exit 1
python3 "$source_dir/$batch/make_provenance.py" --study "$study" --commit "$commit" --phase "$phase"
python3 "$source_dir/$batch/outer_audit.py" --cache-before "$study" --phase "$phase"
worker=/workspace/test/manual/kernels/audit_flashinfer_scoped_skip.py
worker_args=(--phase "$phase" --model-path /model --cache-dir "$ct/sglang-cache" --managed-root "$ct/managed-cache" --provenance "$cr/provenance.json" --output "$cr/$phase.json")
limit=720
sglang_cache="$ct/sglang-cache"
if [[ "$phase" == preflight ]]; then
    worker="/workspace/test/manual/kernels/results/flashinfer_autotune_selection/20261027T160000Z-h100-scoped-cache/import_preflight.py"
    worker_args=(--provenance "$cr/provenance.json" --output "$cr/$phase.json")
    limit=300
    sglang_cache="$ct/preflight-sglang-cache"
fi
if timeout --kill-after=15s "$limit" docker run --rm --name "$active_name" --cidfile "$active_cid" \
    --gpus device=0 --user "$(id -u):$(id -g)" --network none --shm-size=8g \
    --mount "type=bind,source=$source_dir,target=/workspace,readonly" \
    --mount "type=bind,source=$study/input/current,target=/study/input/current,readonly" \
    --mount "type=bind,source=$study/model,target=/model,readonly" \
    --mount "type=bind,source=$results,target=$cr" --mount "type=bind,source=$runtime,target=$ct" \
    --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 --env SGLANG_RUST_BUILD_MODE=never \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env CUDA_VISIBLE_DEVICES=0 --env FLASHINFER_AUTOTUNER_LOAD_FROM_FILE=0 \
    --env FLASHINFER_CUDA_ARCH_LIST=9.0 --env MAX_JOBS=4 --env LD_PRELOAD= \
    --env "HOME=$ct/home/$phase" --env "PYTHONPATH=/study/input/current/tools:/workspace/python:/workspace/test/manual/kernels" \
    --env "FLASHINFER_WORKSPACE_BASE=$ct/cache/current" --env "XDG_CACHE_HOME=$ct/cache/xdg" \
    --env "CUDA_CACHE_PATH=$ct/cache/cuda" --env "TRITON_CACHE_DIR=$ct/cache/triton" --env "TORCHINDUCTOR_CACHE_DIR=$ct/cache/inductor" \
    --env "SGLANG_CACHE_DIR=$sglang_cache" \
    --workdir /workspace --entrypoint /opt/sglang/bin/python "$image" \
    "/workspace/$prior/run_current.py" --source /study/input/current/source \
    --manifest /study/input/current/manifests/candidate-manifest.json \
    --target "$worker" --output "$cr/$phase.json" \
    --native-reference "$cr/native-reference.bootstrap.json" --reuse-manifest "$cr/native-build-cache/reuse-manifest.json" -- \
    "${worker_args[@]}" > "$results/$phase.log" 2>&1; then
    printf '0\n' > "$results/$phase.exit-code"
else
    code=$?; printf '%s\n' "$code" > "$results/$phase.exit-code"
    tail -60 "$results/$phase.log"
    exit "$code"
fi
python3 - "$results/$phase.json" <<'CHECK'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);result=json.loads(p.read_text());bootstrap=json.loads(p.with_suffix('.bootstrap.json').read_text())
assert result['status']=='completed'
assert bootstrap['status']=='passed' and bootstrap['result_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
CHECK
printf 'Completed %s; analysis is separate.\n' "$phase"
