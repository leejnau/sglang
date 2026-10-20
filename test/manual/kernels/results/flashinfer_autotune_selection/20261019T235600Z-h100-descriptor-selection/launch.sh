#!/usr/bin/env bash
set -euo pipefail
study_dir=$(realpath "${1:?usage: launch.sh STUDY_DIR SOURCE_COMMIT ATTEMPT}")
source_commit=${2:?frozen source commit required}
attempt=${3:?unique attempt required}
[[ "$attempt" =~ ^[a-zA-Z0-9_-]+$ ]]
batch_id=20261019T235600Z-h100-descriptor-selection
image=lmsysorg/sglang@sha256:c1f4d9fc6fcf50c2269adbc757ce7b1778883af04155edb888fa9eea3b6648c0
relative_batch="test/manual/kernels/results/flashinfer_autotune_selection/$batch_id"
input_batch=test/manual/kernels/results/flashinfer_autotune_selection/20261019T232945Z-h100-descriptor-qualification
source_dir="$study_dir/source"
results_dir="$study_dir/results/$batch_id/$attempt"
runtime_name="runtime-$attempt"
runtime_dir="$study_dir/$runtime_name"
container_results="/study/results/$batch_id/$attempt"
active_container=
cleanup() {
    local code=$?
    if [[ -n "$active_container" ]]; then docker rm -f "$active_container" >/dev/null 2>&1 || true; fi
    if [[ -d "$results_dir" ]]; then printf '%s\n' "$code" > "$results_dir/launcher.exit-code"; fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$source_commit" ]]
[[ -z "$(git -C "$source_dir" status --porcelain)" ]]
[[ ! -e "$results_dir" && ! -e "$runtime_dir" ]]
mkdir -p "$results_dir" "$runtime_dir/home" "$runtime_dir/caches"
printf '%s\n' "$source_commit" > "$results_dir/execution-commit.txt"
docker image inspect "$image" | python3 -c '
import json,sys
record=json.load(sys.stdin)[0]
assert record["Architecture"] == "amd64"
print(json.dumps({key:record[key] for key in ("Id","RepoDigests","Created","Architecture","Os","Size")},indent=2))
' > "$results_dir/image.json"
nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,power.limit,temperature.gpu,memory.total,memory.used --format=csv > "$results_dir/gpu-before.csv"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader > "$results_dir/gpu-processes-before.csv"
[[ ! -s "$results_dir/gpu-processes-before.csv" ]]
lscpu -J > "$results_dir/host_cpu.json"
python3 - "$source_dir" "$results_dir" "$relative_batch" "$input_batch" <<'PY'
import csv,hashlib,json,sys
from pathlib import Path
source,results=map(Path,sys.argv[1:3]);batch=source/sys.argv[3];inputs=source/sys.argv[4]
plan=json.loads((batch/'plan.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(inputs/'source-overlay.tar.gz')==plan['flashinfer_source']['source_overlay_archive_sha256']
assert sha(inputs/'source-manifest.json')==plan['flashinfer_source']['source_manifest_sha256']
gpu=[{k.strip():v.strip() for k,v in row.items()} for row in csv.DictReader((results/'gpu-before.csv').open())]
assert len(gpu)==1
expected=plan['environment'];assert gpu[0]['name']==expected['gpu']
assert gpu[0]['uuid'].removeprefix('GPU-')==expected['physical_gpu_uuid'] and gpu[0]['driver_version']==expected['driver']
paths=['test/manual/kernels/flashinfer_autotune_descriptor_selection.py','test/manual/kernels/flashinfer_autotune_descriptor_qualification.py','test/manual/kernels/flashinfer_autotune_eager_setup.py','test/manual/kernels/flashinfer_autotune_selection_worker.py','python/sglang/srt/model_executor/runner/flashinfer_autotune.py',sys.argv[3]+'/analyze.py',sys.argv[4]+'/prepare_packages.py']
hashes={path:sha(source/path) for path in paths};assert hashes[paths[4]]==expected['sglang_wrapper_sha256']
(results/'manifest.json').write_text(json.dumps({'plan_sha256':sha(batch/'plan.json'),'source_manifest_sha256':sha(inputs/'source-manifest.json'),'source_sha256':hashes,'gpu':gpu[0],'implementation':'candidate'},indent=2)+'\n')
PY
docker_args=(
    --rm --gpus all --user "$(id -u):$(id -g)"
    --mount "type=bind,source=$source_dir,target=/workspace,readonly"
    --mount "type=bind,source=$study_dir,target=/study"
    --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1
    --env SGLANG_RUST_BUILD_MODE=never --env CUDA_VISIBLE_DEVICES=0
    --env FLASHINFER_AUTOTUNER_LOAD_FROM_FILE=0 --env SGLANG_FLASHINFER_AUTOTUNE_CACHE=1
    --env FLASHINFER_CUDA_ARCH_LIST=9.0 --env MAX_JOBS=4 --env LD_PRELOAD=
    --env "EXPECTED_SOURCE_COMMIT=$source_commit" --workdir /workspace
)
active_container="fi-selection-${attempt}-prepare"
if timeout 300 docker run --name "$active_container" "${docker_args[@]}" \
    --env "HOME=/study/$runtime_name/home" --env PYTHONPATH=/workspace/python \
    --entrypoint /opt/sglang/bin/python "$image" \
    "$input_batch/prepare_packages.py" "/study/$runtime_name" "/workspace/$input_batch" \
    > "$results_dir/prepare.log" 2>&1; then
    printf '0\n' > "$results_dir/prepare.exit-code"
else
    code=$?; printf '%s\n' "$code" > "$results_dir/prepare.exit-code"
    cat "$results_dir/prepare.log"; exit "$code"
fi
active_container=
cp "$runtime_dir/prepared-packages.json" "$results_dir/prepared-packages.json"
run_worker() {
    local m=$1 seed=$2 arm=$3 phase=$4
    local name="m${m}-seed${seed}-${arm}-${phase}"
    mkdir -p "$runtime_dir/home/$name" "$runtime_dir/cache/candidate" "$runtime_dir/caches/m$m/seed$seed"
    active_container="fi-selection-${attempt}-${name}"
    if timeout 900 docker run --name "$active_container" --cidfile "$results_dir/$name.container-id" \
        "${docker_args[@]}" --env "HOME=/study/$runtime_name/home/$name" \
        --env "PYTHONPATH=/study/$runtime_name/packages/candidate:/workspace/python:/workspace/test/manual/kernels" \
        --env "FLASHINFER_WORKSPACE_BASE=/study/$runtime_name/cache/candidate" \
        --env "XDG_CACHE_HOME=/study/$runtime_name/cache/candidate/xdg" \
        --env "CUDA_CACHE_PATH=/study/$runtime_name/cache/candidate/cuda" \
        --env "TRITON_CACHE_DIR=/study/$runtime_name/cache/candidate/triton" \
        --env "TORCHINDUCTOR_CACHE_DIR=/study/$runtime_name/cache/candidate/inductor" \
        --entrypoint /opt/sglang/bin/python "$image" \
        test/manual/kernels/flashinfer_autotune_descriptor_selection.py \
        --arm "$arm" --phase "$phase" --m "$m" --n 1024 --k 1024 --seed "$seed" \
        --cache "/study/$runtime_name/caches/m$m/seed$seed/$arm.json" \
        --manifest "/study/$runtime_name/source-overlay/source-manifest.json" \
        --output "$container_results/$name.json" --oracle \
        --samples 15 --iterations 20 --warmup-ms 500 --min-sample-ms 20 \
        > "$results_dir/$name.log" 2>&1; then
        printf '0\n' > "$results_dir/$name.exit-code"
    else
        code=$?; printf '%s\n' "$code" > "$results_dir/$name.exit-code"
        cat "$results_dir/$name.log"; return "$code"
    fi
    active_container=
}
for m in 128 144; do
    for seed in 17 18 19; do
        order=(default v1 v2)
        if [[ "$seed" == 18 ]]; then order=(v1 v2 default); fi
        if [[ "$seed" == 19 ]]; then order=(v2 default v1); fi
        for arm in "${order[@]}"; do
            for phase in cold warm; do run_worker "$m" "$seed" "$arm" "$phase"; done
        done
    done
    mkdir -p "$results_dir/caches"
    cp -a "$runtime_dir/caches/." "$results_dir/caches/"
    nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,power.limit,temperature.gpu,memory.total,memory.used --format=csv > "$results_dir/gpu-after-m$m.csv"
    printf '0\n' > "$results_dir/m$m.launcher.exit-code"
    active_container="fi-selection-${attempt}-analyze-m$m"
    if docker run --name "$active_container" "${docker_args[@]}" \
        --env "HOME=/study/$runtime_name/home" --env PYTHONPATH= \
        --entrypoint /opt/sglang/bin/python "$image" "$relative_batch/analyze.py" \
        --raw-root "$container_results" --source-root /workspace \
        --expected-source-commit "$source_commit" \
        --source-manifest "/workspace/$input_batch/source-manifest.json" \
        --m "$m" --output "$container_results/checkpoint-m$m-analysis.json" \
        > "$results_dir/checkpoint-m$m-analysis.log" 2>&1; then
        printf '0\n' > "$results_dir/checkpoint-m$m-analysis.exit-code"
    else
        code=$?; printf '%s\n' "$code" > "$results_dir/checkpoint-m$m-analysis.exit-code"
        cat "$results_dir/checkpoint-m$m-analysis.log"; return_code=$code
        exit "$return_code"
    fi
    active_container=
done
