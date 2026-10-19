#!/usr/bin/env bash
set -euo pipefail
study_dir=$(realpath "${1:?usage: launch.sh STUDY_DIR SOURCE_COMMIT ATTEMPT}")
source_commit=${2:?frozen source commit required}
attempt=${3:?unique attempt required}
[[ "$attempt" =~ ^[a-zA-Z0-9_-]+$ ]]
batch_id=20261019T232945Z-h100-descriptor-qualification
image=lmsysorg/sglang@sha256:c1f4d9fc6fcf50c2269adbc757ce7b1778883af04155edb888fa9eea3b6648c0
relative_batch="test/manual/kernels/results/flashinfer_autotune_selection/$batch_id"
source_dir="$study_dir/source"
results_dir="$study_dir/results/$batch_id/$attempt"
runtime_name="runtime-$attempt"
runtime_dir="$study_dir/$runtime_name"
container_results="/study/results/$batch_id/$attempt"
active_container=
cleanup() {
    if [[ -n "$active_container" ]]; then
        docker rm -f "$active_container" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$source_commit" ]]
[[ -z "$(git -C "$source_dir" status --porcelain)" ]]
[[ ! -e "$results_dir" && ! -e "$runtime_dir" ]]
mkdir -p "$results_dir" "$runtime_dir" "$runtime_dir/home" "$runtime_dir/caches"
printf '%s\n' "$source_commit" > "$results_dir/execution-commit.txt"
docker image inspect "$image" | python3 -c '
import json,sys
d=json.load(sys.stdin)[0]
assert d["Architecture"] == "amd64"
print(json.dumps({k:d[k] for k in ("Id","RepoDigests","Created","Architecture","Os","Size")},indent=2))
' > "$results_dir/image.json"
nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,power.limit,temperature.gpu,memory.total,memory.used --format=csv > "$results_dir/gpu-before.csv"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader > "$results_dir/gpu-processes-before.csv"
[[ ! -s "$results_dir/gpu-processes-before.csv" ]]
lscpu -J > "$results_dir/host_cpu.json"
python3 - "$source_dir" "$results_dir" "$relative_batch" <<'PY'
import csv,hashlib,json,sys
from pathlib import Path
source,results=map(Path,sys.argv[1:3]);batch=source/sys.argv[3]
plan=json.loads((batch/'plan.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,key in [('source-overlay.tar.gz','source_overlay_archive_sha256'),('source-manifest.json','source_manifest_sha256'),('upstream-test-support.tar.gz','test_support_archive_sha256')]:
    assert sha(batch/name)==plan['source_arms'][key],name
gpu=[{k.strip():v.strip() for k,v in r.items()} for r in csv.DictReader((results/'gpu-before.csv').open())]
assert len(gpu)==1
expected=plan['environment']
assert gpu[0]['name']==expected['gpu'] and gpu[0]['uuid'].removeprefix('GPU-')==expected['physical_gpu_uuid']
assert gpu[0]['driver_version']==expected['driver']
paths=['test/manual/kernels/flashinfer_autotune_descriptor_qualification.py','test/manual/kernels/flashinfer_autotune_descriptor_native.cpp','test/manual/kernels/build_flashinfer_autotune_descriptor_probe.sh','test/manual/kernels/flashinfer_autotune_eager_setup.py','python/sglang/srt/model_executor/runner/flashinfer_autotune.py']
hashes={p:sha(source/p) for p in paths}
assert hashes[paths[-1]]==expected['sglang_wrapper_sha256']
(results/'manifest.json').write_text(json.dumps({'plan_sha256':sha(batch/'plan.json'),'source_manifest_sha256':sha(batch/'source-manifest.json'),'source_sha256':hashes,'gpu':gpu[0]},indent=2)+'\n')
PY
docker_args=(
    --rm --gpus all --user "$(id -u):$(id -g)"
    --mount "type=bind,source=$source_dir,target=/workspace,readonly"
    --mount "type=bind,source=$study_dir,target=/study"
    --env PYTHONNOUSERSITE=1 --env SGLANG_RUST_BUILD_MODE=never
    --env CUDA_VISIBLE_DEVICES=0 --env FLASHINFER_AUTOTUNER_LOAD_FROM_FILE=0
    --env FLASHINFER_CUDA_ARCH_LIST=9.0 --env MAX_JOBS=4
    --env "EXPECTED_SOURCE_COMMIT=$source_commit" --workdir /workspace
)
active_container="fi-descriptor-${attempt}-prepare"
if timeout 300 docker run --name "$active_container" "${docker_args[@]}" \
    --env "HOME=/study/$runtime_name/home" --env PYTHONPATH=/workspace/python \
    --entrypoint /opt/sglang/bin/python "$image" \
    "$relative_batch/prepare_packages.py" "/study/$runtime_name" "/workspace/$relative_batch" \
    > "$results_dir/prepare.log" 2>&1; then
    printf '0\n' > "$results_dir/prepare.exit-code"
else
    code=$?; printf '%s\n' "$code" > "$results_dir/prepare.exit-code"
    cat "$results_dir/prepare.log"; exit "$code"
fi
active_container=
cp "$runtime_dir/prepared-packages.json" "$results_dir/prepared-packages.json"
active_container="fi-descriptor-${attempt}-probe-build"
if docker run --name "$active_container" "${docker_args[@]}" --entrypoint bash "$image" \
    test/manual/kernels/build_flashinfer_autotune_descriptor_probe.sh "$container_results/probe.so" \
    > "$results_dir/build.log" 2>&1; then
    printf '0\n' > "$results_dir/build.exit-code"
else
    code=$?; printf '%s\n' "$code" > "$results_dir/build.exit-code"
    cat "$results_dir/build.log"; exit "$code"
fi
active_container=

run_worker() {
    local name=$1 arm=$2 mode=$3 seed=$4 samples=$5 version=$6
    local env_args=(--env LD_PRELOAD=)
    local worker_args=()
    if [[ "$arm" == candidate ]]; then
        worker_args+=(--match "$container_results/qualification-baseline-$seed.json")
    fi
    if [[ "$mode" == observe ]]; then
        env_args=(--env "LD_PRELOAD=$container_results/probe.so" --env "FI_NATIVE_PROBE_OUTPUT=$container_results/native-$arm.jsonl")
        worker_args+=(--trace "$container_results/trace-$arm.json.gz")
    fi
    if [[ "$mode" == lifecycle-* ]]; then
        worker_args+=(--cache "/study/$runtime_name/caches/$arm-$version.json")
    fi
    mkdir -p "$runtime_dir/home/$arm" "$runtime_dir/cache/$arm"
    active_container="fi-descriptor-${attempt}-${name}"
    if timeout 900 docker run --name "$active_container" --cidfile "$results_dir/$name.container-id" \
        "${docker_args[@]}" "${env_args[@]}" \
        --env "HOME=/study/$runtime_name/home/$arm" \
        --env "PYTHONPATH=/study/$runtime_name/packages/$arm:/workspace/python" \
        --env "FLASHINFER_WORKSPACE_BASE=/study/$runtime_name/cache/$arm" \
        --env "XDG_CACHE_HOME=/study/$runtime_name/cache/$arm/xdg" \
        --env "CUDA_CACHE_PATH=/study/$runtime_name/cache/$arm/cuda" \
        --env "TRITON_CACHE_DIR=/study/$runtime_name/cache/$arm/triton" \
        --env "TORCHINDUCTOR_CACHE_DIR=/study/$runtime_name/cache/$arm/inductor" \
        --entrypoint /opt/sglang/bin/python "$image" \
        test/manual/kernels/flashinfer_autotune_descriptor_qualification.py \
        --mode "$mode" --implementation "$arm" --seed "$seed" \
        --manifest "/study/$runtime_name/source-overlay/source-manifest.json" \
        --output "$container_results/$name.json" --tuning-version "$version" \
        --samples "$samples" --iterations 20 --warmup-ms 500 --min-sample-ms 20 \
        --observation-calls 5 "${worker_args[@]}" > "$results_dir/$name.log" 2>&1; then
        printf '0\n' > "$results_dir/$name.exit-code"
    else
        code=$?; printf '%s\n' "$code" > "$results_dir/$name.exit-code"
        cat "$results_dir/$name.log"; return "$code"
    fi
    active_container=
}
for seed in 17 18 19; do run_worker "qualification-baseline-$seed" baseline benchmark "$seed" 2 v2; done
run_worker qualification-candidate-17 candidate benchmark 17 2 v2
for seed in 17 18 19; do
    order=(baseline candidate)
    if [[ "$seed" == 18 ]]; then order=(candidate baseline); fi
    for arm in "${order[@]}"; do run_worker "benchmark-$arm-$seed" "$arm" benchmark "$seed" 15 v2; done
done
for arm in baseline candidate; do run_worker "observation-$arm" "$arm" observe 17 15 v2; done
for pair in baseline-v2 candidate-v1 candidate-v2; do
    arm=${pair%-*}; version=${pair##*-}
    for phase in cold warm; do run_worker "lifecycle-$pair-$phase" "$arm" "lifecycle-$phase" 17 15 "$version"; done
done
cp -a "$runtime_dir/caches" "$results_dir/caches"
active_container="fi-descriptor-${attempt}-upstream-tests"
if timeout 900 docker run --name "$active_container" --cidfile "$results_dir/upstream-tests.container-id" \
    "${docker_args[@]}" --env LD_PRELOAD= \
    --env "HOME=/study/$runtime_name/home/candidate" \
    --env "PYTHONPATH=/study/$runtime_name/packages/candidate:/workspace/python:/workspace/test/manual/kernels" \
    --env "FLASHINFER_WORKSPACE_BASE=/study/$runtime_name/cache/candidate" \
    --env "XDG_CACHE_HOME=/study/$runtime_name/cache/candidate/xdg" \
    --entrypoint /opt/sglang/bin/python "$image" "$relative_batch/run_upstream_tests.py" \
    --test-root "/study/$runtime_name/upstream-tests" \
    --manifest "/study/$runtime_name/source-overlay/source-manifest.json" \
    --output "$container_results/upstream-tests.json" > "$results_dir/upstream-tests.log" 2>&1; then
    printf '0\n' > "$results_dir/upstream-tests.exit-code"
else
    code=$?; printf '%s\n' "$code" > "$results_dir/upstream-tests.exit-code"
    cat "$results_dir/upstream-tests.log"; exit "$code"
fi
active_container=
nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,power.limit,temperature.gpu,memory.total,memory.used --format=csv > "$results_dir/gpu-after.csv"
printf '0\n' > "$results_dir/launcher.exit-code"
