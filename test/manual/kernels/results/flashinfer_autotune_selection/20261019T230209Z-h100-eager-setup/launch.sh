#!/usr/bin/env bash
set -euo pipefail
study_dir=$(realpath "${1:?usage: launch.sh STUDY_DIR SOURCE_COMMIT ATTEMPT}")
execution_commit=${2:?exact clean source commit required}
attempt=${3:?unique attempt name required}
[[ "$attempt" =~ ^[a-zA-Z0-9_-]+$ ]]
batch_id=20261019T230209Z-h100-eager-setup
image=lmsysorg/sglang@sha256:c1f4d9fc6fcf50c2269adbc757ce7b1778883af04155edb888fa9eea3b6648c0
source_dir="$study_dir/source"
results_dir="$study_dir/results/$batch_id/$attempt"
active_container=
cleanup() {
    if [[ -n "$active_container" ]]; then
        docker rm -f "$active_container" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$execution_commit" ]]
[[ -z "$(git -C "$source_dir" status --porcelain)" ]]
[[ ! -e "$results_dir" ]]
mkdir -p "$results_dir" "$study_dir/home" "$study_dir/cache"
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
printf '%s\n' "$execution_commit" > "$results_dir/execution-commit.txt"
python3 - "$source_dir" "$results_dir" "$execution_commit" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

source, results = map(Path, sys.argv[1:3])
plan_path = source / 'test/manual/kernels/results/flashinfer_autotune_selection/20261019T230209Z-h100-eager-setup/plan.json'
plan = json.loads(plan_path.read_text())
image = json.loads((results / 'image.json').read_text())
assert plan['container_image'] in image['RepoDigests']
gpu = [{key.strip(): value.strip() for key, value in row.items()}
       for row in csv.DictReader((results / 'gpu-before.csv').open())]
assert len(gpu) == 1
expected = plan['expected_environment']
assert gpu[0]['name'] == expected['gpu']
assert gpu[0]['uuid'].removeprefix('GPU-') == expected['physical_gpu_uuid']
assert gpu[0]['driver_version'] == expected['driver']
files = [
    'test/manual/kernels/flashinfer_autotune_eager_setup.py',
    'test/manual/kernels/flashinfer_autotune_eager_setup_native.cpp',
    'test/manual/kernels/build_flashinfer_autotune_eager_setup_probe.sh',
    'python/sglang/srt/model_executor/runner/flashinfer_autotune.py',
]
hashes = {relative: hashlib.sha256((source / relative).read_bytes()).hexdigest()
          for relative in files}
assert hashes[files[-1]] == expected['sglang_wrapper_sha256']
(results / 'manifest.json').write_text(json.dumps({
    'status': 'success', 'execution_commit': sys.argv[3],
    'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    'source_sha256': hashes, 'gpu': gpu[0],
}, indent=2) + '\n')
PY
docker_args=(
    --rm --gpus all --user "$(id -u):$(id -g)"
    --mount "type=bind,source=$source_dir,target=/workspace,readonly"
    --mount "type=bind,source=$study_dir,target=/study"
    --env HOME=/study/home --env PYTHONNOUSERSITE=1
    --env PYTHONPATH=/workspace/python --env SGLANG_RUST_BUILD_MODE=never
    --env CUDA_VISIBLE_DEVICES=0 --env FLASHINFER_AUTOTUNER_LOAD_FROM_FILE=0
    --env XDG_CACHE_HOME=/study/cache --env CUDA_CACHE_PATH=/study/cache/cuda
    --env TRITON_CACHE_DIR=/study/cache/triton
    --env TORCHINDUCTOR_CACHE_DIR=/study/cache/inductor
    --env "EXPECTED_SOURCE_COMMIT=$execution_commit"
    --workdir /workspace
)
container_results="/study/results/$batch_id/$attempt"
probe_relative="results/$batch_id/$attempt/probe.so"
active_container="fi-eager-${attempt}-build"
if docker run --name "$active_container" "${docker_args[@]}" --entrypoint bash "$image" \
    test/manual/kernels/build_flashinfer_autotune_eager_setup_probe.sh "/study/$probe_relative" \
    > "$results_dir/build.log" 2>&1; then
    printf '0\n' > "$results_dir/build.exit-code"
else
    code=$?
    printf '%s\n' "$code" > "$results_dir/build.exit-code"
    cat "$results_dir/build.log"
    exit "$code"
fi
active_container=

run_worker() {
    local name=$1 mode=$2 seed=$3 samples=$4
    local extra_env=(--env LD_PRELOAD=)
    local extra_args=()
    if [[ "$mode" == observe ]]; then
        extra_env=(--env "LD_PRELOAD=/study/$probe_relative" --env "FI_NATIVE_PROBE_OUTPUT=$container_results/native-events.jsonl")
        extra_args=(--trace "$container_results/torch-trace.json.gz")
    fi
    active_container="fi-eager-${attempt}-${name}"
    if timeout 600 docker run --name "$active_container" "${docker_args[@]}" "${extra_env[@]}" \
        --entrypoint /opt/sglang/bin/python "$image" \
        test/manual/kernels/flashinfer_autotune_eager_setup.py \
        --mode "$mode" --output "$container_results/$name.json" --seed "$seed" \
        --m 128 --n 1024 --k 1024 --samples "$samples" --iterations 20 \
        --warmup-ms 500 --min-sample-ms 20 --observation-calls 5 "${extra_args[@]}" \
        > "$results_dir/$name.log" 2>&1; then
        printf '0\n' > "$results_dir/$name.exit-code"
    else
        code=$?
        printf '%s\n' "$code" > "$results_dir/$name.exit-code"
        cat "$results_dir/$name.log"
        return "$code"
    fi
    active_container=
}
run_worker qualification benchmark 17 2
for seed in 17 18 19; do
    run_worker "benchmark-$seed" benchmark "$seed" 15
done
run_worker observation observe 17 15
nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,power.limit,temperature.gpu,memory.total,memory.used --format=csv > "$results_dir/gpu-after.csv"
printf '0\n' > "$results_dir/launcher.exit-code"
