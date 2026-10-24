#!/usr/bin/env bash
# One bounded, fixed-order cohort. Stop on integrity failure; preserve every attempt.
set -euo pipefail
study=$(realpath "${1:?study directory required}")
commit=${2:?frozen commit required}
attempt=${3:?unique attempt required}
[[ "$commit" =~ ^[0-9a-f]{40}$ && "$attempt" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 1
batch=test/manual/kernels/results/flashinfer_autotune_selection/20261024T180000Z-h100-objective-values
source_dir="$study/source"
input_dir="$study/input/current"
results="$study/results/objective-values/$attempt"
runtime="$study/runtime-$attempt"
cr="/study/results/objective-values/$attempt"
ct="/study/runtime-$attempt"
image=lmsysorg/sglang@sha256:c1f4d9fc6fcf50c2269adbc757ce7b1778883af04155edb888fa9eea3b6648c0
active_cid=
active_name=
created=0
started=$SECONDS
cleanup() {
    local code=$? cid actual
    trap - EXIT
    if [[ -n "$active_cid" && -f "$active_cid" ]]; then
        cid=$(cat "$active_cid")
        if [[ "$cid" =~ ^[0-9a-f]{64}$ ]]; then
            actual=$(docker inspect --format '{{.Name}}' "$cid" 2>/dev/null || true)
            if [[ "$actual" == "/$active_name" ]]; then docker rm -f "$cid" >/dev/null 2>&1 || true; fi
        fi
    fi
    if [[ "$created" == 1 ]]; then
        if [[ -d "$runtime/cache" ]]; then cp -a "$runtime/cache" "$results/native-build-cache" || code=1; fi
        git -C "$source_dir" rev-parse HEAD > "$results/sglang-commit-after.txt" || code=1
        git -C "$source_dir" status --porcelain > "$results/sglang-status-after.txt" || code=1
        [[ "$(cat "$results/sglang-commit-after.txt")" == "$commit" && ! -s "$results/sglang-status-after.txt" ]] || code=1
        nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,power.limit,temperature.gpu,memory.total,memory.used --format=csv > "$results/gpu-after.csv" || code=1
        nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader > "$results/gpu-processes-after.csv" || code=1
        [[ ! -s "$results/gpu-processes-after.csv" ]] || code=1
        printf '%s\n' "$((SECONDS-started))" > "$results/elapsed-seconds.txt"
        printf '%s\n' "$code" > "$results/launcher.exit-code"
    fi
    exit "$code"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$commit" && -z "$(git -C "$source_dir" status --porcelain)" ]] || exit 1
[[ ! -e "$results" && ! -e "$runtime" ]] || exit 1
mkdir -p "$results"
created=1
mkdir -p "$runtime/home" "$runtime/cache/current"
cp -a "$study/fixture" "$results/fixture"
cp "$study/source-snapshot.json" "$results/source-snapshot.json"
printf '%s\n' "$commit" > "$results/sglang-commit-before.txt"
docker image inspect "$image" | python3 -c 'import json,sys;x=json.load(sys.stdin)[0];print(json.dumps({k:x[k] for k in ("Id","RepoDigests","Created","Architecture","Os","Size")},indent=2))' > "$results/image.json"
nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,power.limit,temperature.gpu,memory.total,memory.used --format=csv > "$results/gpu-before.csv"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader > "$results/gpu-processes-before.csv"
[[ ! -s "$results/gpu-processes-before.csv" ]] || exit 1
lscpu -J > "$results/host-cpu.json"
python3 - "$source_dir" "$input_dir" "$results" "$commit" "$batch" <<'PY'
import csv,hashlib,json,sys
from pathlib import Path
s,i,r=map(Path,sys.argv[1:4]);commit,batch=sys.argv[4:]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((s/batch/'plan.json').read_text())
gpu=[{k.strip():v.strip() for k,v in row.items()} for row in csv.DictReader((r/'gpu-before.csv').open())]
assert len(gpu)==1;g=gpu[0]
assert g['name']=='NVIDIA H100 NVL' and g['uuid']=='GPU-240958a2-e8e1-3bac-b767-365c9e0b97bd'
assert g['driver_version']=='580.159.04' and float(g['power.limit [W]'].split()[0])==310 and int(g['memory.used [MiB]'].split()[0])==0
image=json.loads((r/'image.json').read_text())
assert image['Architecture']=='amd64' and image['Os']=='linux' and plan['environment']['image'] in image['RepoDigests']
assert image['Id']=='sha256:2e152727f0a5ce999172ec48203b2b292d41b3cb370976827718870557899caa'
fixture=json.loads((r/'fixture/manifest.json').read_text())
paths={str(p.relative_to(s)) for p in (s/'test/manual/kernels').glob('*.py')}
paths.update(str(p.relative_to(s)) for p in (s/batch).glob('*') if p.suffix in ('.py','.sh','.json'))
paths.update(fixture['source']['sglang_source_sha256'])
paths.add('python/sglang/srt/model_executor/runner/flashinfer_autotune.py')
assert not (i/'source/flashinfer/data').exists()
manifest=dict(schema_version=1,execution_commit=commit,plan_sha256=sha(s/batch/'plan.json'),schedule_sha256=sha(s/batch/'schedule.json'),source_sha256={p:sha(s/p) for p in sorted(paths)},gpu=g,image=image,fixture_manifest_sha256=sha(r/'fixture/manifest.json'),fixture_tensor_file_sha256=sha(r/'fixture/fixture.safetensors'),candidate_manifest_sha256=sha(i/'manifests/candidate-manifest.json'),helper_sha256=sha(i/'tools/container_preflight.py'),private_source_archive_sha256='209b2de9e76c358b94407202a4670613055e963200184e716b50eb67aad71c91')
assert manifest['fixture_manifest_sha256']=='e13a86fab5db5796b22d8dab0f84080b8057b036ca7aef2c7be8832e76505bf1'
assert manifest['fixture_tensor_file_sha256']=='c30aa761eba827bb0b4eb6de0e99d9b8dda8243dd16d00146f9718d2abb05e8b'
assert manifest['candidate_manifest_sha256']==plan['qualified_source']['candidate_manifest_sha256']
assert manifest['helper_sha256']==plan['qualified_source']['container_preflight_sha256']
(r/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
PY
run_worker() {
    local mode=$1 block=$2 cell=$3 capture=$4 name=$5 limit=$6 prepare=$7 budget remaining code
    remaining=$((2700-(SECONDS-started)))
    (( remaining>0 )) || return 124
    budget=$limit
    (( budget<=remaining )) || budget=$remaining
    local mount="type=bind,source=$input_dir,target=/study/input/current"
    local extra=()
    if [[ "$prepare" == yes ]]; then extra=(--prepare-data-links); else mount="$mount,readonly"; fi
    mkdir -p "$runtime/home/$name" "$runtime/tuner/$name"
    active_name="fi-objvalues-${attempt}-$name"
    active_cid="$results/$name.container-id"
    if timeout --kill-after=15s "$budget" docker run --rm --name "$active_name" --cidfile "$active_cid" \
        --gpus all --user "$(id -u):$(id -g)" --network none \
        --mount "type=bind,source=$source_dir,target=/workspace,readonly" --mount "$mount" \
        --mount "type=bind,source=$results,target=$cr" --mount "type=bind,source=$runtime,target=$ct" \
        --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 --env SGLANG_RUST_BUILD_MODE=never \
        --env CUDA_VISIBLE_DEVICES=0 --env FLASHINFER_AUTOTUNER_LOAD_FROM_FILE=0 \
        --env FLASHINFER_CUDA_ARCH_LIST=9.0 --env MAX_JOBS=4 --env LD_PRELOAD= \
        --env "HOME=$ct/home/$name" --env "PYTHONPATH=/study/input/current/tools:/workspace/python" \
        --env "FLASHINFER_WORKSPACE_BASE=$ct/cache/current" --env "XDG_CACHE_HOME=$ct/cache/xdg" \
        --env "CUDA_CACHE_PATH=$ct/cache/cuda" --env "TRITON_CACHE_DIR=$ct/cache/triton" --env "TORCHINDUCTOR_CACHE_DIR=$ct/cache/inductor" \
        --workdir /workspace --entrypoint /opt/sglang/bin/python "$image" \
        "/workspace/$batch/run_current.py" --source /study/input/current/source \
        --manifest /study/input/current/manifests/candidate-manifest.json \
        --target /workspace/test/manual/kernels/bench_flashinfer_objective_values.py \
        --output "$cr/$name.json" "${extra[@]}" -- \
        --mode "$mode" --block "$block" --cell "$cell" --capture-order "$capture" \
        --plan "/workspace/$batch/plan.json" --schedule "/workspace/$batch/schedule.json" \
        --fixture "$cr/fixture/fixture.safetensors" --fixture-manifest "$cr/fixture/manifest.json" \
        --cache "$ct/tuner/$name/cache.json" --output "$cr/$name.json" --trace "$cr/$name.trace.json" \
        > "$results/$name.log" 2>&1; then
        printf '0\n' > "$results/$name.exit-code"
    else
        code=$?; printf '%s\n' "$code" > "$results/$name.exit-code"
        tail -60 "$results/$name.log"; return "$code"
    fi
    active_cid=
    active_name=
    printf 'Completed %s\n' "$name"
}
analyze() {
    local label=$1
    shift
    python3 "$source_dir/$batch/analyze.py" --raw-root "$results" --source-root "$source_dir" \
        --plan "$source_dir/$batch/plan.json" --schedule "$source_dir/$batch/schedule.json" \
        --fixture-manifest "$results/fixture/manifest.json" --output "$results/$label.json" "$@" \
        > "$results/$label.log" 2>&1
}
python3 - "$source_dir/$batch/schedule.json" > "$results/launch-schedule.tsv" <<'PY'
import json,sys
for row in json.load(open(sys.argv[1]))['entries']:
    print(row['block'],row['cell'],row['capture_order'],row['raw_name'].removesuffix('.json'),sep='\t')
PY
first=yes
while IFS=$'\t' read -r block cell capture name; do
    [[ "$block" == 0 ]] || break
    limit=300
    [[ "$first" == no ]] || limit=900
    run_worker admission "$block" "$cell" "$capture" "admission-$cell" "$limit" "$first"
    first=no
done < "$results/launch-schedule.tsv"
analyze admission-analysis --admission
count=0
while IFS=$'\t' read -r block cell capture name; do
    run_worker timing "$block" "$cell" "$capture" "$name" 180 no
    count=$((count+1))
    if [[ "$count" == 4 ]]; then analyze first-block-analysis --blocks 1; fi
done < "$results/launch-schedule.tsv"
[[ "$count" == 32 ]] || exit 1
analyze analysis --blocks 8
