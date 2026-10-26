#!/usr/bin/env bash
# One admission plus eight frozen processes; preserve every attempt and all timing signs.
set -euo pipefail
[[ "$(hostname -s)" == 1u1g-gen-0100 ]] || exit 1
study=$(realpath "${1:?study directory required}")
commit=${2:?frozen commit required}
attempt=${3:?unique attempt required}
phase=${4:?admission or timing phase required}
[[ "$phase" == admission || "$phase" == timing ]] || exit 1
[[ "$commit" =~ ^[0-9a-f]{40}$ && "$attempt" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 1
batch=test/manual/kernels/results/flashinfer_autotune_selection/20261026T220000Z-h100-fixed-pair
source_dir="$study/source"
input_dir="$study/input/current"
results="$study/results/fixed-pair/$attempt"
runtime="$study/runtime-$attempt"
cr="/study/results/fixed-pair/$attempt"
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
        if [[ -d "$runtime/cache" ]]; then
            mkdir -p "$results/native-build-cache" || code=1
            cp -a "$runtime/cache" "$results/native-build-cache/$phase" || code=1
        fi
        git -C "$source_dir" rev-parse HEAD > "$results/sglang-commit-after-$phase.txt" || code=1
        git -C "$source_dir" status --porcelain > "$results/sglang-status-after-$phase.txt" || code=1
        [[ "$(cat "$results/sglang-commit-after-$phase.txt")" == "$commit" && ! -s "$results/sglang-status-after-$phase.txt" ]] || code=1
        nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,power.limit,temperature.gpu,memory.total,memory.used --format=csv > "$results/gpu-after-$phase.csv" || code=1
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
if [[ "$phase" == admission ]]; then
    [[ ! -e "$results" && ! -e "$runtime" ]] || exit 1
    mkdir -p "$results" "$runtime/home" "$runtime/cache/current"
    cp -a "$study/fixture" "$results/fixture"
    cp "$study/source-snapshot.json" "$results/source-snapshot.json"
else
    [[ -d "$results" && -d "$runtime" && ! -e "$results/timing.started" ]] || exit 1
    [[ "$(cat "$results/launcher-admission.exit-code")" == 0 ]] || exit 1
    python3 - "$results" <<'ADMISSION'
import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=r/'admission-auto_synthetic.json'; b=a.with_suffix('.bootstrap.json')
analysis=json.loads((r/'admission-analysis.json').read_text())
assert analysis['status']=='passed'
assert analysis['admission_result_sha256']==sha(a)
assert analysis['admission_bootstrap_sha256']==sha(b)
assert json.loads(a.read_text())['status']=='success'
bootstrap=json.loads(b.read_text()); assert bootstrap['status']=='passed'
assert bootstrap['result_sha256']==sha(a)
ADMISSION
    [[ ! -e "$results/native-build-cache/timing" ]] || exit 1
    touch "$results/timing.started"
fi
created=1
printf '%s\n' "$commit" > "$results/sglang-commit-before-$phase.txt"
docker image inspect "$image" | python3 -c 'import json,sys;x=json.load(sys.stdin)[0];print(json.dumps({k:x[k] for k in ("Id","RepoDigests","Created","Architecture","Os","Size")},indent=2))' > "$results/image-$phase.json"
nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,power.limit,temperature.gpu,memory.total,memory.used --format=csv > "$results/gpu-before-$phase.csv"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader > "$results/gpu-processes-before-$phase.csv"
[[ ! -s "$results/gpu-processes-before-$phase.csv" ]] || exit 1
lscpu -J > "$results/host-cpu-$phase.json"
df -Pk "$study" > "$results/node-disk-before-$phase.txt"
python3 - "$source_dir" "$input_dir" "$results" "$commit" "$batch" "$phase" <<'PY'
import csv,hashlib,json,sys
from pathlib import Path
s,i,r=map(Path,sys.argv[1:4]);commit,batch,phase=sys.argv[4:]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((s/batch/'plan.json').read_text())
gpu=[{k.strip():v.strip() for k,v in row.items()} for row in csv.DictReader((r/f'gpu-before-{phase}.csv').open())]
assert len(gpu)==1;g=gpu[0]
assert g['name']=='NVIDIA H100 NVL' and g['uuid']=='GPU-240958a2-e8e1-3bac-b767-365c9e0b97bd'
assert g['driver_version']=='580.159.04' and float(g['power.limit [W]'].split()[0])==310 and int(g['memory.used [MiB]'].split()[0])==0
image=json.loads((r/f'image-{phase}.json').read_text())
assert image['Architecture']=='amd64' and image['Os']=='linux' and plan['environment']['image'] in image['RepoDigests']
assert image['Id']=='sha256:2e152727f0a5ce999172ec48203b2b292d41b3cb370976827718870557899caa'
assert sha(s/batch/'schedule.json')==plan['schedule_sha256']
assert sha(s/batch/'historical-contract.json')==plan['historical_contract_sha256']
assert plan['protocol']['fresh_processes']==8 and plan['protocol']['admission_workers']==1
fixture=json.loads((r/'fixture/manifest.json').read_text())
paths={str(p.relative_to(s)) for p in (s/'test/manual/kernels').glob('*.py')}
paths.update(str(p.relative_to(s)) for p in (s/batch).glob('*') if p.suffix in ('.py','.sh','.json'))
paths.update(fixture['source']['sglang_source_sha256'])
paths.add('python/sglang/srt/model_executor/runner/flashinfer_autotune.py')
if phase=='admission': assert not (i/'source/flashinfer/data').exists()
manifest=dict(schema_version=1,execution_commit=commit,plan_sha256=sha(s/batch/'plan.json'),schedule_sha256=sha(s/batch/'schedule.json'),historical_contract_sha256=sha(s/batch/'historical-contract.json'),source_sha256={p:sha(s/p) for p in sorted(paths)},gpu=g,image=image,fixture_manifest_sha256=sha(r/'fixture/manifest.json'),fixture_tensor_file_sha256=sha(r/'fixture/fixture.safetensors'),candidate_manifest_sha256=sha(i/'manifests/candidate-manifest.json'),helper_sha256=sha(i/'tools/container_preflight.py'),private_source_archive_sha256='209b2de9e76c358b94407202a4670613055e963200184e716b50eb67aad71c91')
assert manifest['fixture_manifest_sha256']=='e13a86fab5db5796b22d8dab0f84080b8057b036ca7aef2c7be8832e76505bf1'
assert manifest['fixture_tensor_file_sha256']=='c30aa761eba827bb0b4eb6de0e99d9b8dda8243dd16d00146f9718d2abb05e8b'
assert manifest['candidate_manifest_sha256']==plan['qualified_source']['candidate_manifest_sha256']
assert manifest['helper_sha256']==plan['qualified_source']['container_preflight_sha256']
if phase=='timing':
    old=json.loads((r/'manifest.json').read_text())
    for name in ('execution_commit','plan_sha256','schedule_sha256','historical_contract_sha256','source_sha256','fixture_manifest_sha256','fixture_tensor_file_sha256','candidate_manifest_sha256','helper_sha256'):
        assert manifest[name]==old[name], 'admission identity changed: '+name
    assert json.loads((r/'admission-auto_synthetic.bootstrap.json').read_text())['worker_sha256']==manifest['source_sha256']['test/manual/kernels/bench_flashinfer_fixed_pair.py']
    name='manifest-timing.json'
else:
    name='manifest.json'
(r/name).write_text(json.dumps(manifest,indent=2)+'\n')
PY
run_worker() {
    local mode=$1 trial=$2 capture=$3 profile=$4 name=$5 limit=$6 prepare=$7 budget remaining code
    remaining=$((2700-(SECONDS-started)))
    (( remaining>0 )) || return 124
    budget=$limit
    (( budget<=remaining )) || budget=$remaining
    local mount="type=bind,source=$input_dir,target=/study/input/current"
    local extra=()
    if [[ "$prepare" == yes ]]; then extra=(--prepare-data-links); else mount="$mount,readonly"; fi
    mkdir -p "$runtime/home/$name" "$runtime/tuner/$name"
    active_name="fi-fixedpair-${attempt}-$name"
    active_cid="$results/$name.container-id"
    if timeout --kill-after=15s "$budget" docker run --rm --name "$active_name" --cidfile "$active_cid" \
        --gpus all --user "$(id -u):$(id -g)" --network none \
        --mount "type=bind,source=$source_dir,target=/workspace,readonly" --mount "$mount" \
        --mount "type=bind,source=$results,target=$cr" --mount "type=bind,source=$runtime,target=$ct" \
        --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 --env SGLANG_RUST_BUILD_MODE=never \
        --env CUDA_VISIBLE_DEVICES=0 --env FLASHINFER_AUTOTUNER_LOAD_FROM_FILE=0 \
        --env FLASHINFER_CUDA_ARCH_LIST=9.0 --env MAX_JOBS=4 --env LD_PRELOAD= \
        --env "HOME=$ct/home/$name" --env "PYTHONPATH=/study/input/current/tools:/workspace/python:/workspace/test/manual/kernels" \
        --env "FLASHINFER_WORKSPACE_BASE=$ct/cache/current" --env "XDG_CACHE_HOME=$ct/cache/xdg" \
        --env "CUDA_CACHE_PATH=$ct/cache/cuda" --env "TRITON_CACHE_DIR=$ct/cache/triton" --env "TORCHINDUCTOR_CACHE_DIR=$ct/cache/inductor" \
        --workdir /workspace --entrypoint /opt/sglang/bin/python "$image" \
        "/workspace/$batch/run_current.py" --source /study/input/current/source \
        --manifest /study/input/current/manifests/candidate-manifest.json \
        --target /workspace/test/manual/kernels/bench_flashinfer_fixed_pair.py \
        --output "$cr/$name.json" "${extra[@]}" -- \
        --mode "$mode" --trial "$trial" --capture-order "$capture" --profile-order "$profile" \
        --plan "/workspace/$batch/plan.json" --schedule "/workspace/$batch/schedule.json" \
        --historical-contract "/workspace/$batch/historical-contract.json" \
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
    python3 - "$results/$name.json" <<'CHECK'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]); result=json.loads(p.read_text()); bootstrap=json.loads(p.with_suffix('.bootstrap.json').read_text())
assert result['status']=='success', result.get('error', 'worker failed')
assert bootstrap['status']=='passed'
assert bootstrap['result_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
CHECK
    printf 'Completed %s\n' "$name"
}
python3 - "$source_dir/$batch/schedule.json" > "$results/launch-schedule-$phase.tsv" <<'PY_SCHEDULE'
import json,sys
rows=json.load(open(sys.argv[1]))['entries']
assert len(rows)==8 and [x['block'] for x in rows]==list(range(8))
for row in rows:
    assert row['cell']=='auto_synthetic'
    assert row['capture_order'] in ('winner-first','default-first')
    assert row['profiler_objective_order'] in (['auto','graph'],['graph','auto'])
    profile=row['profiler_objective_order'][0]+'-first'
    print(row['block'],row['capture_order'],profile,'block'+str(row['block'])+'-auto_synthetic-timing',sep='\t')
PY_SCHEDULE
IFS=$'\t' read -r trial capture profile name < "$results/launch-schedule-$phase.tsv"
if [[ "$phase" == admission ]]; then
    run_worker admission "$trial" "$capture" "$profile" admission-auto_synthetic 900 yes
    printf 'Admission completed; independent admission analysis is required before timing.\n'
    exit 0
fi
count=0
while IFS=$'\t' read -r trial capture profile name; do
    run_worker timing "$trial" "$capture" "$profile" "$name" 300 no
    count=$((count+1))
done < "$results/launch-schedule-$phase.tsv"
[[ "$count" == 8 ]] || exit 1
printf 'Completed %s timing processes; analysis is separate.\n' "$count"
