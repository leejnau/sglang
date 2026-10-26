#!/usr/bin/env bash
# Capture, six admission processes, then the frozen 36-process comparison.
set -euo pipefail
[[ "$(hostname -s)" == 1u1g-gen-0100 ]] || exit 1
study=$(realpath "${1:?study directory required}")
commit=${2:?frozen commit required}
attempt=${3:?unique attempt required}
phase=${4:?capture, admission or timing phase required}
[[ "$phase" == capture || "$phase" == admission || "$phase" == timing ]] || exit 1
[[ "$commit" =~ ^[0-9a-f]{40}$ && "$attempt" == attempt01 ]] || exit 1
batch=test/manual/kernels/results/flashinfer_autotune_selection/20261027T000000Z-h100-mlp-selection
source_dir="$study/source"
input_dir="$study/input/current"
results="$study/results/mlp-selection/$attempt"
runtime="$study/runtime-$attempt"
cr="/study/results/mlp-selection/$attempt"
ct="/study/runtime-$attempt"
image=lmsysorg/sglang@sha256:c1f4d9fc6fcf50c2269adbc757ce7b1778883af04155edb888fa9eea3b6648c0
active_cid=
active_name=
created=0
started=$SECONDS
verify_reuse() {
    python3 - "$study" "$results/reuse-$1-$phase.json" <<'REUSE_CHECK'
import hashlib,json,os,sys
from pathlib import Path
s=Path(sys.argv[1]);output=Path(sys.argv[2]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((s/'native-reuse.json').read_text());assert m['status']=='passed'
assert sha(s/'native-reference.bootstrap.json')==m['native_reference_sha256']=='7c4727aca3d99adcb3fe2f35d06f4408b512baa4fe9a91b975bdf77358fe205d'
root=s/'input/current'
for rel,value in m['input_files'].items(): assert sha(root/rel)==value,rel
links={str(p.relative_to(root)):os.readlink(p) for p in root.rglob('*') if p.is_symlink()}
assert links==m['input_links']
assert {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and not p.is_symlink()}==set(m['input_files'])
cache=s/'runtime-attempt01/cache'
checked=0
for rel,value in m['cache_files'].items():
    if rel.endswith('/flashinfer_jit.log'): continue
    assert sha(cache/rel)==value,rel
    checked+=1
output.write_text(json.dumps(dict(status='passed',native_reuse_sha256=sha(s/'native-reuse.json'),input_files_verified=len(m['input_files']),cache_files_verified=checked),indent=2)+'\n')
REUSE_CHECK
}
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
        verify_reuse after || code=1
        if [[ -d "$runtime/cache" ]]; then
            mkdir -p "$results/native-build-cache" || code=1
            cp -a "$runtime/cache" "$results/native-build-cache/$phase" || code=1
        fi
        if [[ -d "$runtime/tuner/$phase" ]]; then
            mkdir -p "$results/tuner-cache" || code=1
            cp -a "$runtime/tuner/$phase" "$results/tuner-cache/$phase" || code=1
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
if [[ "$phase" == capture ]]; then
    [[ ! -e "$results" && -d "$runtime/cache/current" && ! -e "$runtime/home" ]] || exit 1
    mkdir -p "$results/native-build-cache" "$runtime/home"
    cp "$study/native-reference.bootstrap.json" "$results/"
    cp "$study/native-reuse.json" "$results/native-build-cache/reuse-manifest.json"
    cp "$study/native-restoration.json" "$results/native-build-cache/restoration.json"
    cp "$study/native-preservation.json" "$results/native-build-cache/prior-preservation.json"
    cp "$study/capture-source-snapshot.json" "$results/capture-source-snapshot.json"
    cp /home/lnau/flashinfer-autotune-study/model-cache/20261027T000000Z-h100-mlp-selection-checkpoint-verification.json "$results/checkpoint-verification.json"
    cp "$source_dir/$batch/plan.json" "$results/capture-plan.json"
else
    [[ -d "$results" && -d "$runtime" && ! -e "$results/$phase.started" ]] || exit 1
    [[ "$(cat "$results/launcher-capture.exit-code")" == 0 && -f "$results/fixture-freeze.json" ]] || exit 1
    if [[ "$phase" == timing ]]; then
        [[ "$(cat "$results/launcher-admission.exit-code")" == 0 ]] || exit 1
        python3 - "$results" <<'ADMISSION'
import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
analysis=json.loads((r/'admission-analysis.json').read_text())
outer=json.loads((r/'admission-outer-audit.json').read_text())
assert analysis['status']=='passed' and outer['status']=='passed' and outer['phase']=='admission'
expected_names={f'admission-trial0-{arm}-{phase}' for arm in ('default','v1','v2') for phase in ('cold','warm')}
rows={x['name']:x for x in analysis['results']}
audited={x['name']:x for x in outer['workers']}
assert len(analysis['results'])==len(outer['workers'])==6 and set(rows)==set(audited)==expected_names
expected={'gemm':'72690cacca4ce1d69ccaa6b9190475483d984bd58de0962d06c874c801c678d6','trtllm_utils':'e442a689064fe3af4ab46510b2d5aaf3597b072949b84f13d14e9e53352109e4'}
for name in sorted(expected_names):
    a=r/(name+'.json');b=a.with_suffix('.bootstrap.json')
    assert rows[name]['result_sha256']==audited[name]['raw_sha256']==sha(a)
    assert rows[name]['bootstrap_sha256']==audited[name]['bootstrap_sha256']==sha(b)
    assert json.loads(a.read_text())['status']=='success'
    bootstrap=json.loads(b.read_text()); assert bootstrap['status']=='passed' and bootstrap['result_sha256']==sha(a)
    reuse=bootstrap['native_reuse']
    assert reuse['reference_sha256']=='7c4727aca3d99adcb3fe2f35d06f4408b512baa4fe9a91b975bdf77358fe205d'
    assert reuse['manifest_sha256']==sha(r/'native-build-cache/reuse-manifest.json')
    for point in ('before','after_build','after_worker'):
        assert reuse[point]['status']=='passed' and reuse[point]['elf_sha256']==expected
ADMISSION
    else
        cp "$study/source-snapshot.json" "$results/source-snapshot.json"
    fi
    touch "$results/$phase.started"
fi
created=1
verify_reuse before
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
assert plan['protocol']['fresh_processes']==36 and plan['protocol']['admission_workers']==6
fixture=json.loads((r/'fixture/manifest.json').read_text()) if phase!='capture' else None
paths={str(p.relative_to(s)) for p in (s/'test/manual/kernels').glob('*.py')}
paths.update(str(p.relative_to(s)) for p in (s/batch).glob('*') if p.suffix in ('.py','.sh','.json'))
if fixture is not None: paths.update(fixture['source']['sglang_source_sha256'])
paths.add('python/sglang/srt/model_executor/runner/flashinfer_autotune.py')
assert (i/'source/flashinfer/data').is_dir()
manifest=dict(schema_version=1,execution_commit=commit,plan_sha256=sha(s/batch/'plan.json'),schedule_sha256=sha(s/batch/'schedule.json'),source_sha256={p:sha(s/p) for p in sorted(paths)},gpu=g,image=image,candidate_manifest_sha256=sha(i/'manifests/candidate-manifest.json'),helper_sha256=sha(i/'tools/container_preflight.py'),private_source_archive_sha256='209b2de9e76c358b94407202a4670613055e963200184e716b50eb67aad71c91')
manifest['native_reference_sha256']=sha(r/'native-reference.bootstrap.json')
manifest['native_reuse_sha256']=sha(r/'native-build-cache/reuse-manifest.json')
assert manifest['candidate_manifest_sha256']==plan['qualified_source']['candidate_manifest_sha256']
assert manifest['helper_sha256']==plan['qualified_source']['container_preflight_sha256']
if phase=='capture':
    assert plan['fixture_sha256'] is None and plan['fixture_manifest_sha256'] is None
    name='capture-environment.json'
else:
    manifest['fixture_manifest_sha256']=sha(r/'fixture/manifest.json')
    manifest['fixture_tensor_file_sha256']=sha(r/'fixture/fixture.safetensors')
    assert manifest['fixture_manifest_sha256']==plan['fixture_manifest_sha256']
    assert manifest['fixture_tensor_file_sha256']==plan['fixture_sha256']
    freeze=json.loads((r/'fixture-freeze.json').read_text())
    assert freeze['status']=='passed' and freeze['execution_commit']==commit and freeze['plan_sha256']==manifest['plan_sha256']
    assert freeze['capture_commit']==fixture['source']['sglang_commit']
    if phase=='timing':
        old=json.loads((r/'manifest.json').read_text())
        for field in ('execution_commit','plan_sha256','schedule_sha256','source_sha256','fixture_manifest_sha256','fixture_tensor_file_sha256','candidate_manifest_sha256','helper_sha256','native_reference_sha256','native_reuse_sha256'):
            assert manifest[field]==old[field], 'admission identity changed: '+field
        name='manifest-timing.json'
    else:
        name='manifest.json'
(r/name).write_text(json.dumps(manifest,indent=2)+'\n')
PY
if [[ "$phase" == capture ]]; then
    active_name="fi-mlp-${attempt}-capture"
    active_cid="$results/capture.container-id"
    mkdir "$runtime/home/capture"
    if timeout --kill-after=15s 900 docker run --rm --name "$active_name" --cidfile "$active_cid" \
        --gpus all --user "$(id -u):$(id -g)" --network none \
        --mount "type=bind,source=$source_dir,target=/workspace,readonly" \
        --mount "type=bind,source=/home/lnau/flashinfer-autotune-study/model-cache,target=/model-cache,readonly" \
        --mount "type=bind,source=$results,target=$cr" --mount "type=bind,source=$runtime,target=$ct" \
        --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES=0 \
        --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env "HOME=$ct/home/capture" \
        --env "XDG_CACHE_HOME=$ct/capture-cache/xdg" --env "CUDA_CACHE_PATH=$ct/capture-cache/cuda" \
        --env "TRITON_CACHE_DIR=$ct/capture-cache/triton" --env LD_PRELOAD= \
        --workdir /workspace --entrypoint /opt/sglang/bin/python "$image" \
        /workspace/test/manual/kernels/capture_flashinfer_mlp_fixture.py \
        --model-cache /model-cache --output-dir "$cr/fixture" --device cuda > "$results/capture.log" 2>&1; then
        printf '0\n' > "$results/capture.exit-code"
    else
        code=$?; printf '%s\n' "$code" > "$results/capture.exit-code"
        tail -60 "$results/capture.log"; exit "$code"
    fi
    active_cid=
    active_name=
    python3 - "$results" "$commit" <<'CAPTURE_CHECK'
import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]);m=json.loads((r/'fixture/manifest.json').read_text())
assert m['schema_version']==2 and m['fixture_kind']=='qwen3_mlp_down_proj'
assert m['source']['sglang_commit']==sys.argv[2]
assert m['tensor_file']['sha256']==hashlib.sha256((r/'fixture/fixture.safetensors').read_bytes()).hexdigest()
assert m['capture']['activation_byte_match'] and m['capture']['checkpoint_tensor_byte_match']
assert m['capture']['full_bf16_forward_completed']
CAPTURE_CHECK
    printf 'Capture completed; freeze fixture hashes and final source before admission.\n'
    exit 0
fi
run_worker() {
    local mode=$1 trial=$2 arm=$3 cache_phase=$4 capture=$5 mode_order=$6 name=$7 budget remaining code
    remaining=$((4500-(SECONDS-started)))
    (( remaining>0 )) || return 124
    budget=360
    (( budget<=remaining )) || budget=$remaining
    local mount="type=bind,source=$input_dir,target=/study/input/current,readonly"
    local cache_dir="$runtime/tuner/$mode/trial$trial-$arm"
    [[ ! -e "$runtime/home/$name" && ! -e "$results/$name.json" && ! -e "$results/$name.container-id" ]] || return 1
    if [[ "$cache_phase" == cold ]]; then
        [[ ! -e "$cache_dir" ]] || return 1
        mkdir -p "$cache_dir"
    else
        [[ -d "$cache_dir" && "$(cat "$results/${name%-warm}-cold.exit-code")" == 0 ]] || return 1
    fi
    mkdir -p "$runtime/home/$name"
    active_name="fi-mlp-${attempt}-$name"
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
        --target /workspace/test/manual/kernels/bench_flashinfer_mlp_selection.py \
        --output "$cr/$name.json" --native-reference "$cr/native-reference.bootstrap.json" --reuse-manifest "$cr/native-build-cache/reuse-manifest.json" -- \
        --mode "$mode" --trial "$trial" --arm "$arm" --phase "$cache_phase" \
        --capture-order "$capture" --mode-order "$mode_order" \
        --plan "/workspace/$batch/plan.json" --schedule "/workspace/$batch/schedule.json" \
        --manifest /study/input/current/manifests/candidate-manifest.json \
        --fixture "$cr/fixture/fixture.safetensors" --fixture-manifest "$cr/fixture/manifest.json" \
        --cache "$ct/tuner/$mode/trial$trial-$arm/cache.json" --output "$cr/$name.json" --trace "$cr/$name.trace.json" \
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
python3 - "$source_dir/$batch/schedule.json" "$phase" > "$results/launch-schedule-$phase.tsv" <<'PY_SCHEDULE'
import itertools,json,sys
rows=json.load(open(sys.argv[1]))['entries'];mode=sys.argv[2]
assert mode in ('admission','timing') and len(rows)==36
assert len({x['id'] for x in rows})==36
orders=[]
for trial in range(6):
    block=[x for x in rows if x['trial']==trial]
    assert len(block)==6
    order=[]
    for i in (0,2,4):
        cold,warm=block[i:i+2]
        assert cold['phase']=='cold' and warm['phase']=='warm'
        assert {k:v for k,v in cold.items() if k not in ('id','phase')}=={k:v for k,v in warm.items() if k not in ('id','phase')}
        order.append(cold['arm'])
    orders.append(tuple(order))
assert set(orders)==set(itertools.permutations(('default','v1','v2')))
for row in rows:
    if mode=='admission' and row['trial']!=0: continue
    assert row['capture_order'] in ('natural-first','default-first')
    assert row['mode_order'] in ('eager-first','graph-first')
    assert row['id']==f"trial{row['trial']}-{row['arm']}-{row['phase']}"
    name=('admission-' if mode=='admission' else '')+row['id']
    print(row['trial'],row['arm'],row['phase'],row['capture_order'],row['mode_order'],name,sep='\t')
PY_SCHEDULE
count=0
while IFS=$'\t' read -r trial arm cache_phase capture mode_order name; do
    run_worker "$phase" "$trial" "$arm" "$cache_phase" "$capture" "$mode_order" "$name"
    count=$((count+1))
done < "$results/launch-schedule-$phase.tsv"
if [[ "$phase" == admission ]]; then [[ "$count" == 6 ]]; else [[ "$count" == 36 ]]; fi
printf 'Completed %s %s processes; analysis is separate.\n' "$count" "$phase"
