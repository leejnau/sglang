#!/usr/bin/env bash
# Restore exact qualified source/native bytes from preserved private archives.
set -euo pipefail
allocation=${1:?allocation required}
source_commit=${2:?frozen source commit required}
phase=${3:?initial or update required}
[[ "$phase" == initial || "$phase" == update ]] || exit 1
[[ "$allocation" =~ ^[0-9]+$ && "$source_commit" =~ ^[0-9a-f]{40}$ ]] || exit 1
srun --jobid "$allocation" --nodes=1 --ntasks=1 bash -s -- "$allocation" "$source_commit" "$phase" <<'NODE'
set -euo pipefail
allocation=$1
source_commit=$2
phase=$3
[[ "$(hostname -s)" == 1u1g-gen-0100 ]] || exit 1
study=/tmp/fi-mlp-selection-$allocation
durable=/home/lnau/flashinfer-autotune-study
batch=20261027T000000Z-h100-mlp-selection
pack="$durable/$batch-$source_commit-snapshot.pack"
archive="$durable/20261026T220000Z-h100-fixed-pair-attempt01-timing-full-raw.tgz"
if [[ "$phase" == initial ]]; then
    [[ ! -e "$study" ]] || exit 1
else
    results="$study/results/mlp-selection/attempt01"
    [[ -d "$study/source" && "$(cat "$results/launcher-capture.exit-code")" == 0 && ! -e "$results/admission.started" ]] || exit 1
    [[ -z "$(git -C "$study/source" status --porcelain)" ]] || exit 1
fi
python3 - "$pack" "$source_commit" "$durable/fp8-current-source-private.tar.gz" "$archive" <<'VERIFY'
import hashlib,json,sys
from pathlib import Path
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=Path(sys.argv[1]);m=json.loads(Path(str(p)+'.json').read_text())
assert m['source_commit']==sys.argv[2] and not m['history_walk'] and not m['lazy_fetch']
assert sha(p)==m['sha256']
assert sha(Path(sys.argv[3]))=='209b2de9e76c358b94407202a4670613055e963200184e716b50eb67aad71c91'
assert sha(Path(sys.argv[4]))=='84e44c807d2a897723e793eeb0dd676d1c254491fdf2c95149511953ee4272f5'
VERIFY
mkdir -p "$study/input/current/manifests" "$study/runtime-attempt01/cache"
export GIT_NO_LAZY_FETCH=1 GIT_TERMINAL_PROMPT=0
if [[ "$phase" == initial ]]; then git init -q "$study/source"; fi
git -C "$study/source" index-pack --stdin < "$pack" > "$study/index-pack-$phase.txt"
capture=a29b31850c7b68cafde022614b00ab01d6ff5cde
{ cat "$study/source/.git/shallow" 2>/dev/null || true; printf '%s\n' "$source_commit" "$capture"; } | sort -u > "$study/shallow-next"
cp "$study/shallow-next" "$study/source/.git/shallow"
git -C "$study/source" config core.sparseCheckout true
git -C "$study/source" config core.sparseCheckoutCone false
printf '%s\n' /python/ /test/manual/kernels/ /.gitignore /pyproject.toml > "$study/source/.git/info/sparse-checkout"
git -C "$study/source" update-ref refs/heads/frozen "$source_commit"
git -C "$study/source" update-ref refs/heads/capture "$capture"
git -C "$study/source" symbolic-ref HEAD refs/heads/frozen
git -C "$study/source" read-tree -mu HEAD
[[ "$(git -C "$study/source" rev-parse HEAD)" == "$source_commit" && -z "$(git -C "$study/source" status --porcelain)" ]] || exit 1
cp "$pack.json" "$study/source-snapshot.json"
if [[ "$phase" == update ]]; then
    python3 - "$study" "$source_commit" "$batch" <<'FREEZE'
import hashlib,json,subprocess,sys
from pathlib import Path
s=Path(sys.argv[1]);commit,batch=sys.argv[2:];r=s/'results/mlp-selection/attempt01'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((s/'source'/ 'test/manual/kernels/results/flashinfer_autotune_selection'/batch/'plan.json').read_text())
fixture=json.loads((r/'fixture/manifest.json').read_text())
assert sha(r/'fixture/fixture.safetensors')==plan['fixture_sha256']
assert sha(r/'fixture/manifest.json')==plan['fixture_manifest_sha256']
assert fixture['schema_version']==2 and fixture['fixture_kind']=='qwen3_mlp_down_proj'
initial=json.loads((r/'capture-source-snapshot.json').read_text())['source_commit']
assert fixture['source']['sglang_commit']==initial and initial!=commit
assert subprocess.check_output(['git','-C',str(s/'source'),'rev-parse','refs/heads/mlp-capture'],text=True).strip()==initial
for rel,value in fixture['source']['sglang_source_sha256'].items():
    assert sha(s/'source'/rel)==value,rel
for rel in ('test/manual/kernels/capture_flashinfer_mlp_fixture.py',):
    captured=subprocess.check_output(['git','-C',str(s/'source'),'show',initial+':'+rel])
    assert hashlib.sha256(captured).hexdigest()==sha(s/'source'/rel)
record=dict(status='passed',capture_commit=initial,execution_commit=commit,fixture_sha256=plan['fixture_sha256'],fixture_manifest_sha256=plan['fixture_manifest_sha256'],plan_sha256=sha(s/'source'/'test/manual/kernels/results/flashinfer_autotune_selection'/batch/'plan.json'),capture_source_snapshot_sha256=sha(r/'capture-source-snapshot.json'),execution_source_snapshot_sha256=sha(s/'source-snapshot.json'))
with (r/'fixture-freeze.json').open('x') as handle: json.dump(record,handle,indent=2);handle.write('\n')
print(json.dumps(record))
FREEZE
    exit 0
fi
git -C "$study/source" update-ref refs/heads/mlp-capture "$source_commit"
cp "$pack.json" "$study/capture-source-snapshot.json"
tar -xzf "$durable/fp8-current-source-private.tar.gz" -C "$study/input/current"
for name in candidate-manifest complete-source-manifest dependency-manifest cutlass-source-manifest cccl-source-manifest spdlog-source-manifest; do
    mv "$study/input/current/$name.json" "$study/input/current/manifests/$name.json"
done
qualified=test/manual/kernels/results/flashinfer_autotune_selection/20261024T090000Z-h100-current-source-qualification/revision02
printf '%s  %s\n' ae7159aba41c3f26199568f9742ab140e7c20985131dbdbd58d7af7c28246829 "$study/source/$qualified/qualification-tools.tar.gz" | sha256sum -c -
tar -xzf "$study/source/$qualified/qualification-tools.tar.gz" -C "$study/input/current"
git -C "$study/input/current/source" apply "$study/input/current/patches/harness-only.patch"
# Recreate only the five known official packaging links at the same container paths.
mkdir "$study/input/current/source/flashinfer/data"
for name in csrc include; do
    ln -s "/study/input/current/source/$name" "$study/input/current/source/flashinfer/data/$name"
done
for name in cutlass cccl spdlog; do
    ln -s "/study/input/current/source/3rdparty/$name" "$study/input/current/source/flashinfer/data/$name"
done
tar -xzf "$archive" -C "$study/runtime-attempt01/cache" --strip-components=3 attempt01/native-build-cache/timing
tar -xzf "$archive" -C "$study" --strip-components=1 attempt01/fixture
mv "$study/fixture" "$study/prior-fixture"
# Docker cannot traverse this shared home mount; retain the exact cache locally.
cp -a "$durable/model-cache" "$study/model-cache"
tar -xOf "$archive" attempt01/block0-auto_synthetic-timing.bootstrap.json > "$study/native-reference.bootstrap.json"
tar -xOf "$archive" attempt01/preservation-manifest-timing.json > "$study/native-preservation.json"
python3 "$study/source/test/manual/kernels/results/flashinfer_autotune_selection/$batch/restore_native_cache.py" --study "$study" --output "$study/native-restoration.json"
python3 - "$study" <<'REUSE'
import hashlib,json,os,subprocess,sys
from pathlib import Path
s=Path(sys.argv[1]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
root=s/'input/current'; manifests=root/'manifests'
expected={
    'candidate-manifest.json':'9a4c51562d7410d785cc544c004bc4f92486f40311cb5c7a7328c2c1213160d5',
    'complete-source-manifest.json':'fec08384276b12a056288725e395c4a8f56ffa7ddb3d34374dcbcdae7f91b4c1',
    'dependency-manifest.json':'65f64226dbe1921bfcc69b8026eaa343a55a4de7467ed04aab9dc3ac1ef13dba',
    'cutlass-source-manifest.json':'788523ddee277a5cce291590f09d42cd840ab9673cc8d57e2f36861e12f88c6d',
    'cccl-source-manifest.json':'308ddb534bf9ea9fd1df54e709513a74669438c52b3691b1b38dff5502fb8b6f',
    'spdlog-source-manifest.json':'d61b2209a11e03cd26c449e9c1de1c7642e0714fa091130c3a81b96b24a28dd2',
}
for name,value in expected.items(): assert sha(manifests/name)==value,name
candidate=json.loads((manifests/'candidate-manifest.json').read_text())
assert candidate['base_commit']=='27c62fdf0ab1cd4014df8a15b66a975541fda83c'
variant=next(x for x in candidate['candidates'] if x['name']=='fp8-runtime-callback')
files={x['path']:x['sha256'] for x in json.loads((manifests/'complete-source-manifest.json').read_text())['files']}
files.update({x['path']:x['candidate_sha256'] for x in variant['tracked_changes']})
for name in ('cutlass','cccl','spdlog'):
    files.update({'3rdparty/'+name+'/'+x['path']:x['sha256'] for x in json.loads((manifests/(name+'-source-manifest.json')).read_text())['files']})
for rel,value in files.items():
    path=root/'source'/rel
    observed=hashlib.sha256(os.readlink(path).encode()).hexdigest() if path.is_symlink() else sha(path)
    assert observed==value,rel
assert sha(root/'tools/container_preflight.py')=='b8d0f6bbc85eaca0e3ef1419ab99a97d5340c96b3da382cba81d7896768287ba'
assert sha(root/'patches/harness-only.patch')=='706ca4d91f3e6ef8c1643977beef72fa39af121e691f9380a175633787fe94ef'
links={}
for name,rel in [('csrc','csrc'),('include','include'),('cutlass','3rdparty/cutlass'),('cccl','3rdparty/cccl'),('spdlog','3rdparty/spdlog')]:
    path=root/'source/flashinfer/data'/name
    assert path.is_symlink() and os.readlink(path)=='/study/input/current/source/'+rel,str(path)
    links[str(path.relative_to(root))]=os.readlink(path)
input_files={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and not p.is_symlink()}
links={str(p.relative_to(root)):os.readlink(p) for p in root.rglob('*') if p.is_symlink()}
cache=s/'runtime-attempt01/cache'
cache_files={str(p.relative_to(cache)):sha(p) for p in sorted(cache.rglob('*')) if p.is_file() and not p.is_symlink()}
native_prefix='current/.cache/flashinfer/0.0.0+unknown/90a/cached_ops/'
native_files={p:h for p,h in cache_files.items() if p.startswith((native_prefix+'gemm/',native_prefix+'trtllm_utils/'))}
assert native_files[native_prefix+'gemm/gemm.so']=='72690cacca4ce1d69ccaa6b9190475483d984bd58de0962d06c874c801c678d6'
assert native_files[native_prefix+'trtllm_utils/trtllm_utils.so']=='e442a689064fe3af4ab46510b2d5aaf3597b072949b84f13d14e9e53352109e4'
assert sha(s/'prior-fixture/manifest.json')=='e13a86fab5db5796b22d8dab0f84080b8057b036ca7aef2c7be8832e76505bf1'
assert sha(s/'prior-fixture/fixture.safetensors')=='c30aa761eba827bb0b4eb6de0e99d9b8dda8243dd16d00146f9718d2abb05e8b'
fixture=json.loads((s/'prior-fixture/manifest.json').read_text())
model_cache=s/'model-cache'
checkpoint=json.loads((model_cache/'20261027T000000Z-h100-mlp-selection-checkpoint-verification.json').read_text())
assert checkpoint['status']=='passed' and checkpoint['revision']=='c1899de289a04d12100db370d81485cdf75e47ca'
snapshot=model_cache/'models--Qwen--Qwen3-0.6B/snapshots'/checkpoint['revision']
assert {name:{key:row[key] for key in ('sha256','size_bytes')} for name,row in checkpoint['files'].items()}==fixture['checkpoint_files']
for name,row in fixture['checkpoint_files'].items():
    path=snapshot/name
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''): digest.update(chunk)
    assert path.stat().st_size==row['size_bytes'] and digest.hexdigest()==row['sha256'],name
checkpoint.update(node_cache_root=str(model_cache),node_local_copy=True,copy_verified_files=sorted(fixture['checkpoint_files']))
(s/'checkpoint-verification.json').write_text(json.dumps(checkpoint,indent=2)+'\n')
for path,value in fixture['source']['sglang_source_sha256'].items():
    captured=subprocess.run(['git','-C',str(s/'source'),'show',fixture['source']['sglang_commit']+':'+path],check=True,capture_output=True).stdout
    assert hashlib.sha256(captured).hexdigest()==value and sha(s/'source'/path)==value
assert sha(s/'native-reference.bootstrap.json')=='7c4727aca3d99adcb3fe2f35d06f4408b512baa4fe9a91b975bdf77358fe205d'
restoration=json.loads((s/'native-restoration.json').read_text());assert restoration['status']=='passed'
record=dict(status='passed',restoration_method='archive-exact-bytes-with-recorded-timestamp-restoration',native_restoration_sha256=sha(s/'native-restoration.json'),native_reference_sha256=sha(s/'native-reference.bootstrap.json'),source_files_verified=len(files),input_files=input_files,input_links=links,native_files=native_files,cache_files=cache_files,copy_preserves_mtime_ns=False)
(s/'native-reuse.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'status':'staged','study':str(s),'source_files_verified':len(files),'native_files':len(native_files),'native_reuse_sha256':sha(s/'native-reuse.json')}))
REUSE
nvidia-smi --query-gpu=name,uuid,driver_version,power.limit,memory.used --format=csv
docker image inspect lmsysorg/sglang@sha256:c1f4d9fc6fcf50c2269adbc757ce7b1778883af04155edb888fa9eea3b6648c0 --format '{{.Id}}'
NODE
