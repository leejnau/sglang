#!/usr/bin/env bash
# Restore exact qualified source/native bytes from preserved private archives.
set -euo pipefail
allocation=${1:?allocation required}
source_commit=${2:?frozen source commit required}
pack=${3:?verified snapshot pack on this login host required}
model_root=${4:?verified checkpoint staging root required}
[[ "$allocation" =~ ^[0-9]+$ && "$source_commit" =~ ^[0-9a-f]{40}$ ]] || exit 1
study=/tmp/fi-scoped-cache-$allocation
[[ -f "$pack.json" && -f "$model_root/model-manifest.json" ]] || exit 1
read -r model_sha stream_sha < <(python3 - "$(dirname "$0")" "$model_root" <<'BEFORE'
import hashlib,json,sys
from pathlib import Path
frozen,root=map(Path,sys.argv[1:]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((frozen/'plan.json').read_text())
a=sha(root/'model-manifest.json');b=sha(root/'model-stream-plan.json')
assert a==sha(frozen/'model-manifest.json')==plan['model']['manifest_sha256']
assert b==plan['model']['stream_plan_sha256']
print(a,b)
BEFORE
)
srun --input=none --jobid "$allocation" --nodes=1 --ntasks=1 mkdir "$study"
srun --input=none --jobid "$allocation" --nodes=1 --ntasks=1 mkdir "$study/model-incoming"
cat "$pack" | srun --jobid "$allocation" --nodes=1 --ntasks=1 tee "$study/source.pack" > /dev/null
cat "$pack.json" | srun --jobid "$allocation" --nodes=1 --ntasks=1 tee "$study/source.pack.json" > /dev/null
python3 "$(dirname "$0")/stream_model.py" --root "$model_root" --expected-manifest "$model_sha" --expected-plan "$stream_sha" | srun --jobid "$allocation" --nodes=1 --ntasks=1 tar -C "$study/model-incoming" -xf -
srun --jobid "$allocation" --nodes=1 --ntasks=1 bash -s -- "$allocation" "$source_commit" <<'NODE'
set -euo pipefail
allocation=$1
source_commit=$2
case "$(hostname -s)" in 1u1g-gen-0100|a1u1g-mil-0836|2u2g-gen-0664) ;; *) exit 1 ;; esac
study=/tmp/fi-scoped-cache-$allocation
durable=/home/lnau/flashinfer-autotune-study
batch=20261027T174000Z-h100-scoped-skips
pack="$study/source.pack"
archive="$durable/20261026T220000Z-h100-fixed-pair-attempt01-timing-full-raw.tgz"
[[ -d "$study" && ! -e "$study/source" ]] || exit 1
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
git init -q "$study/source"
git -C "$study/source" index-pack --stdin < "$pack" > "$study/index-pack.txt"
capture=fb57a74b14b03ec5a35cfc4b43828e46e11261e3
printf '%s\n' "$source_commit" "$capture" > "$study/source/.git/shallow"
git -C "$study/source" config core.sparseCheckout true
git -C "$study/source" config core.sparseCheckoutCone false
printf '%s\n' /python/ /test/manual/kernels/ /.gitignore /pyproject.toml > "$study/source/.git/info/sparse-checkout"
git -C "$study/source" update-ref refs/heads/frozen "$source_commit"
git -C "$study/source" update-ref refs/heads/capture "$capture"
git -C "$study/source" symbolic-ref HEAD refs/heads/frozen
git -C "$study/source" read-tree -mu HEAD
[[ "$(git -C "$study/source" rev-parse HEAD)" == "$source_commit" && -z "$(git -C "$study/source" status --porcelain)" ]] || exit 1
cp "$pack.json" "$study/source-snapshot.json"
python3 - "$study" "$batch" <<'MODEL'
import hashlib,json,sys
from pathlib import Path
s=Path(sys.argv[1]);batch=sys.argv[2];incoming=s/'model-incoming'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while data:=f.read(8*1024*1024):h.update(data)
    return h.hexdigest()
plan=json.loads((s/'source/test/manual/kernels/results/flashinfer_autotune_selection'/batch/'plan.json').read_text())
assert sha(incoming/'model-manifest.json')==plan['model']['manifest_sha256']
assert sha(incoming/'model-manifest.json')==sha(s/'source/test/manual/kernels/results/flashinfer_autotune_selection'/batch/'model-manifest.json')
assert sha(incoming/'model-stream-plan.json')==plan['model']['stream_plan_sha256']
m=json.loads((incoming/'model-manifest.json').read_text())
assert {p.name for p in (incoming/'model').iterdir()}=={r['path'] for r in m['files']}
for row in m['files']:
    p=incoming/'model'/row['path'];assert p.is_file() and not p.is_symlink()
    assert p.stat().st_size==row['size_bytes'] and sha(p)==row['sha256']
assert not (s/'model').exists()
(incoming/'model').rename(s/'model')
for name in ('model-manifest.json','model-stream-plan.json'):(incoming/name).rename(s/name)
print(json.dumps({'model_node_verification':'passed','files':len(m['files'])}))
MODEL
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
tar -xOf "$archive" attempt01/block0-auto_synthetic-timing.bootstrap.json > "$study/native-reference.bootstrap.json"
tar -xOf "$archive" attempt01/preservation-manifest-timing.json > "$study/native-preservation.json"
python3 "$study/source/test/manual/kernels/results/flashinfer_autotune_selection/20261027T020000Z-h100-mlp-confirmation/restore_native_cache.py" --study "$study" --output "$study/native-restoration.json"
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
assert sha(root/'source/flashinfer/autotuner/autotuner.py')=='0db42ad90d1a3fec6f81509a2c520c1892cea0a433f5dbd7f43468a2a10243ca'
model=json.loads((s/'model-manifest.json').read_text())
assert model['status']=='passed' and model['revision']=='42d9515ebd69eea3a87351d079c671c3c5ff0a31'
assert {p.name for p in (s/'model').iterdir()}=={row['path'] for row in model['files']}
for row in model['files']:
    p=s/'model'/row['path'];assert p.stat().st_size==row['size_bytes'] and sha(p)==row['sha256']
assert sha(s/'native-reference.bootstrap.json')=='7c4727aca3d99adcb3fe2f35d06f4408b512baa4fe9a91b975bdf77358fe205d'
restoration=json.loads((s/'native-restoration.json').read_text());assert restoration['status']=='passed'
record=dict(status='passed',restoration_method='archive-exact-bytes-with-recorded-timestamp-restoration',native_restoration_sha256=sha(s/'native-restoration.json'),native_reference_sha256=sha(s/'native-reference.bootstrap.json'),source_files_verified=len(files),input_files=input_files,input_links=links,native_files=native_files,cache_files=cache_files,copy_preserves_mtime_ns=False)
(s/'native-reuse.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'status':'staged','study':str(s),'source_files_verified':len(files),'native_files':len(native_files),'native_reuse_sha256':sha(s/'native-reuse.json')}))
REUSE
nvidia-smi --query-gpu=name,uuid,driver_version,power.limit,memory.used --format=csv
docker image inspect lmsysorg/sglang@sha256:c1f4d9fc6fcf50c2269adbc757ce7b1778883af04155edb888fa9eea3b6648c0 --format '{{.Id}}'
NODE
