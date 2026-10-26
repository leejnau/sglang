#!/usr/bin/env bash
# Stage a frozen snapshot and copy the previous, qualified native build unchanged.
set -euo pipefail
allocation=${1:?allocation required}
source_commit=${2:?frozen source commit required}
[[ "$allocation" =~ ^[0-9]+$ && "$source_commit" =~ ^[0-9a-f]{40}$ ]] || exit 1
srun --jobid "$allocation" --nodes=1 --ntasks=1 bash -s -- "$allocation" "$source_commit" <<'NODE'
set -euo pipefail
allocation=$1
source_commit=$2
[[ "$(hostname -s)" == 1u1g-gen-0100 ]] || exit 1
study=/tmp/fi-finalist-confirmation-$allocation
previous=/tmp/fi-fixed-pair-2861566
durable=/home/lnau/flashinfer-autotune-study
batch=20261026T230000Z-h100-finalist-confirmation
pack="$durable/$batch-$source_commit-snapshot.pack"
[[ ! -e "$study" && -d "$previous/input/current/source" && -d "$previous/runtime-attempt01/cache" ]] || exit 1
python3 - "$pack" "$source_commit" "$durable/fp8-current-source-private.tar.gz" "$previous" <<'VERIFY'
import hashlib,json,sys
from pathlib import Path
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=Path(sys.argv[1]);m=json.loads(Path(str(p)+'.json').read_text());old=Path(sys.argv[4])
assert m['source_commit']==sys.argv[2] and not m['history_walk'] and not m['lazy_fetch']
assert sha(p)==m['sha256']
assert sha(Path(sys.argv[3]))=='209b2de9e76c358b94407202a4670613055e963200184e716b50eb67aad71c91'
r=old/'results/fixed-pair/attempt01'
assert sha(r/'block0-auto_synthetic-timing.bootstrap.json')=='7c4727aca3d99adcb3fe2f35d06f4408b512baa4fe9a91b975bdf77358fe205d'
assert sha(r/'preservation-manifest-timing.json')=='40e684976202ae6eca47e462e5da92efa35ccd2fa4cf1ccaaa8db0716936b6df'
for row in json.loads((r/'preservation-manifest-timing.json').read_text())['files']:
    prefix='native-build-cache/timing/'
    if row['path'].startswith(prefix):
        path=old/'runtime-attempt01/cache'/row['path'][len(prefix):]
        assert path.is_file() and path.stat().st_size==row['size'] and sha(path)==row['sha256'],str(path)
VERIFY
mkdir -p "$study/input" "$study/runtime-attempt01"
cp -a "$previous/input/current" "$study/input/current"
cp -a "$previous/runtime-attempt01/cache" "$study/runtime-attempt01/cache"
cp -a "$previous/fixture" "$study/fixture"
cp "$previous/results/fixed-pair/attempt01/block0-auto_synthetic-timing.bootstrap.json" "$study/native-reference.bootstrap.json"
export GIT_NO_LAZY_FETCH=1 GIT_TERMINAL_PROMPT=0
git init -q "$study/source"
git -C "$study/source" index-pack --stdin < "$pack" > "$study/index-pack.txt"
capture=a29b31850c7b68cafde022614b00ab01d6ff5cde
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
python3 - "$study" "$previous" <<'REUSE'
import hashlib,json,os,subprocess,sys
from pathlib import Path
s,old=map(Path,sys.argv[1:]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
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
def copied_tree(new,previous):
    result={}
    for path in sorted(new.rglob('*')):
        rel=path.relative_to(new);other=previous/rel
        if path.is_symlink():
            assert other.is_symlink() and os.readlink(path)==os.readlink(other),str(rel)
        elif path.is_file():
            value=sha(path)
            assert value==sha(other) and path.stat().st_mtime_ns==other.stat().st_mtime_ns,str(rel)
            result[str(rel)]=value
    assert set(result)=={str(p.relative_to(previous)) for p in previous.rglob('*') if p.is_file() and not p.is_symlink()}
    return result
input_files=copied_tree(root,old/'input/current')
links={str(p.relative_to(root)):os.readlink(p) for p in root.rglob('*') if p.is_symlink()}
cache=s/'runtime-attempt01/cache'
cache_files=copied_tree(cache,old/'runtime-attempt01/cache')
native_prefix='current/.cache/flashinfer/0.0.0+unknown/90a/cached_ops/'
native_files={p:h for p,h in cache_files.items() if p.startswith((native_prefix+'gemm/',native_prefix+'trtllm_utils/'))}
assert native_files[native_prefix+'gemm/gemm.so']=='72690cacca4ce1d69ccaa6b9190475483d984bd58de0962d06c874c801c678d6'
assert native_files[native_prefix+'trtllm_utils/trtllm_utils.so']=='e442a689064fe3af4ab46510b2d5aaf3597b072949b84f13d14e9e53352109e4'
assert sha(s/'fixture/manifest.json')=='e13a86fab5db5796b22d8dab0f84080b8057b036ca7aef2c7be8832e76505bf1'
assert sha(s/'fixture/fixture.safetensors')=='c30aa761eba827bb0b4eb6de0e99d9b8dda8243dd16d00146f9718d2abb05e8b'
fixture=json.loads((s/'fixture/manifest.json').read_text())
for path,value in fixture['source']['sglang_source_sha256'].items():
    captured=subprocess.run(['git','-C',str(s/'source'),'show',fixture['source']['sglang_commit']+':'+path],check=True,capture_output=True).stdout
    assert hashlib.sha256(captured).hexdigest()==value and sha(s/'source'/path)==value
record=dict(status='passed',previous_study=str(old),native_reference_sha256=sha(s/'native-reference.bootstrap.json'),source_files_verified=len(files),input_files=input_files,input_links=links,native_files=native_files,cache_files=cache_files,copy_preserves_mtime_ns=True)
(s/'native-reuse.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'status':'staged','study':str(s),'source_files_verified':len(files),'native_files':len(native_files),'native_reuse_sha256':sha(s/'native-reuse.json')}))
REUSE
nvidia-smi --query-gpu=name,uuid,driver_version,power.limit,memory.used --format=csv
docker image inspect lmsysorg/sglang@sha256:c1f4d9fc6fcf50c2269adbc757ce7b1778883af04155edb888fa9eea3b6648c0 --format '{{.Id}}'
NODE
