#!/usr/bin/env bash
# Run on the cluster login host after the research snapshot is frozen.
set -euo pipefail
allocation=${1:?allocation required}
source_commit=${2:?frozen source commit required}
[[ "$allocation" =~ ^[0-9]+$ && "$source_commit" =~ ^[0-9a-f]{40}$ ]] || exit 1
srun --jobid "$allocation" --nodes=1 --ntasks=1 bash -s -- "$allocation" "$source_commit" <<'NODE'
set -euo pipefail
allocation=$1
source_commit=$2
study=/tmp/fi-objective-values-$allocation
durable=/home/lnau/flashinfer-autotune-study
batch=20261024T180000Z-h100-objective-values
pack="$durable/$batch-$source_commit-snapshot.pack"
[[ ! -e "$study" ]] || exit 1
python3 - "$pack" "$source_commit" "$durable/fp8-current-source-private.tar.gz" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);m=json.loads(Path(str(p)+'.json').read_text())
assert m['source_commit']==sys.argv[2] and not m['history_walk'] and not m['lazy_fetch']
assert hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256']
assert hashlib.sha256(Path(sys.argv[3]).read_bytes()).hexdigest()=='209b2de9e76c358b94407202a4670613055e963200184e716b50eb67aad71c91'
PY
mkdir -p "$study/input/current/manifests"
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
tar -xzf "$durable/fp8-current-source-private.tar.gz" -C "$study/input/current"
for name in candidate-manifest complete-source-manifest dependency-manifest cutlass-source-manifest cccl-source-manifest spdlog-source-manifest; do
    mv "$study/input/current/$name.json" "$study/input/current/manifests/$name.json"
done
qualified=test/manual/kernels/results/flashinfer_autotune_selection/20261024T090000Z-h100-current-source-qualification/revision02
printf '%s  %s\n' ae7159aba41c3f26199568f9742ab140e7c20985131dbdbd58d7af7c28246829 "$study/source/$qualified/qualification-tools.tar.gz" | sha256sum -c -
tar -xzf "$study/source/$qualified/qualification-tools.tar.gz" -C "$study/input/current"
git -C "$study/input/current/source" apply "$study/input/current/patches/harness-only.patch"
cp "$pack.json" "$study/source-snapshot.json"
fixture_archive="$durable/20261020T034000Z-h100-fp8-linear-attempt01-full-raw.tgz"
printf '%s  %s\n' 71793735764772b7fa6c4990bd9afd644cf8cbd113c1aa73a7e8e73159675b0e "$fixture_archive" | sha256sum -c -
tar -xzf "$fixture_archive" -C "$study" --strip-components=1 attempt01/fixture
python3 - "$study" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
s=Path(sys.argv[1]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
root=s/'input/current'
assert sha(root/'manifests/candidate-manifest.json')=='9a4c51562d7410d785cc544c004bc4f92486f40311cb5c7a7328c2c1213160d5'
assert sha(root/'tools/container_preflight.py')=='b8d0f6bbc85eaca0e3ef1419ab99a97d5340c96b3da382cba81d7896768287ba'
assert sha(root/'patches/harness-only.patch')=='706ca4d91f3e6ef8c1643977beef72fa39af121e691f9380a175633787fe94ef'
assert sha(s/'fixture/manifest.json')=='e13a86fab5db5796b22d8dab0f84080b8057b036ca7aef2c7be8832e76505bf1'
assert sha(s/'fixture/fixture.safetensors')=='c30aa761eba827bb0b4eb6de0e99d9b8dda8243dd16d00146f9718d2abb05e8b'
fixture=json.loads((s/'fixture/manifest.json').read_text())
for path,value in fixture['source']['sglang_source_sha256'].items():
    old=subprocess.run(['git','-C',str(s/'source'),'show',fixture['source']['sglang_commit']+':'+path],check=True,capture_output=True).stdout
    assert hashlib.sha256(old).hexdigest()==value and sha(s/'source'/path)==value
assert not (root/'source/flashinfer/data').exists()
print(json.dumps({'status':'staged','study':str(s),'source_snapshot':json.loads((s/'source-snapshot.json').read_text())}))
PY
nvidia-smi --query-gpu=name,uuid,driver_version,power.limit,memory.used --format=csv
docker image inspect lmsysorg/sglang@sha256:c1f4d9fc6fcf50c2269adbc757ce7b1778883af04155edb888fa9eea3b6648c0 --format '{{.Id}}'
NODE
