#!/usr/bin/env bash
set -euo pipefail
study=$(realpath -m "${1:?new study path required}")
source_commit=${2:?frozen source commit required}
pack=$(realpath "${3:?source pack required}")
durable=$(realpath "${4:?preserved archive directory required}")
# The prior restorer is extracted from this frozen pack, not a mutable checkout.
restore=$(mktemp -d)
trap 'rm -rf "$restore"' EXIT
git init -q "$restore"
git -C "$restore" index-pack --stdin < "$pack" > "$restore/index-pack.txt"
GIT_NO_LAZY_FETCH=1 git -C "$restore" show "$source_commit:test/manual/kernels/results/flashinfer_autotune_selection/20261028T010000Z-h100-delay-binding/stage.sh" > "$restore/stage.sh"
bash "$restore/stage.sh" "$study" "$source_commit" "$pack" "$durable"
python3 - "$study" <<'VERIFY'
import hashlib,json,sys
from pathlib import Path
study=Path(sys.argv[1]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record=json.loads((study/'native-reuse-patched.json').read_text())
wrapper=study/'input/patched/source/flashinfer/tllm_utils.py'
assert sha(wrapper)=='4a41c26b8dc5537707ae292c75d6069e59fae73ed881f73d083c15b5a0eb2722'
assert record['declared_overlay']['arm']=='patched'
with (study/'fixed-source.json').open('x') as handle:
 json.dump({'status':'passed','binding_arm':'patched','tuning_arms':['v1','v2'],'wrapper_sha256':sha(wrapper),'reuse_manifest_sha256':sha(study/'native-reuse-patched.json'),'input_files':len(record['input_files']),'same_input_tree_for_both_arms':True},handle,indent=2);handle.write('\n')
VERIFY
