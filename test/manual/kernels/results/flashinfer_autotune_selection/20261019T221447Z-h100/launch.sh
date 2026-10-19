#!/usr/bin/env bash
set -euo pipefail

# Run inside an exclusive, single-H100 allocation. The caller stages a clean,
# self-contained Git checkout at STUDY_DIR/source before invoking this script.
study_dir="$(realpath "${1:?usage: launch.sh STUDY_DIR SOURCE_COMMIT}")"
execution_commit="${2:?the exact clean checkout commit is required}"
batch_id=20261019T221447Z-h100
image=lmsysorg/sglang@sha256:c1f4d9fc6fcf50c2269adbc757ce7b1778883af04155edb888fa9eea3b6648c0
source_dir="$study_dir/source"
results_dir="$study_dir/results/$batch_id"
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
    --workdir /workspace --entrypoint /opt/sglang/bin/python
)

active_container="fi-selection-${batch_id,,}-preflight"
docker run --name "$active_container" "${docker_args[@]}" -i "$image" - > "$results_dir/preflight.json" 2> "$results_dir/preflight.log" <<'PY'
import hashlib
import inspect
import json
import os
import platform
import subprocess
from pathlib import Path

import flashinfer
import torch
from flashinfer.autotuner import AutoTuner, autotune
from sglang.srt.layers.logits_processor import autotune_dummy_run_mode
from sglang.srt.layers.moe.flashinfer_megamoe_autotune import megamoe_autotune_context
from sglang.srt.model_executor.runner import flashinfer_autotune as wrapper

root = Path('/workspace')
plan_path = root / 'test/manual/kernels/results/flashinfer_autotune_selection/20261019T221447Z-h100/plan.json'
plan = json.loads(plan_path.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip()
assert commit == os.environ['EXPECTED_SOURCE_COMMIT']
assert platform.machine() == 'x86_64'
assert torch.__version__ == '2.14.1+cu130'
assert torch.version.cuda == '13.0'
assert flashinfer.__version__ == '0.7.2'
assert torch.cuda.device_count() == 1
assert torch.cuda.get_device_capability() == (9,0)
assert torch.cuda.get_device_name() == plan['expected_environment']['gpu']
assert str(torch.cuda.get_device_properties(0).uuid) == plan['expected_environment']['physical_gpu_uuid']
assert subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip() == plan['expected_environment']['driver']
assert Path(wrapper.__file__).resolve() == root / 'python/sglang/srt/model_executor/runner/flashinfer_autotune.py'
for relative, expected in plan['source_sha256'].items():
    assert sha(root / relative) == expected, relative
tuner_sha = sha(Path(inspect.getfile(AutoTuner)))
assert tuner_sha == plan['flashinfer_tuner_sha256']
torch.manual_seed(17)
torch.backends.cuda.matmul.allow_tf32 = False
a = torch.randn(1,128,1024,device='cuda').to(torch.float8_e4m3fn)
b = torch.randn(1,1024,1024,device='cuda').to(torch.float8_e4m3fn).transpose(1,2)
sa = torch.ones(1,device='cuda')
sb = torch.ones(1,device='cuda')
out = torch.empty(1,128,1024,device='cuda',dtype=torch.bfloat16)
with autotune(False):
    flashinfer.bmm_fp8(a,b,sa,sb,torch.bfloat16,out=out,backend='cublas')
torch.cuda.synchronize()
torch.testing.assert_close(out.float(),torch.bmm(a.float(),b.float()),rtol=.02,atol=.125)
print(json.dumps({'status':'success','source_commit':commit,'plan_sha256':sha(plan_path),
    'source_sha256':plan['source_sha256'],'flashinfer_tuner_sha256':tuner_sha,
    'torch':torch.__version__,'torch_cuda':torch.version.cuda,'flashinfer':flashinfer.__version__,
    'gpu':torch.cuda.get_device_name(),'physical_gpu_uuid':str(torch.cuda.get_device_properties(0).uuid),
    'fp8_numerical_passed':True},indent=2))
PY
active_container=
python3 -m json.tool "$results_dir/preflight.json" >/dev/null

overall_status=0
for m in 128 144; do
    active_container="fi-selection-${batch_id,,}-m$m"
    if docker run --name "$active_container" "${docker_args[@]}" "$image" \
        test/manual/kernels/bench_flashinfer_autotune_selection.py \
        --output-dir "/study/results/$batch_id/m$m" \
        --m "$m" --n 1024 --k 1024 --trials 3 --seed 17 \
        --samples 15 --iterations 20 --warmup-ms 500 --min-sample-ms 20 \
        --timeout 600 --oracle --run > "$results_dir/m$m.log" 2>&1; then
        case_status=0
    else
        case_status=$?
        overall_status=1
    fi
    active_container=
    printf '%s\n' "$case_status" > "$results_dir/m$m.exit-code"
    cat "$results_dir/m$m.log"
done
nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,power.limit,temperature.gpu,memory.total,memory.used --format=csv > "$results_dir/gpu-after.csv"
exit "$overall_status"
