#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/libfi_native_probe.so" >&2
  exit 2
fi
probe_output=$1
if [[ "$probe_output" != /* ]]; then
  echo "The output library path must be absolute" >&2
  exit 2
fi
probe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cuda_include=${FI_PROBE_CUDA_INCLUDE:-/usr/local/cuda/include}
if [[ ! -f "$cuda_include/cublasLt.h" ]]; then
  echo "Set FI_PROBE_CUDA_INCLUDE to the directory containing cublasLt.h" >&2
  exit 2
fi
compiler=${CXX:-g++}
"$compiler" -std=c++17 -O2 -Wall -Wextra -Werror -fPIC -shared \
  -I"$cuda_include" "$probe_dir/flashinfer_autotune_descriptor_native.cpp" -ldl -pthread \
  -o "$probe_output"
sha256sum "$probe_dir/flashinfer_autotune_descriptor_native.cpp" "$probe_output"
