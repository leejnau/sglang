// Original diagnostic code. No production behavior or candidate selection
// changes. Build against the CUDA headers in the experiment container.
// LD_PRELOAD this library only for short observation passes, not the primary
// timing workers.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cublasLt.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <mutex>
#include <string>
#include <time.h>

namespace {
enum Api {
  kDescCreate,
  kDescDestroy,
  kDescSet,
  kLayoutCreate,
  kLayoutDestroy,
  kLayoutSet,
  kPreferenceCreate,
  kPreferenceDestroy,
  kPreferenceSet,
  kHeuristic,
  kAlgoCheck,
  kMatmul,
  kApiCount,
};
const char *const kApiNames[] = {"matmul_desc_create",
                                 "matmul_desc_destroy",
                                 "matmul_desc_set_attribute",
                                 "matrix_layout_create",
                                 "matrix_layout_destroy",
                                 "matrix_layout_set_attribute",
                                 "preference_create",
                                 "preference_destroy",
                                 "preference_set_attribute",
                                 "heuristic",
                                 "algo_check",
                                 "matmul"};
std::atomic<bool> enabled{false};
std::array<std::atomic<uint64_t>, kApiCount> calls{};
std::array<std::atomic<uint64_t>, kApiCount> ns{};
std::array<std::atomic<uint64_t>, kApiCount> failures{};
std::mutex output_mutex;
std::string phase;
FILE *output = nullptr;

uint64_t now_ns() {
  timespec value{};
  clock_gettime(CLOCK_MONOTONIC_RAW, &value);
  return uint64_t(value.tv_sec) * 1000000000ULL + uint64_t(value.tv_nsec);
}

void *resolve(const char *name) {
  void *symbol = dlsym(RTLD_NEXT, name);
  if (symbol != nullptr)
    return symbol;
  // Python extensions can load their cuBLASLt dependency with RTLD_LOCAL.
  for (const char *soname :
       {"libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"}) {
    void *handle = dlopen(soname, RTLD_LAZY | RTLD_NOLOAD);
    if (handle != nullptr) {
      symbol = dlsym(handle, name);
      dlclose(handle);
      if (symbol != nullptr)
        return symbol;
    }
  }
  std::fprintf(stderr, "native probe cannot resolve %s: %s\n", name, dlerror());
  std::abort();
}

void add(Api api, uint64_t elapsed_ns, cublasStatus_t result) {
  calls[api].fetch_add(1, std::memory_order_relaxed);
  ns[api].fetch_add(elapsed_ns, std::memory_order_relaxed);
  if (result != CUBLAS_STATUS_SUCCESS)
    failures[api].fetch_add(1, std::memory_order_relaxed);
}

void write_algorithm(const cublasLtMatmulAlgo_t *algo) {
  if (algo == nullptr) {
    std::fputs("null", output);
    return;
  }
  std::fputs("{\"raw_hex\":\"", output);
  const auto *bytes = reinterpret_cast<const unsigned char *>(algo);
  for (size_t i = 0; i < sizeof(*algo); ++i)
    std::fprintf(output, "%02x", bytes[i]);
  // This is a selected public-attribute view, not the complete opaque config.
  // Equal descriptors corroborate identity within this process. Different raw
  // bytes alone cannot establish different kernels: reserved fields may differ.
  std::fprintf(output,
               "\",\"raw_bytes\":%zu,\"config_scope\":\"selected_public_"
               "attributes\",\"config\":{",
               sizeof(*algo));
  using Get = decltype(&cublasLtMatmulAlgoConfigGetAttribute);
  static Get get =
      reinterpret_cast<Get>(resolve("cublasLtMatmulAlgoConfigGetAttribute"));
  struct Attribute {
    const char *name;
    cublasLtMatmulAlgoConfigAttributes_t attr;
  };
  const Attribute attributes[] = {
      {"id", CUBLASLT_ALGO_CONFIG_ID},
      {"tile_id", CUBLASLT_ALGO_CONFIG_TILE_ID},
      {"split_k", CUBLASLT_ALGO_CONFIG_SPLITK_NUM},
      {"reduction_scheme", CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME},
      {"cta_swizzling", CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING},
      {"custom_option", CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION},
      {"stages_id", CUBLASLT_ALGO_CONFIG_STAGES_ID},
  };
  for (size_t i = 0; i < std::size(attributes); ++i) {
    uint32_t value = 0;
    size_t written = 0;
    cublasStatus_t status =
        get(algo, attributes[i].attr, &value, sizeof(value), &written);
    // The public attributes above are scalar 32-bit values on this API.
    // Preserve status/size; a rejected query is not silently treated as an ID
    // of zero.
    std::fprintf(output,
                 "%s\"%s\":{\"status\":%d,\"bytes\":%zu,\"value\":%llu}",
                 i ? "," : "", attributes[i].name, int(status), written,
                 static_cast<unsigned long long>(value));
  }
  std::fputs("}}", output);
}

void emit_summary_locked() {
  if (output == nullptr || phase.empty())
    return;
  std::fprintf(
      output,
      "{\"event\":\"phase_summary\",\"phase\":\"%s\",\"pid\":%d,\"apis\":{",
      phase.c_str(), int(getpid()));
  for (size_t i = 0; i < kApiCount; ++i) {
    std::fprintf(output,
                 "%s\"%s\":{\"calls\":%llu,\"host_ns\":%llu,\"failures\":%llu}",
                 i ? "," : "", kApiNames[i],
                 static_cast<unsigned long long>(calls[i].load()),
                 static_cast<unsigned long long>(ns[i].load()),
                 static_cast<unsigned long long>(failures[i].load()));
  }
  std::fputs("}}\n", output);
  std::fflush(output);
}
} // namespace

// Call only with no in-flight calls from another thread. Labels are sanitized
// to keep each output record valid JSON without accepting arbitrary text.
extern "C" void fi_probe_set_phase(const char *label) {
  enabled.store(false, std::memory_order_release);
  std::lock_guard<std::mutex> lock(output_mutex);
  emit_summary_locked();
  phase.clear();
  if (label == nullptr || label[0] == '\0')
    return;
  if (output == nullptr) {
    const char *path = std::getenv("FI_NATIVE_PROBE_OUTPUT");
    if (path == nullptr || path[0] == '\0') {
      std::fputs("FI_NATIVE_PROBE_OUTPUT is required when enabling the probe\n",
                 stderr);
      std::abort();
    }
    // Each worker gets its own new output, preventing accidental append/reuse.
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0 || (output = fdopen(fd, "w")) == nullptr) {
      std::perror("native probe output");
      std::abort();
    }
  }
  for (const char *p = label; *p != '\0'; ++p) {
    char ch = *p;
    phase.push_back((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                            (ch >= '0' && ch <= '9') || ch == '_' ||
                            ch == '.' || ch == '-'
                        ? ch
                        : '_');
  }
  for (size_t i = 0; i < kApiCount; ++i) {
    calls[i].store(0);
    ns[i].store(0);
    failures[i].store(0);
  }
  std::fprintf(output,
               "{\"event\":\"phase_begin\",\"phase\":\"%s\",\"pid\":%d,"
               "\"clock\":\"CLOCK_MONOTONIC_RAW\",\"algo_bytes\":%zu}\n",
               phase.c_str(), int(getpid()), sizeof(cublasLtMatmulAlgo_t));
  std::fflush(output);
  enabled.store(true, std::memory_order_release);
}

// The timing interval contains only the real API call. Resolve/log/config-read
// overhead is excluded from each host_ns value, but perturbs surrounding work:
// this is diagnostic evidence, never a replacement for uninstrumented timing.
#define FI_SIMPLE_WRAPPER(name, api, parameters, arguments)                    \
  extern "C" cublasStatus_t CUBLASWINAPI name parameters {                     \
    using Function = decltype(&name);                                          \
    static Function real = reinterpret_cast<Function>(resolve(#name));         \
    if (!enabled.load(std::memory_order_acquire))                              \
      return real arguments;                                                   \
    const uint64_t started = now_ns();                                         \
    const cublasStatus_t result = real arguments;                              \
    add(api, now_ns() - started, result);                                      \
    return result;                                                             \
  }

FI_SIMPLE_WRAPPER(cublasLtMatmulDescCreate, kDescCreate,
                  (cublasLtMatmulDesc_t * desc, cublasComputeType_t compute,
                   cudaDataType_t scale),
                  (desc, compute, scale))
FI_SIMPLE_WRAPPER(cublasLtMatmulDescDestroy, kDescDestroy,
                  (cublasLtMatmulDesc_t desc), (desc))
FI_SIMPLE_WRAPPER(cublasLtMatmulDescSetAttribute, kDescSet,
                  (cublasLtMatmulDesc_t desc,
                   cublasLtMatmulDescAttributes_t attr, const void *buffer,
                   size_t size),
                  (desc, attr, buffer, size))
FI_SIMPLE_WRAPPER(cublasLtMatrixLayoutCreate, kLayoutCreate,
                  (cublasLtMatrixLayout_t * layout, cudaDataType_t type,
                   uint64_t rows, uint64_t cols, int64_t ld),
                  (layout, type, rows, cols, ld))
FI_SIMPLE_WRAPPER(cublasLtMatrixLayoutDestroy, kLayoutDestroy,
                  (cublasLtMatrixLayout_t layout), (layout))
FI_SIMPLE_WRAPPER(cublasLtMatrixLayoutSetAttribute, kLayoutSet,
                  (cublasLtMatrixLayout_t layout,
                   cublasLtMatrixLayoutAttribute_t attr, const void *buffer,
                   size_t size),
                  (layout, attr, buffer, size))
FI_SIMPLE_WRAPPER(cublasLtMatmulPreferenceCreate, kPreferenceCreate,
                  (cublasLtMatmulPreference_t * preference), (preference))
FI_SIMPLE_WRAPPER(cublasLtMatmulPreferenceDestroy, kPreferenceDestroy,
                  (cublasLtMatmulPreference_t preference), (preference))
FI_SIMPLE_WRAPPER(cublasLtMatmulPreferenceSetAttribute, kPreferenceSet,
                  (cublasLtMatmulPreference_t preference,
                   cublasLtMatmulPreferenceAttributes_t attr,
                   const void *buffer, size_t size),
                  (preference, attr, buffer, size))

extern "C" cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoGetHeuristic(
    cublasLtHandle_t handle, cublasLtMatmulDesc_t operation,
    cublasLtMatrixLayout_t a, cublasLtMatrixLayout_t b,
    cublasLtMatrixLayout_t c, cublasLtMatrixLayout_t d,
    cublasLtMatmulPreference_t preference, int requested,
    cublasLtMatmulHeuristicResult_t *results, int *returned) {
  using Function = decltype(&cublasLtMatmulAlgoGetHeuristic);
  static Function real =
      reinterpret_cast<Function>(resolve("cublasLtMatmulAlgoGetHeuristic"));
  if (!enabled.load(std::memory_order_acquire)) {
    return real(handle, operation, a, b, c, d, preference, requested, results,
                returned);
  }
  uint64_t started = now_ns();
  cublasStatus_t status = real(handle, operation, a, b, c, d, preference,
                               requested, results, returned);
  uint64_t elapsed = now_ns() - started;
  add(kHeuristic, elapsed, status);
  std::lock_guard<std::mutex> lock(output_mutex);
  const int count =
      status == CUBLAS_STATUS_SUCCESS && returned != nullptr ? *returned : 0;
  std::fprintf(
      output,
      "{\"event\":\"heuristic\",\"phase\":\"%s\",\"pid\":%d,\"host_ns\":%llu,"
      "\"status\":%d,\"requested\":%d,\"returned\":%d,\"algorithms\":[",
      phase.c_str(), int(getpid()), static_cast<unsigned long long>(elapsed),
      int(status), requested, count);
  for (int i = 0; results != nullptr && i < std::min(count, requested); ++i) {
    std::fprintf(
        output,
        "%s{\"ordinal\":%d,\"state\":%d,\"workspace_bytes\":%zu,\"algo\":",
        i ? "," : "", i, int(results[i].state), results[i].workspaceSize);
    write_algorithm(&results[i].algo);
    std::fputs("}", output);
  }
  std::fputs("]}\n", output);
  std::fflush(output);
  return status;
}

extern "C" cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoCheck(
    cublasLtHandle_t handle, cublasLtMatmulDesc_t operation,
    cublasLtMatrixLayout_t a, cublasLtMatrixLayout_t b,
    cublasLtMatrixLayout_t c, cublasLtMatrixLayout_t d,
    const cublasLtMatmulAlgo_t *algo, cublasLtMatmulHeuristicResult_t *result) {
  using Function = decltype(&cublasLtMatmulAlgoCheck);
  static Function real =
      reinterpret_cast<Function>(resolve("cublasLtMatmulAlgoCheck"));
  if (!enabled.load(std::memory_order_acquire))
    return real(handle, operation, a, b, c, d, algo, result);
  uint64_t started = now_ns();
  cublasStatus_t status = real(handle, operation, a, b, c, d, algo, result);
  uint64_t elapsed = now_ns() - started;
  add(kAlgoCheck, elapsed, status);
  std::lock_guard<std::mutex> lock(output_mutex);
  std::fprintf(
      output,
      "{\"event\":\"algo_check\",\"phase\":\"%s\",\"pid\":%d,\"host_ns\":%llu,"
      "\"status\":%d,\"result\":",
      phase.c_str(), int(getpid()), static_cast<unsigned long long>(elapsed),
      int(status));
  // A successful API call may still reject this descriptor for the problem.
  // Preserve the result state and workspace requirement independently, so a
  // fallback cannot masquerade as a successfully validated candidate.
  if (status == CUBLAS_STATUS_SUCCESS && result != nullptr) {
    std::fprintf(output, "{\"state\":%d,\"workspace_bytes\":%zu}",
                 int(result->state), result->workspaceSize);
  } else {
    std::fputs("null", output);
  }
  std::fputs(",\"algo\":", output);
  write_algorithm(algo);
  std::fputs("}\n", output);
  std::fflush(output);
  return status;
}

extern "C" cublasStatus_t CUBLASWINAPI cublasLtMatmul(
    cublasLtHandle_t handle, cublasLtMatmulDesc_t operation, const void *alpha,
    const void *a, cublasLtMatrixLayout_t a_layout, const void *b,
    cublasLtMatrixLayout_t b_layout, const void *beta, const void *c,
    cublasLtMatrixLayout_t c_layout, void *d, cublasLtMatrixLayout_t d_layout,
    const cublasLtMatmulAlgo_t *algo, void *workspace, size_t workspace_size,
    cudaStream_t stream) {
  using Function = decltype(&cublasLtMatmul);
  static Function real = reinterpret_cast<Function>(resolve("cublasLtMatmul"));
  if (!enabled.load(std::memory_order_acquire)) {
    return real(handle, operation, alpha, a, a_layout, b, b_layout, beta, c,
                c_layout, d, d_layout, algo, workspace, workspace_size, stream);
  }
  uint64_t started = now_ns();
  cublasStatus_t status =
      real(handle, operation, alpha, a, a_layout, b, b_layout, beta, c,
           c_layout, d, d_layout, algo, workspace, workspace_size, stream);
  uint64_t elapsed = now_ns() - started;
  add(kMatmul, elapsed, status);
  std::lock_guard<std::mutex> lock(output_mutex);
  std::fprintf(
      output,
      "{\"event\":\"matmul\",\"phase\":\"%s\",\"pid\":%d,\"host_ns\":%llu,"
      "\"status\":%d,\"workspace_bytes\":%zu,\"algo\":",
      phase.c_str(), int(getpid()), static_cast<unsigned long long>(elapsed),
      int(status), workspace_size);
  write_algorithm(algo);
  std::fputs("}\n", output);
  std::fflush(output);
  return status;
}
