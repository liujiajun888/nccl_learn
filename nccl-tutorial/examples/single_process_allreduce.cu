#include <cuda_runtime.h>
#include <nccl.h>

#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#define CUDA_CHECK(call) do { \
  cudaError_t result = (call); \
  if (result != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(result)); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

#define NCCL_CHECK(call) do { \
  ncclResult_t result = (call); \
  if (result != ncclSuccess) { \
    std::fprintf(stderr, "%s:%d NCCL: %s\n", __FILE__, __LINE__, ncclGetErrorString(result)); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

int main(int argc, char** argv) {
  if (argc > 2) {
    std::fprintf(stderr, "Usage: %s [gpu-count, default 2]\n", argv[0]);
    return EXIT_FAILURE;
  }
  long requested = 2;
  if (argc == 2) {
    char* end = nullptr;
    errno = 0;
    requested = std::strtol(argv[1], &end, 10);
    if (errno || end == argv[1] || *end || requested < 1 || requested > INT_MAX) {
      std::fprintf(stderr, "gpu-count must be a positive integer\n");
      return EXIT_FAILURE;
    }
  }
  int visible = 0;
  CUDA_CHECK(cudaGetDeviceCount(&visible));
  if (requested > visible) {
    std::fprintf(stderr, "Requested %ld GPUs, but only %d are visible\n", requested, visible);
    return EXIT_FAILURE;
  }
  const int ndev = static_cast<int>(requested);
  constexpr size_t count = 1024;
  const size_t bytes = count * sizeof(float);
  std::vector<int> devices(ndev);
  std::vector<ncclComm_t> comms(ndev);
  std::vector<cudaStream_t> streams(ndev);
  std::vector<float*> send(ndev), recv(ndev);
  std::vector<std::vector<float>> inputs(ndev, std::vector<float>(count));
  std::vector<float> output(count);

  int runtime_version = 0;
  NCCL_CHECK(ncclGetVersion(&runtime_version));
  std::printf("NCCL header=%d runtime=%d ranks=%d count=%zu\n",
              NCCL_VERSION_CODE, runtime_version, ndev, count);
  for (int r = 0; r < ndev; ++r) {
    devices[r] = r;
    CUDA_CHECK(cudaSetDevice(devices[r]));
    CUDA_CHECK(cudaStreamCreateWithFlags(&streams[r], cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&send[r]), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&recv[r]), bytes));
    for (size_t i = 0; i < count; ++i) inputs[r][i] = static_cast<float>(r + 1);
    CUDA_CHECK(cudaMemcpyAsync(send[r], inputs[r].data(), bytes,
                               cudaMemcpyHostToDevice, streams[r]));
  }
  NCCL_CHECK(ncclCommInitAll(comms.data(), ndev, devices.data()));

  NCCL_CHECK(ncclGroupStart());
  for (int r = 0; r < ndev; ++r) {
    CUDA_CHECK(cudaSetDevice(devices[r]));
    NCCL_CHECK(ncclAllReduce(send[r], recv[r], count, ncclFloat, ncclSum,
                             comms[r], streams[r]));
  }
  NCCL_CHECK(ncclGroupEnd());

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  for (;;) {
    bool all_done = true;
    for (int r = 0; r < ndev; ++r) {
      CUDA_CHECK(cudaSetDevice(devices[r]));
      ncclResult_t state = ncclSuccess;
      NCCL_CHECK(ncclCommGetAsyncError(comms[r], &state));
      NCCL_CHECK(state);
      cudaError_t status = cudaStreamQuery(streams[r]);
      if (status == cudaErrorNotReady) all_done = false;
      else CUDA_CHECK(status);
    }
    if (all_done) break;
    if (std::chrono::steady_clock::now() >= deadline) {
      std::fprintf(stderr, "Timed out waiting for submitted GPU work\n");
      for (int r = 0; r < ndev; ++r) {
        CUDA_CHECK(cudaSetDevice(devices[r]));
        NCCL_CHECK(ncclCommAbort(comms[r]));
      }
      return EXIT_FAILURE;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  const double expected = static_cast<double>(ndev) * (ndev + 1.0) / 2.0;
  size_t wrong = 0;
  for (int r = 0; r < ndev; ++r) {
    CUDA_CHECK(cudaSetDevice(devices[r]));
    CUDA_CHECK(cudaMemcpy(output.data(), recv[r], bytes, cudaMemcpyDeviceToHost));
    size_t rank_wrong = 0;
    for (float value : output) {
      if (static_cast<double>(value) != expected) ++rank_wrong;
    }
    wrong += rank_wrong;
    std::printf("rank=%d device=%d first=%.0f expected=%.0f wrong=%zu\n",
                r, devices[r], output[0], expected, rank_wrong);
    NCCL_CHECK(ncclCommDestroy(comms[r]));
    CUDA_CHECK(cudaFree(send[r]));
    CUDA_CHECK(cudaFree(recv[r]));
    CUDA_CHECK(cudaStreamDestroy(streams[r]));
  }
  std::printf("%s\n", wrong == 0 ? "PASS" : "FAIL");
  return wrong == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
