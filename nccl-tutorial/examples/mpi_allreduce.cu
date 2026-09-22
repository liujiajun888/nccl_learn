#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define MPI_CHECK(call) do { \
  int result = (call); \
  if (result != MPI_SUCCESS) { \
    char text[MPI_MAX_ERROR_STRING]; int length = 0; \
    MPI_Error_string(result, text, &length); \
    std::fprintf(stderr, "%s:%d MPI: %.*s\n", __FILE__, __LINE__, length, text); \
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

#define CUDA_CHECK(call) do { \
  cudaError_t result = (call); \
  if (result != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(result)); \
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

#define NCCL_CHECK(call) do { \
  ncclResult_t result = (call); \
  if (result != ncclSuccess) { \
    std::fprintf(stderr, "%s:%d NCCL: %s\n", __FILE__, __LINE__, ncclGetErrorString(result)); \
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

int main(int argc, char** argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) return EXIT_FAILURE;
  MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
  int rank = 0, nranks = 0;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
  MPI_Comm local_comm;
  MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                MPI_INFO_NULL, &local_comm));
  int local_rank = 0, local_size = 0;
  MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
  MPI_CHECK(MPI_Comm_size(local_comm, &local_size));
  int visible = 0;
  CUDA_CHECK(cudaGetDeviceCount(&visible));
  const int device = visible == 1 ? 0 : local_rank;
  if (device >= visible) {
    std::fprintf(stderr, "rank=%d local_rank=%d has only %d visible GPUs\n",
                 rank, local_rank, visible);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    return EXIT_FAILURE;
  }
  CUDA_CHECK(cudaSetDevice(device));
  constexpr int bus_id_size = 32;
  char bus_id[bus_id_size] = {};
  CUDA_CHECK(cudaDeviceGetPCIBusId(bus_id, bus_id_size, device));
  std::vector<char> bus_ids(static_cast<size_t>(local_size) * bus_id_size);
  MPI_CHECK(MPI_Allgather(bus_id, bus_id_size, MPI_CHAR, bus_ids.data(),
                          bus_id_size, MPI_CHAR, local_comm));
  for (int peer = 0; peer < local_size; ++peer) {
    if (peer != local_rank && std::strcmp(bus_id, bus_ids.data() + peer * bus_id_size) == 0) {
      std::fprintf(stderr, "rank=%d: multiple local ranks selected PCI device %s\n", rank, bus_id);
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      return EXIT_FAILURE;
    }
  }
  char host[MPI_MAX_PROCESSOR_NAME];
  int host_length = 0;
  MPI_CHECK(MPI_Get_processor_name(host, &host_length));
  std::printf("host=%.*s rank=%d/%d local_rank=%d device=%d pci=%s\n",
              host_length, host, rank, nranks, local_rank, device, bus_id);

  ncclUniqueId id;
  if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
  MPI_CHECK(MPI_Bcast(&id, static_cast<int>(sizeof(id)), MPI_BYTE, 0, MPI_COMM_WORLD));
  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm, nranks, id, rank));
  constexpr size_t count = 1024;
  const size_t bytes = count * sizeof(float);
  std::vector<float> input(count, static_cast<float>(rank + 1)), output(count);
  float* send = nullptr;
  float* recv = nullptr;
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&send), bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&recv), bytes));
  CUDA_CHECK(cudaMemcpyAsync(send, input.data(), bytes, cudaMemcpyHostToDevice, stream));
  NCCL_CHECK(ncclAllReduce(send, recv, count, ncclFloat, ncclSum, comm, stream));

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  for (;;) {
    ncclResult_t state = ncclSuccess;
    NCCL_CHECK(ncclCommGetAsyncError(comm, &state));
    NCCL_CHECK(state);
    cudaError_t status = cudaStreamQuery(stream);
    if (status == cudaSuccess) break;
    if (status != cudaErrorNotReady) CUDA_CHECK(status);
    if (std::chrono::steady_clock::now() >= deadline) {
      std::fprintf(stderr, "rank=%d timed out waiting for submitted GPU work\n", rank);
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      return EXIT_FAILURE;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  CUDA_CHECK(cudaMemcpy(output.data(), recv, bytes, cudaMemcpyDeviceToHost));
  const double expected = static_cast<double>(nranks) * (nranks + 1.0) / 2.0;
  unsigned long long wrong = 0, total_wrong = 0;
  for (float value : output) {
    if (static_cast<double>(value) != expected) ++wrong;
  }
  MPI_CHECK(MPI_Allreduce(&wrong, &total_wrong, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD));
  if (rank == 0) {
    std::printf("first=%.0f expected=%.0f total_wrong=%llu %s\n",
                output[0], expected, total_wrong, total_wrong == 0 ? "PASS" : "FAIL");
  }
  NCCL_CHECK(ncclCommDestroy(comm));
  CUDA_CHECK(cudaFree(send));
  CUDA_CHECK(cudaFree(recv));
  CUDA_CHECK(cudaStreamDestroy(stream));
  MPI_CHECK(MPI_Comm_free(&local_comm));
  MPI_CHECK(MPI_Finalize());
  return total_wrong == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
