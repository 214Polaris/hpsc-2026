#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
              cudaGetErrorString(err));                                      \
      return 1;                                                              \
    }                                                                        \
  } while (0)

__global__ void clear_bucket(int *bucket, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < range)
    bucket[i] = 0;
}

__global__ void count_bucket(const int *key, int *bucket, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    atomicAdd(&bucket[key[i]], 1);
}

__global__ void make_offsets(const int *bucket, int *offset, int range) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    offset[0] = 0;
    for (int i=1; i<range; i++)
      offset[i] = offset[i-1] + bucket[i-1];
  }
}

__global__ void fill_sorted_key(int *key, const int *bucket, const int *offset) {
  int value = blockIdx.x;
  for (int j=threadIdx.x; j<bucket[value]; j+=blockDim.x)
    key[offset[value] + j] = value;
}


int main() {
  int n = 50;
  int range = 5;
  // std::vector<int> key(n);
  int *key, *bucket, *offset;

  CHECK_CUDA(cudaMallocManaged(&key, n*sizeof(int)));
  CHECK_CUDA(cudaMallocManaged(&bucket, range*sizeof(int)));
  CHECK_CUDA(cudaMallocManaged(&offset, range*sizeof(int)));

  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int block_size = 256;
  clear_bucket<<<(range + block_size - 1)/block_size, block_size>>>(bucket, range);
  CHECK_CUDA(cudaGetLastError());
  count_bucket<<<(n + block_size - 1)/block_size, block_size>>>(key, bucket, n);
  CHECK_CUDA(cudaGetLastError());
  make_offsets<<<1, 1>>>(bucket, offset, range);
  CHECK_CUDA(cudaGetLastError());
  fill_sorted_key<<<range, block_size>>>(key, bucket, offset);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  // std::vector<int> bucket(range); 
  // for (int i=0; i<range; i++) {
  //   bucket[i] = 0;
  // }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  CHECK_CUDA(cudaFree(key));
  CHECK_CUDA(cudaFree(bucket));
  CHECK_CUDA(cudaFree(offset));
}
