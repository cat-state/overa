#include <cstdlib>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda/barrier>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include "shared_shim.cuh"

namespace cg = cooperative_groups;

/*  Variable Philox
    32-bit implementation of https://arxiv.org/abs/2106.06161
    "Bandwidth-Optimal Random Shuffling for GPUs"
    Generates a permutation of size (left_side_bits + right_side_bits)
*/
const uint32_t M0 = 0xB1CE6E93;

__device__ uint64_t WVariablePhilox(const uint64_t val,  
    const uint64_t* keys,
    const uint32_t left_side_bits,
    const uint32_t right_side_bits,
    const uint32_t num_rounds
)
{
    auto right_side_mask = (uint64_t(1) << right_side_bits) - 1;
    auto left_side_mask = (uint64_t(1) << left_side_bits) - 1;

    static const uint64_t M0 = UINT64_C(0xD2B74407B1CE6E93);
    uint32_t state[2] = { uint32_t(val >> right_side_bits),
    uint32_t(val & right_side_mask)
    };
    for (int i = 0; i < num_rounds; i++)
    {
    uint32_t hi = __umulhi(M0, state[0]);
    uint32_t lo = M0 * state[0];
    lo = (lo << (right_side_bits - left_side_bits)) |
    state[1] >> left_side_bits;
    state[0] = ((hi ^ keys[i]) ^ state[1]) & left_side_mask;
    state[1] = lo & right_side_mask;
    }
    // Combine the left and right sides together to get result
    return (uint64_t) state[0] << right_side_bits |
    (uint64_t) state[1];
}

__device__ uint32_t variable_philox(
    const uint32_t val,
    const uint32_t* keys,
    const uint32_t left_side_bits,
    const uint32_t right_side_bits,
    const uint32_t num_rounds
) {
    auto right_side_mask = (1 << right_side_bits) - 1;
    auto left_side_mask = (1 << left_side_bits) - 1;

    uint16_t state[2] = { uint16_t(val >> right_side_bits), uint16_t(val & right_side_mask) };
    for (uint32_t i = 0; i < num_rounds; i++) {
        uint32_t hilo = M0 * state[0];
        uint32_t hi = hilo >> 16;
        uint32_t lo = hilo & 0xffff;
                 lo = (lo << (right_side_bits - left_side_bits)) | state[1] >> left_side_bits;
        state[0] = ((hi ^ keys[i]) ^ state[1]) & left_side_mask;
        state[1] = lo & right_side_mask;
    }
    return uint32_t(state[0]) << right_side_bits | uint32_t(state[1]);
}


template <typename scalar_t>
__global__ void parallel_kac_random_walk_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> x,
    uint32_t seed,
    uint32_t n_steps
) {
    auto block = cg::this_thread_block();
    const int32_t b = block.group_index().x;
    const int32_t tid = block.thread_rank();
    const int dim = x.size(1);

    const int32_t par_steps = (n_steps + block.size() - 1) / block.size();

    int bit_width = 32 - __clz(dim) - 1;
    uint32_t left_side_bits = bit_width / 2;
    uint32_t right_side_bits = bit_width - left_side_bits;

    SharedMemory<scalar_t> shared;
    scalar_t* shared_x = shared.getPointer();

    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
    if (block.thread_rank() == 0) {
        init(&barrier, block.size());
    }
    block.sync();

    cuda::memcpy_async(block, shared_x, x[b].data(), dim * sizeof(scalar_t), barrier);
    barrier.arrive_and_wait();

    for(int32_t par_step = 0; par_step < par_steps; par_step++) {
        uint32_t keys[4] = {0};
        for (uint32_t j = 0; j < 1; j++) {
            uint4 result = curand_Philox4x32_10({0, 0, 0, seed}, {par_step, j});
            keys[j * 4] = result.x;
            keys[j * 4 + 1] = result.y;
            keys[j * 4 + 2] = result.z;
            keys[j * 4 + 3] = result.w;
        }

        int32_t step = tid + par_step * block.size();
        if (step < n_steps) {
            uint4 rng = curand_Philox4x32_10({0, 0, tid, seed}, {par_step, 3});
            float a = _curand_uniform(rng.x) * 2.0 * 3.14159265359;
            uint32_t r1 = (uint32_t)variable_philox(tid * 2, keys, left_side_bits, right_side_bits, 4);
            uint32_t r2 = (uint32_t)variable_philox(tid * 2 + 1, keys, left_side_bits, right_side_bits, 4);
            scalar_t x1 = shared_x[r1];
            scalar_t x2 = shared_x[r2];
            scalar_t y1 = cos(a) * x1 - sin(a) * x2;
            scalar_t y2 = sin(a) * x1 + cos(a) * x2;
            shared_x[r1] = y1;
            shared_x[r2] = y2;
        }
        block.sync();
    }

    cuda::memcpy_async(block, x[b].data(), shared_x, dim * sizeof(scalar_t), barrier);
    barrier.arrive_and_wait();
}


template <typename scalar_t>
__global__ void parallel_kac_random_walk_bwd_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> x,
    uint32_t seed,
    uint32_t n_steps
) {
    auto block = cg::this_thread_block();
    const int32_t b = block.group_index().x;
    const int32_t tid = block.thread_rank();

    const int dim = x.size(1);

    const int32_t par_steps = (n_steps + block.size() - 1) / block.size();

    int bit_width = 32 - __clz(dim) - 1;
    uint32_t left_side_bits = bit_width / 2;
    uint32_t right_side_bits = bit_width - left_side_bits;

    SharedMemory<scalar_t> shared;
    scalar_t* shared_x = shared.getPointer();

    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
    if (block.thread_rank() == 0) {
        init(&barrier, block.size());
    }
    block.sync();

    cuda::memcpy_async(block, shared_x, x[b].data(), dim * sizeof(scalar_t), barrier);
    barrier.arrive_and_wait();

    for(int32_t par_step = par_steps - 1; par_step >= 0; par_step--) {
        uint32_t keys[4] = {0};
        for (uint32_t j = 0; j < 1; j++) {
            uint4 result = curand_Philox4x32_10({0, 0, 0, seed}, {par_step, j});
            keys[j * 4] = result.x;
            keys[j * 4 + 1] = result.y;
            keys[j * 4 + 2] = result.z;
            keys[j * 4 + 3] = result.w;
        }

        int32_t step = tid + par_step * block.size();
        if (step < n_steps) {
            uint4 rng = curand_Philox4x32_10({0, 0, tid, seed}, {par_step, 3});
            float a = -_curand_uniform(rng.x) * 2.0 * 3.14159265359;
            uint32_t r1 = variable_philox(tid * 2, keys, left_side_bits, right_side_bits, 4);
            uint32_t r2 = variable_philox(tid * 2 + 1, keys, left_side_bits, right_side_bits, 4);
            scalar_t x1 = shared_x[r1];
            scalar_t x2 = shared_x[r2];
            scalar_t y1 = cos(a) * x1 - sin(a) * x2;
            scalar_t y2 = sin(a) * x1 + cos(a) * x2;
            shared_x[r1] = y1;
            shared_x[r2] = y2;
        }
        block.sync();
    }

    cuda::memcpy_async(block, x[b].data(), shared_x, dim * sizeof(scalar_t), barrier);
    barrier.arrive_and_wait();
}


bool is_even_power_of_two(int n) {
    return (n & (n - 1)) == 0 && n % 4 == 0;
}

torch::Tensor parallel_kac_random_walk(
    torch::Tensor x,
    uint32_t seed,
    uint32_t n_steps
) {
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(is_even_power_of_two(x.size(1)), "x.size(1) must be an even power of 2");

    const dim3 blocks(x.size(0));
    const int threads = min(1024, int(x.size(1)) / 2);
    int shared_size;
    switch (x.scalar_type()) {
        case torch::ScalarType::Double:
            shared_size = x.size(1) * sizeof(double);
            parallel_kac_random_walk_kernel<double><<<blocks, threads, shared_size>>>(
                x.packed_accessor32<double, 2, torch::RestrictPtrTraits>(),
                seed,
                n_steps
            );
            break; 
        case torch::ScalarType::Float:
            shared_size = x.size(1) * sizeof(float);
            parallel_kac_random_walk_kernel<float><<<blocks, threads, shared_size>>>(
                x.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                seed,
                n_steps
            );
            break;
        case torch::ScalarType::Half:
            shared_size = x.size(1) * sizeof(at::Half);
            parallel_kac_random_walk_kernel<at::Half><<<blocks, threads, shared_size>>>(
                x.packed_accessor32<at::Half, 2, torch::RestrictPtrTraits>(),
                seed,
                n_steps
            );
            break;
        case torch::ScalarType::BFloat16:
            shared_size = x.size(1) * sizeof(at::BFloat16);
            parallel_kac_random_walk_kernel<at::BFloat16><<<blocks, threads, shared_size>>>(
                x.packed_accessor32<at::BFloat16, 2, torch::RestrictPtrTraits>(),
                seed,
                n_steps
            );
            break;    
        default:
            TORCH_CHECK(false, "Unsupported dtype");
            break;
    }
    return x;
}


torch::Tensor parallel_kac_random_walk_bwd(
    torch::Tensor x,
    uint32_t seed,
    uint32_t n_steps
) {
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(is_even_power_of_two(x.size(1)), "x.size(1) must be an even power of 2");

    const dim3 blocks(x.size(0));
    const int threads = min(1024, int(x.size(1)) / 2);
    int shared_size;
    switch (x.scalar_type()) {
        case torch::ScalarType::Double:
            shared_size = x.size(1) * sizeof(double);
            parallel_kac_random_walk_bwd_kernel<double><<<blocks, threads, shared_size>>>(
                x.packed_accessor32<double, 2, torch::RestrictPtrTraits>(),
                seed,
                n_steps
            );
            break;
        case torch::ScalarType::Float:
            shared_size = x.size(1) * sizeof(float);
            parallel_kac_random_walk_bwd_kernel<float><<<blocks, threads, shared_size>>>(
                x.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                seed,
                n_steps
            );
            break;
        case torch::ScalarType::Half:
            shared_size = x.size(1) * sizeof(at::Half);
            parallel_kac_random_walk_bwd_kernel<at::Half><<<blocks, threads, shared_size>>>(
                x.packed_accessor32<at::Half, 2, torch::RestrictPtrTraits>(),
                seed,
                n_steps
            );
            break;
        case torch::ScalarType::BFloat16:
            shared_size = x.size(1) * sizeof(at::BFloat16);
            parallel_kac_random_walk_bwd_kernel<at::BFloat16><<<blocks, threads, shared_size>>>(
                x.packed_accessor32<at::BFloat16, 2, torch::RestrictPtrTraits>(),
                seed,
                n_steps
            );
            break;    
        default:
            TORCH_CHECK(false, "Unsupported dtype");
            break;
    }
    return x;
}




__global__ void randperm_kernel(
    int32_t* out_perm,
    uint32_t seed,
    uint32_t N
) {

    auto grid = cg::this_grid();
    int bit_width = 32 - __clz(N) - 1;
    uint32_t left_side_bits = bit_width / 2;
    uint32_t right_side_bits = bit_width - left_side_bits;

    uint4 counter {0, 0, 0, seed};
    uint32_t keys[12] = {0};
    for (uint32_t i = 0; i < 3; i++) {
        uint4 result = curand_Philox4x32_10(counter, {0, i});
        keys[i * 4] = result.x;
        keys[i * 4 + 1] = result.y;
        keys[i * 4 + 2] = result.z;
        keys[i * 4 + 3] = result.w;
    }

    int32_t tid = grid.thread_rank();
    int32_t bid = grid.block_rank();
    out_perm[tid] = cg::this_thread_block().group_index().x;
    // out_perm[tid] = int32_t(variable_philox(tid, keys, left_side_bits, right_side_bits, 12));
}

torch::Tensor randperm(uint32_t N, uint32_t seed) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);
    torch::Tensor out = torch::ones({N}, opts);

    const dim3 blocks(2048);
    randperm_kernel<<<blocks, 2>>>(out.mutable_data_ptr<int32_t>(), seed, N);
    return out;
}   
