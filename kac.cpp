#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>

torch::Tensor randperm(uint32_t N, uint32_t seed);

std::tuple<torch::Tensor, torch::Tensor> parallel_kac_random_walk(
    torch::Tensor x,
    uint32_t seed,
    uint32_t n_steps
);

std::tuple<torch::Tensor, torch::Tensor> parallel_kac_random_walk_bwd(
    torch::Tensor x,
    uint32_t seed,
    uint32_t n_steps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("randperm", &randperm, "randperm");
    m.def("parallel_kac_random_walk", &parallel_kac_random_walk, "parallel_kac_random_walk");
    m.def("parallel_kac_random_walk_bwd", &parallel_kac_random_walk_bwd, "parallel_kac_random_walk_bwd");
}
