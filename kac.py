import math

import torch
import matplotlib.pyplot as plt

from torch.utils.cpp_extension import load, load_inline

kac_kernel = load(name="randperm", sources=["kac_kernel.cu", "kac.cpp"], verbose=True, 
                   extra_ldflags=["-lcurand"], keep_intermediates=True,
                   extra_include_paths=["/home/a/overa"]
                   )


class KacRandomWalk_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, seed: int, n_steps: int) -> torch.Tensor:
        x = kac_kernel.parallel_kac_random_walk(x, seed, n_steps)
        ctx.seed = seed
        ctx.n_steps = n_steps
        ctx.mark_dirty(x)
        return x

    @staticmethod
    def backward(ctx, grad_x: torch.Tensor) -> torch.Tensor:
        grad_x = kac_kernel.parallel_kac_random_walk_bwd(grad_x.clone(), ctx.seed, ctx.n_steps)
        return grad_x, None, None


def kac_random_walk_(x, seed, n_steps=None) -> torch.Tensor:
    if n_steps is None:
        n_steps = math.ceil(math.log2(x.shape[1])) * x.shape[1]
    return KacRandomWalk_.apply(x, seed, n_steps)

if __name__ == "__main__":

    assert(torch.autograd.gradcheck(lambda x,y,z: KacRandomWalk_.apply(x.clone(), y, z), (torch.randn(4, 4096, requires_grad=True).cuda().double(), 5, 12 * 4096), fast_mode=True))


    x = torch.randn(4096, 4096).cuda()
    x.requires_grad_(True)
    KacRandomWalk_.apply(x.clone(), 2024, 12 * 4096).sum().backward()
    print(x.grad)



    print((kac_kernel.randperm(4096, 5)))

    x = torch.eye(4096, 4096).cuda()
    n_steps = math.ceil(math.log2(x.shape[1])) * x.shape[1]
    # n_steps = 512
    w = kac_kernel.parallel_kac_random_walk(x.clone(), 2024, n_steps)
    w2 = kac_kernel.parallel_kac_random_walk_bwd(w.clone(), 2024, n_steps)
    print((x - w2).abs().max())
    # print(x, w, w2)
    assert(torch.allclose(x, w2, atol=1e-5))

    # fig, ax = plt.subplots(3, 1)
    # ax[0].imshow(x.cpu().numpy(), vmin=-1, vmax=1)
    # ax[1].imshow(w.cpu().numpy(), vmin=-1, vmax=1)
    # ax[2].imshow(w2.cpu().numpy(), vmin=-1, vmax=1)
    # plt.show()
    x = torch.randn(4096, 4096).half().cuda()
    z = torch.randn(4096, 4096).half().cuda()
    y = torch.randn(16, 4096).half().cuda()
    n_steps = math.ceil(math.log2(x.shape[1])) * x.shape[1]
    print(n_steps)
    from time import time
    t = time()
    for _ in range(100):
        x = kac_kernel.parallel_kac_random_walk(x, 2024, n_steps)
    torch.cuda.synchronize()
    print("kc time", (time() - t) / 100)

    with torch.no_grad():
        t = time()
        for _ in range(100):
            x = x @ z
        torch.cuda.synchronize()
        dt = (time() - t) / 100
        print("mm time", dt)
    mm_flops = (2 * x.shape[1] - 1) * x.shape[0] * x.shape[1]
    tflops = mm_flops / dt / 1e12
    print("mm tflops", tflops)
