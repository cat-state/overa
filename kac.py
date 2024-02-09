import math

import torch
import matplotlib.pyplot as plt

from torch.utils.cpp_extension import load, load_inline

kac_kernel = load(name="randperm", sources=["kac_kernel.cu", "kac.cpp"], verbose=True, 
                   extra_ldflags=["-lcurand"], keep_intermediates=True,
                   extra_include_paths=["/home/a/overa"]
                   )

print((kac_kernel.randperm(4096, 5)))

x = torch.eye(4096, 4096).cuda()
n_steps = math.ceil(math.log2(x.shape[1])) * x.shape[1]
# n_steps = 512
w, ri = kac_kernel.parallel_kac_random_walk(x.clone(), 2024, n_steps)
w2, ui = kac_kernel.parallel_kac_random_walk_bwd(w.clone(), 2024, n_steps)
print(ri)
print(ui)
print((x - w2).abs().max())
# print(x, w, w2)
print(torch.allclose(x, w2, atol=1e-5))

# fig, ax = plt.subplots(3, 1)
# ax[0].imshow(x.cpu().numpy(), vmin=-1, vmax=1)
# ax[1].imshow(w.cpu().numpy(), vmin=-1, vmax=1)
# ax[2].imshow(w2.cpu().numpy(), vmin=-1, vmax=1)
# plt.show()
x = torch.randn(4096, 4096).half().cuda()
y = torch.randn(4096, 4096).half().cuda()
n_steps = math.ceil(math.log2(x.shape[1])) * x.shape[1]
print(n_steps)
from time import time
t = time()
for _ in range(100):
    x, ri = kac_kernel.parallel_kac_random_walk(x, 2024, n_steps)
torch.cuda.synchronize()
print("time", (time() - t) / 100)

with torch.no_grad():
    t = time()
    for _ in range(100):
        x = x @ y
    torch.cuda.synchronize()
    dt = (time() - t) / 100
    print("time", dt)
mm_flops = (2 * x.shape[1] - 1) * x.shape[0] * x.shape[1]
tflops = mm_flops / dt / 1e12
print("mm tflops", tflops)
