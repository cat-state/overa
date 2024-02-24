import jax
import jax.numpy as np

def tst(x, a):
    x1, x2 = x[:1], x[1:]
    y1 = np.cos(a) * x1 - np.sin(a) * x2
    y2 = np.sin(a) * x1 + np.cos(a) * x2
    return np.concatenate([y1, y2]).sum()

jaxpr = jax.make_jaxpr(tst)(np.array([1., 2.]).astype(np.float32), 3.14)
grad_jaxpr = jax.make_jaxpr(jax.grad(tst, argnums=(0, 1)))(np.array([1., 2.]).astype(np.float32), 3.14)
print(jaxpr)
print(grad_jaxpr)

def tstg(x, a, grad_output):
    x1, x2 = x[:1], x[1:]
    y = grad_output[0]
    z = grad_output[1]
    y1 = np.cos(a) * x1 - np.sin(a) * x2
    y2 = np.sin(a) * x1 + np.cos(a) * x2
    g_y1 = np.cos(-a) * y - np.sin(-a) * z
    g_y2 = np.sin(-a) * y + np.cos(-a) * z
    y = grad_output[0]
    z = grad_output[1]
    grad_a = z * y1 + -y * y2
    return np.concatenate([np.atleast_1d(g_y1), np.atleast_1d(g_y2)]), grad_a
print(jax.grad(tst, argnums=(0, 1))(np.array([1.234, -2.]).astype(np.float32), 3.14))
print(tstg(np.array([1.234, -2.]).astype(np.float32), 3.14, np.array([1., 1.])))


def tst2(x, an):
    x1, x2 = x[:1], x[1:]

    for i in range(len(an)):
        a = an[i]
        y1 = np.cos(a) * x1 - np.sin(a) * x2
        y2 = np.sin(a) * x1 + np.cos(a) * x2
        x1, x2 = y1, y2

    return np.concatenate([y1, y2]) 

def tst2g(y, an, grad_output):
    y1, y2 = y[0], y[1]
    g_y, g_z = grad_output[0], grad_output[1]
    grad_an = np.zeros(len(an))
    for i in range(len(an) - 1, -1, -1):
        a = an[i]
        grad_a = g_z * y1 + -g_y * y2
        grad_an = grad_an.at[i].set(grad_a)

        g_y1 = np.cos(-a) * g_y - np.sin(-a) * g_z
        g_y2 = np.sin(-a) * g_y + np.cos(-a) * g_z
        x1 = np.cos(-a) * y1 - np.sin(-a) * y2
        x2 = np.sin(-a) * y1 + np.cos(-a) * y2
        g_y, g_z = g_y1, g_y2
        y1, y2 = x1, x2

        print(y1, y2, g_y, g_z, i)

    g_y = np.atleast_1d(g_y)
    g_z = np.atleast_1d(g_z)
    return np.concatenate([g_y, g_z]), grad_an

print(jax.grad(lambda x, a: tst2(x, a).sum(), argnums=(0, 1))(np.array([1.234, -2.]).astype(np.float32), np.array([3.14, 0.3, 2.5])))
output = tst2(np.array([1.234, -2.]).astype(np.float32), np.array([3.14, 0.3, 2.5]))
print(tst2g(output, np.array([3.14, 0.3, 2.5]), np.array([1., 1.])))

