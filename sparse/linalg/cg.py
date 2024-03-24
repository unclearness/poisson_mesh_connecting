import numpy as np


def spsolve(A, b):
    return cg(A, b)


def cg(A, b, x0=None, maxiter=None, M=None, callback=None, atol=0.0, rtol=1e-05):
    x = x0
    if x is None:
        x = np.zeros_like(b)
    t = A @ x
    p = b - t
    r = p.copy()
    r_square = (r * r).sum()
    terminate_th = max(rtol * np.linalg.norm(b), atol)
    terminate_th2 = terminate_th ** 2
    if maxiter is None:
        maxiter = 1000
    for i in range(maxiter):
        t = A @ p
        alpha = r_square / (t * p).sum()
        x += alpha * p
        r -= alpha * t
        r_square_new = (r * r).sum()
        if r_square_new < terminate_th2:
            break
        beta = r_square_new / r_square
        p = r + beta * p
        r_square = r_square_new
    return x
