import math
import time

def denominator(i, j):
    ij = i + j
    return ij * (ij + 1) // 2 + i + 1

def multiply_av(v, out):
    n = len(v)
    for i in range(n):
        out[i] = sum(v[j] / denominator(i, j) for j in range(n))

def multiply_atv(v, out):
    n = len(v)
    for i in range(n):
        out[i] = sum(v[j] / denominator(j, i) for j in range(n))

def spectral_norm(n):
    u = [1.0] * n
    v = [0.0] * n
    tmp = [0.0] * n
    for _ in range(10):
        multiply_av(u, tmp)
        multiply_atv(tmp, v)
        multiply_av(v, tmp)
        multiply_atv(tmp, u)
    vbv = sum(a * b for a, b in zip(u, v))
    vv = sum(b * b for b in v)
    return math.sqrt(vbv / vv)

start = time.time()
result = spectral_norm(500)
elapsed = time.time() - start
print("result: %.13f" % result)
print("elapsed: %s" % elapsed)
