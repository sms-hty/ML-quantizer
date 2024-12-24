import numpy as np
from tqdm import tqdm
from lll import LLL_reduction as RED

def ORTH(B):
    return np.linalg.cholesky(B @ B.T)
"""
def RED(B):
    return B
"""
def URAN(n):
    return np.random.randn(n)
def GRAN(n, m):
    return np.random.normal(size = (n, m))
def CLP(G, r):
    n = G.shape[0]
    C = np.float64("inf")
    i = n
    d = np.array([n - 1] * n)
    Lambda = np.zeros(n + 1)
    F = np.zeros((n, n))
    F[n - 1] = r.copy()
    p = np.zeros(n)
    u = np.zeros(n)
    Delta = np.zeros(n)
    while True:
        while True:
            if i != 0:
                i = i - 1
                for j in range(d[i], i, -1):
                    F[j - 1, i] = F[j, i] - u[j] * G[j, i]
                p[i] = F[i, i] / G[i, i]
                u[i] = np.round(p[i])
                y = (p[i] - u[i]) * G[i, i]
                Delta[i] = np.sign(y)
                Lambda[i] = Lambda[i + 1] + y ** 2
            else:
                uu = u.copy()
                C = Lambda[0]
            if Lambda[i] >= C:
                break
        m = i
        while True:
            if i == n - 1:
                return uu
            else:
                i = i + 1
                u[i] = u[i] + Delta[i]
                Delta[i] = -Delta[i] - np.sign(Delta[i])
                y = (p[i] - u[i]) * G[i][i]
                Lambda[i] = Lambda[i + 1] + y ** 2
            if Lambda[i] < C:
                break
        for j in range(m, i):
            d[j] = i
        for j in range(m - 1, -1, -1):
            if d[j] < i:
                d[j] = i
            else:
                break
def det(B):
    res = 1
    for i in range(n):
        res = res * B[i, i]
    return res


Tr = 100
T = Tr * 10000
mu0 = 0.001
v = 500
n = 10

B = ORTH(RED(GRAN(n, n)))
B = B / (det(B) ** (1 / n))
for t in tqdm(range(T)):
    mu = mu0 * (v ** (-t / (T - 1)))
    z = URAN(n)
    y = z - CLP(B, z @ B)
    e = y @ B
    e2 = np.linalg.norm(e) ** 2
    for i in range(n):
        for j in range(i):
            B[i, j] -= mu * y[i] * e[j]
        B[i, i] -= mu * (y[i] * e[i] - e2 / (n * B[i, i]))
    if t % Tr == Tr - 1:
        B = ORTH(RED(B))
        B = B / (det(B) ** (1 / n))

test = 10000
G = 0
sigma = 0
for i in tqdm(range(test)):
    z = URAN(n)
    y = z - CLP(B, z @ B)
    e = y @ B
    e2 = np.linalg.norm(e) ** 2
    val = 1 / n * e2
    G += val
    sigma += val ** 2

G = G / test
sigma = (sigma / test - G ** 2) / (test - 1)

print("G:", G, " sigma:", sigma)