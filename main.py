import numpy as np
from tqdm import tqdm
from lll import LLL_reduction as RED

from chol_diff import chol_rev


def ORTH(B):
    return np.linalg.cholesky(B @ B.T)

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


Tr = 1
T = Tr * 1000000
mu0 = 0.001
v = 500
n = 2

I = np.eye(n)
I_swapped = I.copy() 
I_swapped[[0, 1]] = I_swapped[[1, 0]]  
G = [I, I_swapped]

L = ORTH(RED(GRAN(n, n)))
# L = np.array([[1,2],[3,1]])
L = L / (det(L) ** (1 / n))

for t in tqdm(range(T)):
    mu = mu0 * (v ** (-t / (T - 1)))

    # print("L: ", L)

    A = np.zeros((n, n))
    for g in G: 
        A += g@L@((g@L).T)
    A /= len(G)

    # print("A: ", A)

    B = np.linalg.cholesky(A)
    B_diff = np.zeros((n, n))

    # print("B: ", B)

    z = URAN(n)
    y = z - CLP(B, z @ B)
    e = y @ B
    e2 = np.linalg.norm(e) ** 2
    for i in range(n):
        for j in range(i):
            B_diff[i, j] = y[i] * e[j]
        B_diff[i, i] = (y[i] * e[i] - e2 / (n * B[i, i]))

    # print("B_diff: ", B_diff)

    A_diff = chol_rev(B, B_diff) 

    # print("A_diff: ", A_diff)
    """
    for i in range(n): 
        for j in range(i): 
            A_diff[j, i] = A_diff[i, j]
    """

    A_diff = np.tril(A_diff) + np.tril(A_diff, -1).T
    # print("A_diff: ", A_diff)

    L_diff = np.zeros((n, n))
    for g in G:
        gL_diff = A_diff @ g @ L * 2
        L_diff += g.T @ gL_diff
    L_diff /= len(G)

    L -= mu * L_diff

    if t % Tr == Tr - 1:
        L = ORTH(RED(L))
        L = L / (det(L) ** (1 / n))

print("L_diff: ", L_diff)
A = np.zeros((n, n))
for g in G: 
    A += g@L@((g@L).T)
A /= len(G)

B = np.linalg.cholesky(A)
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
    sigma += val * val

G = G / test
sigma = (sigma / test - G ** 2) / (test - 1)

print("G:", G, " sigma:", sigma)
print("B: ", B)