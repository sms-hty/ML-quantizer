from numba import prange, jit
import numpy as np
import math
from numpy import linalg as la
from tqdm import tqdm
import matplotlib.pyplot as plt

def ORTH(B):
	return la.cholesky(B @ B.T)

def URAN(n):
	return np.random.rand(n)

def URAN_matrix(n, m):
	return np.random.rand(n, m)

def GRAN(n, m):
	return np.random.normal(size=(n, m))

@jit(nopython=True, fastmath=True)
def CLP_single(G, r):
	n = G.shape[0]
	C = np.inf
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
				Lambda[i] = Lambda[i + 1] + y**2
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
				Lambda[i] = Lambda[i + 1] + y**2
			if Lambda[i] < C:
				break

		d[m:i] = i
		for j in range(m - 1, -1, -1):
			if d[j] < i:
				d[j] = i
			else:
				break


@jit(nopython=True, parallel=True, fastmath=True)
def CLP(G, r_batch):
	res = np.zeros_like(r_batch)
	for i in prange(r_batch.shape[0]):
		res[i] = CLP_single(G, r_batch[i])
	return res

def det(B):
	return np.prod(np.diagonal(B, axis1=-2, axis2=-1), axis=-1)


def grader(B, test = 100000, batchsize = 128):
    n = B.shape[0]
    G = 0
    sigma = 0
    for i in tqdm(range(test)):
        z = URAN_matrix(batchsize, n)
        y = z - CLP(B, z @ B)
        e = y @ B
        e2 = la.norm(e, axis = -1) ** 2
        val = 1 / n * e2
        G += val.sum()
        sigma += (val ** 2).sum()
    T = test * batchsize
    G = G / T
    sigma = math.sqrt((sigma / T - G ** 2) / (T - 1))
    print(G, sigma)
    return (G, sigma)

def theta_image(B, UP = 5):
    n = B.shape[0]
    UP = 5
    data = []
    def dfs(dep, sum, nowvec):
        if sum > UP:
            return
        if dep == -1:
            data.append(sum)
            return
        # sum + (now[dep] + i * B[dep][dep]) ** 2 <= UP
        # |now[dep] + i * B[dep][dep]| <= sqrt(UP - sum)
        # -now[dep] - sqrt(UP - sum) <= i * B[dep][dep] <= -now[dep] + sqrt(UP - sum)
        
        l = math.ceil((-nowvec[dep] - math.sqrt(UP - sum)) / B[dep, dep])
        r = math.floor((-nowvec[dep] + math.sqrt(UP - sum)) / B[dep, dep])
        
        for i in range(l, r + 1):
            newvec = nowvec + i * B[dep]
            dfs(dep - 1, sum + newvec[dep] ** 2, newvec)

    dfs(n - 1, 0, np.zeros(n))

    sorted_data = list(np.sort(data))
    cdf = [i for i in range(1, len(sorted_data) + 1)]
    cdf.append(len(sorted_data))
    sorted_data.append(UP)
    plt.figure(figsize=(8, 6))
    plt.step(sorted_data, cdf, where='post', linestyle='-', markersize=0)
    plt.xlabel('r^2')
    plt.xlim(0, UP)
    plt.ylabel('N(B,r)')
    plt.yscale('log')
    plt.grid(True)
    plt.show()