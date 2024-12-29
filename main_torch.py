import numpy as np
from tqdm import tqdm
from lll import LLL_reduction as RED
from chol_diff import chol_rev
from numpy import linalg as la
from numba import prange, jit
import matplotlib.pyplot as plt
import torch
from schedulers import CosineAnnealingRestartLRScheduler, ExponentialLRScheduler, StepLRScheduler

np.random.seed(19260817)

#TODO:(perhaps) change numpy to cupy for GPU acceleration
#TODO: generate theta-image
#TODO: design G
#TODO: add covariance to error


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


def calc_B(G, L):
	return la.cholesky(
	    np.mean(np.matmul(np.matmul(G, L), np.swapaxes(np.matmul(G, L), -1,
	                                                   -2)),
	            axis=0))


def calc_NSM(B_t, batch_size, n):
	B = B_t.detach().numpy()
	z = URAN_matrix(batch_size, n)
	y = z - CLP(B, z @ B)
	e = torch.tensor(y) @ B_t
	e2 = torch.norm(e, dim=-1) ** 2

	NSM = (torch.prod(torch.diagonal(B_t))**(-2 / n)) * e2 / n
	return torch.mean(NSM)


def reduce_L(L):
	L = ORTH(RED(L))
	L = L / (det(L)**(1 / n))
	return L


def train(T, G, L, scheduler, n, batch_size):
	
	G = torch.tensor(G)

	for t in tqdm(range(T)):
		mu = scheduler.step()

		leaf_L = torch.tensor(L, requires_grad = True)
		B_t = torch.linalg.cholesky(
	    torch.mean((G @ leaf_L) @ (G @ leaf_L).transpose(-1, -2),
	            dim=0))

		NSM = calc_NSM(B_t, batch_size, n)

		NSM.backward()

		L -= mu * leaf_L.grad.numpy()

		if t % Tr == Tr - 1:
			L = reduce_L(L)

	return L


if __name__ == "__main__":

	Tr = 100
	T = Tr * 1000
	mu0 = 0.5
	v = 1000
	n = 10
	batch_size = 128

	I = np.eye(n)
	I_swapped = I.copy()
	I_swapped[[0, 1]] = I_swapped[[1, 0]]
	G = [I]
	# G = [
	#     np.diag([1, 1]),
	#     np.diag([-1, 1]),
	#     np.diag([1, -1]),
	#     np.diag([-1, -1])
	# ]
	G = np.array(G)
	L = ORTH(RED(GRAN(n, n)))
	L = L / (det(L)**(1 / n))

	scheduler = CosineAnnealingRestartLRScheduler(initial_lr=mu0)
	# scheduler = ExponentialLRScheduler(initial_lr=mu0, gamma=v**(-1 / T))

	L = train(T, G, L, scheduler, n, batch_size)

	A = np.mean(np.matmul(np.matmul(G, L), np.swapaxes(np.matmul(G, L), -1,
	                                                   -2)),
	            axis=0)

	B = la.cholesky(A)
	B = B / (det(B)**(1 / n))

	test = 100000
	G = 0
	sigma = 0
	for i in tqdm(range(test)):
		z = URAN_matrix(1, n)
		y = z - CLP(B, z @ B)
		e = y @ B
		e2 = la.norm(e)**2
		val = 1 / n * e2
		G += val
		sigma += val * val

	G = G / test
	sigma = (sigma / test - G**2) / (test - 1)

	print("G:", G, " sigma:", sigma)

	# print("B: ", B)
