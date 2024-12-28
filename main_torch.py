import numpy as np
from tqdm import tqdm
from lll import LLL_reduction as RED
from chol_diff import chol_rev
import torch

torch.set_default_dtype(torch.float64)


def ORTH(B):
	return torch.linalg.cholesky(B @ B.T)


def URAN(n):
	# return torch.randn(n)
	return torch.rand(n)


def GRAN(n, m):
	return torch.randn((n, m))


# def URAN_TEST(n):
#     return torch.tensor([0.342,0.46556,0.8])


def CLP(GG, rr):
	G = GG.numpy()
	r = rr.numpy()

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
				Lambda[i] = Lambda[i + 1] + y**2
			else:
				uu = u.copy()
				C = Lambda[0]
			if Lambda[i] >= C:
				break
		m = i
		while True:
			if i == n - 1:
				ret = torch.from_numpy(uu)
				return ret
			else:
				i = i + 1
				u[i] = u[i] + Delta[i]
				Delta[i] = -Delta[i] - np.sign(Delta[i])
				y = (p[i] - u[i]) * G[i][i]
				Lambda[i] = Lambda[i + 1] + y**2
			if Lambda[i] < C:
				break
		for j in range(m, i):
			d[j] = i
		for j in range(m - 1, -1, -1):
			if d[j] < i:
				d[j] = i
			else:
				break


def mydet(B):
	# res = 1
	# for i in range(n):
	# 	res = res * B[i, i]
	res = torch.prod(torch.diag(B))
	return res


Tr = 100
T = Tr * 1000
mu0 = 0.1
v = 50
n = 2

I = np.eye(n)
I_swapped = I.copy()
I_swapped[[0, 1]] = I_swapped[[1, 0]]
# G = [I]
G = [np.diag([1, 1]), np.diag([-1, 1]), np.diag([1, -1]), np.diag([-1, -1])]
torch_G = torch.tensor(np.array(G)).to(torch.float64)

L = ORTH(RED(GRAN(n, n)))
# L = ORTH(RED(torch.tensor([[1,3,5],[2,4,3],[6,6,6]]).float()))
L = L / (torch.linalg.det(L)**(1 / n))
L.requires_grad_()
optimizer = torch.optim.SGD([L], lr=mu0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, Tr, 500**(-Tr / T))

for t in tqdm(range(T)):
	# mu = mu0 * (v**(-t / (T - 1)))
	# mu = 0.1

	# L.requires_grad_()
	optimizer.zero_grad()

	A = torch.mean(torch.matmul(torch_G, L).matmul(
	    torch.permute(torch.matmul(torch_G, L), (0, 2, 1))),
	               dim=0)

	B = torch.linalg.cholesky(A)

	with torch.no_grad():
		z = URAN(n)
		y = z - CLP(B, z @ B)
	e = torch.matmul(y, B)
	V = mydet(B)

	e2 = torch.linalg.vector_norm(e, dim=-1)**2
	NSM = (V**(-2 / n)) * e2 / n

	NSM.backward()
	optimizer.step()
	scheduler.step()
	if t % Tr == Tr - 1:
		with torch.no_grad():
			L.data = ORTH(RED(L.data))
			L.data = L.data / (torch.linalg.det(L.data)**(1 / n))
		# L.requires_grad_()

with torch.no_grad():
	A = torch.mean(torch.matmul(torch_G, L).matmul(
	    torch.permute(torch.matmul(torch_G, L), (0, 2, 1))),
	               dim=0)
	B = torch.linalg.cholesky(A)
	# B = ORTH(RED(B))
	B = B / (mydet(B)**(1 / n))

	test = 1000
	G = 0
	sigma = 0
	for i in tqdm(range(test)):
		z = URAN(n)
		y = z - CLP(B, z @ B)
		e = torch.matmul(y, B)
		e2 = torch.linalg.vector_norm(e, dim=-1)**2
		val = 1 / n * e2
		G += val
		sigma += val * val

	G = G / test
	sigma_squared = (sigma / test - G**2) / (test - 1)

	print("G:", G, " sigma:", sigma_squared)
	print("B: ", B)
