import numpy as np
from tqdm import tqdm
from lll import LLL_reduction as RED
from chol_diff import chol_rev
from numpy import linalg as la
import matplotlib.pyplot as plt
import torch
from schedulers import CosineAnnealingRestartLRScheduler, ExponentialLRScheduler, StepLRScheduler
from util import ORTH, URAN_matrix, GRAN, CLP, det, grader
import time
import os

np.random.seed(19260817)

#TODO:(perhaps) change numpy to cupy for GPU acceleration
#TODO: add covariance to error


def calc_NSM(B_t, batch_size, n):
	B = B_t.detach().numpy()
	z = URAN_matrix(batch_size, n)
	y = z - CLP(B, z @ B)
	e = torch.tensor(y) @ B_t
	e2 = torch.norm(e, dim=-1)**2

	NSM = (torch.prod(torch.diagonal(B_t))**(-2 / n)) * e2 / n
	return torch.mean(NSM)


def reduce_L(L):
	# L = ORTH(RED(L))
	L = L / (det(L)**(1 / n))
	return L


def train(T, hook_fn, L, scheduler, n, batch_size):

	for t in tqdm(range(T)):
		mu = scheduler.step()

		leaf_L = torch.tensor(L, requires_grad=True)
		leaf_L.register_hook(hook_fn)

		NSM = calc_NSM(leaf_L, batch_size, n)

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

	L = ORTH(GRAN(n, n))
	L[1, 0] = 0
	L = L / (det(L)**(1 / n))

	scheduler = CosineAnnealingRestartLRScheduler(initial_lr=mu0)
	# scheduler = ExponentialLRScheduler(initial_lr=mu0, gamma=v**(-1 / T))

	mask = torch.zeros((10,10), dtype=torch.float32)
	for i in range(n):
		for j in range(i + 1):
			if i != 1 or j != 0:
				mask[i, j] = 1
	# print(mask)

	L = train(T, lambda grad: grad * mask, L, scheduler, n, batch_size)
	L = L / (det(L)**(1 / n))

	NSM, sigma_squared = grader(L)
	sigma = np.sqrt(sigma_squared)
	data = {
	    'B': L,
	    'NSM': NSM,
		'G':None,
	    'sigma': sigma,
	    'n': n,
	    'batch_size': batch_size,
	    'T': T,
	    'mu0': mu0
	}
	save_path = f"./data/{n}_dim/"
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	# print("B: ", B)

	np.savez(
	    save_path + "B" + time.strftime("%Y%m%d-%H-%M-%S", time.localtime()),
	    **data)
