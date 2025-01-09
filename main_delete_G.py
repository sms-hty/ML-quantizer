import numpy as np
from tqdm import tqdm
from lll import LLL_reduction as RED
from chol_diff import chol_rev
from numpy import linalg as la
from numba import prange, jit
import matplotlib.pyplot as plt
from schedulers import CosineAnnealingRestartLRScheduler, ExponentialLRScheduler, StepLRScheduler
from util import ORTH, URAN_matrix, GRAN, CLP, det, grader, theta_image
import time
import os

np.random.seed(1919810)

#TODO:(perhaps) change numpy to cupy for GPU acceleration
#TODO: design G
#TODO: add covariance to error

def calc_NSM(B, batch_size, n):
	z = URAN_matrix(batch_size, n)
	y = z - CLP(B, z @ B)
	e = y @ B
	e2 = la.norm(e, axis=-1)**2

	NSM = (det(B)**(-2 / n)) * e2 / n
	return y, e, e2, np.mean(NSM)


def calc_B_diff(y, e, e2, B, n):
	B_diff = np.tril(np.einsum('ij,ik->ijk', y, e))
	B_diff.transpose(1, 2, 0)[np.diag_indices(n)] -= np.outer(
	    1 / np.diag(B), e2 / n)
	B_diff = np.mean(B_diff, axis=0)
	B_diff = B_diff * 2 * (det(B)**(-2 / n)) / n
	return B_diff


def reduce_B(B):
	# B = ORTH(RED(B))
	B = B / (det(B)**(1 / n))
	return B


def train(T, mask, B, scheduler, n, batch_size):

	for t in tqdm(range(T)):
		mu = scheduler.step()

		y, e, e2, NSM = calc_NSM(B, batch_size, n)

		B_diff = calc_B_diff(y, e, e2, B, n)

		B -= mu * (B_diff * mask)

		if t % Tr == Tr - 1:
			B = reduce_B(B)

	return B

def initB(n):
	B = np.zeros((n,n))
	B = ORTH(RED(GRAN(n,n)))
	B[12, 0:12] = 0
	B = B / (det(B)**(1 / n))
	return B

def initmask(n):
	mask = np.zeros((n,n))
	for i in range(n):
		for j in range(i + 1):
			if i < 12 or j >= 12:
				mask[i, j] = 1
	return mask

if __name__ == "__main__":
	Tr = 100
	T = Tr * 400
	mu0 = 0.5
	v = 1000
	n = 13
	batch_size = 128

	
	B = initB(n)
	mask = initmask(n)

	scheduler = CosineAnnealingRestartLRScheduler(initial_lr=mu0)
	# scheduler = ExponentialLRScheduler(initial_lr=mu0, gamma=v**(-1 / T))

	B = train(T, mask, B, scheduler, n, batch_size)

	B = B / (det(B)**(1 / n))

	NSM, sigma_squared = grader(B)
	sigma = np.sqrt(sigma_squared)

	np.set_printoptions(suppress=True)
	np.set_printoptions(precision=4)
	print("B: ", B)

	data = {
	    'B': B,
	    'NSM': NSM,
	    'sigma': sigma,
	    'n': n,
	    'batch_size': batch_size,
	    'T': T,
	    'mu0': mu0
	}
	save_path = f"./data/{n}_dim/"
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	np.savez(
	    save_path + "B" + time.strftime("%Y%m%d-%H-%M-%S", time.localtime()),
	    **data)
