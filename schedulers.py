import numpy as np


class CosineAnnealingRestartLRScheduler:

	def __init__(self, initial_lr=0.1, T_0=50, T_mult=2, eta_min=0):
		self.initial_lr = initial_lr
		self.T_0 = T_0
		self.T_mult = T_mult
		self.eta_min = eta_min
		self.current_step = 0
		self.T_cur = T_0

	def step(self):
		lr = self.eta_min + (self.initial_lr - self.eta_min) * (
		    1 + np.cos(np.pi * self.current_step / self.T_cur)) / 2
		self.current_step += 1
		if self.current_step >= self.T_cur:
			self.current_step = 0
			self.T_cur *= self.T_mult
		return lr


class ExponentialLRScheduler:

	def __init__(self, initial_lr, gamma):
		self.initial_lr = initial_lr
		self.gamma = gamma
		self.current_step = 0

	def step(self):
		lr = self.initial_lr * (self.gamma**self.current_step)
		self.current_step += 1
		return lr


class StepLRScheduler:

	def __init__(self, initial_lr, step_size, gamma):
		self.initial_lr = initial_lr
		self.step_size = step_size
		self.gamma = gamma
		self.current_step = 0

	def step(self):
		lr = self.initial_lr * (self.gamma
		                        **(self.current_step // self.step_size))
		self.current_step += 1
		return lr
