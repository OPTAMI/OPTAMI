import torch
import math
from torch.optim.optimizer import Optimizer
import sys
# import OPTAMI as opt

from Hyperfast_v2.OPTAMI.subsolver import fourth_newton as fn
from Hyperfast_v2.OPTAMI.sup import tuple_to_vec as ttv
from Hyperfast_v2.OPTAMI.sup import derivatives as de
from Hyperfast_v2.OPTAMI.subsolver.subproblem_solving import sub_solve


def quad_func(flat_vk, flat_grad, flat_hess, L):
	vk_norm = flat_vk.norm().square().item()
	total = flat_grad.mul(flat_vk).sum().item() + 0.5 * flat_hess.mv(flat_vk).mul(
		flat_vk).sum().item() + vk_norm ** 2 * L / 4.
	return total


def quad_func_grad(flat_vk, flat_grad, flat_hess, L):
	vk_norm = flat_vk.norm().square().item()
	total = flat_grad.add(flat_hess.mv(flat_vk)).add(flat_vk.mul(vk_norm * L)).abs().max()
	return total


class BDGM(Optimizer):
	"""Implements BDGM. 

	Arguments:
		params (iterable): iterable of parameters to optimize or dicts defining
			parameter groups
		lr (float, optional): learning rate of the gradient descent (default: 1e-1)
		L (float, optional): Lipshitz constant of the Hessian (default: 1e+1)
		eps (float, optional): Desired accuracy for the norm of the model's gradient, used for the stopping criterion  (default: 1e-4)
		max_itet (integer, optional): maximal number of inner iterations of the gradient descent to solve subproblem (default: 10)
		subsolver_bdgm (torch.opt): optimizer for solving
		tol_subsolve (float) : 1/tol_subsolve iterations will be performed
		subsolver_args (dict) : arguments for `subsolver_bdgm`
	"""

	def __init__(self, params, L=1e+1, max_iter_outer=20, subroutine_eps=1e-4, subsolver_bdgm=None, tol_subsolve=None, subsolver_args=None, restarted = False):
		if not 0.0 <= L:
			raise ValueError("Invalid L: {}".format(L))

		defaults = dict(L=L, max_iter_outer=max_iter_outer, subroutine_eps=subroutine_eps,
						   subsolver_bdgm=subsolver_bdgm, tol_subsolve=tol_subsolve, subsolver_args=subsolver_args, restarted = restarted)
		super(BDGM, self).__init__(params, defaults)

		# for group in self.param_groups:
		# for p in group['params']:
		# state = self.state[p]
		# state['step'] = 0.
		# state['ak'] = 0.

	# def share_memory(self):
	# for group in self.param_groups:
	# or p in group['params']:
	# state = self.state[p]
	# state['ak'].share_memory_()

	def step(self, closure=None):
		"""Performs a single optimization step.

		Arguments:
			closure (callable, optional): A closure that reevaluates the model
				and returns the loss.
		"""
		for group in self.param_groups:
			output = closure()
			params = group['params']
			subroutine_eps = group['subroutine_eps']
			L = group['L']
			max_iter_outer = group['max_iter_outer']
			restarted = group['restarted']
			
			subsolver_bdgm = group['subsolver_bdgm']
			tol_subsolve = group['tol_subsolve']
			subsolver_args = group['subsolver_args']

			third_order = True  # do more presize calculation of third model
			test_active = False  # add tests

			length = len(list(params))
			sqrt_const = 2. + math.sqrt(2.)

			# Gradient computation
			grads = torch.autograd.grad(output, list(params), create_graph=True)

			# print(grads)
			grad_norm_xk = torch.sqrt(ttv.tuple_norm_square(grads).detach())
			# delta = 0.1
			# tau = torch.tensor([3.], dtype = torch.double)
			# tau = tau.mul(delta / 8. / sqrt_const).div(grad_norm_xk)

			# Hessian computation
			flat_grads = ttv.tuple_to_vector(grads, length)
			# print(flat_grads)

			flat_grad_xk = flat_grads.clone().detach().to(torch.double)

			if ((subsolver_bdgm == None) & (tol_subsolve == None) & (subsolver_args == None)):
				full_hessian = de.flat_hessian(flat_grads, list(params)).to(torch.double)
				# print(full_hessian)

				# SVD decomposition
				eigenvalues, eigenvectors = torch.symeig(full_hessian, eigenvectors=True)

				if test_active:
					if eigenvectors.mm(torch.diag(eigenvalues)).mm(eigenvectors).sub(full_hessian).max().ge(1e-3).item():
						ValueError('Error: inaccurate computation of svd')

			# Zero step
			vk = []
			# grad_xk = []
			for i in range(length):
				# grad_xk.append(grads[i].detach().clone())
				vk.append(grads[i].clone().detach().mul(0.))

			vk_numel = ttv.tuple_numel(vk)
			# print('vk', vk)
			# print('vk_numel', vk_numel)

			# Computation cycle that solve subproblem by gradient descent
			k = 0
			if torch.cuda.is_available():
				norm_vk = torch.tensor([0.], dtype=torch.double).cuda()
				norm_g_phi = torch.tensor([100000000.], dtype=torch.double).cuda()
				grad_vk_norm = torch.tensor([100000000.], dtype=torch.double).cuda()
			else:
				norm_vk = torch.tensor([0.], dtype=torch.double)
				norm_g_phi = torch.tensor([100000000.], dtype=torch.double)
				grad_vk_norm = torch.tensor([100000000.], dtype=torch.double)
			while k < max_iter_outer:
				# print(vk)

				if third_order:
					tdv, hvp = de.third_derivative_vec(closure, list(params), vk)
					#tdv = de.third_derivative_vec(closure, list(params), vk)
					flat_tdv = ttv.tuple_to_vector(tdv, length).to(torch.double)
					flat_hvp = ttv.tuple_to_vector(hvp, length).to(torch.double)
					flat_g_phi = flat_tdv.div(2.)
				else:

					with torch.no_grad():
						for i in range(length):
							list(params)[i].add_(vk[i].mul(tau))

					# print('+', list(params))
					# Compute gradient in shifted point plus v
					output_plus = closure()
					grad_tau_plus = torch.autograd.grad(output_plus, list(params), retain_graph=True)
					flat_grad_plus = ttv.tuple_to_vector(grad_tau_plus, length).to(torch.double)
					# print('flat_grad_plus', flat_grad_plus)
					# Shift to point minus v

					with torch.no_grad():
						for i in range(length):
							list(params)[i].sub_(vk[i].mul(tau))
					# print('0', list(params))
					with torch.no_grad():
						for i in range(length):
							list(params)[i].sub_(vk[i].mul(tau))
					# print('-', list(params))
					# Compute gradient in shifted point minus v
					output_minus = closure()
					grad_tau_minus = torch.autograd.grad(output_minus, list(params), retain_graph=True)
					flat_grad_minus = ttv.tuple_to_vector(grad_tau_minus, length).to(torch.double)
					# print('flat_grad_minus', flat_grad_minus)
					# Shift back to point xk

					with torch.no_grad():
						for i in range(length):
							list(params)[i].add_(vk[i].mul(tau))

					flat_grad_plus.sub_(flat_grad_xk).div_(tau)  # A = (f'(x_k+tau*v_k) - f'(x_k))/tau
					# flat_grad_plus = flat_grad_plus.add(flat_grad_minus).sub(flat_grad_xk, alpha = 2)#.div_(tau) # A = (f'(x_k+tau*v_k) - f'(x_k))/tau
					# print('0', flat_grad_plus)
					flat_grad_minus.sub_(flat_grad_xk).div_(tau)  # B = (f'(x_k-tau*v_k) - f'(x_k))/tau
					# flat_grad_minus.sub_(flat_grad_xk)#.div_(tau) # B = (f'(x_k-tau*v_k) - f'(x_k))/tau
					# print('1', flat_grad_minus)
					flat_g_phi = flat_grad_plus.add(flat_grad_minus).div(tau.mul(2.))  # G = (A + B)/(2*tau)
					# flat_g_phi = flat_grad_plus.add(flat_grad_minus).div(tau.square().mul(2.)) #G = (A + B)/(2*tau)
					# flat_g_phi = flat_grad_plus.div(tau.square().mul(2.))

				# print('0', list(params))

				# Compute g_phi
				flat_vk = ttv.tuple_to_vector(vk, length).to(torch.double)
				# print('flat_vk', flat_vk)

				# print('2', flat_g_phi)
				#flat_hvp = full_hessian.mv(flat_vk)
				flat_g_phi.add_(flat_hvp)  # Get G += f"(x_k)v_k
				# print('+hessian', flat_g_phi)
				vk_tri_norm = flat_vk.mul(norm_vk.square()).mul(L)
				flat_g_phi.add_(vk_tri_norm)
				# print('+norm', flat_g_phi)
				flat_g_phi.add_(flat_grad_xk)
				# print('flat_g_phi', flat_g_phi)

				norm_g_phi_new = flat_g_phi.norm()
				# print('norm_g_phi_new', norm_g_phi_new)
				if norm_g_phi_new < norm_g_phi:
					norm_g_phi = norm_g_phi_new
				else:
					ValueError('Error: norm greater, then previous')

				flat_const_grad = flat_hvp.add(vk_tri_norm).mul(sqrt_const)
				flat_const_grad = flat_g_phi.sub(flat_const_grad)

				# Stoping criterion

				# Shift to point v
				with torch.no_grad():
					for i in range(length):
						list(params)[i].add_(vk[i])
				# print(list(params))
				# Compute gradient in shifted point minus v
				output_vk = closure()
				grad_vk = torch.autograd.grad(output_vk, list(params), retain_graph=False)
				grad_vk_norm_new = ttv.tuple_norm_square(grad_vk).to(torch.double).sqrt()
				if grad_vk_norm_new < grad_vk_norm:
					grad_vk_norm = grad_vk_norm_new
				else:
					ValueError('Error: gradient of a function greater, then previous')

				with torch.no_grad():
					for i in range(length):
						list(params)[i].sub_(vk[i])

				if test_active:
					print('exit criterea ', norm_g_phi, 'delta ', delta, 'grad_vk / 6. ', grad_vk_norm.div(6.))

					# if norm_g_phi.add(delta) <= grad_vk_norm.div(6.) or grad_vk_norm.div(3.) <= delta:
				if norm_g_phi <= grad_vk_norm.div(6.):
					# print('Exit by stopping criterion. BDGM iterantions = ', k)
					k = max_iter_outer
				# Stoping criterion finish

				# print('model before suvbsolver = ', quad_func(flat_vk, flat_const_grad, full_hessian.mul(sqrt_const), L * sqrt_const))

				# computing for subproblem
				with torch.no_grad():
					init_point = ttv.tuple_to_vector(params).clone().detach()
				if ((subsolver_bdgm==None) & (tol_subsolve==None) & (subsolver_args==None)):
					flat_vk = fn.fourth_subsolver(flat_const_grad.div(sqrt_const), full_hessian,
												  eigenvalues, eigenvectors, L, eps=1e-9)
				else:
					#flat_vk, _ = sub_solve(tol_subsolve, subsolver_bdgm, subsolver_args, {'c': flat_const_grad.detach(), 'A': full_hessian.detach(), 'L': L}, x_tilde=x_tilde)
					#flat_vk, _ = sub_solve(tol_subsolve, subsolver_bdgm, subsolver_args, {'c': flat_const_grad.detach(), 'A': full_hessian.detach(), 'L': L}, params)
					if restarted:
						N_iter = int(1./tol_subsolve)
						N = 0
						k_ = 1
						C = 3
						alp = 1.
						while(N < N_iter):
							N_k = C * math.exp(alp * k_)
							eps = 1./N_k
							flat_vk, _ = sub_solve(eps, init_point, subsolver_bdgm, subsolver_args, {'c': flat_const_grad.div(sqrt_const).detach(),  'L': L}, params, closure)
							init_point = flat_vk
							N += N_k
							k_ += 1
					else:
						if torch.cuda.is_available():
							init_point = torch.zeros_like(init_point).cuda()
						else:
							init_point = torch.zeros_like(init_point)
						flat_vk, _ = sub_solve(tol_subsolve, init_point, subsolver_bdgm, subsolver_args, {'c': flat_const_grad.div(sqrt_const).detach(),  'L': L}, params, closure)
				# print('model after suvbsolver = ', quad_func(flat_vk, flat_const_grad, full_hessian.mul(sqrt_const), L * sqrt_const))
				# print('gradient of quadratic =', quad_func_grad(flat_vk, flat_const_grad, full_hessian, L))
				# print('model g_phi after =', quad_func_grad(flat_vk, flat_const_grad, full_hessian.mul(sqrt_const), L * sqrt_const))

				norm_vk = flat_vk.norm()

				# Projection on the set if outside
				#radius = (grad_norm_xk.mul(sqrt_const / L)).pow(1. / 3).mul(2.)
				#if norm_vk > radius:
				#	print('make a projection. norm = ', norm_vk, 'radius = ', radius)
				#	flat_vk = flat_vk.div(norm_vk).mul(radius)
				#	norm_vk = radius

				# transform vk back to tuple
				vk = ttv.rollup_vector(flat_vk, vk, length, vk_numel)
				# print(vk)
				k += 1
				# print('k=', k)
				# print('vk=',vk, 'size= ', vk[0].size(), vk[1].size())

			# Full step update of parameters
			with torch.no_grad():
				for i in range(length):
					list(params)[i].add_(vk[i])

		return None
