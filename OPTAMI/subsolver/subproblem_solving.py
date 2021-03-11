import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import OPTAMI as opt
import matplotlib.pyplot as plt
import math
import numpy as np

from OPTAMI.sup.tuple_to_vec import tuple_to_vector as ttv

def _optim_test(x, c, A, L):
    # c + |x|^2 * L * x + A * x
    first = x.mul(x.square().sum()).mul(L)
    return first.add(c).add(A.mv(x)).abs().max()

def _optim_test_hess_v(x, c, Ax, L):
    # c + |x|^2 * L * x + A * x
    first = x.mul(x.square().sum()).mul(L)
    return first.add(c).add(Ax).abs().max()


#def sub_solve(tol, subsolver, subsolver_args, closure_args, x_tilde):
def sub_solve(tol, subsolver, subsolver_args, closure_args, params, closure):
    """
    Solves the subproblem with given parameters
        min_x c^T x + 0.5 x^T A x + L / 4 \| x \|^4                          (1)
    tol: float
        solution tolerance
    subsolver: optimizer
        method that solves the problem
    subsolver_args: dict
        arguments for subsolver
    closure_args: dict
        contains c, L from (1) !!!! c MUST NOT BE DETACHED!!!!!!!!!!!!!!!!
    x_tilde_params : model.parameters()
        the point, where we need to compute hessian
    """

    c, L = closure_args.values()
    x_ = torch.zeros_like(c, requires_grad=True)
    params_local = [x_]

    optimizer = subsolver(params_local, **subsolver_args)

    zeroing_optimizer = subsolver(params, **subsolver_args)  # optimizer for zeroing gradients of params

    for i in range(int(1/tol)):
        optimizer.zero_grad()

        # computation hessian vector part
        hess_vec, grad_closure = opt.sup.derivatives.flat_hvp(closure, list(params), x_)
        zeroing_optimizer.zero_grad() # zeroing gradients of params

        x_.grad = c + hess_vec + x_.mul(L * x_.norm() ** 2)

        optimizer.step()

    x_sol = params_local[0].detach()
    #if _optim_test(x_sol, c, A, L).ge(0.01).item():

    #uncomment print to monitor subproblem tolerance
    #print("_optim_test = ",_optim_test_hess_v(x_sol, c, hess_vec, L))

    #zeroing_optimizer.zero_grad()
    #    raise ValueError('Error: x in subproblem is not optimal. GradNorm = ' + str(_optim_test(x_sol, c, A, L).item()))
    #print("_optim_test = ", _optim_test(x_sol, c, A, L).item())

    return x_sol