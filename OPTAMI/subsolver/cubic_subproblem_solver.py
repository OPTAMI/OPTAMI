import torch
import OPTAMI as opt
from OPTAMI.sup.tuple_to_vec import tuple_to_vector as ttv


def subsolve_cubic_problem(params, closure, L, subsolver, subsolver_args, number_inner_iter, cubic_linear_part):
    """
    Solves the subproblem with given parameters
        min_x c^T x + 0.5 x^T A x + L / 3 \| x \|^3
    L (float): Lipshitz constant of the Second-order

    subsolver(Optimizer): Optimization method to solve inner problem by gradient steps

    number_inner_iter(int): number of inner iteration of Subsolver to solve the inner problem

    cubic_linear_part(tensor): Should be a flat vector. If used, then function for optimization
        equal to <cubic_linear_part,x> + f(x).
        To be used for inexact, stochastic or distributed versions of Cubic Newton Method.
    """

    x_ = torch.zeros_like(cubic_linear_part, requires_grad=True)

    params_local = [x_]

    optimizer = subsolver(params_local, **subsolver_args)

    zeroing_optimizer = subsolver(params, **subsolver_args)  # optimizer for zeroing gradients of params

    for i in range(number_inner_iter):
        optimizer.zero_grad()

        # computation hessian vector part
        hess_vec, grad_closure = opt.sup.derivatives.flat_hvp(closure, list(params), x_)
        zeroing_optimizer.zero_grad()

        # computation of gradient for the problem
        full_linear_grad = grad_closure + cubic_linear_part  # step for cubic_linear_part * x + f(x)
        x_.grad = full_linear_grad + hess_vec + x_.mul(L * x_.norm())

        # step of the problem
        optimizer.step()

    return x_.detach()  # flat vector
