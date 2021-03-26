import torch
import OPTAMI as opt
from OPTAMI.sup.tuple_to_vec import tuple_to_vector as ttv


def subsolve_cubic_problem(params, closure, L, zeros_tuple, subsolver, subsolver_args, number_inner_iter, inner_rel_err):
    """
    Solves the subproblem with given parameters
        min_x c^T x + 0.5 x^T A x + L / 3 \| x \|^3

    zeros_tuple (tuple): tuple of zeros is used to generate the first point on the correct device like cuda.

    """

    x_ = torch.zeros_like(ttv(zeros_tuple), requires_grad=True)

    params_local = [x_]

    optimizer = subsolver(params_local, **subsolver_args)

    zeroing_optimizer = subsolver(params, **subsolver_args)  # optimizer for zeroing gradients of params

    i = 0
    while i < number_inner_iter:
        optimizer.zero_grad()

        # computation hessian vector part
        hess_vec, full_grad = opt.sup.derivatives.flat_hvp(closure, list(params), x_)
        zeroing_optimizer.zero_grad()

        # computation of gradient for the problem
        x_.grad = full_grad + hess_vec + x_.mul(L * x_.norm())

        if x_.grad.norm() < inner_rel_err * full_grad.norm():
            print('Inner method reach stopping criterion on', i)
            i = number_inner_iter + 1
        # step of the problem
        optimizer.step()
        i = i+1
    if i == number_inner_iter:
        print('Inner method not reach stopping criterion')
    return x_.detach()  # flat vector
