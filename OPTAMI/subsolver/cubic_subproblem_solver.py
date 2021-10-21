import torch
import OPTAMI as opt
from OPTAMI.sup.tuple_to_vec import tuple_to_vector as ttv
import math
from OPTAMI.sup.derivatives import flat_hessian as flat_hes
from OPTAMI.sup import _line_search as l_s


def _optim_test(x, c, A, L):
    # c + |x| * L * x / 2 + A * x
    first = x.mul(x.square().sum().sqrt()).mul(L / 2.)
    return first.add(c).add(A.mv(x)).abs().max()


# def _optim_dual(tau, T, ct, L):
#    # tau^2  - ct * ct * inv^2 * sqrt(2 * L) / 2
#    return tau.sub(ct.square().mul(inversion(T, L, tau).square()).sum(), alpha=math.sqrt(L / 2.))


# def _first_ineq_test(x, tau, L, U):
#    return (U.t().mv(x)).square().sum().mul(math.sqrt(L / 2)).sub(tau).abs()


def inversion(T, L, tau):
    return T.add(tau.mul(L / 2.)).reciprocal()


def dual_func(tau, ct, T, L):
    # tau^3 *L/12 + inv * ct * ct * 0.5
    first = tau.pow(3).mul(L / 12.)
    second = inversion(T, L, tau).mul(ct.square()).sum().div(2.)
    return first.add(second)


def cubic_subsolver(L=0., closure=None, params=None, c=None, A=None, T=None, U=None, cubic_line_search_eps=1e-10):
    # Almost exact subsolver via dual problem solving
    if L < 0.:
        raise ValueError("Invalid Lipshitz constant: {}".format(L))
    if c is None or A is None:
        if closure is None or params is None:
            raise ValueError("Invalid Input: {}".format(closure))
        grads = torch.autograd.grad(closure(), list(params), create_graph=True)

        flat_grads = ttv(grads)
        # print(flat_grads)

        c = flat_grads.clone().detach().to(torch.double)
        # Hessian computation

        full_hessian = flat_hes(flat_grads, list(params))
        A = full_hessian.clone().detach().to(torch.double)
        # SVD decomposition
        T, U = torch.linalg.eigh(A)

    if c.dim() != 1:
        raise ValueError("Should be a vector: {}".format(c))
    if A.dim() > 2:
        raise ValueError("Should be a matrix: {}".format(A))
    if c.size()[0] != A.size()[0]:
        raise ValueError("Vector and matrix should have the same dimensions")
    if (A.t() - A).max() > 0.1:
        raise ValueError("Non-symmetric matrix A")

    if U is None or T is None:
        T, U = torch.linalg.eigh(A)

    # ct = U^T * c
    ct = U.t().mv(c)
    # ct = c

    # to solve -min g(tau), tau>0
    # g := 
    g = lambda tau: dual_func(tau, ct, T, L)

    left_point = torch.tensor([0.])
    middle_point = torch.tensor([2.])
    tau_best = l_s.ray_line_search(g, middle_point, left_point, eps=cubic_line_search_eps)
    # print(tau_best)
    invert = inversion(T, L, tau_best)

    x_sol = - U.mv(invert.mul(ct).type_as(U))

    # print('gradient of dual problem ', _optim_dual(tau_best, T, ct, L))
    # print('optimality test ', _optim_test(x_sol, c, A, L))
    # print('first ineq ', _first_ineq_test(x_sol, tau_best, L, U))

    if _optim_test(x_sol, c, A, L).ge(0.01).item():
        raise ValueError('Error: x in subproblem is not optimal')

    # if _first_ineq_test(x_sol, tau_best, L, U).ge(0.01).item():
    #    raise ValueError('Error: x or tau is incorrect')

    return x_sol


def subsolve_cubic_problem(params, closure, L, zeros_tuple, subsolver, subsolver_args, number_inner_iter,
                           inner_rel_err):
    """
    Solves the subproblem with given parameters
        min_x c^T x + 0.5 x^T A x + L / 6 \| x \|^3

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
        x_.grad = full_grad + hess_vec + x_.mul(L * x_.norm() / 2.)

        if x_.grad.norm() < inner_rel_err * full_grad.norm():
            print('Inner method reach stopping criterion on', i)
            i = number_inner_iter + 1
        # step of the problem
        optimizer.step()
        i += 1
    if i == number_inner_iter:
        print('Inner method not reach stopping criterion')
    return x_.detach()  # flat vector
