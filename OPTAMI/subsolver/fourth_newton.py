import torch
import math
from OPTAMI.sup import _line_search as l_s


def _optim_test(x, c, A, L):
    # c + |x|^2 * L * x + A * x
    first = x.mul(x.square().sum()).mul(L)
    return first.add(c).add(A.mv(x)).abs().max()


def _optim_dual(tau, T, ct, L):
    # tau - ct * ct * inv^2 * sqrt(2 * L) / 2 
    return tau.sub(ct.square().mul(inversion(T, L, tau).square()).sum(), alpha=math.sqrt(L / 2.))


def _first_ineq_test(x, tau, L, U):
    return (U.t().mv(x)).square().sum().mul(math.sqrt(L / 2)).sub(tau).abs()


def inversion(T, L, tau):
    return T.add(tau.mul(math.sqrt(2 * L))).reciprocal()


def dual_func(tau, ct, T, L):
    # tau^2 * 0.5 + inv * ct * ct * 0.5
    first = tau.square().mul(0.5)
    second = inversion(T, L, tau).mul(ct.square()).sum().div(2.)
    return first.add(second)


def fourth_subsolver(c=None, A=None, T=None, U=None, L=0., fourth_line_search_eps=1e-10):
    if L < 0.0:
        raise ValueError("Invalid Lipshitz constant: {}".format(L))
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
    # g := beta/2 * tau^2 + 1/2 ct^T * (T +  alpha * tau * I)^{-1} ct

    g = lambda tau: dual_func(tau, ct, T, L)

    left_point = torch.tensor([0.])
    middle_point = torch.tensor([2.])
    tau_best = l_s.ray_line_search(g, middle_point, left_point, eps=fourth_line_search_eps)
    # print(tau_best)
    invert = inversion(T, L, tau_best)

    x_sol = - U.mv(invert.mul(ct).type_as(U))

    # print('gradient of dual problem ', _optim_dual(tau_best, T, ct, L))
    # print('optimality test ', _optim_test(x_sol, c, A, L))
    # print('first ineq ', _first_ineq_test(x_sol, tau_best, L, U))

    if _optim_test(x_sol, c, A, L).ge(0.01).item():
        raise ValueError('Error: x in subproblem is not optimal')

    if _first_ineq_test(x_sol, tau_best, L, U).ge(0.01).item():
        raise ValueError('Error: x or tau is incorrect')

    return x_sol
