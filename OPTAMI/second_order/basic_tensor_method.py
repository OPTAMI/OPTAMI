
import torch
from torch.optim.optimizer import Optimizer
from OPTAMI.utils import tuple_to_vec, derivatives, line_search


class BasicTensorMethod(Optimizer):
    """Implements Inexact Third-order Basic Tensor Method with Bregman Distance Gradient Method as a subsolver.

    Exact version was proposed in "Implementable Tensor Methods in Unconstrained Convex Optimization." by Yu.Nesterov.
    2021. Mathematical Programming, 186, pp.157-183.
    https://doi.org/10.1007/s10107-019-01449-1
    Detailed inexact version was proposed in "Superfast Second-Order Methods for Unconstrained Convex Optimization" by Yu.Nesterov.
    2021. Journal of Optimization Theory and Applications, 191, pp.1-30.
    https://doi.org/10.1007/s10957-021-01930-y (formula 21)

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        L (float, optional): Lipshitz constant of the Hessian (default: 1e+1)
        max_iters (integer, optional): maximal number of inner iterations of the gradient descent to solve subproblem (default: 10)
        subsolver (torch.opt): optimizer for solving
        subsolver_args (dict) : arguments for `subsolver`
    """

    MONOTONE = True
    ORDER = 3

    def __init__(self, params, L: float = 1., subsolver: Optimizer = None, max_iters_outer: int = 50,
                 subsolver_args: dict = None, max_iters: int = None, verbose: bool = True, testing: bool = False):
        if L <= 0:
            raise ValueError(f"Invalid learning rate: L = {L}")
        super().__init__(params, dict(
            L=L, subsolver=subsolver, max_iters_outer=max_iters_outer,
            subsolver_args=subsolver_args or {'lr': 1e-2}, max_iters=max_iters))
        self.verbose = verbose
        self.testing = testing

        if len(self.param_groups) != 1:
            raise ValueError("Basic Tensor Method doesn't support per-parameter options "
                             "(parameter groups)")


    def _add_v(self, params, vector, alpha=1.0):
        with torch.no_grad():
            for p, v in zip(params, vector):
                p.add_(v, alpha=alpha)

    def _check_stopping_condition(self, closure, params, v, g_norm):
        self._add_v(params, v)
        df_norm = tuple_to_vec.tuple_to_vector(
            torch.autograd.grad(closure(), list(params))).norm()
        self._add_v(params, v, alpha=-1)
        return g_norm <= 1 / 6 * df_norm

    def step(self, closure):
        """Solves a subproblem.
        Arguments:
            closure (callable): a closure that reevaluates the model and returns the loss.
        """
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        params = group['params']


        L = group['L']
        subsolver = group['subsolver']
        max_iters = group['max_iters']
        subsolver_args = group['subsolver_args']
        max_iters_outer = group['max_iters_outer']

        df = tuple_to_vec.tuple_to_vector(
            torch.autograd.grad(closure(), list(params), create_graph=True))

        v_flat = torch.zeros_like(df)
        v = tuple_to_vec.rollup_vector(v_flat, list(params))

        if subsolver is None:
            full_hessian = derivatives.flat_hessian(df, list(params))
            eigenvalues, eigenvectors = torch.linalg.eigh(full_hessian)

        for _ in range(max_iters_outer):
            D3xx, Hx = derivatives.third_derivative_vec(
                closure, list(params), v, flat=True)
            with torch.no_grad():

                Lv3 = v_flat * L * v_flat.norm() ** 2
                mid = Hx + Lv3
                g = df.add(D3xx, alpha=0.5).add(mid)

            if self._check_stopping_condition(closure, params, v, g.norm()):
                self._add_v(params, v)
                return None

            with torch.no_grad():
                c = g.div(2. + 2 ** 0.5).sub(mid)

            if subsolver is None:
                v_flat = exact(L, c.detach(), T=eigenvalues, U=eigenvectors)
            else:
                v_flat = iterative(params, closure, L, c.detach(),
                              subsolver, subsolver_args, max_iters)
            with torch.no_grad():
                v = tuple_to_vec.rollup_vector(v_flat, list(params))

        self._add_v(params, v)
        return None


@torch.no_grad()
def exact(L, c, T, U, tol=1e-10):
    ct = U.t().mv(c)
    def inv(T, L, tau): return  T.add((2 * L) ** 0.5 * tau).reciprocal()
    def dual(tau): return  tau ** 2 + inv(T, L, tau).mul(ct.square()).sum()

    tau_best = line_search.ray_line_search(dual, left_point=0., middle_point=2., eps=tol)

    invert = inv(T, L, tau_best)
    x = -U.mv(invert.mul(ct).type_as(U))

    return x


def iterative(params, closure, L, c, subsolver, subsolver_args, max_iters):
    x = torch.zeros_like(tuple_to_vec.tuple_to_vector(list(params)), requires_grad=True)
    optimizer = subsolver([x], **subsolver_args)

    for _ in range(max_iters):
        optimizer.zero_grad()
        Hx, __ = derivatives.flat_hvp(closure, list(params), x)
        x.grad = c + Hx + x.mul(L * x.norm() ** 2)
        optimizer.step()

    return x.detach()
