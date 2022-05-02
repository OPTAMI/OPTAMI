import math
import torch
from torch.optim.optimizer import Optimizer
from OPTAMI.utils import tuple_to_vec, derivatives, line_search


class BasicTensorMethod(Optimizer):
    """Implements Inexact Third-order Basic Tensor Method with Bregman Distance Gradient Method as a subsolver.
    Exact version was proposed by Yu.Nesterov in "Implementable tensor methods in unconstrained convex optimization"
    https://doi.org/10.1007/s10107-019-01449-1
    Detailed inexact version was proposed be Yu.Nesterov in "Superfast Second-Order Methods for Unconstrained Convex Optimization"
    https://doi.org/10.1007/s10957-021-01930-y (formula 21)
    Contributors:
        Dmitry Kamzolov
        Dmitry Vilensky-Pasechnyuk
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        L (float, optional): Lipshitz constant of the Hessian (default: 1e+1)
        max_iters (integer, optional): maximal number of inner iterations of the gradient descent to solve subproblem (default: 10)
        subsolver (torch.opt): optimizer for solving
        subsolver_args (dict) : arguments for `subsolver`
    """
    MONOTONE = True

    def __init__(self, params, L: float = 1e+2, subsolver: Optimizer = None, max_iters_outer: int = 50,
                 subsolver_args: dict = None, max_iters: int = None, verbose: bool = True):
        if L <= 0:
            raise ValueError(f"Invalid learning rate: L = {L}")

        super().__init__(params, dict(
            L=L, subsolver=subsolver, max_iters_outer=max_iters_outer,
            subsolver_args=subsolver_args or {'lr': 1e-2}, max_iters=max_iters))
        self.verbose = verbose

    def _add_v(self, params, vector, alpha=1):
        with torch.no_grad():
            for p, v in zip(params, vector):
                p.add_(v, alpha=alpha)

    def _check_stopping_condition(self, closure, params, v, g_norm):
        self._add_v(params, v)
        df_norm = tuple_to_vec.tuple_to_vector(
            torch.autograd.grad(closure(), list(params))).norm()
        self._add_v(params, v, alpha=-1)
        return g_norm <= 1/6 * df_norm

    def step(self, closure):
        """Solves a subproblem.
        Arguments:
            closure (callable): a closure that reevaluates the model and returns the loss.
        """
        closure = torch.enable_grad()(closure)

        for group in self.param_groups:
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
                full_hessian = derivatives.flat_hessian(df, list(params)).to(torch.double)
                eigenvalues, eigenvectors = torch.linalg.eigh(full_hessian)

            for _ in range(max_iters_outer):
                D3xx, Hx = derivatives.third_derivative_vec(
                    closure, list(params), v, flat=True)
                with torch.no_grad():
                    D3xx = D3xx.to(torch.double)
                    Hx = Hx.to(torch.double)
                    Lv3 = v_flat * (v_flat.norm().square() * L)
                    g = df.add(Hx).add(D3xx, alpha=0.5).add(Lv3)

                if self._check_stopping_condition(closure, params, v, g.norm()):
                    self._add_v(params, v)
                    return True

                with torch.no_grad():
                    c = g.div(2. + math.sqrt(2)).sub(Hx).sub(Lv3)

                if subsolver is None:
                    v_flat = exact(L, c.detach(), T=eigenvalues, U=eigenvectors)
                else:
                    v_flat = iterative(params, closure, L, c.detach(),
                                  subsolver, subsolver_args, max_iters)
                with torch.no_grad():
                    v = tuple_to_vec.rollup_vector(v_flat, list(params))

            self._add_v(params, v)
        return False


@torch.no_grad()
def exact(L, c, T, U, tol=1e-10):
    ct = U.t().mv(c)

    def inv(T, L, tau): return T.add(math.sqrt(2 * L) * tau).reciprocal()
    def dual(tau): return  tau.square() + inv(T, L, tau).mul(ct.square()).sum()

    tau_best = line_search.ray_line_search(
        dual,
        middle_point=torch.tensor([2.]),
        left_point=torch.tensor([0.]),
        eps=tol)

    invert = inv(T, L, tau_best)
    x = -U.mv(invert.mul(ct).type_as(U))

    return x


def iterative(params, closure, L, c, subsolver, subsolver_args, max_iters):
    x = torch.zeros_like(tuple_to_vec.tuple_to_vector(
        list(params)), requires_grad=True)
    optimizer = subsolver([x], **subsolver_args)

    for _ in range(max_iters):
        optimizer.zero_grad()
        Hx, __ = derivatives.flat_hvp(closure, list(params), x)

        for p in params:
            if p.grad is not None:
                if p.grad.grad_fn is not None:
                    p.grad.detach_()
                else:
                    p.grad.requires_grad_(False)
                p.grad.zero_()

        x.grad = c + Hx + x.mul(L * x.norm() ** 2)
        optimizer.step()

    return x.detach()
