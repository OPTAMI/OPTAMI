import math
import torch
from torch_optimizer import Optimizer
from OPTAMI.utils import tuple_to_vec, derivatives, line_search


class BDGM:
    """Implements Bregman Distance Gradient Method for 4th degree Taylor polynomial.
    It had been proposed in `Superfast Second-Order Methods for Unconstrained Convex Optimization` 
    https://link.springer.com/article/10.1007/s10957-021-01930-y and listed in
    `Near-Optimal Hyperfast Second-Order Method for Convex Optimization`
    https://link.springer.com/chapter/10.1007/978-3-030-58657-7_15
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

    def __init__(self, params, L: float = 1e+2, subsolver: Optimizer = None,
                 subsolver_args: dict = None, max_iters: int = None, verbose: bool = True):
        if L <= 0:
            raise ValueError(f"Invalid learning rate: L = {L}")

        self.L = L
        self.params = params
        self.max_iters = max_iters
        self.subsolver = subsolver
        self.subsolver_args = subsolver_args
        self.verbose = verbose

    def _add_x(self, params, x, alpha=1):
        x_ = tuple_to_vec.rollup_vector(x, list(params))
        with torch.no_grad():
            for i, p in enumerate(params):
                p.add_(x_[i], alpha=alpha)

    def _check_stopping_condition(self, closure, params, x, g_norm):
        self._add_x(params, x)
        df_norm = tuple_to_vec.tuple_to_vector(
            torch.autograd.grad(closure(), list(params))).norm()
        self._add_x(params, x, alpha=-1)

        if self.verbose:
            print(f"g_norm = {g_norm}, df_norm = {df_norm}")

        return g_norm <= 1/6 * df_norm

    def solve(self, closure, max_iters_outer: int = 50):
        """Solves a subproblem.
                Arguments:
                        closure (callable): A closure that reevaluates the model and returns the loss.
                """
        closure = torch.enable_grad()(closure)

        df = tuple_to_vec.tuple_to_vector(
            torch.autograd.grad(closure(), list(self.params), create_graph=True))
        x = torch.zeros_like(df)

        if self.subsolver is None:
            H = derivatives.flat_hessian(df, self.params)

        for _ in range(max_iters_outer):
            D3xx, Hx = derivatives.third_derivative_vec(
                closure, list(self.params), x)
            g = df + Hx + 1/2 * D3xx + self.L * x * x.norm().square()

            if self._check_stopping_condition(closure, self.params, x, g.norm()):
                self._add_x(self.params, x)
                return True

            c = g.sub(Hx + self.L * x * x.norm().square(),
                      alpha=(2 + math.sqrt(2))) / (2 + math.sqrt(2))
            if self.subsolver is None:
                x = exact(self.L, c.detach(), H)
            else:
                x = iterative(self.params, closure, self.L, c.detach(),
                              self.subsolver, self.subsolver_args, self.max_iters)

        self._add_x(self.params, x)
        return False


def exact(L, c, A, tol=1e-10):
    if c.dim() != 1:
        raise ValueError(f"`c` must be a vector, but c = {c}")

    if A.dim() > 2:
        raise ValueError(f"`A` must be a matrix, but A = {A}")

    if c.size()[0] != A.size()[0]:
        raise ValueError("`c` and `A` mush have the same 1st dimension")

    if (A.t() - A).max() > 0.1:
        raise ValueError("`A` is not symmetric")

    T, U = torch.linalg.eigh(A)
    ct = U.t().mv(c)

    def inv(T, L, tau): return (T + math.sqrt(2 * L) * tau).reciprocal()
    def dual(tau): return 1/2 * tau.square() + 1 / \
        2 * inv(T, L, tau).mul(ct.square()).sum()

    tau_best = line_search.ray_line_search(
        dual, eps=tol,
        middle_point=torch.tensor([2.]),
        left_point=torch.tensor([0.]))

    invert = inv(T, L, tau_best)
    x = -U.mv(invert.mul(ct).type_as(U))

    if not (c + L * x.norm()**2 * x + A.mv(x)).abs().max().item() < 0.01:
        raise ValueError('obtained `x` is not optimal')

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
