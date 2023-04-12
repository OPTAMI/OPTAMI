import torch
from torch.optim.optimizer import Optimizer
from OPTAMI.utils import tuple_to_vec, derivatives
import math

class DampedNewton(Optimizer):
    """Implements different versions of Damped Newton Method.
    x_{k+1} = x_k - alpha (\nabla^2 f(x) + lambd I)^{-1}\nabla f(x)
    A) classical Damped Newton Method, (variant = None)
    B) Affine-Invariant Cubic Newton, (variant = 'AIC') from
    "A Damped Newton Method Achieves Global O(1/k^2) and Local Quadratic Convergence Rate"
    by Slavomír Hanzely, Dmitry Kamzolov, Dmitry Pasechnyuk, Alexander Gasnikov,
    Peter Richtárik, and Martin Takáč
https://arxiv.org/pdf/2211.00140.pdf
    C) Gradient Regularized Newton (variant = 'GradReg') from
    1) "Regularized Newton Method with Global Convergence O(1/k^2)" by Konstantin Mishchenko
    https://arxiv.org/abs/2112.02089
    2) "Gradient Regularization of Newton Method with Bregman Distances" by Nikita Doikov, Yurii Nesterov
    https://arxiv.org/abs/2112.02952

    Contributors:
        Dmitry Kamzolov
        Dmitry Vilensky-Pasechnyuk
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        variant (str):
        'Classic' - classical Damped Newton Method;
        'AIC' - Affine-Invariant Cubic Newton;
        'GradReg' - Gradient Regularized Newton (default: 'GradReg')
        alpha (float): step-size for Damped Newton Method (default: 1.)
        L (float): estimated value of Lipschitz constant of the Hessian (default: 1.)
        lambd (float): estimated value of Hessian regularizer (default: 0.)
        subsolver (Optimizer): optimization method to solve the inner problem by gradient steps (default: None)
    """
    MONOTONE = True

    def __init__(self, params, variant: str = 'GradReg', alpha: float = 1., L: float = 1.,
                 lambd: float = 0., subsolver: Optimizer = None, verbose: bool = True, testing: bool = False):
        if L <= 0:
            raise ValueError(f"Invalid learning rate: L = {L}")

        super().__init__(params, dict(
            alpha=alpha, variant=variant,
            lambd=lambd, L=L, subsolver=subsolver))

        self.verbose = verbose
        self.testing = testing

    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): a closure that reevaluates the model and returns the loss
        """
        closure = torch.enable_grad()(closure)

        for group in self.param_groups:
            params = group['params']
            variant = group['variant']
            alpha = group['alpha']
            L = group['L']
            lambd = group['lambd']
            subsolver = group['subsolver']
            g = torch.autograd.grad(closure(), list(params), create_graph=True)

            if variant == 'Classic':
                lambd = 0.
                alpha = 1.
            elif variant == 'GradReg':
                g_norm = torch.sqrt(tuple_to_vec.tuple_norm_square(g))
                lambd = torch.sqrt(L * g_norm)
                alpha = 1.

            if subsolver is None:
                h = exact(g, params, lambd, self.testing)
            else:
                raise NotImplementedError()

            if variant == 'AIC':
                G = 0.
                for h_i, g_i in zip(h, g):
                    G += g_i.mul(-h_i).sum()
                G = L * math.sqrt(G)
                alpha = (math.sqrt(1 + 2 * G) - 1) / G


            with torch.no_grad():
                for p, h in zip(params, h):
                    p.add_(h, alpha=alpha)
        return None


def exact(g, params, lambd, testing):
    g_flat = tuple_to_vec.tuple_to_vector(g)
    A = derivatives.flat_hessian(g_flat, list(params))

    c = g_flat.detach().to(torch.double)
    A = A.detach().to(torch.double)

    if c.dim() != 1:
        raise ValueError(f"`c` must be a vector, but c = {c}")

    if A.dim() > 2:
        raise ValueError(f"`A` must be a matrix, but A = {A}")

    if c.size()[0] != A.size()[0]:
        raise ValueError("`c` and `A` mush have the same 1st dimension")

    if testing and (A.t() - A).max() > 0.1:
        raise ValueError("`A` is not symmetric")

    h = torch.linalg.solve(A + torch.diag(torch.ones_like(c)).mul_(lambd), -c)

    return tuple_to_vec.rollup_vector(h, list(params))
