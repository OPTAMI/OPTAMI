import torch
from torch.optim.optimizer import Optimizer
from OPTAMI.utils import tuple_to_vec, derivatives
import math

class DampedNewton(Optimizer):
    """Implements Damped Newton Method.
    It had been proposed in `TODO`
    TODO
    Contributors:
        Dmitry Kamzolov
        Dmitry Vilensky-Pasechnyuk
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        is_adaptive (boolean): if False then method works with fixed L. If True then method tunes L adaptively (default: False)
        L (float): estimated value of Lipschitz constant of the Hessian
        subsolver (Optimizer): optimization method to solve the inner problem by gradient steps
        subsolver_args (dict): arguments for the subsolver such as a learning rate and others
        max_iters (int): number of the inner iterations of the subsolver to solve the inner problem
        rel_acc (float): relative stopping criterion for the inner problem
    """
    MONOTONE = True

    def __init__(self, params, alpha: float = 1, L: float = 1e+2,
                 lambd: float = 0., subsolver: Optimizer = None,
                 affine_invariant: bool = False, verbose: bool = True):
        if L <= 0:
            raise ValueError(f"Invalid learning rate: L = {L}")

        super().__init__(params, dict(
            alpha=alpha, affine_invariant=affine_invariant, 
            lambd=lambd, L=L, subsolver=subsolver))

        self.verbose = verbose

    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): a closure that reevaluates the model and returns the loss
        """
        closure = torch.enable_grad()(closure)

        for group in self.param_groups:
            params = group['params']

            alpha = group['alpha']
            affine_invariant = group['affine_invariant']
            L = group['L']
            lambd = group['lambd']
            subsolver = group['subsolver']
            g = torch.autograd.grad(closure(), list(params), create_graph=True)

            if subsolver is None:
                h = exact(g, params, lambd)
            else:
                raise NotImplementedError()

            if affine_invariant:
                G = 0.
                for h_i, g_i in zip(h, g):
                    G += g_i.mul(-h_i).sum()
                G = L * math.sqrt(G)
                alpha = (math.sqrt(1 + 2*G) - 1) / G

            #if gradient_regularization:

            with torch.no_grad():
                for i, p in enumerate(params):
                    p.add_(h[i], alpha=alpha)
        return None


def exact(g, params, lambd):
    g_flat = tuple_to_vec.tuple_to_vector(g)
    H = derivatives.flat_hessian(g_flat, list(params))

    c = g_flat.clone().detach().to(torch.double)
    A = H.clone().detach().to(torch.double) # to check do we actually need this copy

    if c.dim() != 1:
        raise ValueError(f"`c` must be a vector, but c = {c}")

    if A.dim() > 2:
        raise ValueError(f"`A` must be a matrix, but A = {A}")

    if c.size()[0] != A.size()[0]:
        raise ValueError("`c` and `A` mush have the same 1st dimension")

    if (A.t() - A).max() > 0.1:
        raise ValueError("`A` is not symmetric")

    h = torch.linalg.solve(A + lambd * torch.eye(*c.size()), -c)
    # add assert Ax = with verbose = true

    return tuple_to_vec.rollup_vector(h, list(params))
