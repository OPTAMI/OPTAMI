import torch
from torch.optim.optimizer import Optimizer
from OPTAMI.utils import tuple_to_vec, derivatives


class GlobalNewton(Optimizer):
    """Implements Globally Regularized Newton Method.
    It had been proposed in `Regularized Newton Method with Global O(1/k^2) Convergence`
    https://arxiv.org/abs/2112.02089
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

    def __init__(self, params, L: float = 1e+2, subsolver: Optimizer = None, verbose: bool = True):
        if L <= 0:
            raise ValueError(f"Invalid learning rate: L = {L}")

        super().__init__(params, dict(L=L, subsolver=subsolver))

        self.verbose = verbose

    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): a closure that reevaluates the model and returns the loss
        """
        closure = torch.enable_grad()(closure)

        for group in self.param_groups:
            params = group['params']

            L = group['L']
            subsolver = group['subsolver']

            g = tuple_to_vec.tuple_to_vector(
                torch.autograd.grad(closure(), list(params), create_graph=True))

            lambd = torch.sqrt(L * torch.norm(g))

            if subsolver is None:
                x = exact(g, params, lambd)
            else:
                raise NotImplementedError()

            with torch.no_grad():
                for i, p in enumerate(params):
                    p.add_(x[i])
        return None


def exact(g, params, lambd):
    H = derivatives.flat_hessian(g, list(params))

    c = g.clone().detach().to(torch.double)
    A = H.clone().detach().to(torch.double)

    if c.dim() != 1:
        raise ValueError(f"`c` must be a vector, but c = {c}")

    if A.dim() > 2:
        raise ValueError(f"`A` must be a matrix, but A = {A}")

    if c.size()[0] != A.size()[0]:
        raise ValueError("`c` and `A` mush have the same 1st dimension")

    if (A.t() - A).max() > 0.1:
        raise ValueError("`A` is not symmetric")

    x = torch.linalg.solve(A + lambd * torch.eye(*c.size()), -c)
    return tuple_to_vec.rollup_vector(x, list(params))
