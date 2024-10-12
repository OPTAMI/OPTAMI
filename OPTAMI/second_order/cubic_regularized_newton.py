import torch
from torch.optim.optimizer import Optimizer
from OPTAMI.utils import tuple_to_vec, derivatives, line_search, subproblem_solver

class CubicRegularizedNewton(Optimizer):
    """Implements Cubic Regularized Newton Method.

    It was proposed in `Cubic regularization of Newton method and its global performance`
    by Yurii Nesterov and Boris Polyak. 2006. Mathematical Programming. 108, pp. 177–205.
    https://doi.org/10.1007/s10107-006-0706-8

    Contributors:
        Dmitry Kamzolov
        Dmitry Vilensky-Pasechnyuk

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        L (float): estimated value of Lipschitz constant of the Hessian
        subsolver (Optimizer): optimization method to solve the inner problem by gradient steps
        subsolver_args (dict): arguments for the subsolver such as a learning rate and others
        max_iters (int): number of the inner iterations of the subsolver to solve the inner problem
        rel_acc (float): relative stopping criterion for the inner problem (default: 1e-1)
        testing (bool): if True, it computes some additional tests (default: False)
    """
    ORDER = 2
    MONOTONE = True

    def __init__(self, params, L: float = 1., subsolver: Optimizer = None,
                 subsolver_args: dict = None, max_iters: int = 100,
                 rel_acc: float = 1e-1,
                 verbose: bool = False, testing: bool = False):
        if L <= 0:
            raise ValueError(f"Invalid learning rate: L = {L}")

        super().__init__(params, dict(
            L=L, subsolver=subsolver,
            subsolver_args=subsolver_args or {'lr': 1e-2},
            max_iters=max_iters, rel_acc=rel_acc))
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
            L = group['L']
            rel_acc = group['rel_acc']
            max_iters = group['max_iters']
            subsolver = group['subsolver']
            subsolver_args = group['subsolver_args']

            grad = tuple_to_vec.tuple_to_vector(
                torch.autograd.grad(closure(), list(params), create_graph=True))

            if subsolver is None:
                hessian = derivatives.flat_hessian(grad, list(params))
                h = subproblem_solver.cubic_exact(params=params, grad_approx=grad, hessian_approx=hessian, L=L,
                                                  testing=self.testing)
            else:
                is_satisfactory, h = iterative(params=params, grad=grad, L=L, subsolver=subsolver,
                                               subsolver_args=subsolver_args, max_iters=max_iters, rel_acc=rel_acc)

            if self.testing:
                f_k = closure()
                with torch.no_grad():
                    for i, p in enumerate(params):
                        p.add_(h[i])
                f_k_plus = closure()
                grad = torch.autograd.grad(f_k_plus, params)
                sq_norm = tuple_to_vec.tuple_norm_square(grad)
                norm = sq_norm ** (1/2)
                with torch.no_grad():
                    for i, p in enumerate(params):
                        p.sub_(h[i])
                if self.verbose:
                    print('norm', norm)
                    print('success', f_k - f_k_plus - norm ** (4/3) / (2 * L ** (1/3)))
                    print()
                assert f_k - f_k_plus - norm ** (3/2) / (2 * L ** (1/2)) >= 0

            with torch.no_grad():
                for i, p in enumerate(params):
                    p.add_(h[i])
        return None


def iterative(params, grad, L, subsolver, subsolver_args, max_iters, rel_acc):
    x = torch.zeros_like(tuple_to_vec.tuple_to_vector(
        list(params)), requires_grad=True)
    optimizer = subsolver([x], **subsolver_args)

    for _ in range(max_iters):
        optimizer.zero_grad()
        Hx = derivatives.hvp_from_grad(grad, list(params), x)

        x.grad = grad + Hx + x.mul(L * x.norm() / 2.)
        optimizer.step()

        if x.grad.norm() < rel_acc * grad.norm():
            return True, tuple_to_vec.rollup_vector(x.detach(), list(params))

    return False, tuple_to_vec.rollup_vector(x.detach(), list(params))

