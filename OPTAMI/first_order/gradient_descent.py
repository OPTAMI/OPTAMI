import torch
from torch.optim.optimizer import Optimizer


class GradientDescent(Optimizer):
    """Implements Gradient Descent

    Contributors:
        Dmitry Kamzolov

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        L (float): estimated value of Lipschitz constant of the gradient
    """
    MONOTONE = True
    ORDER = 1

    def __init__(self, params, L: float = 1., verbose: bool = True, testing: bool = False):
        if L <= 0:
            raise ValueError(f"Invalid Lipschitz constant: L = {L}")

        super().__init__(params, dict(L=L))

        self.verbose = verbose
        self.testing = testing
        if len(self.param_groups) != 1:
            raise ValueError("Method doesn't support per-parameter options "
                             "(parameter groups)")

    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): a closure that reevaluates the model and returns the loss
        """
        closure = torch.enable_grad()(closure)
        group = self.param_groups[0]
        params = group['params']
        L = group['L']
        closure().backward()
        with torch.no_grad():
            for p in params:
                p.sub_(p.grad / L)
        return None

