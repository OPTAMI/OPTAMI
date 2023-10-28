import torch
import math
from torch.optim.optimizer import Optimizer
from OPTAMI.higher_order._supplemetrary import step_definer


class Superfast(Optimizer):
    """Implements Inexact Accelerated Tensor Method.
     Exact version was proposed by Yu.Nesterov in "Implementable tensor methods in unconstrained convex optimization"
     https://doi.org/10.1007/s10107-019-01449-1
     Detailed inexact version was proposed be Yu.Nesterov in "Superfast Second-Order Methods for Unconstrained Convex Optimization"
     https://doi.org/10.1007/s10957-021-01930-y
    Contributors:
        Dmitry Kamzolov
        Dmitry Vilensky-Pasechnyuk
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        L (float): estimated value of Lipschitz constant for the hessian (default: 1.)
        order (int): order of the method (default: 3)
        a_step (float): constant controlling the step size; bigger constant is bigger step;
        theoretical a_step=1/604 for p=3, a_step=1/80 for p=2 (default: None)
        TensorStepMethod (Optimizer): method to be accelerated;
        for p=3 - BasicTensorMethod
        for p=2 - CubicRegularizedNewton
        for p=2 - GradientDescent
        (default: None)
        tensor_step_kwargs (dict): kwargs for TensorStepMethod (default: None)
        subsolver (Optimizer): method to solve the inner problem (default: None)
        subsolver_args (dict): arguments for the subsolver (default: None)
        max_iters (int): number of the inner iterations of the subsolver to solve the inner problem (default: None)
    """
    MONOTONE = False
    SKIP_TEST_LOGREG = False

    def __init__(self, params, L: float = 1., order: int = 3, a_step: float = None,
                 TensorStepMethod: Optimizer = None, tensor_step_kwargs: dict = None,
                 subsolver: Optimizer = None, subsolver_args: dict = None,
                 max_iters: int = None, verbose: bool = True, testing: bool = False):
        if L <= 0:
            raise ValueError(f"Invalid learning rate: L = {L}")

        super().__init__(params, dict(L=L))

        self.verbose = verbose
        self.testing = testing
        if len(self.param_groups) != 1:
            raise ValueError("Superfast doesn't support per-parameter options "
                             "(parameter groups)")
        group = self.param_groups[0]
        params = group['params']
        p = next(iter(params))
        state_common = self.state[p]
        state_common['k'] = 0

        self.order = order

        if a_step is None:
            a_step = (2 * order - 1) / (2 * (order + 1) * (2 * order + 1)) \
                     * math.factorial(order - 1) * 2 * (1 / (2 * order)) ** order
        self.a_step = a_step

        # Step initialization: if order = 3 then Basic Tensor step,
        # if order = 2 then Cubic Newton, if order = 1 then Gradient Descent
        self.tensor_step_method = step_definer(params=params, L=L, order=order,
                 TensorStepMethod=TensorStepMethod, tensor_step_kwargs=tensor_step_kwargs,
                 subsolver=subsolver, subsolver_args=subsolver_args,
                 max_iters=max_iters, verbose=verbose, testing=testing)

        # Initialization of intermediate points
        for p in params:
            state = self.state[p]
            state['x0'] = p.detach().clone()
            state['x'] = state['x0'].clone()
            state['v'] = state['x0'].clone()
            state['df_sum'] = torch.zeros_like(p)

    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): a closure that reevaluates the model and returns the loss.
        """
        closure = torch.enable_grad()(closure)

        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        params = group['params']
        p = next(iter(params))
        state_common = self.state[p]

        k = state_common['k']

        L = group['L']

        alpha = (1. - 1 / (k + 1)) ** (self.order + 1)

        with torch.no_grad():
            for p in params:
                state = self.state[p]
                p.mul_(alpha).add_(state['v'], alpha=1 - alpha)

        self.tensor_step_method.step(closure)
        self.zero_grad()

        closure().backward()

        with torch.no_grad():
            a = self.a_step * ((k + 1) ** (self.order + 1) - k ** (self.order + 1)) / L
            for p in params:
                state = self.state[p]
                state['x'].zero_().add_(p)
                state['df_sum'].add_(p.grad, alpha=a)

            if self.order == 1:
                scalier = 1.
            else:
                norm_squared = 0.
                for p in params:
                    state = self.state[p]
                    norm_squared += state['df_sum'].square().sum()

                power = (1. - self.order) / (2. * self.order)
                scalier = norm_squared ** power

            for p in params:
                state = self.state[p]
                state['v'].zero_().add_(state['x0']).sub_(state['df_sum'], alpha=scalier)

            state_common['k'] += 1

        return None
