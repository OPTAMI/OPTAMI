from torch.optim.optimizer import Optimizer
from OPTAMI.utils import tuple_to_vec
import warnings
import OPTAMI
import torch
import math


class Superfast(Optimizer):
    """Implements Inexact Accelerated Tensor Method.
     Exact version was proposed by Yu.Nesterov in "Implementable tensor methods in unconstrained convex optimization"
     https://doi.org/10.1007/s10107-019-01449-1
     Detailed inexact version was proposed by Yu.Nesterov in "Superfast Second-Order Methods for Unconstrained Convex Optimization"
     https://doi.org/10.1007/s10957-021-01930-y
    Contributors:
        Dmitry Kamzolov
        Dmitry Vilensky-Pasechnyuk
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        L (float): estimated value of Lipschitz constant for the hessian (default: 1e+3)
        divider (float): constant controlling the step size; lower constant is a bigger step (default: 604.8, by theory)
        subsolver (Optimizer): method to solve the inner problem (default: BDGM)
    """
    MONOTONE = False

    def __init__(self, params, L: float = 1e+3, order: int = 3, a_step: float = None,
                 TensorStepMethod: Optimizer = None, tensor_step_kwargs: dict = None,
                 subsolver: Optimizer = None, subsolver_args: dict = None,
                 max_iters: int = None, verbose: bool = True):
        if L <= 0:
            raise ValueError(f"Invalid learning rate: L = {L}")

        super().__init__(params, dict(L=L))
        if len(self.param_groups) != 1:
            raise ValueError("Superfast doesn't support per-parameter options "
                             "(parameter groups)")

        self.order = order
        self.TensorStepMethod = TensorStepMethod
        self.subsolver = subsolver
        self.subsolver_args = subsolver_args
        self.max_iters = max_iters
        self.tensor_step_kwargs = tensor_step_kwargs
        self.tensor_step_method = None

        if a_step is None:
            a_step = (2 * order - 1) / (2 * (order + 1) * (2 * order + 1)) \
                     * math.factorial(order - 1) * 2 * (1 / (2 * order)) ** order
        self.a_step = a_step

        self.verbose = verbose


    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): a closure that reevaluates the model and returns the loss.
        """
        closure = torch.enable_grad()(closure)
        assert len(self.param_groups) == 1

        group = self.param_groups[0]
        params = group['params']
        L = group['L']

        if self.tensor_step_method is None:
            if self.TensorStepMethod is None:
                if self.order == 3:
                    self.tensor_step_method = OPTAMI.BasicTensorMethod(
                        params, L=L, subsolver=self.subsolver, verbose=self.verbose,
                        subsolver_args=self.subsolver_args, max_iters=self.max_iters)
                elif self.order == 2:
                    self.tensor_step_method = OPTAMI.CubicRegularizedNewton(
                        params, L=L, subsolver=self.subsolver, verbose=self.verbose,
                        subsolver_args=self.subsolver_args, max_iters=self.max_iters)
                else:  # order = 1
                    self.tensor_step_method = torch.optim.SGD(params, lr=1. / L)
            else:
                if not hasattr(self.TensorStepMethod, 'MONOTONE') or not self.TensorStepMethod.MONOTONE:
                    warnings.warn("`TensorStepMethod` should be monotone!")
                self.tensor_step_method = self.TensorStepMethod(params, **self.tensor_step_kwargs)

        p = next(iter(params))
        state_common = self.state[p]

        if 'k' not in state_common:
            state_common['k'] = 0

        k = state_common['k']
        alpha = (1. - 1 / (k + 1)) ** (self.order + 1)

        for p in params:
            state = self.state[p]

            if ('v' not in state) or ('x' not in state):
                state['x0'] = p.detach().clone()
                state['x'] = state['x0'].clone()
                state['v'] = state['x0'].clone()
                state['df_sum'] = torch.zeros_like(p)

            with torch.no_grad():
                v = state['v']
                p.mul_(alpha).add_(v, alpha=1 - alpha)

        self.tensor_step_method.step(closure)
        self.zero_grad()

        closure().backward()

        a = self.a_step * ((k + 1) ** (self.order + 1) - k ** (self.order + 1)) / L
        for p in params:
            state = self.state[p]
            with torch.no_grad():
                state['x'].zero_().add_(p)
                state['df_sum'].add_(p.grad, alpha=a)

        if self.order == 1:
            scalier = 1.
        else:
            norm_squared = 0.
            for p in params:
                state = self.state[p]
                with torch.no_grad():
                    norm_squared += tuple_to_vec.tuple_norm_square(state['df_sum'])

            power = (1. - self.order) / (2. * self.order)
            scalier = norm_squared ** power

        for p in params:
            state = self.state[p]
            with torch.no_grad():
                state['v'].zero_().add_(state['x0']).sub_(state['df_sum'], alpha=scalier)

        state_common['k'] += 1

        return None
