import torch
from math import factorial
from torch.optim.optimizer import Optimizer
from ._supplemetrary import step_definer


class NesterovAcceleration(Optimizer):
    """Implements Nesterov Accelerated Tensor Method.

     Exact version was proposed in "Implementable Tensor Methods in Unconstrained Convex Optimization." by Yu.Nesterov.
    2021. Mathematical Programming, 186, pp.157-183.
    https://doi.org/10.1007/s10107-019-01449-1
    Detailed inexact version was proposed in "Superfast Second-Order Methods for Unconstrained Convex Optimization" by Yu.Nesterov.
    2021. Journal of Optimization Theory and Applications, 191, pp.1-30.
    https://doi.org/10.1007/s10957-021-01930-y

    Contributors:
        Dmitry Kamzolov
        Dmitry Vilensky-Pasechnyuk

    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups
        L (float): Estimated value of Lipschitz constant for the Hessian (default: 1.0)
        order (int): Order of the method (default: 2)
        a_step (float): Constant controlling the step size; bigger constant results in a bigger step.
        Theoretical values are a_step=1/604 for p=3, a_step=1/24 for p=2 (default: None)
        TensorStepMethod (Optimizer): Method to be accelerated;
        for p=3 - BasicTensorMethod
        for p=2 - CubicRegularizedNewton
        for p=1 - GradientDescent
        (default: None)
        tensor_step_kwargs (dict): kwargs for TensorStepMethod (default: None)
        subsolver (Optimizer): Method to solve the inner problem (default: None)
        subsolver_args (dict): Arguments for the subsolver (default: None)
        max_subsolver_iterations (int): Maximal number of iterations of the subsolver to solve the inner problem (default: None)
    """
    MONOTONE = False
    ACCELERATION = True

    def __init__(self, params, L: float = 1., order: int = 2, a_step: float = None,
                 TensorStepMethod: Optimizer = None, tensor_step_kwargs: dict = None,
                 subsolver: Optimizer = None, subsolver_args: dict = None,
                 max_subsolver_iterations: int = None, verbose: bool = True, testing: bool = False):
        if L <= 0:
            raise ValueError(f"Invalid Lipschitz constant: L = {L}")

        super().__init__(params, dict(L=L))

        self.verbose = verbose
        self.testing = testing
        if len(self.param_groups) != 1:
            raise ValueError("Method doesn't support per-parameter options "
                             "(parameter groups)")
        group = self.param_groups[0]
        params = group['params']

        self.iteration = 0

        self.order = order
        self.L = L

        if a_step is None:
            if order == 2:
                a_step = 1 / 24
            elif order == 3:
                a_step = 5 / 3024
            else:
                a_step = ((2 * order - 1) * factorial(order - 1))  / ((order + 1) * (2 * order + 1) *  (2 * order) ** order)
        self.a_step = a_step

        # Step initialization: if order = 3 then Basic Tensor step,
        # if order = 2 then Cubic Newton, if order = 1 then Gradient Descent
        self.tensor_step_method = step_definer(params=params, L=L, order=order,
                                               TensorStepMethod=TensorStepMethod, tensor_step_kwargs=tensor_step_kwargs,
                                               subsolver=subsolver, subsolver_args=subsolver_args,
                                               max_subsolver_iterations=max_subsolver_iterations, verbose=verbose, testing=testing)

        # Initialization of intermediate points
        for p in params:
            state = self.state[p]
            state['x0'] = p.detach().clone()
            state['x'] = state['x0'].clone()
            state['v'] = state['x0'].clone()
            state['grads_sum'] = torch.zeros_like(p)

    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable): a closure that reevaluates the model and returns the loss.
        """
        closure = torch.enable_grad()(closure)

        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        params = group['params']

        alpha = (1. - 1. / (self.iteration + 1)) ** (self.order + 1)

        with torch.no_grad():
            for p in params:
                state = self.state[p]
                p.mul_(alpha).add_(state['v'], alpha=1 - alpha)

        self.tensor_step_method.step(closure)
        self.zero_grad()

        closure().backward()

        with torch.no_grad():
            a = self.a_step * ((self.iteration + 1) ** (self.order + 1) - self.iteration ** (self.order + 1)) / self.L
            for p in params:
                state = self.state[p]
                state['x'].zero_().add_(p)
                state['grads_sum'].add_(p.grad, alpha=a)

            if self.order == 1:
                scaling = 1.
            else:
                norm_squared = 0.
                for p in params:
                    state = self.state[p]
                    norm_squared += state['grads_sum'].square().sum()

                power = (1. - self.order) / (2. * self.order)
                scaling = norm_squared ** power

            for p in params:
                state = self.state[p]
                state['v'].zero_().add_(state['x0']).sub_(state['grads_sum'], alpha=scaling)

            self.iteration += 1

        return None
