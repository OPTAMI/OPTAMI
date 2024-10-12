from math import factorial
import torch
from torch.optim.optimizer import Optimizer
from ._supplemetrary import step_definer


class NearOptimalAcceleration(Optimizer):
    """Implements Near-Optimal Accelerated Tensor Method.

    Exact version was proposed by Bubeck, S., Jiang, Q., Lee, Y.T., Li, Y. and Sidford, A., 2019.
    "Near-Optimal Method for Highly Smooth Convex Optimization." In Conference on Learning Theory (pp. 492-507). PMLR.
    https://proceedings.mlr.press/v99/bubeck19a.html
    and
    Gasnikov, A., Dvurechensky, P., Gorbunov, E., Vorontsova, E., Selikhanovych, D., Uribe, C.A.,
    Jiang, B., Wang, H., Zhang, S., Bubeck, S. and Jiang, Q., 2019.
    "Near-Optimal Methods for Minimizing Convex Functions with Lipschitz $p$-th Derivatives."
    In Conference on Learning Theory (pp. 1392-1393). PMLR.
    https://proceedings.mlr.press/v99/gasnikov19b.html

    Inexact version was proposed by Kamzolov D., 2020.
    "Near-Optimal Hyperfast Second-order Method for Convex Optimization."
    In International Conference on Mathematical Optimization Theory and Operations Research (pp. 167-178). Springer, Cham.
    https://doi.org/10.1007/978-3-030-58657-7_15

    Contributors:
        Dmitry Kamzolov
        Dmitry Vilensky-Pasechnyuk

    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups
        L (float): Estimated value of Lipschitz constant for the Hessian (default: 1.0)
        order (int): Order of the method (order = 1,2,3)
        TensorStepMethod (Optimizer): method to be accelerated;
        for p=3 - BasicTensorMethod
        for p=2 - CubicRegularizedNewton
        for p=1 - GradientDescent
        (default: None)
        tensor_step_kwargs (dict): kwargs for TensorStepMethod (default: None)
        subsolver (Optimizer): Method to solve the inner problem (default: None)
        subsolver_args (dict): Arguments for the subsolver (default: None)
        max_subsolver_iterations (int): Maximal number of the inner iterations of the subsolver to solve the inner problem (default: None)
        max_iterations_ls (int): Maximal number of the line-search iterations (default: 20)
    """

    MONOTONE = False
    ACCELERATION = True

    def __init__(self, params, L: float = 1., order: int = 3,
                 TensorStepMethod: Optimizer = None, tensor_step_kwargs: dict = None,
                 subsolver: Optimizer = None, subsolver_args: dict = None,
                 max_subsolver_iterations: int = None, max_iterations_ls: int = 20,
                 verbose: bool = True, testing: bool = False):
        if L <= 0:
            raise ValueError(f"Invalid Lipschitz constant: L = {L}")

        super().__init__(params, dict(L=L))

        self.verbose = verbose
        self.testing = testing
        if len(self.param_groups) != 1:
            raise ValueError("Method doesn't support per-parameter options "
                             "(parameter groups)")

        self.theta = 1.
        self.A = 0.
        self.iteration = 0
        self.average_iterations = 0.
        self.total_iterations = [0]

        self.order = order
        self.L = L
        self.max_iterations_ls = max_iterations_ls

        self.tensor_step_method = step_definer(params=params, L=L, order=order,
                                               TensorStepMethod=TensorStepMethod, tensor_step_kwargs=tensor_step_kwargs,
                                               subsolver=subsolver, subsolver_args=subsolver_args,
                                               max_subsolver_iterations=max_subsolver_iterations, verbose=verbose, testing=testing)

        for p in params:
            state = self.state[p]
            state['x'] = p.detach().clone()
            state['y'] = state['x'].clone()
            state['x_wave'] = state['x'].clone()


    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable): a closure that reevaluates the model and returns the loss.
        """
        closure = torch.enable_grad()(closure)

        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        params = group['params']


        factorial_ = factorial(self.order - 1)
        upper_bound = self.order / (self.order + 1)
        medium = (upper_bound + 0.5) / 2
        left, right = 0., 1.
        A_new = self.A + 0.
        a = 0.
        inner_iteration = 0
        stop = False
        while not stop and inner_iteration < self.max_iterations_ls:
            inner_iteration += 1
            A_new = self.A / self.theta
            a = A_new - self.A

            for p in params:
                state = self.state[p]
                with torch.no_grad():
                    state['x_wave'] = state['y'].mul(self.theta).add(state['x'], alpha=1-self.theta)
                    p.zero_().add_(state['x_wave'])

            self.tensor_step_method.step(closure)
            self.zero_grad()


            with torch.no_grad():
                norm_squared = 0.
                for p in params:
                    state = self.state[p]
                    state['x_wave'].sub_(p)
                    norm_squared += state['x_wave'].square().sum().item()
                norm = norm_squared ** ((self.order - 1) / 2.)

            H = 1.5 * self.L
            inequality = ((1 - self.theta) ** 2 * self.A * H / self.theta) * norm / factorial_

            if self.A == 0:
                a = factorial_ / (2 * H * norm)
                A_new = self.A + a
                stop = True
            elif 0.5 <= inequality <= upper_bound:
                stop = True
            elif inequality < medium:
                self.theta, right = (self.theta + left) / 2, self.theta
            else:
                left, self.theta = self.theta, (right + self.theta) / 2


        with torch.no_grad():
            for p in params:
                state = self.state[p]
                state['y'] = p.detach().clone()

        closure().backward()

        with torch.no_grad():
            for p in params:
                state = self.state[p]
                state['x'].sub_(p.grad, alpha=a)

            self.average_iterations = (self.average_iterations * self.iteration + inner_iteration) / (self.iteration + 1)
            self.total_iterations.append(self.total_iterations[-1] + inner_iteration)
            self.iteration += 1
            self.A = A_new + 0.
        return None
