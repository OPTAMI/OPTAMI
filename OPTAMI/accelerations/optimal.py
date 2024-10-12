from torch.optim.optimizer import Optimizer
import torch
from math import factorial
from ._supplemetrary import step_definer


class Optimal(Optimizer):
    """Implements Optimal Tensor Method.

     It was proposed in "The First Optimal Acceleration of High-Order Methods in Smooth Convex Optimization"
     by Dmitry Kovalev  and Alexander Gasnikov. 2022. Advances in Neural Information Processing Systems, 35, pp.35339-35351.
     https://proceedings.neurips.cc/paper_files/paper/2022/hash/e56f394bbd4f0ec81393d767caa5a31b-Abstract-Conference.html

    Contributors:
        Dmitry Vilensky-Pasechnyuk
        Dmitry Kamzolov

    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups
        L (float): Lipschitz constant of p-th derivative
        eta0 (float): Initial step size (default: 0.0)
        sigma (float): in (0, 1), factor for stopping condition (default: 0.5)
        order (int): Order of the method (default: 2)
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

    def __init__(self, params, L: float = 1., eta0: float = 0., sigma: float = 0.5, order: int = 2,
                 TensorStepMethod: Optimizer = None, tensor_step_kwargs: dict = None,
                 subsolver: Optimizer = None, subsolver_args: dict = None,
                 max_subsolver_iterations: int = None, max_iters_ls: int = 20, verbose: bool = False, testing: bool = False):
        if L <= 0:
            raise ValueError(f"Invalid learning rate: L = {L}")

        super().__init__(params, dict(L=L))

        self.verbose = verbose
        self.testing = testing
        if len(self.param_groups) != 1:
            raise ValueError("Optimal Tensor Method doesn't support per-parameter options "
                             "(parameter groups)")

        group = self.param_groups[0]
        params = group['params']

        self.iteration = 0
        self.average_iterations = 0.
        self.total_iterations = [0]
        self.beta = 0
        self.order = order
        self.L = L
        self.sigma = sigma
        self.max_iters_ls = max_iters_ls

        if eta0 <= 0.:

            if self.order == 1:
                #Cm = L * (1 + 1 / sigma) *  (q + 1) #** (q / 2 - 1)
                #self.eta0 = 2  / (Cm * 4 * (1 + sigma) / (1 - sigma))
                self.eta0 = 4.
            else:
                q = self.order
                Cm = L * q ** q * (1 + 1 / sigma) / (factorial(q) * (q - 1) ** (q / 2) * (q + 1) ** (q / 2 - 1))
                self.eta0 = 2 ** q * q ** (1 / 2) / (Cm * (3 * q + 1) ** q * ((1 + sigma) / (1 - sigma)) ** ((q - 1) / 2))
        else:
            self.eta0 = eta0
        # Step initialization: if order = 3 then Basic Tensor step,
        # if order = 2 then Cubic Newton, if order = 1 then Gradient Descent
        self.tensor_step_method = step_definer(params=params, L=L, order=order,
                                               TensorStepMethod=TensorStepMethod, tensor_step_kwargs=tensor_step_kwargs,
                                               subsolver=subsolver, subsolver_args=subsolver_args,
                                               max_subsolver_iterations=max_subsolver_iterations, verbose=verbose, testing=testing)
        with torch.no_grad():
            for p in params:
                state = self.state[p]
                state['x'] = p.detach().clone()
                state['xg'] = state['x'].clone()
                state['xt'] = state['x'].clone()

    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): a closure that reevaluates the model and returns the loss.
        """
        closure = torch.enable_grad()(closure)

        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        params = group['params']


        sigma = self.sigma

        L = self.L

        # Steps 5-6
        eta = self.eta0 * (1 + self.iteration) ** ((3 * self.order - 1) / 2)
        self. beta += eta
        alpha = eta / self.beta #eta_0 - free
        lambd = eta * alpha #linear dependence on eta_0
        # Steps 7-8
        with torch.no_grad():
            for p in params:
                state = self.state[p]
                state['xg'].mul_(alpha).add_(p.detach(), alpha=1 - alpha)
                state['xt'] = state['xg'].clone()
        inner_iteration = 0
        stop = False
        while not stop and inner_iteration < self.max_iters_ls:
            for p in params:
                state = self.state[p]
                with torch.no_grad():
                    p.zero_().add_(state['xt'])

            def regularized_closure():
                self.tensor_step_method.zero_grad()
                norm = 0.
                for p in params:
                    state = self.state[p]
                    norm += (p.sub(state['xg'])).square().sum()
                return closure() + norm / (2 * lambd)

            inner_iteration += 1
            self.tensor_step_method.step(regularized_closure)
            self.zero_grad()

            for p in params:
                state = self.state[p]
                state['xt+1/2'] = p.detach().clone()


            closure().backward()

            dA_norm = 0.
            shift_norm = 0.
            step_norm = 0.
            with torch.no_grad():
                for p in group['params']:
                    state = self.state[p]
                    state['dA'] = p.grad.clone()
                    temp = state['xt+1/2'].sub(state['xg'])
                    state['dA'].add_(temp.div(lambd))
                    dA_norm += state['dA'].square().sum()
                    shift_norm += temp.square().sum()
                    step_norm += (state['xt+1/2'] - state['xt']).square().sum()
                dA_norm = dA_norm ** 0.5
                shift_norm = shift_norm ** 0.5
                step_norm = step_norm ** 0.5
                step_size = factorial(self.order - 1) / (L * step_norm ** (self.order - 1))
                for p in params:
                    state = self.state[p]
                    state['xt'].sub_(state['dA'], alpha=step_size)

            stop = dA_norm <= (sigma / lambd) * shift_norm
        for p in params:
            state = self.state[p]
            with torch.no_grad():
                p.zero_().add_(state['xt+1/2'])


        closure().backward()

        with torch.no_grad():
            for p in params:
                state = self.state[p]
                state['x'].sub_(p.grad, alpha=eta)

        self.average_iterations = (self.average_iterations * self.iteration + inner_iteration) / (self.iteration + 1)
        self.total_iterations.append(self.total_iterations[-1] + inner_iteration)
        self.iteration += 1
        if self.verbose and self.iteration % 10 == 0:
            print("Iteration", self.iteration, "average iterations", self.average_iterations)
        return None