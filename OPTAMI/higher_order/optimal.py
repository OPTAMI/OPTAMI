from torch.optim.optimizer import Optimizer
import torch
import math
from OPTAMI.higher_order._supplemetrary import step_definer


class Optimal(Optimizer):
    """Implements Optimal Tensor Method.
     It was proposed by Dmitry Kovalev  and Alexander Gasnikov in "The First Optimal Acceleration of High-Order Methods in Smooth Convex Optimization"
     https://arxiv.org/pdf/2205.09647.pdf
    Contributors:
        Dmitry Vilensky-Pasechnyuk
        Dmitry Kamzolov
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        eta0 (float): initial step size
        L (float): Lipschitz constant of p-th derivative
        sigma (float): in (0, 1), factor for stopping condition
        order (int): order of the method
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

    def __init__(self, params, L: float = 1., eta0: float = 0., sigma: float = 0.5, order: int = 3,
                 TensorStepMethod: Optimizer = None, tensor_step_kwargs: dict = None,
                 subsolver: Optimizer = None, subsolver_args: dict = None,
                 max_iters: int = None, max_iters_ls: int = 20, verbose: bool = True, testing: bool = False):
        if L <= 0:
            raise ValueError(f"Invalid learning rate: L = {L}")

        super().__init__(params, dict(L=L, eta0=eta0, sigma=sigma))

        self.verbose = verbose
        self.testing = testing
        if len(self.param_groups) != 1:
            raise ValueError("Optimal Tensor Method doesn't support per-parameter options "
                             "(parameter groups)")

        group = self.param_groups[0]
        params = group['params']
        p = next(iter(params))
        state_common = self.state[p]
        state_common['k'] = 0
        state_common['beta'] = 0
        state_common['average_iterations'] = 0
        state_common['total_iterations'] = [0]

        self.order = order
        self.L = L
        self.eta0 = eta0
        self.sigma = sigma
        self.max_iters_ls = max_iters_ls

        # Step initialization: if order = 3 then Basic Tensor step,
        # if order = 2 then Cubic Newton, if order = 1 then Gradient Descent
        self.tensor_step_method = step_definer(params=params, L=L, order=order,
                                               TensorStepMethod=TensorStepMethod, tensor_step_kwargs=tensor_step_kwargs,
                                               subsolver=subsolver, subsolver_args=subsolver_args,
                                               max_iters=max_iters, verbose=verbose, testing=testing)
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
        p = next(iter(params))
        state_common = self.state[p]

        k = state_common['k']
        eta0 = self.eta0
        sigma = self.sigma

        L = self.L
        if eta0 <= 0.:
            q = self.order
            Cm = L * q ** q * (1 + 1 / sigma) / (math.factorial(q) * ((q - 1)) ** (q / 2) * ((q + 1)) ** (q / 2 - 1))
            eta0 = 2 ** q * q ** (1 / 2) / (Cm * (3 * q + 1) ** q * ((1 + sigma) / (1 - sigma)) ** ((q - 1) / 2))

        # Steps 5-6
        eta = eta0 * (1 + k) ** ((3 * self.order - 1) / 2)
        beta = state_common['beta'] + eta
        alpha = eta / beta #eta_0 - free
        lambd = eta * alpha #linear dependence on eta_0

        # Steps 7-8
        with torch.no_grad():
            for p in params:
                state = self.state[p]
                state['xg'].mul_(alpha).add_(p.detach(), alpha=1 - alpha)
                state['xt'] = state['xg'].clone()
        it = 0
        stop = False
        while not stop and it < self.max_iters_ls:
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

            it += 1
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
                step_size = math.factorial(self.order - 1) / (L * step_norm ** (self.order - 1))

                for p in params:
                    state = self.state[p]
                    state['xt'].sub_(state['dA'], alpha=step_size)



            stop = dA_norm <= (sigma / lambd) * shift_norm
        for p in params:
            state = self.state[p]
            with torch.no_grad():
                p.zero_().add_(state['xt+1/2'])

        with torch.enable_grad():
            closure().backward()

        with torch.no_grad():
            for p in params:
                state = self.state[p]
                state['x'].sub_(p.grad, alpha=eta)

        state_common['total_iterations'].append(state_common['total_iterations'][-1] + it)
        state_common['average_iterations'] = (state_common['average_iterations'] * state_common['k'] + it) / (
                    state_common['k'] + 1)
        state_common['k'] += 1
        state_common['beta'] = beta
        if self.verbose and state_common['k'] % 10 == 0:
            print("Iteration", state_common['k'], "average iterations", state_common['average_iterations'])
        return None