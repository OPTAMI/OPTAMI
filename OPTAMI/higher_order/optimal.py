from torch.optim.optimizer import Optimizer
from OPTAMI.utils import tuple_to_vec
import warnings
import OPTAMI
import torch
import math


class Optimal(Optimizer):
    """Implements Optimal Tensor Method.
     It was proposed by D.Kovalev in "The First Optimal Acceleration of High-Order Methods in Smooth Convex Optimization"
     https://arxiv.org/pdf/2205.09647.pdf
    Contributors:
        Dmitry Vilensky-Pasechnyuk
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        eta0 (float): initial step size
        L (float): Lipschitz constant of p-th derivative
        sigma (float): in (0, 1), factor for stopping condition
        order (int): order of the method
    """
    MONOTONE = False

    def __init__(self, params, eta0: float = 1e-3, L: float = 1e+3, sigma: float = 0.5, order: int = 2,
                 TensorStepMethod: Optimizer = None, tensor_step_kwargs: dict = None,
                 subsolver: Optimizer = None, subsolver_args: dict = None,
                 max_iters: int = None, verbose: bool = True):
        if L <= 0:
            raise ValueError(f"Invalid learning rate: L = {L}")

        super().__init__(params, dict(L=L, eta0=eta0, sigma=sigma))
        if len(self.param_groups) != 1:
            raise ValueError("Optimal Tensor Method doesn't support per-parameter options "
                             "(parameter groups)")

        self.order = order
        self.TensorStepMethod = TensorStepMethod
        self.subsolver = subsolver
        self.subsolver_args = subsolver_args
        self.max_iters = max_iters
        self.tensor_step_kwargs = tensor_step_kwargs
        self.tensor_step_method = None

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
        eta0 = group['eta0']
        sigma = group['sigma']
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
            state_common['beta'] = 0

        k = state_common['k']
        eta = eta0 * (1 + k) ** (3*self.order/2 - 0.5)
        beta = state_common['beta'] + eta
        lambd = eta**2/beta
        alpha = eta/beta

        for p in params:
            state = self.state[p]

            if 'x' not in state:
                state['x'] = p.detach().clone()
                state['xg'] = p.detach().clone()

            with torch.no_grad():
                state['xg'].mul_(alpha).add_(p, alpha=1 - alpha)
                state['xt'] = state['xg'].clone()

        stop = False
        while not stop:
            for p in params:
                state = self.state[p]
                with torch.no_grad():
                    p.zero_().add_(state['xt'])

            def regularized_closure():
                f = closure()
                norm = 0
                for p in params:
                    state = self.state[p]
                    norm += torch.linalg.norm(p.sub(state['xg'].detach())).item()**2
                f += 1/(2*lambd) * norm
                return f

            self.tensor_step_method.step(regularized_closure)
            self.zero_grad()

            for p in params:
                state = self.state[p]
                state['xt+1/2'] = p.detach().clone()

            with torch.enable_grad():
                closure().backward()
                for p in group['params']:
                    state = self.state[p]
                    state['dA'] = p.grad.clone()
                    state['dA'].add_(state['xt+1/2'], alpha=lambd).sub_(state['xg'], alpha=lambd)

            step_norm = 0
            for p in params:
                state = self.state[p]
                with torch.no_grad():
                    step_norm += torch.linalg.norm(state['xt+1/2'].sub(state['xt'])).item()**2
            step_norm = step_norm**(0.5)

            step_size = 1/(L * step_norm**(self.order-1) / math.factorial(self.order - 1))
            for p in params:
                state = self.state[p]
                with torch.no_grad():
                    state['xt'].sub_(state['dA'], alpha=step_size)
            
            dA_norm = 0
            shift_norm = 0
            for p in params:
                state = self.state[p]
                with torch.no_grad():
                    dA_norm += torch.linalg.norm(state['dA']).item()**2
                    shift_norm += torch.linalg.norm(state['xt+1/2'].sub(state['xg'])).item()**2
            dA_norm = dA_norm**(0.5)
            shift_norm = shift_norm**(0.5)

            stop = dA_norm <= (sigma / lambd) * shift_norm

        for p in params:
            state = self.state[p]
            with torch.no_grad():
                p.zero_().add_(state['xt+1/2'])

        with torch.enable_grad():
            closure().backward()
            for p in group['params']:
                state = self.state[p]
                state['df'] = p.grad.clone()
        
        for p in params:
            state = self.state[p]
            with torch.no_grad():
                state['x'].sub_(state['df'], alpha=eta)

        state_common['k'] += 1
        state_common['beta'] = beta

        return None
