import torch
import warnings
from torch.optim.optimizer import Optimizer
from .basic_tensor_method import BasicTensorMethod
from OPTAMI.utils import tuple_to_vec


class Superfast(Optimizer):
    """Implements Superfast Second-Order Method.

    It had been proposed in `Superfast second-order methods for unconstrained convex optimization`
    https://dial.uclouvain.be/pr/boreal/object/boreal%3A227146/
    Contributors:
        Dmitry Kamzolov
        T. Golubeva
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        L (float): estimated value of Lipschitz constant for the hessian (default: 1e+3)
        divider (float): constant controlling the step size; lower constant is a bigger step (default: 604.8, by theory)
        subsolver (Optimizer): method to solve the inner problem (default: BDGM)
    """
    MONOTONE = True

    def __init__(self, params, L: float = 1e+3, divider: float = 604.8, 
                 TensorStepMethod: Optimizer = None,
                 subsolver: Optimizer = None, subsolver_args: dict = None,
                 max_iters: int = None, verbose: bool = True, tensor_step_kwargs: dict = None):
        if L <= 0:
            raise ValueError(f"Invalid learning rate: L = {L}")

        super().__init__(params, dict(L=L, divider=divider))

        self.tensor_step_method = None
        self.TensorStepMethod = TensorStepMethod
        self.subsolver = subsolver
        self.subsolver_args = subsolver_args
        self.max_iters = max_iters
        self.tensor_step_kwargs = tensor_step_kwargs

        self.verbose = verbose

    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): a closure that reevaluates the model and returns the loss.
        """
        closure = torch.enable_grad()(closure)

        for group in self.param_groups:
            params = group['params']
            p = next(iter(params))
            state_common = self.state[p]

            if 'k' not in state_common:
                state_common['k'] = 0

            L = group['L']
            k = state_common['k']
            divider = group['divider']

            if self.tensor_step_method is None:
                if self.TensorStepMethod is None:
                    self.tensor_step_method = BasicTensorMethod(
                        params, L=L, subsolver=self.subsolver, verbose=self.verbose, 
                        subsolver_args=self.subsolver_args, max_iters=self.max_iters)
                else:
                    if not hasattr(self.TensorStepMethod, 'MONOTONE') or not self.TensorStepMethod.MONOTONE:
                        warnings.warn("`TensorStepMethod` should be monotone!")
                    self.tensor_step_method = self.TensorStepMethod(params, **self.tensor_step_kwargs)

            alpha = (1 - 1./(k+1))**4

            for p in group['params']:
                state = self.state[p]

                if ('v' not in state) or ('x' not in state):
                    state['v'] = p.clone()
                    state['x'] = p.clone()
                    state['x0'] = p.detach().clone()
                    state['df_sum'] = torch.zeros_like(p)

                with torch.no_grad():
                    v = state['v']
                    p.mul_(alpha).add_(v, alpha=1-alpha)

            self.tensor_step_method.step(closure)
            self.zero_grad()

            for p in group['params']:
                state = self.state[p]
                with torch.no_grad():
                    state['x'].zero_().add_(p)

            closure(backward=True)
            a = (2*k+1.) * (2*k*(k+1)+1) / (divider*L)

            for p in group['params']:
                state = self.state[p]
                with torch.no_grad():
                    state['df_sum'].add_(p.grad, alpha=a)

            norm = 0
            for p in group['params']:
                state = self.state[p]
                with torch.no_grad():
                    norm += tuple_to_vec.tuple_norm_square(state['df_sum'])
            norm = norm.pow(1./3)

            for p in group['params']:
                state = self.state[p]
                with torch.no_grad():
                    x0 = state['x0']
                    state['v'].zero_().add_(x0).sub_(state['df_sum'], alpha=1/norm)

            state_common['k'] += 1

        return None
