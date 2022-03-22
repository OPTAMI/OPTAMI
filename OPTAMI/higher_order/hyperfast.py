import math
import torch
from OPTAMI.utils import tuple_to_vec
from torch.optim.optimizer import Optimizer
from .bregman_distance_gradient_method import BDGM


class Hyperfast(Optimizer):
    """Implements Hyperfast Second-Order Method.

    It had been proposed in `Near-optimal hyperfast second-order method for convex optimization`
    https://link.springer.com/chapter/10.1007/978-3-030-58657-7_15
    Contributors:
        Dmitry Kamzolov
        T. Golubeva
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        L (float): estimated value of Lipschitz constant for the hessian (default: 1e+3)
        divider (float): constant controlling the step size; lower constant is a bigger step (default: 604.8, by theory)
        subsolver (Optimizer): method to solve the inner problem
    """
    MONOTONE = True

    def __init__(self, params, L: float = 1e+3, order: int = 3,
                 subsolver: Optimizer = None, subsolver_args: dict = None,
                 max_iters_ls: int = 50, max_iters: int = None, verbose: bool = True):
        if L <= 0:
            raise ValueError(f"Invalid learning rate: L = {L}")

        super().__init__(params, dict(
            L=L, order=order, fac=math.factorial(order-1), subsolver=subsolver, max_iters_ls=max_iters_ls,
            max_iters=max_iters, subsolver_args=subsolver_args))
        self.verbose = verbose


    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): a closure that reevaluates the model and returns the loss.
        """
        closure = torch.enable_grad()(closure)

        for group in self.param_groups:
            p = next(p for p in group['params'])
            state_common = self.state[p]

            if 'theta' not in state_common:
                state_common['theta'] = 1.
                state_common['A'] = 0.
            
            L = group['L']
            fac = group['fac']
            A = state_common['A']
            order = group['order']
            theta = state_common['theta']
            max_iters = group['max_iters']
            subsolver = group['subsolver']
            max_iters_ls = group['max_iters_ls']
            subsolver_args = group['subsolver_args']

            for p in group['params']:
                state = self.state[p]

                if ('x' not in state) or ('y' not in state):
                    state['x'] = p.detach().clone()
                    state['y'] = p.detach().clone()

            A_new = A
            s = order/(order+1)
            m = (s + 0.5) / 2

            l, u = 0., 1.
            for _ in range(max_iters_ls):
                A_new = A / theta
                a = A_new - A

                for p in group['params']:
                    state = self.state[p]
                    with torch.no_grad():
                        p.zero_().add_(state['y'], alpha=theta).add_(state['x'], alpha=1-theta)
                        state['x_wave'] = p.detach().clone()

                BDGM(
                    group['params'], L=L, subsolver=subsolver, verbose=self.verbose, 
                    subsolver_args=subsolver_args, max_iters=max_iters
                ).solve(closure)
                
                norm = 0.
                with torch.no_grad():
                    for p in group['params']:
                        state = self.state[p]
                        state['x_wave'].sub_(p)
                        norm += tuple_to_vec.tuple_norm_square(state['x_wave'])
                norm = norm.sqrt().pow(order-1)

                H = 1.5 * L
                inequality = ((1-theta)**2 * A * H / theta) * norm / fac

                if A == 0:
                    a = fac / (2 * H * norm)
                    A_new = A + a
                    theta = 1.
                    break
                elif 0.5 <= inequality <= s:
                    break
                elif inequality < m:
                    theta, u = (theta + l) / 2, theta
                else:
                    l, theta = theta, (u + theta) / 2
            else:
                if self.verbose:
                    print('line-search failed')

            with torch.no_grad():
                for p in group['params']:
                    state = self.state[p]
                    state['y'] = p.detach().clone()

            closure(backward=True)

            with torch.no_grad():
                for p in group['params']:
                    state = self.state[p]
                    state['x'].sub_(p.grad, alpha=a)

            state_common['A'] = A_new
            state_common['theta'] = theta
        return None