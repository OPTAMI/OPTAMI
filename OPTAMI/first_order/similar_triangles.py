import math
import torch
from torch.optim.optimizer import Optimizer


class SimilarTriangles(Optimizer):
    """Implements Method of Similar Triangles.
    It had been proposed in `Universal Method for Stochastic Composite Optimization Problems` 
    https://link.springer.com/content/pdf/10.1134/S0965542518010050.pdf
    Contributors:
        Dmitry Kamzolov
        Dmitry Vilensky-Pasechnyuk
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        is_adaptive (boolean): if False then method works with fixed L. If True then method tunes L adaptively (default: False)
        L (float): estimated value of Lipschitz smoothness constant; if `adaptive` is True, then L should be greater than theoretically tight (default: 1e+2)
        max_adapt_iters (int): maximal number of iterations to multiplicatively adapt L if `adaptive` is True (defaulr: 10)
        zeta (float): coefficient to multiply/divide L on with the aim of adaptation (default: 2.)
        verbose (bool): flag to control additional logs, here - on adaptation of L (default: True)
    """
    MONOTONE=False

    def __init__(self, params, L: float = 1e+2, is_adaptive: bool = True,
                 max_adapt_iters: int = 10, zeta: float = 2., verbose: bool = True):
        if L <= 0:
            raise ValueError(f"Invalid learning rate: L = {L}")

        super().__init__(params, dict(
            L=L, is_adaptive=is_adaptive,
            max_adapt_iters=max_adapt_iters, zeta=zeta))
        self.verbose = verbose

    @torch.no_grad()
    def _check_relaxation(self, closure, params, fy, L):
        fx = closure().item()
        gap = fx - fy
        for p in params:
            state = self.state[p]
            y = state['y']
            dfy = state['dfy']

            gap += dfy.mul(y.sub(p)).sum().item()
            gap -= L/2 * y.sub(p).norm().item() ** 2

        return gap

    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): a closure that reevaluates the model and returns the loss
        """
        closure = torch.enable_grad()(closure)

        for group in self.param_groups:
            p = next(iter(group['params']))
            state_common = self.state[p]

            if ('A' not in state_common) or ('L' not in state_common):
                state_common['A'] = 0.
                state_common['L'] = group['L']

            A = state_common['A']
            L = state_common['L']
            is_adaptive = group['is_adaptive']
            max_adapt_iters = group['max_adapt_iters']
            zeta = group['zeta']

            for p in group['params']:
                state = self.state[p]
                state['x'] = p.clone()

            for _ in range(max_adapt_iters):
                for p in group['params']:
                    state = self.state[p]

                    if 'u' not in state:
                        state['u'] = p.clone()

                    u = state['u']
                    a = math.sqrt(1./(4*L**2) + A/L) + 1./(2*L)
                    alpha = A / (A + a)
                    state['alpha'] = alpha

                    p.mul_(alpha).add_(u, alpha=1-alpha)
                    state['y'] = p.clone().detach()

                with torch.enable_grad():
                    fy = closure()
                    fy.backward()
                fy = fy.item()
                for p in group['params']:
                    state = self.state[p]
                    state['dfy'] = p.grad.clone()

                for p in group['params']:
                    state = self.state[p]
                    u = state['u']
                    x = state['x']
                    alpha = state['alpha']

                    u.sub_(p.grad, alpha=a)
                    p.zero_().add_(x,alpha = alpha).add_(u, alpha=1-alpha)

                    state['u'] = u

                if not is_adaptive:
                    break
                elif self._check_relaxation(closure, group['params'], fy, L) < 0:
                    if self.verbose:
                        print(f"/ {zeta}")
                    L /= zeta
                    break
                else:
                    if self.verbose:
                        print(f"* {zeta}")
                    L *= zeta
                    for p in group['params']:
                        state = self.state[p]
                        x = state['x']
                        p.zero_().add_(x)

            state_common['A'] = A + a
            state_common['L'] = L
        return None
