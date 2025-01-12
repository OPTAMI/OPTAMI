import torch
from torch.optim.optimizer import Optimizer


class UniversalSGD(Optimizer):
    """Implements Universal Stochastic Gradient Method
    Universal Stochastic Gradient Method (Algorithm 4.1) was proposed by Anton Rodomanov et.al
    https://arxiv.org/pdf/2402.03210
    Contributors:
        Dmitry Kamzolov
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        D (float): size of the ball that includes starting point and solution.
        passive_start (bool): if True, then the starting point is not included in the ball.
    """
    MONOTONE = False
    SKIP_TEST_LOGREG = True

    def __init__(self, params, D: float = 1., passive_start: bool = False, verbose: bool = True, testing: bool = False):
        if D <= 0:
            raise ValueError(f"Invalid set size: D = {D}")

        super().__init__(params, dict(D=D))
        self.D = D
        self.Hk = 0.
        self.passive_start = passive_start
        self.verbose = verbose
        self.testing = testing
        self.iter_count = 0
        if len(self.param_groups) != 1:
            raise ValueError("Method doesn't support per-parameter options "
                             "(parameter groups)")
        group = self.param_groups[0]
        params = group['params']

        # Initialization of intermediate points
        for p in params:
            state = self.state[p]
            state['x0'] = p.detach().clone()
            state['x'] = state['x0'].clone()
            state['grad'] = torch.zeros_like(state['x0'])
            state['x_av'] = state['x0'].clone()

    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): a closure that reevaluates the model and returns the loss
        """
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        params = group['params']

        if self.iter_count == 0:
            # First step is a projection step as H_k = 0.
            closure().backward()
            with torch.no_grad():
                grad_sq = 0.
                for p in params:
                    state = self.state[p]
                    state['grad'].copy_(p.grad)
                    grad_sq += p.grad.square().sum().item()
                # Step-size h counted to be a projection for first step
                h = self.D / grad_sq ** 0.5

                # For passive start, we don't move after first iteration but increase Hk
                if not self.passive_start:
                    self.iter_count = 1
        else:
            # Regular step-size
            h = 1. / self.Hk


        with torch.no_grad():
            # Making a projected gradient step
            # Calculating the distance of potential step
            potential_step_distance_sq = 0.
            for p in params:
                state = self.state[p]
                potential_step_distance_sq += (state['x'] - h * state['grad'] - state['x0']).square().sum().item()
            potential_step_distance = potential_step_distance_sq ** 0.5

            # Making a projection step and calculating the distance of final step
            final_step_distance_sq = 0.
            if potential_step_distance > self.D:
                # Step with projection
                for p in params:
                    state = self.state[p]
                    dif = p - state['x0']
                    p.copy_(state['x0']).add_(self.D * dif / potential_step_distance)
                    final_step_distance_sq += dif.square().sum().item()
            else:
                # Regular step without projection
                for p in params:
                    state = self.state[p]
                    dif = h * state['grad']
                    p.copy_(state['x'] - dif)
                    final_step_distance_sq += dif.square().sum().item()

        # Calculating new step-size H_k
        closure().backward()
        with torch.no_grad():
            bregman_distance = 0.
            for p in params:
                state = self.state[p]
                bregman_distance += (p - state['x']).mul(p.grad - state['grad']).sum().item()
            self.Hk += max((bregman_distance - final_step_distance_sq * self.Hk / 2) / (self.D ** 2 + final_step_distance_sq / 2), 0.)

            # Saving the current point and current gradient for all cases except first step of passive_start
            if self.iter_count > 0:
                for p in params:
                    state = self.state[p]
                    state['grad'].copy_(p.grad)
                    state['x'].copy_(p)

            # Convergence is for average iteration, so the exit params is average point
            for p in params:
                state = self.state[p]
                state['x_av'].mul_(self.iter_count).add_(state['x']).div_(self.iter_count + 1.)
                p.copy_(state['x_av'])
            self.iter_count += 1
        return None