from torch.optim.optimizer import Optimizer
import torch
from ._supplemetrary import step_definer

class ProxPointSegmentSearch(Optimizer):
    """Implements Near-Optimal Proximal-Point Acceleration Method with Segment Search.

     Method was proposed in "Inexact High-Order Proximal-Point Methods with Auxiliary Search Procedure" by Yu.Nesterov.
     2021. SIAM Journal on Optimization, 31(4), pp.2807-2828.
     https://doi.org/10.1137/20M134705X

    Contributors:
        Dmitry Kamzolov
        Dmitry Vilensky-Pasechnyuk

    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups
        L (float): Estimated value of Lipschitz constant for the hessian (default: 1.0)
        approx (float): Approximation parameter connected with precision of inner subsolver (default: None);
        order (int): Order of the method (order = 1,2,3)TensorStepMethod (Optimizer): method to be accelerated;
        for p=3 - BasicTensorMethod
        for p=2 - CubicRegularizedNewton
        for p=1 - GradientDescent
        (default: None)
        tensor_step_kwargs (dict): kwargs for TensorStepMethod (default: None)
        subsolver (Optimizer): Method to solve the inner problem (default: None)
        subsolver_args (dict): Arguments for the subsolver (default: None)
        max_subsolver_iterations (int): Maximal number of the inner iterations of the subsolver to solve the inner problem (default: None)
        max_iterations_ls (int): Maximal number of the line-search iterations (default: 20)

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        L (float): estimated value of Lipschitz constant for the hessian (default: 1e+3)
    """
    MONOTONE = False
    ACCELERATION = True

    def __init__(self, params, L: float = 1., approx: float = None, order: int = 2,
                 TensorStepMethod: Optimizer = None, tensor_step_kwargs: dict = None,
                 subsolver: Optimizer = None, subsolver_args: dict = None,
                 max_subsolver_iterations: int = None, max_iterations_ls: int = 20,
                 verbose: bool = True, testing: bool = False):
        if L <= 0:
            raise ValueError(f"Invalid learning rate: L = {L}")

        super().__init__(params, dict(L=L))
        if len(self.param_groups) != 1:
            raise ValueError("Method doesn't support per-parameter options "
                             "(parameter groups)")

        self.order = order
        if approx is None or approx > 1:
            if self.order == 2:
                self.approx = 1.
            elif self.order == 3:
                self.approx = 5/36
        self.A = 0.
        self.verbose = verbose
        self.testing = testing

        self.iteration = 0
        self.average_iterations = 0.
        self.total_iterations = [0]
        self.max_iterations_ls = max_iterations_ls

        self.tensor_step_method = step_definer(params=params, L=L, order=order,
                                               TensorStepMethod=TensorStepMethod, tensor_step_kwargs=tensor_step_kwargs,
                                               subsolver=subsolver, subsolver_args=subsolver_args,
                                               max_subsolver_iterations=max_subsolver_iterations, verbose=verbose, testing=testing)

        for p in params:
            state = self.state[p]
            state['x'] = p.detach().clone()
            state['v'] = state['x'].clone()

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

        inner_iteration = 0

        with torch.no_grad():
            H_scaled = (self.approx / L) ** (1 / self.order)
            for p in params:
                state = self.state[p]
                state['u'] = state['v'] - state['x']
        
        self.tensor_step_method.step(closure)
        self.zero_grad()
        inner_iteration += 1

        closure().backward()
        with torch.no_grad():
            scal_1 = 0.
            df_norm_1 = 0.
            for p in params:
                state = self.state[p]
                state['T_1'] = p.detach().clone()
                state['df_det_1'] = p.grad.detach().clone()
                scal_1 += state['df_det_1'].mul(state['u']).sum().item()
                df_norm_1 += state['df_det_1'].square().sum().item()
            df_norm_1 = df_norm_1 ** 0.5

        if scal_1 >= 0:
            with torch.no_grad():
                g_norm = df_norm_1 + 0.
                for p in params:
                    state = self.state[p]
                    state['phi'] = state['df_det_1'].clone()
                    state['x'] = state['T_1'].clone()
        else:
            with torch.no_grad():
                for p in params:
                    state = self.state[p]
                    p.zero_().add_(state['v'])
            
            self.tensor_step_method.step(closure)
            self.zero_grad()
            inner_iteration += 1

            closure().backward()

            with torch.no_grad():
                scal_2 = 0.
                df_norm_2 = 0.
                for p in params:
                    state = self.state[p]
                    state['T_2'] = p.detach().clone()
                    state['df_det_2'] = p.grad.detach().clone()
                    scal_2 += state['df_det_2'].mul(state['u']).sum().item()
                    df_norm_2 += state['df_det_2'].square().sum().item()
                df_norm_2 = df_norm_2 ** 0.5
            if scal_2 <= 0:
                with torch.no_grad():
                    g_norm = df_norm_2 + 0.
                    for p in params:
                        state = self.state[p]
                        state['phi'] = state['df_det_2'].clone()
                        state['x'] = state['T_2'].clone()
            else:
                with torch.no_grad():
                    tau_1 = 0.
                    tau_2 = 1.

                    alpha = scal_2 / (scal_2 - scal_1)
                    g_pow = alpha * df_norm_1 ** ((self.order + 1) / self.order) + (1 - alpha) * df_norm_2 ** ((self.order + 1) / self.order)

                while scal_1 * 2 * (tau_1 - tau_2) * alpha > g_pow * H_scaled and inner_iteration < self.max_iterations_ls:
                    with torch.no_grad():
                        tau = (tau_1 + tau_2) / 2

                        for p in params:
                            state = self.state[p]
                            p.zero_().add_(state['x']).add_(state['u'], alpha=tau)

                    self.tensor_step_method.step(closure)
                    self.zero_grad()
                    inner_iteration += 1

                    closure().backward()

                    with torch.no_grad():
                        scal = 0.
                        for p in params:
                            state = self.state[p]
                            scal += p.grad.mul(state['u']).sum().item()

                        if scal > 0:
                            tau_2 = tau + 0.
                            scal_2 = scal + 0.

                            for p in params:
                                state = self.state[p]
                                state['T_2'] = p.detach().clone()
                                state['df_det_2'] = p.grad.detach().clone()
                        else:
                            tau_1 = tau + 0.
                            scal_1 = scal + 0.

                            for p in params:
                                state = self.state[p]
                                state['T_1'] = p.detach().clone()
                                state['df_det_1'] = p.grad.detach().clone()
                    alpha = scal_2 / (scal_2 - scal_1)
                    g_pow = alpha * df_norm_1 ** ((self.order + 1) / self.order) + (1 - alpha) * df_norm_2 ** (
                                (self.order + 1) / self.order)

                with torch.no_grad():
                    g_norm = g_pow ** (self.order / (self.order + 1))
                    for p in params:
                        state = self.state[p]
                        state['phi'].zero_().add_(state['df_det_1'], alpha=alpha).add_(state['df_det_2'], alpha=1-alpha)
                        state['x'].zero_().add_(state['T_1'], alpha=alpha).add_(state['T_2'], alpha=1-alpha)

        with torch.no_grad():
            c = 0.5 * H_scaled / (g_norm ** ((self.order - 1) / self.order))
            inner = c ** 2 + 4. * c  * self.A
            a = (inner ** 0.5 + c) / 2.

            for p in params:
                state = self.state[p]
                state['v'].sub_(state['phi'], alpha=a)
                p.zero_().add_(state['x'])

        self.A += a
        self.average_iterations = (self.average_iterations * self.iteration + inner_iteration) / (self.iteration + 1)
        self.total_iterations.append(self.total_iterations[-1] + inner_iteration)
        self.iteration += 1
        return None
