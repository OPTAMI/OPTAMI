import torch
from math import factorial
from torch.optim.optimizer import Optimizer
from ._supplemetrary import step_definer


class NATA(Optimizer):
    """Implements Nesterov Accelerated Tensor Method with A-Adaptation (NATA).

    It was proposed by D. Kamzolov, D. Pasechnyuk, A. Agafonov,A. Gasnikov, and Martin Takác
    in "OPTAMI: Global Superlinear Convergence of High-order Methods"
    https://arxiv.org/abs/2410.04083

    Contributors:
        Dmitry Kamzolov

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        L (float): estimated value of Lipschitz constant for the hessian (default: 1.)
        order (int): order of the method (default: 2)
        nu0 (float): constant controlling the step size; bigger constant is bigger step;
        theoretical nu0=1/604 for p=3, nu0=1/24 for p=2 (default: None)
        nu_adaptation (bool): Flag to activate adaptation of nu (default: True)
        aggressive_nu_adaptation (bool): Flag allowing increasing of nu during adaptation (default: True)
        nu_max_limiter (bool): Flag allowing limiting of nu during adaptation (default: True)
        TensorStepMethod (Optimizer): method to be accelerated;
        for p=3 - BasicTensorMethod
        for p=2 - CubicRegularizedNewton
        for p=1 - GradientDescent
        (default: None)
        tensor_step_kwargs (dict): kwargs for TensorStepMethod (default: None)
        subsolver (Optimizer): method to solve the inner problem (default: None)
        subsolver_args (dict): arguments for the subsolver (default: None)
        max_subsolver_iterations (int): Maximal number of iterations of the subsolver to solve the inner problem (default: None)
        max_iterations_ls (int): Maximal number of the line-search iterations (default: 20)
    """

    MONOTONE = False
    ACCELERATION = True

    def __init__(self, params, L: float = 1., order: int = 2, nu0: float = 10., nu_adaptation: bool = True,
        theta: float = 2.,
                 aggressive_nu_adaptation: bool = True, nu_max_limiter: bool = True, nu_max: float = 10000.,
                 TensorStepMethod: Optimizer = None, tensor_step_kwargs: dict = None,
                 subsolver: Optimizer = None, subsolver_args: dict = None,
                 max_subsolver_iterations: int = None, max_iterations_ls: int = 20,
                 verbose: bool = False, testing: bool = False):
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
        self.A = 0.
        if order == 2:
            a_step = 1 / 24
        elif order == 3:
            a_step = 5 / 3024
        else:
            a_step = ((2 * order - 1) * factorial(order - 1)) / (
                    (order + 1) * (2 * order + 1) * (2 * order) ** order)
        if nu0 is None:
            nu0 = a_step + 0.

        self.nu_t = nu0
        self.theta = theta
        self.nu_min = a_step + 0.
        self.nu_max_limiter = nu_max_limiter
        self.nu_adaptation = nu_adaptation
        self.nu_max = nu_max
        if self.nu_adaptation and self.nu_max_limiter:
            self.nu_t = min(self.nu_t, self.nu_max)
        self.psi_agr = 0.
        self.aggressive_nu_adaptation = aggressive_nu_adaptation
        self.max_iterations_ls = max_iterations_ls
        self.average_iterations = 0.
        self.total_iterations = [0]

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
            state['v_new'] = state['x0'].clone()
            state['y'] = state['x0'].clone()
            state['grads_sum'] = torch.zeros_like(p)
            state['grads_sum_new'] = torch.zeros_like(p)

    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): a closure that reevaluates the model and returns the loss.
        """
        closure = torch.enable_grad()(closure)

        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        params = group['params']


        inner_iteration = 0
        flag = True
        if self.nu_adaptation:
            self.nu_t *= self.theta

        while flag and inner_iteration < self.max_iterations_ls:
            inner_iteration += 1
            if self.nu_adaptation:
                self.nu_t = max(self.nu_t/self.theta, self.nu_min)
            else:
                flag = False
            a_t = self.nu_t / self.L * ((self.iteration + 1) ** (self.order + 1) - self.iteration ** (self.order + 1))
            A_new = self.A + a_t
            alpha = self.A / A_new

            with torch.no_grad():
                for p in params:
                    state = self.state[p]
                    state['y'] = state['x'].mul(alpha).add(state['v'], alpha=1 - alpha)
                    p.zero_().add_(state['y'])
            self.tensor_step_method.step(closure)
            self.zero_grad()

            f_xk1 = closure()
            f_xk1.backward()

            with torch.no_grad():

                for p in params:
                    state = self.state[p]
                    state['grads_sum_new'] = state['grads_sum'].add(p.grad, alpha=a_t)

                if self.order == 1:
                    scaling = 1.
                else:
                    norm_squared = 0.
                    for p in params:
                        state = self.state[p]
                        norm_squared += state['grads_sum_new'].square().sum()

                    power = (1. - self.order) / (2. * self.order)
                    scaling = norm_squared ** power
                for p in params:
                    state = self.state[p]
                    state['v_new'] = state['x0'].sub(state['grads_sum_new'], alpha=scaling)

                v_distance_sq = 0.
                for p in params:
                    state = self.state[p]
                    v_distance_sq += (state['v_new'] - state['x0']).square().sum()
                v_distance = v_distance_sq ** (1 / 2)

                if self.testing:
                    grad_v_norm = 0.
                    for p in params:
                        state = self.state[p]
                        grad_v_norm += (state['grads_sum_new'] + v_distance ** (self.order-1) * (state['v_new'] - state['x0'])).square().sum()
                    assert grad_v_norm < 1e-8


                if self.nu_adaptation:
                    grad_mul_xk = 0.
                    grads_sum_mul = 0.

                    for p in params:
                        state = self.state[p]
                        grads_sum_mul += state['v_new'].mul(state['grads_sum_new']).sum()
                        grad_mul_xk += p.mul(p.grad).sum()
                    psi_agr_new = (f_xk1.item() - grad_mul_xk) * a_t
                    total = self.psi_agr + psi_agr_new + grads_sum_mul + v_distance ** (self.order + 1) / (self.order + 1)

                    if total >= A_new * f_xk1.item():
                        flag = False

        if self.nu_adaptation:
            self.psi_agr += psi_agr_new
        self.iteration += 1
        self.A = A_new + 0.
        if self.aggressive_nu_adaptation and self.nu_adaptation:
            self.nu_t *= self.theta
            if self.nu_max_limiter:
                self.nu_t = min(self.nu_t, self.nu_max)
        for p in params:
            state = self.state[p]
            state['x'].zero_().add_(p)
            state['v'].zero_().add_(state['v_new'])
            state['grads_sum'].zero_().add_(state['grads_sum_new'])
        self.average_iterations = (self.average_iterations * self.iteration + inner_iteration) / (
                self.iteration + 1)
        self.total_iterations.append(self.total_iterations[-1] + inner_iteration)

        return None