import math
import torch
from torch.optim.optimizer import Optimizer
from OPTAMI.higher_order._supplemetrary import step_definer


class Hyperfast(Optimizer):
    """Implements Inexact Near-optimal Accelerated Tensor Method.

    Exact version was proposed by Bubeck, S., Jiang, Q., Lee, Y.T., Li, Y. and Sidford, A., 2019, June.
    "Near-optimal method for highly smooth convex optimization." In Conference on Learning Theory (pp. 492-507). PMLR.
    https://proceedings.mlr.press/v99/bubeck19a.html
    and
    Gasnikov, A., Dvurechensky, P., Gorbunov, E., Vorontsova, E., Selikhanovych, D., Uribe, C.A.,
    Jiang, B., Wang, H., Zhang, S., Bubeck, S. and Jiang, Q., 2019, June.
    "Near optimal methods for minimizing convex functions with lipschitz $ p $-th derivatives."
    In Conference on Learning Theory (pp. 1392-1393). PMLR.
    https://proceedings.mlr.press/v99/gasnikov19b.html

    Inexact version was proposed by Kamzolov D., 2020, July.
    "Near-optimal hyperfast second-order method for convex optimization."
    In International Conference on Mathematical Optimization Theory and Operations Research (pp. 167-178). Springer, Cham.
    https://doi.org/10.1007/978-3-030-58657-7_15

    Contributors:
        Dmitry Kamzolov
        Dmitry Vilensky-Pasechnyuk
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        L (float): estimated value of Lipschitz constant for the hessian (default: 1e+3)
        order (int): order of the method (order = 1,2,3)TensorStepMethod (Optimizer): method to be accelerated;
        for p=3 - BasicTensorMethod
        for p=2 - CubicRegularizedNewton
        for p=2 - GradientDescent
        (default: None)
        tensor_step_kwargs (dict): kwargs for TensorStepMethod (default: None)
        subsolver (Optimizer): method to solve the inner problem (default: None)
        subsolver_args (dict): arguments for the subsolver (default: None)
        max_iters (int): number of the inner iterations of the subsolver to solve the inner problem (default: None)
        max_iters_ls (int): number of the line-search iterations (default: 50)
    """
    MONOTONE = False
    SKIP_TEST_LOGREG = False

    def __init__(self, params, L: float = 1., order: int = 3,
                 TensorStepMethod: Optimizer = None,
                 tensor_step_kwargs: dict = None,
                 subsolver: Optimizer = None, subsolver_args: dict = None,
                max_iters: int = None, max_iters_ls: int = 50,
                 verbose: bool = True, testing: bool = False):
        if L <= 0:
            raise ValueError(f"Invalid learning rate: L = {L}")

        super().__init__(params, dict(
            L=L, max_iters_ls=max_iters_ls))

        self.verbose = verbose
        self.testing = testing
        if len(self.param_groups) != 1:
            raise ValueError("Superfast doesn't support per-parameter options "
                             "(parameter groups)")
        group = self.param_groups[0]
        params = group['params']
        p = next(iter(params))
        state_common = self.state[p]
        state_common['theta'] = 1.
        state_common['A'] = 0.

        self.order = order

        self.tensor_step_method = step_definer(params=params, L=L, order=order,
                                               TensorStepMethod=TensorStepMethod, tensor_step_kwargs=tensor_step_kwargs,
                                               subsolver=subsolver, subsolver_args=subsolver_args,
                                               max_iters=max_iters, verbose=verbose, testing=testing)

        for p in params:
            state = self.state[p]
            state['x'] = p.detach().clone()
            state['y'] = state['x'].clone()


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


        A = state_common['A']
        theta = state_common['theta']

        L = group['L']
        max_iters_ls = group['max_iters_ls']

        fac = math.factorial(self.order - 1)
        s = self.order/(self.order+1)
        m = (s + 0.5) / 2
        l, u = 0., 1.
        A_new = A + 0.

        for _ in range(max_iters_ls):
            A_new = A / theta
            a = A_new - A

            for p in params:
                state = self.state[p]
                with torch.no_grad():
                    state['x_wave'] = state['y'].mul(theta).add(state['x'], alpha=1-theta)
                    p.zero_().add_(state['x_wave'])

            self.tensor_step_method.step(closure)
            self.zero_grad()


            with torch.no_grad():
                norm_squared = torch.tensor(0.)
                for p in params:
                    state = self.state[p]
                    state['x_wave'].sub_(p)
                    norm_squared += state['x_wave'].square().sum()
                norm = norm_squared.pow((self.order-1)/2.)

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


        with torch.no_grad():
            for p in params:
                state = self.state[p]
                state['y'] = p.detach().clone()

        closure().backward()

        with torch.no_grad():
            for p in params:
                state = self.state[p]
                state['x'].sub_(p.grad, alpha=a)

            state_common['A'] = A_new
            state_common['theta'] = theta
        return None
