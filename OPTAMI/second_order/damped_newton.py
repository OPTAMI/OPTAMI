import torch
from torch.optim.optimizer import Optimizer
from OPTAMI.utils import tuple_to_vec, subproblem_solver

class DampedNewton(Optimizer):
    """Implements different versions of Damped Newton Method.

    x_{k+1} = x_k - alpha (\nabla^2 f(x) + lambd I)^{-1}\nabla f(x)
    A) classical Damped Newton Method, (variant = None)

    B) Affine-Invariant Cubic Newton, (variant = 'AIC') from
    "A Damped Newton Method Achieves Global O(1/k^2) and Local Quadratic Convergence Rate"
    by Slavomír Hanzely, Dmitry Kamzolov, Dmitry Pasechnyuk, Alexander Gasnikov,
    Peter Richtárik, and Martin Takáč. 2022. Advances in Neural Information Processing Systems, 35, pp.25320-25334.
https://proceedings.neurips.cc/paper_files/paper/2022/hash/a1f0c0cd6caaa4863af5f12608edf63e-Abstract-Conference.html

    C) Gradient Regularized Newton (variant = 'GradReg') from
    1) "Regularized Newton Method with Global $\mathcal O\left (\frac {1}{k^ 2}\right) $ Convergence." by Konstantin Mishchenko. 2023. SIAM Journal on Optimization, 33(3), pp.1440-1462.
    https://doi.org/10.1137/22M1488752
    2) "Gradient Regularization of Newton Method with Bregman Distances" by Nikita Doikov, Yurii Nesterov. 2024. Mathematical Programming, 204(1), pp.1-25.
    https://doi.org/10.1007/s10107-023-01943-7

    Contributors:
        Dmitry Kamzolov
        Dmitry Vilensky-Pasechnyuk

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        variant (str):
        'Classic' - classical Damped Newton Method;
        'AIC' - Affine-Invariant Cubic Newton;
        'GradReg' - Gradient Regularized Newton (default: 'GradReg')
        alpha (float): step-size for Damped Newton Method (default: 1.)
        L (float): estimated value of Lipschitz constant of the Hessian (default: 1.)
        reg (float): estimated value of Hessian regularizer (default: 0.)
        CG_subsolver (bool): if True, it uses CG as a subsolver (default: True)
        testing (bool): if True, it may compute some additional tests. (default: False)
    """
    MONOTONE = True
    ORDER = 2

    def __init__(self, params, variant: str = 'GradReg', alpha: float = 1., L: float = 1.,
                 reg: float = 0., CG_subsolver: bool = True,  verbose: bool = True, testing: bool = False):
        if L <= 0:
            raise ValueError(f"Invalid learning rate: L = {L}")

        super().__init__(params, dict(
            alpha=alpha, variant=variant,
            reg=reg, L=L))
        self.CG_subsolver = CG_subsolver
        self.verbose = verbose
        self.testing = testing

    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): a closure that reevaluates the model and returns the loss
        """
        closure = torch.enable_grad()(closure)

        for group in self.param_groups:
            params = group['params']
            variant = group['variant']
            alpha = group['alpha']
            L = group['L']
            reg = group['reg']
            grad = torch.autograd.grad(closure(), list(params), create_graph=True)

            if variant == 'Classic':
                reg = 0.
                alpha = alpha
            elif variant == 'GradReg':
                g_norm = torch.sqrt(tuple_to_vec.tuple_norm_square(grad))
                reg = (L * g_norm).item() ** 0.5
                alpha = 1.
            elif variant == 'AIC':
                reg = 0.

            if self.CG_subsolver:
                h = subproblem_solver.CG_subsolver(params=params, grad=grad, reg=reg, testing=self.testing)
            else:
                h = subproblem_solver.quadratic_exact_solve(params=params, grad=grad, reg=reg, testing=self.testing)

            if variant == 'AIC':
                G = 0.
                for h_i, g_i in zip(h, grad):
                    G += g_i.mul(h_i).sum()
                G = L * G ** 0.5
                alpha = ((1 + 2 * G) ** 0.5 - 1) / G

            with torch.no_grad():
                for p, h in zip(params, h):
                    p.sub_(h, alpha=alpha)
        return None


