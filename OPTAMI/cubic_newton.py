import torch
from torch.optim.optimizer import Optimizer
from OPTAMI.sup import tuple_to_vec as ttv
from OPTAMI.subsolver import cubic_subproblem_solver as css


class Cubic_Newton(Optimizer):
    """Implements Cubic Newton Method.
    Nesterov Y, Polyak BT. "Cubic regularization of
    Newton method and its global performance." Mathematical Programming. 2006 Aug;108(1):177-205.
    https://link.springer.com/content/pdf/10.1007/s10107-006-0706-8.pdf

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        L (float): Lipshitz constant of the Second-order
        subsolver(Optimizer): Optimization method to solve inner problem by gradient steps
        number_inner_iter(int): number of inner iteration of Subsolver to solve the inner problem
        cubic_linear_part(tensor): Should be a flat vector. If used, then function for optimization
            equal to <cubic_linear_part,x> + f(x).
            To be used for inexact, stochastic or distributed versions of Cubic Newton Method.


    """

    def __init__(self, params, L, subsolver, subsolver_args, number_inner_iter, cubic_linear_part=None):

        if not L >= 0.0:
            raise ValueError("Invalid L: {}".format(L))

        defaults = dict(L=L,  subsolver=subsolver, subsolver_args=subsolver_args,
                        number_iter=number_inner_iter, cubic_linear_part=cubic_linear_part)
        super(Cubic_Newton, self).__init__(params, defaults)

        for group in self.param_groups:
            params = group['params']
            state = self.state[list(params)[0]]
            xk = []
            for i in range(len(list(params))):
                xk.append(list(params)[i].clone().detach())
            state['xk'] = xk


    def share_memory(self):
        for group in self.param_groups:
            params = group['params']
            state = self.state[list(params)[0]]
            state['xk'].share_memory_()


    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        if closure is None:
            raise ValueError("Closure is None. Closure is necessary for this method. ")
        for group in self.param_groups:

            params = group['params']
            state = self.state[list(params)[0]]
            xk = state['xk']
            L = group['L']
            subsolver = group['subsolver']
            subsolver_args = group['subsolver_args']
            number_inner_iter = group['number_inner_iter']
            cubic_linear_part = group['cubic_linear_part']
            if cubic_linear_part is None:
                cubic_linear_part = torch.zeros(ttv.tuple_numel(xk))
            flat_xk = css.subsolve_cubic_problem(params, closure, L, subsolver, subsolver_args,
                                                 number_inner_iter, cubic_linear_part)

            xk = ttv.rollup_vector(flat_xk, xk)
            with torch.no_grad():
                for i in range(len(xk)):
                    list(params)[i].add_(xk[i])

            state['xk'] = xk

        return None
