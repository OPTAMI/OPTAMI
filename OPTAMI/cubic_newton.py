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
        L (float): Lipshitz constant of the Hessian.
        subsolver(Optimizer): Optimization method to solve the inner problem by gradient steps.
        subsolver_args(dict): arguments for the subsolver such as a learning rate and others.
        number_inner_iter(int): number of the inner iterations of the Subsolver to solve the inner problem.
        inner_rel_err(float): Should be < 1. Relative stopping criterion for the inner problem.


    """

    def __init__(self, params, L, subsolver=None, subsolver_args=None,
                 number_inner_iter=100, inner_rel_err=0.1):

        if subsolver_args is None:
            subsolver_args = {'lr': 0.01}
        if not L >= 0.0:
            raise ValueError("Invalid L: {}".format(L))

        defaults = dict(L=L, subsolver=subsolver, subsolver_args=subsolver_args,
                        number_inner_iter=number_inner_iter, inner_rel_err=inner_rel_err)
        super(Cubic_Newton, self).__init__(params, defaults)

        for group in self.param_groups:
            params = group['params']
            state = self.state[list(params)[0]]
            zeros_tuple = []
            for i in range(len(list(params))):
                zeros_tuple.append(list(params)[i].clone().detach())
            state['zeros_tuple'] = zeros_tuple

    def share_memory(self):
        for group in self.param_groups:
            params = group['params']
            state = self.state[list(params)[0]]
            state['zeros_tuple'].share_memory_()

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
            zeros_tuple = state['zeros_tuple']
            L = group['L']
            subsolver = group['subsolver']
            subsolver_args = group['subsolver_args']
            number_inner_iter = group['number_inner_iter']
            inner_rel_err = group['inner_rel_err']
            if subsolver is None:
                flat_xk = css.cubic_subsolver(L, closure, params)
            else:
                flat_xk = css.subsolve_cubic_problem(params, closure, L, zeros_tuple,
                                                 subsolver, subsolver_args, number_inner_iter, inner_rel_err)

            xk = ttv.rollup_vector(flat_xk, zeros_tuple)
            with torch.no_grad():
                for i in range(len(xk)):
                    list(params)[i].add_(xk[i])

        return None
