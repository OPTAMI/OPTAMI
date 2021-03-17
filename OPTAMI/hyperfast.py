import torch
from torch.optim.optimizer import Optimizer
import OPTAMI as opt
from OPTAMI.sup import tuple_to_vec as ttv


class Hyperfast(Optimizer):
    """Implements Hyperfast Second-Order Method.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        L (float, optional): Lipshitz constant of the Third-order (default: 1e+3)
    """

    def __init__(self, params, L=1e+3, eps=1e-1, p_order=3, subsolver=opt.BDGM,
                 subsolver_bdgm=None, tol_subsolve=None,
                 subsolver_args=None):

        if not L >= 0.0:
            raise ValueError("Invalid L: {}".format(L))

        defaults = dict(L=L, eps=eps, p_order=p_order, subsolver=subsolver,
                        subsolver_bdgm=subsolver_bdgm, tol_subsolve=tol_subsolve, subsolver_args=subsolver_args)
        super(Hyperfast, self).__init__(params, defaults)

        for group in self.param_groups:
            params = group['params']
            state = self.state[list(params)[0]]
            xk = []
            yk = []
            for i in range(len(list(params))):
                yk.append(list(params)[i].clone().detach())
                xk.append(list(params)[i].clone().detach())
            state['xk'] = xk
            state['yk'] = yk
            state['theta'] = 1.
            state['bigA'] = 0.

    def share_memory(self):
        for group in self.param_groups:
            params = group['params']
            state = self.state[list(params)[0]]
            state['theta'].share_memory_()
            state['bigA'].share_memory_()
            state['xk'].share_memory_()
            state['yk'].share_memory_()


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
            subsolver = group['subsolver']
            p_order = group['p_order']
            theta = state['theta']
            bigA = state['bigA']
            xk = state['xk']
            yk = state['yk']
            L = group['L']

            Hf = L * 3 / 2
            bigAplus = bigA
            line_search = True
            line_search_count = 0
            theta_up = 1
            theta_down = 0
            right_eq = p_order / (p_order + 1.)
            middle_eq = (right_eq + 0.5) / 2.
            fac = ttv.factorial(p_order - 1)
                
            subsolver_bdgm = group['subsolver_bdgm']
            tol_subsolve = group['tol_subsolve']
            subsolver_args = group['subsolver_args']
            
            optimizer = subsolver(params, L=L, subsolver_bdgm=subsolver_bdgm, tol_subsolve=tol_subsolve,
                                  subsolver_args=subsolver_args)  # define subsolver

            while line_search:

                bigAplus = bigA / theta
                ak = bigAplus - bigA
                xk_tilda = []
                with torch.no_grad():
                    for i in range(len(list(params))):
                        list(params)[i].zero_().add_(yk[i], alpha=theta).add_(xk[i], alpha=1 - theta)
                        xk_tilda.append(list(params)[i].clone().detach())

                # BDGM get the solution by one step because of inexact criterea
                optimizer.step(closure)
                for i in range(len(list(params))):
                    xk_tilda[i].sub_(list(params)[i])

                norm_xk_vk = ttv.tuple_norm_square(xk_tilda).pow(p_order - 1).sqrt().item()

                # linear-search------------

                inequality = ((1. - theta) / theta * bigA * (1. - theta)) * Hf * norm_xk_vk / fac  # Hf = L * 3/2
                line_search_count += 1

                if bigA == 0.:
                    ak = fac / (Hf * norm_xk_vk * 2.)
                    bigAplus = bigA + ak
                    state['theta'] = 1.
                    line_search = False
                else:
                    # print('ls:', line_search_count)
                    # print('theta_up:', theta_up)
                    # print('theta:', theta)
                    # print('theta_down:', theta_down)
                    # print('ineq:', inequality)
                    if 0.5 <= inequality <= right_eq:
                        state['theta'] = theta
                        print('theta after ls', theta)
                        line_search = False
                    else:
                        if inequality < middle_eq:
                            theta_up = theta
                            theta = (theta + theta_down) / 2.
                        else:
                            theta_down = theta
                            theta = (theta_up + theta) / 2.

            with torch.no_grad():
                for i in range(len(list(params))):
                    yk[i] = list(params)[i].clone().detach()
            print('End of line search. Total line_search calc:', line_search_count)

            # Compute gradient in point y
            output_y = closure()
            grad_y = torch.autograd.grad(output_y, list(params), retain_graph=False)
            # print(output_y)

            # Update xk
            with torch.no_grad():
                for i in range(len(yk)):
                    xk[i].sub_(grad_y[i], alpha=ak)
            state['bigA'] = bigAplus
            state['xk'] = xk
            state['yk'] = yk

        return None
