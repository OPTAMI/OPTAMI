import torch
from torch.optim.optimizer import Optimizer
import OPTAMI as opt
from OPTAMI.sup import tuple_to_vec as ttv


class Superfast(Optimizer):
    """Implements Superfast Second-Order Method.
    Nesterov, Y., 2020. Superfast second-order methods for unconstrained convex optimization UCL-Université Catholique de Louvain, CORE.
    https://dial.uclouvain.be/pr/boreal/object/boreal%3A227146/
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        L (float): Lipshitz constant of the Third-order (default: 1e+3)
        divisor (float): constant connected with the step's length. Lower constant is a bigger step.
        divisor = 604.8 by theory.
    """

    def __init__(self, params, L=1e+3, divisor=604.8, subsolver=opt.BDGM,
                 subsolver_params=None, subsolver_bdgm=None, tol_subsolve=None,
                 subsolver_args=None):

        if not L >= 0.0:
            raise ValueError("Invalid L: {}".format(L))

        defaults = dict(L=L, divisor=divisor, subsolver=subsolver, subsolver_params=subsolver_params,
                        subsolver_bdgm=subsolver_bdgm, tol_subsolve=tol_subsolve, subsolver_args=subsolver_args)
        super(Superfast, self).__init__(params, defaults)

        for group in self.param_groups:
            params = group['params']
            state = self.state[list(params)[0]]
            xk = []
            vk = []
            grad_sum = []
            for i in range(len(list(params))):
                vk.append(list(params)[i].clone().detach())
                xk.append(list(params)[i].clone().detach())
                grad_sum.append(torch.zeros_like(list(params)[i]))
            state['xk'] = xk
            state['x0'] = xk
            state['vk'] = vk
            state['grad_sum'] = grad_sum
            state['k_step'] = 0

    def share_memory(self):
        for group in self.param_groups:
            params = group['params']
            state = self.state[list(params)[0]]
            state['grad_sum'].share_memory_()
            state['k_step'].share_memory_()
            state['xk'].share_memory_()
            state['vk'].share_memory_()

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
            subsolver_bdgm = group['subsolver_bdgm']
            tol_subsolve = group['tol_subsolve']
            subsolver_args = group['subsolver_args']
            divisor = group['divisor']
            L = group['L']

            grad_sum = state['grad_sum']
            k = state['k_step']
            xk = state['xk']
            x0 = state['x0']
            vk = state['vk']

            #Form of A_k: Ak = k**4 / divisor / L, but we don't need it in code

            # Precomputed A_k/A_{k+1}
            Ak_div_Ak_plus = (1.-1./(k+1.))**4
            #print(Ak_div_Ak_plus)
            # Some precomputed form of A_{k+1}-A_{k} for p = 3
            ak = (2*k+1.)*(2*(k*(k+1.))+1.)/divisor/L
            #print(ak)
            # Compute y_k = (A_k/A_{k+1})*x_k+ (1-(A_k/A_{k+1}))*v_k
            # In params x_k
            with torch.no_grad():
                for i in range(len(list(params))):
                    list(params)[i].mul_(Ak_div_Ak_plus).add_(vk[i], alpha=1.-Ak_div_Ak_plus)
            # In params y_k

            # BDGM step from y_k
            optimizer = subsolver(params, L=L, subsolver_bdgm=subsolver_bdgm, tol_subsolve=tol_subsolve,
                                  subsolver_args=subsolver_args)
            optimizer.step(closure)
            # In params x_{k+1}

            # Saving x_{k+1}
            for i in range(len(list(params))):
                xk[i].zero_().add(list(params)[i])

            #Adding scaled gradient to sum for psi_{k+1}:
            #grad_sum = grad_sum + a_{k+1} * \nabla f(x_{k+1})
            grad_xk = torch.autograd.grad(closure(), params)
            for i in range(len(list(params))):
                grad_sum[i].add_(grad_xk[i].mul_(ak))

            #Computing norm of v_{k+1}
            scaled_grad_sum = ttv.tuple_norm_square(grad_sum).pow(1/3) #pow(1/3) maybe bad
            #print(scaled_grad_sum)
            #Computing v_{k+1} = = x_0 - grad_sum/||grad_sum||**(2/3)
            for i in range(len(list(params))):
                vk[i].zero_().add_(x0[i]).sub_(grad_sum[i].div(scaled_grad_sum))

            # In params x_{k+1}
            state['grad_sum'] = grad_sum
            state['xk'] = xk
            state['vk'] = vk
            state['k_step'] = state['k_step'] + 1

        return None
