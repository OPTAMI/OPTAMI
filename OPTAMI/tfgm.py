import torch
from torch.optim.optimizer import Optimizer


class TFGM(Optimizer):
    """Implements Triangle Fast Gradient Method.

    It has been proposed in `Universal Method for Stochastic Composite
Optimization Problems https://link.springer.com/content/pdf/10.1134/S0965542518010050.pdf

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
    """

    def __init__(self, params, L=1e+2):
        if not 0.0 <= L:
            raise ValueError("Invalid learning rate: {}".format(L))

        defaults = dict(L=L)
        super(TFGM, self).__init__(params, defaults)

        for group in self.param_groups:
            params = group['params']
            state = self.state[list(params)[0]]

            uk = []
            for i in range(len(list(params))):
                uk.append(list(params)[i].clone().detach())
            state['uk'] = uk
            state['Ak'] = 0.

    def share_memory(self):
        for group in self.param_groups:
            params = group['params']
            state = self.state[list(params)[0]]
            state['Ak'].share_memory_()
            state['uk'].share_memory_()
            state['xk'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if closure is None:
            raise ValueError("Closure is None. Closure is necessary for this method. ")
        for group in self.param_groups:
            params = group['params']
            state = self.state[list(params)[0]]
            uk = state['uk']
            leng = len(list(params))
            # Compute a_{k+1}
            c = 1. / group['L'] / 2.
            ak = (c * c + state['Ak'] / group['L']) ** 0.5 + c
            # Compute A_{k+1}
            Ak_plus = state['Ak'] + ak
            alpha = state['Ak'] / Ak_plus
            # Change current point to y_{k+1}
            xk = []
            with torch.no_grad():
                for i in range(leng):
                    xk.append(list(params)[i].clone().detach())
                    list(params)[i].mul_(alpha).add_(uk[i].mul(1. - alpha))
            # Compute gradient in y_{k+1}
            output_y = closure()
            grad_yk = torch.autograd.grad(output_y, list(params))
            # Compute u_{k+1} and change point to x_{k+1}
            with torch.no_grad():
                for i in range(leng):
                    uk[i].sub_(grad_yk[i].mul(ak))
                    list(params)[i].zero_().add_(xk[i].mul(alpha)).add_(uk[i].mul(1.-alpha))
            state['Ak'] = Ak_plus
            state['uk'] = uk
        return None
