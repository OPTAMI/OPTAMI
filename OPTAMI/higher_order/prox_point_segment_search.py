from torch.optim.optimizer import Optimizer
from OPTAMI.utils import tuple_to_vec
import warnings
import OPTAMI
import torch


class ProxPointSS(Optimizer):
    """Implements Superfast Second-Order Method.
     Inexact version was proposed by Yu.Nesterov in "Superfast Second-Order Methods for Unconstrained Convex Optimization"
     https://doi.org/10.1007/s10957-021-01930-y
    Contributors:
        Dmitry Kamzolov
        Dmitry Vilensky-Pasechnyuk
        Golubeva T.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        L (float): estimated value of Lipschitz constant for the hessian (default: 1e+3)
    """
    MONOTONE = False

    def __init__(self, params, L: float = 1e+3, approx: float = 16., 
                 TensorStepMethod: Optimizer = None, tensor_step_kwargs: dict = None,
                 subsolver: Optimizer = None, subsolver_args: dict = None,
                 max_iters: int = None, verbose: bool = True):
        if L <= 0:
            raise ValueError(f"Invalid learning rate: L = {L}")

        super().__init__(params, dict(L=L, approx=approx))
        if len(self.param_groups) != 1:
            raise ValueError("Superfast doesn't support per-parameter options "
                             "(parameter groups)")

        self.TensorStepMethod = TensorStepMethod
        self.subsolver = subsolver
        self.subsolver_args = subsolver_args
        self.max_iters = max_iters
        self.tensor_step_kwargs = tensor_step_kwargs
        self.tensor_step_method = None

        self.verbose = verbose


    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): a closure that reevaluates the model and returns the loss.
        """
        closure = torch.enable_grad()(closure)
        assert len(self.param_groups) == 1

        group = self.param_groups[0]
        params = group['params']
        approx = group['approx']
        L = group['L']

        if self.tensor_step_method is None:
            if self.TensorStepMethod is None:
                self.tensor_step_method = OPTAMI.BasicTensorMethod(
                    params, L=L, subsolver=self.subsolver, verbose=self.verbose,
                    subsolver_args=self.subsolver_args, max_iters=self.max_iters)
            else:
                if not hasattr(self.TensorStepMethod, 'MONOTONE') or not self.TensorStepMethod.MONOTONE:
                    warnings.warn("`TensorStepMethod` should be monotone!")
                self.tensor_step_method = self.TensorStepMethod(params, **self.tensor_step_kwargs)

        p = next(iter(params))
        state_common = self.state[p]

        if 'A' not in state_common:
            state_common['A'] = 0.

        for p in params:
            state = self.state[p]

            if ('v' not in state) or ('x' not in state):
                state['x'] = p.detach().clone()
                state['v'] = state['x'].clone()

        A = state_common['A']
        H_div = approx * L

        for p in params:
            state = self.state[p]
            with torch.no_grad():
                state['u'] = state['v'].sub_(state['x'])
        
        self.tensor_step_method.step(closure)
        self.zero_grad()

        closure().backward()

        scal_1 = torch.tensor(0.)
        for p in params:
            state = self.state[p]
            with torch.no_grad():
                state['T_1'] = p.detach().clone()
                state['df_det_1'] = p.grad
                scal_1.add_(state['df_det_1'].mul(state['u']).sum())

        df_norm_1 = torch.tensor(0.)
        for p in params:
            state = self.state[p]
            with torch.no_grad():
                df_norm_1 += tuple_to_vec.tuple_norm_square(state['df_det_1'])
        df_norm_1 = df_norm_1.sqrt()

        if scal_1.ge(0):
            g = df_norm_1.clone()
            for p in params:
                state = self.state[p]
                with torch.no_grad():
                    state['phi'] = state['df_det_1']
                    state['x'] = state['T_1']
        else:
            for p in params:
                state = self.state[p]
                with torch.no_grad():
                    p.zero_().add(state['v'])
            
            self.tensor_step_method.step(closure)
            self.zero_grad()

            closure().backward()

            scal_2 = torch.tensor(0.)
            for p in params:
                state = self.state[p]
                with torch.no_grad():
                    state['T_2'] = p.detach().clone()
                    state['df_det_2'] = p.grad
                    scal_2.add_(state['df_det_2'].mul(state['u']).sum())

            df_norm_2 = torch.tensor(0.)
            for p in params:
                state = self.state[p]
                with torch.no_grad():
                    df_norm_2 += tuple_to_vec.tuple_norm_square(state['df_det_2'])
            df_norm_2 = df_norm_2.sqrt()

            if scal_2.mul(-1).ge(0):
                g = df_norm_2.clone()
                for p in params:
                    state = self.state[p]
                    with torch.no_grad():
                        state['phi'] = state['df_det_2']
                        state['x'] = state['T_2']
            else:
                tau_1 = 0.
                tau_2 = 1.

                alpha = scal_2.div(scal_2.sub(scal_1)).item()
                g_pow = df_norm_1.pow(4/3).mul(alpha).add(df_norm_2.pow(4/3).mul(1-alpha))

                iters = 1

                while scal_1.mul(2*(tau_1-tau_2)).ge(g_pow.mul(H_div**(1/3))):
                    tau = (tau_1 + tau_2) / 2
                    
                    for p in params:
                        state = self.state[p]
                        with torch.no_grad():
                            p.zero_().add_(state['x']).add_(state['u'], alpha=tau)

                    self.tensor_step_method.step(closure)
                    self.zero_grad()

                    closure().backward()

                    scal = torch.tensor(0.)
                    for p in params:
                        state = self.state[p]
                        with torch.no_grad():
                            state['df_det'] = p.grad
                            scal.add_(state['df_det'].mul(state['u']).sum())

                    if scal.ge(0):
                        tau_2 = tau
                        scal_2 = scal

                        for p in params:
                            state = self.state[p]
                            with torch.no_grad():
                                state['T_2'] = p.detach().clone()
                                state['df_det_2'] = state['df_det']
                    else:
                        tau_1 = tau
                        scal_1 = scal

                        for p in params:
                            state = self.state[p]
                            with torch.no_grad():
                                state['T_1'] = p.detach().clone()
                                state['df_det_1'] = state['df_det']

                    alpha = scal_2.div(scal_2.sub(scal_1)).item()
                    g_pow = df_norm_1.pow(4/3).mul(alpha).add(df_norm_2.pow(4/3).mul(1-alpha))

                    iters += 1
                
                if self.verbose:
                    print('line-search iterations:', iters)
                
                g = g_pow.pow(3/4)
                for p in params:
                    state = self.state[p]
                    with torch.no_grad():
                        state['phi'].zero_().add_(state['df_det_1'], alpha=alpha).add_(state['df_det_2'], alpha=1-alpha)
                        state['x'].zero_().add_(state['T_1'], alpha=alpha).add_(state['T_2'], alpha=1-alpha)

        divis = g.pow(2).mul_(H_div).pow(1/3)
        c = torch.tensor(1.).div(divis).div(2.)
        inner = c.pow(2).add(c.mul(4. * A))
        a = inner.sqrt().add(c).div(2.)

        for p in params:
            state = self.state[p]
            with torch.no_grad():
                state['v'].sub_(state['phi'], alpha=a)
                p.zero_().add_(state['x'])

        state_common['A'] = A + a
        return None
