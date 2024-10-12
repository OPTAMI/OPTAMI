import torch
import warnings
from torch.optim.optimizer import Optimizer
import OPTAMI

def step_definer(params, L: float = 1., order: int = 3,
                 TensorStepMethod: Optimizer = None, tensor_step_kwargs: dict = None,
                 subsolver: Optimizer = None, subsolver_args: dict = None,
                 max_subsolver_iterations: int = None, verbose: bool = True, testing: bool = False):
    if TensorStepMethod is None:
        if order == 3:
            tensor_step_method = OPTAMI.BasicTensorMethod(
                params, L=L, subsolver=subsolver, subsolver_args=subsolver_args,
                max_iters=max_subsolver_iterations, verbose=verbose, testing=testing)
        elif order == 2:
            tensor_step_method = OPTAMI.CubicRegularizedNewton(
                params, L=L, subsolver=subsolver, verbose=verbose,
                subsolver_args=subsolver_args, max_iters=max_subsolver_iterations, testing=testing)
        else:  # order = 1
            tensor_step_method = OPTAMI.GradientDescent(params, L=L, testing=testing)
    else:
        if not hasattr(TensorStepMethod, 'MONOTONE') or not TensorStepMethod.MONOTONE:
            warnings.warn("`TensorStepMethod` should be monotone!")
        tensor_step_method = TensorStepMethod(params, **tensor_step_kwargs)
    return tensor_step_method