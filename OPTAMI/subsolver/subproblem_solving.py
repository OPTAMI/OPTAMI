import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import OPTAMI as opt
import matplotlib.pyplot as plt
import math
import numpy as np

from Hyperfast_v2.OPTAMI.sup.tuple_to_vec import tuple_to_vector as ttv

def _optim_test(x, c, A, L):
    # c + |x|^2 * L * x + A * x
    first = x.mul(x.square().sum()).mul(L)
    return first.add(c).add(A.mv(x)).abs().max()

def _optim_test_hess_v(x, c, Ax, L):
    # c + |x|^2 * L * x + A * x
    first = x.mul(x.square().sum()).mul(L)
    return first.add(c).add(Ax).abs().max()


#def sub_solve(tol, subsolver, subsolver_args, closure_args, x_tilde):
def sub_solve(tol, init_point, subsolver, subsolver_args, closure_args, params, closure):
    """
    Solves the subproblem with given parameters
        min_x c^T x + 0.5 x^T A x + L / 4 \| x \|^4                          (1)
    tol: float
        solution tolerance
    subsolver: optimizer
        method that solves the problem
    subsolver_args: dict
        arguments for subsolver
    closure_args: dict
        contains c, L from (1) !!!! c MUST NOT BE DETACHED!!!!!!!!!!!!!!!!
    x_tilde_params : model.parameters()
        the point, where we need to compute hessian
    """
    
    #c, A, L = closure_args.values()
    c, L = closure_args.values()
    #x_tilde = x_tilde_params.clone()

    #if torch.cuda.is_available():
    #	x_ = torch.tensor(init_point, requires_grad=True).cuda()
    #else:
    #	x_ = torch.tensor(init_point, requires_grad=True)
    #x_ = torch.tensor(init_point, requires_grad=True)
    #the x_ above with commented below yields type(x_grad) = NoneType
    x_ = torch.zeros_like(c, requires_grad=True)
    #with such x_ everything works
    params_local = [x_]
    if subsolver.__name__ == 'APDGD':
    	optimizer = subsolver(params_local, **{'L': L})
    else:
    	optimizer = subsolver(params_local, **subsolver_args)
    optimizer.zero_grad()

    if subsolver.__name__ == 'APDGD':
    	zeroing_optimizer = subsolver(params, **{'L': L})
    else:
    	zeroing_optimizer = subsolver(params, **subsolver_args)
    
    #x = torch.tensor(list(params_local)[0]).flatten().requires_grad_()
    loss_history = []
    if ((subsolver.__name__ != 'LBFGS') & (subsolver.__name__!='APDGD') & (subsolver.__name__!= 'UAGMsDR')):
        for i in range(int(1/tol)):
            optimizer.zero_grad()

            out_for_grad = closure()
            grads_x_tilde = ttv(torch.autograd.grad(out_for_grad, list(params), create_graph=True))
            grad_vec = grads_x_tilde.mul(params_local[0]).sum()
            #print('grad_vec = ', grad_vec)
            #print('x_tilde = ', ttv(list(params)))
            hess_vec = ttv(torch.autograd.grad(grad_vec, list(params), retain_graph=False))

            #print("type x_.grad = ", type(x_.grad))
            #print("type assigning = ", type(c + hess_vec + L * params_local[0].mul(params_local[0]).sum() * params_local[0]))
            x_.grad = c + hess_vec + L * params_local[0].mul(params_local[0]).sum() * params_local[0]
            zeroing_optimizer.zero_grad()
            #print(loss_tmp)
            #loss.backward()
            optimizer.step()
            # loss_history.append(loss.detach().item())

    elif ((subsolver.__name__=='APDGD') | (subsolver.__name__== 'UAGMsDR')):
        for i in range(int(1/tol)):
            #optimizer.zero_grad()
            out_for_grad = closure()
            grads_x_tilde = ttv(torch.autograd.grad(out_for_grad, list(params), create_graph=True))
            grad_vec = grads_x_tilde.mul(params_local[0]).sum()
            #print('grad_vec = ', grad_vec)
            #print('x_tilde = ', ttv(list(params)))
            hess_vec = ttv(torch.autograd.grad(grad_vec, list(params), retain_graph=False))

            def local_closure():
            	optimizer.zero_grad()
            	return c.mul(params_local[0]).sum() + 0.5 * hess_vec.mul(params_local[0]).sum() + L / 4. * params_local[0].mul(params_local[0]).sum() ** 2
            loss = local_closure()
            zeroing_optimizer.zero_grad()
            #loss.backward
            optimizer.step(local_closure)
            # loss_history.append(loss.detach().item())
    else:
        raise ValueError('Error: LBFGS if not supported')
        # for i in range(int(1/tol)):
        #     #closure(x, c, A, L, optimizer)
        #     #cls(t) = closure(t, c=c, A=A, L=L, optimizer=optimizer)
        #     def cls():
        #         optimizer.zero_grad()
        #         x = params[0]#tuple_to_vector(list(params))
        #         lin = (x.mul(c)).sum()
        #         quad = 0.5 * A.mv(x).mul(x).sum()
        #         reg = L / 4. * x.mul(x).sum() ** 2
        #         loss = lin + quad + reg
        #         loss_history.append(loss.item())
        #         loss.backward()
        #         return loss
        #     #loss = cls(model, optimizer, c, A, L)
        #     #print(loss_tmp)
        #     #loss.backward()
        #     optimizer.step(cls)
        #     #loss_history.append(loss.detach().item())
    #return tuple_to_vector(list(params.detach())), loss_history
    x_sol = params_local[0].detach() 
    #if _optim_test(x_sol, c, A, L).ge(0.01).item():

    #uncomment print to monitor subproblem tolerance
    #print("_optim_test = ",_optim_test_hess_v(x_sol, c, hess_vec, L))

    #zeroing_optimizer.zero_grad()
    #    raise ValueError('Error: x in subproblem is not optimal. GradNorm = ' + str(_optim_test(x_sol, c, A, L).item()))
    #print("_optim_test = ", _optim_test(x_sol, c, A, L).item())
    
    return x_sol, loss_history