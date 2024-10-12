import torch
import OPTAMI as opt
import time
from OPTAMI.utils import derivatives
from OPTAMI.utils import tuple_to_vec as ttv


def func_fit(optimizer, iter_num, func, x, precision: float = 1e-10, **kwargs):
    losses = []
    norm_grads = []
    time_steps = [0.]

    def closure():
        optimizer.zero_grad()
        return func(x, **kwargs)

    iters = 0
    func_loss_prev = 1e+10
    flag = True
    while iters < iter_num and flag:
        func_loss = closure()
        losses.append(func_loss.item())
        norm_grad = torch.autograd.grad(func_loss, [x])[0].square().sum().item()
        norm_grads.append(norm_grad)
        step_timer = time.time()
        optimizer.step(closure)
        time_steps.append(time.time() - step_timer + time_steps[-1])
        if (func_loss_prev - func_loss).abs() < precision:
            flag = False
        func_loss_prev = func_loss + 0.
        iters += 1

    func_loss = closure()
    norm_grad = torch.autograd.grad(func_loss, [x])[0].square().sum().item()
    norm_grads.append(norm_grad)
    losses.append(func_loss.item())

    return losses, time_steps, norm_grads


def L_1_exact(func, x, *args):
    flat_grads = ttv.tuple_to_vector(torch.autograd.grad(func(x), [x], create_graph=True))
    hes = derivatives.flat_hessian(flat_grads, [x])
    L = torch.linalg.norm(hes, ord=2)
    return L

def L_1_stochastic(points_number, closure, x, power = True, *args):
    L = 0.
    y = ttv.tuple_to_vector(x.clone().detach())
    dimension = y.size()
    z = torch.randn(dimension)
    z.div_(z.norm())
    flat_grad = ttv.tuple_to_vector(torch.autograd.grad(closure(), [x], create_graph=True))

    for j in range(points_number):
        dot = flat_grad.mul(z).sum()
        hvp = ttv.tuple_to_vector(torch.autograd.grad(dot, [x], retain_graph=True))
        L_h = hvp @ z
        if power:
            z = hvp.div(hvp.norm())
        else:
            z = torch.randn(dimension)
            z.div_(z.norm())
        if L < L_h:
            L = L_h.item()
    return L, z

def L_3_stochastic(points_number, vector_number, func, x, *args):
    L = 0.
    dimension = x.size()
    z_w = torch.randn(dimension)
    y_w = torch.randn(dimension)
    vec_w = torch.randn(dimension)

    for j in range(points_number):
        z = torch.randn(dimension)
        z.div_(z.norm()).requires_grad_()
        y = torch.randn(dimension)
        y.div_(y.norm()).requires_grad_()
        optimizer_z = torch.optim.Adam([z])
        optimizer_y = torch.optim.Adam([y])

        def closure_z():
            optimizer_z.zero_grad()
            return func(z, *args)

        def closure_y():
            optimizer_y.zero_grad()
            return func(y, *args)

        for i in range(vector_number):
            vec = torch.randn(dimension)
            vec.div_(vec.norm())
            third_vp_z, _ = derivatives.third_derivative_vec(closure_z, [z], vec)
            third_vp_y, _ = derivatives.third_derivative_vec(closure_y, [y], vec)
            L_h = third_vp_z[0].sub(third_vp_y[0]).norm().div(z.detach().sub(y.detach()).norm())
            if L < L_h:
                L = L_h
                z_w = z.detach().clone()
                y_w = y.detach().clone()
                vec_w = vec.clone()
    return L, z_w, y_w, vec_w


def L_2_stochastic(points_number, vector_number, func, x, *args):
    L = 0.
    dimension = x.size()
    z_w = torch.randn(dimension)
    y_w = torch.randn(dimension)
    vec_w = torch.randn(dimension)

    for j in range(points_number):
        z = torch.randn(dimension)
        z.div_(z.norm()).requires_grad_()
        y = torch.randn(dimension)
        y.div_(y.norm()).requires_grad_()
        optimizer_z = torch.optim.Adam([z])
        optimizer_y = torch.optim.Adam([y])

        def closure_z():
            optimizer_z.zero_grad()
            return func(z, *args)

        def closure_y():
            optimizer_y.zero_grad()
            return func(y, *args)

        for i in range(vector_number):
            vec = torch.randn(dimension)
            vec.div_(vec.norm())
            hvp_z, _ = derivatives.hess_vec_prod(closure_z, [z], vec)
            hvp_y, _ = derivatives.hess_vec_prod(closure_y, [y], vec)
            L_h = hvp_z[0].sub(hvp_y[0]).norm().div(z.detach().sub(y.detach()).norm())
            if L < L_h:
                L = L_h
                z_w = z.detach().clone()
                y_w = y.detach().clone()
                vec_w = vec.clone()
    return L, z_w, y_w, vec_w


