import torch
from OPTAMI.sup import tuple_to_vec as ttv


def hess_vec_prod(closure, list_params, vec_tuple):  # Return tuple format of hessian-vector-product
    # should be more friendly for sparse tensors
    output = closure()
    grads = torch.autograd.grad(output, list_params, create_graph=True)
    dot = 0.
    for i in range(len(grads)):
        dot += grads[i].mul(vec_tuple[i]).sum()
    hvp = torch.autograd.grad(dot, list_params, retain_graph=False)
    return hvp, grads


def flat_hvp(closure, list_params, vector):
    output = closure()
    flat_grad = ttv.tuple_to_vector(torch.autograd.grad(output, list_params, create_graph=True))
    dot = flat_grad.mul(vector).sum()
    hvp = ttv.tuple_to_vector(torch.autograd.grad(dot, list_params, create_graph=True))
    return hvp, flat_grad


def third_derivative_vec(closure, list_params, vector):
    output = closure()
    grads = torch.autograd.grad(output, list_params, create_graph=True)
    dot = 0.
    for i in range(len(grads)):
        dot += grads[i].mul(vector[i]).sum()
    hvp = torch.autograd.grad(dot, list_params, create_graph=True)
    dot_hes = 0.
    for i in range(len(grads)):
        dot_hes += hvp[i].mul(vector[i]).sum()
    third_vp = torch.autograd.grad(dot_hes, list_params, retain_graph=False)
    hvp_det = []
    for pa in range(len(grads)):
        hvp_det.append(hvp[pa].detach())
    return third_vp, hvp_det


def flat_hessian(flat_grads, params):
    full_hessian = []
    for i in range(flat_grads.size()[0]):
        temp_hess = torch.autograd.grad(flat_grads[i], params,
                                        retain_graph=True)
        # print(temp_hess)
        full_hessian.append(ttv.tuple_to_vector(temp_hess))
    return torch.stack(full_hessian)
