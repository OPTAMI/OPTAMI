import torch
from . import tuple_to_vec


# Return tuple format of hessian-vector-product
def hvp_from_grad(grads_tuple, list_params, vec_tuple):
    # don't damage grads_tuple. Grads_tuple should be calculated with create_graph=True
    dot = 0.
    for grad, vec in zip(grads_tuple, vec_tuple):
        dot += grad.mul(vec).sum()
    return torch.autograd.grad(dot, list_params, retain_graph=True)


# Return tuple format of hessian-vector-product
def hess_vec_prod(closure, list_params, vec_tuple):
    # should be more friendly for sparse tensors
    output = closure()
    grads = torch.autograd.grad(output, list_params, create_graph=True)
    hvp = hvp_from_grad(grads, list_params, vec_tuple)
    return hvp, grads


def flat_hvp(closure, list_params, vector):
    output = closure()
    flat_grad = tuple_to_vec.tuple_to_vector(
        torch.autograd.grad(output, list_params, create_graph=True))
    dot = flat_grad.mul(vector).sum()
    hvp = tuple_to_vec.tuple_to_vector(torch.autograd.grad(dot, list_params))
    return hvp, flat_grad


def third_derivative_vec(closure, params, vector, flat=False):
    output = closure()
    grads = torch.autograd.grad(output, params, create_graph=True)
    dot = 0.
    for i in range(len(grads)):
        dot += grads[i].mul(vector[i]).sum()
    hvp = torch.autograd.grad(dot, params, create_graph=True)
    dot_hes = 0.
    for i in range(len(grads)):
        dot_hes += hvp[i].mul(vector[i]).sum()
    third_vp = torch.autograd.grad(dot_hes, params)
    hvp_det = [hvp[i].detach() for i in range(len(grads))]
    if flat:
        return tuple_to_vec.tuple_to_vector(third_vp), tuple_to_vec.tuple_to_vector(hvp_det)
    else:
        return third_vp, hvp_det


def flat_hessian(flat_grads, params):
    full_hessian = []
    for i in range(flat_grads.size()[0]):
        temp_hess = torch.autograd.grad(flat_grads[i], params,
                                        retain_graph=True)
        full_hessian.append(tuple_to_vec.tuple_to_vector(temp_hess))
    return torch.stack(full_hessian)
