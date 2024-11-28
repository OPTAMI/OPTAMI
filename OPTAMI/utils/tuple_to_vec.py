import torch
from typing import List

def norm_of_tensors(list_tensor: List[torch.Tensor]) -> float:
    return sum(x.square().sum().item() for x in list_tensor) ** 0.5

def tuple_norm_square(list_tensor: List[torch.Tensor])-> float:
    return sum(x.square().sum().item() for x in list_tensor)


# return a flat vector from a tuple of vectors and matrices.
def tuple_to_vector(tuple_in):
    return torch.cat([t.view(-1) for t in tuple_in])


# return a tuple with number of elements in each vector.
def tuple_numel(tuple_in):
    return [t.numel() for t in tuple_in]


# return a tuple of vectors from a flat vec
def rollup_vector(flat_vector, tuple_in):
    new_vec = torch.split(flat_vector, tuple_numel(tuple_in))
    return [v.view_as(t) for v, t in zip(new_vec, tuple_in)]