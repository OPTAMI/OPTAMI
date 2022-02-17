import torch


def tuple_norm_square(tuple_in, length=None):
    norm_in = 0.
    if length is None:
        length = len(tuple_in)
    for i in range(length):
        norm_in += tuple_in[i].square().sum()
    return norm_in


# return a flat vector from a tuple of vectors and matrices.
def tuple_to_vector(tuple_in, length=None):
    if length is None:
        length = len(tuple_in)
    flat_vector = []
    for j in range(length):
        flat_vector.append(tuple_in[j].view(-1))
    flat_vector = torch.cat(flat_vector)
    if flat_vector.dim() > 1:
        print('Error: result is not a vector')
    return flat_vector


# return a tuple with number of elements in each vector.
def tuple_numel(tuple_in, length=None):
    if length is None:
        length = len(tuple_in)
    t_numel = []
    for j in range(length):
        t_numel.append(tuple_in[j].numel())
    return t_numel


# return a tuple of vectors from a flat vec
def rollup_vector(flat_vector, tuple_in, length=None, vec_numel=None):
    if length is None:
        length = len(tuple_in)
    if vec_numel is None:
        vec_numel = tuple_numel(tuple_in)
    new_vec = torch.split(flat_vector, vec_numel)
    good_vec = []
    for j in range(length):
        good_vec.append(new_vec[j].view_as(tuple_in[j]))
    return good_vec
