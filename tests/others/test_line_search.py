import sys
sys.path.append("./")
import OPTAMI.utils.line_search as ls
import torch


def unbound(x):
    return -x


def square(x):
    return (x - 10) ** 2


def inverse(x):
    return 1. / (1+x)


def exp(x):
    return torch.exp(x)


def test_ray_search():
    left = torch.tensor([0.])
    middle = torch.tensor([1.])
    eps = 1e-4
    delta = 1e-8
    assert ls.ray_line_search(unbound, left_point=0., middle_point=1.)[0] is None
    assert ls.ray_line_search(unbound, left_point=left, middle_point=middle)[0] is None
    assert - eps < ls.ray_line_search(square, left_point=left, middle_point=middle) - 10.0 < eps
    assert inverse(ls.ray_line_search(inverse, left_point=left, middle_point=middle, delta=delta)) < delta


def test_segment_search():
    eps = 1e-4
    left = torch.tensor([0.])
    right = torch.tensor([20.])
    for segment_search in [ls.segment_search_ternary, ls.segment_search_gold]:
        assert abs(segment_search(square, left_point=0., right_point=20.) - 10.) < eps
        assert abs(segment_search(square, left_point=left, right_point=right) - 10.) < eps
        assert segment_search(exp, left_point=left, right_point=right) == 0.
        assert abs(segment_search(inverse, left_point=0., right_point=right) - 20.) < 0.001




