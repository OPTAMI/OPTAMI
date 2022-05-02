import torch
import math


def segment_line_search_fib(g, left_point, right_point, eps=1e-10):
    fib = 0.5 + math.sqrt(5)/2.

    diff = right_point - left_point
    x1 = right_point - diff / fib
    x2 = left_point + diff / fib
    g1 = g(x1)
    g2 = g(x2)
    while diff > eps:
        if g1 > g2:
            left_point = x1
            x1 = x2
            g1 = g2
            diff = right_point - left_point
            x2 = left_point + diff / fib
            g2 = g(x2)
        else:
            right_point = x2
            x2 = x1
            g2 = g1
            diff = right_point - left_point
            x1 = right_point - diff / fib
            g1 = g(x1)

    return (right_point + left_point) / 2.


def segment_line_search_tri(g, left_point, right_point, eps=1e-10):
    left_point = left_point.to(torch.double)
    right_point = right_point.to(torch.double)
    x1 = right_point.add(left_point, alpha=2.).div_(3.)
    x2 = left_point.add(right_point, alpha=2.).div_(3.)
    while right_point.sub(left_point).ge(eps).item():
        if g(x1).sub(g(x2)).ge(0.).item():
            left_point = x1.clone()
        else:
            right_point = x2.clone()
        x1 = right_point.add(left_point, alpha=2.).div_(3.)
        x2 = left_point.add(right_point, alpha=2.).div_(3.)
    return right_point.add(left_point).div(2.)


def ray_line_search(g, middle_point, left_point, eps=1e-10):
    if g(middle_point).sub(g(left_point)).ge(0.).item():
        right_point = middle_point
    else:
        right_point = middle_point.mul(2.)

        while g(middle_point).sub(g(right_point)).ge(0.).item():
            left_point = middle_point.clone()
            middle_point = right_point.clone()
            right_point = middle_point.mul(2.)
    return segment_line_search_tri(g, left_point, right_point, eps)