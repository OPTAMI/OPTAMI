import torch
import math


def segment_search_gold(g, left_point, right_point, eps=1e-9, delta=1e-6):
    """Golden-section segment search on the segment from the left_point to the right_point.
    https://en.wikipedia.org/wiki/Golden-section_search
    Arguments:
        g (function): one-dimensional function to optimize, g(x)
        left_point (float or Tensor): left point of the segment for search, a for [a,b]
        right_point (float or Tensor): right point of the segment for search, b for [a,b]
        eps (float): pointwise accuracy, |x_k - x^{*}| < eps (default: 1e-9)
        delta (float): functional accuracy, f(x_k) - f(x^{*}) < delta (default: 1e-6)
    """
    assert right_point > left_point
    right_point = check_left_point_(g, left_point, right_point, delta=1e-6)
    diff = right_point - left_point
    if diff < delta:
        return left_point

    fib = 0.5 + math.sqrt(5) / 2.

    x1 = right_point - diff / fib
    x2 = left_point + diff / fib
    g_1 = g(x1)
    g_2 = g(x2)
    iter = 0
    while right_point - left_point > eps and abs(g_1 - g_2) > delta:
        if g_1 > g_2:
            left_point = x1 + 0.
            x1 = x2 + 0.
            g_1 = g_2 + 0.
            x2 = left_point + (right_point - left_point) / fib
            g_2 = g(x2)
        else:
            right_point = x2 + 0.
            x2 = x1 + 0.
            g_2 = g_1 + 0.
            x1 = right_point - (right_point - left_point) / fib
            g_1 = g(x1)
        iter += 1
        if iter == 40:
            return None, ValueError("Bad solution by segment-search")

    return (right_point + left_point) / 2.


def segment_search_ternary(g, left_point, right_point, eps=1e-9, delta=1e-6):
    """Ternary segment search on the segment from the left_point to the right_point.
    https://en.wikipedia.org/wiki/Ternary_search
    Arguments:
        g (function): one-dimensional function to optimize, g(x)
        left_point (float or Tensor): left point of the segment for search, a for [a,b]
        right_point (float or Tensor): right point of the segment for search, b for [a,b]
        eps (float): pointwise accuracy, |x_k - x^{*}| < eps (default: 1e-9)
        delta (float): functional accuracy, f(x_k) - f(x^{*}) < delta (default: 1e-6)
    """
    assert right_point > left_point

    right_point = check_left_point_(g, left_point, right_point,  delta=1e-6)
    diff = right_point - left_point
    if diff < delta:
        return left_point

    x1 = (2 * left_point + right_point) / 3.
    g_1 = g(x1)
    x2 = (left_point + 2 * right_point) / 3.
    g_2 = g(x2)
    iter = 0
    while right_point - left_point > eps and abs(g_1 - g_2) > delta:
        if g_1 > g_2:
            left_point = x1 + 0.
        else:
            right_point = x2 + 0.
        x1 = (2 * left_point + right_point) / 3.
        x2 = (left_point + 2 * right_point) / 3.
        g_1 = g(x1)
        g_2 = g(x2)
        iter += 1
        if iter == 40:
            return None, ValueError("Bad solution by segment-search")

    return (right_point + left_point) / 2.


def ray_line_search(g, left_point, middle_point, eps=1e-8, delta=1e-6, segment='gold'):
    """Ray search on the ray from the left_point to the direction of the middle_point
    Arguments:
        g (function): one-dimensional function to optimize, g(x)
        left_point (float or Tensor): left point of the ray for search, a for [a,b,+inf)
        middle_point (float or Tensor): middle point of the ray for search, b for [a,b,+inf)
        eps (float): pointwise accuracy, |x_k - x^{*}| < eps (default: 1e-9)
        delta (float): functional accuracy, f(x_k) - f(x^{*}) < delta (default: 1e-6)
        segment (str): type of segment search after finding a segment with minimum, (default: 'gold')
    """
    assert middle_point > left_point

    if segment == 'ternary':
        segment_search = segment_search_ternary
    else:
        segment_search = segment_search_gold
    g_mid = g(middle_point)
    g_left = g(left_point)
    if g_mid >= g_left:
        right_point = middle_point + 0.
    else:
        right_point = middle_point * 2.
        g_right = g(right_point)
        iter = 0
        while g_mid >= g_right:
            iter += 1
            if g_mid - g_right < delta:
                return right_point
            left_point = middle_point + 0.
            middle_point = right_point + 0.
            g_mid = g_right + 0.
            right_point = middle_point * 2.
            g_right = g(right_point)
            if iter > 36:
                return None, ValueError(f"Function {g} is unbounded from below. There is no solution.")

    return segment_search(g=g, left_point=left_point, right_point=right_point, eps=eps, delta=delta)


def check_left_point_(g, left_point, right_point, delta=1e-6):
    """Auxilary procedure for checking that the solution of segment search is not at the left point
    """
    iter = 0
    x1 = (2 * left_point + right_point) / 3.
    g_left = g(left_point)
    g_1 = g(x1)
    while g_left <= g_1 + delta:
        right_point = x1 + 0.
        x1 = (2 * left_point + right_point) / 3.
        g_1 = g(x1)
        iter += 1
        if iter == 10:
            return left_point
    return right_point


