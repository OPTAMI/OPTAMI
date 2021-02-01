import torch
import math


def func_compute(closure, params, first_point, second_point, theta_search):
    with torch.no_grad():
        for i in range(len(list(params))):
            list(params)[i].zero_().add_(first_point[i].mul(1. - theta_search)).add_(second_point[i].mul(theta_search))
    loss = closure()
    return loss

def func_compute_ray(closure, params, first_point, second_point, theta_search):
    with torch.no_grad():
        for i in range(len(list(params))):
            list(params)[i].zero_().add_(first_point[i].sub(second_point[i].mul(theta_search)))
    loss = closure()
    return loss

def point_search(closure, params, first_point, second_point, eps=1e-10):
    save_point = []
    with torch.no_grad():
        for i in range(len(list(params))):
            save_point.append(list(params)[i])
    func = lambda theta: func_compute(closure, params, first_point, second_point, theta)
    proportion = segment_line_search_fib(func)
    xk_tilde = []
    with torch.no_grad():
        for i in range(len(list(params))):
            list(params)[i].zero_().add_(save_point[i])     # return point back
            xk_tilde.append(first_point[i].mul(1. - proportion).add(second_point[i].mul(proportion)))
            
    return xk_tilde


def point_search_ray(closure, params, first_point, second_point, eps=1e-10):
    save_point = []
    with torch.no_grad():
        for i in range(len(list(params))):
            save_point.append(list(params)[i])
    func = lambda theta: func_compute_ray(closure, params, first_point, second_point, theta)
    proportion = ray_line_search(func)
    xk_tilde = []
    with torch.no_grad():
        for i in range(len(list(params))):
            list(params)[i].zero_().add_(save_point[i])  # return point back
            xk_tilde.append(first_point[i].sub(second_point[i].mul(proportion)))
    
    return xk_tilde

def segment_line_search_fib(g, left_point = 0., right_point = 1., eps=1e-10):
    fib = 0.5 + math.sqrt(5) / 2.

    diff = right_point - left_point
    x1 = right_point - diff / fib
    x2 = left_point + diff / fib
    g1 = g(x1)
    g2 = g(x2)
    while diff > eps:
        # print(left_point, x1, x2, right_point)
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
    #left_point = left_point.to(torch.double)
    #right_point = right_point.to(torch.double)
    #x1 = right_point.add(left_point, alpha=2.).div_(3.)
    x1 = (right_point + 2. * left_point)/3.
    #x2 = left_point.add(right_point, alpha=2.).div_(3.)
    x2 = (left_point + 2. * right_point) / 3.
    #while right_point.sub(left_point).norm().ge(eps).item():
    while abs(right_point - left_point) > eps:
        if g(x1).sub(g(x2)).ge(0.).item():
            #left_point = x1.clone()
            left_point = x1
            #x1 = right_point.add(left_point, alpha=2.).div_(3.)
            x1 = (right_point + 2. * left_point)/3.
            #x2 = left_point.add(right_point, alpha=2.).div_(3.)
            x2 = (left_point + 2. * right_point) / 3.
        else:
            #right_point = x2.clone()
            right_point = x2
            #x1 = right_point.add(left_point, alpha=2.).div_(3.)
            x1 = (right_point + 2. * left_point)/3.
            #x2 = left_point.add(right_point, alpha=2.).div_(3.)
            x2 = (left_point + 2. * right_point) / 3.
    #return right_point.add(left_point).div(2.)
    return (right_point + left_point) / 2.


def ray_line_search(g, middle_point = 1., left_point = 0., eps=1e-10):
    #if g(middle_point).sub(g(left_point)).ge(0.).item():
    if ((g(middle_point)- g(left_point)) >= 0.):
        right_point = middle_point
    else:
        #right_point = middle_point.mul(2.).sub(left_point)
        right_point = 2. * middle_point - left_point
        #while g(middle_point).sub(g(right_point)).ge(0.).item():
        while (g(middle_point)-g(right_point))>=0:
            #left_point = middle_point.clone()
            left_point = middle_point
            #middle_point = right_point.clone()
            middle_point = right_point
            #right_point = middle_point.mul(2.).sub(left_point)
            right_point = 2. * middle_point - left_point
    return segment_line_search_tri(g, left_point, right_point, eps)
    
