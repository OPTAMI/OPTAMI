import json
import torch
import OPTAMI as opt
from numpy import ndim
import sys
sys.path.append("./")
torch.set_default_dtype(torch.float64)

def f_small_pow_4(x):
    return x.square().square().sum()


def nesterov_lower_3(x):
    d = 10
    A = torch.eye(d)
    for i in range(d - 1):
        A[i][i + 1] = -1.
    return A.mv(x).square().square().sum().div(4) - x[0]


def test_steps():
    methods = {
        'Optimal': opt.Optimal,
        'ProxPointSegmentSearch': opt.ProxPointSegmentSearch,
        'NATA': opt.NATA,
        'NearOptimalAcceleration': opt.NearOptimalAcceleration,
        'NesterovAcceleration': opt.NesterovAcceleration,
        'BasicTensorMethod': opt.BasicTensorMethod
    }

    problems = {'f_small_pow_4': f_small_pow_4, 'nesterov_lower_3': nesterov_lower_3}

    #with open('./tests/third_order/test_problems_third.json', 'r') as re: #for git
    with open('./third_order/test_problems_third.json', 'r') as re:
        problem_setup = json.loads(re.read())

    #with open('./tests/third_order/test_methods_third.json', 'r') as re: #for git
    with open('./third_order/test_methods_third.json', 'r') as re:
        tests = json.loads(re.read())

    for problem in problem_setup:
        for test in tests:
            method = methods[test['algorithm']]
            for outer_setup in test['outer_setup']:
                f = problems[problem["test_problem"]]
                # To define x by generator "test_starting_point" should be with two dimensions.
                # The first dimension is a vector dimension. The second dimension is a value
                if ndim(problem["test_starting_point"]) == 2:
                    x = torch.ones(problem["test_starting_point"][0][0])
                    x = x.mul(problem["test_starting_point"][1][0])
                    x.requires_grad_()
                else:
                    x = torch.tensor(problem["test_starting_point"]).requires_grad_()
                optimizer = method([x], problem["L"], **outer_setup["config"], testing=True)
                precision = max(problem["problem_precision"], outer_setup["algorithms_precision"])
                iteration = problem["problem_iteration"] * outer_setup["algorithms_iteration_mul"]
                min_solution = 1000000000.

                def closure():
                    optimizer.zero_grad()
                    return f(x)

                i = 0
                while i < iteration:
                    optimizer.step(closure)
                    loss = closure().item()
                    if loss < min_solution:
                        min_solution = loss
                    if loss - problem["test_func_min"] < precision:
                        i = iteration
                    i += 1

                assert (min_solution - problem["test_func_min"]) < precision
