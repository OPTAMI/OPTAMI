import json
import torch
import OPTAMI as opt



def f_pow_4(x):
    return x.square().square().sum()



def test_steps():
    methods = {
        'Hyperfast': opt.Hyperfast,
        'Superfast': opt.Superfast,
        'BDGM': opt.BDGM,
        'Cubic_Newton': opt.Cubic_Newton
    }

    problems = {'f_pow_4': f_pow_4}

    with open('test_methods.json', 'r') as re:
        tests = json.loads(re.read())

    with open('test_problems.json', 'r') as re:
        problem_setup = json.loads(re.read())

    for problem in problem_setup:
        for test in tests:
            method = methods[test['algorithm']]
            for outer_setup in test['outer_setup']:
                f = problems[problem["test_problem"]]
                x = torch.tensor(problem["test_starting_point"]).requires_grad_()
                optimizer = method([x], problem["L"], **outer_setup["config"])

                def closure():
                    optimizer.zero_grad()
                    return f(x)

                for i in range(outer_setup["algorithms_iteration"]):
                    optimizer.step(closure)
                assert (closure() - problem["test_func_min"]) < \
                       min(problem["problem_precision"], outer_setup["algorithms_precision"])
