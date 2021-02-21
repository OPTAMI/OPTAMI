import json
import torch
import OPTAMI as opt



def f_pow_4(x):
    return x.square().square().sum()



def test_steps():
    methods = {
        'Hyperfast': opt.Hyperfast,
        'BDGM': opt.BDGM
    }

    problems = {'f_pow_4': f_pow_4}

    with open('test_methods.json', 'r') as re:
        tests = json.loads(re.read())

    with open('test_problems.json', 'r') as re:
        problem_setup = json.loads(re.read())

    for problem in problem_setup:
        for test in tests:
            method = methods[test['algorithm']]
            for config in test['config']:
                f = problems[problem["test_problem"]]
                x = torch.tensor(problem["test_starting_point"]).requires_grad_()
                optimizer = method([x], problem["L"], **config)

                def closure():
                    optimizer.zero_grad()
                    return f(x)

                for i in range(config["algorithms_iteration"]):
                    optimizer.step(closure)
                assert (closure() - problem["test_func_min"]) < \
                       min(problem["problem_precision"], config["algorithms_precision"])
