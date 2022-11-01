
import torch
import time


def timer_2(func, a, y):
    start_time = time.time()
    res = func(a, y)
    end_time = time.time() - start_time
    return res, end_time


def test_LogSigmoid_vs_SoftMargin():
    # For big dimensions (n>1e+7) SoftMargin almost always faster than log(sigmoid))
    for i in range(2):
        n = 100000000
        a = torch.randn(n).mul_(100.)
        y = torch.randint(0, 2, [n]).mul(2.).sub(1.)

        def func_logreg(a, y):
            return torch.mean(- torch.log(torch.sigmoid(a.mul(y))))

        res_sm, time_sm = timer_2(torch.nn.functional.soft_margin_loss, a, y)
        res_fun, time_fun = timer_2(func_logreg, a, y)
        assert time_sm - time_fun < 0.
