#!/usr/bin/env python3

from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import normalize
from models_utils import minimize
import warnings
import OPTAMI
import torch
import sys


if not sys.warnoptions:
    warnings.simplefilter("ignore")

DATASET = "a9a"
F_STAR = 0.32501597692415846

EPOCHS = 25
EPSILON = 0.08
EPSILON_MAX = 0.2

L = 0.5
mu = 1e-5

if DATASET == "a9a":
    dataset = load_svmlight_file('./data/LibSVM/a9a.txt')
    x = torch.tensor(normalize(dataset[0].toarray(), norm='l2', axis=1), dtype=torch.double)
    y = torch.tensor(dataset[1], dtype=torch.double)
    INPUT_SIZE = x.size()[1]
else:
    raise AttributeError(f"dataset {DATASET} undefined")

for classname in filter(lambda attr: attr[0].isupper(), dir(OPTAMI)):
    Algorithm = getattr(OPTAMI, classname)
    torch.manual_seed(777)

    failed_counter = 0

    def logreg(w):
        return torch.nn.functional.soft_margin_loss(x.mv(w), y) + mu/2 * torch.norm(w, p=2)**2

    w = torch.zeros(INPUT_SIZE).double().requires_grad_()
    optimizer = Algorithm([w], L=L, verbose=False)
    name = str(Algorithm).split('.')[-1][:-2]

    print(name)
    times, losses, grads = minimize(logreg, w, optimizer, epochs=EPOCHS, verbose=True, tqdm_on=False)
    print()

    # tic = toc = time.time()
    # losses = []

    # while toc - tic < TIME_LIMIT:
    #     def closure():
    #         optimizer.zero_grad()
    #         return logreg(w)

    #     loss = closure()
    #     f_val = loss.item()
    #     print(f_val)
    #     losses.append(f_val)

    #     loss.backward()
    #     optimizer.step(closure)
    #     toc = time.time()

    if Algorithm.MONOTONE:
        print(f"test_monotonicity ({classname}) ... ", end="")
        if all(x >= y for x, y in zip(losses, losses[1:])):
            print("ok")
        else:
            print("FAIL")
            failed_counter += 1
    
    print(f"test_obtained_solution ({classname}) ... ", end="")
    if losses[-1] < F_STAR + EPSILON:
        print("ok")
    else:
        print("FAIL")
        failed_counter += 1

    print(f"test_divergence ({classname}) ... ", end="")
    try:
        if next(filter(lambda i: all(x > F_STAR + EPSILON_MAX for x in losses[i:]), range(len(losses)))) / len(losses) > 0.9:
            print("ok")
        else:
            print("FAIL")
            print(next(filter(lambda i: all(x > F_STAR + EPSILON_MAX for x in losses[i:]), range(len(losses)))) / len(losses))
            failed_counter += 1
    except StopIteration:
        print("ok")

    print(f"test_infinities ({classname}) ... ", end="")
    if not any(x == float('+inf') or x == float('-inf') for x in losses):
        print("ok")
    else:
        print("FAIL")
        failed_counter += 1

    print(f"test_none ({classname}) ... ", end="")
    if not any(x is None for x in losses):
        print("ok")
    else:
        print("FAIL")
        failed_counter += 1

    print("\n----------------------------------------------------------------------\n")
    print(f"OK" if failed_counter == 0 else f"FAILED (failures={failed_counter})")

    if failed_counter > 0:
        raise Exception(f"Universal tests failed with {failed_counter} failures")
