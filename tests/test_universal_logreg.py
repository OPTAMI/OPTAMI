#!/usr/bin/env python3

import torch
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import normalize
import warnings
import sys
sys.path.append("./")
import OPTAMI
from OPTAMI.utils.fit import func_fit
torch.set_default_dtype(torch.float64)

def test_universal_logreg():

    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    DATASET = "a9a"
    f_star = 0.3361787035767108


    EPSILON = 0.1
    EPSILON_MAX = 0.2

    L1 = 0.25
    L2 = 0.1
    L3 = 0.1
    mu = 1e-4
    epochs = 25

    if DATASET != "a9a":
        raise AttributeError(f"dataset {DATASET} undefined")

    #dataset = load_svmlight_file(f'./data/LibSVM/{DATASET}.txt') # for git
    dataset = load_svmlight_file(f'../data/LibSVM/{DATASET}.txt')
    x = torch.tensor(normalize(dataset[0].toarray(), norm='l2', axis=1))
    y = torch.tensor(dataset[1])
    INPUT_SIZE = x.size()[1]

    for classname in filter(lambda attr: attr[0].isupper(), dir(OPTAMI)):
        Algorithm = getattr(OPTAMI, classname)
        try:
            if Algorithm.SKIP_TEST_LOGREG:
                return True
        except AttributeError:
            pass

        torch.manual_seed(42)

        def logreg(w):
            return torch.nn.functional.soft_margin_loss(x @ w, y) + mu/2 * w.square().sum()
        w0 = torch.ones(INPUT_SIZE).mul_(1)

        try:
            if Algorithm.ACCELERATION:
                for p in [2,3]:
                    Algorithm.ORDER = p
                    w = w0.requires_grad_()
                    L, epochs_ = L_from_order(Algorithm, L1, L2, L3, epochs)
                    optimizer = Algorithm([w], L=L, order=p, verbose=False, testing=True)
                    print(f'Algorithm {Algorithm} with order {p} and epochs={epochs_}')
                    run_learn(Algorithm, optimizer, epochs_, logreg, w0, classname, f_star, EPSILON, EPSILON_MAX)
            else:
                L, epochs_ = L_from_order(Algorithm, L1, L2, L3, epochs)
                epochs_ *= 2
                w = w0.requires_grad_()
                optimizer = Algorithm([w], L=L, verbose=False, testing=True)
                print(f'Algorithm {Algorithm} with epochs={epochs_}')
                run_learn(Algorithm, optimizer, epochs_, logreg, w0, classname, f_star, EPSILON, EPSILON_MAX)
        except AttributeError:
            L,epochs_ = L_from_order(Algorithm, L1,L2,L3,epochs)
            epochs_ *= 2
            w = w0.requires_grad_()
            optimizer = Algorithm([w], L=L, verbose=False, testing=True)
            print(f'Algorithm {Algorithm} with epochs={epochs_}')
            run_learn(Algorithm, optimizer, epochs_, logreg, w0, classname, f_star, EPSILON, EPSILON_MAX)


def L_from_order(Algorithm, L1,L2,L3,epochs):
        try:
            if Algorithm.ORDER == 1:
                return L1, 5 * epochs
            elif Algorithm.ORDER == 2:
                return L2, epochs
            elif Algorithm.ORDER == 3:
                return L3, epochs
            else:
                raise AttributeError(f"algorithm {Algorithm} not supported")
        except AttributeError:
            return L1, epochs

def run_learn(Algorithm, optimizer, EPOCHS, func, params, classname, f_star, EPSILON, EPSILON_MAX):
    failed_counter = 0
    name = str(Algorithm).split('.')[-1][:-2]

    print(name)
    losses, times, grads = func_fit(optimizer, EPOCHS, func, params)
    print()

    if Algorithm.MONOTONE:
        print(f"test_monotonicity ({classname}) ... ", end="")
        if all(x >= y for x, y in zip(losses, losses[1:])):
            print("ok")
        else:
            print("FAIL")
            failed_counter += 1

    print(f"test_obtained_solution ({classname}) ... ", end="")
    if losses[-1] < f_star + EPSILON:
        print("ok")
    else:
        print("FAIL")
        failed_counter += 1

    print(f"test_divergence ({classname}) ... ", end="")
    try:
        if next(filter(lambda i: all(x > f_star + EPSILON_MAX for x in losses[i:]), range(len(losses)))) / len(
                losses) > 0.9:
            print("ok")
        else:
            print("FAIL")
            print(next(filter(lambda i: all(x > f_star + EPSILON_MAX for x in losses[i:]), range(len(losses)))) / len(
                losses))
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
