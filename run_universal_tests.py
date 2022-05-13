#!/usr/bin/env python3

from torchvision.transforms import ToTensor, Compose, Resize
from models_utils import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import warnings
import OPTAMI
import torch
import time
import sys


if not sys.warnoptions:
    warnings.simplefilter("ignore")

for classname in filter(lambda attr: attr[0].isupper(), dir(OPTAMI)):
    Algorithm = getattr(OPTAMI, classname)
    torch.manual_seed(777)

    IMG_SIZE = 28
    INPUT_DIM = IMG_SIZE**2
    OUTPUT_DIM = 2

    TIME_LIMIT = 100
    L = 4.0
    F_STAR_PLUS_EPSILON = 0.15
    F_REASONABLE = 0.25

    failed_counter = 0

    train_loader = DataLoader(
        dataset=MNIST(root='./data', train=True, download=True, 
        transform=Compose([ToTensor(), Resize(IMG_SIZE), lambda x: x.double().view(IMG_SIZE**2)]),
        target_transform=lambda y: y % 2),
        batch_size=100, shuffle=False)

    model = LogisticRegression(INPUT_DIM, OUTPUT_DIM)
    optimizer = Algorithm(model.parameters(), L=L, verbose=False)

    tic = toc = time.time()
    losses = []

    while toc - tic < TIME_LIMIT:
        for i, (image, label) in enumerate(train_loader):
            if i != 0:
                continue

            def closure():
                optimizer.zero_grad()
                prediction = model(image)
                return model.criterion(prediction, label)

            loss = closure().item()
            losses.append(loss)

            optimizer.step(closure)
            toc = time.time()

            if toc - tic > TIME_LIMIT:
                break

    print(losses)
    if Algorithm.MONOTONE:
        print(f"test_monotonicity ({classname}) ... ", end="")
        if all(x >= y for x, y in zip(losses, losses[1:])):
            print("ok")
        else:
            print("FAIL")
            failed_counter += 1
    
    print(f"test_obtained_solution ({classname}) ... ", end="")
    if losses[-1] < F_STAR_PLUS_EPSILON:
        print("ok")
    else:
        print("FAIL")
        failed_counter += 1

    print(f"test_divergence ({classname}) ... ", end="")
    try:
        if next(filter(lambda i: all(x > F_REASONABLE for x in losses[i:]), range(len(losses)))) / len(losses) > 0.9:
            print("ok")
        else:
            print("FAIL")
            print(next(filter(lambda i: all(x > F_REASONABLE for x in losses[i:]), range(len(losses)))) / len(losses))
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
