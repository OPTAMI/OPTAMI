#!/usr/bin/env python3

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torch.autograd import Variable
import OPTAMI
import torch
import time
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def zero_all(model):
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
    return model


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim, gamma=0.):
        super().__init__()
        self.gamma = gamma
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

    def criterion(self, hypothesis, reference):
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(hypothesis, reference)

        if self.gamma > 0.:
            for param in model.parameters():
                loss += param.square().sum().mul(self.gamma)
        return loss


for classname in filter(lambda attr: attr[0].isupper(), dir(OPTAMI)):
    Algorithm = getattr(OPTAMI, classname)
    torch.manual_seed(777)


    INPUT_DIM = 784
    OUTPUT_DIM = 2

    TIME_LIMIT = 30
    L = 4.0
    F_STAR_PLUS_EPSILON = 0.3
    F_REASONABLE = 0.4

    failed_counter = 0

    train_loader = DataLoader(dataset=MNIST(root='./data', train=True, transform=ToTensor(), download=True),
                            batch_size=5000, shuffle=False)

    model = zero_all(LogisticRegression(INPUT_DIM, OUTPUT_DIM))
    optimizer = Algorithm(model.parameters(), L=L, verbose=False)

    tic = toc = time.time()
    losses = []

    while toc - tic < TIME_LIMIT:
        for i, (images, labels) in enumerate(train_loader):
            def closure(backward=False):
                prediction = model(image)
                loss = model.criterion(prediction, label)
                optimizer.zero_grad()
                if backward:
                    loss.backward()
                return loss

            image = Variable(images.view(-1, 28 ** 2))
            label = Variable(labels).fmod(2)

            loss = closure().item()
            losses.append(loss)

            optimizer.step(closure)
            toc = time.time()

            if toc - tic > TIME_LIMIT:
                break

    if Algorithm.MONOTONE:
        print(f"test_monotonicity ({classname}) ... ", end="")
        print(losses)
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
