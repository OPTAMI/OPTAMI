#!/usr/bin/env python3

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torch.autograd import Variable
import OPTAMI
import torch
import time


def zero_all(model):
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
    return model


def train(model, optimizer, dataloader, epochs=10, verbose=True):
    zero_all(model)

    tic = time.time()
    losses = []

    for _ in range(epochs):
        for i, (images, labels) in enumerate(dataloader):
            if i != 0:
                continue

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

            if verbose:
                print(f'loss = {loss}')

            optimizer.step(closure)

    toc = time.time()
    return losses, toc - tic


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


BATCH_SIZE = 5000
INPUT_DIM = 784
OUTPUT_DIM = 2
EPOCHS = 7
PLOT = True
NORMALIZE_PLOT = True
SEED = 777
torch.manual_seed(SEED)

train_loader = DataLoader(dataset=MNIST(root='./data', train=True, transform=ToTensor(), download=True),
                          batch_size=BATCH_SIZE, shuffle=False)

model = zero_all(LogisticRegression(INPUT_DIM, OUTPUT_DIM))
L = 4.0

optimizers = {
    'Superfast': OPTAMI.Superfast(model.parameters(), L=L, subsolver=torch.optim.Adam, max_iters=100)#,
    # 'CubicRegularizedNewton': OPTAMI.CubicRegularizedNewton(model.parameters(), L=L, subsolver=torch.optim.Adam),
    # 'SimilarTriangles': OPTAMI.SimilarTriangles(model.parameters(), L=L, is_adaptive=True)
}

if PLOT:
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 16})

f_star = 0.
divider = 1.
for i, (name, optimizer) in enumerate(optimizers.items()):
    losses, working_time = train(model, optimizer, train_loader, epochs=EPOCHS)

    if i == 0:
        f_star = min(losses)
        divider = losses[0] - min(losses)

    if PLOT:
        if NORMALIZE_PLOT:
            plt.plot(torch.tensor(losses).sub(f_star).div(divider), label=name)
        else:
            plt.plot(losses, label=name)
    else:
        print(name, losses, working_time, sep='\n', end='\n\n')

if PLOT:
    plt.yscale('log')

    plt.title('Logistic Regression for MNIST')
    plt.xlabel('Iterations')
    plt.ylabel('$\\log\\;\\frac{f(x_i)-f(x_*)}{f(x_0)-f(x_*)}$')

    plt.legend()
    plt.tight_layout()
    plt.show()
