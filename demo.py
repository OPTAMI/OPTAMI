#!/usr/bin/env python3

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize
from torchvision.datasets import MNIST
import OPTAMI
import torch
import time


def zero_all(model):
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
    return model


def train(model, optimizer, dataloader, epochs=10, verbose=True, return_grads=False):
    zero_all(model)

    tic = time.time()
    losses = []
    grads = []

    for _ in range(epochs):
        for i, (images, labels) in enumerate(dataloader):
            if i != 0:
                continue

            def closure():
                optimizer.zero_grad()
                prediction = model(image)
                return model.criterion(prediction, label)

            image = images.view(-1, 28 ** 2)
            label = labels.fmod(2)

            loss = model.criterion(model(image), label)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            grad_norm = 0.
            for p in model.parameters():
                if p.grad is None:
                    continue
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item()**2
            grads.append(grad_norm**0.5)
            optimizer.zero_grad()

            if verbose:
                print(f'loss = {loss.item()}')

            optimizer.step(closure)

    toc = time.time()
    if return_grads:
        return losses, grads, toc - tic
    return losses, toc - tic


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim, gamma=0.):
        super().__init__()
        self.gamma = gamma
        self.linear = torch.nn.Linear(input_dim, output_dim).double()

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

    def pure_loss(self, hypothesis, reference):
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(hypothesis, reference)


BATCH_SIZE = 1000
IMG_SIZE = 15
INPUT_DIM = IMG_SIZE**2
OUTPUT_DIM = 2
EPOCHS = 21
PLOT = True
NORMALIZE_PLOT = False
SEED = 777
torch.manual_seed(SEED)

train_loader = DataLoader(dataset=MNIST(root='./data', train=True, download=True, 
                          transform=Compose([ToTensor(), Resize(IMG_SIZE), lambda x: x.double()])),
                          batch_size=BATCH_SIZE, shuffle=False)

model = zero_all(LogisticRegression(INPUT_DIM, OUTPUT_DIM, gamma=1e-2))
L = 4.0

optimizers = {
    # 'Hyperfast': OPTAMI.Hyperfast(model.parameters(), L=L),
    # 'Superfast': OPTAMI.Superfast(model.parameters(), L=L),
    'Cubic Regularized Newton': OPTAMI.CubicRegularizedNewton(model.parameters(), L=L),
    'Damped Newton': OPTAMI.DampedNewton(model.parameters(), alpha=5e-1, L=L),
    'Affine Invariant Newton': OPTAMI.DampedNewton(model.parameters(), L=L, affine_invariant=True),
    # 'Hyperfast accelerated': OPTAMI.Hyperfast(model.parameters(), L=L, TensorStepMethod=OPTAMI.DampedNewton, tensor_step_kwargs={'affine_invariant': True, 'alpha': 1e-1}),
    # 'Superfast accelerated': OPTAMI.Superfast(model.parameters(), L=L, TensorStepMethod=OPTAMI.DampedNewton, tensor_step_kwargs={'affine_invariant': True, 'alpha': 1e-1}),
    # 'SimilarTriangles': OPTAMI.SimilarTriangles(model.parameters(), L=L, is_adaptive=True)
}

f_star = 0.
divider = 1.

f, axs = plt.subplots(1, 2, figsize=(10, 4))

markers = ["o", "v", "d"]

for i, (name, optimizer) in enumerate(optimizers.items()):
    losses, grads, working_time = train(model, optimizer, train_loader, epochs=EPOCHS, return_grads=True)

    if i == 0:
        f_star = min(losses)
        divider = losses[0] - min(losses)

    if PLOT:
        if NORMALIZE_PLOT:
            axs[0].plot(torch.tensor(losses).sub(f_star).div(divider), label=name)
        else:
            axs[0].plot(losses, label=name, marker=markers[i], markevery=3)
        axs[1].semilogy(grads, label=name, marker=markers[i], markevery=3)
    else:
        print(name, losses, grads, working_time, sep='\n', end='\n\n')

if PLOT:
    axs[0].set_xlabel("T")
    axs[0].set_ylabel("$F(x_T)$")
    axs[0].grid(alpha=0.4)
    axs[0].legend()

    axs[1].set_xlabel("T")
    axs[1].set_ylabel("$||\\nabla F(x_T)||$")
    axs[1].grid(alpha=0.4)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("figure.pdf")
    plt.show()
