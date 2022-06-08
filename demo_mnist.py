#!/usr/bin/env python3

from torchvision.datasets import MNIST
from sklearn.metrics import f1_score
from functools import partial
import argparse
import OPTAMI
import pickle
import torch
import time
import os


SEED = 777
torch.manual_seed(SEED)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('epochs', nargs='?', default=10, type=int)
args = parser.parse_args()
EPOCHS = args.epochs

img_size = 28

dataset = MNIST(root='./data', train=True, download=True)
x = dataset.data.reshape(-1, img_size**2).double()
y = dataset.targets
ys = [(y == i).int() * 2 - 1 for i in range(10)]

INPUT_SIZE = x.size()[1]
initial_point = 1/(7*INPUT_SIZE) * torch.ones(INPUT_SIZE).double()

mu = 1/INPUT_SIZE
print("mu =", mu)

def logreg(w, y):
    return torch.nn.functional.soft_margin_loss(x.mv(w), y) + mu/2 * torch.norm(w, p=2)**2

LOG_PATH = f"logs_mnist_all_2"
os.makedirs(LOG_PATH, exist_ok=True)

custom_list = [
    # ("AIC Newton", OPTAMI.DampedNewton, {"affine_invariant": True, "L": 40}),
    # ("Damped Newton", OPTAMI.DampedNewton, {"alpha": 0.05}),
    ("Globally Reg. Newton", OPTAMI.GlobalNewton, {"L": 450_000}),
    ("Cubic Newton", OPTAMI.CubicRegularizedNewton, {"L": 200_000})
]

@torch.no_grad()
def f1(ws):
    prob = [torch.sigmoid(x.mv(ws[i])) for i in range(10)]
    pred = torch.argmax(torch.vstack(prob), dim=0)
    return f1_score(y, pred, average='weighted')

for name, Algorithm, kwargs in custom_list:
    torch.manual_seed(777)

    ws = [initial_point.clone().requires_grad_() for _ in range(10)]
    optimizers = [Algorithm([ws[i]], verbose=False, **kwargs) for i in range(10)]

    print(name)
    
    times = []
    losses = []
    f1_scores = []

    def closure(optimizer, w, y):
        optimizer.zero_grad()
        return logreg(w, y)

    closures = [partial(closure, optimizers[i], ws[i], ys[i]) for i in range(10)]

    loss = [closures[i]() for i in range(10)]
    for i in range(10):
        loss[i].backward()
        optimizers[i].zero_grad()

    times.append(0.)
    losses.append(sum([loss[i].item() for i in range(10)]) / 10)
    f1_scores.append(f1(ws))

    tic = time.time()

    for _ in range(EPOCHS):
        loss = [closures[i]() for i in range(10)]
        for i in range(10):
            loss[i].backward()

        f_val = sum([loss[i].item() for i in range(10)]) / 10
        f1_val = f1(ws)
        print("f =", f_val, "f1-score =", f1_val)

        times.append(time.time() - tic)
        losses.append(f_val)
        f1_scores.append(f1_val)

        for i in range(10):
            optimizers[i].step(closures[i])

    print()

    with open(os.path.join(LOG_PATH, f"{name}.pkl"), "wb") as f:
        pickle.dump((times, losses, f1_scores), f)
