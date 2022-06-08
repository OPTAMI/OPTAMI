#!/usr/bin/env python3

from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import normalize
from torchvision.datasets import MNIST
from models_utils import minimize
import argparse
import OPTAMI
import pickle
import torch
import os


SEED = 777
torch.manual_seed(SEED)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['a9a', 'mnist'])
parser.add_argument('epochs', nargs='?', default=100, type=int)
parser.add_argument('--perftest', default=False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()
DATASET = args.dataset
EPOCHS = args.epochs

if DATASET == 'a9a':
    mu = 1e-5

    dataset = load_svmlight_file('./data/LibSVM/a9a.txt')
    x = torch.tensor(normalize(dataset[0].toarray(), norm='l2', axis=1), dtype=torch.double)
    y = torch.tensor(dataset[1], dtype=torch.double)

    INPUT_SIZE = x.size()[1]
    initial_point = torch.zeros(INPUT_SIZE).double()
elif DATASET == 'mnist':
    img_size = 28

    dataset = MNIST(root='./data', train=True, download=True)
    x = dataset.data.reshape(-1, img_size**2).double()
    y = (dataset.targets == 0).int() * 2 - 1

    INPUT_SIZE = x.size()[1]
    initial_point = 1/(7*INPUT_SIZE) * torch.ones(INPUT_SIZE).double()

    mu = 1/INPUT_SIZE
    print("mu =", mu)
else:
    raise AttributeError(f"dataset {DATASET} undefined")

def logreg(w):
    return torch.nn.functional.soft_margin_loss(x.mv(w), y) + mu/2 * torch.norm(w, p=2)**2

LOG_PATH = f"logs_{DATASET}_?"
os.makedirs(LOG_PATH, exist_ok=True)

def all_methods():
    for classname in filter(lambda attr: attr[0].isupper(), dir(OPTAMI)):
        Algorithm = getattr(OPTAMI, classname)
        name = str(Algorithm).split('.')[-1][:-2]
        yield name, Algorithm, {"L": 0.5}

custom_list = [
    ("AIC Newton", OPTAMI.DampedNewton, {"affine_invariant": True, "L": 40}),
    ("Damped Newton", OPTAMI.DampedNewton, {"alpha": 0.05}),
    ("Globally Reg. Newton", OPTAMI.GlobalNewton, {"L": 450_000}),
    ("Cubic Newton", OPTAMI.CubicRegularizedNewton, {"L": 200_000})
]

methods_iterator = all_methods() if args.perftest else custom_list

for name, Algorithm, kwargs in methods_iterator:
    torch.manual_seed(777)
    w = initial_point.clone().requires_grad_()
    optimizer = Algorithm([w], verbose=False, **kwargs)

    print(name)
    times, losses, grads = minimize(logreg, w, optimizer, epochs=EPOCHS, verbose=True, tqdm_on=False)
    print()

    with open(os.path.join(LOG_PATH, f"{name}.pkl"), "wb") as f:
        pickle.dump((times, losses, grads), f)
