#!/usr/bin/env python3

from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import normalize
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
parser.add_argument('dataset', choices=['a9a'])
parser.add_argument('epochs', nargs='?', default=25, type=int)
args = parser.parse_args()
DATASET = args.dataset
EPOCHS = args.epochs

# L = os.environ("L")
# mu = os.environ("mu")
L = 0.5
mu = 1e-5

if DATASET == "a9a":
    dataset = load_svmlight_file('./data/LibSVM/a9a.txt')
    x = torch.tensor(normalize(dataset[0].toarray(), norm='l2', axis=1), dtype=torch.double)
    y = torch.tensor(dataset[1], dtype=torch.double)
    INPUT_SIZE = x.size()[1]
else:
    raise AttributeError(f"dataset {DATASET} undefined")

def logreg(w):
    return torch.nn.functional.soft_margin_loss(x.mv(w), y) + mu/2 * torch.norm(w, p=2)**2

LOG_PATH = f"logs_{DATASET}"
os.makedirs(LOG_PATH, exist_ok=True)

for classname in filter(lambda attr: attr[0].isupper(), dir(OPTAMI)):
    Algorithm = getattr(OPTAMI, classname)
    name = str(Algorithm).split('.')[-1][:-2]

    torch.manual_seed(777)
    w = torch.zeros(INPUT_SIZE).double().requires_grad_()
    optimizer = Algorithm([w], L=L, verbose=False)

    print(name)
    times, losses, grads = minimize(logreg, w, optimizer, epochs=EPOCHS, verbose=True, tqdm_on=False)
    print()

    with open(os.path.join(LOG_PATH, f"{name}.pkl"), "wb") as f:
        pickle.dump((times, losses, grads), f)
