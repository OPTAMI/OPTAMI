#!/usr/bin/env python3

from argparse import ArgumentError
# from torchvision.transforms import ToTensor, Compose, Resize
# from torch.utils.data import TensorDataset, DataLoader
# from models_utils import LogisticRegression, train
# from torchvision.datasets import MNIST
from models_utils import minimize
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import normalize
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
parser.add_argument('epochs', nargs='?', default=20, type=int)
args = parser.parse_args()
DATASET = args.dataset
EPOCHS = args.epochs

L = os.environ("L")
mu = os.environ("mu")

# if DATASET == "mnist":
#     IMG_SIZE = 28
#     INPUT_SIZE = IMG_SIZE**2
#     DATASET_SIZE = 60000
#     train_loader = DataLoader(
#         dataset=MNIST(root='./data', train=True, download=True, 
#         transform=Compose([ToTensor(), Resize(IMG_SIZE), 
#         lambda x: x.double().view(IMG_SIZE**2)]),
#         target_transform=lambda y: y % 2),
#         batch_size=DATASET_SIZE, shuffle=False
#     )
# elif DATASET == "a9a":
#     INPUT_SIZE = 123
#     DATASET_SIZE = 32561
#     dataset = load_svmlight_file('./data/LibSVM/a9a.txt')
#     train_loader = DataLoader(
#         TensorDataset(torch.tensor(normalize(dataset[0].toarray(), norm='l2', axis=1)).double(), 
#         ((torch.tensor(dataset[1]) + 1)/2).long()), 
#         batch_size=DATASET_SIZE, shuffle=False
#     )
# else:
#     raise ArgumentError(f"dataset {DATASET} undefined")

# model = LogisticRegression(INPUT_SIZE, 2, gamma=mu).to(device)
# print(sum(p.numel() for p in model.parameters() if p.requires_grad))

# optimizers = {
#     'Cubic regularized Newton': OPTAMI.CubicRegularizedNewton(model.parameters(), L=L),
#     'Damped Newton': OPTAMI.DampedNewton(model.parameters(), alpha=5e-1, L=L),
#     'Globally regularized Newton': OPTAMI.GlobalNewton(model.parameters(), L=L),
#     'AIC Newton': OPTAMI.DampedNewton(model.parameters(), L=L, affine_invariant=True)
# }

if DATASET == "a9a":
    dataset = load_svmlight_file('./data/LibSVM/a9a.txt')
    x = torch.tensor(normalize(dataset[0].toarray(), norm='l2', axis=1), dtype=torch.double)
    y = torch.tensor(dataset[1], dtype=torch.double)
    INPUT_SIZE = x.size()[1]
else:
    raise ArgumentError(f"dataset {DATASET} undefined")

def logreg(w):
    return torch.nn.functional.soft_margin_loss(x.mv(w), y) + mu/2 * torch.norm(w, p=2)**2

w = torch.zeros(INPUT_SIZE).double().requires_grad_()
optimizers = {
    'Cubic regularized Newton': OPTAMI.CubicRegularizedNewton([w], L=L),
    'Damped Newton': OPTAMI.DampedNewton([w], alpha=L),
    'Globally regularized Newton': OPTAMI.GlobalNewton([w], L=L),
    'AIC Newton': OPTAMI.DampedNewton([w], L=L, affine_invariant=True)
}

LOG_PATH = f"logs_{DATASET}"
os.makedirs(LOG_PATH, exist_ok=True)
for name, optimizer in optimizers.items():
    losses, grads = minimize(logreg, w, optimizer, epochs=EPOCHS, verbose=True)
    with open(os.path.join(LOG_PATH, f"{name}_L={L}_mu={mu}.pkl"), "wb") as f:
        pickle.dump((losses, grads), f)
