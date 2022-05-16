#!/usr/bin/env python3

from argparse import ArgumentError
from torchvision.transforms import ToTensor, Compose, Resize
from torch.utils.data import TensorDataset, DataLoader
from models_utils import LogisticRegression, train
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import normalize
from torchvision.datasets import MNIST
import argparse
import OPTAMI
import pickle
import torch
import os


SEED = 777
torch.manual_seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['a9a', 'mnist'])
parser.add_argument('epochs', nargs='?', default=25, type=int)
args = parser.parse_args()
DATASET = args.dataset
EPOCHS = args.epochs

if DATASET == "mnist":
    IMG_SIZE = 28
    INPUT_SIZE = IMG_SIZE**2
    DATASET_SIZE = 60000
    train_loader = DataLoader(
        dataset=MNIST(root='./data', train=True, download=True, 
        transform=Compose([ToTensor(), Resize(IMG_SIZE), 
        lambda x: x.double().view(IMG_SIZE**2)]),
        target_transform=lambda y: y % 2),
        batch_size=DATASET_SIZE, shuffle=False
    )
elif DATASET == "a9a":
    INPUT_SIZE = 123
    DATASET_SIZE = 32561
    dataset = load_svmlight_file('./data/LibSVM/a9a.txt')
    train_loader = DataLoader(
        TensorDataset(torch.tensor(normalize(dataset[0].toarray(), norm='l2', axis=1)).double(), 
        ((torch.tensor(dataset[1]) + 1)/2).long()), 
        batch_size=DATASET_SIZE, shuffle=False
    )
else:
    raise ArgumentError(f"dataset {DATASET} undefined")

model = LogisticRegression(INPUT_SIZE, 2, gamma=1e-3).to(device)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
L = 4.0

optimizers = {
    'Cubic regularized Newton': OPTAMI.CubicRegularizedNewton(model.parameters(), L=L),
    'Damped Newton': OPTAMI.DampedNewton(model.parameters(), alpha=5e-1, L=L),
    'Globally regularized Newton': OPTAMI.GlobalNewton(model.parameters(), L=L),
    'AI Newton': OPTAMI.DampedNewton(model.parameters(), L=L, affine_invariant=True),
    # 'Hyperfast accelerated': OPTAMI.Hyperfast(model.parameters(), L=L, TensorStepMethod=OPTAMI.DampedNewton, tensor_step_kwargs={'affine_invariant': True, 'alpha': 1e-1}),
    # 'Superfast accelerated': OPTAMI.Superfast(model.parameters(), L=L, TensorStepMethod=OPTAMI.DampedNewton, tensor_step_kwargs={'affine_invariant': True, 'alpha': 1e-1})
}

LOG_PATH = f"logs_{DATASET}"
os.makedirs(LOG_PATH, exist_ok=True)
for name, optimizer in optimizers.items():
    losses, grads, working_time = train(model, optimizer, train_loader, device, epochs=EPOCHS, return_grads=True)
    with open(os.path.join(LOG_PATH, name + ".pkl"), "wb") as f:
        pickle.dump((losses, grads), f)
