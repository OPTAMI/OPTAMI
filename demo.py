#!/usr/bin/env python3

from torchvision.transforms import ToTensor, Compose, Resize
from models_utils import LogisticRegression, train
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import OPTAMI
import pickle
import torch
import os


SEED = 777
torch.manual_seed(SEED)

IMG_SIZE = 28
DATASET_SIZE = 1000
train_loader = DataLoader(dataset=MNIST(root='./data', train=True, download=True, 
                          transform=Compose([ToTensor(), Resize(IMG_SIZE), lambda x: x.double()])),
                          batch_size=DATASET_SIZE, shuffle=False)
model = LogisticRegression(IMG_SIZE**2, 2, gamma=1e-2)

L = 4.0
optimizers = {
    'Cubic regularized Newton': OPTAMI.CubicRegularizedNewton(model.parameters(), L=L),
    'Damped Newton': OPTAMI.DampedNewton(model.parameters(), alpha=5e-1, L=L),
    'Globally regularized Newton': OPTAMI.GlobalNewton(model.parameters(), L=L),
    'AI Newton': OPTAMI.DampedNewton(model.parameters(), L=L, affine_invariant=True),
    # 'Hyperfast accelerated': OPTAMI.Hyperfast(model.parameters(), L=L, TensorStepMethod=OPTAMI.DampedNewton, tensor_step_kwargs={'affine_invariant': True, 'alpha': 1e-1}),
    # 'Superfast accelerated': OPTAMI.Superfast(model.parameters(), L=L, TensorStepMethod=OPTAMI.DampedNewton, tensor_step_kwargs={'affine_invariant': True, 'alpha': 1e-1})
}

EPOCHS = 4
LOG_PATH = "logs"
os.makedirs(LOG_PATH, exist_ok=True)
for name, optimizer in optimizers.items():
    losses, grads, working_time = train(model, optimizer, train_loader, IMG_SIZE, epochs=EPOCHS, return_grads=True)
    with open(os.path.join(LOG_PATH, name + ".pkl"), "wb") as f:
        pickle.dump((losses, grads), f)
