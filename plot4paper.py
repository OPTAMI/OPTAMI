#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['a9a', 'mnist', 'mnist_all'])
args = parser.parse_args()
DATASET = args.dataset

if DATASET == "mnist":
    F_START = 4.1581224710762
    F_STAR  = 0.021645584427361638
elif DATASET == "mnist_all":
    F_START = 4.305707573420709
else:
    F_START = 0.6931471805599453
    F_STAR  = 0.32501597692415846

LOG_PATH = f"logs_{DATASET}"

styles = ['r', 'b--', 'g:', 'c-.']

for i, file in enumerate(sorted(os.listdir(LOG_PATH))):
    with open(os.path.join(LOG_PATH, file), "rb") as f:
        times, losses, grads = pickle.load(f)
    name = ".".join(file.split(".")[:-1])
    plt.plot(np.array(losses)[1:], styles[i], label=name)
    print(name, losses[-1])

plt.xlabel("Iterations, k")
plt.ylabel("$f(x_k)$")
plt.ylim(None, losses[0] + (losses[0] - losses[-1])*0.1)
plt.legend()
plt.grid(alpha=0.4)
plt.tight_layout()
# plt.savefig(f"{DATASET}_f.eps")
plt.show()

if DATASET != 'mnist_all':
    for i, file in enumerate(sorted(os.listdir(LOG_PATH))):
        with open(os.path.join(LOG_PATH, file), "rb") as f:
            times, losses, grads = pickle.load(f)
        name = ".".join(file.split(".")[:-1])
        plt.semilogy(np.array(losses) - F_STAR, styles[i], label=name)

    plt.xlabel("Iterations, k")
    plt.ylabel("$f(x_k) - f(x^*)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{DATASET}_logf.eps")
    plt.show()

    for i, file in enumerate(sorted(os.listdir(LOG_PATH))):
        with open(os.path.join(LOG_PATH, file), "rb") as f:
            times, losses, grads = pickle.load(f)
        name = ".".join(file.split(".")[:-1])
        plt.semilogy(grads, styles[i], label=name)

    plt.xlabel("Iterations, k")
    plt.ylabel("$||\\nabla f(x_k)||^2$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{DATASET}_norm.eps")
    plt.show()
else:
    for i, file in enumerate(sorted(os.listdir(LOG_PATH))):
        with open(os.path.join(LOG_PATH, file), "rb") as f:
            times, losses, f1_scores = pickle.load(f)
        name = ".".join(file.split(".")[:-1])
        print(name, f1_scores[-1])
        plt.plot(f1_scores[1:], styles[i], label=name)

    plt.xlabel("Iterations, k")
    plt.ylabel("f1-score (weighted)")
    plt.legend(loc=4)
    plt.grid(alpha=0.4)
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.tight_layout()
    # plt.savefig(f"{DATASET}_f1.eps")
    plt.show()
