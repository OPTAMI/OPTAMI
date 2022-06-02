#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


f, axs = plt.subplots(1, 2, figsize=(10, 4))
markers = ["o", "v", "d", "s", "^", "<", ">", "+"]

DATASET = "a9a"
F_START = 0.6931471805599453
F_STAR  = 0.32501597692415846
WITH_TIME = False

LOG_PATH = f"logs_{DATASET}"
for i, file in enumerate(sorted(os.listdir(LOG_PATH))):
    with open(os.path.join(LOG_PATH, file), "rb") as f:
        times, losses, grads = pickle.load(f)

    name = file.split(".")[0]

    if WITH_TIME:
        axs[0].semilogy(times, np.array(losses) - F_STAR, label=name, marker=markers[i], markevery=max(1, len(losses) // 10))
        axs[1].semilogy(times, grads, label=name, marker=markers[i], markevery=max(1, len(grads) // 10))
    else:
        axs[0].semilogy(np.array(losses) - F_STAR, label=name, marker=markers[i], markevery=max(1, len(losses) // 10))
        axs[1].semilogy(grads, label=name, marker=markers[i], markevery=max(1, len(grads) // 10))

if WITH_TIME:
    axs[0].set_xlabel("t, sec.")
    axs[1].set_xlabel("t, sec.")
else:
    axs[0].set_xlabel("N, iterations")
    axs[1].set_xlabel("N, iterations")

axs[0].set_ylabel("$f(x_N)$")
axs[0].grid(alpha=0.4)
axs[0].legend()
axs[0].set_ylim(1e-3, F_START - F_STAR)

axs[1].set_ylabel("$||\\nabla f(x_N)||^2$")
axs[1].grid(alpha=0.4)
axs[1].set_ylim(1e-8, 1e-1)

plt.tight_layout()
postfix = "time" if WITH_TIME else "iters"
plt.savefig(f"figure_{postfix}.jpg", dpi=400)
