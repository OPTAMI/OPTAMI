#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pickle
import os


f, axs = plt.subplots(1, 2, figsize=(10, 4))
markers = ["o", "v", "d", "s", "^", "<", ">"]

LOG_PATH = "logs"
for i, file in enumerate(sorted(os.listdir(LOG_PATH))):
    with open(os.path.join(LOG_PATH, file), "rb") as f:
        losses, grads = pickle.load(f)

    name = file.split(".")[0]
    axs[0].plot(losses, label=name, marker=markers[i], markevery=max(1, len(losses) // 10))
    axs[1].semilogy(grads, label=name, marker=markers[i], markevery=max(1, len(grads) // 10))

axs[0].set_xlabel("T, iterations")
axs[0].set_ylabel("$f(x_T)$")
axs[0].grid(alpha=0.4)
axs[0].legend()

axs[1].set_xlabel("T, iterations")
axs[1].set_ylabel("$||\\nabla f(x_T)||^2$")
axs[1].grid(alpha=0.4)
axs[1].legend()

plt.tight_layout()
plt.savefig("figure.pdf")
