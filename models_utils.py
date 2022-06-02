from tqdm import tqdm
import torch
import time


def zero_all(model):
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()


def minimize(f, w, optimizer, epochs=10, verbose=False, tqdm_on=False):
    times = []
    losses = []
    grads = []

    def closure():
        optimizer.zero_grad()
        return f(w)

    times.append(0.)
    loss = closure()
    losses.append(loss.item())
    loss.backward()
    grads.append(torch.norm(w.grad.data, p=2)**2)
    optimizer.zero_grad()

    tic = time.time()

    r = range(epochs)
    if tqdm_on:
        r = tqdm(r)

    for _ in r:
        loss = closure()
        loss.backward()

        f_val = loss.item()
        grad_norm_squared = torch.norm(w.grad.data, p=2)**2

        times.append(time.time() - tic)
        losses.append(f_val)
        grads.append(grad_norm_squared)

        if verbose:
            print(f'f = {f_val}, grad = {grad_norm_squared}')
        optimizer.step(closure)

    return times, losses, grads
