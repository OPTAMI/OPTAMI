from tqdm import tqdm
import torch
import time


def zero_all(model):
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()


def train(model, optimizer, dataloader, img_size, epochs=10, verbose=False, return_grads=False):
    zero_all(model)

    tic = time.time()
    losses = []
    grads = []

    for _ in tqdm(range(epochs)):
        for i, (images, labels) in enumerate(dataloader):
            if i != 0:
                continue

            def closure():
                optimizer.zero_grad()
                prediction = model(image)
                return model.criterion(prediction, label)

            image = images.view(-1, img_size ** 2)
            label = labels.fmod(2)

            loss = model.criterion(model(image), label)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            grad_norm = 0.
            for p in model.parameters():
                if p.grad is None:
                    continue
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item()**2
            grads.append(grad_norm)
            optimizer.zero_grad()

            if verbose:
                print(f'loss = {loss.item()}')

            optimizer.step(closure)

    toc = time.time()
    if return_grads:
        return losses, grads, toc - tic
    return losses, toc - tic


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim, gamma=0.):
        super().__init__()
        self.gamma = gamma
        self.linear = torch.nn.Linear(input_dim, output_dim).double()
        zero_all(self)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

    def criterion(self, hypothesis, reference):
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(hypothesis, reference)

        if self.gamma > 0.:
            for param in self.parameters():
                loss += param.square().sum().mul(self.gamma)
        return loss

    def pure_loss(self, hypothesis, reference):
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(hypothesis, reference)
