"""PyTorch Quickstart — FashionMNIST classifier.

Follows https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
Picks CUDA (NVIDIA), MPS (Apple Silicon), or CPU automatically.
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pth"
BATCH_SIZE = 64
EPOCHS = 5


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            current = (batch + 1) * len(X)
            print(f"  loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0.0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test:  accuracy {100 * correct:>5.1f}%  avg loss {test_loss:>8f}")


def main():
    device = get_device()
    print(f"Using device: {device}")

    training_data = datasets.FashionMNIST(
        root=str(DATA_DIR), train=True, download=True, transform=ToTensor()
    )
    test_data = datasets.FashionMNIST(
        root=str(DATA_DIR), train=False, download=True, transform=ToTensor()
    )
    train_loader = DataLoader(training_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train(train_loader, model, loss_fn, optimizer, device)
        test(test_loader, model, loss_fn, device)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    # Reload + single-sample prediction demo
    reloaded = NeuralNetwork().to(device)
    reloaded.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    reloaded.eval()

    classes = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        pred = reloaded(x.to(device))
        print(f'\nPredicted: "{classes[pred.argmax(1).item()]}"  Actual: "{classes[y]}"')


if __name__ == "__main__":
    main()
