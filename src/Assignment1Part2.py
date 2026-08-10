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

def saveModel(model, path):
    torch.save(model.state_dict(), path)
    print(f"Saved PyTorch Model State to {path}")

def loadModel(model: NeuralNetwork, path: Path, device: str = "mps") -> NeuralNetwork:
    model.load_state_dict(torch.load(path, weights_only=True))
    model.to(device)
    model.eval()
    print(f"Loaded PyTorch Model State from {path}")
    return model

def predict(model: NeuralNetwork, x, device: str = "mps"):
    # model.eval()
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        return pred.argmax(1).item()



def main():
    device = get_device()
    print(f"Using device: {device}")

    training_data = datasets.MNIST(
        root=str(DATA_DIR), train=True, download=True, transform=ToTensor()
    )
    test_data = datasets.MNIST(
        root=str(DATA_DIR), train=False, download=True, transform=ToTensor()
    )
    train_loader = DataLoader(training_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Train and test the model
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train(train_loader, model, loss_fn, optimizer, device)
        test(test_loader, model, loss_fn, device)

    # Save the model
    # torch.save(model.state_dict(), MODEL_PATH)
    saveModel(model, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    # Load the model 
    # model = NeuralNetwork().to(device)
    modelLoaded = loadModel(NeuralNetwork(), MODEL_PATH, device)

    classes = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]

    """
    test_data behaves like a list of tuples
    test_data[i]          →  a tuple: (image_tensor, label)
    test_data[i][0]       →  the image tensor  (shape: 1 x 28 x 28)
    test_data[i][1]       →  the integer label (0 - 9, the actual digit)

    test_data[10][0]
    [10] — select the 10th sample in the dataset (i.e., which handwritten digit image you're looking at)
    [0] — take the image tensor from that (index 0 of the tuple), not the label (which would be index [1])

    test_data[10][1]
    [10] — select the 10th sample in the dataset (i.e., which handwritten digit image you're looking at)
    [1] — take the label from that (index 1 of the tuple), not the image tensor (which would be index [0])
    """
    random_index = torch.randint(len(test_data), size=(1,)).item()

    # x, y = test_data[10][0], test_data[10][1] 
    x = test_data[1][0] # x = image tensor  (tuple index 0)
    y = test_data[1][1] # y = label integer (tuple index 1)
    # print (f'\nPredicted x,y: "{x, y}"  \n')
    print(test_data[10][1])  # prints the true label integer

    # pred = predict(modelLoaded, x, device) # Cannot just use x by itself
    pred = predict(modelLoaded, x.unsqueeze(0), device) # adds a batch dimension to the image tensor (1 x 28 x 28) → (1 x 1 x 28 x 28)
    predicted = classes[pred]
    actual = classes[y]
    
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
    # with torch.no_grad():
    #     x = x.to(device)
    #     pred = modelLoaded(x)
    #     predicted = classes[pred[0].argmax(0)]
    #     actual = classes[y]
    #     print(f'Predicted: "{predicted}", Actual: "{actual}"')


if __name__ == "__main__":
    main()
