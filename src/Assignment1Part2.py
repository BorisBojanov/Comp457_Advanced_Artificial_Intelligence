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
import random
import matplotlib.pyplot as plt

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

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 1: learn 32 filters on the raw grayscale image
        # 1 CNN the convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=0)
        # shape after: 32 × 14 × 14
        #   in_channels: grayscale=1        (color images would be 3)
        #   out_channels: means it learns 32 different filters -> produces 32 feature maps (32 features are learned) 
        #   kernel_size: 3x3 filter    
        #   padding = keeps same size -> adds a border of zeros so the output is the same size as the input

        # 2. The pooling layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # divides the spatial dimensions by 2
        # 28x28 → 14x14, 14x14 → 7x7
        # Shrinks the feature maps by taking the max value in each 2x2 window. This:
        # Reduces computation
        # Makes the model robust to small shifts in position



        """
            Tensor shape after the convolutional and pooling layers:
            Formula for the output size:
            out = floor((W - K + 2P) / S) + 1

            Input: 1 x 28 x 28
            Conv2d(1→32, k=3, pad=1)	same spatial size	32 x 28 x 28
            MaxPool2d(2)	            halve spatial dims	32 x 14 x 14
            Conv2d(32→64, k=3, pad=1)	same spatial size	64 x 14 x 14
            MaxPool2d(2)	            halve again	        64 x 7 x 7
            Flatten	                    multiply all dims	3136
            Linear(3136→128)		                        128
            Linear(128→10)		                            10

            64*7*7 = 3136 : The number of hard-coded features that are learned by the convolutional layers. 
            This is the input size for the first fully connected layer.
        """


        # Block 2: learn 64 filters on the 32 feature maps from block 1
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # shape after: 64 × 7 × 7

        # Block 3: A third convolutional layer to learn 128 filters on the 64 feature maps from block 2
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # shape after: 128 × 3 × 3 (2.5x2.5 + 1 → 3.5x3.5 after round down to 3x3)
        
        # Classifier head
        self.flatten = nn.Flatten()
        # 3. activation function
        nn.ReLU()

        # Chanign number of convolutional layers
        # self.fc1 = nn.Linear(32 * 14 * 14, 64)   # 6272 → 64 (1 convolutional layer)
        # self.fc2 = nn.Linear(64, 10)             # 64   → 10 classes
        
        # self.fc1 = nn.Linear(64 * 7 * 7, 128)   # 3136 → 128 (2 convolutional layers)
        # self.fc2 = nn.Linear(128, 10)             # 128  → 10 classes

        # self.fc1 = nn.Linear(128 * 3 * 3, 256)    # 1152 → 256 (3 convolutional layers)
        # self.fc2 = nn.Linear(256, 10)             # 256  → 10 classes

        # Changing Padding and Strides
        self.fc1 = nn.Linear(32 * 13 * 13, 64)   # 5408 → 64 (1 convolutional layer)
        self.fc2 = nn.Linear(64, 10)             # 64   → 10 classes

    # def forward(self, x):
    #     # x enters as shape: (batch, 1, 28, 28)
    #     x = self.pool1(torch.relu(self.conv1(x)))  # → (batch, 32, 14, 14)
    #     # x = self.pool2(torch.relu(self.conv2(x)))  # → (batch, 64,  7,  7)
    #     # x = self.pool3(torch.relu(self.conv3(x)))  # → (batch, 128,  3,  3)
    #     x = self.flatten(x)                        # → (batch, 3136) or (1152)
    #     x = torch.relu(self.fc1(x))                # → (batch, 128) or (256)
    #     return self.fc2(x)                         # → (batch, 10)

    def forward(self, x):
        # x enters as shape: (batch, 1, 28, 28)
        # padding = 0, stride = 1, kernel_size = 3, W = 28
        # If we set padding = 0 then our size formula must be
        # out = floor((W - K + 2(0)) / S) + 1
        # out = floor((W - K) / S) + 1

        x = self.pool1(torch.relu(self.conv1(x)))  # → (batch, 32, 26, 26)
        #each convolution layer eats 2 pixels
        x = self.flatten(x)                        # → (batch, 21632)
        x = torch.relu(self.fc1(x))                # → (batch, 64)
        return self.fc2(x)                         # → (batch, 10)


def imagePlotter(image_tensor , label):
    pass

def imagesPlotter(image_tensors: list, label, num_cols: int = 4):
    """
    Plots all image tensors with its label.
    image_tensors = [img1, img2, img3, img4]
    
    The tensor is shape (1, 28, 28) — channel first. 
        imshow wants 2D, so pass x.squeeze(). 
        If the tensor has been moved to MPS, add .cpu() first: 
        x.squeeze().cpu()

epochs
    """
    # Convert single 4D batch tensor to a list of 3D tensors
    images = []
    if isinstance(image_tensors, torch.Tensor) and image_tensors.ndim == 4:
        images = [image_tensors[i] for i in range(image_tensors.size(0))]
    else:
        images = image_tensors

    num_images = len(images)
    num_rows = (num_images + num_cols - 1) // num_cols

    # Create the figure grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3), squeeze=False)
    axes = axes.flatten()

    for i in range(len(axes)):
        if i < num_images:
            img = images[i].detach().cpu()

            # Handle Grayscale (1, H, W) vs RGB (3, H, W)
            if img.shape[0] == 1:
                img = img.squeeze(0)  # Convert to (H, W)
                cmap = "gray"
            else:
                img = img.permute(1, 2, 0)  # Convert (C, H, W) to (H, W, C)
                cmap = None

            # Unnormalize if your images are scaled to [-1, 1]
            # img = img * 0.5 + 0.5

            # Clip values to [0, 1] to prevent Matplotlib rendering artifacts
            img = torch.clamp(img, 0, 1)

            axes[i].imshow(img.numpy(), cmap=cmap)
            axes[i].axis("off")  # Hide axis labels
        else:
            axes[i].axis("off")  # Hide unused grid subplots

    # fig.savefig(f"data/figures/image_plot_{label}.png", dpi=150, bbox_inches="tight")
    # 3. Optimize layout spacing and render
    plt.tight_layout()
    # plt.show() # Will be called outside of this function

    

def curvePlotter(train_losses: list, test_losses: list, test_accs: list):
    """
    Left: train_losses and test_losses — two lines, same axes (both are loss, same units, so they share a scaler).
    Right: test_accs — its own axes.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_losses, label="train loss")
    ax.plot(test_losses, label="test loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training and Test Loss")
    ax.legend() # makes a legend for multiple lines on the same plot
    fig.savefig("data/figures/loss_plot.png", dpi=150, bbox_inches="tight")
    # plt.show() # Will be called outside of this function


def train(dataloader, model, loss_fn: nn.CrossEntropyLoss, optimizer, device):
    size = len(dataloader.dataset)
    lossEpoch = 0
    avgTrainingloss = 0.0 # average training loss per epoch
    model.train()
    num_batches = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        lossEpoch += loss.item()
        if batch % 100 == 0:
            current = (batch + 1) * len(X)
            print(f"  loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

    avgTrainingloss = lossEpoch / num_batches
    return avgTrainingloss

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
    # print(f"Test:  accuracy {100 * correct:>5.1f}%  avg loss {test_loss:>8f}")
    return correct, test_loss

def saveModel(model, path):
    torch.save(model.state_dict(), path)
    print(f"Saved PyTorch Model State to {path}")

def loadModel(model: NeuralNetwork | CNN, path: Path, device: str = "mps") -> NeuralNetwork | CNN:
    model.load_state_dict(torch.load(path, weights_only=True))
    model.to(device)
    model.eval()
    print(f"Loaded PyTorch Model State from {path}")
    return model

def predict(model: NeuralNetwork | CNN, x, device: str = "mps"):
    # model.eval()
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        return pred.argmax(1).item()

"""
current model Image (28x28) → Flatten to 784 numbers → Linear → Linear → 10 outputs

The flattening throws away the 2D structure of the image.
"""
def problem1(trainAndloadModel: bool = True):
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
    train_losses, test_losses, test_accs = [], [], []

    if trainAndloadModel:
            # Train and test the model
        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")
            
            train_losses.append(train(train_loader, model, loss_fn, optimizer, device))
            test_accs.append(test(test_loader, model, loss_fn, device)[0])
            test_losses.append(test(test_loader, model, loss_fn, device)[1])
        saveModel(model, MODEL_PATH)
    else:
        for epoch in range(EPOCHS):
            test_accs.append(test(test_loader, model, loss_fn, device)[0])
            test_losses.append(test(test_loader, model, loss_fn, device)[1])
    # Load the model 
    # model = NeuralNetwork().to(device)
    modelLoaded = loadModel(NeuralNetwork(), MODEL_PATH, device)
    


    classes = ["0","1", "2", "3", "4", "5", "6", "7", "8", "9"]

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
    random_index = random.randint(0, len(test_data)) 
    print(f"Length of test data: {len(test_data)} \n Random index: {random_index}")

    # x, y = test_data[10][0], test_data[10][1] 
    x = test_data[random_index][0] # x = image tensor  (tuple index 0)
    y = test_data[random_index][1] # y = label integer (tuple index 1)
    # print (f'\nPredicted x,y: "{x, y}"  \n')
    # print(test_data[10][1])  # prints the true label integer
    testimage0 = test_data[random_index][0] #.permute(1, 2, 0).detach().cpu().numpy()  # shows the image of the handwritten digit
    testimages = [testimage0]
    # pred = predict(modelLoaded, x, device) # Cannot just use x by itself
    pred = predict(modelLoaded, x.unsqueeze(0), device) # adds a batch dimension to the image tensor (1 x 28 x 28) → (1 x 1 x 28 x 28)
    predicted = classes[pred]
    actual = classes[y]
    
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
    print(f'Image {actual} is shown below:')
    imagesPlotter(testimages, actual)
    curvePlotter(train_losses, test_losses, test_accs)
    plt.show()

    # with torch.no_grad():
    #     x = x.to(device)
    #     pred = modelLoaded(x)
    #     predicted = classes[pred[0].argmax(0)]
    #     actual = classes[y]
    #     print(f'Predicted: "{predicted}", Actual: "{actual}"')

""""""
def problem2(trainAndloadModel: bool = True):
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

    model = CNN().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_losses, test_losses, test_accs = [], [], []

    if trainAndloadModel:
            # Train and test the model
        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")

            train_losses.append(train(train_loader, model, loss_fn, optimizer, device))
            test_accs.append(test(test_loader, model, loss_fn, device)[0])
            test_losses.append(test(test_loader, model, loss_fn, device)[1])
        saveModel(model, MODEL_PATH)

    else:
        for epoch in range(EPOCHS):
            test_accs.append(test(test_loader, model, loss_fn, device)[0])
            test_losses.append(test(test_loader, model, loss_fn, device)[1])
    # Load the model 
    # model = CNN().to(device)
    modelLoaded = loadModel(CNN(), MODEL_PATH, device)



    random_index = random.randint(0, len(test_data) - 1)
    print(f"Length of test data: {len(test_data)} \n Random index: {random_index}")

    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    # test_data[0 to len(datasets.MNIST)] a tuple: (image_tensor, label)
    tensor = test_data[random_index][0] #test_data[i][0] x = image tensor  (tuple index 0)
    label = test_data[random_index][1] #test_data[i][1] y = label integer (tuple index 1)

    # The image tensor is in the shape (1 x 28 x 28) and needs to be permuted to (28 x 28 x 1) for plotting
    testimage0 = test_data[random_index][0] #.permute(1, 2, 0).detach().cpu().numpy()  # shows the image of the handwritten digit
    testimages = [testimage0]
    pred = predict(modelLoaded, tensor.unsqueeze(0), device) # adds a batch dimension to the image tensor (1 x 28 x 28) → (1 x 1 x 28 x 28)
    predicted = classes[pred]
    actual = classes[label]
    
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
    print(f'Image {actual} is shown below:')
    imagesPlotter(testimages, actual)
    curvePlotter(train_losses, test_losses, test_accs)
    plt.show()



def main():
    # problem1(trainAndloadModel=True)
    problem2(trainAndloadModel=True) # True/False to train-load or load model

if __name__ == "__main__":
    main()
