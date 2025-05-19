import sys
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
except ModuleNotFoundError as e:
    missing = e.name
    sys.exit(
        f"Missing dependency: {missing}.\n"
        "Install required packages with `pip install torch torchvision matplotlib`."
    )
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an MNIST CNN with live accuracy animation"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--save",
        metavar="GIF",
        help="save animation to the specified GIF file",
    )
    parser.add_argument(
        "--checkpoint",
        metavar="FILE",
        default="mnist_cnn.pth",
        help="path to save or load model weights",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume training from the checkpoint if it exists",
    )
    return parser.parse_args()


class SimpleCNN(nn.Module):
    """Small convolutional network for MNIST."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.fc1 = nn.Linear(3200, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)
        x = self.relu(self.conv3(x))
        x = self.max_pool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train(model, device, train_loader, optimizer, loss_fn):
    model.train()
    running_loss = 0.0
    correct = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
    return running_loss / len(train_loader.dataset), correct / len(train_loader.dataset)


def test(model, device, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            test_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return test_loss / len(test_loader.dataset), correct / len(test_loader.dataset)


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = SimpleCNN().to(device)
    if args.resume and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    fig, ax = plt.subplots()
    ax.set_xlim(0, args.epochs)
    ax.set_ylim(0, 1)
    train_line, = ax.plot([], [], label='train accuracy')
    test_line, = ax.plot([], [], label='test accuracy')
    ax.legend()
    accuracies = {'train': [], 'test': []}

    def update(epoch):
        train_loss, train_acc = train(model, device, train_loader, optimizer, loss_fn)
        test_loss, test_acc = test(model, device, test_loader, loss_fn)
        accuracies['train'].append(train_acc)
        accuracies['test'].append(test_acc)
        x_data = list(range(1, len(accuracies['train']) + 1))
        train_line.set_data(x_data, accuracies['train'])
        test_line.set_data(x_data, accuracies['test'])
        ax.set_xlim(0, len(x_data) + 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Epoch {epoch + 1}\nTrain Acc: {train_acc:.2f} Test Acc: {test_acc:.2f}')
        torch.save(model.state_dict(), args.checkpoint)
        return train_line, test_line

    ani = FuncAnimation(fig, update, frames=range(args.epochs), blit=False, repeat=False)
    if args.save:
        ani.save(args.save, writer=PillowWriter(fps=1))
    else:
        plt.show()


if __name__ == '__main__':
    main()
