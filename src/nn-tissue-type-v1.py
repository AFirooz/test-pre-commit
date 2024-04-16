import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

import nn_modules as nnm
import nn_preprocessor as nnp

# import filestructure as fs


def main():
    hyp = nnp.Hyperparams(
        device="cpu",
        patience=30,
        epochs=100,
    )

    labels = nnp.Labels("mz")
    custom_dataset = nnp.CustomDataset("mz", labels.tissue_type)
    custom_dataset.init_transforms(
        transform=transforms.Compose(
            [nnp.ColPadding(custom_dataset.all_cols), nnp.CustomToTensor()]
        )
    )

    train_dataset, val_dataset, test_dataset = nnp.DatasetSplitter(
        custom_dataset, 0.7, 0.15
    ).group_split()

    train_loader = DataLoader(
        train_dataset, batch_size=hyp.batch_size, shuffle=hyp.shuffle
    )
    val_loader = DataLoader(val_dataset, batch_size=hyp.batch_size, shuffle=hyp.shuffle)
    test_loader = DataLoader(
        test_dataset, batch_size=hyp.batch_size, shuffle=hyp.shuffle
    )

    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            # Define the layers
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
            self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
            self.pool = nn.MaxPool1d(kernel_size=2)
            self.fc1 = nn.Linear(
                32 * 122, 128
            )  # Calculate the output size based on the input dimension
            self.fc2 = nn.Linear(
                128, 2
            )  # Output layer dimension is 2 for binary classification

        def forward(self, x):
            # Define the forward pass
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 32 * 122)  # Reshape before passing to fully connected layers
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Initialize the model
    # model = SimpleCNN()

    model = nnm.FullyConnectedModel(
        49_654,
        [10_000, 2000, 500, 50, 2],
        [True for _ in range(5)],
        ["relu", "relu", "relu", "relu", "sigmoid"],
    )
    criterion = nn.CrossEntropyLoss()

    model = nnm.train_network(
        train_loader, val_loader, hyp, model, criterion, scheduler=False
    )

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            # Run the forward pass
            predictions = model(data)
            # calculate the loss
            # loss = criterion(predictions, label)

            # For classification accuracy
            _, predicted = torch.max(predictions, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    # Calculate final evaluation metrics
    average_loss = test_loss / len(test_dataset)
    accuracy = correct / total

    print(f"Average Test Loss: {average_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
