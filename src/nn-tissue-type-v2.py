# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
# ---

# %%
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

# %%
import nn_modules as nnm
import nn_preprocessor as nnp


# %% [markdown]
# import filestructure as fs


# %%
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


# %%
def main():
    hyp = nnp.Hyperparams(
        batch_size=10_000,
        learning_rate=0.001,
        patience=30,
        epochs=100,
        eval_every=2,
    )
    labels = nnp.Labels("mz")
    grouping = nnp.Grouping("mz", group_init="person")

    custom_dataset = nnp.CustomDataset(
        "mz", labels.tissue_type, grouping.result, transpose=True
    )
    custom_dataset.pre_transforms(
        transform=transforms.Compose([nnp.ColPadding(custom_dataset.all_cols)])
    )

    splitter = nnp.DatasetSplitter(custom_dataset, 0.7, 0.15)
    train_dataset, val_dataset, test_dataset = splitter.group_split()

    train_loader = DataLoader(
        train_dataset, batch_size=hyp.batch_size, shuffle=hyp.shuffle
    )
    val_loader = DataLoader(val_dataset, batch_size=hyp.batch_size, shuffle=hyp.shuffle)
    test_loader = DataLoader(
        test_dataset, batch_size=hyp.batch_size, shuffle=hyp.shuffle
    )

    # Initialize the model
    # model = SimpleCNN()
    model = nnm.FullyConnectedModel(
        input_dim=custom_dataset.get_input_dim,  # 704
        layer_dims=[350, 175, 88, 44, 22, 11, 1],
        layer_batchnorm=[True for _ in range(7)],
        layer_acts=["relu", "relu", "relu", "relu", "relu", "relu", "linear"],
    )

    criterion = nn.BCEWithLogitsLoss()
    try:
        model = nnm.train_network(
            train_loader, val_loader, hyp, model, criterion, scheduler=False
        )
    finally:
        # just to make sure we save our state
        splitter.save_state(hyp.output_path)

    model.eval()
    threshold = 0.5
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        model.to(hyp.device)
        for i, (data, label) in enumerate(test_loader):
            data: torch.Tensor = data.to(hyp.device)
            label: torch.Tensor = label.to(hyp.device)
            # Run the forward pass
            logit_pred = model(data).squeeze()
            # calculate the loss
            test_loss = criterion(logit_pred, label)

            # For classification accuracy
            total += label.size(0)
            y_prob = nn.functional.sigmoid(logit_pred)
            correct += ((y_prob > threshold) == label).float().sum().item()

    # Calculate final evaluation metrics
    average_loss = test_loss / len(test_dataset)
    accuracy = correct / total

    print(f"Average Test Loss: {average_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")


# %%
if __name__ == "__main__":
    main()
