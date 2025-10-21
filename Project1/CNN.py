import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader


# ----------------------------- Data Loading Utilities ----------------------------- #

def load_dataset(data_dir: str, image_size=(28, 28)):
    """
    Load grayscale PNG images organized by numeric label folders.

    Args:
        data_dir (str): Path to dataset root directory.
        image_size (tuple): Desired image dimensions (width, height).

    Returns:
        (np.ndarray, np.ndarray): Normalized image arrays and corresponding labels.
    """
    images = []
    labels = []

    for label_str in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label_str)
        if not os.path.isdir(label_dir):
            continue

        # Skip folders that are not numeric labels
        try:
            label = int(label_str)
        except ValueError:
            print(f"Skipping folder {label_str} (not numeric)")
            continue

        # Load all PNG images within the label directory
        for filename in os.listdir(label_dir):
            if filename.endswith(".png"):
                img_path = os.path.join(label_dir, filename)
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img = img.resize(image_size)             # Resize to target size
                img_array = np.array(img) / 255.0        # Normalize to [0, 1]
                images.append(img_array)
                labels.append(label)

    print(f"Loaded {len(images)} images from {data_dir}")
    return np.array(images), np.array(labels)


def split_dataset(X, y, val_ratio=0.2, shuffle=True):
    """
    Split dataset into training and validation subsets.

    Args:
        X (np.ndarray): Input feature data.
        y (np.ndarray): Corresponding labels.
        val_ratio (float): Fraction of data to reserve for validation.
        shuffle (bool): Whether to shuffle before splitting.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Training and validation sets (X_train, y_train, X_val, y_val).
    """
    num_samples = X.shape[0]
    indices = np.arange(num_samples)

    if shuffle:
        np.random.shuffle(indices)

    split_idx = int(num_samples * (1 - val_ratio))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


# ----------------------------- CNN Model Definition ----------------------------- #

class CNN_Model(nn.Module):
    """
    Convolutional Neural Network for MNIST-like digit classification.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)   # First convolution layer
        self.pool = nn.MaxPool2d(2, 2)                 # Max pooling layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # Second convolution layer

        # Fully connected layers
        self.fc = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        """
        Forward pass through the CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, 1, 28, 28).

        Returns:
            torch.Tensor: Raw output logits (no softmax applied).
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)     # Flatten for fully connected layers
        x = self.fc(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # NOTE: Do not apply softmax here; CrossEntropyLoss handles it internally.
        return x


# ----------------------------- Training Function ----------------------------- #

def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.01):
    """
    Train the CNN model and track training/validation loss.

    Args:
        model (nn.Module): CNN model instance.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate.

    Returns:
        Tuple[List[float], List[float]]: Lists of training and validation losses.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # -------- Training Phase -------- #
        for inputs, targets in train_loader:
            inputs, targets = inputs.float(), targets.long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        # -------- Validation Phase -------- #
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.float(), targets.long()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_losses.append(val_loss / len(val_loader))
        acc = 100 * correct / total

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {train_losses[-1]:.4f}, "
            f"Val Loss: {val_losses[-1]:.4f}, "
            f"Val Acc: {acc:.2f}%"
        )

    return train_losses, val_losses


# ----------------------------- Main Script ----------------------------- #

if __name__ == "__main__":
    data_dir = "MNIST"

    print("Loading dataset...")
    X, y = load_dataset(data_dir)
    print(f"Loaded {X.shape[0]} samples with shape {X.shape[1:]}")

    print("Splitting dataset into train and validation sets...")
    X_train, y_train, X_val, y_val = split_dataset(X, y)
    print(f"Train samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    # Add channel dimension for CNN input
    X_train = X_train[:, None, :, :]
    X_val = X_val[:, None, :, :]

    print("Converting to PyTorch tensors...")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    print("Creating DataLoaders...")
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    print("Training CNN Model...")
    model = CNN_Model()

    train_losses, val_losses = train_model(model, train_loader, val_loader,
                                           num_epochs=10, lr=0.01)

    # ----------------------------- Plot Training Curve ----------------------------- #
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o',
             color='r', label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker='o',
             color='b', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
