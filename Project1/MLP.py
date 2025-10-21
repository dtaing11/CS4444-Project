import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader


# ----------------------------- Load Dataset ----------------------------- #
def load_dataset(data_dir: str):
    """
    Load grayscale PNG images from folders named with numeric labels.
    Each image is flattened and normalized to [0,1].
    """
    images = []
    labels = []
    for label_str in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label_str)
        if not os.path.isdir(label_dir):
            continue
        try:
            # Convert folder name to integer label
            label = int(label_str)
        except ValueError:
            print(f"Skipping folder {label_str} (not numeric)")
            continue
        for filename in os.listdir(label_dir):
            if filename.endswith(".png"):
                img_path = os.path.join(label_dir, filename)
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img_array = np.array(img).flatten() / 255.0  # Normalize and flatten
                images.append(img_array)
                labels.append(label)
    print(f"Loaded {len(images)} images from {data_dir}")
    return np.array(images), np.array(labels)


# ----------------------------- Split Dataset ----------------------------- #
def split_dataset(X, y, val_ratio=0.2, shuffle=True):
    """
    Split dataset into training and validation sets.

    Args:
        X: feature matrix
        y: labels
        val_ratio: fraction of data for validation
        shuffle: whether to shuffle before splitting
    """
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)
    split_idx = int(num_samples * (1 - val_ratio))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


# ----------------------------- Define MLP Model ----------------------------- #
class MLP(nn.Module): 
    """
    Simple Multilayer Perceptron (MLP) with two hidden layers.
    Input size = 784 (28x28), Output size = 10 classes.
    """
    def __init__(self, inputDim=784, outputDim=10):
        super().__init__()
        self.input_fc = nn.Linear(inputDim, 256)      # First fully connected layer
        self.hidden_layer1 = nn.Linear(256, 128)      # Second fully connected layer
        self.output_fc = nn.Linear(128, outputDim)    # Output layer (10 classes)

    def forward(self, x):
        """
        Forward pass through the network.
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)                    # Flatten input
        h_1 = F.relu(self.input_fc(x))                # Hidden layer 1 with ReLU
        h_2 = F.relu(self.hidden_layer1(h_1))         # Hidden layer 2 with ReLU
        y_pred = self.output_fc(h_2)                  # Output layer (no softmax)
        return y_pred


# ----------------------------- Train Model ----------------------------- #
def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.01):
    """
    Train the MLP model and evaluate on validation data.
    """
    criterion = nn.CrossEntropyLoss()                 # Cross-entropy loss for classification
    optimizer = optim.SGD(model.parameters(), lr=lr)  # Stochastic Gradient Descent optimizer

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()                                 # Set model to training mode
        running_loss = 0.0

        # ---------------- Training Loop ---------------- #
        for inputs, targets in train_loader:
            inputs = inputs.float()
            targets = targets.long()

            optimizer.zero_grad()                     # Reset gradients
            outputs = model(inputs)                   # Forward pass
            loss = criterion(outputs, targets)        # Compute loss
            loss.backward()                           # Backpropagation
            optimizer.step()                          # Update weights
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        # ---------------- Validation Loop ---------------- #
        model.eval()                                  # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0 
        with torch.no_grad():                         # Disable gradient tracking
            for inputs, targets in val_loader: 
                inputs = inputs.float()
                targets = targets.long()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)  # Get predicted class
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_losses.append(val_loss / len(val_loader))
        acc = 100 * correct / total                   # Compute accuracy

        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_losses[-1]:.4f}, "
              f"Val Loss: {val_losses[-1]:.4f}, "
              f"Val Acc: {acc:.2f}%")

    return train_losses, val_losses


# ----------------------------- Main Execution ----------------------------- #
if __name__ == "__main__":
    data_dir = 'MNIST' 
    print("Loading dataset...")
    X, y = load_dataset(data_dir)
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features each.")

    print("Splitting dataset into train and validation sets...")
    X_train, y_train, X_val, y_val = split_dataset(X, y)
    print(f"Train samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    # Convert NumPy arrays to PyTorch tensors
    print("Converting to PyTorch tensors......")
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)
    X_val_tensor = torch.tensor(X_val)
    y_val_tensor = torch.tensor(y_val)

    # Create DataLoaders for batching
    print("Create Dataloaders......")
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Initialize and train the MLP model
    print("Training Multilayer Perceptron......")
    model = MLP()
    val_losses = []
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=10, lr=0.01)

    # Plot training and validation loss
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', color='r', label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker='o', color='b', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
