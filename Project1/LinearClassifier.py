import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


def load_datasets(data_dir: str):
    """
    Load grayscale PNG images from folders named by numeric labels.

    Args:
        data_dir (str): Root dataset directory where each subfolder corresponds to a class label.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Flattened normalized image data and numeric labels.
    """
    images = []
    labels = []

    for label_str in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label_str)
        if not os.path.isdir(label_dir):
            continue
        try:
            label = int(label_str)  # Convert folder name to numeric label
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


def split_dataset(X, y, val_ratio=0.2, shuffle=True, random_seed=None):
    """
    Split dataset into training and validation sets.

    Args:
        X (np.ndarray): Feature data.
        y (np.ndarray): Labels.
        val_ratio (float): Fraction of data for validation.
        shuffle (bool): Whether to shuffle the dataset.
        random_seed (int): Optional random seed for reproducibility.

    Returns:
        Tuple: (X_train, y_train, X_val, y_val)
    """
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    if shuffle:
        if random_seed is not None:
            np.random.seed(random_seed)
        np.random.shuffle(indices)
    split_idx = int(num_samples * (1 - val_ratio))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


class LinearClassifier:
    """
    Simple linear classifier using Mean Squared Error (MSE) loss.
    """

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W = None
        self.b = None

    def fit(self, X, y):
        """
        Train the linear classifier using gradient descent.

        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Training labels.
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Convert labels to one-hot encoding
        y_one_hot = np.zeros((n_samples, n_classes))
        y_one_hot[np.arange(n_samples), y] = 1

        # Initialize weights and biases
        self.W = np.random.randn(n_features, n_classes) * 0.01
        self.b = np.zeros((1, n_classes))

        # Training loop
        for epoch in range(self.epochs):
            y_pred = X.dot(self.W) + self.b
            loss = np.mean(np.square(y_pred - y_one_hot))  # MSE loss
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss {loss:.4f}")

            # Compute gradients
            dW = (2 / n_samples) * X.T.dot(y_pred - y_one_hot)
            db = (2 / n_samples) * np.sum(y_pred - y_one_hot, axis=0, keepdims=True)

            # Update weights and biases
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

    def predict(self, X):
        """
        Predict class labels for the given input.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted class indices.
        """
        y_pred = X.dot(self.W) + self.b
        return np.argmax(y_pred, axis=1)


# --------------------------- CHATGPT --------------------------- #

def pca_numpy(X, n_components=2):
    """
    Perform Principal Component Analysis (PCA) using NumPy.

    Args:
        X (np.ndarray): Data matrix.
        n_components (int): Number of principal components to keep.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (Reduced data, PCA components)
    """
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvectors by descending eigenvalues
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_idx]
    eigenvalues = eigenvalues[sorted_idx]

    # Select top components and project
    components = eigenvectors[:, :n_components]
    X_reduced = np.dot(X_centered, components)
    return X_reduced, components


def plot_decision_boundary(clf, X_2d, y, resolution=0.02):
    """
    Plot decision boundary and data points for a 2D classifier.

    Args:
        clf (LinearClassifier): Trained classifier.
        X_2d (np.ndarray): 2D feature data.
        y (np.ndarray): Class labels.
        resolution (float): Grid step size for plotting.
    """
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.tab10)

    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=plt.cm.tab10, edgecolor='k')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title("Decision Boundary and Data Points")
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.show()

# ---------------------------End of CHATGPT --------------------------- #

# ------------------------------- Main Execution ------------------------------- #

if __name__ == "__main__":
    dataset_dir = "MNIST"

    print("Loading dataset...")
    X, y = load_datasets(dataset_dir)
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features each.")

    print("Splitting dataset into train and validation sets...")
    X_train, y_train, X_val, y_val = split_dataset(X, y, val_ratio=0.2, shuffle=True, random_seed=42)
    print(f"Train samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    print("Training linear classifier...")
    clf = LinearClassifier(learning_rate=0.01, epochs=1000)
    clf.fit(X_train, y_train)

    print("Predicting on validation set...")
    y_pred = clf.predict(X_val)
    accuracy = np.mean(y_pred == y_val)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    # PCA visualization
    X_train_2d, components = pca_numpy(X_train, n_components=2)
    X_val_centered = X_val - np.mean(X_train, axis=0)
    X_val_2d = np.dot(X_val_centered, components)

    clf_2d = LinearClassifier(learning_rate=0.01, epochs=1000)
    clf_2d.fit(X_train_2d, y_train)

    plot_decision_boundary(clf_2d, X_val_2d, y_val)
