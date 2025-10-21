import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt


def load_dataset(data_dir: str):
    """
    Load a dataset of grayscale PNG images organized in numeric-labeled folders.

    Args:
        data_dir (str): Path to the root dataset directory. Each subfolder name is the label.

    Returns:
        (np.ndarray, np.ndarray): Flattened normalized image data and corresponding labels.
    """
    images = []
    labels = []

    for label_str in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label_str)
        if not os.path.isdir(label_dir):
            continue

        # Skip non-numeric folder names
        try:
            label = int(label_str)
        except ValueError:
            print(f"Skipping folder {label_str} (not numeric)")
            continue

        # Load and normalize images
        for filename in os.listdir(label_dir):
            if filename.endswith(".png"):
                img_path = os.path.join(label_dir, filename)
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img_array = np.array(img).flatten() / 255.0
                images.append(img_array)
                labels.append(label)

    print(f"Loaded {len(images)} images from {data_dir}")
    return np.array(images), np.array(labels)


def split_dataset(X, y, val_ratio=0.2, shuffle=True, random_seed=None):
    """
    Split dataset into training and validation subsets.

    Args:
        X (np.ndarray): Input features.
        y (np.ndarray): Labels.
        val_ratio (float): Fraction of data for validation.
        shuffle (bool): Whether to shuffle before splitting.
        random_seed (int, optional): Random seed for reproducibility.

    Returns:
        Tuple of train and validation sets: (X_train, y_train, X_val, y_val)
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


def knn_predict(X_train, y_train, X_test, k=1):
    """
    Predict labels for test samples using k-Nearest Neighbors.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test features.
        k (int): Number of nearest neighbors.

    Returns:
        np.ndarray: Predicted labels for X_test.
    """
    y_pred = []
    for test_sample in X_test:
        distances = np.linalg.norm(X_train - test_sample, axis=1)
        knn_indices = np.argsort(distances)[:k]
        knn_labels = y_train[knn_indices]
        count = np.bincount(knn_labels.astype(int))
        y_pred.append(np.argmax(count))
    return np.array(y_pred)


# -----------------------------CHATGPT ----------------------------- #

def fit_pca_2d(X):
    """
    Compute the mean and top-2 principal components for 2D visualization.
    """
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    cov = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    top2 = eigvecs[:, :2]
    return mean, top2


def transform_pca_2d(X, mean, components):
    """
    Transform data to 2D using precomputed PCA components.
    """
    X_centered = X - mean
    return np.dot(X_centered, components)


def plot_2d_embedding(X_embedded, y, title="2D PCA embedding", show_legend=True):
    """
    Plot a 2D PCA embedding with color-coded labels.
    """
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y,
                          cmap='tab10', alpha=0.6, s=15)
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    if show_legend:
        plt.legend(*scatter.legend_elements(), title="Digits")
    plt.grid(True)
    plt.show()

# -----------------------------End of CHATGPT ----------------------------- #

# --------------------------------- Main Script --------------------------------- #

if __name__ == "__main__":
    dataset_dir = "MNIST"

    # Load dataset
    print("Loading dataset...")
    X, y = load_dataset(dataset_dir)
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features each.")

    # Split dataset
    print("Splitting dataset into train and validation sets...")
    X_train, y_train, X_val, y_val = split_dataset(
        X, y, val_ratio=0.2, shuffle=True, random_seed=42)
    print(f"Train samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    # Evaluate KNN for different k values
    ks = [1, 3, 5]
    accuracies = []

    for k in ks:
        print(f"Running {k}-NN classifier...")
        y_pred = knn_predict(X_train, y_train, X_val, k=k)
        accuracy = np.mean(y_pred == y_val)
        print(f"Validation Accuracy for k={k}: {accuracy * 100:.2f}%")
        accuracies.append(accuracy)

    # Fit PCA on training data and visualize embeddings
    mean_train, pca_components = fit_pca_2d(X_train)
    X_train_2d = transform_pca_2d(X_train, mean_train, pca_components)
    X_val_2d = transform_pca_2d(X_val, mean_train, pca_components)

    plot_2d_embedding(X_train_2d, y_train, title="Train data PCA embedding (true labels)")
    plot_2d_embedding(X_val_2d, y_val, title="Validation data PCA embedding (true labels)")

    # Visualize predicted labels for k=3
    k_example = 3
    y_pred_example = knn_predict(X_train, y_train, X_val, k=k_example)
    plot_2d_embedding(X_val_2d, y_pred_example,
                      title=f"Validation data PCA embedding (predicted labels k={k_example})")

    # Plot KNN accuracy comparison
    plt.figure(figsize=(6, 4))
    plt.bar([str(k) for k in ks], [acc * 100 for acc in accuracies], color='skyblue')
    plt.xlabel('k in KNN')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('KNN Validation Accuracy for Different k Values')
    plt.ylim(0, 100)
    plt.show()
