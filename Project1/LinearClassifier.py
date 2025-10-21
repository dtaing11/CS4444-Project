import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def load_datasets(data_dir: str):
    images = []
    labels = []
    
    for label_str in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label_str)
        if not os.path.isdir(label_dir):
            continue
        try:
            label = int(label_str)  
        except ValueError:
            print(f"Skipping folder {label_str} (not numeric)")
            continue
        for filename in os.listdir(label_dir):
            if filename.endswith(".png"):
                img_path = os.path.join(label_dir, filename)
                img = Image.open(img_path).convert('L')  
                img_array = np.array(img).flatten() / 255.0 
                images.append(img_array)
                labels.append(label)
    print(f"Loaded {len(images)} images from {data_dir}")
    return np.array(images), np.array(labels)

def split_dataset(X, y, val_ratio=0.2, shuffle=True, random_seed=None):
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
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        y_one_hot = np.zeros((n_samples, n_classes))
        y_one_hot[np.arange(n_samples), y] = 1
        
        self.W = np.random.randn(n_features, n_classes) * 0.01
        self.b = np.zeros((1, n_classes))

        for epoch in range(self.epochs):
            y_pred = X.dot(self.W) + self.b
            
            loss = np.mean(np.square(y_pred - y_one_hot)) 
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss {loss:.4f}")
            
            dW = (2 / n_samples) * X.T.dot(y_pred - y_one_hot)
            db = (2 / n_samples) * np.sum(y_pred - y_one_hot, axis=0, keepdims=True)
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

    def predict(self, X):
        y_pred = X.dot(self.W) + self.b
        return np.argmax(y_pred, axis=1) 
## CHATGPT
def pca_numpy(X, n_components=2):
    # Center the data (subtract mean)
    X_centered = X - np.mean(X, axis=0)

    # Compute covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvectors by descending eigenvalues
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_idx]
    eigenvalues = eigenvalues[sorted_idx]

    # Select the first n_components eigenvectors
    components = eigenvectors[:, :n_components]

    # Project data
    X_reduced = np.dot(X_centered, components)
    return X_reduced, components

def plot_decision_boundary(clf, X_2d, y, resolution=0.02):
    # Define bounds of the domain
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

    # Create a grid of points with distance 'resolution' between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    
    # Flatten grid to pass through classifier
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict class labels for each point on the grid
    Z = clf.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.tab10)
    
    # Plot also the training points
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=plt.cm.tab10, edgecolor='k')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title("Decision Boundary and Data Points")
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.show()
## End of CHATGPT
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

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_val)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")


    ##CHATGPT
    X_train_2d, components = pca_numpy(X_train, n_components=2)

    # For validation data, center with the same mean as training
    X_val_centered = X_val - np.mean(X_train, axis=0)
    X_val_2d = np.dot(X_val_centered, components)
    # Train linear classifier on 2D data
    clf_2d = LinearClassifier(learning_rate=0.01, epochs=1000)
    clf_2d.fit(X_train_2d, y_train)

    # Plot decision boundary
    plot_decision_boundary(clf_2d, X_val_2d, y_val)
    ## END OF CHATGPT

 
