import numpy as np 
import matplotlib.pyplot as plt
import os 
from PIL import Image

# Function to load the dataset from a directory structure
def load_datasets (data_dir: str):
    images = []
    labels = []
    
    # Loop through each folder (each represents a label)
    for label_str in os.listdir(data_dir):
        label_dir = os.path.join(data_dir,label_str)
        if not os.path.isdir(label_dir):
            continue
        try:
            # Convert folder name to numeric label
            label = int(label_str)
        except ValueError:
            print(f"Skipping folder {label_str} (not numeric)")
            continue
        # Loop through image files in each label directory
        for filename in os.listdir(label_dir):
            if filename.endswith(".png"):
                img_path = os.path.join(label_dir, filename)
                img = Image.open(img_path).convert('L')  # Convert image to grayscale
                img_array = np.array(img).flatten() / 255.0  # Normalize and flatten image
                images.append(img_array)
                labels.append(label)
    print(f"Loaded {len(images)} images from {data_dir}")
    return np.array(images), np.array(labels)

# Function to split dataset into training and validation sets
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


# Naive Bayes classifier using binary features (pixels thresholded at 0.5)
class NaiveBayesBinary:
    def fit(self, X, y):
        # Get unique classes and feature count
        self.classes = np.unique(y)
        n_features = X.shape[1]
        n_classes = len(self.classes)

        # Convert features to binary (1 if pixel > 0.5 else 0)
        X_bin = (X > 0.5).astype(int)

        # Calculate class priors (probability of each class)
        self.class_priors = np.zeros(n_classes)
        for i, c in enumerate(self.classes):
            self.class_priors[i] = np.mean(y == c)

        # Calculate likelihoods for each pixel given a class
        self.likelihoods = np.zeros((n_classes, n_features))
        for i, c in enumerate(self.classes):
            X_c = X_bin[y == c]  # Subset samples of class c
            pixel_counts = X_c.sum(axis=0)  # Count of pixel=1 per feature
            # Laplace smoothing applied to avoid zero probabilities
            self.likelihoods[i] = (pixel_counts + 1) / (X_c.shape[0] + 2)  

    def predict(self, X):
        # Convert input features to binary
        X_bin = (X > 0.5).astype(int)
        n_samples = X.shape[0]
        n_classes = len(self.classes)

        # Log priors for numerical stability
        log_priors = np.log(self.class_priors)
        likelihoods = np.clip(self.likelihoods, 1e-9, 1 - 1e-9)  # Prevent log(0)

        y_pred = np.zeros(n_samples, dtype=int)
        # For each sample, compute log posterior for every class
        for i in range(n_samples):
            log_posteriors = []
            for c in range(n_classes):
                # Compute log likelihood for pixel being 1 or 0
                log_likelihood_pixel_1 = np.log(likelihoods[c])
                log_likelihood_pixel_0 = np.log(1 - likelihoods[c])
                log_likelihood = X_bin[i] * log_likelihood_pixel_1 + (1 - X_bin[i]) * log_likelihood_pixel_0
                log_posterior = np.sum(log_likelihood) + log_priors[c]
                log_posteriors.append(log_posterior)
            # Class with max posterior probability is chosen
            y_pred[i] = self.classes[np.argmax(log_posteriors)]
        return y_pred
    
    
## ChatGPT helper function for visualization
def plot_misclassified_images(X, y_true, y_pred, misclassified_idx, max_images_per_plot=100):
    """
    Plot misclassified images in subsets if they exceed the limit per plot.
    """
    num_images = len(misclassified_idx)
    
    # Determine how many plots are needed
    num_plots = (num_images // max_images_per_plot) + (1 if num_images % max_images_per_plot != 0 else 0)

    # Loop through subsets and plot them
    for plot_idx in range(num_plots):
        start_idx = plot_idx * max_images_per_plot
        end_idx = min((plot_idx + 1) * max_images_per_plot, num_images)
        subset_idx = misclassified_idx[start_idx:end_idx]

        # Grid setup for subplots
        num_cols = 10  # Fixed number of columns
        num_rows = (len(subset_idx) // num_cols) + (1 if len(subset_idx) % num_cols != 0 else 0)

        # Create the figure
        plt.figure(figsize=(num_cols * 2, num_rows * 2))
        for i, idx in enumerate(subset_idx):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow(X[idx].reshape(28, 28), cmap='gray')  # Reshape flat vector to 28x28
            plt.title(f'True: {y_true[idx]} Pred: {y_pred[idx]}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()
## End of ChatGPT helper


if __name__ == "__main__":
    dataset_dir = "MNIST"
    print("Loading dataset...")
    X, y = load_datasets(dataset_dir)
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features each.")

    print("Splitting dataset into train and validation sets...")
    X_train, y_train, X_val, y_val = split_dataset(X, y, val_ratio=0.2, shuffle=True, random_seed=42)
    print(f"Train samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    print("Training Naive Bayes Binary classifier...")
    nb_bin = NaiveBayesBinary()
    nb_bin.fit(X_train, y_train)

    print("Predicting validation set...")
    y_pred = nb_bin.predict(X_val)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_val)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    # Identify misclassified samples
    misclassified_idx = np.where(y_pred != y_val)[0] 
    print(f"The model misclassified = {misclassified_idx.shape[0]} images")

    # Visualize misclassified images
    plot_misclassified_images(X_val, y_val, y_pred, misclassified_idx, max_images_per_plot=100)
