import numpy as np 
import matplotlib.pyplot as plt
import os 
from PIL import Image


def load_datasets (data_dir: str):
    images = []
    labels = []
    
    for label_str in os.listdir(data_dir):
        label_dir = os.path.join(data_dir,label_str)
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
                img = Image.open(img_path).convert('L')  # grayscale
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

class NaiveBayesBinary:
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_features = X.shape[1]
        n_classes = len(self.classes)

        X_bin = (X > 0.5).astype(int)

        self.class_priors = np.zeros(n_classes)
        for i, c in enumerate(self.classes):
            self.class_priors[i] = np.mean(y == c)

        self.likelihoods = np.zeros((n_classes, n_features))
        for i, c in enumerate(self.classes):
            X_c = X_bin[y == c]
            pixel_counts = X_c.sum(axis=0)
            self.likelihoods[i] = (pixel_counts + 1) / (X_c.shape[0] + 2)  

    def predict(self, X):
        X_bin = (X > 0.5).astype(int)
        n_samples = X.shape[0]
        n_classes = len(self.classes)

        log_priors = np.log(self.class_priors)
        likelihoods = np.clip(self.likelihoods, 1e-9, 1 - 1e-9)

        y_pred = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):

            log_posteriors = []
            for c in range(n_classes):
                log_likelihood_pixel_1 = np.log(likelihoods[c])
                log_likelihood_pixel_0 = np.log(1 - likelihoods[c])
                log_likelihood = X_bin[i] * log_likelihood_pixel_1 + (1 - X_bin[i]) * log_likelihood_pixel_0
                log_posterior = np.sum(log_likelihood) + log_priors[c]
                log_posteriors.append(log_posterior)
            y_pred[i] = self.classes[np.argmax(log_posteriors)]
        return y_pred
    
    
    
## ChatGPT
def plot_misclassified_images(X, y_true, y_pred, misclassified_idx, max_images_per_plot=100):
    num_images = len(misclassified_idx)
    
    # Calculate how many subsets we need
    num_plots = (num_images // max_images_per_plot) + (1 if num_images % max_images_per_plot != 0 else 0)

    # Loop through subsets and plot them
    for plot_idx in range(num_plots):
        start_idx = plot_idx * max_images_per_plot
        end_idx = min((plot_idx + 1) * max_images_per_plot, num_images)
        subset_idx = misclassified_idx[start_idx:end_idx]

        # Calculate the number of rows and columns for the subplot grid
        num_cols = 10  # Keep 10 columns for the grid
        num_rows = (len(subset_idx) // num_cols) + (1 if len(subset_idx) % num_cols != 0 else 0)

        # Create a plot for the current subset of misclassified images
        plt.figure(figsize=(num_cols * 2, num_rows * 2))  # Adjust the figure size dynamically
        for i, idx in enumerate(subset_idx):  # Show the subset of misclassified samples
            plt.subplot(num_rows, num_cols, i + 1)  # Adjust subplot grid
            plt.imshow(X[idx].reshape(28, 28), cmap='gray')
            plt.title(f'True: {y_true[idx]} Pred: {y_pred[idx]}')
            plt.axis('off')

        plt.tight_layout()  # Adjust subplots to fit into the figure area
        plt.show()
## End of ChatGPT
    
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

    accuracy = np.mean(y_pred == y_val)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    misclassified_idx = np.where(y_pred != y_val)[0] 
    print(f"The model misclassified = {misclassified_idx.shape[0]} images")
    plot_misclassified_images(X_val, y_val, y_pred, misclassified_idx, max_images_per_plot=100)