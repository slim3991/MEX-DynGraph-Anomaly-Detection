from re import fullmatch
import numpy as np
import struct
from array import array
from os.path import join, exists
import random
import matplotlib.pyplot as plt
import tensorly as tl
from sklearn.decomposition import KernelPCA


class MnistDataloader:
    def __init__(
        self,
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    ):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        # Load labels
        with open(labels_filepath, "rb") as f:
            magic, size = struct.unpack(">II", f.read(8))
            if magic != 2049:
                raise ValueError(
                    f"Label file magic mismatch: expected 2049, got {magic}"
                )
            labels = np.frombuffer(f.read(), dtype=np.uint8)

        # Load images
        with open(images_filepath, "rb") as f:
            magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
            if magic != 2051:
                raise ValueError(
                    f"Image file magic mismatch: expected 2051, got {magic}"
                )
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(size, rows, cols)

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath, self.training_labels_filepath
        )
        x_test, y_test = self.read_images_labels(
            self.test_images_filepath, self.test_labels_filepath
        )
        return (x_train, y_train), (x_test, y_test)


def load_mnist(input_path="raw_data"):
    """Load MNIST dataset from IDX files."""

    def read_images_labels(images_filepath, labels_filepath):
        with open(labels_filepath, "rb") as f:
            magic, size = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)

        with open(images_filepath, "rb") as f:
            magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(size, rows, cols)

        return images, labels

    training_images_filepath = join(
        input_path, "train-images-idx3-ubyte/train-images.idx3-ubyte"
    )
    training_labels_filepath = join(
        input_path, "train-labels-idx1-ubyte/train-labels.idx1-ubyte"
    )

    test_images_filepath = join(
        input_path, "t10k-images-idx3-ubyte/t10k-images.idx3-ubyte"
    )
    test_labels_filepath = join(
        input_path, "t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte"
    )

    x_train, y_train = read_images_labels(
        training_images_filepath, training_labels_filepath
    )
    x_test, y_test = read_images_labels(test_images_filepath, test_labels_filepath)

    return (x_train, y_train), (x_test, y_test)


def create_anomaly_tensor(
    normal_digit=8,
    num_images=256,
    num_anomalies=20,
    seed=None,
    input_path="raw_data",
):
    """
    Creates a tensor (28 x 28 x num_images) mostly containing one digit
    with random anomaly images inserted.
    """

    if seed is not None:
        np.random.seed(seed)

    (x_train, y_train), _ = load_mnist(input_path)

    normal_imgs = x_train[y_train == normal_digit]
    anomaly_imgs = x_train[y_train != normal_digit]

    if len(normal_imgs) < num_images:
        raise ValueError("Not enough normal images.")

    # Sample normal images
    normal_indices = np.random.choice(len(normal_imgs), num_images, replace=False)
    tensor = normal_imgs[normal_indices].copy()

    # Choose positions for anomalies
    anomaly_positions = np.random.choice(num_images, num_anomalies, replace=False)
    anomaly_indices = np.random.choice(len(anomaly_imgs), num_anomalies, replace=False)

    for i, pos in enumerate(anomaly_positions):
        tensor[pos] = anomaly_imgs[anomaly_indices[i]]

    # Convert to (28,28,K)
    tensor = np.transpose(tensor, (1, 2, 0))

    return tensor, anomaly_positions


def visualize_tensor_grid(tensor, grid=(8, 8), random_select=True):
    """
    Visualize slices of a tensor as a grid of images.

    tensor shape: (28,28,K)
    grid: (rows, cols)
    """

    rows, cols = grid
    total = rows * cols
    K = tensor.shape[2]

    if random_select:
        indices = np.random.choice(K, total, replace=False)
    else:
        indices = np.arange(total)

    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):
        img = tensor[:, :, indices[i]]
        ax.imshow(img, cmap="gray")
        ax.axis("off")

    plt.tight_layout()


if __name__ == "__main__":
    tensor, anomaly_pos = create_anomaly_tensor(
        normal_digit=8, num_images=256, num_anomalies=100, seed=42
    )

    print("Tensor shape:", tensor.shape)
    print("Anomalies at:", anomaly_pos)

    visualize_tensor_grid(tensor, grid=(10, 10))
