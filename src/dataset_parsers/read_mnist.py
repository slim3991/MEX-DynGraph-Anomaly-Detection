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


# File paths
input_path = "raw_data/"
training_images_filepath = join(
    input_path, "train-images-idx3-ubyte/train-images.idx3-ubyte"
)
training_labels_filepath = join(
    input_path, "train-labels-idx1-ubyte/train-labels.idx1-ubyte"
)
test_images_filepath = join(input_path, "t10k-images-idx3-ubyte/t10k-images.idx3-ubyte")
test_labels_filepath = join(input_path, "t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte")

# Load MNIST dataset
mnist_dataloader = MnistDataloader(
    training_images_filepath,
    training_labels_filepath,
    test_images_filepath,
    test_labels_filepath,
)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# twos = x_train[y_train == 8]
# # Transpose to (784, 5958) so each COLUMN is an image
# twos_flat = twos.reshape(twos.shape[0], -1).T
# twos_subset = twos_flat[:, :1000]  # Grab first 1000 samples


x_train = x_train.reshape(-1, 28 * 28).T  # (784, N)
mean_vec = np.mean(x_train, axis=1, keepdims=True)
x_train = x_train - mean_vec


x_test = x_test.reshape(-1, 28 * 28).T
x_test = x_test - mean_vec

x_train = x_train[:, :1000]
y_train = y_train[:1000]

subspaces = []

for i in range(10):
    print(f"Processing digit: {i}")

    current_class = x_train[:, y_train == i]
    other_classes = x_train[:, y_train != i]

    # 1. Compute orthonormal basis for OTHER classes
    Uo, _, _ = np.linalg.svd(other_classes, full_matrices=False)
    Uo = Uo[:, :50]  # Keep top components (tune this)

    # 2. Project current_class onto OTHER subspace
    projection = Uo @ (Uo.T @ current_class)

    # 3. Remove shared components
    unique_features = current_class - projection

    # 4. SVD on residual
    Q_digit, _, _ = np.linalg.svd(unique_features, full_matrices=False)

    # Keep first 10 unique components
    subspaces.append(Q_digit[:, :10])


def classify_sample(x, subspaces):
    """
    x: (784,) single flattened image
    subspaces: list of 10 matrices, each (784, k)
    """
    errors = []

    for Q in subspaces:
        # Project onto digit subspace
        projection = Q @ (Q.T @ x)

        # Reconstruction error
        error = np.linalg.norm(x - projection)
        errors.append(error)

    # Pick digit with smallest reconstruction error
    return np.argmin(errors)


def classify_batch(X, subspaces):
    """
    X: (784, N)
    Returns: predicted labels (N,)
    """
    N = X.shape[1]
    errors = np.zeros((10, N))

    for digit, Q in enumerate(subspaces):
        projection = Q @ (Q.T @ X)
        errors[digit] = np.linalg.norm(X - projection, axis=0)

    return np.argmin(errors, axis=0), errors


preds, errors = classify_batch(x_test, subspaces)

accuracy = np.mean(preds == y_test)
print("Overall accuracy:", accuracy)
conf_matrix = np.zeros((10, 10), dtype=int)

for true_label, pred_label in zip(y_test, preds):
    conf_matrix[true_label, pred_label] += 1
precision = np.zeros(10)
recall = np.zeros(10)
f1 = np.zeros(10)

for i in range(10):
    TP = conf_matrix[i, i]
    FP = conf_matrix[:, i].sum() - TP
    FN = conf_matrix[i, :].sum() - TP

    precision[i] = TP / (TP + FP + 1e-8)
    recall[i] = TP / (TP + FN + 1e-8)
    f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-8)

    print(f"Digit {i}:")
    print(f"  Precision: {precision[i]:.4f}")
    print(f"  Recall:    {recall[i]:.4f}")
    print(f"  F1 Score:  {f1[i]:.4f}")
# plt.imshow(subspaces[5][:, 0].reshape(28, 28), cmap="gray")
# plt.title(f"Reconstruction")
# plt.show()
