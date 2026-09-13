import gzip
import numpy as np

directory = "data/"

def load_images(filename):
    with gzip.open(directory + filename, "rb") as f:
        f.read(16)  # skip image header

        data = np.frombuffer(
            f.read(),
            dtype=np.uint8
        )

        images = data.reshape(-1, 28, 28)

        return images / 255.0

def load_labels(filename):
    with gzip.open(directory + filename, "rb") as f:
        f.read(8)  # skip label header

        labels = np.frombuffer(
            f.read(),
            dtype=np.uint8
        )

        return labels

if __name__ == "__main__":
    train_images = load_images(
        "train-images-idx3-ubyte.gz"
    )

    print(train_images.shape)

    train_labels = load_labels(
        "train-labels-idx1-ubyte.gz"
    )

    print(train_labels.shape)
