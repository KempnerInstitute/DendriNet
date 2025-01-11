# synthetic_datasets.py
import torch
import numpy as np
from torch.utils.data import Dataset, random_split, TensorDataset

from dendritic_modeling.utils import load_MNIST  # Adjust path if needed


########################################
# 1) LINE DETECTION DATASET
########################################
class LineDataset(Dataset):
    """
    Creates images of size (image_size x image_size) containing:
      - 0: no line
      - 1: horizontal line
      - 2: vertical line
    """
    def __init__(self, n_samples=10000, image_size=10):
        super().__init__()
        self.n_samples = n_samples
        self.image_size = image_size
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        X, Y = [], []
        for _ in range(self.n_samples):
            img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            label = np.random.choice([0, 1, 2])
            if label == 1:  # horizontal line
                row = np.random.randint(0, self.image_size)
                img[row, :] = 1.0
            elif label == 2:  # vertical line
                col = np.random.randint(0, self.image_size)
                img[:, col] = 1.0

            X.append(img.flatten())
            Y.append(label)

        return torch.tensor(np.stack(X)), torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


########################################
# 2) CENTER-SURROUND DATASET
########################################
class CenterSurroundDataset(Dataset):
    """
    Creates images of size (image_size x image_size) with:
      - class 0: uniform image
      - class 1: bright center, dark surround
      - class 2: dark center, bright surround
    """
    def __init__(self, n_samples=10000, image_size=15):
        super().__init__()
        self.n_samples = n_samples
        self.image_size = image_size
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        X, Y = [], []
        radius = self.image_size // 4
        center_x = center_y = self.image_size // 2

        for _ in range(self.n_samples):
            label = np.random.choice([0, 1, 2])
            img = np.zeros((self.image_size, self.image_size), dtype=np.float32)

            if label == 0:
                # uniform random brightness
                val = np.random.rand() * 0.2 + 0.4
                img[:] = val
            elif label == 1:
                # bright center, dark surround
                for i in range(self.image_size):
                    for j in range(self.image_size):
                        dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                        if dist < radius:
                            img[i, j] = 1.0
                        else:
                            img[i, j] = 0.0
            else:
                # dark center, bright surround
                for i in range(self.image_size):
                    for j in range(self.image_size):
                        dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                        if dist < radius:
                            img[i, j] = 0.0
                        else:
                            img[i, j] = 1.0

            X.append(img.flatten())
            Y.append(label)

        return torch.tensor(np.stack(X)), torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


########################################
# 3) ORIENTATION BARS DATASET
########################################
class OrientationBarsDataset(Dataset):
    """
    Each sample is an (image_size x image_size) image with a bar
    in one of n_orient_classes orientations.
    """
    def __init__(self, n_samples=10000, image_size=16, n_orient_classes=8):
        super().__init__()
        self.n_samples = n_samples
        self.image_size = image_size
        self.n_orient_classes = n_orient_classes
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        X, Y = [], []
        for _ in range(self.n_samples):
            label = np.random.randint(0, self.n_orient_classes)
            angle_deg = (180 / self.n_orient_classes) * label
            angle_rad = np.deg2rad(angle_deg)

            img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            cx = cy = self.image_size // 2
            for r in range(-cx, cx):
                x = int(cx + r * np.cos(angle_rad))
                y = int(cy + r * np.sin(angle_rad))
                if 0 <= x < self.image_size and 0 <= y < self.image_size:
                    img[y, x] = 1.0

            X.append(img.flatten())
            Y.append(label)

        return torch.tensor(np.stack(X)), torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


########################################
# 4) MULTI-XOR DATASET
########################################
class MultiXORDataset(Dataset):
    """
    Each sample is an n_bits-bit input. We define multiple XOR sub-tasks
    and combine them for a single label with multiple classes.
    e.g., XOR(x0, x1) => y1, XOR(x2, x3, x4) => y2 => label in [0..3].
    """
    def __init__(self, n_samples=10000, n_bits=8):
        super().__init__()
        self.n_samples = n_samples
        self.n_bits = n_bits
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        X, Y = [], []
        for _ in range(self.n_samples):
            bits = np.random.randint(0, 2, size=self.n_bits).astype(np.float32)
            # y1 = XOR(x0, x1)
            y1 = bits[0] ^ bits[1]
            # y2 = XOR(x2, x3, x4)
            y2 = bits[2] ^ bits[3] ^ bits[4]
            label = (y1 << 1) + y2  # 0..3
            X.append(bits)
            Y.append(label)
        return torch.tensor(X), torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


########################################
# MNIST LOADER (RAW DATASETS)
########################################
def load_mnist_as_datasets(train_valid_split=0.9):
    """
    Loads MNIST using your existing load_MNIST() utility, then wraps
    each split in a TensorDataset. Returns (train_dataset, valid_dataset, test_dataset).
    """
    data = load_MNIST(train_valid_split=train_valid_split)
    # data['train'] => (X_train, Y_train)
    train_dataset = torch.utils.data.TensorDataset(*data['train'][:])
    valid_dataset = torch.utils.data.TensorDataset(*data['valid'][:])
    test_dataset  = torch.utils.data.TensorDataset(*data['test'][:])
    return train_dataset, valid_dataset, test_dataset


########################################
# 5) Unified function to get ANY dataset
########################################
def get_unified_datasets(
    dataset_name="mnist",
    train_size=8000,
    valid_size=1000,
    test_size=1000,
    image_size=10,
    n_orient_classes=8,
    n_bits=8,
    mnist_train_valid_split=0.9
):
    """
    Returns (train_dataset, valid_dataset, test_dataset) as raw Dataset objects
    for the specified dataset_name.
    The main script or trainer can then create DataLoaders from these Datasets.
    """

    if dataset_name == "mnist":
        # If we want MNIST, load from your utility
        return load_mnist_as_datasets(train_valid_split=mnist_train_valid_split)

    elif dataset_name == "line":
        total = train_size + valid_size + test_size
        full_dataset = LineDataset(n_samples=total, image_size=image_size)
        train_ds, valid_ds, test_ds = torch.utils.data.random_split(
            full_dataset, [train_size, valid_size, test_size]
        )
        return train_ds, valid_ds, test_ds

    elif dataset_name == "center_surround":
        total = train_size + valid_size + test_size
        full_dataset = CenterSurroundDataset(n_samples=total, image_size=image_size)
        train_ds, valid_ds, test_ds = torch.utils.data.random_split(
            full_dataset, [train_size, valid_size, test_size]
        )
        return train_ds, valid_ds, test_ds

    elif dataset_name == "orientation_bars":
        total = train_size + valid_size + test_size
        full_dataset = OrientationBarsDataset(
            n_samples=total,
            image_size=image_size,
            n_orient_classes=n_orient_classes
        )
        train_ds, valid_ds, test_ds = torch.utils.data.random_split(
            full_dataset, [train_size, valid_size, test_size]
        )
        return train_ds, valid_ds, test_ds

    elif dataset_name == "multi_xor":
        total = train_size + valid_size + test_size
        full_dataset = MultiXORDataset(n_samples=total, n_bits=n_bits)
        train_ds, valid_ds, test_ds = torch.utils.data.random_split(
            full_dataset, [train_size, valid_size, test_size]
        )
        return train_ds, valid_ds, test_ds

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")