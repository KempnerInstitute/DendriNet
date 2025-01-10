# datasets.py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Import your MNIST loader from your existing utils
# (Adjust import if your module is in a different location)
from dendritic_modeling.utils import load_MNIST

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
            # y1 = XOR of bits[0], bits[1]
            y1 = bits[0] ^ bits[1]
            # y2 = XOR of bits[2], bits[3], bits[4]
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
# 5) MNIST LOADER
########################################
# We rely on load_MNIST(...) from dendritic_modeling.utils
# which typically returns a dictionary: {'train': (X_train, Y_train), 
#                                       'valid': (X_valid, Y_valid), 
#                                       'test': (X_test, Y_test)}
# We just need to wrap each partition in a DataLoader or something equivalent.

def get_mnist_dataloaders(
    train_valid_split=0.9, 
    batch_size=64, 
    # you can pass your cache_dir if needed
):
    data = load_MNIST(train_valid_split=train_valid_split)
    # data['train'] => (X_train, Y_train), each is torch.Tensor
    # We'll create DataLoaders from those:

    train_dataset = torch.utils.data.TensorDataset(*data['train'][:])
    valid_dataset = torch.utils.data.TensorDataset(*data['valid'][:])
    test_dataset  = torch.utils.data.TensorDataset(*data['test'][:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


########################################
# 6) Unified function to get ANY dataset
########################################
def get_unified_dataloaders(
    dataset_name="mnist",
    train_size=8000,
    valid_size=1000,
    test_size=1000,
    image_size=10,
    n_orient_classes=8,
    n_bits=8,
    batch_size=64,
    mnist_train_valid_split=0.9
):
    """
    Returns train_loader, valid_loader, test_loader 
    for the selected dataset.
    """
    if dataset_name == "mnist":
        return get_mnist_dataloaders(
            train_valid_split=mnist_train_valid_split,
            batch_size=batch_size
        )
    elif dataset_name == "line":
        total = train_size + valid_size + test_size
        dataset = LineDataset(n_samples=total, image_size=image_size)
    elif dataset_name == "center_surround":
        total = train_size + valid_size + test_size
        dataset = CenterSurroundDataset(n_samples=total, image_size=image_size)
    elif dataset_name == "orientation_bars":
        total = train_size + valid_size + test_size
        dataset = OrientationBarsDataset(
            n_samples=total, 
            image_size=image_size,
            n_orient_classes=n_orient_classes
        )
    elif dataset_name == "multi_xor":
        total = train_size + valid_size + test_size
        dataset = MultiXORDataset(n_samples=total, n_bits=n_bits)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    # if not MNIST, we split + create dataloaders
    train_data, valid_data, test_data = torch.utils.data.random_split(
        dataset, [train_size, valid_size, test_size]
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader