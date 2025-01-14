# synthetic_datasets.py

import torch
import numpy as np
from torch.utils.data import Dataset, random_split, TensorDataset

from dendritic_modeling.utils import load_MNIST  # Adjust path if needed
from torchvision.datasets import MNIST
import torchvision.transforms as T


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
    train_dataset = TensorDataset(*data['train'][:])
    valid_dataset = TensorDataset(*data['valid'][:])
    test_dataset  = TensorDataset(*data['test'][:])
    return train_dataset, valid_dataset, test_dataset


#### NEW DATASETS ####

class MNISTContextualFG(Dataset):
    """
    Takes a standard MNIST sample (28x28),
    transforms left half into random noise, keeps the right half as original digit.
    Label remains the digit (0..9).
    """
    def __init__(self, base_mnist, noise_level=0.1):
        super().__init__()
        self.base_mnist = base_mnist  # e.g. torchvision MNIST
        self.noise_level = noise_level

    def __len__(self):
        return len(self.base_mnist)

    def __getitem__(self, idx):
        img, label = self.base_mnist[idx]
        # If using torchvision transforms, 'img' might be a 1x28x28 tensor or a PIL image
        img_arr = np.array(img, dtype=np.float32)  # shape: [28,28]
        w = 28
        # random noise on left half
        new_img = np.random.rand(w, w) * self.noise_level
        boundary = w//2
        new_img[:, boundary:] = img_arr[:, boundary:]
        # flatten
        new_img_tensor = torch.tensor(new_img, dtype=torch.float32).view(-1)
        return new_img_tensor, label


class MNISTNoisyPartial(Dataset):
    """
    Takes a standard MNIST sample, adds random noise and partial occlusion.
    occlusion_prob: fraction of pixels set to 0
    noise_level: random [0, noise_level] added
    """
    def __init__(self, base_mnist, occlusion_prob=0.3, noise_level=0.2):
        super().__init__()
        self.base_mnist = base_mnist
        self.occlusion_prob = occlusion_prob
        self.noise_level = noise_level

    def __len__(self):
        return len(self.base_mnist)

    def __getitem__(self, idx):
        img, label = self.base_mnist[idx]
        img_arr = np.array(img, dtype=np.float32)  # shape [28,28]
        # add noise
        noise = np.random.rand(*img_arr.shape) * self.noise_level
        img_arr += noise
        # occlusion
        mask = np.random.rand(*img_arr.shape) < self.occlusion_prob
        img_arr[mask] = 0.0
        # flatten
        tensor_img = torch.tensor(img_arr, dtype=torch.float32).view(-1)
        return tensor_img, label


class MNISTTemporal(Dataset):
    """
    Splits each 28x28 digit into seq_len vertical slices.
    Then concatenates them in time order into a single vector.
    """
    def __init__(self, base_mnist, seq_len=4):
        super().__init__()
        self.base_mnist = base_mnist
        self.seq_len = seq_len
        self.slice_width = 28 // seq_len

    def __len__(self):
        return len(self.base_mnist)

    def __getitem__(self, idx):
        img, label = self.base_mnist[idx]
        img_arr = np.array(img, dtype=np.float32)  # [28,28]

        slices = []
        for s in range(self.seq_len):
            sc = s * self.slice_width
            ec = (s+1)*self.slice_width
            part = img_arr[:, sc:ec].flatten()
            slices.append(part)
        data_np = np.concatenate(slices, axis=0)  # shape: seq_len*(28*slice_width)

        data_tensor = torch.tensor(data_np, dtype=torch.float32)
        return data_tensor, label


class MNISTSparse(Dataset):
    """
    Binarizes MNIST images above a threshold => sparser input.
    """
    def __init__(self, base_mnist, threshold=0.3):
        super().__init__()
        self.base_mnist = base_mnist
        self.threshold = threshold

    def __len__(self):
        return len(self.base_mnist)

    def __getitem__(self, idx):
        img, label = self.base_mnist[idx]
        arr = np.array(img, dtype=np.float32)/255.0
        bin_arr = (arr > self.threshold).astype(np.float32)
        bin_tensor = torch.tensor(bin_arr, dtype=torch.float32).view(-1)
        return bin_tensor, label


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
    mnist_train_valid_split=0.9,
    # NEW ARGS
    seq_len=4,
    occlusion_prob=0.3,
    noise_level=0.2,
    threshold=0.3
):
    """
    Returns (train_dataset, valid_dataset, test_dataset) as raw Dataset objects
    for the specified dataset_name.
    The main script or trainer can then create DataLoaders from these Datasets.
    """

    if dataset_name == "mnist":
        return load_mnist_as_datasets(train_valid_split=mnist_train_valid_split)

    elif dataset_name == "line":
        total = train_size + valid_size + test_size
        full_dataset = LineDataset(n_samples=total, image_size=image_size)
        train_ds, valid_ds, test_ds = random_split(
            full_dataset, [train_size, valid_size, test_size]
        )
        return train_ds, valid_ds, test_ds

    elif dataset_name == "center_surround":
        total = train_size + valid_size + test_size
        full_dataset = CenterSurroundDataset(n_samples=total, image_size=image_size)
        train_ds, valid_ds, test_ds = random_split(
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
        train_ds, valid_ds, test_ds = random_split(
            full_dataset, [train_size, valid_size, test_size]
        )
        return train_ds, valid_ds, test_ds

    elif dataset_name == "multi_xor":
        total = train_size + valid_size + test_size
        full_dataset = MultiXORDataset(n_samples=total, n_bits=n_bits)
        train_ds, valid_ds, test_ds = random_split(
            full_dataset, [train_size, valid_size, test_size]
        )
        return train_ds, valid_ds, test_ds

    #### NEW MNIST-based tasks ####
    elif dataset_name == "mnist_contextual":
        # We'll load raw torchvision MNIST
        from torchvision.datasets import MNIST
        base_mnist = MNIST(root=".", download=True, train=True)
        # wrap
        full_dataset = MNISTContextualFG(base_mnist, noise_level=noise_level)
        train_ds, valid_ds, test_ds = random_split(
            full_dataset, [train_size, valid_size, test_size]
        )
        return train_ds, valid_ds, test_ds

    elif dataset_name == "mnist_noisy_partial":
        from torchvision.datasets import MNIST
        base_mnist = MNIST(root=".", download=True, train=True)
        full_dataset = MNISTNoisyPartial(
            base_mnist,
            occlusion_prob=occlusion_prob,
            noise_level=noise_level
        )
        train_ds, valid_ds, test_ds = random_split(
            full_dataset, [train_size, valid_size, test_size]
        )
        return train_ds, valid_ds, test_ds

    elif dataset_name == "mnist_temporal":
        from torchvision.datasets import MNIST
        base_mnist = MNIST(root=".", download=True, train=True)
        full_dataset = MNISTTemporal(base_mnist, seq_len=seq_len)
        train_ds, valid_ds, test_ds = random_split(
            full_dataset, [train_size, valid_size, test_size]
        )
        return train_ds, valid_ds, test_ds

    elif dataset_name == "mnist_sparse":
        from torchvision.datasets import MNIST
        base_mnist = MNIST(root=".", download=True, train=True)
        full_dataset = MNISTSparse(base_mnist, threshold=threshold)
        train_ds, valid_ds, test_ds = random_split(
            full_dataset, [train_size, valid_size, test_size]
        )
        return train_ds, valid_ds, test_ds

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")