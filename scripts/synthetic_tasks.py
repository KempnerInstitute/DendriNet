import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

#############################
# 1) LINE DETECTION DATASET
#############################
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
        X = []
        Y = []
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


#############################
# 2) CENTER-SURROUND DATASET
#############################
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
            label = np.random.choice([0,1,2])
            img = np.zeros((self.image_size, self.image_size), dtype=np.float32)

            if label == 0:
                # uniform random brightness
                val = np.random.rand() * 0.2 + 0.4
                img[:] = val
            elif label == 1:
                # bright center, dark surround
                for i in range(self.image_size):
                    for j in range(self.image_size):
                        dist = np.sqrt((i-center_x)**2 + (j-center_y)**2)
                        if dist < radius:
                            img[i, j] = 1.0
                        else:
                            img[i, j] = 0.0
            else:
                # dark center, bright surround
                for i in range(self.image_size):
                    for j in range(self.image_size):
                        dist = np.sqrt((i-center_x)**2 + (j-center_y)**2)
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


#############################
# 3) ORIENTATION BARS DATASET
#############################
class OrientationBarsDataset(Dataset):
    """
    Each sample is an image_size x image_size image with a bar in one of N orientation classes.
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
            # draw the bar
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


#############################
# 4) MULTI-XOR DATASET
#############################
class MultiXORDataset(Dataset):
    """
    Each sample is an n_bits-bit input. We define multiple XOR sub-tasks
    and combine them into a single label to produce multiple classes.
    Example: XOR(x1,x2) and XOR(x3,x4,x5) => up to 4 classes.
    """
    def __init__(self, n_samples=10000, n_bits=8):
        super().__init__()
        self.n_samples = n_samples
        self.n_bits = n_bits
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        X, Y = [], []
        for _ in range(self.n_samples):
            bits = np.random.randint(0,2,size=self.n_bits).astype(np.float32)
            # y1 = XOR of the first 2 bits
            y1 = bits[0] ^ bits[1]
            # y2 = XOR of the next 3 bits
            y2 = bits[2] ^ bits[3] ^ bits[4]
            label = (y1 << 1) + y2  # range 0..3
            X.append(bits)
            Y.append(label)
        return torch.tensor(X), torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


#############################
#  HELPER: get_dataloaders
#############################
def get_synthetic_dataloaders(
    dataset_name="line",
    train_size=8000,
    valid_size=1000,
    test_size=1000,
    image_size=10,
    n_orient_classes=8,
    n_bits=8,
    batch_size=64
):
    """
    Unified function that returns train/valid/test dataloaders 
    for the specified dataset.
    """
    total = train_size + valid_size + test_size

    if dataset_name == "line":
        dataset = LineDataset(n_samples=total, image_size=image_size)
    elif dataset_name == "center_surround":
        dataset = CenterSurroundDataset(n_samples=total, image_size=image_size)
    elif dataset_name == "orientation_bars":
        dataset = OrientationBarsDataset(
            n_samples=total,
            image_size=image_size,
            n_orient_classes=n_orient_classes
        )
    elif dataset_name == "multi_xor":
        dataset = MultiXORDataset(n_samples=total, n_bits=n_bits)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    train_data, valid_data, test_data = torch.utils.data.random_split(
        dataset, [train_size, valid_size, test_size]
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader