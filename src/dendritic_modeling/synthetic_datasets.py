# synthetic_datasets.py

import os
import shutil
from copy import deepcopy

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, TensorDataset, random_split
from torchvision import datasets, transforms
from torchvision.datasets import MNIST

from typing import List, Dict, Optional, Union

########################################
# Original tasks: line, center_surround, orientation_bars, multi_xor, plus standard MNIST
########################################


# Create a centralized function to get or create a data directory
def get_data_directory(data_type, custom_path=None):
    """
    Get or create a data directory for the specified data type.

    Args:
        data_type (str): Type of data (e.g., 'mnist', 'cifar')
        custom_path (str, optional): Custom path provided by the user

    Returns:
        str: Path to the data directory
    """
    if custom_path:
        # Use the custom path if provided
        data_path = os.path.join(custom_path, data_type)
    else:
        # Use 'data' directory in the current working directory
        data_path = os.path.join("data", data_type)

    # Ensure the directory exists
    os.makedirs(data_path, exist_ok=True)

    return data_path


class LineDataset(Dataset):
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
            if label == 1:
                row = np.random.randint(0, self.image_size)
                img[row, :] = 1.0
            elif label == 2:
                col = np.random.randint(0, self.image_size)
                img[:, col] = 1.0
            X.append(img.flatten())
            Y.append(label)
        return torch.tensor(np.stack(X)), torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class CenterSurroundDataset(Dataset):
    def __init__(self, n_samples=10000, image_size=15):
        super().__init__()
        self.n_samples = n_samples
        self.image_size = image_size
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        X, Y = [], []
        radius = self.image_size // 4
        cx = cy = self.image_size // 2
        for _ in range(self.n_samples):
            X.append(
                np.zeros((self.image_size, self.image_size), dtype=np.float32).flatten()
            )
            label = np.random.choice([0, 1, 2])
            img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            if label == 0:
                val = np.random.rand() * 0.2 + 0.4
                img[:] = val
            elif label == 1:
                for i in range(self.image_size):
                    for j in range(self.image_size):
                        dist = np.sqrt((i - cx) ** 2 + (j - cy) ** 2)
                        if dist < radius:
                            img[i, j] = 1.0
                        else:
                            img[i, j] = 0.0
            else:
                for i in range(self.image_size):
                    for j in range(self.image_size):
                        dist = np.sqrt((i - cx) ** 2 + (j - cy) ** 2)
                        if dist < radius:
                            img[i, j] = 0.0
                        else:
                            img[i, j] = 1.0
            X[-1] = img.flatten()
            Y.append(label)
        return torch.tensor(np.stack(X)), torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class OrientationBarsDataset(Dataset):
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


class MultiXORDataset(Dataset):
    def __init__(self, n_samples=10000, n_bits=8):
        super().__init__()
        self.n_samples = n_samples
        self.n_bits = n_bits
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        X, Y = [], []
        for _ in range(self.n_samples):
            bits = np.random.randint(0, 2, size=self.n_bits).astype(np.int32)
            y1 = bits[0] ^ bits[1]
            y2 = bits[2] ^ bits[3] ^ bits[4]
            label = (y1 << 1) + y2
            X.append(bits)
            Y.append(label)
        return torch.tensor(np.stack(X)), torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_mnist_as_datasets(
    train_valid_split=0.8,
    flatten=False,
    normalize=False,
    data_path=None
):
    """
    Load the MNIST dataset and split into train, validation, and test sets.

    Args:
        train_valid_split (float): Proportion of training data to use for training (vs validation)
        data_path (str, optional): Path to the data directory

    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    mnist_dir = get_data_directory("mnist", data_path)

    tf_list = [transforms.ToTensor()]
    if normalize:
        tf_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    transform = transforms.Compose(tf_list)

    train_dataset = datasets.MNIST(
        root=mnist_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=mnist_dir, train=False, download=True, transform=transform
    )

    train_size = int(len(train_dataset) * train_valid_split)
    valid_size = len(train_dataset) - train_size

    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, valid_size]
    )

    x_train = torch.stack([img for img, _ in train_dataset])
    y_train = torch.tensor([label for _, label in train_dataset])

    x_valid = torch.stack([img for img, _ in valid_dataset])
    y_valid = torch.tensor([label for _, label in valid_dataset])

    x_test = torch.stack([img for img, _ in test_dataset])
    y_test = torch.tensor([label for _, label in test_dataset])

    if flatten:
        x_train = x_train.view(x_train.shape[0], -1)
        x_valid = x_valid.view(x_valid.shape[0], -1)
        x_test = x_test.view(x_test.shape[0], -1)

    train_ds = torch.utils.data.TensorDataset(x_train, y_train)
    valid_ds = torch.utils.data.TensorDataset(x_valid, y_valid)
    test_ds = torch.utils.data.TensorDataset(x_test, y_test)

    return train_ds, valid_ds, test_ds


def load_MNIST_modulo10(
    shuffle_iterations=2, 
    train_valid_split=1, 
    flatten=False,
    normalize=False,
    cache_dir=None
):
    train_ds, valid_ds, test_ds = load_mnist_as_datasets(
        train_valid_split=train_valid_split, 
        flatten=flatten, 
        normalize=normalize, 
        data_path=cache_dir
    )
    data = {"train": train_ds, "valid": valid_ds, "test": test_ds}

    for key, ds in data.items():
        ix = torch.randperm(len(ds))
        ix0 = ix[: int(len(ds) / 2)]
        ix1 = ix[int(len(ds) / 2) :]

        input0, label0 = ds[ix0]
        input1, label1 = ds[ix1]
        assert label0.shape[0] == label1.shape[0]

        inputs = []
        labels = []
        for _ in range(shuffle_iterations):
            inputs.append(torch.cat([input0, input1], dim=-1))
            labels.append((label0 + label1) % 10)

            ix0 = torch.randperm(input0.shape[0])
            input0 = input0[ix0]
            label0 = label0[ix0]

            ix1 = torch.randperm(input1.shape[0])
            input1 = input1[ix1]
            label1 = label1[ix1]

        inputs = torch.cat(inputs, dim=0)
        labels = torch.cat(labels, dim=0)

        data[key] = TensorDataset(inputs, labels)

    return data


class InfoShuntingDataset(Dataset):
    """
    half = Gaussian, half = Uniform; label=0 or 1
    """

    def __init__(self, n_samples=20000, input_dim=10):
        super().__init__()
        self.n_samples = n_samples
        self.input_dim = input_dim
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        half = self.n_samples // 2
        gauss = np.random.normal(0, 1, size=(half, self.input_dim)).astype(np.float32)
        unif = np.random.uniform(
            -1, 1, size=(self.n_samples - half, self.input_dim)
        ).astype(np.float32)
        Xfull = np.concatenate([gauss, unif], axis=0)
        Yfull = np.concatenate([np.zeros(half), np.ones(self.n_samples - half)])
        idx = np.arange(self.n_samples)
        np.random.shuffle(idx)
        Xfull = Xfull[idx]
        Yfull = Yfull[idx]
        return torch.tensor(Xfull), torch.tensor(Yfull, dtype=torch.long)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class NoisyLineDataset(Dataset):
    """
    line dataset but add random Gaussian noise
    """

    def __init__(self, n_samples=10000, image_size=10, noise_level=0.2):
        super().__init__()
        self.n_samples = n_samples
        self.image_size = image_size
        self.noise_level = noise_level
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        X, Y = [], []
        for _ in range(self.n_samples):
            img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            label = np.random.choice([0, 1, 2])
            if label == 1:
                row = np.random.randint(0, self.image_size)
                img[row, :] = 1.0
            elif label == 2:
                col = np.random.randint(0, self.image_size)
                img[:, col] = 1.0
            noise = np.random.normal(0, self.noise_level, size=img.shape)
            img += noise.astype(np.float32)
            X.append(img.flatten())
            Y.append(label)
        return torch.tensor(np.stack(X)), torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SparseMNISTDataset(Dataset):
    """
    binarize MNIST
    """

    def __init__(self, base_mnist, threshold=0.3):
        super().__init__()
        self.base_mnist = base_mnist
        self.threshold = threshold

    def __len__(self):
        return len(self.base_mnist)

    def __getitem__(self, idx):
        img, label = self.base_mnist[idx]
        arr = np.array(img, dtype=np.float32) / 255.0
        bin_arr = (arr > self.threshold).astype(np.float32)
        return torch.tensor(bin_arr.flatten(), dtype=torch.float32), label


class ContextualFigureGroundMNIST(Dataset):
    """
    left half random noise, right half original digit
    """

    def __init__(self, base_mnist, noise_level=0.1):
        super().__init__()
        self.base_mnist = base_mnist
        self.noise_level = noise_level

    def __len__(self):
        return len(self.base_mnist)

    def __getitem__(self, idx):
        img, label = self.base_mnist[idx]
        img_arr = np.array(img, dtype=np.float32)
        w = 28
        new_img = np.random.rand(w, w) * self.noise_level
        boundary = w // 2
        new_img[:, boundary:] = img_arr[:, boundary:]
        return torch.tensor(new_img.flatten(), dtype=torch.float32), label


class MultiTaskLineDataset(Dataset):
    """
    stub: half data is line dataset (0,1,2), second half inverts labels
    """

    def __init__(self, total_size=12000, image_size=10):
        super().__init__()
        self.total_size = total_size
        self.image_size = image_size
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        half = self.total_size // 2
        Xlist, Ylist = [], []
        for _ in range(half):
            img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            label = np.random.choice([0, 1, 2])
            if label == 1:
                row = np.random.randint(0, self.image_size)
                img[row, :] = 1.0
            elif label == 2:
                col = np.random.randint(0, self.image_size)
                img[:, col] = 1.0
            Xlist.append(img.flatten())
            Ylist.append(label)
        for _ in range(self.total_size - half):
            img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            label = np.random.choice([0, 1, 2])
            if label == 1:
                row = np.random.randint(0, self.image_size)
                img[row, :] = 1.0
            elif label == 2:
                col = np.random.randint(0, self.image_size)
                img[:, col] = 1.0
            new_label = {0: 0, 1: 2, 2: 1}[label]
            Xlist.append(img.flatten())
            Ylist.append(new_label + 3)
        Xarr = np.stack(Xlist).astype(np.float32)
        Yarr = np.array(Ylist, dtype=np.int64)
        return torch.tensor(Xarr), torch.tensor(Yarr)

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class CompositionalMNIST(Dataset):
    """
    merges two MNIST digits horizontally => label = digit1*10+digit2
    """

    def __init__(self, base_mnist, n_samples=20000):
        super().__init__()
        self.base_mnist = base_mnist
        self.n_samples = n_samples
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        length = len(self.base_mnist)
        Xlist = []
        Ylist = []
        for _ in range(self.n_samples):
            idx1 = np.random.randint(0, length)
            idx2 = np.random.randint(0, length)
            img1, lbl1 = self.base_mnist[idx1]
            img2, lbl2 = self.base_mnist[idx2]
            arr1 = np.array(img1, dtype=np.float32)
            arr2 = np.array(img2, dtype=np.float32)
            comp = np.concatenate([arr1, arr2], axis=1)
            Xlist.append(comp.flatten())
            Ylist.append(lbl1 * 10 + lbl2)
        Xarr = np.stack(Xlist).astype(np.float32)
        Yarr = np.array(Ylist, dtype=np.int64)
        return torch.tensor(Xarr), torch.tensor(Yarr)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_cifar10_as_datasets(
    train_valid_split=1.0,
    flatten=False,
    normalize=False,
    data_path=None
):
    """
    Load the CIFAR-10 dataset and split into train, validation, and test sets.

    Args:
        train_valid_split (float): Proportion of training data to use for training (vs validation)
        data_path (str, optional): Path to the data directory

    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    tf_list = [transforms.ToTensor()]
    if normalize:
        tf_list.append(transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)))
    
    transform = transforms.Compose(tf_list)

    cifar_dir = get_data_directory("cifar", data_path)

    train_dataset = datasets.CIFAR10(
        root=cifar_dir, train=True, download=True, transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root=cifar_dir, train=False, download=True, transform=transform
    )

    def convert_to_tensor(dataset):
        data = []
        labels = []
        for img, label in dataset:
            data.append(img)
            labels.append(label)
        data_tensor = torch.stack(data)
        label_tensor = torch.tensor(labels)
        if flatten:
            data_tensor = data_tensor.view(data_tensor.size(0), -1)
        return data_tensor, label_tensor

    if train_valid_split < 1.0:
        train_size = int(len(train_dataset) * train_valid_split)
        valid_size = len(train_dataset) - train_size

        train_dataset, valid_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, valid_size]
        )

        x_train, y_train = convert_to_tensor(train_dataset)
        x_valid, y_valid = convert_to_tensor(valid_dataset)
        x_test, y_test = convert_to_tensor(test_dataset)

        train_ds = torch.utils.data.TensorDataset(x_train, y_train)
        valid_ds = torch.utils.data.TensorDataset(x_valid, y_valid)
        test_ds = torch.utils.data.TensorDataset(x_test, y_test)
    else:
        x_train, y_train = convert_to_tensor(train_dataset)
        x_test, y_test = convert_to_tensor(test_dataset)
        train_ds = torch.utils.data.TensorDataset(x_train, y_train)
        valid_ds = train_ds
        test_ds = torch.utils.data.TensorDataset(x_test, y_test)

    return train_ds, valid_ds, test_ds


########################################
# The single get_unified_datasets(...) function
########################################


def applyRandomFlip(images, labels, setAll=False, setValue=0):
    """
    This function modifies images and labels in-place to flip the label
    or shift it. The 'first 3 pixels' logic sets a code:
      pixel0=10 => normal
      pixel1=10 => flip to 9-digit
      pixel2=10 => shift by -1 mod 10
    """
    images_flip = images.clone()
    labels_flip = labels.clone()
    if setAll:
        # set all flips to a single mode
        if setValue == 0:
            # normal
            images_flip[:, 0] = 10
            images_flip[:, 1] = 0
            images_flip[:, 2] = 0
        elif setValue == 1:
            # 9 - label
            labels_flip = 9 - labels_flip
            images_flip[:, 0] = 0
            images_flip[:, 1] = 10
            images_flip[:, 2] = 0
        else:
            # label-1 mod 10
            labels_flip = (labels_flip - 1) % 10
            images_flip[:, 0] = 0
            images_flip[:, 1] = 0
            images_flip[:, 2] = 10
        return images_flip, labels_flip, None
    else:
        # random assignment
        randFlips = torch.randint(
            0, 3, (images_flip.shape[0],), device=images_flip.device
        )
        # mode=0 => normal, mode=1 => flip 9-lbl, mode=2 => lbl-1 mod10
        # set the first pixel code
        images_flip[randFlips == 0, 0] = 10
        images_flip[randFlips == 0, 1] = 0
        images_flip[randFlips == 0, 2] = 0

        labels_flip[randFlips == 1] = 9 - labels_flip[randFlips == 1]
        images_flip[randFlips == 1, 0] = 0
        images_flip[randFlips == 1, 1] = 10
        images_flip[randFlips == 1, 2] = 0

        labels_flip[randFlips == 2] = (labels_flip[randFlips == 2] - 1) % 10
        images_flip[randFlips == 2, 0] = 0
        images_flip[randFlips == 2, 1] = 0
        images_flip[randFlips == 2, 2] = 10

        return images_flip, labels_flip, randFlips


class RandomFlipMNISTDataset(Dataset):
    """
    Loads standard MNIST and applies the applyRandomFlip() transform
    to each sample in __getitem__ if random_flip=True.
    This modifies the first 3 pixels and possibly flips/shifts the label.
    """

    def __init__(
        self,
        split="train",
        transform=None,
        data_path=None,
        random_flip=True,
        setAll=None,
        setValue=0,
    ):
        """
        split: 'train', 'valid', or 'test'
        random_flip: if False, no flipping is applied
        setAll: if an integer in [0,1,2], then all flips set to that mode
        setValue: same as above
        """
        super().__init__()
        self.transform = transform
        self.random_flip = random_flip
        self.setAll = setAll
        self.setValue = setValue
        # load full MNIST or we can do partial
        mnist_dir = get_data_directory("mnist", data_path)
        if split == "test":
            base_ds = datasets.MNIST(root=mnist_dir, train=False, download=True)
        else:
            base_ds = datasets.MNIST(root=mnist_dir, train=True, download=True)

        # For simplicity, we'll just store the entire split in memory
        data_tensors = []
        label_tensors = []
        for img, lbl in base_ds:
            arr = np.array(img, dtype=np.float32)
            arr = arr.reshape(-1)
            data_tensors.append(arr)
            label_tensors.append(lbl)
        X = torch.tensor(np.stack(data_tensors))
        Y = torch.tensor(label_tensors, dtype=torch.long)

        # If user wants a valid split, we do that
        if split == "train":
            # we'll take first 50k
            self.X = X[:50000]
            self.Y = Y[:50000]
        elif split == "valid":
            # next 10k
            self.X = X[50000:]
            self.Y = Y[50000:]
        else:  # test
            self.X = X
            self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        if self.random_flip:
            if self.setAll is not None:
                x, y, _ = applyRandomFlip(
                    x.unsqueeze(0), y.unsqueeze(0), setAll=True, setValue=self.setAll
                )
                x = x.squeeze(0)
                y = y.squeeze(0)
            else:
                x, y, _ = applyRandomFlip(x.unsqueeze(0), y.unsqueeze(0), setAll=False)
                x = x.squeeze(0)
                y = y.squeeze(0)
        return x, y


def poisson_generator_dataset(
    base_dataset: str,
    rate_multiplier: float = 1,
    n_samples: int = 10,
    train_valid_split: float = 0.8,
    flatten: bool = False,
    data_path: str = None
):
    """
    Generate a dataset of Poisson-distributed samples from a base dataset.

    Args:
        base_dataset: The base dataset to sample from
        n_samples: The number of samples to generate
        train_valid_split: The proportion of samples to use for training vs validation
        flatten: Whether to flatten the samples
        normalize: Whether to normalize the samples
        data_path: The path to save the generated dataset

    Returns:
        A dataset of Poisson-distributed samples
    """
    # Load the base dataset
    if base_dataset == "mnist":
        train_ds, valid_ds, test_ds = load_mnist_as_datasets(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=False,
            data_path=data_path
        )    
    elif base_dataset == "cifar10":
        train_ds, valid_ds, test_ds = load_cifar10_as_datasets(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=False,
            data_path=data_path
        )
    else:
        raise ValueError(f"Unknown base dataset: {base_dataset}")
    
    data = {"train": train_ds, "valid": valid_ds, "test": test_ds}

    for key, ds in data.items():
        inputs, labels = ds[:]

        scaled_inputs = inputs * rate_multiplier
        noisy_scaled_inputs = torch.poisson(
            scaled_inputs[None, ...].expand(n_samples, *inputs.shape)
        )
        noisy_inputs = noisy_scaled_inputs / rate_multiplier
        noisy_inputs = noisy_inputs.flatten(0, 1)
        labels = labels[None, ...].expand(n_samples, *labels.shape)
        labels = labels.flatten(0, 1)

        data[key] = TensorDataset(noisy_inputs, labels)

    return data


class OrthoNet(torch.nn.Module):
    """ Orthogonal sigmoid network (Invertible network) """

    def __init__(self,d, nb_layers, nonlinearity='sigmoid'):
        super().__init__()
        for i in range(nb_layers):
            # register as we don't want to update it
            self.register_buffer(f'Q_{i}', self.rand_ortho(d))
        self.nb_layer = nb_layers
        if nonlinearity is None:
            self.nonlinearity = torch.tanh
        elif nonlinearity == 'sigmoid':
            self.nonlinearity = torch.sigmoid
        elif nonlinearity == 'tanh':
            self.nonlinearity = torch.tanh
        else:
            raise ValueError(f"Invalid nonlinearity: {nonlinearity}")


    def forward(self,x):
        for i in range(self.nb_layer):
            Q = getattr(self,f'Q_{i}')
            x = self.nonlinearity(x @ Q)
        return x
    
    def rand_ortho(self,d):
        """ Generate a random orthogonal matrix of dimension d """
        Q, R = torch.linalg.qr(torch.randn(d,d))
        return Q
    

class OrthoNetRepeat(torch.nn.Module):
    """ Orthogonal sigmoid network with repeated layer weights plus noise """

    def __init__(self, d, nb_layers, nonlinearity=None, noise_std=0.1):
        super().__init__()
        # Generate the first layer's orthogonal matrix
        base_Q = self.rand_ortho(d)
        self.register_buffer('Q_0', base_Q)
        
        # Create subsequent layers by adding noise to the first layer
        for i in range(1, nb_layers):
            noise = torch.randn_like(base_Q) * noise_std
            noisy_Q = base_Q + noise
            # Re-orthogonalize the matrix
            Q, R = torch.linalg.qr(noisy_Q)
            self.register_buffer(f'Q_{i}', Q)
            
        self.nb_layer = nb_layers
        if nonlinearity is None:
            self.nonlinearity = torch.tanh
        elif nonlinearity == 'sigmoid':
            self.nonlinearity = torch.sigmoid
        elif nonlinearity == 'tanh':
            self.nonlinearity = torch.tanh
        else:
            raise ValueError(f"Invalid nonlinearity: {nonlinearity}")

    def forward(self, x):
        for i in range(self.nb_layer):
            Q = getattr(self, f'Q_{i}')
            x = self.nonlinearity(x @ Q)
        return x
    
    def rand_ortho(self, d):
        """ Generate a random orthogonal matrix of dimension d """
        Q, R = torch.linalg.qr(torch.randn(d, d))
        return Q
    
class OrthoNetDataset(Dataset):
    """
    Dataset that generates data using invertible nonlinear transformations.
    Input data is uniformly random in [-1,1]^d, transformed through an OrthoNet network.
    Labels are the original inputs, making this an inverse problem dataset.
    """
    def __init__(self, n_samples=10000, input_dim=10, n_layers=5, nonlinearity='sigmoid', 
                 positive_input=True, seed=None, repeat_layers=False, noise_std=0.1):
        super().__init__()
        self.n_samples = n_samples
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.nonlinearity = nonlinearity
        self.positive_input = positive_input
        self.repeat_layers = repeat_layers
        self.noise_std = noise_std
        self.seed = seed

        if seed is not None:
            self.set_seed(seed)
            
        self.data, self.labels = self._generate_data()

    def set_seed(self, seed):
        """Set the seed for the network."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _generate_data(self):
        # Generate random input data in [-1,1]^d
        if self.positive_input:
            X = torch.rand(self.n_samples, self.input_dim)
        else:
            X = 2 * torch.rand(self.n_samples, self.input_dim) - 1
        
        # Create and apply transformation
        if self.repeat_layers:
            F = OrthoNetRepeat(self.input_dim, self.n_layers, 
                             self.nonlinearity, self.noise_std)
        else:
            F = OrthoNet(self.input_dim, self.n_layers, self.nonlinearity)
            
        with torch.no_grad():
            Y = F(X)
        
        return Y, X

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # if isinstance(idx, slice):
        #     return self.data[idx], self.labels[idx]
        return self.data[idx], self.labels[idx]

class CorrelatedFeatureDataset(Dataset):
    """
    Create a binary classification dataset with heterogeneous correlation structures.
    
    This dataset creates features with log-normal distributions to ensure positivity,
    and allows for defining heterogeneous correlation structures to test dendritic
    network inductive biases toward clustering correlated inputs.
    """
    
    def __init__(
        self, 
        input_dim: int,
        n_samples: int,
        correlation_blocks: Optional[List[Dict]] = None,
        block_proportions: Optional[List[float]] = None,
        high_corr_means: Optional[List[float]] = None,
        high_corr_stds: Optional[List[float]] = None,
        low_corr_means: Optional[List[float]] = None,
        low_corr_stds: Optional[List[float]] = None,
        shuffle_dims: bool = True,
        per_pair_correlation: bool = False,
        normalize_features: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize the dataset with correlation structures.
        
        Parameters:
        -----------
        input_dim : int
            Dimensionality of input features
        n_samples : int
            Total number of samples
        correlation_blocks : List[Dict], optional
            Pre-defined correlation blocks. Each dict should have keys:
            - 'dims': list of feature indices forming the block
            - 'high_corr': correlation value(s) for high-corr class
            - 'low_corr': correlation value(s) for low-corr class
            If provided, block_proportions and related parameters are ignored.
        block_proportions : List[float], optional
            Proportion of dimensions for each block (should sum to ≤ 1.0)
        high_corr_means : List[float], optional
            Mean correlation value for each block in high correlation class
        high_corr_stds : List[float], optional
            Standard deviation of correlation values for high correlation class
        low_corr_means : List[float], optional
            Mean correlation value for each block in low correlation class
        low_corr_stds : List[float], optional
            Standard deviation of correlation values for low correlation class
        shuffle_dims : bool, default=True
            Whether to randomly assign dimensions to blocks
        per_pair_correlation : bool, default=False
            If True, generate unique correlation values for each feature pair
        normalize_features : bool, default=True
            Whether to normalize features to have mean 1.0
        seed : int, optional
            Random seed for reproducibility
        """
        super().__init__()
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.input_dim = input_dim
        self.n_samples = n_samples
        self.n_samples_per_class = n_samples // 2
        self.normalize_features = normalize_features
        
        # If correlation blocks not provided, generate them from parameters
        if correlation_blocks is None:
            if any(param is None for param in [block_proportions, high_corr_means, 
                                              high_corr_stds, low_corr_means, low_corr_stds]):
                raise ValueError("Either correlation_blocks or all block parameters must be provided")
                
            correlation_blocks = self.generate_correlation_blocks(
                input_dim=input_dim,
                block_proportions=block_proportions,
                high_corr_means=high_corr_means,
                high_corr_stds=high_corr_stds,
                low_corr_means=low_corr_means,
                low_corr_stds=low_corr_stds,
                shuffle_dims=shuffle_dims,
                per_pair_correlation=per_pair_correlation,
                seed=seed
            )
        
        # Store correlation blocks for reference
        self.correlation_blocks = correlation_blocks
        
        # Create the dataset based on correlation blocks
        self._create_dataset()
    
    def _create_dataset(self):
        """
        Create the dataset based on correlation blocks.
        """
        # Generate independent log-normal base variables
        high_X_base = np.exp(np.random.randn(self.n_samples_per_class, self.input_dim))
        low_X_base = np.exp(np.random.randn(self.n_samples_per_class, self.input_dim))
        
        # Initialize data arrays
        self.data_high = np.zeros((self.n_samples_per_class, self.input_dim))
        self.data_low = np.zeros((self.n_samples_per_class, self.input_dim))
        
        # Set of dimensions already processed
        processed_dims = set()
        
        # Process each correlation block
        for block in self.correlation_blocks:
            dims = block['dims']
            high_corr = block['high_corr']
            low_corr = block['low_corr']
            
            # Track processed dimensions
            processed_dims.update(dims)
            
            if len(dims) < 2:
                # Single dimension has no correlation to enforce
                for dim in dims:
                    self.data_high[:, dim] = high_X_base[:, dim]
                    self.data_low[:, dim] = low_X_base[:, dim]
                continue
            
            # For each block, use the first dimension as reference
            ref_dim = dims[0]
            
            # Set reference dimension directly
            self.data_high[:, ref_dim] = high_X_base[:, ref_dim]
            self.data_low[:, ref_dim] = low_X_base[:, ref_dim]
            
            # If correlation value is a single number (uniform within block)
            if isinstance(high_corr, (float, int)) and isinstance(low_corr, (float, int)):
                for i in dims[1:]:
                    # High correlation class
                    self.data_high[:, i] = (high_corr * high_X_base[:, ref_dim] + 
                                          np.sqrt(1 - high_corr**2) * high_X_base[:, i])
                    
                    # Low correlation class
                    self.data_low[:, i] = (low_corr * low_X_base[:, ref_dim] + 
                                         np.sqrt(1 - low_corr**2) * low_X_base[:, i])
            
            # If correlation is a matrix (unique values per pair)
            elif isinstance(high_corr, np.ndarray) and isinstance(low_corr, np.ndarray):
                for idx, i in enumerate(dims[1:], 1):
                    ref_idx = 0  # Index in the correlation matrix for the reference dimension
                    
                    # Get correlation value from the matrix
                    h_corr = high_corr[ref_idx, idx]
                    l_corr = low_corr[ref_idx, idx]
                    
                    # High correlation class
                    self.data_high[:, i] = (h_corr * high_X_base[:, ref_dim] + 
                                          np.sqrt(1 - h_corr**2) * high_X_base[:, i])
                    
                    # Low correlation class
                    self.data_low[:, i] = (l_corr * low_X_base[:, ref_dim] + 
                                         np.sqrt(1 - l_corr**2) * low_X_base[:, i])
        
        # Handle any remaining dimensions not in any block (independent)
        remaining_dims = set(range(self.input_dim)) - processed_dims
        for dim in remaining_dims:
            self.data_high[:, dim] = high_X_base[:, dim]
            self.data_low[:, dim] = low_X_base[:, dim]
        
        # Normalize features if requested
        if self.normalize_features:
            for i in range(self.input_dim):
                # Normalize each feature dimension to have mean 1
                self.data_high[:, i] /= np.mean(self.data_high[:, i])
                self.data_low[:, i] /= np.mean(self.data_low[:, i])
        
        # Concatenate data
        self.data = np.vstack([self.data_high, self.data_low])
        
        # Create binary labels (1 for high correlation, 0 for low correlation)
        self.labels = np.hstack([np.ones(self.n_samples_per_class), 
                                np.zeros(self.n_samples_per_class)])
        
        # Convert to torch tensors
        self.data = torch.from_numpy(self.data).float()
        self.labels = torch.from_numpy(self.labels).long()
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.n_samples
    
    def __getitem__(self, idx):
        """Return a data sample and its label."""
        return self.data[idx], self.labels[idx]
    
    @staticmethod
    def generate_correlation_blocks(
        input_dim: int,
        block_proportions: List[float],
        high_corr_means: List[float],
        high_corr_stds: List[float],
        low_corr_means: List[float],
        low_corr_stds: List[float],
        shuffle_dims: bool = True,
        per_pair_correlation: bool = False,
        seed: Optional[int] = None
    ) -> List[Dict]:
        
        if seed is not None:
            np.random.seed(seed)
        
        # Input validation
        if len(block_proportions) != len(high_corr_means) or \
          len(block_proportions) != len(high_corr_stds) or \
          len(block_proportions) != len(low_corr_means) or \
          len(block_proportions) != len(low_corr_stds):
            raise ValueError("All input lists must have the same length")
        
        if sum(block_proportions) > 1.0:
            raise ValueError("Sum of block_proportions should not exceed 1.0")
        
        # Calculate number of dimensions for each block
        block_dims = [int(p * input_dim) for p in block_proportions]
        
        # Adjust the last block to account for rounding errors
        total_allocated = sum(block_dims)
        if total_allocated < int(sum(block_proportions) * input_dim):
            # Add remaining dimensions to the last block
            remaining = int(sum(block_proportions) * input_dim) - total_allocated
            block_dims[-1] += remaining
        
        # Create array of dimension indices
        all_dims = np.arange(input_dim)
        
        # Shuffle dimensions if requested
        if shuffle_dims:
            np.random.shuffle(all_dims)
        
        # Allocate dimensions to blocks
        blocks = []
        start_idx = 0
        
        for i, block_size in enumerate(block_dims):
            if block_size < 2:
                print(f"Warning: Block {i} has fewer than 2 dimensions, skipping")
                continue
                
            # Select dimensions for this block
            block_dims_indices = all_dims[start_idx:start_idx + block_size]
            start_idx += block_size
            
            # Generate correlation values
            high_corr = CorrelatedFeatureDataset._generate_correlation_values(
                block_size,
                high_corr_means[i],
                high_corr_stds[i],
                per_pair_correlation
            )
            
            low_corr = CorrelatedFeatureDataset._generate_correlation_values(
                block_size,
                low_corr_means[i],
                low_corr_stds[i],
                per_pair_correlation
            )
            
            # Create block dictionary
            blocks.append({
                'dims': block_dims_indices.tolist(),
                'high_corr': high_corr,
                'low_corr': low_corr
            })
        
        return blocks
    
    @staticmethod
    def _generate_correlation_values(
        block_size: int,
        mean: float,
        std: float,
        per_pair_correlation: bool
    ) -> Union[float, np.ndarray]:
        """
        Generate correlation values for a block.
        
        Parameters:
        -----------
        block_size : int
            Number of dimensions in the block
        mean : float
            Mean correlation value
        std : float
            Standard deviation of correlation values
        per_pair_correlation : bool
            If True, generate unique correlation values for each feature pair
            
        Returns:
        --------
        Union[float, np.ndarray]
            Either a single correlation value or a matrix of correlation values
        """
        if per_pair_correlation and block_size > 1:
            # Initialize correlation matrix (identity matrix)
            corr_matrix = np.eye(block_size)
            
            # Fill upper triangle with random correlations
            for i in range(block_size):
                for j in range(i+1, block_size):
                    # Generate random correlation value
                    corr_value = np.random.normal(mean, std)
                    # Clip to valid range
                    corr_value = np.clip(corr_value, 0.0, 0.99)
                    # Make matrix symmetric
                    corr_matrix[i, j] = corr_value
                    corr_matrix[j, i] = corr_value
            
            return corr_matrix
        else:
            # Generate a single correlation value
            corr_value = np.random.normal(mean, std)
            # Clip to valid range
            corr_value = np.clip(corr_value, 0.0, 0.99)
            return corr_value
    
    def get_correlation_matrix(self, class_idx=None):
        """Get correlation matrix for specified class."""
        if class_idx == 1:
            data = self.data_high
        elif class_idx == 0:
            data = self.data_low
        else:
            data = self.data.numpy()
        
        return np.corrcoef(data.T)
    
    def visualize_correlation_matrix(self, class_idx=None, ax=None):
        """Visualize correlation matrix for specified class."""
        corr = self.get_correlation_matrix(class_idx)
        title = {1: "High Correlation Class", 
                0: "Low Correlation Class",
                None: "Full Dataset"}[class_idx]
        return plot_correlation_matrix(corr, title, ax)
    
    def visualize_correlation_structure(self, class_name="high", ax=None):
        """Visualize planned correlation structure."""
        return plot_correlation_structure(
            self.correlation_blocks,
            self.input_dim,
            class_name,
            ax
        )
    
    def plot_sample_distributions(self, n_dims=4, class_idx=None):
        """Plot feature distributions."""
        return plot_feature_distributions(
            self.data_high,
            self.data_low,
            n_dims,
            class_idx
        )

def get_unified_datasets(task_cfg, train_cfg):
    """
    Get datasets based on configuration.

    Args:
        task_cfg: Task configuration
        train_cfg: Training configuration

    Returns:
        dict: Dictionary with 'train', 'valid', and 'test' datasets
    """
    try:
        dataset_name = task_cfg["dataset"]
        data_path = task_cfg.get("data_path", None)
        p = task_cfg.get("parameters", {})
    except TypeError:
        dataset_name = task_cfg.dataset
        data_path = getattr(task_cfg, "data_path", None)
        p = task_cfg.parameters

    # If data_path is empty string, set it to None
    if data_path == "":
        data_path = None

    # If data_path is None, use the outputs directory in project root instead of current working directory
    if data_path is None:
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        data_path = os.path.join(project_root, "outputs", "data")
        os.makedirs(data_path, exist_ok=True)

    train_valid_split = getattr(train_cfg, "train_valid_split", 1.0)
    train_size = p.get("train_size", 8000)
    valid_size = p.get("valid_size", 1000)
    test_size = p.get("test_size", 1000)
    image_size = p.get("image_size", 784)
    flatten = p.get("flatten", True)
    normalize = p.get("normalize", False)
    base_dataset = p.get("base_dataset", "mnist")
    rate_multiplier = p.get("rate_multiplier", 1)
    n_samples = p.get("n_samples", 10)
    n_orient_classes = p.get("n_orient_classes", 8)
    n_bits = p.get("n_bits", 8)
    total = train_size + valid_size + test_size

    if dataset_name == "line":
        full_ds = LineDataset(n_samples=total, image_size=image_size)
        return random_split(full_ds, [train_size, valid_size, test_size])
    elif dataset_name == "center_surround":
        full_ds = CenterSurroundDataset(n_samples=total, image_size=image_size)
        return random_split(full_ds, [train_size, valid_size, test_size])
    elif dataset_name == "orientation_bars":
        full_ds = OrientationBarsDataset(
            n_samples=total, image_size=image_size, n_orient_classes=n_orient_classes
        )
        return random_split(full_ds, [train_size, valid_size, test_size])
    elif dataset_name == "multi_xor":
        full_ds = MultiXORDataset(n_samples=total, n_bits=n_bits)
        return random_split(full_ds, [train_size, valid_size, test_size])
    elif dataset_name == "mnist":
        train_ds, valid_ds, test_ds = load_mnist_as_datasets(
            train_valid_split=train_valid_split, 
            flatten=flatten,
            normalize=normalize,
            data_path=data_path
        )
        return train_ds, valid_ds, test_ds
    elif dataset_name == "random_flip_mnist":
        # Our newly added dataset that uses RandomFlipMNISTDataset
        # We'll treat train vs valid vs test accordingly
        train_dataset = RandomFlipMNISTDataset(
            split="train", data_path=data_path, random_flip=True
        )
        valid_dataset = RandomFlipMNISTDataset(
            split="valid", data_path=data_path, random_flip=True
        )
        test_dataset = RandomFlipMNISTDataset(
            split="test", data_path=data_path, random_flip=True
        )
        return train_dataset, valid_dataset, test_dataset
    elif dataset_name == "mnist_modulo10":
        data = load_MNIST_modulo10(
            train_valid_split=train_valid_split, 
            flatten=flatten,
            normalize=normalize,
            cache_dir=data_path
        )
        train_ds = data["train"]
        if "valid" in data:
            valid_ds = data["valid"]
        else:
            valid_ds = data["test"]
        test_ds = data["test"]
        return train_ds, valid_ds, test_ds
    elif dataset_name == "info_shunting":
        full_ds = InfoShuntingDataset(n_samples=total, input_dim=10)
        return random_split(full_ds, [train_size, valid_size, test_size])
    elif dataset_name == "noise_resilience":
        full_ds = NoisyLineDataset(
            n_samples=total, image_size=image_size, noise_level=0.2
        )
        return random_split(full_ds, [train_size, valid_size, test_size])
    elif dataset_name == "feature_selectivity":
        base_mnist = MNIST(root=data_path, train=True, download=True)
        full_ds = SparseMNISTDataset(base_mnist, threshold=0.3)
        return random_split(full_ds, [train_size, valid_size, test_size])
    elif dataset_name == "context_gating":
        base_mnist = MNIST(root=data_path, train=True, download=True)
        full_ds = ContextualFigureGroundMNIST(base_mnist, noise_level=0.1)
        return random_split(full_ds, [train_size, valid_size, test_size])
    elif dataset_name == "learning_dynamics":
        full_ds = MultiTaskLineDataset(total_size=total, image_size=image_size)
        return random_split(full_ds, [train_size, valid_size, test_size])
    elif dataset_name == "hierarchical_processing":
        base_mnist = MNIST(root=data_path, train=True, download=True)
        full_ds = CompositionalMNIST(base_mnist, n_samples=total)
        return random_split(full_ds, [train_size, valid_size, test_size])
    elif dataset_name == "cifar10":
        return load_cifar10_as_datasets(
            train_valid_split=train_valid_split, 
            flatten=flatten,
            normalize=normalize,
            data_path=data_path
        )
    elif dataset_name == "poisson_generator":
        data = poisson_generator_dataset(
            base_dataset=base_dataset,
            rate_multiplier=rate_multiplier,
            n_samples=n_samples,
            train_valid_split=train_valid_split,
            flatten=flatten,
            data_path=data_path
        )
        train_ds = data["train"]
        if "valid" in data:
            valid_ds = data["valid"]
        else:
            valid_ds = data["test"]
        test_ds = data["test"]
        return train_ds, valid_ds, test_ds
    elif dataset_name == "orthonet":
        input_dim = p.get('input_dim', 10)
        n_layers = p.get('n_layers', 5)
        nonlinearity = p.get('nonlinearity', 'sigmoid')
        repeat_layers = p.get('repeat_layers', False)
        noise_std = p.get('noise_std', 0.1)
        seed = getattr(train_cfg, 'seed', 0)
        full_ds = OrthoNetDataset(
            n_samples=total, 
            input_dim=input_dim, 
            n_layers=n_layers, 
            nonlinearity=nonlinearity, 
            seed=seed,
            repeat_layers=repeat_layers,
            noise_std=noise_std
        )
        return random_split(full_ds, [train_size, valid_size, test_size])
    elif dataset_name == "correlated_features":
        input_dim = p.get('input_dim', 10)
        n_samples = p.get('n_samples', 10000)
        block_proportions = p.get('block_proportions', [0.5])
        high_corr_means = p.get('high_corr_means', [0.8])
        high_corr_stds = p.get('high_corr_stds', [0.1])
        low_corr_means = p.get('low_corr_means', [0.2])
        low_corr_stds = p.get('low_corr_stds', [0.1])
        shuffle_dims = p.get('shuffle_dims', True)
        per_pair_correlation = p.get('per_pair_correlation', True)
        normalize_features = p.get('normalize_features', True)
        seed = getattr(train_cfg, 'seed', 0)
        full_ds = CorrelatedFeatureDataset(
            input_dim=input_dim,
            n_samples=n_samples,
            block_proportions=block_proportions,
            high_corr_means=high_corr_means,
            high_corr_stds=high_corr_stds,
            low_corr_means=low_corr_means,
            low_corr_stds=low_corr_stds,
            shuffle_dims=shuffle_dims,
            per_pair_correlation=per_pair_correlation,
            normalize_features=normalize_features,
            seed=seed
        )
        return random_split(full_ds, [train_size, valid_size, test_size])
    
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


# Added function from the older codebase for splitting MNIST inputs.
def split_MNIST_inputs(inputs, labels):
    unique_label = torch.unique(labels)
    input_list = []
    for unique in unique_label:
        ix = torch.where(labels == unique, True, False)
        input_list.append(inputs[ix])
    return input_list
