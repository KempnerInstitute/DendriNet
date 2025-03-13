# synthetic_datasets.py
import os
import shutil
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset, random_split, TensorDataset

from torchvision.datasets import MNIST
import torchvision.transforms as T
import logging

logger = logging.getLogger(__name__)

########################################
# Original tasks: line, center_surround, orientation_bars, multi_xor, plus standard MNIST
########################################

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
            label = np.random.choice([0,1,2])
            if label==1:
                row = np.random.randint(0,self.image_size)
                img[row,:] = 1.0
            elif label==2:
                col = np.random.randint(0,self.image_size)
                img[:,col] = 1.0
            X.append(img.flatten())
            Y.append(label)
        return torch.tensor(np.stack(X)), torch.tensor(Y,dtype=torch.long)

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
            label = np.random.choice([0,1,2])
            img = np.zeros((self.image_size,self.image_size), dtype=np.float32)
            if label==0:
                val = np.random.rand()*0.2+0.4
                img[:]=val
            elif label==1:
                for i in range(self.image_size):
                    for j in range(self.image_size):
                        dist = np.sqrt((i-cx)**2+(j-cy)**2)
                        if dist<radius:
                            img[i,j]=1.0
                        else:
                            img[i,j]=0.0
            else:
                for i in range(self.image_size):
                    for j in range(self.image_size):
                        dist = np.sqrt((i-cx)**2+(j-cy)**2)
                        if dist<radius:
                            img[i,j]=0.0
                        else:
                            img[i,j]=1.0
            X.append(img.flatten())
            Y.append(label)
        return torch.tensor(np.stack(X)), torch.tensor(Y,dtype=torch.long)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class OrientationBarsDataset(Dataset):
    def __init__(self, n_samples=10000, image_size=16, n_orient_classes=8):
        super().__init__()
        self.n_samples=n_samples
        self.image_size=image_size
        self.n_orient_classes=n_orient_classes
        self.data,self.labels=self._generate_data()

    def _generate_data(self):
        X,Y=[],[]
        for _ in range(self.n_samples):
            label=np.random.randint(0,self.n_orient_classes)
            angle_deg=(180/self.n_orient_classes)*label
            angle_rad=np.deg2rad(angle_deg)
            img=np.zeros((self.image_size,self.image_size), dtype=np.float32)
            cx=cy=self.image_size//2
            for r in range(-cx,cx):
                x=int(cx+r*np.cos(angle_rad))
                y=int(cy+r*np.sin(angle_rad))
                if 0<=x<self.image_size and 0<=y<self.image_size:
                    img[y,x]=1.0
            X.append(img.flatten())
            Y.append(label)
        return torch.tensor(np.stack(X)), torch.tensor(Y,dtype=torch.long)
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        return self.data[idx],self.labels[idx]


class MultiXORDataset(Dataset):
    def __init__(self,n_samples=10000,n_bits=8):
        super().__init__()
        self.n_samples=n_samples
        self.n_bits=n_bits
        self.data,self.labels=self._generate_data()

    def _generate_data(self):
        X,Y=[],[]
        for _ in range(self.n_samples):
            bits=np.random.randint(0,2,size=self.n_bits).astype(np.int32)
            y1=bits[0]^bits[1]
            y2=bits[2]^bits[3]^bits[4]
            label=(y1<<1)+y2
            X.append(bits)
            Y.append(label)
        return torch.tensor(np.stack(X)), torch.tensor(Y,dtype=torch.long)
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_data_dir(data_path=None, dataset_type=None):
    """
    Centralized function to get or create a data directory.
    
    Args:
        data_path: Optional path override for data storage
        dataset_type: Subdirectory for specific dataset (e.g., 'mnist', 'cifar')
        
    Returns:
        Path to the data directory
    """
    if data_path is None:
        # Use a 'data' directory in the current working directory
        data_dir = os.path.join(os.getcwd(), 'data')
    else:
        data_dir = data_path
        
    # Create the base data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # If a dataset type is specified, create and return that subdirectory
    if dataset_type:
        dataset_dir = os.path.join(data_dir, dataset_type)
        os.makedirs(dataset_dir, exist_ok=True)
        return dataset_dir
    
    return data_dir

def get_inputs_labels_from_dataset(dataset):
    """
    Extract input tensors and labels from a PyTorch dataset.
    
    Args:
        dataset: A PyTorch Dataset object
        
    Returns:
        Tuple of (inputs, labels) tensors
    """
    # For datasets that return pre-made tensors
    if isinstance(dataset, TensorDataset):
        return dataset.tensors
    
    # For torchvision datasets like MNIST and CIFAR10
    if hasattr(dataset, 'data') and hasattr(dataset, 'targets'):
        # Convert data to tensor if it's not already
        if not isinstance(dataset.data, torch.Tensor):
            inputs = torch.tensor(dataset.data, dtype=torch.float32)
        else:
            inputs = dataset.data.float()
        
        # Flatten images if they're not already flat
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), -1)
            
        # Normalize to [0, 1] if data is in uint8 format
        if inputs.dtype == torch.uint8:
            inputs = inputs / 255.0
            
        # Convert targets to tensor if needed
        if not isinstance(dataset.targets, torch.Tensor):
            labels = torch.tensor(dataset.targets, dtype=torch.long)
        else:
            labels = dataset.targets
            
        return inputs, labels
    
    # For Subset datasets from random_split
    if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
        parent_inputs, parent_labels = get_inputs_labels_from_dataset(dataset.dataset)
        indices = dataset.indices
        return parent_inputs[indices], parent_labels[indices]
    
    # For general case, collect all data
    inputs = []
    labels = []
    for i in range(len(dataset)):
        x, y = dataset[i]
        # Handle case where x is an image
        if isinstance(x, torch.Tensor) and x.dim() > 1:
            x = x.view(-1)  # Flatten
        inputs.append(x)
        labels.append(y)
    
    # Stack inputs and convert labels
    inputs = torch.stack(inputs)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return inputs, labels

def load_mnist_as_datasets(train_valid_split=0.8, data_path=None):
    """
    Load the MNIST dataset and return train, validation, and test datasets.
    
    Args:
        train_valid_split: Ratio of training data to use for training (vs. validation)
        data_path: Optional override for where to store/find the data
        
    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset)
    """
    # Get the appropriate data directory
    data_dir = get_data_dir(data_path, 'mnist')
    
    # Download and load the dataset
    train_dataset = MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=T.Compose([
            T.ToTensor(),
        ])
    )
    
    test_dataset = MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=T.Compose([
            T.ToTensor(),
        ])
    )
    
    # Split training data into train and validation
    if train_valid_split < 1.0:
        train_size = int(len(train_dataset) * train_valid_split)
        valid_size = len(train_dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, valid_size])
    else:
        valid_dataset = test_dataset
    
    # Convert to tensor datasets for consistent interface
    train_inputs, train_labels = get_inputs_labels_from_dataset(train_dataset)
    valid_inputs, valid_labels = get_inputs_labels_from_dataset(valid_dataset)
    test_inputs, test_labels = get_inputs_labels_from_dataset(test_dataset)
    
    train_dataset = TensorDataset(train_inputs, train_labels)
    valid_dataset = TensorDataset(valid_inputs, valid_labels)
    test_dataset = TensorDataset(test_inputs, test_labels)
    
    return train_dataset, valid_dataset, test_dataset

def load_cifar10_as_datasets(train_valid_split=1.0, data_path=None):
    """
    Load the CIFAR10 dataset and return train, validation, and test datasets.
    
    Args:
        train_valid_split: Ratio of training data to use for training (vs. validation)
        data_path: Optional override for where to store/find the data
        
    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset)
    """
    # Get the appropriate data directory
    data_dir = get_data_dir(data_path, 'cifar10')
    
    # Download and load the dataset
    train_dataset = CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=T.Compose([
            T.ToTensor(),
        ])
    )
    
    test_dataset = CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=T.Compose([
            T.ToTensor(),
        ])
    )
    
    # Split training data into train and validation
    if train_valid_split < 1.0:
        train_size = int(len(train_dataset) * train_valid_split)
        valid_size = len(train_dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, valid_size])
    else:
        valid_dataset = test_dataset
    
    # Convert to tensor datasets for consistent interface
    train_inputs, train_labels = get_inputs_labels_from_dataset(train_dataset)
    valid_inputs, valid_labels = get_inputs_labels_from_dataset(valid_dataset)
    test_inputs, test_labels = get_inputs_labels_from_dataset(test_dataset)
    
    train_dataset = TensorDataset(train_inputs, train_labels)
    valid_dataset = TensorDataset(valid_inputs, valid_labels)
    test_dataset = TensorDataset(test_inputs, test_labels)
    
    return train_dataset, valid_dataset, test_dataset

class InfoShuntingDataset(Dataset):
    """
    half = Gaussian, half = Uniform; label=0 or 1
    """
    def __init__(self,n_samples=20000, input_dim=10):
        super().__init__()
        self.n_samples=n_samples
        self.input_dim=input_dim
        self.data,self.labels=self._generate_data()

    def _generate_data(self):
        half=self.n_samples//2
        gauss=np.random.normal(0,1,size=(half,self.input_dim)).astype(np.float32)
        unif=np.random.uniform(-1,1,size=(self.n_samples-half,self.input_dim)).astype(np.float32)
        Xfull=np.concatenate([gauss,unif],axis=0)
        Yfull=np.concatenate([np.zeros(half),np.ones(self.n_samples-half)])
        idx=np.arange(self.n_samples)
        np.random.shuffle(idx)
        Xfull=Xfull[idx]
        Yfull=Yfull[idx]
        return torch.tensor(Xfull), torch.tensor(Yfull,dtype=torch.long)
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class NoisyLineDataset(Dataset):
    """
    line dataset but add random Gaussian noise
    """
    def __init__(self,n_samples=10000,image_size=10,noise_level=0.2):
        super().__init__()
        self.n_samples=n_samples
        self.image_size=image_size
        self.noise_level=noise_level
        self.data,self.labels=self._generate_data()
    def _generate_data(self):
        X,Y=[],[]
        for _ in range(self.n_samples):
            img=np.zeros((self.image_size,self.image_size),dtype=np.float32)
            label=np.random.choice([0,1,2])
            if label==1:
                row=np.random.randint(0,self.image_size)
                img[row,:]=1.0
            elif label==2:
                col=np.random.randint(0,self.image_size)
                img[:,col]=1.0
            noise=np.random.normal(0,self.noise_level,size=img.shape)
            img+=noise.astype(np.float32)
            X.append(img.flatten())
            Y.append(label)
        return torch.tensor(np.stack(X)), torch.tensor(Y,dtype=torch.long)
    def __len__(self):return self.n_samples
    def __getitem__(self,idx):
        return self.data[idx], self.labels[idx]


class SparseMNISTDataset(Dataset):
    """
    binarize MNIST
    """
    def __init__(self, base_mnist, threshold=0.3):
        super().__init__()
        self.base_mnist=base_mnist
        self.threshold=threshold
    def __len__(self):
        return len(self.base_mnist)
    def __getitem__(self,idx):
        img,label=self.base_mnist[idx]
        arr=np.array(img,dtype=np.float32)/255.0
        bin_arr=(arr>self.threshold).astype(np.float32)
        return torch.tensor(bin_arr.flatten(),dtype=torch.float32), label


class ContextualFigureGroundMNIST(Dataset):
    """
    left half random noise, right half original digit
    """
    def __init__(self, base_mnist, noise_level=0.1):
        super().__init__()
        self.base_mnist=base_mnist
        self.noise_level=noise_level
    def __len__(self):
        return len(self.base_mnist)
    def __getitem__(self,idx):
        img,label=self.base_mnist[idx]
        img_arr=np.array(img,dtype=np.float32)
        w=28
        new_img=np.random.rand(w,w)*self.noise_level
        boundary=w//2
        new_img[:,boundary:]=img_arr[:,boundary:]
        return torch.tensor(new_img.flatten(),dtype=torch.float32), label


class MultiTaskLineDataset(Dataset):
    """
    stub: half data is line dataset (0,1,2), second half inverts labels
    """
    def __init__(self,total_size=12000,image_size=10):
        super().__init__()
        self.total_size=total_size
        self.image_size=image_size
        self.data,self.labels=self._generate_data()
    def _generate_data(self):
        half=self.total_size//2
        Xlist,Ylist=[],[]
        for _ in range(half):
            img=np.zeros((self.image_size,self.image_size),dtype=np.float32)
            label=np.random.choice([0,1,2])
            if label==1:
                row=np.random.randint(0,self.image_size)
                img[row,:]=1.0
            elif label==2:
                col=np.random.randint(0,self.image_size)
                img[:,col]=1.0
            Xlist.append(img.flatten())
            Ylist.append(label)
        for _ in range(self.total_size-half):
            img=np.zeros((self.image_size,self.image_size),dtype=np.float32)
            label=np.random.choice([0,1,2])
            if label==1:
                row=np.random.randint(0,self.image_size)
                img[row,:]=1.0
            elif label==2:
                col=np.random.randint(0,self.image_size)
                img[:,col]=1.0
            new_label={0:0,1:2,2:1}[label]
            Xlist.append(img.flatten())
            Ylist.append(new_label+3)
        Xarr=np.stack(Xlist).astype(np.float32)
        Yarr=np.array(Ylist,dtype=np.int64)
        return torch.tensor(Xarr), torch.tensor(Yarr)
    def __len__(self):
        return self.total_size
    def __getitem__(self,idx):
        return self.data[idx], self.labels[idx]


class CompositionalMNIST(Dataset):
    """
    merges two MNIST digits horizontally => label = digit1*10+digit2
    """
    def __init__(self, base_mnist, n_samples=20000):
        super().__init__()
        self.base_mnist=base_mnist
        self.n_samples=n_samples
        self.data,self.labels=self._generate_data()
    def _generate_data(self):
        length=len(self.base_mnist)
        Xlist=[]
        Ylist=[]
        for _ in range(self.n_samples):
            idx1=np.random.randint(0,length)
            idx2=np.random.randint(0,length)
            img1,lbl1=self.base_mnist[idx1]
            img2,lbl2=self.base_mnist[idx2]
            arr1=np.array(img1,dtype=np.float32)
            arr2=np.array(img2,dtype=np.float32)
            comp=np.concatenate([arr1,arr2],axis=1)
            Xlist.append(comp.flatten())
            Ylist.append(lbl1*10+lbl2)
        Xarr=np.stack(Xlist).astype(np.float32)
        Yarr=np.array(Ylist,dtype=np.int64)
        return torch.tensor(Xarr), torch.tensor(Yarr)
    def __len__(self):return self.n_samples
    def __getitem__(self,idx):
        return self.data[idx],self.labels[idx]




def load_MNIST(train_valid_split = 1, cache_dir=None):
    if cache_dir is None:
        # Navigate up from the current file to the project root, then use outputs/data
        # instead of using the current working directory which creates src/data
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_path = os.path.join(project_root, 'outputs', 'data', 'mnist')
        os.makedirs(data_path, exist_ok=True)
    else:
        data_path = os.path.join(cache_dir, 'data', 'mnist')

    if os.path.exists(data_path):
        train_data = TensorDataset(
            torch.load(os.path.join(data_path, 'train_inputs.pt'), 
                       weights_only = True),
            torch.load(os.path.join(data_path, 'train_labels.pt'), 
                       weights_only = True),
        )
        test_data = TensorDataset(
            torch.load(os.path.join(data_path, 'test_inputs.pt'), 
                       weights_only = True),
            torch.load(os.path.join(data_path, 'test_labels.pt'), 
                       weights_only = True),
        )
    else:
        os.makedirs(data_path, exist_ok = True)

        transform = T.ToTensor()

        def loader_saver(train = True):
            dataset = MNIST(root = data_path, 
                            train = train, 
                            download = True)
            imgs, labels = zip(*(dataset[i] for i in range(len(dataset))))
            
            imgs = torch.stack([transform(img).flatten() for img in imgs], 
                               dim = 0)
            labels = torch.tensor(labels)
                        
            name = 'train' if train else 'test'
            torch.save(imgs, os.path.join(data_path, f'{name}_inputs.pt'))
            torch.save(labels, os.path.join(data_path, f'{name}_labels.pt'))
            
            return TensorDataset(imgs, labels)

        train_data = loader_saver(train = True)
        test_data = loader_saver(train = False)

        shutil.rmtree(os.path.join(data_path, 'MNIST'))

    data = {}

    if train_valid_split < 1:
        train_ix = torch.randperm(len(train_data))[
            :int(train_valid_split*len(train_data))
        ]
        train_mask = torch.zeros((len(train_data),)).bool()
        train_mask[train_ix] = True

        valid_data = TensorDataset(*train_data[~train_mask])
        train_data = TensorDataset(*train_data[train_mask])

        data['valid'] = valid_data

    data['train'] = train_data
    data['test'] = test_data

    return data

def load_MNIST_modulo10(shuffle_iterations = 2, 
                        train_valid_split = 1,
                        cache_dir = None):
    
    data = load_MNIST(train_valid_split, cache_dir)

    for key, ds in data.items():
        ix = torch.randperm(len(ds))
        ix0 = ix[:int(len(ds)/2)]
        ix1 = ix[int(len(ds)/2):]

        input0, label0 = ds[ix0]
        input1, label1 = ds[ix1]
        assert(label0.shape[0] == label1.shape[0])

        inputs = []
        labels = []
        for _ in range(shuffle_iterations):
            inputs.append(torch.cat([input0, input1], dim = -1))
            labels.append((label0 + label1) % 10)

            ix0 = torch.randperm(input0.shape[0])
            input0 = input0[ix0]
            label0 = label0[ix0]

            ix1 = torch.randperm(input1.shape[0])
            input1 = input1[ix1]
            label1 = label1[ix1]
        
        inputs = torch.cat(inputs, dim = 0)
        labels = torch.cat(labels, dim = 0)

        data[key] = TensorDataset(inputs, labels)

    return data

def load_MNIST_task_switch(train_valid_split = 1, cache_dir = None):
    """
    MNIST Switch: 
    Turns the first two input pixels into 'task bits'. 
    For one task bit pattern, labels remain normal. 
    For the other, labels are inverted (digit -> 9 - digit). 
    """
    data = load_MNIST(train_valid_split, cache_dir)

    for key, ds in data.items():
        input0, label0 = ds[:]
        input1 = deepcopy(input0)

        input0[:,0] = 1
        input0[:,1] = 0

        input1[:,0] = 0
        input1[:,1] = 1
        label1 = 9 - label0

        inputs = torch.cat([input0, input1], dim = 0)
        labels = torch.cat([label0, label1], dim = 0)
        
        data[key] = TensorDataset(inputs, labels)
    
    return data

def load_MNIST_task_switch0(train_valid_split = 1, cache_dir = None):
    """
    MNIST Split: 
    Zeros out the first 10 pixels in the input. Then for each digit i in 0..9, 
    we set pixel i to 1 and also shift the label by i (mod 10). 
    This effectively creates 10 'subtasks' inside one dataset.
    """
    data = load_MNIST(train_valid_split, cache_dir)

    for key, ds in data.items():
        raw_input, raw_label = ds[:]
        raw_input[:,:10] = 0

        inputs = []
        labels = []

        for i in range(10):
            task_input = deepcopy(raw_input)
            task_input[:,i] = 1

            task_label = (raw_label + i) % 10

            inputs.append(task_input)
            labels.append(task_label)
        
        inputs = torch.cat(inputs, dim = 0)
        labels = torch.cat(labels, dim = 0)

        data[key] = TensorDataset(inputs, labels)

    return data

def split_MNIST_inputs(inputs, labels):
    unique_label = torch.unique(labels)
    input_list = []
    for unique in unique_label:
        ix = torch.where(labels == unique, True, False)
        input_list.append(inputs[ix])
    return input_list



def load_cifar10_as_datasets(train_valid_split=1.0, data_path=None):
    """
    Loads CIFAR-10, converts it to shape [N, 3*32*32], 
    and optionally splits into train/valid if train_valid_split < 1. 
    """
    from torchvision.datasets import CIFAR10
    transform = T.Compose([
        T.ToTensor()
    ])
    
    if data_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        root_dir = os.path.join(project_root, 'outputs', 'data', 'cifar')
        os.makedirs(root_dir, exist_ok=True)
    else:
        root_dir = os.path.join(data_path, "cifar")
        
    train_set = CIFAR10(root=root_dir, train=True, download=True, transform=transform)
    test_set = CIFAR10(root=root_dir, train=False, download=True, transform=transform)
    
    # Convert each dataset to Tensors: shape [N, 3*32*32]
    def convert_to_tensor(dataset):
        X = []
        Y = []
        for img, lbl in dataset:
            # img is shape [3, 32, 32], flatten to 3*32*32
            X.append(img.view(-1))
            Y.append(lbl)
        X = torch.stack(X, dim=0)
        Y = torch.tensor(Y, dtype=torch.long)
        return X, Y

    Xtrain, Ytrain = convert_to_tensor(train_set)
    Xtest, Ytest = convert_to_tensor(test_set)

    data = {}
    # If we want to do a train/valid split:
    if train_valid_split < 1.0:
        n_train = int(len(Ytrain)*train_valid_split)
        perm = torch.randperm(len(Ytrain))
        idx_train = perm[:n_train]
        idx_valid = perm[n_train:]
        Xtrain_, Ytrain_ = Xtrain[idx_train], Ytrain[idx_train]
        Xvalid_, Yvalid_ = Xtrain[idx_valid], Ytrain[idx_valid]
        data['train'] = (Xtrain_, Ytrain_)
        data['valid'] = (Xvalid_, Yvalid_)
    else:
        data['train'] = (Xtrain, Ytrain)

    data['test'] = (Xtest, Ytest)

    train_ds = TensorDataset(*data['train'])
    if 'valid' in data:
        valid_ds = TensorDataset(*data['valid'])
    else:
        valid_ds = TensorDataset(*data['test'])
    test_ds = TensorDataset(*data['test'])
    return train_ds, valid_ds, test_ds

########################################
# The single get_unified_datasets(...) function
########################################
def get_unified_datasets(task_cfg, train_cfg):
    """
    Get datasets based on configuration, handling both standard and synthetic datasets.
    
    Args:
        task_cfg: Task configuration with dataset info
        train_cfg: Training configuration with split info
        
    Returns:
        Dictionary of datasets (train, valid, test)
    """
    # Extract necessary parameters
    dataset_name = task_cfg.dataset
    data_path = task_cfg.data_path if task_cfg.data_path else None
    
    # Use the central data path management
    data_dir = get_data_dir(data_path)
    logger.info(f"Using data directory: {data_dir}")
    
    # Load the appropriate dataset
    if dataset_name == "mnist":
        train_ds, valid_ds, test_ds = load_mnist_as_datasets(
            train_valid_split=train_cfg.train_valid_split, 
            data_path=data_dir
        )
    elif dataset_name == "cifar10":
        train_ds, valid_ds, test_ds = load_cifar10_as_datasets(
            train_valid_split=train_cfg.train_valid_split, 
            data_path=data_dir
        )
    # Synthetic datasets don't need the data path, they're generated on-the-fly
    elif dataset_name == "line":
        train_ds, valid_ds, test_ds = get_line_datasets(task_cfg.parameters)
    elif dataset_name == "center_surround":
        train_ds, valid_ds, test_ds = get_center_surround_datasets(task_cfg.parameters)
    elif dataset_name == "orientation_bars":
        train_ds, valid_ds, test_ds = get_orientation_bars_datasets(task_cfg.parameters)
    elif dataset_name == "multi_xor":
        train_ds, valid_ds, test_ds = get_multi_xor_datasets(task_cfg.parameters)
    elif dataset_name == "mnist_modulo10":
        train_ds, valid_ds, test_ds = load_MNIST_modulo10(
            train_valid_split=train_cfg.train_valid_split,
            cache_dir=data_dir
        )
    elif dataset_name == "mnist_switch2":
        train_ds, valid_ds, test_ds = load_MNIST_task_switch(
            train_valid_split=train_cfg.train_valid_split,
            cache_dir=data_dir
        )
    elif dataset_name == "info_shunting":
        train_ds, valid_ds, test_ds = get_info_shunting_datasets(task_cfg.parameters)
    elif dataset_name == "noise_resilience":
        train_ds, valid_ds, test_ds = get_noisy_line_datasets(task_cfg.parameters)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return {
        "train": train_ds,
        "valid": valid_ds,
        "test": test_ds
    }
    
    
    
    
    
    # 1.	Synthetic
	# •	"line"
	# •	"center_surround"
	# •	"orientation_bars"
	# •	"multi_xor"
	# •	"info_shunting"
	# •	"noise_resilience"
	# •	"learning_dynamics"
	# 2.	MNIST-based
	# •	"mnist"
	# •	"mnist_modulo10"
	# •	"mnist_switch"
	# •	"mnist_split"
	# •	"feature_selectivity"
	# •	"context_gating"
	# •	"hierarchical_processing"
	# 3.	CIFAR-based
	# •	"cifar10"
