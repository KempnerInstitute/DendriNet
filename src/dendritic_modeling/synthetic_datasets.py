# synthetic_datasets.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, TensorDataset

from torchvision.datasets import MNIST
import torchvision.transforms as T

from dendritic_modeling.utils import load_MNIST

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
            bits=np.random.randint(0,2,size=self.n_bits).astype(np.float32)
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

def load_mnist_as_datasets(train_valid_split=0.8):
    data = load_MNIST(train_valid_split=train_valid_split)
    train_ds=TensorDataset(*data['train'][:])
    if 'valid' in data:
        valid_ds=TensorDataset(*data['valid'][:])
    else:
        valid_ds=TensorDataset(*data['test'][:])  # fallback if no separate valid
    test_ds=TensorDataset(*data['test'][:])
    return train_ds, valid_ds, test_ds

########################################
# New tasks: info_shunting, noise_resilience, feature_selectivity,
#            context_gating, learning_dynamics, hierarchical_processing
########################################

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


########################################
# The single get_unified_datasets(...) function
########################################

def get_unified_datasets(task_cfg, train_cfg):
    """
    Returns (train_dataset, valid_dataset, test_dataset)
    for any of the recognized dataset_name from 'task_cfg["dataset"]'.

    The function uses train_cfg["train_valid_split"] if needed for MNIST
    or other tasks. 
    Each dataset is of type torch.utils.data.Dataset.
    """

    dataset_name = task_cfg['dataset']  # e.g. "line", "mnist", etc.
    p = task_cfg['parameters']

    train_size = p['train_size',8000]
    valid_size = p['valid_size',1000]
    test_size  = p['test_size',1000]
    image_size = p['image_size',784]
    n_orient_classes = p.get('n_orient_classes',8)
    n_bits = p.get('n_bits',8)

    train_valid_split = train_cfg.get("train_valid_split",1.0)

    total = train_size+valid_size+test_size

    if dataset_name=="line":
        full_ds=LineDataset(n_samples=total, image_size=image_size)
        return random_split(full_ds,[train_size, valid_size, test_size])

    elif dataset_name=="center_surround":
        full_ds=CenterSurroundDataset(n_samples=total, image_size=image_size)
        return random_split(full_ds,[train_size, valid_size, test_size])

    elif dataset_name=="orientation_bars":
        full_ds=OrientationBarsDataset(n_samples=total,
                                       image_size=image_size,
                                       n_orient_classes=n_orient_classes)
        return random_split(full_ds,[train_size, valid_size, test_size])

    elif dataset_name=="multi_xor":
        full_ds=MultiXORDataset(n_samples=total,n_bits=n_bits)
        return random_split(full_ds,[train_size, valid_size, test_size])

    elif dataset_name=="mnist":
        # returns train_ds, valid_ds, test_ds from your load_mnist_as_datasets
        # ignoring train_size, valid_size, test_size for now, unless you want partial
        train_ds, valid_ds, test_ds=load_mnist_as_datasets(train_valid_split=train_valid_split)

        # if you want partial:
        # e.g. if train_size < len(train_ds), then random_split further
        # for now, let's keep it as is:
        return train_ds, valid_ds, test_ds

    elif dataset_name=="info_shunting":
        full_ds=InfoShuntingDataset(n_samples=total,input_dim=10)
        return random_split(full_ds,[train_size, valid_size, test_size])

    elif dataset_name=="noise_resilience":
        full_ds=NoisyLineDataset(n_samples=total,
                                 image_size=image_size,
                                 noise_level=0.2)
        return random_split(full_ds,[train_size, valid_size, test_size])

    elif dataset_name=="feature_selectivity":
        # We threshold MNIST => Sparse
        base_mnist=MNIST(root='.',train=True,download=True)
        full_ds=SparseMNISTDataset(base_mnist,threshold=0.3)
        return random_split(full_ds,[train_size, valid_size, test_size])

    elif dataset_name=="context_gating":
        base_mnist=MNIST(root='.',train=True,download=True)
        full_ds=ContextualFigureGroundMNIST(base_mnist,noise_level=0.1)
        return random_split(full_ds,[train_size, valid_size, test_size])

    elif dataset_name=="learning_dynamics":
        full_ds=MultiTaskLineDataset(total_size=total,image_size=image_size)
        return random_split(full_ds,[train_size, valid_size, test_size])

    elif dataset_name=="hierarchical_processing":
        base_mnist=MNIST(root='.',train=True,download=True)
        full_ds=CompositionalMNIST(base_mnist,n_samples=total)
        return random_split(full_ds,[train_size, valid_size, test_size])

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")