import torch
import torchvision



class Data(torch.utils.data.Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, i):
        return self.x[i], self.y[i]
    
    def __len__(self):
        return self.y.shape[0]



def load_MNIST(train_valid_split = 1):
    data_root = 'mnist'

    train_data = torchvision.datasets.MNIST(root = data_root, train = True, download = True)

    test_data = torchvision.datasets.MNIST(root = data_root, train = False, download = True)

    if train_valid_split < 1:
        train_ix = torch.randperm(len(train_data))[:int(train_valid_split*len(train_data))]

        valid_data = Data(train_data[~train_ix])
        train_data = Data(train_data[train_ix])

        return train_data, valid_data, test_data
    
    return train_data, test_data














