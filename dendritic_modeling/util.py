import torch
import torchvision
import os
import json



class Data(torch.utils.data.Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, i):
        return self.x[i], self.y[i]
    
    def __len__(self):
        return self.y.shape[0]


def save_dict(dict, save_path, fname):
    with open(os.path.join(save_path, fname), 'w') as f:
        json.dump(dict, f)


def load_MNIST(train_valid_split = 1):
    data_root = '../data/mnist'

    train_data = torchvision.datasets.MNIST(root = data_root, train = True, download = True)
    test_data = torchvision.datasets.MNIST(root = data_root, train = False, download = True)

    data = {}

    if train_valid_split < 1:
        train_ix = torch.randperm(len(train_data))[:int(train_valid_split*len(train_data))]

        valid_data = Data(train_data[~train_ix])
        train_data = Data(train_data[train_ix])

        data['valid'] = valid_data

    data['train'] = train_data
    data['test'] = test_data

    return data



def accuracy_score(classifier, inputs, labels):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    classifier = classifier.to(device)
    inputs = inputs.to(device)
    labels = labels.to(device)

    yhat = classifier.predict(inputs)

    accuracy = (yhat == labels).mean(dim = -1)
    return accuracy








