import torch
import torchvision
from torchvision import transforms
import os
import shutil
import json



class Data(torch.utils.data.Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, i):
        return self.x[i], self.y[i]
    
    def __len__(self):
        return self.y.shape[0]


class Shaper(object):

    def __init__(self, shape):
        self.shape = shape
    
    def reshape(self, x):
        return x.view(*self.shape)


def save_dict(dict, save_path, fname):
    with open(os.path.join(save_path, fname), 'w') as f:
        json.dump(dict, f, indent = 4)



def load_MNIST(train_valid_split = 1):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root, 'data', 'mnist')

    if os.path.exists(data_path):
        train_data = Data(
            torch.load(os.path.join(data_path, 'train_inputs.pt'), weights_only = True),
            torch.load(os.path.join(data_path, 'train_labels.pt'), weights_only = True),
        )
        test_data = Data(
            torch.load(os.path.join(data_path, 'test_inputs.pt'), weights_only = True),
            torch.load(os.path.join(data_path, 'test_labels.pt'), weights_only = True),
        )

    else:
        os.mkdir(data_path)

        transform = transforms.ToTensor()

        def loader_saver(train = True):
            dataset = torchvision.datasets.MNIST(root = data_path, train = train, download = True)
            imgs, labels = zip(*(dataset[i] for i in range(len(dataset))))
            
            imgs = torch.stack([transform(img).flatten() for img in imgs], dim = 0)
            labels = torch.tensor(labels)
                        
            name = 'train' if train else 'test'
            torch.save(imgs, os.path.join(data_path, f'{name}_inputs.pt'))
            torch.save(labels, os.path.join(data_path, f'{name}_labels.pt'))
            
            return Data(imgs, labels)

        train_data = loader_saver(train = True)
        test_data = loader_saver(train = False)

        shutil.rmtree(os.path.join(data_path, 'MNIST'))

    data = {}

    if train_valid_split < 1:
        train_ix = torch.randperm(len(train_data))[:int(train_valid_split*len(train_data))]
        train_mask = torch.zeros((len(train_data),)).bool()
        train_mask[train_ix] = True

        valid_data = Data(*train_data[~train_mask])
        train_data = Data(*train_data[train_mask])

        data['valid'] = valid_data

    data['train'] = train_data
    data['test'] = test_data

    return data


def load_MNIST_modulo10(shuffle_iterations = 2, train_valid_split = 1):
    data = load_MNIST(train_valid_split)

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

        data[key] = Data(inputs, labels)

    return data


def split_MNIST_inputs(inputs, labels):
    unique_label = torch.unique(labels)

    input_list = []

    for unique in unique_label:
        ix = torch.where(labels == unique, True, False)
        input_list.append(inputs[ix])

    return input_list


def accuracy_score(classifier, inputs, labels):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    classifier = classifier.to(device)
    inputs = inputs.to(device)
    labels = labels.to(device)

    yhat = classifier.predict(inputs)

    accuracy = (yhat == labels).float().mean(dim = -1)
    return accuracy








