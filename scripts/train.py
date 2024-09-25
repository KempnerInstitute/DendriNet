import os
import time
from datetime import datetime

import torch

from dendritic_modeling.utils import load_MNIST, accuracy_score, save_dict
from dendritic_modeling.training import TrainerMLE
from dendritic_modeling.models import EINetClassifier

print('\nimports complete\n', flush = True)


if __name__ == '__main__':

    name = 'train'
    
    model_hp = {
        'input_dim' : 784,# * 2,
        'excitatory_layer_sizes' : [10,],
        'inhibitory_layer_sizes' : [20,],
        'excitatory_branch_factors' : [12,],
        'inhibitory_branch_factors' : [],
        'reactivate' : True,
        'somatic_synapses' : True,
    }

    model_hp['ee_synapses_per_branch_per_layer'] = [24,]
    model_hp['ei_synapses_per_branch_per_layer'] = [100,]
    model_hp['ie_synapses_per_branch_per_layer'] = [1,]
    model_hp['ii_synapses_per_branch_per_layer'] = []

    train_hp = {
        'train_valid_split' : 0.8,
        'lr' : 1e-3,
        'epochs' : 300,
        'batch_size' : 512,
    }

    print('loading mnist.....\n', flush = True)
    # data = load_MNIST_modulo10(shuffle_iterations = 5, train_valid_split  = 0.8)
    data = load_MNIST(train_valid_split = train_hp['train_valid_split'])

    clf = EINetClassifier(**model_hp)
    

    trainer = TrainerMLE(
        optimizer = torch.optim.Adam(clf.parameters(), lr = train_hp['lr']),
        suppress_prints = False, print_every = 10,
    )

    now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(root, 'results', f'{name}_{now}')
    os.makedirs(save_path, exist_ok = True)

    print('training.....\n')
    stime = time.time()
    trainer.train(
        model = clf, 
        train_data = data['train'], valid_data = data['valid'],
        epochs = train_hp['epochs'], batch_size = train_hp['batch_size'], shuffle = True,
        plot_losses = True, save_path = save_path,
        )
    etime = time.time() - stime

    train_time = '%.6f sec -> %.6f min' % (etime, etime/60)
    

    print('evaluating performance.....\n', flush = True)
    stime = time.time()
    performance = {
        'train accuracy' : accuracy_score(clf, *data['train'][:]).item(),
        'valid accuracy' : accuracy_score(clf, *data['valid'][:]).item(),
        'test accuracy' : accuracy_score(clf, *data['test'][:]).item(),
    }
    etime = time.time() - stime

    performance['training time'] = train_time
    performance['inference time'] = '%.6f sec -> %.6f min' % (etime, etime/60)


    save_dict(dict = model_hp, save_path = save_path, fname = 'model_hparams.json')
    save_dict(dict = train_hp, save_path = save_path, fname = 'training_hparams.json')
    save_dict(dict = performance, save_path = save_path, fname = 'performance.json')

    print('  done..')



