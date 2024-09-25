
import os
import json

import torch

from dendritic_modeling.models import EINetClassifier
from dendritic_modeling.utils import load_MNIST, Shaper, split_MNIST_inputs
from dendritic_modeling.plot_utils import (plot_einet_params, 
                                           plot_einet_activations, 
                                           einet_activations_to_csv)


print('imports completed.....', flush = True)


if __name__ == '__main__':

    trained_name = 'test5'

    make_csv = False
    
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trained_path = os.path.join(root, 'results', trained_name)

    with open(os.path.join(trained_path, 'model_hparams.json'), 'r') as f:
        hparams = json.load(f)

    clf = EINetClassifier(**hparams)
    clf.load_state_dict(torch.load(os.path.join(trained_path, 
                                                'state_dict.pt'), 
                                                weights_only = True))

    shaper = Shaper(shape = (28,28))

    print('creating param visuals.....', flush = True)
    plot_einet_params(
        einet = clf.net, save_root = os.path.join(trained_path, 
                                                  'param_visuals'),
        reshape_fn = shaper.reshape, reshape_exc_syn = True,
    )
    print('  done..', flush = True)
    
    inputs, labels = load_MNIST(train_valid_split = 1)['train'][:]
    input_list = split_MNIST_inputs(inputs, labels)

    print('creating activation visuals.....')
    plot_einet_activations(
        einet = clf.net, input_list = input_list,
        save_root = os.path.join(trained_path, 'activation_visuals'),
    )
    print('  done..')

    if make_csv:
        print('making activation component csv files.....', flush = True)
        einet_activations_to_csv(
            einet = clf.net, input_list = input_list,
            save_root = os.path.join(trained_path, 'activation_components'),
        )
        print('  done..')




