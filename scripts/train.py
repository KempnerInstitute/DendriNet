import torch
from dendritic_modeling.util import load_MNIST, accuracy_score, save_dict
from dendritic_modeling.schemes import TrainerMLE
from dendritic_modeling.models import TopKLogisticRegression

import os
from datetime import datetime
import time
print('imports complete\n\n', flush = True)


if __name__ == '__main__':

    name = 'TopKLogReg'

    hparams = {}
    hparams['train-valid split'] = 0.8
    
    hparams['K'] = 300
    hparams['param freezing'] = False

    hparams['lr'] = 1e-3
    hparams['epochs'] = 100
    hparams['batch size'] = 512

    print('loading mnist.....\n\n', flush = True)
    data = load_MNIST(train_valid_split = hparams['train-valid split'])
    
    logreg = TopKLogisticRegression(
        input_dim = 784, n_classes = 10, K = hparams['K'], param_freezing = hparams['param freezing'],
        )

    trainer = TrainerMLE(
        optimizer = torch.optim.Adam(logreg.parameters(), lr = hparams['lr']),
        suppress_prints = False, print_every = 10,
    )

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(root, 'trained', f'{name}_{now}')

    print('training.....\n\n')
    stime = time.time()
    trainer.train(
        model = logreg, 
        train_data = data['train'], valid_data = data['valid'],
        epochs = hparams['epochs'], batch_size = hparams['batch size'], shuffle = True,
        plot_losses = True, save_path = save_path,
        )
    etime = time.time() - stime

    timing = {}
    timing['training'] = '%.3f sec -> %.3f min' % (etime, etime/60)
    
    train_input, train_label = data['train'][:]
    valid_input, valid_label = data['valid'][:]
    test_input, test_label = data['test'][:]

    print('evaluating performance.....\n\n', flush = True)
    performance = {}
    stime = time.time()
    performance['train accuracy'] = accuracy_score(
        classifier = logreg, inputs = train_input, labels = train_label,
    )
    performance['valid accuracy'] = accuracy_score(
        classifier = logreg, inputs = valid_input, labels = valid_label,
    )
    performance['test accuracy'] = accuracy_score(
        classifier = logreg, inputs = test_input, labels = test_label,
    )
    etime = time.time() - stime

    timing['inference'] = '%.3f sec -> %.3f min' % (etime, etime/60)


    info = {
        'hparams' : hparams, 'performance' : performance, 'timing' : timing
    }

    save_dict(dict = info, save_path = save_path, fname = 'info.json')










