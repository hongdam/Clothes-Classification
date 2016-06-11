import cPickle as pickle
import os

import lasagne as nn

__all__ = [
    'read_model_data',
    'write_model_data',
]

PARAM_EXTENSION = 'pkl'


def read_model_data(model, filename):
    filename = os.path.join('./pkl/', '%s.%s' % (filename, PARAM_EXTENSION))
    with open(filename, 'r') as f:
        data = pickle.load(f)
    nn.layers.set_all_param_values(model, data)


def write_model_data(model, filename):
    data = nn.layers.get_all_param_values(model)
    filename = os.path.join('./pkl/', filename)
    filename = '%s.%s' % (filename, PARAM_EXTENSION)
    with open(filename, 'w') as f:
        pickle.dump(data, f)