
import numpy as np


def num_params_count(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    return parameters


def num_params(model):
    print('Trainable Parameters: %.3f million' % num_params_count(model))