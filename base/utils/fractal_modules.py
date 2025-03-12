import sys, os

def getDir():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(getDir())

import numpy as np
import torch
from easydict import EasyDict as edict

def fractal_code(contractive_functions, probs, seed):
    '''
    Also called an IFS code.
    contractive_functions : list of Contractive functions (contains the weights and biases)
    probs : list / numpy array
    '''
    code = edict()
    code.contractive_functions = contractive_functions
    code.p = probs

    return code

def sample_functions(ifs_code, num_points, batch=1):
    '''
    probs : numpy array of shape (n,) or a list that is later converted to numpy
    num_points : Number of samples to generate
    '''
    probs = ifs_code.p

    if isinstance(probs, torch.Tensor):
        probs = probs.cpu()
    if not isinstance(probs, np.ndarray):
        probs = np.array(probs)
    N = probs.shape[0]
    a = np.arange(0, N, 1)
    indices = np.random.choice(a=a, size=(batch, num_points), replace=True, p=probs)
    return indices