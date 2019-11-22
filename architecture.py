import torch.nn as nn
from torch.optim import Adam
from random import random, sample
import numpy as np

SEARCH_SPACE = {
    # "input" : (0, 1, 2),
    "fine_tune_input" : (True, False),
    "use_conv" : (True, False),
    "conv_act" : (nn.ReLU(), nn.LeakyReLU(), nn.Sigmoid(), nn.Tanh()),
    "conv_batch_norm": (True, False),
    "conv_max_ngram_length": (2, 3),
    "conv_dropout": (.0, .1, .2, .3, .4),
    "conv_filter_nb": (32, 64, 128),
    "hid_layer_nb": (0, 1, 2, 3, 5),
    "hid_layer_size" : (64, 128, 256),
    "hid_layer_act" : (nn.ReLU(), nn.LeakyReLU(), nn.Sigmoid(), nn.Tanh()),
    "hid_layer_norma" : (None, "batch", "layer"),
    "hid_layer_dropout" : (.0, .1, .2, .3, .4),
    # "optim" : (),
    "batch_size" : (128, 256),
    "deep_tower_lr" : np.logspace(1e-3, 1e-1, 5),
    "deep_tower_regularization_weight" : (0., 1e-4, 1e-3, 1e-2),
    "wide_tower_lr" : np.logspace(1e-3, 1e-1, 5),
    "wide_tower_regularization_weight" : (0., 1e-4, 1e-3, 1e-2),
    "nb_training_samples" : (1e5, 1e6),
}


class Architecture:
    def __init__(self, search_space:dict, archi:dict):
        self.search_space = search_space
        self.hyperparam_value = archi
        self.net = self._create_model(archi)
        self.loss = nn.BCEWithLogitsLoss()
        self.optim = Adam(self.net.parameters())
    
    @staticmethod
    def _create_model(archi):
        # text embedding module
        
        # fc stack
        
        
        return nn.Sequential(*[])
    
    #wide shallow layer OneHot->softmax, L1 loss
    
    def mutate(self, r=.5):
        archi = {}
        for k in self.search_space:
            archi[k] = sample(self.search_space[k]) if random() < r else self.hyperparam_value[k]
        
        return Architecture(self.search_space, archi)

    @staticmethod
    def random_sample(search_space):
        archi = {k: sample(search_space[k]) for k in search_space}
        return Architecture(search_space, archi)

