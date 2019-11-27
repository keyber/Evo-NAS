import torch.nn as nn
import numpy as np


_SEARCH_SPACE_TEXT = {
    # "input" : (0, 1, 2),
    "fine_tune_input": (True, False),
    "use_conv": (True, False),
    "conv_act": (nn.ReLU(), nn.LeakyReLU(), nn.Sigmoid(), nn.Tanh()),
    "conv_batch_norm": (True, False),
    "conv_max_ngram_length": (2, 3),
    "conv_dropout": (.0, .1, .2, .3, .4),
    "conv_filter_nb": (32, 64, 128),
    "hid_layer_nb": (0, 1, 2, 3, 5),
    "hid_layer_size": (64, 128, 256),
    "hid_layer_act": (nn.ReLU(), nn.LeakyReLU(), nn.Sigmoid(), nn.Tanh()),
    "hid_layer_norma": (None, "batch", "layer"),
    "hid_layer_dropout": (.0, .1, .2, .3, .4),
    # "optim" : (),
    "batch_size": (128, 256),
    "deep_tower_lr": np.logspace(1e-3, 1e-1, 5),
    "deep_tower_regularization_weight": (0., 1e-4, 1e-3, 1e-2),
    "wide_tower_lr": np.logspace(1e-3, 1e-1, 5),
    "wide_tower_regularization_weight": (0., 1e-4, 1e-3, 1e-2),
    "nb_training_samples": (1e5, 1e6),
}


class SearchSpaceText:
    def __init__(self):
        self.space = _SEARCH_SPACE_TEXT
    
    @staticmethod
    def create_model(choices):
        # todo
        # text embedding module
        l = []
        #l.append(embedding)
        
        if choices["use_conv"]:
            l.append()
        
        # fc stack
        
        return nn.Sequential(*[])
        
        #wide shallow layer OneHot->softmax, L1 loss

