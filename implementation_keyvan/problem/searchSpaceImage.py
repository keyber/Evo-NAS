from collections import namedtuple
from implementation_keyvan.problem.searchSpace import SearchSpace
import torch.nn as nn
from random import random, choice
import sys
from implementation_keyvan.problem.architectureNN import ArchitectureNN


class SearchSpaceImage(SearchSpace):
    """Définit les choix d'architecture possibles et la manière de créer le réseau"""
    _warn = True
    
    def __init__(self, dl_train, dl_test, out_channels, imagewidth, reduced_space=False, in_channels=3, train_params=None):
        self.train_data = dl_train
        self.test_data = dl_test
        self.imagewidth = imagewidth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.space = _SEARCH_SPACE_IMAGE_REDUCED if reduced_space else _SEARCH_SPACE_IMAGE
        self.train_params = train_params if train_params is not None else {}

    def _create_model(self, gbl_values, loc_values):
        """
        crée toutes les convolutions décrites,
        insert des ReLU entre
        finit par un Fully Connected
        """
        assert gbl_values["layer_number"] == len(loc_values)
        conv = []
        in_channels = self.in_channels
        
        # maintient le calcul de cette valeur pour le fully connected
        image_width = self.imagewidth
    
        for layer in loc_values:
            n = layer["filter_number"]
            w = layer["filter_width"]
            s = layer["filter_stride"]
            p = layer["filter_padding"]
        
            if w > image_width:
                if SearchSpaceImage._warn:
                    SearchSpaceImage._warn = False
                    print("EarlyStoppingWarning: le kernel est plus grand que l'image", file=sys.stderr)
            else:
                conv.append(nn.Conv2d(in_channels=in_channels, out_channels=n,
                                      kernel_size=w, stride=s, padding=p))
            
                conv.append(nn.ReLU())
            
                in_channels = n
                image_width = (image_width + 2 * p - w) // s + 1  # cf tp 6 de RDFIA
    
        conv = nn.Sequential(*conv)
    
        fc = nn.Linear(in_channels * image_width ** 2, self.out_channels)
    
        return Net(self, conv, fc)

    def score(self, archi:ArchitectureNN):
        """fit le modele puis retourne ses performances en test"""
        archi.fit_model(self.train_data, **self.train_params)
    
        return archi.compute_test_score(self.test_data)

    def random_sample(self, **kwargs):
        """Retourne une architecture aléatoire de l'espace"""
        gbl_values = {k: choice(v) for (k, v) in self.space.gbl.items()}
    
        n = gbl_values["layer_number"]
        loc_values = [{k: choice(v) for (k, v) in self.space.loc.items()} for _ in range(n)]
        
        net = self._create_model(gbl_values, loc_values)
        return ArchitectureNN(gbl_values, loc_values, net, **kwargs)

    def mutate(self, archi: ArchitectureNN, r=.5, **kwargs):
        """Retourne une nouvelle architecture en re-samplant les caractéristiques avec une probabilité r"""
        gbl_values = {k: choice(v) if random() < r else archi.gbl_values[k] for (k, v) in self.space.gbl.items()}
    
        n = gbl_values["layer_number"]
        loc_values = []
        for i in range(n):
            if i < archi.gbl_values["layer_number"]:
                loc_values.append({k: choice(v) if random() < r else archi.loc_values[i][k]
                                   for (k, v) in self.space.loc.items()})
            else:
                loc_values.append({k: choice(v) for (k, v) in self.space.loc.items()})
    
        net = self._create_model(gbl_values, loc_values)
        return ArchitectureNN(gbl_values, loc_values, net, **kwargs)


class Net(nn.Module):
    def __init__(self, search_space: SearchSpaceImage, conv, fc):
        super().__init__()
        self.search_space = search_space
        self.conv = conv
        self.fc = fc
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv(x)
        x = x.reshape((batch_size, -1))
        x = self.fc(x)
        return x


_space_type = namedtuple("Space", ("gbl", "loc"))

_SEARCH_SPACE_IMAGE = _space_type(
    gbl = {
        "layer_number": (5, 10, 15),
    },
    loc = {
        "filter_width" : (1, 3, 5, 7),
        "filter_number" : (24, 36, 48, 64),
        "filter_stride" : (1, 2, 3),
        "filter_padding" : (0,),
    })
_SEARCH_SPACE_IMAGE_REDUCED = _space_type(
    gbl = {
        "layer_number": (1, 2, 3, 4),
    },
    loc = {
        "filter_width" : (1, 3, 5),
        "filter_number" : (24, 36, 48, 64),
        "filter_stride" : (1, 2, 3),
        "filter_padding" : (0,),
    })
