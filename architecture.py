import torch.nn as nn
from torch.optim import Adam
from random import random, choice


class Architecture:
    """Contient un réseau de neurones, une loss et un optimiser"""
    
    def __init__(self, search_space, architecture, lr_scheduler=None):
        self.search_space = search_space
        self.architecture = architecture
        
        self.net = search_space.create_model(architecture)
        self.loss = nn.CrossEntropyLoss()
        self.optim = Adam(self.net.parameters())
        self.lr_scheduler = lr_scheduler
    
    def mutate(self, r=.5):
        """Retourne une nouvelle architecture en rechoisissant les caractéristiques avec une proba r"""
        space = self.search_space.space
        
        global_values = {k: choice(space["global"][k]) for k in space["global"]}
        
        n = global_values["layer_number"]
        local_values = []
        for i in range(n):
            if i < self.architecture["global"]["layer_number"]:
                local_values.append({k: choice(space["local"][k]) if random() < r 
                                     else self.architecture["local"][i][k]
                                     for k in space["local"]})
            else:
                local_values.append({k: choice(space["local"][k]) for k in space["local"]})
        
        archi = {"global": global_values, "local": local_values}
        return Architecture(self.search_space, archi)
    
    @staticmethod
    def random_sample(search_space):
        """Retourne une architecture aléatoire de l'espace"""
        space = search_space.space
        
        global_values = {k: choice(space["global"][k]) for k in space["global"]}
        
        n = global_values["layer_number"]
        local_values = [{k: choice(space["local"][k]) for k in space["local"]} for _ in range(n)]
        
        archi = {"global": global_values, "local": local_values}
        return Architecture(search_space, archi)
