from problem.searchSpace import SearchSpace
from random import shuffle, random
import numpy as np


class SearchSpaceL2C(SearchSpace):
    def __init__(self, search_space: int):
        self.space = search_space
    
    def score(self, archi):
        a = archi
        n = self.space
        
        s = a[0] ** 2
        
        for k in range(0, n - 1):
            s += (a[k + 1] - a[k]) ** 2
        
        s += (a[n - 1] - (n + 1)) ** 2
        
        return (n + 1) / s
    
    def mutate(self, archi, r=.5, **kwargs):
        values = archi.copy()
        for i in range(self.space - 1):
            if random() < r:
                values[i], values[i+1] = values[i+1], values[i]
    
        return values
    
    def random_sample(self, **kwargs):
        """Retourne une architecture aléatoire de l'espace"""
        l = np.arange(1, self.space + 1)
        shuffle(l)
        return l
