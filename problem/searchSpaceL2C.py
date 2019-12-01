from problem.searchSpace import SearchSpace
import numpy as np


class SearchSpaceL2C(SearchSpace):
    def __init__(self, search_space: int):
        self.space = search_space
    
    def score(self, archi):
        """cf article: retourne
        1 pour des nombres dans l'ordre
        un nombre entre 0 et 1 sinon"""
        a = archi
        n = self.space
        
        s = a[0] ** 2
        
        for k in range(0, n - 1):
            s += (a[k + 1] - a[k]) ** 2
        
        s += (a[n - 1] - (n + 1)) ** 2
        
        return (n + 1) / s

    def mutate(self, archi, r=.5, **kwargs):
        values = archi.copy()
        ind = np.random.random(self.space) < r
        values[ind] = np.random.randint(1, self.space+1, len(values[ind]))
    
        return values
    
    def random_sample(self, **kwargs):
        """Retourne une architecture aléatoire de l'espace"""
        return np.random.randint(1, self.space + 1, size=self.space)
