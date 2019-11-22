import numpy as np
from collections import deque


class AgentEvo:
    def __init__(self, space, P, S):
        self.space = space
        self.P = P
        self.S = S
        self.population = deque()
        self.last_action = None
        
    
    def act(self, last_reward=None):
        # prise en compte du feedback
        if self.last_action is not None:
            self.population.append((self.last_action, last_reward))
            
        # enlève la plus ancienne des architectures
        if len(self.population) == self.P + 1:
            self.population.popleft()
        
        # choisit S architectures aléatoirement
        selection = np.random.choice(self.population, self.S)
        
        # en extrait la meilleure
        selection = max(selection, key=lambda x:x[1])
        
        # lui fait subir une mutation
        selection = selection.mutate()
        
        return selection
