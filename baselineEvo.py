from collections import deque
from architecture import Architecture
from random import sample


class AgentEvo:
    def __init__(self, space, min_p, max_p, s, mutation_proba=.5):
        """
        :param min_p: les min_p premiers indivisus sont tirés aléatoirement
        :param max_p: au plus max_p individus sont gardés
        :param s: on fait muter le meilleur parmi s individus
        """
        self.space = space
        self.s = s
        self.max_p = max_p
        self.min_p = min_p
        self.mutation_proba = mutation_proba
        
        self.population = deque()
        self.last_action = None
        
    
    def act(self, last_reward=None):
        # prise en compte du feedback
        if self.last_action is not None:
            self.population.append((self.last_action, last_reward))
        
        
        if len(self.population) < self.min_p:
            # les min_p premiers individus sont tirés aléatoirement
            selection = Architecture.random_sample(self.space)
            
        else:
            # choisit S architectures aléatoirement
            selection = sample(self.population, self.s)
            
            # en extrait la meilleure
            selection = max(selection, key=lambda x:x[1])[0]
            
            # lui fait subir une mutation
            selection = selection.mutate(r=self.mutation_proba)
        
        
        if len(self.population) > self.max_p:
            # enlève la plus ancienne des architectures
            self.population.popleft()
        
        
        self.last_action = selection
        
        return selection
