from implementation_keyvan.agent.agent import Agent
from collections import deque
from random import sample
import sys
sys.path.append("../problem")
from implementation_keyvan.problem.searchSpace import SearchSpace


class AgentEvo(Agent):
    def __init__(self, space:SearchSpace, min_p, max_p, s, mutation_proba=.5, **kw_params):
        """
        :param min_p: les min_p premiers indivisus sont tirés aléatoirement
        :param max_p: au plus max_p individus sont gardés
        :param s: on fait muter le meilleur parmi s individus
        :param kw_params: paramètres pour l'architecture (loss, optim, ...)
        """
        assert min_p >= s
        self.space = space
        self.s = s
        self.max_p = max_p
        self.min_p = min_p
        self.mutation_proba = mutation_proba
        self.kw_params = kw_params
        
        self.population = deque()
        self.last_action = None
        
    
    def act(self, last_reward=None):
        # prise en compte du feedback
        if self.last_action is not None:
            self.population.append((self.last_action, last_reward))
        
        
        if len(self.population) < self.min_p:
            # les min_p premiers individus sont tirés aléatoirement
            selection = self.space.random_sample(**self.kw_params)
            
        else:
            # choisit S architectures aléatoirement
            selection_list = sample(self.population, self.s)
            
            # en extrait la meilleure
            selection = max(selection_list, key=lambda x:x[1])[0]
            
            # lui fait subir une mutation
            selection = self.space.mutate(selection, r=self.mutation_proba, **self.kw_params)
        
        
        if len(self.population) > self.max_p:
            # enlève la plus ancienne des architectures
            self.population.popleft()
        
        
        self.last_action = selection
        
        return selection
