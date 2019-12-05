from implementation_keyvan.agent.agent import Agent
import sys
sys.path.append("../problem")
from implementation_keyvan.problem.searchSpace import SearchSpace


class AgentRS(Agent):
    """Retourne une architecture aléatoire à chaque fois"""
    def __init__(self, space:SearchSpace, **kw_params):
        self.space = space
        self.kw_params = kw_params

    def act(self, last_reward=None):
        return self.space.random_sample(**self.kw_params)
