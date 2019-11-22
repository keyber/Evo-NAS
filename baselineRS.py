import random


class AgentRS:
    def __init__(self, space):
        self.space = space
    
    def act(self, last_reward=None):
        return random.choice(self.space)
