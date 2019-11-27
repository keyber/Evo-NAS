from architecture import Architecture

class AgentRS:
    def __init__(self, space):
        """Retourne une architecture aléatoire à chaque fois"""
        self.space = space
    
    def act(self, last_reward=None):
        return Architecture.random_sample(self.space)
