from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def act(self, last_reward=None):
        pass
    