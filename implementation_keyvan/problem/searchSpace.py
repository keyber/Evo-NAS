from abc import ABC, abstractmethod


class SearchSpace(ABC):
    @abstractmethod
    def random_sample(self, **kwargs):
        pass
    
    @abstractmethod
    def mutate(self, archi, r=.5, **kwargs):
        pass
    # 
    # @abstractmethod
    # def rnn_mutate(self, archi, **kwargs):
    #     pass

    @abstractmethod
    def score(self, archi):
        pass
