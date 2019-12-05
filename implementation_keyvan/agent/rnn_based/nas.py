from implementation_keyvan.agent.agent import Agent
from implementation_keyvan.agent.rnn_based.controller import Controller
import sys
sys.path.append("../problem")
from implementation_keyvan.problem.searchSpace import SearchSpace
import torch.nn as nn
from torch.optim import Adam

class AgentNAS(nn.Module, Agent):
    """Retourne une architecture aléatoire à chaque fois"""
    def __init__(self, space:SearchSpace, emb_size, hid_size, update_freq=100, **kw_params):
        super().__init__()
        self.space = space
        self.kw_params = kw_params
        
        self.iteration = 0
        self.update_freq = update_freq
        
        self.controller = Controller(space, emb_size, hid_size) # type: Controller
        self.optim = Adam(self.controller.parameters())
        
        self.buffer = [] # liste de couples (action, reward)
        self.last_action = None
    
    
    def act(self, last_reward=None):
        if last_reward is not None:
            assert self.last_action is not None
            self.buffer.append((self.last_action, last_reward))
        
        self.iteration += 1
        if self.iteration % self.update_freq == 0 and self.iteration!=0:
            self._update()
        
        action = self.space.rnn_sample(self.controller, **self.kw_params)
        archi, log_p = action
        
        self.last_action = action
        return archi
    
    def _update(self):
        self.optim.zero_grad()
        
        s = None
        for ((_archi, log_proba), reward) in self.buffer:
            if s is None:
                s = log_proba * reward
            else:
                s += log_proba * reward
        
        s.backward(retain_graph=True)
        
        self.optim.step()
        
        self.buffer = []
        