import torch
import torch.utils.data
from problem import instances
from agent.baselineRS import AgentRS
from agent.baselineEvo import AgentEvo
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

def learn_to_count_score(sol):
    return 0
    

def run_agent(agent, n_iter=10, verbose=0):
    """appelle n_iter fois agent.act et retourne la meilleure architecture trouvée"""
    best_a, best_v = None, -np.inf
    
    archi = agent.act(last_reward=None)
    
    for i in range(n_iter):
        score = learn_to_count_score(archi)
        
        if score > best_v:
            best_v = score
            best_a = archi
            if verbose == 1:
                print(score, archi)
        
        if verbose == 2:
            print(score, archi)
        
        archi = agent.act(last_reward=score)
    
    return best_a, best_v


def main():
    space, ds_train, ds_test = instances.cifar10(reduced=True)
    
    agents = [
        AgentRS(space, loss=(nn.CrossEntropyLoss, {}), optim=(Adam, {"lr": 1e-5}),
                lr_scheduler=(ExponentialLR, {"gamma": .9})),
        AgentEvo(space, min_p=2, max_p=3, s=2),
    ]
    
    for agent in agents:
        print("Agent", agent.__class__)
        best_archi, best_val = run_agent(ds_train, ds_test, agent, n_iter=4, train_max_iter=2, verbose=2)
        print("best archi", best_archi.gbl_values, best_archi.loc_values)
        print("best val", best_val)
        print("\n\n\n")


main()
