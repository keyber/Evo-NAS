from implementation_keyvan.problem import instances
from implementation_keyvan.agent.agent import Agent
from implementation_keyvan.agent.baselineRS import AgentRS
from implementation_keyvan.agent.baselineEvo import AgentEvo
from implementation_keyvan.agent.rnn_based.nas import AgentNAS
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from implementation_keyvan.problem.searchSpace import SearchSpace


def run_agent(space:SearchSpace, agent:Agent, n_iter=10, verbose=0):
    """appelle n_iter fois agent.act et retourne la meilleure architecture trouvée"""
    best_a, best_v = None, -np.inf
    score_sum = 0
    
    archi = agent.act(last_reward=None)
    
    s = len(str(n_iter-1))
    
    if verbose:
        print("iter".ljust(s+2), " val    moy     archi")
    
    
    for i in range(1, n_iter+1):
        score = space.score(archi)
        score_sum += score
        
        if score > best_v:
            best_v = score
            best_a = archi
            if verbose==1:
                print(str(i).rjust(s, "0"), "  %.2f   %.2f   "%(score, score_sum/i), archi)
                
        if verbose==2:
            print(str(i).rjust(s, "0"), "  %.2f   %.2f   "%(score, score_sum/i), archi)
        
        archi = agent.act(last_reward=score)
        
    
    return best_a, best_v


def main_learn_to_count():
    space = instances.learn_to_count(space_size=8)
    
    agents = [
        AgentNAS(space, emb_size=50, hid_size=50),
        # AgentRS(space),
        # AgentEvo(space, min_p=100, max_p=100, s=20),
    ]
    
    for agent in agents:
        print("\n\n\nAgent", agent.__class__)
        run_agent(space, agent, n_iter=100000, verbose=1)
        break


def main_cifar10():
    # entraînement très réduit juste pour voir le code tourner
    space = instances.cifar10(batch_size=3, reduced=True, train_params={"n_epochs":2, "max_iter":2})
    
    # on peut changer loss, optim, et lr_scheduler et leurs paramètres
    agents = [
        AgentRS(space, loss_param=(nn.CrossEntropyLoss, {}), optim_param=(Adam, {"lr":1e-5}),
                lr_scheduler_param=(ExponentialLR, {"gamma": .9})),
        AgentEvo(space, min_p=3, max_p=5, s=2),
    ]
    
    for agent in agents:
        print("\n\n\nAgent", agent.__class__)
        best_archi, best_val = run_agent(space, agent, n_iter=10, verbose=1)
        print("best archi", best_archi)
        print("best val", best_val)

main_learn_to_count()
# main_cifar10()
