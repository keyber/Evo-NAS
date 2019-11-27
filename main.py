import torch
import torch.utils.data
import datasets
from baselineRS import AgentRS
from baselineEvo import AgentEvo
import numpy as np

def fit_model(model, data, n_epochs=2, max_iter=1e8):
    i = 0
    for epoch in range(n_epochs):
        for x, y in data:
            model.net.zero_grad()
            y_pred = model.net(x)
            loss = model.loss(y_pred, y)
            loss.backward()
            model.optim.step()
            i += 1
            
            if i==max_iter:
                return 
        
        if model.lr_scheduler is not None:
            model.lr_scheduler.step(epoch)
        

def score_model(model, data):
    model.net.eval()
    loss = 0
    
    with torch.no_grad():
        for x, y in data:
            y_pred = model.net(x)
            loss += model.loss(y_pred, y).item()
    
    return loss


def run_agent(ds_train, ds_test, agent, n_iter=10, train_max_iter=1e8, batch_size=7, verbose=0):
    train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, drop_last=True)
    test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, drop_last=True)
    best_a, best_v = None, -np.inf
    
    archi = agent.act(last_reward=None)
    
    
    for i in range(n_iter):
        fit_model(archi, train, max_iter=train_max_iter)
        score = score_model(archi, test)
        
        if score > best_v:
            best_v = score
            best_a = archi
            if verbose==1:
                print(score, archi.architecture)
                
        if verbose==2:
            print(score, archi.architecture)
        
        archi = agent.act(last_reward=score)
        
    
    return best_a, best_v
        

def main():
    space, ds_train, ds_test = datasets.cifar10(reduced=True)
    agents = [AgentEvo(space, min_p=3, max_p=5, s=2), AgentRS(space)]
    
    for agent in agents:
        print("Agent", agent.__class__)
        best_arch, best_val = run_agent(ds_train, ds_test, agent, n_iter=10, train_max_iter=2, verbose=2)
        print("best archi", best_arch.architecture)
        print("best val", best_val)
        print("\n\n\n")

main()
