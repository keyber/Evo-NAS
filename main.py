import torch
import torch.utils.data

from baselineRS import AgentRS
from baselineEvo import AgentEvo
import numpy as np

def fit_model(model, data, n_epochs=10):
    for epoch in range(n_epochs):
        for x, y in data:
            model.net.zero_grads()
            y_pred = model.net(x)
            loss = model.loss(y_pred, y)
            loss.backward()
            model.optim.step()
        
        model.lr_scheduler.step(epoch)
        

def score_model(model, data):
    model.net.eval()
    loss = 0
    
    with torch.no_grad():
        for x, y in data:
            y_pred = model.net(x)
            loss += model.loss(y_pred, y).item()
    
    return loss


def run_agent(ds_train, ds_test, agent, n_iter=10, batch_size=32, verbose=0):
    train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, drop_last=True)
    test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, drop_last=True)
    best_a, best_v = None, -np.inf
    
    archi = agent.act(reward=None)
    
    
    for i in range(n_iter):
        model = archi.model()
        fit_model(model, train)
        score = score_model(model, test)
        
        if score > best_v:
            best_v = score
            best_a = archi
            if verbose==1:
                print("archi", archi, "score", score)
                
        if verbose==2:
            print("archi", archi, "score", score)
        
        archi = agent.act(reward=score)
        
    
    return best_a, best_v
        

def main():
    space = []
    agents = [AgentRS(space), AgentEvo(space, P=10, S=5)]
    
    for agent in agents:
        print("Agent", agent)
        best_arch, best_val = run_agent(ds_train, ds_test, agent)
        print("best_arch", best_arch, "best_val", best_val)
