# from archi.architecture import Architecture
import torch
import torch.nn as nn
from torch.optim import Adam


class ArchitectureNN:
    """Contient un réseau de neurones, une loss et un optimiser"""
    
    def __init__(self, gbl_values, loc_values, net, loss_param=None, optim_param=None, lr_scheduler_param=None):
        self.gbl_values = gbl_values
        self.loc_values = loc_values
        self.net = net
        
        self.loss_param = loss_param
        self.optim_param = optim_param
        self.lr_scheduler_param = lr_scheduler_param
        
        self.loss = loss_param[0](**loss_param[1]) if loss_param is not None \
            else nn.CrossEntropyLoss()
        self.optim = optim_param[0](self.net.parameters(), **optim_param[1]) if optim_param is not None \
            else Adam(self.net.parameters())
        self.lr_scheduler = lr_scheduler_param[0](self.optim, **lr_scheduler_param[1]) if lr_scheduler_param is not None \
            else None
    
    def __str__(self):
        return str(self.gbl_values) + " " + str(self.loc_values)
    
    def fit_model(self, data, n_epochs=2, max_iter=1e8):
        i = 0
        for epoch in range(n_epochs):
            for x, y in data:
                self.net.zero_grad()
                y_pred = self.net(x)
                loss = self.loss(y_pred, y)
                loss.backward()
                self.optim.step()
                i += 1
                
                if i == max_iter:
                    return
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch)
    
    def compute_test_score(self, data):
        self.net.eval()
        loss = 0
        tot = 0
        
        with torch.no_grad():
            for x, y in data:
                y_pred = self.net(x)
                loss += self.loss(y_pred, y).item()
                tot += len(x)
        
        return loss / tot
    
    # def score_model(self):
    #     self._fit_model(self.search_space.get_train_data())
    #     
    #     return self._compute_test_score(self.search_space.get_test_data())
    # 
    # def mutate(self, r=.5):
    #     """Retourne une nouvelle architecture en re-samplant les caractéristiques avec une probabilité r"""
    #     gbl_space = self.search_space.get_space().gbl
    #     loc_space = self.search_space.get_space().loc
    #     
    #     gbl_values = {k: choice(gbl_space[k]) for k in gbl_space}
    #     
    #     n = gbl_values["layer_number"]
    #     loc_values = []
    #     for i in range(n):
    #         if i < self.gbl_values["layer_number"]:
    #             loc_values.append({k: choice(loc_space[k]) if random() < r
    #             else self.loc_values[i][k]
    #                                for k in loc_space})
    #         else:
    #             loc_values.append({k: choice(loc_space[k]) for k in loc_space})
    #     
    #     return ArchitectureNN(self.search_space, gbl_values, loc_values, loss_param=self.loss_param,
    #                         optim_param=self.optim_param, lr_scheduler_param=self.lr_scheduler_param)
    # 
    # @staticmethod
    # def random_sample(search_space: SearchSpaceNN, loss=None, optim=None, lr_scheduler=None):
    #     """Retourne une architecture aléatoire de l'espace"""
    #     gbl_space = search_space.get_space().gbl
    #     loc_space = search_space.get_space().loc
    #     
    #     gbl_values = {k: choice(gbl_space[k]) for k in gbl_space}
    #     
    #     n = gbl_values["layer_number"]
    #     loc_values = [{k: choice(loc_space[k]) for k in loc_space} for _ in range(n)]
    #     
    #     return ArchitectureNN(search_space, gbl_values, loc_values, loss_param=loss,
    #                         optim_param=optim, lr_scheduler_param=lr_scheduler)
