from implementation_keyvan.problem.searchSpace import SearchSpace
import numpy as np
from random import random
import torch
from torch.functional import F
from implementation_keyvan.agent.rnn_based.controller import Controller

class SearchSpaceL2C(SearchSpace):
    def __init__(self, search_space: int):
        self.space = search_space
        self.ind_to_val = {i:i+1 for i in range(self.space)}
    
    
    def output_dims(self):
        return [self.space] * self.space
        # return {i:self.space for i in range(self.space)}
    
    
    def score(self, archi):
        """cf article: retourne
        1 pour des nombres dans l'ordre
        un nombre entre 0 et 1 sinon"""
        a = archi
        n = self.space
        
        s = a[0] ** 2
        
        for k in range(0, n - 1):
            s += (a[k + 1] - a[k]) ** 2
        
        s += (a[n - 1] - (n + 1)) ** 2
        
        return (n + 1) / s
    
    
    def random_sample(self, **kwargs):
        """Retourne une architecture aléatoire de l'espace"""
        return np.random.randint(1, self.space + 1, size=self.space)
    
    
    def mutate(self, archi, r=.5, **kwargs):
        values = archi.copy()
        ind = np.random.random(self.space) < r
        values[ind] = np.random.randint(1, self.space + 1, len(values[ind]))
    
        return values
    
    
    def rnn_sample(self, controller:Controller, **kwargs):
        list_result = []
        sum_log_proba = 0
        
        emb = controller.start_of_sequence
        h, c = controller.lstm_cell(emb)
        
        for i in range(self.space):
            logits = controller.decoders[i](h).squeeze(0)
            probas = F.softmax(logits, dim=0)
            log_probas = F.log_softmax(logits, dim=0)
            
            # result = torch.argmax(probas)
            result = torch.multinomial(probas, num_samples=1)[0]
            
            emb = controller.encoders[i](result.unsqueeze(0))
            h, c = controller.lstm_cell(emb, (h, c))

            list_result.append(self.ind_to_val[result.item()])
            sum_log_proba += log_probas[result.item()]
    
        return list_result, sum_log_proba
    
     
    def rnn_mutate(self, archi, controller, r=.5, **kwargs):
        result = archi.copy()
        
        emb = controller.emb(torch.zeros())
        h = controller.rnn_cell(emb)
        
        for i in range(self.space):
            if random() < r:
                result[i] = controller.classifier[i](h)
            
            emb = controller.emb(result[i])
            h = controller.rnn_cell(emb)
    
        return np.array(result)
