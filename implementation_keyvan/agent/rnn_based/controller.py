import torch
import torch.nn as nn
from implementation_keyvan.problem.searchSpace import SearchSpace


class Controller(nn.ModuleList):
    def __init__(self, space:SearchSpace, emb_size, hid_size):
        super().__init__()

        self.lstm_cell = nn.LSTMCell(emb_size, hid_size)

        encoders = []
        decoders = []

        for out_size in space.output_dims():
            decoders.append(nn.Linear(hid_size, out_size))
            encoders.append(nn.Embedding(out_size, emb_size))

        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        
        self.start_of_sequence = torch.zeros((1, emb_size))
    
    def modules(self):
        return self.lstm_cell, self.encoders, self.decoders, self.start_of_sequence
    


