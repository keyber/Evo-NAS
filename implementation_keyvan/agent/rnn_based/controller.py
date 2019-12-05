import torch.nn as nn
from implementation_keyvan.problem.searchSpace import SearchSpace


class Controller(nn.Module):
    def __init__(self, space:SearchSpace, emb_size, hid_size):
        super().__init__()

        self.lstm = nn.LSTMCell(emb_size, hid_size)

        encoders = []
        decoders = []

        for out_size in space.output_dims():
            decoders.append(nn.Linear(hid_size, out_size))
            encoders.append(nn.Embedding(out_size, emb_size))

        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)


