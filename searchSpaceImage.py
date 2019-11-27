import torch.nn as nn


_SEARCH_SPACE_IMAGE = {
    "global" : {
        "layer_number": (5, 10, 15),
    },
    "local" : {
        "filter_width" : (1, 3, 5, 7),
        "filter_number" : (24, 36, 48, 64),
        "filter_stride" : (1, 2, 3),
        "filter_padding" : (0,),
    }
}
_SEARCH_SPACE_IMAGE_REDUCED = {
    "global" : {
        "layer_number": (1, 2, 3, 4),
    },
    "local" : {
        "filter_width" : (1, 3, 5),
        "filter_number" : (24, 36, 48, 64),
        "filter_stride" : (1, 2, 3),
        "filter_padding" : (0,),
    }
}


class SearchSpaceImage:
    """Définit les choix d'architecture possibles et la manière de créer le réseau"""
    _warn = True
    
    def __init__(self, out_channels, imagewidth, reduced_space=False, in_channels=3):
        self.imagewidth = imagewidth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.space = _SEARCH_SPACE_IMAGE_REDUCED if reduced_space else _SEARCH_SPACE_IMAGE
    
    def create_model(self, choices):
        conv = []
        in_channels = self.in_channels
        image_width = self.imagewidth
        
        for layer in choices["local"]:
            n = layer["filter_number"]
            w = layer["filter_width"]
            s = layer["filter_stride"]
            p = layer["filter_padding"]
            
            if w > image_width:
                if SearchSpaceImage._warn:
                    SearchSpaceImage._warn = False
                    print("EarlyStoppingWarning: le kernel est plus grand que l'image")
            else:
                conv.append(nn.Conv2d(in_channels=in_channels, out_channels=n,
                                   kernel_size=w, stride=s, padding=p))
                
                conv.append(nn.ReLU())
                
                in_channels = n
                image_width = (image_width + 2 * p - w) // s + 1  # cf tp 6 de RDFIA
        
        
        conv = nn.Sequential(*conv)
        
        fc = nn.Linear(in_channels * image_width**2, self.out_channels)
        
        return SearchSpaceImage.Net(conv, fc)
    
    class Net(nn.Module):
        def __init__(self, conv, fc):
            super().__init__()
            self.conv = conv
            self.fc = fc
        
        def forward(self, x):
            batch_size = x.shape[0]
            x = self.conv(x)
            x = x.reshape((batch_size, -1))
            x = self.fc(x)
            return x

