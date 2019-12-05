import pandas as pd
from implementation_keyvan.problem.searchSpaceImage import SearchSpaceImage
from implementation_keyvan.problem.searchSpaceL2C import SearchSpaceL2C
import torchvision.datasets
import torch.utils
from torchvision import transforms


def learn_to_count(space_size):
    return SearchSpaceL2C(space_size)
    
    
def cifar10(batch_size, reduced=False, train_params=None):
    t = transforms.Compose([
            # transforms.Resize((224, 224),interpolation=Image.NEAREST),
            #transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), #mean std
        ])
    
    train = torchvision.datasets.CIFAR10(root="~/datasets", download=True, transform=t, train=True)
    test = torchvision.datasets.CIFAR10(root="~/datasets", download=True, transform=t, train=False)
    
    train = torch.utils.data.DataLoader(train, batch_size=batch_size, drop_last=True)
    test = torch.utils.data.DataLoader(test, batch_size=batch_size, drop_last=True)
    
    return SearchSpaceImage(train, test, out_channels=10, in_channels=3, imagewidth=32,
                            reduced_space=reduced, train_params=train_params)


def newsAggregator():
    path = "~/datasets/input/news-aggregator-dataset/uci-news-aggregator.csv"
    df = pd.read_csv(path)[:1500]
    # df["spacy_title"] = df["TITLE"].apply(lambda x : nlp(x))
    # todo

