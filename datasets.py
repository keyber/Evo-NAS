import pandas as pd
from searchSpaceImage import SearchSpaceImage
from searchSpaceText import SearchSpaceText
import torchvision.datasets
from torchvision import transforms


def newsAggregator():
    path = "~/datasets/input/news-aggregator-dataset/uci-news-aggregator.csv"
    df = pd.read_csv(path)[:1500]
    # df["spacy_title"] = df["TITLE"].apply(lambda x : nlp(x))
    # todo

def cifar10(reduced=False):
    t = transforms.Compose([
            # transforms.Resize((224, 224),interpolation=Image.NEAREST),
            #transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), #mean std
        ])
    
    return \
        SearchSpaceImage(out_channels=10, in_channels=3, imagewidth=32, reduced_space=reduced),\
        torchvision.datasets.CIFAR10(root="~/datasets", download=True, transform=t, train=True),\
        torchvision.datasets.CIFAR10(root="~/datasets", download=True, transform=t, train=False)
