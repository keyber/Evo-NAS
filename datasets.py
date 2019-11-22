import pandas as pd

def newsAggregator():
    path = "~/datasets/input/news-aggregator-dataset/uci-news-aggregator.csv"
    df = pd.read_csv(path)[:1500]
    # df["spacy_title"] = df["TITLE"].apply(lambda x : nlp(x))
