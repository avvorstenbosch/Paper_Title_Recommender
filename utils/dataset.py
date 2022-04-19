# import packages
import os
import re
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch

# setup logging
import logging

logger = logging.getLogger(__name__)

# Dataset class
class ArxivDataset(Dataset):
    """
    Class for loading and processing Arxiv Dataset from Kaggle

    Parameters
    ----------
    Dataset : Pandas.DataFrame
        Dataframe containing the arxiv dataset
    """

    def __init__(self, df, tokenizer, max_length):
        """
        Initializes objects to be used for __getitem__ and __len__.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the columns 'categorie', 'title' and 'abstract'
        tokenizer : huggingface.PreTrainedTokenizer
            tokenizer used to proces the data
        max_length : int
            maximum length of the text, it will either be cast or truncated to this length in terms of tokens.
        """
        self.input_ids = []
        self.attn_masks = []
        self.labels = []

        logger.info("Casting input strings to correct text format.")
        # Apply correct text formatting to the data
        prep_txt = df.apply(
            lambda x: f"<|startoftext|> Categorie: {x.categories} \nAbstract: {x.abstract}\nTitle: {x.title}<|endoftext|>",
            axis=1,
        )

        # Tokenize processed tetxt
        logger.info("Applying pretrained GPT2-tokenizer to input text.")
        for txt in prep_txt:
            # tokenize
            encodings_dict = tokenizer(
                txt, truncation=True, max_length=max_length, padding="max_length"
            )
            # append to lists
            self.input_ids.append(torch.tensor(encodings_dict["input_ids"]))
            self.attn_masks.append(torch.tensor(encodings_dict["attention_mask"]))
            self.labels.append(txt.split("Title:")[1].split("<")[0])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.labels[idx]


# Data loading function
def load_arxiv_dataset(tokenizer, samples=2e5):
    """
    Load Arxiv dataset and select N samples to use for finetuning

    Parameters
    ----------
    tokenizer : huggingface.PreTrainedTokenizer
        tokenizer used to proces the data
    samples : int
        Number of samples to draw from full Arxiv dataset

    Returns
    -------
    train_dataset : torch.Dataset
        Dataset class object containing processed data for finetuning
    X_test : pd.DataFrame
        Unprocessed samples to be used for testing the algorithm performance. 
    """
    # load dataset and sample N Abstracts.
    logger.info("Loading dataset from processed CSV.")
    file_path = "./data/arxiv_metadata_small.csv"
    df_full = pd.read_csv(file_path)
    try:
        df = df.sample(samples, random_state=2112)
    except ValueError as e:
        logger.exception(
            f"The max amount of samples is {len(df_full):.0f}, you tried to select {samples:.0f} samples."
        )
    # divide into test and train
    logger.info("Creating train and test sets.")
    X_train_raw, X_test = train_test_split(
        df, shuffle=True, test_size=0.1, random_state=2112
    )

    # format into SentimentDataset class
    logger.info("Preprocess data into final format torch.Dataset format.")
    X_train = ArxivDataset(X_train_raw, tokenizer, max_length=1024)

    # return
    return X_train, X_test


def process_raw_arxiv_dataset(
    path_raw_data="./data/arxiv-metadata-oai-snapshot.json",
    savepath="./data/arxiv_metadata_small.csv",
):
    """
    Generate a smaller usefull dataset from the full metadata set

    Parameters
    ----------
    path_raw_data : str, optional
        location of the raw datafile, by default "./data/arxiv-metadata-oai-snapshot.json"
    savepath : str, optional
        location where to save processe data, by default "./data/arxiv_metadata_small.csv"
    """
    df = pd.read_json(path_raw_data, lines=True)
    df = df[["title", "categories", "abstract"]]

    # Remove abstracts that are to long for the model
    # 2000 is chosen as a safe value
    MAX_LEN_ABSTRACT = 2000
    df = df[df.abstract.str.len() <= MAX_LEN_ABSTRACT]

    # Select first categorie as THE categorie
    df.categories = df.categories.str.split(" ").str[0]
    df.to_csv(savepath, index=False)
