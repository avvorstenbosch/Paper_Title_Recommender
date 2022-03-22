# download packages
#!pip install transformers==4.8.2

# import packages
from asyncio.log import logger
import os
import re
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPT2LMHeadModel

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
        logger.info("Applying pretrained tokenizer to input text.")
        for txt in prep_txt:
            # tokenize
            encodings_dict = tokenizer(
                txt, truncation=True, max_length=max_length, padding="max_length"
            )
            # append to list
            self.input_ids.append(torch.tensor(encodings_dict["input_ids"]))
            self.attn_masks.append(torch.tensor(
                encodings_dict["attention_mask"]))
            self.labels.append(txt.split("Title:")[1].split("<")[0])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.labels[idx]


# Data load function
def load_arxiv_dataset(tokenizer, samples):
    """
    load Arxiv dataset and select N samples to use for tinetuning

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
    # load dataset and sample 10k reviews.
    logger.info("Loading dataset from processed CSV.")
    file_path = "../data/arxiv_metadata_small.csv"
    df = pd.read_csv(file_path)
    df = df.sample(100, random_state=2112)

    # divide into test and train
    logger.info("Creating train and test sets.")
    X_train, X_test = train_test_split(
        df, shuffle=True, test_size=20, random_state=21172
    )  # , stratify=df['categories'])

    # format into SentimentDataset class
    logger.info("Preprocess data into final format torch.Dataset format.")
    train_dataset = ArxivDataset(X_train, tokenizer, max_length=1024)

    # return
    return train_dataset, X_test
