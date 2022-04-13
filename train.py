import os
import random
import pandas as pd
from datetime import today

import click
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPT2LMHeadModel
from kaggle.api.kaggle_api_extended import KaggleApi
from utils.dataset import process_raw_arxiv_dataset, load_arxiv_dataset

import logging

logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
savepath = "./results/arxiv-model-" + today.strftime("%Y%m%d")


@click.command()
@click.option("--savepath", help="Where to save trained model", default=savepath)
@click.option(
    "--override", help="If savepath occupied, should it be overwriten?", default=False
)
@click.option("--device", help="Select on which device to train the NN", default=device)
@click.option("--samples", help="How many samples to train on", default=2e5)
@click.option(
    "--num_train_epochs", help="How many training epochs to perform", default=1
)
@click.option("--save_steps", help="Save model every N training steps", default=50000)
@click.otpion(
    "--per_device_train_batch_size", help="Batchsize as sent to device", default=1
)
@click.option(
    "--gradient_accumalation_steps",
    help="How many per_device_batches to accumalate per training step",
    default=8,
)
def train():
    if device == "cuda" and not torch.cuda.is_available():
        logger.error("No GPU detected, please select a different device for training.")
        raise RuntimeError("Cuda device not available")

    if os.path.exists(savepath) and override == False:
        logger.error("Trained model already exists.")
        logger.error("Switch to inference mode, or set 'override'=True")

    if not os.path.exists("./data/arxiv-metadata-oai-snapshot.json"):
        logger.error(
            "The training data cannot be found, please make sure the data is located in ./data"
        )
        logger.error(
            "You can fetch the data from kaggle.com by running 'fetch_dataset.py'"
        )
        raise RuntimeError("Dataset wasn't found")
    elif not os.path.exists("./data/arxiv_metadata_small.csv"):
        logger.info("Processing raw data into smaller usable dataset.")
        process_raw_arxiv_dataset()
        logger.info("Finished processing raw data.")

    # Choose correct model
    model_name = "gpt2-medium"
    # Set random seed for reproducibility
    torch.manual_seed(42)

    logger.info("Setting base tokenizer from Huggingface.")
    tokenizer = GPT2Tokenizer.from_pretrained(
        model_name,
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
    )
    X_train, X_test = load_arxiv_dataset(tokenizer, samples=samples)

    logger.info("Setting base model from Huggingface, rescaling token embeddings")
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.resize_token_embeddings(len(tokenizer))

    logger.info("Setting training arguments.")
    training_args = TrainingArguments(
        output_dir="results",
        num_train_epochs=num_train_epochs,
        logging_steps=10,
        load_best_model_at_end=False,
        save_strategy="steps",
        save_steps=save_steps,
        do_eval=False,
        evaluation_strategy="no",
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        weight_decay=0.001,
        logging_dir="logs",
    )

    logger.info("Starting model training.")
    Trainer(
        model=model,
        args=training_args,
        train_dataset=X_train,
        data_collator=lambda data: {
            "input_ids": torch.stack([f[0] for f in data]),
            "attention_mask": torch.stack([f[1] for f in data]),
            "labels": torch.stack([f[0] for f in data]),
        },
    ).train()

    logger.info("Finished training sequence, saving model.")
    model.save_pretrained(savepath)


if __name__ == "__main__":
    train()
