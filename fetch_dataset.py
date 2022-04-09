import os
from kaggle.api.kaggle_api_extended import KaggleApi
import click
import logging

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--dataset",
    default="Cornell-University/arxiv",
    help="kaggle path to dataset",
    type=str,
)
@click.option(
    "--file_name",
    default="arxiv-metadata-oai-snapshot.json",
    help="Name of target file in dataset",
)
@click.option(
    "--savepath",
    default="./data/arxiv-metadata-oai-snapshot.json",
    help="Where dataset is stored",
    type=str,
)
def fetch_dataset(dataset, file_name, savepath):
    """
    Use the kaggle api to fetch a dataset from kaggle.
    Put the kaggle API .JSON file in '~/.kaggle/',
    or the api is not usable. 

    Parameters
    ----------
    dataset : str
        Kaggle path to dataset
    savepath : str
        Where to save the retrieved dataset
    """
    if not os.path.exists("./data"):
        os.makedir("./data")
        logger.info("Created directory: './data'.")

    kaggle_api = KaggleApi()
    kaggle_api.authenticate()
    logger.info("Authentication completed")
    kaggle_api.dataset_download_file(
        dataset=dataset, file_name=file_name, path=savepath, unzip=True
    )


if __name__ == "__main__":
    fetch_dataset()
