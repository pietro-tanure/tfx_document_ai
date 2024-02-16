# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv
import datasets
# from tqdm import tqdm
# from itertools import islice

@click.command()
@click.argument('output_filepath', default= 'data/raw/DocLayNet-small', type=click.Path())#, required=False)
def download_dataset(output_filepath):
    """Download raw dataset from HuggingFace

    Args:
        output_filepath (str, optional): Path to cloud dataset on hugging face library or local dataset in PC.
        Defaults to 'data/raw/DocLayNet-small'. 
    """
    dataset_small = datasets.load_dataset("pierreguillou/DocLayNet-small")
    dataset_small.save_to_disk(output_filepath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # # find .env automagically by walking up directories until it's found, then
    # # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    download_dataset()
