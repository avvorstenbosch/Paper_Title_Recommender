# download packages
#!pip install transformers==4.8.2

# import packages
import os
import re
import torch
import random
import pandas as pd
from tqdm import tqdm
## Define class and functions
#--------

# Dataset class
df = pd.read_json('arxiv-metadata-oai-snapshot.json', lines=True)
df = df[['title', 'comments', 'categories', 'abstract']]
df = df[df.abstract.str.len() < 2000]
df.categories = df.categories.str.split(" ").str[0]
df.to_csv("arxiv_metadata_small.csv", index=False)
