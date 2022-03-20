# download packages
#!pip install transformers==4.8.2

# import packages
import os
import re
import torch
import random
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPT2LMHeadModel

## Define class and functions
#--------

# Dataset class
class ArxivDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        # define variables    
        print("encoding data with tokenizer")
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        # iterate through the dataset
        prep_txt = df.apply(lambda x: f'<|startoftext|> Categorie: {x.categories} \nAbstract: {x.abstract}\nTitle: {x.title}<|endoftext|>', axis=1)
        # tokenize,axis=1)
        for txt in prep_txt:
            # tokenize
            encodings_dict = tokenizer(txt, truncation=True,
                                       max_length=max_length, padding="max_length")
            # append to list
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
            self.labels.append(txt.split("Title:")[1].split("<")[0])
            
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.labels[idx]

# Data load function
def load_arxiv_dataset(tokenizer, samples=170000):
    # load dataset and sample 10k reviews.
    print("Loading data")
    file_path = "arxiv_metadata_small.csv"
    df = pd.read_csv(file_path)
    df = df.sample(samples)#350000, random_state=1)

    # divide into test and train
    X_train, X_test = \
              train_test_split(df, shuffle=True, test_size=20, random_state=1)#, stratify=df['categories'])

    # format into SentimentDataset class
    train_dataset = ArxivDataset(X_train, tokenizer, max_length=1024)

    # return
    return train_dataset, X_test

## Load model and data
#--------
    
save_name = "/home/alex/DeepLearning/sentiment_transformer/results/arxiv-model-medium-20220308"
override = 0
if os.path.exists(save_name) and override==False:
    model_name = save_name
    model = GPT2LMHeadModel .from_pretrained(model_name).cuda()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium', bos_token='<|startoftext|>',
                                              eos_token='<|endoftext|>', pad_token='<|pad|>')
    train_dataset, test_dataset = load_arxiv_dataset(tokenizer, samples=30)
    
else:
    # set model name
    model_name = "gpt2-medium"
    # seed
    torch.manual_seed(42)
    
    # load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, bos_token='<|startoftext|>',
                                              eos_token='<|endoftext|>', pad_token='<|pad|>')
    model = GPT2LMHeadModel.from_pretrained(model_name).cuda()
    model.resize_token_embeddings(len(tokenizer))
    
    # prepare and load dataset
    train_dataset, test_dataset = load_arxiv_dataset(tokenizer)
    
    ## Train
    #--------
    # creating training arguments
    training_args = TrainingArguments(output_dir='results', num_train_epochs=1, logging_steps=10,
                                     load_best_model_at_end=False, save_strategy="steps", save_steps=55000, do_eval=False, evaluation_strategy="no",
                                     per_device_train_batch_size=1,gradient_accumulation_steps=8,
                                     warmup_steps=100, weight_decay=0.001, logging_dir='logs')
    
    # start training
    print("Train model")
    Trainer(model=model, args=training_args, train_dataset=train_dataset,
            data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                        'attention_mask': torch.stack([f[1] for f in data]),
                                        'labels': torch.stack([f[0] for f in data])}).train()
    
        
    model.save_pretrained(save_name)
## Test
# set the model to eval mode
_ = model.eval()

# run model inference on all test data
original_label, predicted_label, original_text, predicted_text = [], [], [], []
titles = test_dataset.title
test_txt = test_dataset.apply(lambda x: f'<|startoftext|> Categorie: {x.categories} \nAbstract: {x.abstract}\nTitle: ', axis=1)
        
for (txt, title) in zip(test_txt, titles):
    # generate tokens
    generated = tokenizer(f"{txt}", return_tensors="pt").input_ids.cuda()
    # perform prediction
    MLE_output = model.generate(generated, max_length=1024,  
            num_return_sequences=1, early_stopping=True, num_beams=30)
    random_outputs = model.generate(generated, max_length=1024, do_sample=True, top_p=0.9,  
            temperature=1.5, num_return_sequences=4, early_stopping=True)
    # decode the predicted tokens into texts
    pred_texts_mle = [tokenizer.decode(decoded, skip_special_tokens=True) for decoded in MLE_output]
    pred_texts_random = [tokenizer.decode(decoded, skip_special_tokens=True) for decoded in random_outputs]
    # extract the predicted sentiment
    print("\n\n----------REAL----------\n",txt+title, "\n--------------Generated------------------\n")
    for pred in pred_texts_mle:
        print("\nGenerated MLE Title:", pred.split("\nTitle:")[1],"\n")
    for pred in pred_texts_random:
        print("\nGenerated random Title:", pred.split("\nTitle:")[1],"\n")
    