#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 18:18:48 2022

@author: alex
"""

# import
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# model name
model_name = "gpt2"

# load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).cuda()

# create prompt
prompt = """
Below are some examples for sentiment detection of movie reviews.

Review: I am sad that the hero died.
Sentiment: Negative

Review: The ending was perfect.
Sentiment: Positive

Review: The plot was not so good!
Sentiment:"""

# generate tokens
generated = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

# perform prediction 
sample_outputs = model.generate(generated, do_sample=False, top_k=50, max_length=512, top_p=0.90, 
        temperature=0, num_return_sequences=0)

# decode the predicted tokens into texts
predicted_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
print(predicted_text)
