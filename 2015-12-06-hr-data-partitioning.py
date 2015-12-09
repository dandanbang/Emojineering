
# coding: utf-8

# #Data partitioning
# Spliting data into training and test set, that will be held out until the very end of the project.
# Split by 80% training, 20% test

# In[1]:

import json
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import pandas as pd
pd.options.display.max_colwidth = 140
import nltk
import re


# In[2]:

with open('./data/tweets_1M.json','r') as f:
    tweets_df = pd.DataFrame(json.load(f))

train_percent = 0.8

training_size = round(len(tweets_df) * train_percent)
training_df = tweets_df.sample(training_size)
training_df.reset_index(drop=True, inplace=True)

test_size = len(tweets_df) - training_size
test_df = tweets_df.sample(test_size)
test_df.reset_index(drop=True, inplace=True)


# In[3]:

training_df.to_json('./data/tweets_training.json', force_ascii=False)
test_df.to_json('./data/tweets_test.json', force_ascii=False)


# In[5]:

training_df


# In[6]:

with open('./data/tweets_training.json','r') as f:
    test = pd.DataFrame(json.load(f))
test

