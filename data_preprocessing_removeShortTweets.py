
# coding: utf-8

# In[6]:

import nltk
import pandas as pd
import json
import itertools

from data_cleaning import loader


# In[7]:

tweets_df = loader("./data/tweets_training_clean_preprocessing.json")
# tweets_df = loader("./data/tweets_test_clean_preprocessing.json")


# In[8]:

# Filter tweets by those of more than 50 characters 
tweets_df_filtered = tweets_df[tweets_df.only_text_splithashtag.apply(lambda x: len(x)) > 50]
sum(tweets_df_filtered.only_emoji.apply(lambda x: len(x) > 0))


# In[9]:

# Look at distribution of emojis after filtering short ones
temp = tweets_df_filtered.only_emoji.values.flatten().tolist()
chain = itertools.chain(*temp)
fdist = nltk.FreqDist(chain)
fdist.most_common()


# In[10]:

tweets_df_filtered.to_json('./data/tweets_training_clean_preprocessingv2.json', force_ascii=False)
# tweets_df_filtered.to_json('./data/tweets_test_clean_preprocessingv2.json', force_ascii=False)

