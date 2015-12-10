
# coding: utf-8

# The goal is to try to do hierarchical clustering of emojis, based on word2vec, ie cosine similarity between word contexts. 
# 
# Inspired from data_exploration_dc.ipynb (but no functions so I couldn't import it! ;))

# In[12]:

import gensim
from gensim.models import Word2Vec

import data_cleaning_hr as clean


# In[4]:

emoji_model = gensim.models.Word2Vec.load('emoji.embedding')


# In[7]:

emoji_model.most_similar(positive = ['😊'], topn=)


# In[ ]:

# Load list of emojis from CSV that Carlo will finish
# 

