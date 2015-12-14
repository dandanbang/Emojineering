
# coding: utf-8

# In[3]:

## import all necessary packages
import json
import re
import pandas as pd

import nltk, string
from nltk.collocations import *

import numpy as np
import matplotlib.pyplot as plt

from pandas import *

from collections import defaultdict

import string
import happyfuntokenizing
import json


# In[4]:

with open("./data/categories_manual.json") as json_file:
    category_dict = json.load(json_file)


# In[5]:

with open('./data/emoji_webscraped_expanded.json','r') as f_emoji:
    emoji_df = DataFrame(json.load(f_emoji))


# In[6]:

with open('./data/tweets_test_clean.json','r') as f:
    tweet_df = DataFrame(json.load(f))


# In[7]:

tweet_df.head()


# ## Statistics on all emojis

# **emoji_array**:  list of all emojis

# In[8]:

emoji_array = [word for item in tweet_df.only_emoji for word in item]
len(emoji_array)


# **unique_emoji**:  number of unique emojis

# In[9]:

unique_emoji = set(emoji_array)
len(unique_emoji)


# **full_dict**: percentage of emojis by emoji

# In[10]:

full_dict = defaultdict(int)
for item in emoji_array:
    full_dict[item] += 1


# In[11]:

for item in sorted(full_dict.items(), key=lambda x:x[1], reverse=True)[:50]:
    print(item[0], (float(item[1])/len(emoji_array))*100)


# ## Statistics on only faces

# **unique_face**: unique faces from category_dict file

# In[12]:

unique_face = [face for item in category_dict.values() for face in item]
len(unique_face)


# **all_faces**: from emoji_array (list of all emojis), subset by unique_face

# In[13]:

all_faces = [unique for face in emoji_array for unique in unique_face if face in unique]
len(all_faces)


# In[14]:

all_faces_unique = set(all_faces)
len(all_faces_unique)


# **categorize**: list of all_faces defined by their respective category from cateogry_dict

# In[15]:

cateogrize = []
for category, faces in category_dict.items():
    for every_face in all_faces:
            if every_face in faces:
                cateogrize.append((category, every_face))


# In[16]:

# cateogrize


# **categorize_percentage**: dictionary with count of each categroy form categorize

# In[17]:

category_percentage = defaultdict(int)
for item in cateogrize:
    category_percentage[item[0]] += 1


# In[18]:

for item in sorted(category_percentage.items(), key=lambda x:x[1], reverse=True)[:50]:
    print(item[0], (float(item[1])/len(cateogrize))*100)


# ## Add category column to dataframe

# In[19]:

def cateogry_column(text):
    if len(text) == 0:
        return None
    
    boolean_face = [True if face in unique_face else False for face in text]
#     print(boolean_face)
    
    for item in boolean_face:
        if all(boolean_face) == True:
        
            cateogrize = []
            for category, faces in category_dict.items():
                for face in text:
                        if face in faces:
                            cateogrize.append(category)
            cateogrize = list(set(cateogrize))
            if len(cateogrize)==1:
                return cateogrize[0]
            else:
                return "Mix_P"
                
            return list(set(cateogrize))
        
        elif all(not element for element in boolean_face):
            return "Other"
#         else:
#             return "Mix_P_NonP"
        else:
            true_boolean = [index for index, item in enumerate(boolean_face) if item == True]
            keep_true = [text[i] for i in true_boolean]
            
            cateogrize = []
            for category, faces in category_dict.items():
                for face in keep_true:
                        if face in faces:
                            cateogrize.append(category)
            cateogrize = list(set(cateogrize))
            if len(cateogrize)==1:
                return cateogrize[0]
            else:
                return "Mix_P"

    return


# In[20]:

print(cateogry_column([]))
print(cateogry_column(["❤"]))
print(cateogry_column(["👌"]))
print(cateogry_column(["😒", "😀"]))
print(cateogry_column(["😒", "👌"]))
print(cateogry_column(["😒","❤","😩", "👌"]))


# In[21]:

tweet_df["cateogry_prediction"] = tweet_df.only_emoji.apply(cateogry_column)


# In[22]:

dict_category_prediction = defaultdict(int)
for item in tweet_df.cateogry_prediction:
    dict_category_prediction[item] += 1


# In[23]:

dict_category_prediction


# In[24]:

(11559 + 52558 + 38702 + 2345) / (len(tweet_df) - 625122)


# In[25]:

(11559 + 52558 + 38702 + 2345)


# ## Make Final column numeric

# In[26]:

tweet_df.head()


# In[27]:

numeric_category = [(index, value) for index, value in enumerate(list(set(tweet_df.cateogry_prediction)))]
numeric_category


# In[28]:

def make_numeric(text):
    _num = [item[0] for item in numeric_category if text == item[1]]
    return _num[0]


# In[29]:

make_numeric(None)


# In[30]:

tweet_df['category_numeric'] = tweet_df.cateogry_prediction.apply(make_numeric)


# In[31]:

tweet_df.shape


# In[32]:

tweet_df.to_json('./data/tweets_test_clean_preprocessing.json', force_ascii=False)


# In[33]:

tweet_df


# In[ ]:



