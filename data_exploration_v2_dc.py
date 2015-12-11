
# coding: utf-8

# #Load Library 

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

import data_cleaning_hr as tc

import string
import happyfuntokenizing


# #Load Data

# In[ ]:

with open('./data/tweets_training.json','r') as f:
    tweet_df = DataFrame(json.load(f))


# #Clean Data (Handle, URL, Emoticon Conversion)

# In[21]:

tc.cleanHandle(tweet_df)
tc.cleanURL(tweet_df)
tc.convertEmoticon(tweet_df)
tc.cleanRetweets(tweet_df)
tweet_df.head()


# #Emoji Finder

# In[22]:

try:
    # Wide UCS-4 build
    highpoints = re.compile(u'['
        u'\U0001F300-\U0001F64F'
        u'\U0001F680-\U0001F6FF'
        u'\u2600-\u26FF\u2700-\u27BF]+', 
        re.UNICODE)
except re.error:
    # Narrow UCS-2 build
    highpoints = re.compile(u'('
        u'\ud83c[\udf00-\udfff]|'
        u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
        u'[\u2600-\u26FF\u2700-\u27BF])+', 
        re.UNICODE)


# In[1]:

faces = re.compile(u'['
        u'\U0001F600-\U0001F64F]',
        re.UNICODE)
# Function that takes a list of text and return text that contains just faces
def just_face(text):
    return (faces.findall(text))
# Function that take a list of text and return text with just emojis
def just_emojis(text):
    return (highpoints.findall(text))
# Function that a list of text and return the number of emojis in the text.
def count_emojis(text):
    return len(highpoints.findall(text))

# Functions to check whether there's an emoji in the text, return 1 if true, 0 if false
def is_emoji(text):
    if highpoints.search(text):
        return 1
    else:
        return 0

    # Functions to check whether there's a face emoji in the text, return 1 if true, 0 if false
def is_face(text):
    if faces.search(text):
        return 1
    else:
        return 0


# In[8]:

emoji_df["Emoji Count"] = emoji_df["text"].apply(count_emojis)
emoji_df["Text Length"] = emoji_df["text"].apply(lambda x: len(x))
emoji_df = emoji_df[['id', 'text', 'timeStamp', 'user_id', 'Emoji Count', 'Text Length']]
emoji_df.describe()


# # Exploring Emoji vs no Emoji (20% Text has Emojis, 15% has face)

# ###Functions to check whether it's emoji or face

# In[73]:

tweet_df["is_emoji"] = tweet_df.text.apply(is_emoji)
tweet_df["is_face"] = tweet_df.text.apply(is_face)
tweet_df.head()


# In[25]:

tweet_df.describe()


# ##Take a look at the imported tokenizer and test it

# In[29]:

tok = happyfuntokenizing.TweetTokenizer(preserve_case=False)
samples = (
    u"RT @ #happyfuncoding: this is a typical Twitter tweet😖",
    u"😂😂😂 RT @Yours_Truly3x: Bitch brush yoo mouth; other Web oddities can be an &aacute;cute <em class='grumpy'>pain</em> >:(",
    u"Yay my cat is cuddling🔫 with me tonight❤ +1 (800) 123-4567, (800) 123-4567, and 123-4567 are treated as words despite their whitespace."
    )


# In[31]:

for s in samples:
        print("======================================================================")
        print(s)
        tokenized = tok.tokenize(s)
        print(list(tokenized))
        print("\n")


# #Add a new column that display just the emojis for the text

# In[32]:

def emojiExtract(sent):
    return [word for word in tok.tokenize(sent) if is_emoji(word) == 1]

def textExtract(sent):
    return [word for word in tok.tokenize(sent) if is_emoji(word) == 0]

def textTokenized(sent):
    return [word for word in tok.tokenize(sent)]

def addTokenizedText(df):
    df['Tokenized'] = [textTokenized(word) for word in df.text]

def addEmojiCol(df):
    df['Emoji'] = [emojiExtract(word) for word in df.text]

def addText(df):
    df['only_Text'] = [textExtract(word) for word in df.text]

def addHashTag(df):
    df['only_HashTag'] = [re.findall(r"(#\w+)", word) for word in df.text]


# In[40]:

addEmojiCol(tweet_df)
addText(tweet_df)
addTokenizedText(tweet_df)
addHashTag(tweet_df)
addTokenizedText(tweet_df)
tweet_df.head()


# # Word2Vec Trained Model with Twitter Corpus

# In[34]:

import gensim
from gensim.models import Word2Vec


# In[38]:

#emoji_model = gensim.models.Word2Vec(list(tweet_df.Tokenized))
# It might take some time to train the model. So, after it is trained, it can be saved as follows:
#emoji_model.save('emoji.embedding')
emoji_model = gensim.models.Word2Vec.load('emoji.embedding')


# In[39]:

emoji_model.most_similar(positive = ['😂'], topn = 20)


# In[233]:

len(tweet_df[tweet_df.text.str.contains(r'#bye')])


# #How many Hashtags? 155979

# In[454]:

tweet_df[tweet_df.only_HashTag.str.len() != 0].count(0)


# #Language Cleaning

# In[142]:

punctuation = string.punctuation
ex = ['“', '—', '’', ' ️', '️', '...', '”', '…', ' , , ,', '?', ' ', ' ⃣', '∞', '🆒']
for pun in [word for word in ex if word not in punctuation]:
    punctuation += pun
def isEnglish(list):
    try:
        [word.encode('ascii') for word in list if word not in punctuation]
    except Exception:
        return False
    else:
        return True


# In[170]:

def cleanNonEnglish(df):
    text_list = df['only_Text'].values
    english_Boolean = [isEnglish(sent) for sent in text_list]
    return df[english_Boolean]


# In[171]:

tweet_df = cleanNonEnglish(tweet_df)
len(tweet_df)

