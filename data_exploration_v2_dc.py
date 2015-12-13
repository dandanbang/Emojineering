
# coding: utf-8

# #Load Library 

# In[1]:

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

# In[74]:

with open('./data/tweets_training_clean.json','r') as f:
    tweet_df = DataFrame(json.load(f))


# #Clean Data (Handle, URL, Emoticon Conversion)

# In[3]:

tc.clean(tweet_df)
tweet_df.head()


# #Emoji Finder

# In[4]:

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


# In[5]:

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

# In[6]:

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

# In[20]:

ex = 'hdl not funny! Sad 😞😭 bye money 💸. Now you have to drive me places Donna'
def textExtract(sent):
    return ''.join([word for word in sent if is_emoji(word) == 0])

def emojiExtract(sent):
    return [word for word in tok.tokenize(sent) if is_emoji(word) == 1]

textExtract(ex)


# In[7]:

def emojiExtract(sent):
    return [word for word in tok.tokenize(sent) if is_emoji(word) == 1]

def textExtract(sent):
    return ''.join([word for word in sent if is_emoji(word) == 0])

def addTokenizedText(df):
    def textTokenized(sent):
        return [word for word in tok.tokenize(sent)]
    df['Tokenized'] = [textTokenized(word) for word in df.text]

def addEmojiCol(df):
    df['Emoji'] = [emojiExtract(word) for word in df.text]

def addText(df):
    df['only_Text'] = [textExtract(word) for word in df.text]

def addHashTag(df):
    df['only_HashTag'] = [re.findall(r"(#\w+)", word) for word in df.text]


# In[71]:

def splitTextEmoji(df):
    def emojiExtract(sent):
        return [word for word in tok.tokenize(sent) if is_emoji(word) == 1]

    def textExtract(sent):
        return ''.join([word for word in sent if is_emoji(word) == 0])

    def addEmoji(df):
        df['only_Emoji'] = [emojiExtract(word) for word in df.text]

    def addText(df):
        df['only_Text'] = [textExtract(word) for word in df.text]
    
    addText(df)
    addEmoji(df)
    return


# In[75]:

splitTextEmoji(tweet_df)


# In[76]:

tweet_df.head()


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

# In[29]:

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


# In[83]:

def cleanNonEnglish(df):
    temp = df.copy()
    text_list = df['only_text'].values
    english_Boolean = [isEnglish(sent) for sent in text_list]
    df = temp[english_Boolean]
    return temp[english_Boolean]


# In[84]:

tweet_df_clean = cleanNonEnglish(tweet_df.iloc[0:10000])
len(tweet_df_clean)


# In[82]:

cleanNonEnglish(tweet_df.iloc[0:10000])
len(tweet_df)


# In[46]:

get_ipython().magic('pinfo2 pandas.DataFrame.drop')


# In[93]:

def apply_hashtag_split(_hashtag):
    # for each hashtag in list of hashtags, split on # and take second item
    item = [item.split('#')[1] for item in _hashtag if len(item) > 0]
    
    # regex pattern
    pattern = r'''([A-Z]{2,})      | (1) split on two caps or more, and only keep caps
                  ([A-Z]{1}[a-z]*)   (2) split on exactly one cap or more, and keep 
                                         trailing letters '''
    
    # loop through each item in list of hashtags
    final_hashtags = []
    for word in item:
        
        # if len of word is 0, then there is no hashtag
        if len(word) == 0:
            print("empty_hashtag")
            final_hashtags.append("empty_hashtag")
        
        # use (1)) regex: funciton lowercase, as "Treatlowercase" can be treated as lowercase
        elif word[0].isupper() and word[1:].islower():
            print("lower_forced: " + word + " : ", end="")
            print(infer_spaces(word.lower()))
            final_hashtags.append(infer_spaces(word.lower()))
        
        # use (1) regex: funciton lowercase
        elif word.islower():
            print("lower       :: " + word + " : ", end="")
            print(infer_spaces(word))
            final_hashtags.append(infer_spaces(word))
        
        # use (2) regex: customized uppercase
        else:
            print("upper       : " + word + " : ", end="")
            print(list(filter(None, re.split(pattern, word))))
            final_hashtags.append(list(filter(None, re.split(pattern, word))))
    return final_hashtags


# In[87]:

addHashTag(tweet_df)
tweet_df


# In[90]:

tweet_df['only_HashTag'].apply(apply_hashtag_split())


# In[95]:

def infer_spaces(s):
    """Uses dynamic programming to infer the location of spaces in a string
    without spaces."""
    
    # Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).

    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
        return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

    # Build the cost array.
    cost = [0]
    for i in range(1,len(s)+1):
        c,k = best_match(i)
        cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i>0:
        c,k = best_match(i)
        assert c == cost[i]
        out.append(s[i-k:i])
        i -= k

    return list(reversed(out))


# In[96]:

apply_hashtag_split(['#whitegirlprobs'])

