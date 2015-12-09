
# coding: utf-8

# #Data cleaning

# Data cleaning steps:
# - replace @ handles with: hdl (chose a word with no punctuation so no extra step is needed to process, hdl = handle)
# - replace urls by: url
# - replace emoticons with corresponding emojis
# 
# 
# For all of these, first check what is the prevalence and if it is worth the effort. 
# 
# Ideas:
# - split hashtags into words
#     - Some people seperate words in hashtags with capitals. Make it much easier to seperate. 
# - replace contractions
# - spellchecking
# - what does quoted messages mean? Someone quoting somebody else? Should we consider this text or remove it? 
# - there are some smileys that are still in punctuation and not in unicode, eg :-P, :-)
# - Tweets not in english
# - Character ngram will probably be more efficient due to the really low quality of speach
# - Retweets have two formats:
#     - Either finish with RT &lt;content of retweet>
#     - OR "&lt;content of retweet>" &lt;content of tweet>
#     - Can also have mutliple embedings with “ for second level. E.g. : 
# "@letwerkaaaaa: “@Palmira_0: HAHA WHEN I WAS LITTLE I WAS FAT AS FUCK:joy:” Same I was so fat that they thought my vagina was a dick" LMFAO
#     - For now will leave them here. We might have to consider while training if it is retweeted content or original content. Actually impacts the single response rate, ie when people just add an emoji to a tweet (as in the emoji is the sole content) 

# In[1]:

import json
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import pandas as pd
pd.options.display.max_colwidth = 140
import nltk
import re


# In[ ]:




# ### Replace @ handles with hdl

# No "hdl" words for confusion in the txt. Good replacement name. 
# 
# Best to replace handles with "\ hdl\ ", so for tokenization it will be easier to identify as a word. 

# In[16]:

def cleanHandle(df):
    """ Replace in-place handles with hdl keyword
    
    Returns None
    """
    pattern = r"@[a-zA-Z0-9_]{1,15}" #from http://kagan.mactane.org/blog/2009/09/22/what-characters-are-allowed-in-twitter-usernames/
    print("{} handles replaced".format(np.sum(df.text.str.contains(pattern).values)))
    df.text = df.text.str.replace(pattern, " hdl ")
    return


# ### Replace URLs with url

# URLs are quiet well formed and are generally at the end of tweets. No risk of engulfing in the cleaning some more text after the url.
# 
# keyword url is used only 4 times in dataset, no risk of confusion

# In[1]:

def cleanURL(df):
    """ Replace in-place URLs with url keyword
    
    Returns None
    """
    pattern = r"http://\S+"
    print("{} urls replaced".format(np.sum(clean_tweets_df.text.str.contains(pattern).values)))
    df.text = df.text.str.replace(pattern, " url ")
    return


# ### Convert emoticons to emojis

# In[2]:

# Based on:
# https://slack.zendesk.com/hc/en-us/articles/202931348-Emoji-and-emoticons
# http://unicodey.com/emoji-data/table.htm
# http://www.unicode.org/emoji/charts/emoji-list.html

def convertEmoticon(df):
    """ Replace in-place common emoticons to emojis.
    
    Returns None
    """
    emoticon2emoji = {
        r"<3": "\u2764",
        r"</3": "\U0001F494",
        r"8\)": "\U0001F60E",
        r"D:": "\U0001F627",
        r":'\(": "\U0001F622",
        r":o\)": "\U0001F435",
        r":-*\*": "\U0001F48B",
        r"=-*\)": "\U0001F600",
        r":-*D": "\U0001F600",
        r";-*\)": "\U0001F609",
        r":-*>": "\U0001F606",
        r":-*\|": "\U0001F610",
        r":-*[Oo]": "\U0001F62E",
        r">:-*\(": "\U0001F620",
        r":-*\)|\(:": "\U0001F603",
        r":-*\(|\):": "\U0001F61E",
        r":-*[/\\]": "\U0001F615",
        r":-*[PpbB]": "\U0001F61B",
        r";-*[PpbB]": "\U0001F61C"
    }
    
#     for emoticon in emoticon2emoji:
#         print("{:10} -> {:>5}".format(emoticon, emoticon2emoji[emoticon]))
    
    total = 0
    for emoticon in emoticon2emoji:
        nreplacements = np.sum(df.text.str.contains(emoticon).values)
        total += nreplacements
        print("{:10} -> {:>5} replaced {:6} times".format(emoticon, emoticon2emoji[emoticon], nreplacements))
        df.text = df.text.str.replace(emoticon, emoticon2emoji[emoticon])
    print("{:3} replaced {} times".format("ALL", total))
    return



# ### Explore retweets

# In[10]:

pattern = r"""(?:\W|^)RT(?:[ \":“]|$)| # Retweets with RT keyword
            [\"“]\s*hdl                # Retweets with quotes"""
temp = clean_tweets_df.text.str.contains(pattern, flags=re.X)
print("There are {} retweets".format(np.sum(temp.values)))
clean_tweets_df[temp].sample(10)


# ###Twitter hashtags

# In[ ]:




# ##Supportive functions

# In[4]:

import re
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


# In[ ]:

# Functions to check whether there's an emoji in the text, return 1 if true, 0 if false
def is_emoji(text):
    if highpoints.search(text):
        return 1
    else:
        return 0


# #Functions to extract only emoji or only text from input

# In[1]:

def emojiExtract(sent):
    return [word for word in tok.tokenize(sent) if is_emoji(word) == 1]

def textExtract(sent):
    return [word for word in tok.tokenize(sent) if is_emoji(word) == 0]


# ##Functions to Create Two New Columns [Text Only] & [Emoji Only]

# In[2]:

def textEmojiOnly(df):
    df['Emoji'] = [emojiExtract(word) for word in df.text]
    df['only_Text'] = [textExtract(word) for word in df.text]


# In[ ]:



