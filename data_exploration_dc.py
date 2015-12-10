
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

# In[20]:

with open('./data/tweets_1M.json','r') as f:
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


# # Subset Dataframe with Emojis Only (New Dataframe emoij_df)

# In[69]:

emoji_list = []
for index, value in enumerate(tweet_df.text):
    if highpoints.search(value):
        emoji_list.append((index, value))
emoji_index = [x[0] for x in emoji_list]
emoji_df = tweet_df.ix[emoji_index]


# # Count Emoji Per Text (1.12 Average Emoji/Text, 53.2 Average Text Length)

# In[23]:

# List of functions for emoji search
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


# In[8]:

emoji_df["Emoji Count"] = emoji_df["text"].apply(count_emojis)
emoji_df["Text Length"] = emoji_df["text"].apply(lambda x: len(x))
emoji_df = emoji_df[['id', 'text', 'timeStamp', 'user_id', 'Emoji Count', 'Text Length']]
emoji_df.describe()


# # Emoji Distribution

# In[ ]:

reset_df = emoji_df.reset_index(drop=True)
emoji_array = [reset_df.loc[[index]].text.apply(just_emojis) for index in range(len(reset_df))]
full_list = []
for item in emoji_array:
    for emoji in item:
        for sinlge in emoji:
            full_list.append(sinlge)


# In[9]:

full_dict = defaultdict(int)
for item in full_list:
    full_dict[item] += 1


# ### 30469 Unique Emoji 

# In[10]:

len(unique(full_dict.keys()))


# In[11]:

for item in sorted(full_dict.items(), key=lambda x:x[1], reverse=True)[:50]:
    print(item[0], (float(item[1])/len(full_list))*100)


# # Emoji Face Distribution

# In[12]:

face_array = [reset_df.loc[[index]].text.apply(just_face) for index in range(len(reset_df))]

face_list = []
for item in face_array:
    for emoji in item:
        for sinlge in emoji:
            face_list.append(sinlge)

print("number of tweets with emojis {0}".format(len(full_list)))
print("number of tweets with faces  {0}".format(len(face_array)))
print("percentage of tweets with emojis with faces {0}%".format(round((float(len(face_array))/len(full_list)*100),1)))


# In[13]:

face_dict = defaultdict(int)
for item in face_list:
    face_dict[item] += 1


# In[14]:

for item in sorted(face_dict.items(), key=lambda x:x[1], reverse=True)[:50]:
    print(item[0], (float(item[1])/len(full_list))*100)


# # Exploring Emoji vs no Emoji (20% Text has Emojis, 15% has face)

# ###Functions to check whether it's emoji or face

# In[24]:

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


# # Unigrams Frequency

# In[21]:

tok = Tokenizer(preserve_case=False)
text_list = list(tweet_df.text)
tokenized = [list(tok.tokenize(item)) for item in text_list]
print(tokenized[:10])


# In[31]:

stop_words = nltk.corpus.stopwords.words('english') + ["http", 'hdl', 'url'] 
punctuation_words = list(set(string.punctuation)) + [":", ":/"]

def real_unigrams(text):
    real_unigrams = [word for sent in text for word in sent if word.lower() not in stop_words and word not in punctuation_words and is_emoji(word) == 1] 
    real_unigrams_freq = nltk.FreqDist(real_unigrams)
    top_unigrams = real_unigrams_freq.most_common(100)
    return top_unigrams


# In[32]:

real_unigrams(tokenized)


# # Bigram Frequency

# In[26]:

bigram_tokenized = [list(tok.tokenize(item)) for item in text_list]


# In[35]:

def bigrams(text):
    all_bigrams = [nltk.bigrams(sent) for sent in text]
    all_bigrams = [pair for _list in all_bigrams for pair in list(_list)                    if pair[0] not in stop_words and pair[1] not in stop_words                   and pair[0] not in punctuation_words and pair[1] not in punctuation_words
                  and is_emoji(pair[0]) == 1 and is_emoji(pair[1]) == 1 
                  and pair[0] != pair[1]]
    
    bi_freq = nltk.FreqDist(all_bigrams)
    top_bigrams = bi_freq.most_common(100)
    return top_bigrams


# In[36]:

bigrams(bigram_tokenized)


# ## Collocations

# In[38]:

text_joined = " ".join(tweet_df.text)


# In[41]:

bigram_measures = nltk.collocations.BigramAssocMeasures()

def collocation_bigrams(text):    
    finder = BigramCollocationFinder.from_words(text)
    finder.apply_freq_filter(9)
    bigram_coll = (finder.nbest(bigram_measures.pmi, 500))
    return bigram_coll


# In[42]:

collocation_bigrams = collocation_bigrams(text_joined)


# In[53]:

[pair for pair in collocation_bigrams if is_emoji(pair[0]) == 1 and is_emoji(pair[1]) == 1]


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

#addEmojiCol(tweet_df)
#addText(tweet_df)
addTokenizedText(tweet_df)
tweet_df.head()


# In[449]:

ex = ['My Friday night... #collegeanatomy #kickingmybutt #study @ The Price Family Inn url']
[re.findall(r"(#\w+)", word) for word in ex]


# ## Create Only Hashtag Column and Tokenized Columns

# In[453]:

#addHashTag(tweet_df)
#addTokenizedText(tweet_df)


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


# In[333]:

from nltk.corpus import words
"fuck" in words.words()


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


# In[131]:

#with pd.option_context('display.max_rows', 999):
#    print (tweet_df_non_en['text'])
#tweet_df_non_en['text'].to_csv('non_en.csv')


# In[171]:

tweet_df = cleanNonEnglish(tweet_df)


# In[172]:

len(tweet_df)


# In[ ]:



