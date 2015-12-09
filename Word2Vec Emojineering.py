
# coding: utf-8

# ##Word2Vec Demo##
# From https://github.com/nltk/nltk/blob/develop/nltk/test/gensim.doctest
# 

# In[1]:

# to get gensim, to to https://radimrehurek.com/gensim/
# OR run this on your command line: easy_install -U gensim 

import nltk
import numpy as np
import gensim
from gensim.models import Word2Vec
from nltk.data import find
import pandas as pd


# In[2]:

# To get the model file needed, do the following one time only:
#one time only: Run download; view the UI that pops up; switch to the models tab, and download the word2vec_sample model
# nltk.download()


# In[9]:

word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
model = gensim.models.Word2Vec.load_word2vec_format(word2vec_sample, binary=False)


# In[7]:

for doc in word2vec_sample:
    words = filter(lambda x: x in model.vocab, doc.words)


# # OLD STUFF FOR REFERENCE

# We pruned the model to only include the most common words (~44k words).
# 

# In[8]:

len(model.vocab)


# Each word is represented in the space of 300 dimensions:
# 

# In[10]:

len(model['university'])


# Finding the top n words that are similar to a target word is simple. The result is the list of n words with the score.
# 

# In[11]:

model.most_similar(positive=['university'], topn = 10)


# Finding a word that is not in a list is also supported in the API.
# 

# In[12]:

model.doesnt_match('breakfast cereal dinner lunch'.split())


# Mikolov et al. (2013) figured out the following famous exampe:  word embedding captures much of syntactic and semantic regularities. For example,
# the vector 'King - Man + Woman' is close to 'Queen' and 'Germany - Berlin + Paris' is close to 'France'.

# In[13]:

model.most_similar(positive=['woman','king'], negative=['man'], topn = 1)


# In[9]:

model.most_similar(positive=["face", "person", 'triumph', 'won'], topn = 10)


# In[10]:

model.most_similar(positive=['Paris','Germany'], negative=['Berlin'], topn = 1)


# In[11]:

model.most_similar(positive=['president', 'university'], topn=30)


# You can train your own models.  Here is an example using NLTK corpora.  This will be an exercise in seeing how different corpora yield different results.

# In[12]:

from nltk.corpus import brown
brown_model = gensim.models.Word2Vec(brown.sents())

# It might take some time to train the model. So, after it is trained, it can be saved as follows:

brown_model.save('brown.embedding')
new_model = gensim.models.Word2Vec.load('brown.embedding')


# In[13]:

brown_model.most_similar('president')


# # START EMOJINEERING 

# In[49]:

def convert_scraped_txt(txt):
    with open("emoji_webscraped.txt") as f_in:
        titles = []
        descriptions = []
        annotations = []
        for line in f_in:
            line = line.strip()
            temp = line.split(", ")
            titles.append(temp[0])
            descriptions.append(temp[1])
            annotations.append(temp[2:len(temp)])
        return titles, descriptions, annotations

titles, descriptions, annotations = convert_scraped_txt("emoji_webscraped.txt")


# In[50]:

print(len(titles))
print(len(descriptions))
print(len(annotations))


# In[59]:

d = {'titles' : (titles),
     'annotations' : (annotations),
     'descriptions': (descriptions)}
df = pd.DataFrame(d)


# In[60]:

df.head()


# count vectorizer on annoations descriptions
# clustering with binary data, possibly asocaition rules
# tfidf
# feature vector
# k_means on either full vector (or on lower dimensional space)

# In[112]:

list_titles = [list(item) for item in list(df.annotations)]
index_face_person = [index for index,value in enumerate(list_titles) if 'face' in value or 'person' in value]
print(len(index_face_person))
df_face_person = df.iloc[index_face_person]
print(df_face_person.shape)
df_face_person.head()


# In[89]:

list_words = [word for item in list(df.annotations) for word in item]


# In[90]:

nltk.FreqDist(list_words).most_common(50)


# In[79]:

def list_word2vec(_list):
    single_words = [model.most_similar(positive=item, topn = 1)  for item in _list]
    return single_words


# In[80]:

list_word2vec(df.annotations)


# In[ ]:



