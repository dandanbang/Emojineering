
# coding: utf-8

# ## 1. Load Library 

# In[115]:

## import all necessary packages
import json
import re
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt

from pandas import *
from collections import defaultdict
import string
import happyfuntokenizing
from textblob import TextBlob
import random

# scikit-learn
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
#from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier


# NLTK
from nltk import word_tokenize, wordpunct_tokenize, pos_tag
from nltk.wsd import lesk
import nltk, string
from nltk.collocations import *
from nltk.stem import WordNetLemmatizer
from xgboost import XGBClassifier

# abydos
from abydos.qgram import QGrams
from abydos.phonetic import double_metaphone, soundex
from abydos.clustering import skeleton_key, omission_key


# ## 2. Load Data

# In[169]:

with open('./data/tweets_training_clean_preprocessing_v2.json','r') as f:
    tweet_df = DataFrame(json.load(f))
with open('./data/tweets_test_clean_preprocessingv2.json','r') as f:
    test_df = DataFrame(json.load(f))


# In[187]:

tweet_random = tweet_df.sample(n=338134 ,random_state=666,axis=0)
tweet_random = tweet_random[['category_numeric', 'only_emoji', 'only_text_splithashtag', 'retweet', 'split_hashtag', 'text']]
test_df = test_df[['category_numeric', 'only_emoji', 'only_text_splithashtag', 'retweet', 'split_hashtag', 'text']]


# In[188]:

tweet_random = tweet_random[tweet_random["category_numeric"] != 6] 
test_df = test_df[test_df["category_numeric"] != 6] 


# ## 3. Initializations

# In[177]:

# set testing to True to keep a consistent seed for functions that take a seed
TESTING = True
def seed():
    if TESTING:
        return 256
    else:
        return None

# WordNet Lemmatizer
wnl = WordNetLemmatizer()


# ## 4. Functions

# ### 4.1 Function to label emoji

# In[176]:

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

def is_emoji(text):
    if highpoints.search(text):
        return 1
    else:
        return 0


# ### 4.2 Function to split training and testing data

# In[175]:

#splitting the training data into 
def create_training_sets (training_data):
    # Create the features sets.  Call the function that was passed in.
    # For names data, key is the name, and value is the gender

    # Divided training and testing in thirds.  Could divide in other proportions instead.
    third = int(float(len(training_data)) / 3.0)    
    train_set, test_set = training_data[0:third*2], training_data[third*2:]
    return train_set, test_set


# ### 4.3 Function to transform pandas for classification

# In[174]:

# Useful Transformer from http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html
# This pulls a single column from a supplied pandas dataframe for classification.
class ColumnExtractor(TransformerMixin):
    def __init__(self, columns=[]):
        self.columns = columns

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def transform(self, X, **transform_params):
        return X[self.columns]

    def fit(self, X, y=None, **fit_params):
        return self


# ### 4.4 Feature Building Function

# In[173]:

def otherFeature(sentence, retweet, hashtag):
    blob = TextBlob(sentence)
    sentimentSum = 0
    tweet = 0
    for sentence in blob.sentences:
        sentimentSum += sentence.sentiment.polarity
    if retweet == None:
        tweet =  0
    else:
        tweet = 1
    count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
    a_punct = count(sentence, string.punctuation)
    dict_ = {'sentiment' : sentimentSum,
    'textLength' : len(sentence),
    'isRetweet': tweet,
    'numPunct': a_punct,
    'hashTag': len(hashtag)}
    return dict_

def sentimentScore(sentence):
    blob = TextBlob(sentence)
    sentimentSum = 0
    for sentence in blob.sentences:
        sentimentSum += sentence.sentiment.polarity
    return sentimentSum

def textLength(text):
    return len(text)

def isRetweet(text):
    if text == None:
        return 0
    else:
        return 1

def numPunct(text):
    count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
    a_punct = count(text, string.punctuation)
    return a_punct

def leskize(word_pos_str):
    """Returns the most probable WordNet lemmas for each term in a string by applying the lesk aglorithm
    to each term, given its tagged POS and the string context
    
    Arguments:
    word_pos_str -- str consisting of "word POS word POS word POS ..."
    
    Returns:
    list
    """
    wn_pos = {'VERB': 'v', 'NOUN': 'n', 'ADV': 'r', 'ADJ': 'a'}
    
    ss_list = []

    word_pos_pairs = word_pos_str.split()
    words = word_pos_pairs[::2]
    for i in range(0, len(word_pos_pairs), 2):
        if word_pos_pairs[i+1] in wn_pos:
            ss = lesk(words, word_pos_pairs[i], wn_pos[word_pos_pairs[i+1]])
            if not ss:
                ss = lesk(words, word_pos_pairs[i])
            if ss:
                ss_list.append(ss)

    return ss_list

def build_features(df):
    """Do basic processing of the input text and generate features based on it in different columns

    Arguments:
    df -- DataFrame (the pandas dataframe with Category & Text columns already defined)
    
    Returns:
    None
    """
#     # Part of Speech in the form "word POS word POS word POS ..."
#     df['pos'] = df.only_text_splithashtag.apply(lambda sent: ' '.join([' '.join([wnl.lemmatize(word), tag]) for word, tag
#                                                               in pos_tag(wordpunct_tokenize(sent),
#                                                                          tagset='universal') if
#                                                               tag[0] not in string.punctuation]))

#     # A list of most probable lemma synsets (not used directly, but useful)
#     df['synsets'] = df.pos.apply(leskize)

#     # The definitions of each word, concatenated
#     df['definition'] = df.synsets.apply(lambda sss: ' '.join([ss.definition() for ss in sss]))

    # q-grams (generated from Chris' abydos package, which performed a little better than TfidfVectorizer)
    # These are also known as k-grams, shingles, k-mers, and (character-wise) n-grams
    df['other_features'] = df.apply(lambda x: otherFeature(x['only_text_splithashtag'], x['retweet'], x['split_hashtag']), axis=1)
    
    
    df['qgrams'] = df.only_text_splithashtag.apply(lambda s: dict(QGrams(s, 4, start_stop='') + QGrams(s, 5, start_stop='')))


# ## 5. Building the Pipeline Model

# In[179]:

tok = happyfuntokenizing.TweetTokenizer(preserve_case=False)
# Logistic classifier, using WordNet-lemmatized unigrams & bigrams as features
log_lem12_pipeline = Pipeline([
            ('tfidf_lemmatized', Pipeline([
                    ('extract', ColumnExtractor('tokenized')),
                    ('vectorize', CountVectorizer(ngram_range=(1, 2),
                                                  lowercase=True, tokenizer=tok.tokenize)),
            ])),
            ('classifier', LogisticRegression(random_state=seed()))])

# lsvc_lem12_pipeline = Pipeline([
#             ('tfidf_lemmatized', Pipeline([
#                     ('extract', ColumnExtractor('tokenized')),
#                     ('vectorize', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, norm='l2',
#                                                   lowercase=True, tokenizer=tok.tokenize)),
#                     # LSA is cool, in theory, but didn't work here, except with huge values & overfitting
#                     #('lsa', TruncatedSVD(n_components=2100, algorithm='arpack', random_state=seed())),
#             ])),
#             ('classifier', LinesarSVC(loss='hinge', C=1, random_state=seed()))])

# SVM with a linear kernel, using unigrams & bigrams of sentences passed through soundex and through
# double metaphone as features

lsvc_sdx_pipeline = Pipeline([
            ('features', FeatureUnion([
                ('definition', Pipeline([
                    ('extract', ColumnExtractor('only_text_splithashtag')),
                    ('vectorize', TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, norm='l2',
                                                  lowercase=False, stop_words="english")),
                ])),
                ('definition', Pipeline([
                    ('extract', ColumnExtractor('only_text_splithashtag')),
                    ('vectorize', CountVectorizer(ngram_range=(1, 2),lowercase=True)),
                ])),
                ('definition', Pipeline([
                        ('extract', ColumnExtractor('qgrams')),
                        ('vectorize', DictVectorizer()),
                ])),
                ('definition', Pipeline([
                        ('extract', ColumnExtractor('other_features')),
                         ('vectorize', DictVectorizer()),
                ])),
                    ])),
                ('classifier', LogisticRegression(C=0.001, random_state=seed()))
#                ('classifier', XGBClassifier(max_depth=8,n_estimators=128,))
        ])


# ##6. Building the Feature

# In[189]:

build_features(tweet_random)
build_features(test_df)


# ##7. Splitting the training and test set

# In[134]:

train_set, test_set = create_training_sets(tweet_random)


# ##8. Prediction

# In[ ]:

start_time = time.time()

TESTING = False
model = lsvc_sdx_pipeline.fit(train_set, train_set['category_numeric'])
prediction = model.predict(test_df)
score = accuracy_score(test_df['category_numeric'], prediction)
print(score)

sec = time.time() - start_time
hours, remainder = divmod(sec, 3600)
minutes, seconds = divmod(remainder, 60)
print('time to completion: %02d:%02d:%02d' % (hours, minutes, seconds))


# In[183]:

from nltk import FreqDist
f = FreqDist(prediction) 
f.most_common(10)


# In[45]:

from sklearn.metrics import accuracy_score
start_time = time.time()

TESTING = False
model = lsvc_sdx_pipeline.fit(train_set, train_set['category_numeric'])
prediction = model.predict(test_set)
score = accuracy_score(test_set['category_numeric'], prediction)
print(score)

sec = time.time() - start_time
hours, remainder = divmod(sec, 3600)
minutes, seconds = divmod(remainder, 60)
print('time to completion: %02d:%02d:%02d' % (hours, minutes, seconds))


# ##9. Saving The Model

# In[184]:

import pickle
with open('logisticRegression.pickle', 'wb') as fin:
    pickle.dump(model, fin)

