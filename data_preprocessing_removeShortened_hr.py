
# coding: utf-8

# In[1]:

import nltk
from data_cleaning_hr import loader
def loader(filename):
    """ Load tweets from filename. Resets the index. Returns the loaded data frame"""
    with open(filename,'r') as f:
        df = pd.DataFrame(json.load(f))
    df.reset_index(inplace=True, drop=True)
    return df


# In[ ]:

tweets_df = loader("./data/tweets_test_clean_preprocessing.json")


# In[ ]:

tweets_df.head()
len(tweets_df)


# In[ ]:

# Filter tweets by those of more than 50 characters 
tweets_df_filtered = tweets_df[tweets_df.only_text_splithashtag.apply(lambda x: len(x)) > 50]


# In[ ]:

tweets_df_filtered.head()


# In[ ]:

sum(tweets_df_filtered.only_emoji.apply(lambda x: len(x) > 0))


# In[ ]:

temp = tweets_df_filtered.only_emoji.values.flatten().tolist()
import itertools
chain = itertools.chain(*temp)
fdist = nltk.FreqDist(chain)
fdist.most_common()


# In[ ]:

tweets_df_filtered.to_json('./data/tweets_test_clean_preprocessingv2.json', force_ascii=False)

