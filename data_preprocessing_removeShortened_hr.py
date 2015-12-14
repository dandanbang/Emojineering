
# coding: utf-8

# In[18]:

import nltk
from data_cleaning_hr import loader


# In[2]:

tweets_df = loader("./data/tweets_training_clean_preprocessing.json")


# In[11]:

tweets_df.head()
len(tweets_df)


# In[15]:

# Filter tweets by those of more than 50 characters 
tweets_df_filtered = tweets_df[tweets_df.only_text_splithashtag.apply(lambda x: len(x)) > 50]


# In[14]:

tweets_df_filtered.head()


# In[17]:

sum(tweets_df_filtered.only_emoji.apply(lambda x: len(x) > 0))


# In[34]:

temp = tweets_df_filtered.only_emoji.values.flatten().tolist()
import itertools
chain = itertools.chain(*temp)
fdist = nltk.FreqDist(chain)
fdist.most_common()


# In[ ]:

tweets_df_filtered.to_json('./data/tweets_training_clean_preprocessingv2.json', force_ascii=False)

