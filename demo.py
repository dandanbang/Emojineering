
# coding: utf-8

# # Word2Vec with emojis demo
# The goal of this demo is to experience the power of word2vec which works well also with emojis. 

# In[21]:

from data_cleaning import loader
from gensim.models import Word2Vec


# ### Load word2vec model
# For training see word2vec_hierarchical_clustering. The model is trained on 800k tweets cleaned, with a window size of 10 (for more semantic similarities), a feature vector of size 200 and otherwise standard parameters. 

# In[22]:

emoji_model = Word2Vec.load('emoji.embedding')


# ### Load emojis

# In[23]:

def convertEmojis(df):
    """Converts emojis df to printable format """
    emojis = list(map(lambda x: bytes("{}{}".format(*x), 'ascii').decode('unicode-escape'), zip(list(df.byteCode1), list(df.byteCode2)))) 
    return emojis


# In[24]:

def subset_present(df, model):
    # Select only emojis that are in our model (ie in the corpus)
    return df[df["emojis"].map(lambda x: x in model.vocab.keys())]


# In[25]:

emojis_df = loader("./data/emoji_webscraped_expanded.json")
emojis_df["emojis"] = convertEmojis(emojis_df) 
emojis_df_sub = subset_present(emojis_df, emoji_model) # Subset to emojis present in our data (at least 100 times)


# ## Pick one emoji from this list

# In[26]:

for emoji in emojis_df_sub.emojis:
    print(emoji, end="")


# ## Drag and drop emojis here

# In[47]:

emoji_model.most_similar(positive = ['👑', "girl"], negative= ["guy"], topn=1)


# ## Examples

# In[30]:

emoji_model.most_similar(positive = ['👑', "girl"], negative= ["guy"], topn=1)


# In[29]:

emoji_model.most_similar(positive = ['🐪'], negative= [], topn=1)


# In[31]:

emoji_model.most_similar(positive = ['🍺'], negative= [], topn=1)


# In[32]:

emoji_model.most_similar(positive = ['🍻'], negative= [], topn=1)


# In[37]:

emoji_model.most_similar(positive = ['🍴'], negative= ["🍺"], topn=10)


# In[39]:

emoji_model.most_similar(positive = ['sport', "🏆"], negative= [], topn=10)


# In[45]:

emoji_model.most_similar(positive = ["snow"], negative= [], topn=10)


# In[46]:

emoji_model.most_similar(positive = ["💲", "💍"], negative= [], topn=10)


# In[ ]:



