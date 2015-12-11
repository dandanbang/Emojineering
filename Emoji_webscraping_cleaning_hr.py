
# coding: utf-8

# In[7]:

import gensim
from gensim.models import Word2Vec

from data_cleaning_hr import loader
import src.happyfuntokenizing

import pandas as pd
import re

from IPython.display import display


# ## Expand unicodes

# In[54]:

# Load list of emojis from CSV that Carlo will finish
# Copy pasted from "Clustering with Annotations.ipynb" -> to replace with module import and function calls
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

# Create dataframe from three arrays
emojis = {'byteCode' : (titles),
     'annotations' : (annotations),
     'descriptions': (descriptions)}
emojis_df = pd.DataFrame(emojis)


# In[55]:

# Expand bytecodes
pattern = r"U\+(\d?)([A-F0-9]{4})(?: U\+(\d?)([A-F0-9]{4}))?"
p = re.compile(pattern)
temp = emojis_df.byteCode.str.extract(pattern)
temp[0].replace("1", r"\U0001", inplace=True)
temp[0].replace("", r"\U0000", inplace=True)
temp[2].replace("1", r"\U0001", inplace=True)
temp[2].replace("", r"\U0000", inplace=True)
temp.fillna("",inplace=True)
display(temp)
emojis_df["byteCode1"] = temp.apply(lambda x: "".join(x[:2]), axis=1)
emojis_df["byteCode2"] = temp.apply(lambda x: "".join(x[2:]), axis=1)


# In[56]:

emojis_df


# In[60]:

# Verifying
for byteCode1, byteCode2 in zip(list(emojis_df.byteCode1), list(emojis_df.byteCode2)):
    print(bytes("{}{}".format(byteCode1, byteCode2), 'ascii').decode('unicode-escape'))


# In[61]:

emojis_df.to_json("./data/emoji_webscraped_expanded.json")


# In[ ]:



