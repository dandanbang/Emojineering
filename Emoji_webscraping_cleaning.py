
# coding: utf-8

# In[110]:

import gensim
from gensim.models import Word2Vec

from data_cleaning_hr import loader
import src.happyfuntokenizing

import pandas as pd
import re

from IPython.display import display


# ## Expand unicodes

# In[116]:

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


# In[118]:

# Expand bytecodes
pattern = r"U\+(\d?)([A-F0-9]{4})(?: U\+(\d?)([A-F0-9]{4}))?"
p = re.compile(pattern)
temp = emojis_df.byteCode.str.extract(pattern)
temp[0].replace("1", "\\U0001", inplace=True)
temp[0].replace("", "\\U0000", inplace=True)
temp[2].replace("1", "\\U0001", inplace=True)
temp[2].replace("", "\\U0000", inplace=True)
temp.fillna("",inplace=True)
emojis_df.byteCode = temp.apply(lambda x: "".join(x), axis=1)


# In[119]:

emojis_df


# In[107]:

# Verifying
for byteCode in emojis_df.byteCode:
    print(bytes("{0}".format(byteCode), 'ascii').decode('unicode-escape'))


# In[120]:

emojis_df.to_csv("./data/emoji_webscraped_expanded.csv")

