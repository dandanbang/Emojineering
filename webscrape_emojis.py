################################################################
######### Import Packages ######################################
################################################################
import sys
import pandas as pd
import re
import urllib
from bs4 import BeautifulSoup

################################################################
################ Steal from yelp hw ############################
################################################################
def preprocess_yelp_page(content):
    ''' Remove extra spaces between HTML tags. '''
    content = ''.join([line.strip() for line in content.split('\n')])
    return content

content_final = []

url = 'http://unicode.org/emoji/charts/full-emoji-list.html#1f602'
content = urllib.urlopen(url).read()
content = preprocess_yelp_page(content) # Now *content* is a string containing the first page of search results, ready for processing with BeautifulSoup
content_final.append(url)

soup = BeautifulSoup(content, 'html.parser')

################################################################
####### Look for :    
####### 	- unicode titles of each emoji through class code
#######		- descriptions and annotations through class name
################################################################

titles = soup.findAll("td", { "class" : "code" })
titles = [(item.string).encode('ascii','ignore') for item in titles]

# print(titles)
# print(len(titles))

all_descriptions = soup.findAll("td", {"class": "name"})
# print((descriptions))
# print(len(descriptions))

descriptions = [(all_descriptions[i].string).encode('ascii','ignore') for i in xrange(0,len(all_descriptions),2)]

annotations = [all_descriptions[i].findAll(text=True) for i in xrange(1,len(all_descriptions),2)]



full = []
for i in annotations:
	temp = []
	for j in i:
		if re.findall(r'(?<!,)\b\w+', j.encode('ascii','ignore')):
			temp.append(j.encode('ascii','ignore'))
	full.append(temp)

zipped = zip(titles, descriptions, full)
print(zipped[0])


# def make_txt(zipped):
# 	for item in range(len(zipped)):
# 		print("{0}, {1}, {2}\n".format(titles[item],descriptions[item],", ".join(full[2]))) 
# # make_txt(zipped)

def txt_file(zipped):
	with open('emoji_webscraped.txt','w') as f_in:
		for item in range(len(zipped)):
			f_in.write("{0}, {1}, {2}\n".format(titles[item],descriptions[item],", ".join(full[item])))
		# for line in raw_text:
		# 	f_in.write("{0}\n".format(line))
txt_file(zipped)

################################################################
###### Convert to pandas df to submit as csv ###################
################################################################
# df_zipped = pd.DataFrame(zipped)
# df_zipped.to_csv("emoji_webscraped.csv")

