#!/usr/bin/env python
# coding: utf-8

# The following codes are intended only as an exploration from the perspective of NLP.

# In[1]:


import pandas as pd 
import numpy as np 
from IPython.display import display

import matplotlib.pyplot as plt 
import re
import string

import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
nltk.download('stopwords')
nltk.download('vader_lexicon')


from collections import Counter

from matplotlib import pyplot as plt
from matplotlib import ticker
import seaborn as sns
import plotly.express as px

sns.set(style="darkgrid")


# In[2]:


df=pd.read_csv('Bitcoin_tweets.csv')


# In[3]:


df['Date'] = pd.to_datetime(df['user_created'],format='%Y-%m-%d %H:%M:%S', errors='coerce')


# In[4]:


df['Dates'] = pd.to_datetime(df['Date']).dt.date
df['Time'] = pd.to_datetime(df['Date']).dt.time


# In[5]:


df.shape


# In[6]:


df.head(2)


# In[7]:


needed_columns=['user_name','Dates','text']
df=df[needed_columns]
df.head()


# In[10]:


df.user_name=df.user_name.astype('category')

df.Dates=pd.to_datetime(df.Dates).dt.date


# In[11]:


df.head(3)


# In the next lines, we are going to work with the tweets (text) it self 

# In[12]:


tweets=df.text
tweets


# In[13]:


#removing URLs 
remove_url=lambda x:re.sub(r'http\S+','',str(x))
tweets_lr=tweets.apply(remove_url)
tweets_lr


# In[14]:


#convertring tweets to lowercase 
to_lower=lambda x: x.lower()
tweets_lr_lc=tweets_lr.apply(to_lower)
tweets_lr_lc


# In[15]:


#removing punctations 
remove_puncs= lambda x:x.translate(str.maketrans('','',string.punctuation))
tweets_lr_lc_np=tweets_lr_lc.apply(remove_puncs)
tweets_lr_lc_np


# In[16]:


stop_words=set(stopwords.words('english')) #nltk package
remove_words=lambda x: ' '.join([word for word in x.split() if word not in stop_words]) #.join is from package string
tweets_lr_lc_np_ns=r=tweets_lr_lc_np.apply(remove_words)
tweets_lr_lc_np_ns


# In[17]:


#common words in the tweets 
words_list=[word for line in tweets_lr_lc_np_ns for word in line.split()]
words_list[:5]


# In[18]:


word_counts=Counter(words_list).most_common(50)
word_df=pd.DataFrame(word_counts)
word_df.columns=['word','frq']
display(word_df.head(5))
# px=import plotly.express
px.bar(word_df,x='word',y='frq',title='Most common words')


# It is interesting to see that the most frequently occurring words are from the financial area. Other cryptocurrencies such as Etherum are also mentioned, as described in the concept. It is also curious that elon musk is often mentioned.His name was often mentioned in the media in connection with market manipulation in the crypto market. According to some sources, Elon Musk (an individual) has more influence on the crypto market than China, for example.The term airdrop has its own meaning in the crypto scene. It means that some coins or tokens are distributed for free.Binance is a crypto trading platform.The word Defi is a crypto abbreviation. It stands for decentralised finance system.Dodge or dodgecoin refers to a cryptocurrency. This currency became famous due to its skyrocketing price. This cryptocurrency originated on the social media platform reddit.
# 
# 
# As expected, the tweets are not only about Bitcoin. Rather, there are tweets about all kinds of cryptocurrencies. For more recent topics in the crypto scene such as the NFTs, for example, are also included in the tweets.
# 

# In[20]:


#adding clean text to the main dataframe 
display(df.head(5))
df.text=tweets_lr_lc_np_ns
display(df.head(5))


# In[21]:


#additional Cleaning 
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
df['text'] = df['text'].apply(lambda x: clean_text(x))
display(df)


# In[22]:


# removing emoticons,  I used some hints that I found on stackoverflow
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


# In[23]:


df['text']=df['text'].apply(lambda x: remove_emoji(x))
display(df)


# Sentiment Analysis 

# The code below could take some time (depends on the capacity/ RAM)

# In[25]:


sid=SentimentIntensityAnalyzer()
ps=lambda x:sid.polarity_scores(x)
sentiment_scores=df.text.apply(ps)
#sentiment_scores


# In[26]:


sentiment_scores


# In[27]:


sentiment_df=pd.DataFrame(data=list(sentiment_scores))
display(sentiment_df)


# In[28]:


#Labeling the scores based on the compound polarity value
labelize=lambda x:'neutral' if x==0 else('positive' if x>0 else 'negative')
sentiment_df['label']=sentiment_df.compound.apply(labelize)
display(sentiment_df.head(10))


# In[29]:


#joining the sentiment data to the tweets data 
display(df.head(5))
data=df.join(sentiment_df.label)
display(data.head(5))


# In[30]:


#sentiment score coutns plot 
counts_df=data.label.value_counts().reset_index()
display(counts_df)


# In[31]:


plt.figure(figsize=(8,5)) 
sns.barplot(x='index',y='label',data=counts_df)


# In[33]:


#plotting with the regard to date 
data_agg=data[['user_name','Dates','label']]
display(data_agg.head(5))


# In[32]:


data.head(2)


# In[34]:


data_agg=data_agg.groupby(['Dates','label'])
display(data_agg.head(5))


# In[35]:


data_agg=data_agg.count()
display(data_agg.head(5))


# In[36]:


data_agg=data_agg.reset_index()
display(data_agg.head(5))


# In[37]:


#the 'user_name' is the count of users, so need to change the column name
data_agg.columns=['date','label','counts']
display(data_agg.head())


# In[38]:


px.line(data_agg,x='date',y='counts',color='label',
       title=' Tweet Sentimental Analysis')


# In[39]:


df['text']=df['text'].apply(lambda x: remove_emoji(x))
display(df)


# In[40]:


from wordcloud import WordCloud


# In[41]:


cut_text = " ".join(df.text)
max_words=100
word_cloud = WordCloud(
                    background_color='white',
                    stopwords=set(stop_words),
                    max_words=max_words,
                    max_font_size=30,
                    scale=5,
    colormap='magma',
                    random_state=1).generate(cut_text)
fig = plt.figure(1, figsize=(50,50))
plt.axis('off')
plt.title('Word Cloud for Top '+str(max_words)+' words with # Bitcoin or BTC in Twitter\n', fontsize=100,color='blue')
fig.subplots_adjust(top=2.3)
plt.imshow(word_cloud)
plt.show()

