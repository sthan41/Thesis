#!/usr/bin/env python
# coding: utf-8

# In[64]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure 
import plotly.express as px 
from wordcloud import WordCloud, ImageColorGenerator 
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist 
from nltk.corpus import stopwords 


# In[47]:


df=pd.read_csv('Bitcoin_tweets.csv')


# In[49]:


df.head(4)


# In[50]:


#feature set and nr. of obs.
df.shape


# In[51]:


df['user_friends'] = pd.to_numeric(df['user_friends'],errors='coerce')
df['user_favourites'] = pd.to_numeric(df['user_favourites'],errors='coerce')


# In[52]:


df.dtypes


# In[53]:


#Generative Descriptive statistics,
df.describe()


# # univariate Anaylsis 
# 
# ## Distribution of column values across the dataset.
# ## Outlier Detection
# ## Number of Null values.
# ## Number of Unique values.
# ## Aggregation(wherever necessary)

# # user_location 

# In[71]:


#number of unique locations in the datasets 
df["user_location"].nunique()


# In[55]:


#distribution of location of all tweets 
df["user_location"].value_counts()


# In[56]:


#missing values 
df["user_location"].isna().sum()


# In[57]:


# storing in a df to plot it later 
user_location_df = df["user_location"].value_counts().rename_axis("place").reset_index(name="counts")
user_location_df.head()


# In[72]:


user_location_threshold_data = user_location_df[user_location_df["counts"]>1000].head(50)

fig = px.bar(user_location_threshold_data,x="place",y="counts", title="Top 50 Locations tweets originate from")
fig.show()


# # user_verified

# Wiriting function , which helps to inspect the columns

# In[60]:


def inspect_column(data,column):
    print("------------------------------------------------------------------------------------------------------------")
    print("Basic Preliminary Information of column '{}'\n\n".format(column))
    print("Total number of Unique ",column,"values: ",data[column].nunique())
    print("----Quick overview of the distribution of the variable------")
    print(data[column].value_counts())
    print("The number of tweets where the ", column ,"specific data is unkown : ", data[column].isna().sum())
    sub_data_df = data[column].value_counts().rename_axis(column).reset_index(name="counts")
    sub_data_threshold_df = sub_data_df[sub_data_df["counts"]>100].head(100)#set to 100 according to the size of the data 
    fig = px.bar(sub_data_threshold_df,x=column,y="counts", title="Distribution of values of column '{}'".format(column))
    fig.show()


# In[61]:


df['user_verified'] = df['user_verified'].astype('bool')


# In[62]:


inspect_column(df,"user_verified")


# # hashtags

# In[63]:


inspect_column(df,"hashtags")


# maybe we can analyze this feature further. For this it is better to clean the data 

# In[31]:


all_hashtag_list = []


for each_row in df.itertuples():
    if not str(each_row.hashtags).lower() == "nan":
        each_hashtag = str(each_row.hashtags)
        each_hashtag = each_hashtag.strip('[]').replace("'","")
        all_hashtag_list += each_hashtag.split(",")
        
print("Total number of hashtags",len(all_hashtag_list))


# In[32]:


hashtag_df = pd.DataFrame(all_hashtag_list,columns=["hashtags"])
hashtag_df.head()


# we could also look at the individual counts of the hashtags 

# In[66]:


count_df = hashtag_df["hashtags"].value_counts().rename_axis("hashtags").reset_index(name="counts")
count_df.head()


# there are a lot of duplicates. I think more cleaning is needed

# In[67]:


hashtag_final_count_dic = {}
for each_row in count_df.itertuples():
    if str(each_row.hashtags).strip().lower() == "bitcoin":
        if "bitcoin" not in hashtag_final_count_dic:
            hashtag_final_count_dic["bitcoin"] = each_row.counts
        else:
            hashtag_final_count_dic["bitcoin"] += each_row.counts
    else:
        hashtag_final_count_dic[str(each_row.hashtags).strip()] = each_row.counts
        
print("The aggregated hashtags count has {} hashtags".format(len(hashtag_final_count_dic)))


# In[68]:


final_hashtag_count_df = pd.DataFrame(hashtag_final_count_dic.items(),columns=['hashtag','count'])
final_hashtag_count_df


# In[69]:


final_df = final_hashtag_count_df.sort_values(by='count',ascending=False).head(10)
fig = px.bar(final_df,"hashtag","count",title="Top 10 hashtags in this dataframe")
fig.show()


# # user_followers

# Analyzing the distribution of the followers for each twitter handler 

# In[ ]:


fig = px.box(df,y="user_followers", title="The overall distribution of user followers")
fig.show()


# In[31]:


fig = px.box(df[(df["user_followers"]>0) & (df["user_followers"]<=4000)],y="user_followers",title="The distribution of User followers within 4000 user followers")
fig.show()


# # user_favourites

# In[76]:


fig = px.box(df,y="user_favourites",title="Distribution of user favourites")
fig.show()


# # source

# In[6]:


count_df = df["source"].value_counts().rename_axis("source").reset_index(name="counts")
count_df


# In[7]:


#variation of top 10 data sources 
fig = px.bar(count_df.head(10), x='source', y='counts',title="Top 10 Sources to make tweets")
fig.show()


# # is_retweet
# 

# This variable indicates whether a particular tweet was retweeted or not.

# In[8]:


df['is_retweet'] = df['is_retweet'].astype('bool')


# In[9]:


count_df = df["is_retweet"].value_counts()
count_df


# date (user_created)

# In[12]:


df['user_created'] = pd.to_datetime(df['user_created'],format='%Y-%m-%d %H:%M:%S', errors='coerce')


# In[13]:


print("Date column is of '{}' type".format(df["user_created"].dtype))
df["user_created"].head()


# In[14]:


df["day_of_tweet"] = pd.to_datetime(df['user_created']).dt.date
df["day_of_tweet"].head()


# In[15]:


#number of tweets per day 
date_time_series = df.groupby("day_of_tweet").size().rename_axis("day_of_tweet").reset_index(name="number_of_tweets")
date_time_series


# In[16]:


fig = px.line(date_time_series, x='day_of_tweet', y="number_of_tweets", title="Time series for number of tweets made per day")
fig.show()


# # Multi-Variate Exploratory Analysis

# Lets explore and understand the change caused by one of the variables on other variables. Mutlivariate analysis is a prominent technique to draw concrete insights into the behaviour of data.

#  user location / account verification status.

# Now I try to understand what country/location has been making most of the tweets and how many of them are verified.

# In[17]:


# grouping by user_location and user_verified and gathering top 50 entries
user_loc_df = df.groupby(["user_location","user_verified"])["user_verified"].count().reset_index(name="count").sort_values(by=['count'], ascending=False).head(50)
user_loc_df.head()


# In[18]:


# bar plot to show the user_location and count by user_verified
fig = px.bar(user_loc_df, x='user_location',y='count',color='user_verified',barmode="group",
            title="Relationship between the user locations v/s user verified")
fig.show()


# In[19]:


fig = px.pie(user_loc_df, values='count', names='user_verified', title='Ratio of Verified acounts v/s Unverified accounts')
fig.show()


# user_location / Hashtag

# In[21]:


# extracting the user location and their respective hashtags
user_loc_hastag_data = df[["user_location","hashtags"]]
user_loc_hastag_data.head()


# In[22]:


# converting the dataframe to dictionary to aggregate by location
user_loc_hastag_data_dic = user_loc_hastag_data.to_dict(orient='records')
print("There are a total of {} records in the dictionary".format(len(user_loc_hastag_data_dic)))


# In[23]:


# code block to perform string manipulation and extract location keys and aggregated values
cleaned_dic_container = []
for each in user_loc_hastag_data_dic:
    if str(each["user_location"]).lower() != 'nan' and str(each["hashtags"]).lower() != 'nan':
        cleaned_dic = {}
        each["hashtags"] = str(each["hashtags"]).strip('[]').replace("'","").split(",")
        cleaned_dic["user_location"] = str(each["user_location"])
        cleaned_dic["hashtags"] = each["hashtags"]
        cleaned_dic_container.append(cleaned_dic)
cleaned_dic_container[0:5]


# In[24]:


# converting the processed list of dictionaries to a dataframe
user_loc_hashtags_df = pd.DataFrame(cleaned_dic_container)
user_loc_hashtags_df = user_loc_hashtags_df.explode('hashtags')
user_loc_hashtags_df


# In[25]:


# applying final manipulations using lambda functions
hashtag_loc_df = user_loc_hashtags_df.groupby(['user_location',"hashtags"])["hashtags"].count().reset_index(name="count").sort_values(by=['count'], ascending=False).head(100)
hashtag_loc_df["user_location"] = hashtag_loc_df["user_location"].apply(lambda x : x.strip())
hashtag_loc_df["hashtags"] = hashtag_loc_df["hashtags"].apply(lambda x : x.strip())
hashtag_loc_df


# In[26]:


#the most popular hastags aggregated by the user location.
fig = px.bar(hashtag_loc_df,x = "user_location",y="count",color="hashtags",title="What are these countries talking about the most ?")
fig.show()


# user_location / tweet source

# In[28]:


# grouping by user location , user verified and the source. extracting the top 50 most commonly used sources where the users are verified.
user_loc_source_df = df.groupby(["user_location","user_verified","source"])["source"].count().reset_index(name="count").sort_values(by=['count'], ascending=False).head(50)
user_loc_source_df.head()


# In[29]:


fig = px.bar(user_loc_source_df,x="user_location", y="count",color="source", facet_col="user_verified",title="Exploring the relationship between the user location v/s source of the tweet v/s user verification status")
fig.show()

