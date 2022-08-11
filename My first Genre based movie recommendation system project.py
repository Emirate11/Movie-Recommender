#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# In[25]:


df = pd.read_csv("movies.csv")
df.head(20)


# In[14]:


df.info()


# In[3]:


df["genres"] = df["genres"].str.split("|")
df["genres"] = df["genres"].fillna("").astype("str")


# In[4]:


tf = TfidfVectorizer(analyzer="word",ngram_range=(1, 2),min_df=0, stop_words="english")
tfidf_matrix = tf.fit_transform(df["genres"])
tfidf_matrix


# In[39]:


cosine_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_similarity[:4, :4]


# In[24]:


titles = df["title"]
indices = pd.Series(df.index, index=df["title"])


# In[40]:


def my_recommendations(title):
    idx = indices[title]
    similar_score = list(enumerate(cosine_similarity[idx]))
    similar_score = sorted(similar_score, key=lambda x: x[1], reverse=True)
    similar_score = similar_score[1:10]
    df_indices = [i[0] for i in similar_score]
    return titles.iloc[df_indices]


# In[41]:


my_recommendations("Casino (1995)")


# ## Thank you

# **Justin**
# 

# In[ ]:




