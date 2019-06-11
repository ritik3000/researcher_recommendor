
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import pickle
import json


# In[3]:


df=pd.read_json("jsons/dblp_merge.json")
df.reset_index(inplace=True)


# In[4]:


authors=list(df.authors)


# In[5]:


author_set=set()


# In[6]:


for author_list in authors:
    for author in author_list:
        author_set.add(author)


# In[3]:


aut_list=pickle.load(open("pickles/aut_list.pkl","rb"))


# In[ ]:


alist=[0]*(len(author_set))
coauthor_vector={key: list(alist) for key in aut_list}


# In[ ]:


print(coauthor_vector)


# In[5]:


# pickle.dump(aut_list,open("pickles/python2aut_list.pkl","wb"),protocol=2)


# # In[10]:


# coauthor_list=pickle.load(open("pickles/coauthor_list.pickle","rb"))


# # In[16]:


# pickle.dump(coauthor_list,open("pickles/python2new_coauthor_list.pkl","wb"),protocol=2)


# In[14]:


coauthor_list

