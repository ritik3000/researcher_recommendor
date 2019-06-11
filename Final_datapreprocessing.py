
# coding: utf-8

# In[4]:


import pandas as pd
import itertools
import math
import time
import gensim
from collections import Counter
from gensim.parsing.preprocessing import remove_stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Doc2Vec
import pickle


# In[2]:
print("----Start----\n")

# df1=pd.read_json("jsons/dblp-ref-0.json",lines=True)
# df2=pd.read_json("jsons/dblp-ref-1.json",lines=True)
# df3=pd.read_json("jsons/dblp-ref-2.json",lines=True)
# df4=pd.read_json("jsons/dblp-ref-3.json",lines=True)
# frames=[df1,df2,df3,df4]


# # In[3]:


# df=pd.concat(frames)
# # df.head()

# print("Data Loaded\n")
# # In[4]:


# df.drop(['id','references','title','venue'],axis=1,inplace=True)


# # In[5]:


# print("With Nan: ",df.shape)
# df.dropna(inplace=True)
# print("Without Nan: ",df.shape)


# # In[6]:


# def handle_abstract(old_abstract):
#     if 'Background#R##N#' in old_abstract:
#         return(old_abstract[16:])
#     return(old_abstract)


# # In[7]:


# df['abstract'] = df['abstract'].apply(handle_abstract)
# df=df.sort_values('year')
# df=df.reset_index(drop=True)


# # In[8]:


# # df.shape


# # In[9]:


# df=df.drop(df[df.authors.map(len)<2].index)
# df=df.reset_index(drop=True)


# # In[10]:


# # df.shape


# # In[11]:
# print("Preprocessing Completed\n")

# def extract_coauthors(df):
#     coauthor_list=[]
#     for i in range(df.shape[0]):
#         if(i%10000==0):
#             print(i//10000,end=' ')
#         comb=list(itertools.combinations(df.iloc[i].authors, 2))
#         coauthor_list.extend(list(map(lambda data: (data,df.iloc[i].year),comb)))

#     return coauthor_list


# # In[12]:


# coauthor_list=extract_coauthors(df)


# # In[13]:
# author_dict=dict()

# for coauthor,score in coauthor_list:
#     author_dict[coauthor[0]] = ''
#     author_dict[coauthor[1]] = ''


# # In[15]:

# print("Start collecting author_dict")
# count = 0
# for i in range(df.shape[0]):
#     if count%100000 == 0:
#         print(count//100000,end=' ')
#     count+=1
#     for author in df.iloc[i].authors:
#         author_dict[author] += df.iloc[i].abstract


# # In[16]:


# present_year=df['year'].max()


# # In[17]:


# coauthor_set = Counter()
# for k,v in coauthor_list:
#     coauthor_set[tuple(sorted(k))] += 1/math.log(present_year-v+2,2)

# # In[18]:


# coauthor_set=list(coauthor_set.items())


# In[19]:


# len(coauthor_set)


# In[20]:

# print("start making DataFrame")
# main_df = {'Author1':[], 'Author2':[], 'Score':[]}
# for coauthor,score in coauthor_set:
#     main_df['Author1'].append(coauthor[0])
#     main_df['Author2'].append(coauthor[1])
#     main_df['Score'].append(score)
# main_df=pd.DataFrame(main_df)


# # In[21]:


# main_df['Score']=(main_df['Score']-main_df['Score'].min())/(main_df['Score'].max()-main_df['Score'].min())

# main_df.to_json("jsons/final_df.json")

model=Doc2Vec.load('others/doc2vec_model')




# In[ ]:
author_dict = pickle.load(open("pickles/author_dict_final_datapreprocessing.pkl","rb"))
print("start making final_df\n")
count = 0
author_vector_dict=dict()
start=time.time()
for author in author_dict:
    if count% 10000 == 0:
        print(count//10000,end=' ')
        print(time.time()-start,end=' ')
    count+=1
    #preprocessed_abstract=gensim.utils.simple_preprocess(remove_stopwords(author_dict[author]))
    author_vector_dict[author]=list(model.infer_vector(gensim.utils.simple_preprocess(remove_stopwords(author_dict[author]))))


# In[1]:


# author_vector_dict


# In[2]:


# len(author_list)


# In[6]:


# pickle.dump(author_dict,open("pickles/author_dict.pkl","wb"))
# author_dict=pickle.load(open("pickles/author_dict.pkl","rb"))


# In[101]:




print("Done\n")
# In[95]:


pickle.dump(dict(author_vector_dict),open("pickles/author_vector_dict_dl.pkl","wb"))


# In[90]:


print(author_vector_dict[author_list[0]])

print("complete\n")