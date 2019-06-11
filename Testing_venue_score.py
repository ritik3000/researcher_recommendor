
# coding: utf-8

# In[55]:


import pickle
import json
import numpy
import operator
from sklearn.metrics.pairwise import cosine_similarity
import csv


# In[82]:


def cosine(aut_vec):
    vec_array = []
    for name in loa:
        vec_array.append(aut_vec[name])
    data = cosine_similarity(vec_array)
    return data


# In[83]:
alpha=[1,1.5,2,2.5,3,3.5,4]
beta=[1,1.5,2,2.5,3,3.5,4]

ma_f1=[0]*120
org_f1=pickle.load(open("pickles/f1_org.pkl","rb"))
max_diff=0
max_alpha=0
f1_scores=[]
loa=pickle.load(open("pickles/list_of_authors.pkl","rb"))
aut_vec = pickle.load(open("pickles/Aut_vectitle.pkl","rb"), encoding='latin1')
aut_vec = pickle.load(open("pickles/Aut_vec.pkl","rb"), encoding='latin1')
citation_score=pickle.load(open("pickles/Coauthor_citation_score_dict.pkl","rb"))

for alp in alpha:
    print(alp)
    col = json.load(open("jsons/Author_collab_train_5000.json"))
    
    #loa = list(aut_vec.keys())
    
    cos1 = cosine(aut_vec)
    
    #loa = aut_vec.keys()
    cos2 = cosine(aut_vec)
    cn = 0
    for i in range(5000):
        if len(col[loa[i]]) == 0:
            cn = cn + 1
            #print loa[i]
    #print(cn)


    # In[84]:


    #pickle.dump(loa,open("list_of_authors.pkl","wb"))


    # In[85]:

    m=1
    cos = m*cos1 + (1-m)*cos2
    cos


    # In[86]:


    
   # venue_score=pickle.load(open("pickles/normalised_coauthor_venue_score_dict.pkl","rb"))


    # In[87]:


    #citation_score


    # In[88]:


    for i in range(len(loa)):
        for j in range(len(loa)):
            if((loa[i],loa[j]) in citation_score):
                cos[i][j]=alp*cos[i][j]


    # In[89]:


    for i in range(5000):
        cos[i][i] = 0
        for j in range(5000):
            if i != j:

                if loa[i] not in col[loa[j]]:
                    cos[i][j] = 0.0
                    cos[j][i] = 0.0
                    #pass


    # In[90]:


    for i in range(5000):
        su = sum(cos[i])
        if su != 0:
            cos[i] = cos[i]/float(su)


    # In[91]:


    def graphrec(name):

        target = name
        s = [0]*5000
        index = -1
        for i in range(5000):
            if target == loa[i]:
                index = i
                s[i] = 1

        q = s
        for i in range(50):
            s = numpy.dot(0.85,(numpy.dot(s,cos)))+0.15*numpy.array(q)
            #print s[:10]
        rec = []
        for i in range(len(loa)):
            rec.append((loa[i],s[i]))
        rec.sort(key=operator.itemgetter(1),reverse=True)
        return rec


    # In[92]:


    col = json.load(open("jsons/Author_newcollab_test.json"))
    col_tr = json.load(open("jsons/Author_collab_list_test1.json"))

    data = json.load(open("jsons/Author_numnewcollab_test.json"))
    lon = []
    num = 50
    for i in range(num):
        lon.append(data[i][0])
    mrrn = 0
    mrr = 0
    rlo = {}
    rln = {}
    lor = []
    lorn = []
    prec = numpy.zeros(120)
    recall = numpy.zeros(120)
    f1 = numpy.zeros(120)
   # print(lon)


    # In[93]:


    for name in lon:

        rel = col[name]
        rel1 = col_tr[name]
        cnt = cnt1 = 0
        retr = graphrec(name)
        lst = []
        lstn = []
        for i in range(120):
            if retr[i][0] in rel:
                lstn.append(1)
                cnt= cnt + 1
            else:
                lstn.append(0)


            if retr[i][0] in rel1:
                lst.append(1)
                cnt1= cnt1 + 1
            else:
                lst.append(0)
            p = float(cnt1) / (i + 1)
            r = float(cnt1) / len(rel1)
            if p!=0 and r!=0:
                f = 2*p*r/(p+r)
            else:
                f = 0
            prec[i] = prec[i] + p
            recall[i] = recall[i] + r
            f1[i] = f1[i] + f

        #print cnt
        #print cnt1
        lor.append(lst)
        lorn.append(lstn)


    # In[94]:


    prec = prec/num
    recall = recall/num
    f1 = f1/num

    diff=0
    for i in range(120):
        diff+=f1[i]-org_f1[i]
    print(diff/120)
    if (diff/120)>=max_diff:
        print("Changed:",max_diff-diff/120)
        max_diff=diff/120
        max_alpha=alp
        ma_f1=f1
    f1_scores.append([m,alp,diff/120])


print(max_alpha)
print(max_diff)
print(ma_f1)
pickle.dump(ma_f1,open("max_f1_alpha.pkl","wb"))
# pickle.dump(prec,open("prec_tnrec.pkl","wb"))
# pickle.dump(recall,open("recall_tnrec.pkl","wb"))
# pickle.dump(f1,open("f1_tnrec.pkl","wb"))


# In[95]:

out = open('alpha.csv', 'w')
for row in f1_scores:
    for column in row:
        out.write('%f;' % column)
    out.write('\n')
out.close()
# pickle.dump(prec,open("prec_tnrec.pkl","wb"))
# pickle.dump(recall,open("recall_tnrec.pkl","wb"))
#pickle.dump(f1,open("pickles/f1_orgacit.pkl","wb"))

