#Author-Author graph creation
from __future__ import division
import pickle
import json
import numpy

import operator
from sklearn.metrics.pairwise import cosine_similarity


def cosine(aut_vec):
    vec_array = []
    for name in loa:
        vec_array.append(aut_vec[name])
    data = cosine_similarity(vec_array)
    return data
col = json.load(open("jsons/Author_collab_train_5000.json"))

aut_vec = pickle.load(open("pickles/Aut_vectitle.pkl","rb"),encoding='latin1')
#print len(aut_vec)
loa = list(aut_vec.keys())
cos1 = cosine(aut_vec)
aut_vec = pickle.load(open("pickles/Aut_vec.pkl","rb"),encoding='latin1')
#loa = aut_vec.keys()
cos2 = cosine(aut_vec)
m = 1
cn = 0
for i in range(5000):
    if len(col[loa[i]]) == 0:
        cn = cn + 1
        #print loa[i]
#print cn
cos = m*cos1 + (1-m)*cos2

for i in range(5000):
    cos[i][i] = 0
    for j in range(5000):
        if i != j:

            if loa[i] not in col[loa[j]]:
                cos[i][j] = 0.0
                cos[j][i] = 0.0
                #pass

for i in range(5000):
    su = sum(cos[i])
    if su != 0:
     cos[i] = cos[i]/su
    #else:
     #print "sum = zero occured"

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
print(len(lon))
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
          cnt = cnt + 1
        else:
          lstn.append(0)


        if retr[i][0] in rel1:
          lst.append(1)
          cnt1 = cnt1 + 1
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
prec = prec/num
recall = recall/num
f1 = f1/num

# pickle.dump(prec,open("prec_tnrec.pkl","wb"))
# pickle.dump(recall,open("recall_tnrec.pkl","wb"))
# pickle.dump(f1,open("f1_tnrec.pkl","wb"))


"""from math import log


def precision_at_k(r, k):
    assert k >= 1
    r = numpy.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return numpy.mean(r)


def average_precision(r):
    r = numpy.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return numpy.mean(out)


def mean_average_precision(rs):
    return numpy.mean([average_precision(r) for r in rs])


def dcg_at_k(scores):
    assert scores

    return scores[0] + sum(sc / log(ind, 2) for sc, ind in zip(scores[1:], range(2, len(scores) + 1)))


def ndcg_at_k(predicted_scores, user_scores, k):
    predicted_scores = predicted_scores[:k]
    user_scores = user_scores[:k]
    assert len(predicted_scores) == len(user_scores)
    idcg = dcg_at_k(sorted(user_scores, reverse=True))
    return (dcg_at_k(predicted_scores) / idcg) if idcg > 0.0 else 0.0


def mean_reciprocal_rank(rs):
    rs = (numpy.asarray(r).nonzero()[0] for r in rs)
    return numpy.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])
print "Map for new",mean_average_precision(lorn)
print "Map for total",mean_average_precision(lor)

print "mrr for new", mean_reciprocal_rank(lorn)
print "mrr for total", mean_reciprocal_rank(lor)
ndcg_120 = 0
ndcgn_120 = 0
ndcg_40 = 0
ndcgn_40 = 0
ndcg_80 = 0
ndcgn_80 = 0
idl = [1] * 120
for i in range(len(lor)):
    ndcg_120 = ndcg_120 + ndcg_at_k(lor[i], idl, 120)
    ndcgn_120 = ndcgn_120 + ndcg_at_k(lorn[i],idl, 120)
    ndcg_40 = ndcg_40 + ndcg_at_k(lor[i], idl, 40)
    ndcgn_40 = ndcgn_40 + ndcg_at_k(lorn[i], idl, 40)
    ndcg_80 = ndcg_80 + ndcg_at_k(lor[i], idl, 80)
    ndcgn_80 = ndcgn_80 + ndcg_at_k(lorn[i], idl, 80)

ndcg_40 = ndcg_40 / num
ndcgn_40 = ndcgn_40 / num
ndcg_80 = ndcg_80 / num
ndcgn_80 = ndcgn_80 / num
ndcg_120 = ndcg_120 / num
ndcgn_120 = ndcgn_120 / num

print "ndcg for new_40", ndcgn_40
print "ndcg for total_40", ndcg_40
print "ndcg for new_80", ndcgn_80
print "ndcg for total_80", ndcg_80
print "ndcg for new_120", ndcgn_120
print "ndcg for total_120", ndcg_120

#json.dump(rlo,open("graphrecomtotal.json","wb"))
#json.dump(rln, open("graphrecomnew.json","wb"))"""




