import json
dict = {}
cnt = 0
j = []
#j.append(json.load(open('dblp_0_train.json')))

#j.append(json.load(open('dblp_1_train.json')))
#j.append(json.load(open('dblp_2_train.json')))
#j.append(json.load(open('dblp_3_train.json')))
#fin = j[0] +j[1]
#json.dump(fin, open("dblp_total_train.json","wb"))
#print len(fin)

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.stem.porter import *
import json
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
stemmer = PorterStemmer()
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(stemmer.stem(word) for word in punc_free.split())
    return normalized

loa = json.load(open("loa.json"))
loa = loa[:50000]

print(loa[:5])
# loa1 = [i[0] for i in loa]
# j0 = json.load(open("dblp_0_test.json"))
# j1 = json.load(open("dblp_1_test.json"))
# j2 = json.load(open("dblp_2_test.json"))
# j3 = json.load(open("dblp_3_test.json"))
# ldic = {}
# for i in loa1:
#     ldic[i]=1
# js = j0+j1+j2+j3
# print js[0].keys()
# print len(js)
# for el in js:

#         loa = el['authors']
#         yr = el['year']
#         if len(loa) != 1:
#             cln = clean(el['abstract'])

#             for str in loa:
#                 if ldic.get(str,0)==1:

#                     dict.setdefault(str, {})

#                     dict[str][yr] = dict[str].get(yr,"") + " " + cln



#json.dump(dict, open("Author_Abstract_small_test.json","wb"))




