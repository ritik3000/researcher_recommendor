# from keras.layers import Input, Dense, RNN, LSTM, Concatenate, MaxPooling1D, Embedding
# from keras.models import Model
# from keras.losses import binary_crossentropy
# from keras.optimizers import Adam
# from keras.callbacks import TensorBoard,ModelCheckpoint
# from keras.models import model_from_json
import numpy as np
import pandas as pd
from time import time
import operator
import sys
import pickle

print("-----Start-----")

df = pd.read_hdf("jsons/final_df_dl.h5",'main_df')
#vector_dict=pickle.load(open("pickles/author_vector_dict_dl_final.pkl","rb"))

# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# model.load_weights("model.h5")
# print("Loaded model from disk")

# model.compile(optimizer='RMSprop',
#             loss={'main_output': 'mean_squared_error'},
#             metrics=['mse', 'mae', 'mape', 'cosine'])

Author1=set(df.Author1)
Author2=set(df.Author2)
AuthorList=list(Author1|Author2)
print(len(AuthorList))
Author_old = list(set(df.Author1.iloc[:5999234])|set(df.Author2.iloc[:5999234]))

Author_new = list(set(df.Author1.iloc[5999234:])|set(df.Author2.iloc[5999234:]))

Author_new_test = list(np.setdiff1d(np.array(Author_old),np.setdiff1d(np.array(Author_old), np.array(Author_new), assume_unique=True), assume_unique=True))
print("-----Loaded-----")
score_list=pickle.load(open("pickles/score_dl.pkl","rb"))
print(score_list[('Mark McCann', 'Nicholas Pippenger')])
def graphrec(name):
    target = name
    rec=[]
    #count=0
    Xf1=[]
    Xf2=[] 
    score=list()
    for i in range(len(AuthorList)):
        if(i%100000==0):
            print(i//100000,end=' ')
            sys.stdout.flush()
        try:
            score+=score_list[(target,AuthorList[i])]
        except:
            score+='0'


        # Xeval1=np.array(vector_dict[target])
            
        # Xeval2=np.array(vector_dict[AuthorList[i]])

        # #y  = extract.y

        # X1=np.array([])
        # X2=np.array([])

        # X1=np.append([X1],[Xeval1],axis=1)
        # X2=np.append([X2],[Xeval2],axis=1)

        # X1=X1.reshape(1,50,1)
        # X2=X2.reshape(1,50,1)
        # Xf1.append(X1)
        # Xf2.append(X2)
    
    # X1=np.array(Xf1).reshape(len(AuthorList),50,1)
    # X2=np.array(Xf2).reshape(len(AuthorList),50,1)
    # score = model.predict([X1,X2],batch_size=512,verbose=1)
    # score=list(score.reshape(len(AuthorList)))
    rec=list(zip(AuthorList,score))
    #print(count)
    rec.sort(key=operator.itemgetter(1),reverse=True)
    print(rec[:50])
    return rec[:150]

df_train=df.iloc[:5999234]
df_test=df.iloc[5999234:]

mrrn = 0
mrr = 0
rlo = {}
rln = {}
lor = []
lorn = []
prec = np.zeros(120)
recall = np.zeros(120)
f1 = np.zeros(120)
num=50

for i in range(50):
    print(i,end=' ')
    sys.stdout.flush()
    cnt = cnt1 = 0
    name = Author_new_test[i]
    col1 = df_test[df_test.Author1==name]
    col2 = df_test[df_test.Author2==name] 
    col = pd.concat([col1,col2])
    col_tr1 = df_train  [df_train.Author1==name]
    col_tr2 = df_train[df_train.Author2==name]
    col_tr = pd.concat([col_tr1,col_tr2])
    rel=[]
    rel1=[]
    for i in range(col.shape[0]):
        if col.iloc[i].Author1==name:
            rel.append(col.iloc[i].Author2)
        else:
            rel.append(col.iloc[i].Author1)

    for i in range(col_tr.shape[0]):
        if col_tr.iloc[i].Author1==name:
            rel1.append(col_tr.iloc[i].Author2)
        else:
            rel1.append(col_tr.iloc[i].Author1)

    lst = []
    lstn = []
    retr = graphrec(name)

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
print(f1)

pickle.dump(prec,open("pickles/prec_dl.pkl","wb"))
pickle.dump(f1,open("pickles/f1_dl.pkl","wb"))
pickle.dump(recall,open("pickles/recall_dl.pkl","wb"))