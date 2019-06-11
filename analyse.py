from keras.layers import Input, Dense, RNN, LSTM, Concatenate, MaxPooling1D, Embedding
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.callbacks import TensorBoard,ModelCheckpoint
from keras.models import model_from_json
import numpy as np
import pandas as pd
from time import time
import pickle
import sys

print("-----Start-----")

df1 = pd.read_hdf("jsons/yearwiseAuthor1.h5",'main_dfAuthor1')

df2 = pd.read_hdf("jsons/yearwiseAuthor2.h5",'main_dfAuthor2')

print("-----Data Loaded-----")

a1=np.array(list(df1.iloc[5999234:]))
a2=np.array(list(df2.iloc[5999234:]))

author1=set(df1.iloc[5999234:])
author2=set(df2.iloc[5999234:])

author1=list(author1|author2)

author1x=set(df1.iloc[:5999234])
author2x=set(df2.iloc[:5999234])
author1x=list(author1x|author2x)

new = list(np.setdiff1d(np.array(author1),np.array(author1x), assume_unique=True))
print("-----All Set-----")
new_dict={key:[] for key in new}
print("-----Dictionary Made-----")

count = 0
for author in author1:
    if count%1000==0:
        print(count//1000,end=' ')
        sys.stdout.flush()
    count+=1
    try:
        new_dict[author].append(list(a2[np.array(a1==author)]))
    except:
        pass
print("-----Complete-----")
pickle.dump(new_dict,open("pickles/new_dict.pkl","wb"))
print("-----End-----")

