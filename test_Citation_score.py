import pandas as pd
import pickle
from signal import signal, SIGPIPE, SIG_DFL

signal(SIGPIPE, SIG_DFL)

with open('Aut_list.txt', 'rb') as f:
    author_list = pickle.load(f)

print("Read Author_list")

df1=pd.read_json("dblp-ref-0.json",lines=True)
print("a")
df2=pd.read_json("dblp-ref-1.json",lines=True)
print("b")
df3=pd.read_json("dblp-ref-2.json",lines=True)
print("c")
df4=pd.read_json("dblp-ref-3.json",lines=True)
print("d")

frames=[df1,df2,df3,df4]
df=pd.concat(frames)
frames=[df1,df2,df3,df4]
df=pd.concat(frames)
df.dropna(inplace=True)

df=df.reset_index()

print("Read DBLP")




