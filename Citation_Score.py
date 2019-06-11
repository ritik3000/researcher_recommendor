import pandas as pd
import time
import json
import multiprocessing
from multiprocessing import Process
import math
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

def thread_names(l,r):
    for author in author_list[l:r]:
        for i in range(df.shape[0]):
            if i%100000==0:
                print(l,":",r,":",i//100000)
        #print(df.iloc[i].authors)
            if author in df.iloc[i].authors:
                if author in authors_dict:
                    authors_dict[author].append(df.iloc[i].id)
                else:
                    authors_dict[author]=[df.iloc[i].id]

def multiprocessed():
    processes = []
    n = 0
    threads = 40
    for i in range(0, threads):
        stop = n + math.floor(1/threads + 1) if n + threads <= 1 else 1
        p = Process(target=thread_names, args=(n, stop))
        n = stop + 1
        processes.append(p)
    # Start the processes
    for p in processes:
        p.start()
    # Ensure all processes have finished execution
    count = 0
    for p in processes:
        p.join()
        print("Process %d is over and time taked is : %f" %(count, (time.time() - start)))
        count += 1

start = time.time()
if __name__=="__main__":
    manager = multiprocessing.Manager()
    authors_dict = manager.dict()
    multiprocessed()

print(type(authors_dict))
authors_dict=dict(authors_dict)
print(type(authors_dict))

pickle.dump( authors_dict, open( "Coauthor_Citation_score.pkl", "wb" ) )

for i in authors_dict:
    print(i)

