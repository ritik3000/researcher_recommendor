import pandas as pd

df1=pd.read_json("dblp-ref-0.json",lines=True)
df2=pd.read_json("dblp-ref-1.json",lines=True)
df3=pd.read_json("dblp-ref-2.json",lines=True)
df4=pd.read_json("dblp-ref-3.json",lines=True)
frames=[df1,df2,df3,df4]
df=pd.concat(frames)
print("jbgkljhgvhjk")
frames=[df1,df2,df3,df4]
df=pd.concat(frames)
df.dropna(inplace=True)

df=df.reset_index()

df.to_json("dblp_merge.json")