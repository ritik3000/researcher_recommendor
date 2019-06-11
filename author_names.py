import pickle

with open('Aut_vec.pkl', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    author_list = u.load()
author_dict_list=list(author_list.keys())

with open('Aut_list.txt','wb') as f:
	pickle.dump(author_dict_list,f)