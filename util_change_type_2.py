# read embs
import base64
import numpy as np
dataset = 'imagenet'
nc = 'c2'
filenames = ['./'+nc+'_gate1_'+dataset, './'+nc+'_gate2_'+dataset, './'+nc+'_gate3_'+dataset]
for fn in filenames:
    with open(fn, 'rb') as f:
        data = f.read()
        data = base64.b64decode(data)
        data = np.frombuffer(data, dtype=np.int8)
        print(data[3])