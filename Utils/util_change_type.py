# read embs
import base64
import numpy as np
dataset = 'imagenet'
nc = 'c1'
filenames = ['./'+nc+'_gate1_'+dataset, './'+nc+'_gate2_'+dataset, './'+nc+'_gate3_'+dataset]
for fn in filenames:
    with open(fn, 'rb') as f:
        data = f.read()
        data = base64.b64decode(data)
        data = np.frombuffer(data, dtype=np.float32)
        data = data*255
        data = data.astype(np.uint8)
        
        nfn = fn+'_uint8'
        data = data.tobytes()
        data = base64.b64encode(data)
        with open(nfn, 'wb') as nf:
            nf.write(data)