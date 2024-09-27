# read embs
import base64
import numpy as np
import os

datapath = '../../data/'
subpath = ['imagenet-20-raw-image-224/', 'cifar-10-raw-image/', 'ccpd-raw-image/']
subpath2 = ['imagenet-20-raw-encode/', 'cifar-10-raw-encode/', 'ccpd-raw-encode/']

for ind, sp in enumerate(subpath):
    for i in range(600):
        with open(datapath+sp+str(i)+'.bmp', 'rb') as f:
            data = f.read()
            data = base64.b64encode(data)
            if not os.path.exists(datapath+subpath2[ind]):
                os.makedirs(datapath+subpath2[ind])
            with open(datapath+subpath2[ind]+str(i), 'wb') as f2:
                f2.write(data)
