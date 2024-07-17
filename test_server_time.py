# test server speed
import cv2
from Models import mobilenetv2
import numpy as np
from PIL import Image
import sys
import time
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from Utils import utils

dataset = 'cifar-10'
batch = 100
classify = 10
resize = (1,3,32,32)

client, server = mobilenetv2.mobilenetv2_splitter(num_classes=classify,weight_root='./Weights/' + dataset + '/', partition=-1)
client.eval()
server.eval()
client = client.to('cuda:0')
server = server.to('cuda:0')


cr = 19
server_latency = np.zeros((batch, cr))
jpg_acc = np.zeros((batch, cr))
original_acc = np.zeros((batch, 1))

mean, std = utils.image_transform(dataset)
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

with open ('../data/' + dataset + '-client/labels.txt', 'r') as f:
    lines = f.readlines()
    labels = []
    for line in lines:
        labels.append([int(line.strip())])

for i in tqdm(range(batch)):
    for j in range(cr):
        fp = '../data/' + dataset + '-client/%s_%s.jpg' % (str(i), str((j+1)*5))
        img = Image.open(fp)
        img = transform(img)
        # show image

        img = torch.tensor(img).float()
        img = img.view(resize)
        img = img.to('cuda:0')

        label = labels[i]

        t1 = time.time()
        emb = client(img)
        out = server(emb)
        _, pred = torch.max(out, 1)
        pred = pred.cpu().numpy()
        t2 = time.time()

        server_latency[i,j] = t2 - t1
        jpg_acc[i,j] = 1 if label == pred else 0
    fp = '../data/' + dataset + '-client/%s.bmp' % (str(i))

    img = Image.open(fp)
    img = transform(img)
    img = torch.tensor(img).float()
    img = img.view(resize)
    img = img.to('cuda:0')
    emb = client(img)
    out = server(emb)
    _, pred = torch.max(out, 1)
    pred = pred.cpu().numpy()
    
    lable = labels[i]
    original_acc[i,0] = 1 if label == pred else 0
    
np.save('server_latency_' + dataset, server_latency)
np.save('jpg_acc_' + dataset, jpg_acc)
np.save('original_acc_' + dataset, original_acc)