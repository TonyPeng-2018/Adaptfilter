from Models import resnet
import sys

middle_size = int(sys.argv[1])
client, server = resnet.resnet_splitter(num_classes=1000, 
                                        weight_root='/home/tonypeng/Workspace1/adaptfilter/Adaptfilter/Weights/imagenet', 
                                        device='cuda:0',
                                        layers=50)

from Dataloaders import dataloader_image_20

train, _, val = dataloader_image_20.Dataloader_imagenet_20_integrated(device='home',
                                                                         train_batch=64,
                                                                         test_batch=50)

import torch
import torch.optim as optim
import torch.nn as nn

device = torch.device('cuda:0')
client = client.to(device)
server = server.to(device)
client = client.eval()
server = server.eval()

middle = resnet.resnet_middle(middle=middle_size)
middle = middle.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(middle.parameters(), lr=0.001)

from tqdm import tqdm

epochs = 100
min_val_loss = 1000000
for epoch in tqdm(range(epochs)):
    middle.train()
    for i, data in enumerate(train):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = client(inputs).detach()
        outputs = middle(outputs)
        outputs = server(outputs)
        # print(outputs.size(), labels.size())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    middle.eval()
    val_loss = 0.0
    for i, data in enumerate(val):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = client(inputs).detach()
        outputs = middle(outputs)
        outputs = server(outputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
    print('val loss: ', val_loss)
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(middle.state_dict(), 'resnet_imagenet_middle_'+str(middle_size)+'.pth')