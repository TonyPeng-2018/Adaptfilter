from Models import mobilenetv2
import sys

middle_sizes = [1,2,4,8,16]
client, server = mobilenetv2.mobilenetv2_splitter(num_classes=1000, 
                                        weight_root='/home/tonypeng/Workspace1/adaptfilter/Adaptfilter/Weights/imagenet', 
                                        device='cuda:0')

from Dataloaders import dataloader_image_20

train, _, val, _ = dataloader_image_20.Dataloader_imagenet_20_integrated(device='home',
                                                                         train_batch=64,
                                                                         test_batch=100)

import torch
import torch.optim as optim
import torch.nn as nn

device = torch.device('cuda:0')
client = client.to(device)
server = server.to(device)
client = client.eval()
server = server.eval()

middles = []
for middle_size in middle_sizes:
    middle = mobilenetv2.MobileNetV2_middle(middle=middle_size)
    middle = middle.to(device)
    middles.append(middle)

criterion = nn.CrossEntropyLoss()
ops = []
for middle in middles:
    optimizer = optim.Adam(middle.parameters(), lr=0.001)
    ops.append(optimizer)

from tqdm import tqdm

epochs = 100
min_val_loss = [1000000] * len(middles)
for epoch in tqdm(range(epochs)):
    for middle in middles:
        middle.train()
    for i, data in enumerate(tqdm(train)):
        inputs, labels, newlabel = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        client_out = client(inputs).detach()
        for i in range(len(middles)):
            middle = middles[i]
            optimizer = ops[i]

            optimizer.zero_grad()

            outputs = middle(client_out)
            outputs = server(outputs)
            # print(outputs.size(), labels.size())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    for middle in middles:
        middle.eval()

    val_loss = [0.0] * len(middles)
    correct = [0] * len(middles)
    for i, data in enumerate(val):
        inputs, labels, newlabel = data
        inputs, labels = inputs.to(device), labels.to(device)
        client_out = client(inputs).detach()
        for i in range(len(middles)):
            middle = middles[i]

            outputs = middle(client_out)
            outputs = server(outputs)
            loss = criterion(outputs, labels)
            val_loss[i] += loss.item()
            correct[i] += (outputs.max(1)[1] == labels).sum().item()

    for i in range(len(middles)):
        print('middle size: ', middle_sizes[i])
        print('val loss: ', val_loss[i])
        print('correct: ', correct[i]/len(val.dataset))

        if val_loss[i] <= min_val_loss[i]:
            min_val_loss[i] = val_loss[i]
            torch.save(middles[i].state_dict(), 'mobile_imagenet_middle_'+str(middle_sizes[i])+'.pth')