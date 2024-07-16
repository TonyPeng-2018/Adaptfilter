from Models import mobilenetv2

middle_size = 8
client, server = mobilenetv2.mobilenetv2_splitter(num_classes=10,
                                                  weight_root='/home/tonypeng/Workspace1/adaptfilter/Adaptfilter/Weights/cifar-10',
                                                  device='cuda:0',partition=-1)

from Dataloaders import dataloader_cifar10

train, _, val = dataloader_cifar10.Dataloader_cifar10_val(train_batch=128, test_batch=100, seed=2024)

import torch
import torch.optim as optim
import torch.nn as nn

device = torch.device('cuda:0')
client = client.to(device)
server = server.to(device)
client = client.eval()
server = server.eval()

middle = mobilenetv2.MobileNetV2_middle(middle=middle_size)
middle = middle.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(middle.parameters(), lr=0.001)

from tqdm import tqdm

epochs = 200
min_val_loss = 1000000
for epoch in tqdm(range(epochs)):
    middle.train()
    for i, data in enumerate(train):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = client(inputs)
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
        outputs = client(inputs)
        outputs = middle(outputs)
        outputs = server(outputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
    print('val loss: ', val_loss)
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(middle.state_dict(), 'model_middle_'+str(middle_size)+'.pth')