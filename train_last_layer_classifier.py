from Models import mobilenetv2, resnet
import sys

dataset = "imagenet"
model = sys.argv[1]
if model == "mobile":
    cmodel, smodel = mobilenetv2.mobilenetv2_splitter(weight_root="./Weights/" + dataset + "/")
elif model == "resnet":
    cmodel, smodel = resnet.resnet_splitter(weight_root="./Weights/" + dataset + "/", layers=50)
from Models import last_classifier
linear = last_classifier.last_layer_classifier(1000,20)
# import imagenet-20
from Dataloaders import dataloader_image_20
from torchvision import transforms
import torch.nn as nn
import torch
train, _, val = dataloader_image_20.Dataloader_imagenet_20_integrated(train_batch=64, test_batch=50)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(linear.parameters(), lr=0.001)
epochs = 50

from tqdm import tqdm
cmodel = cmodel.to("cuda:0")
smodel = smodel.to("cuda:0")
linear = linear.to("cuda:0")

cmodel.eval()
smodel.eval()
for param in cmodel.parameters():
    param.requires_grad = False
for param in smodel.parameters():  
    param.requires_grad = False
epochs = 50
min_val_loss = 1000000

import last_classifier_mapping
mapping = last_classifier_mapping.last_layer_mapping
for epoch in tqdm(range(epochs)):
    train_loss = 0.0
    linear.train()
    for i, data in enumerate(train):
        inputs, labels = data
        # transfer labels to 20 classes
        labels = [mapping[label.item()] for label in labels]
        labels = torch.tensor(labels)
        # print(labels)
        inputs, labels = inputs.to("cuda:0"), labels.to("cuda:0")

        optimizer.zero_grad()

        outputs = cmodel(inputs)
        outputs = smodel(outputs).detach()
        outputs = linear(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    print(f"Epoch: {epoch}, Train Loss: {train_loss}")

    val_loss = 0.0
    linear.eval()
    with torch.no_grad():
        for i, data in enumerate(val):
            inputs, labels = data
            labels = [mapping[label.item()] for label in labels]
            labels = torch.tensor(labels)
            inputs, labels = inputs.to("cuda:0"), labels.to("cuda:0")

            outputs = cmodel(inputs)
            outputs = smodel(outputs)
            outputs = linear(outputs)
            # mapping output
            loss = criterion(outputs, labels)
            val_loss += loss.item()
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            print(f"Epoch: {epoch}, Val Loss: {train_loss}")
            torch.save(linear.state_dict(), 'model_last_layer.pth')