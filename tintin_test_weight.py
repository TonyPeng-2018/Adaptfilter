from Models import mobilenetv2
client, server = mobilenetv2.mobilenetv2_splitter(num_classes=10,
                                                    weight_root='./Weights/cifar-10',
                                                    device='cuda:0',partition=-1)
from Dataloaders import dataloader_cifar10
_, test, _ = dataloader_cifar10.Dataloader_cifar10_val(datasetpath = '/data/anp407/',
    train_batch=128, test_batch=100, seed=2024)
import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device('cuda:0')
client = client.to(device)
server = server.to(device)
client = client.eval()
server = server.eval()

correct = 0
with torch.no_grad():
    for i, data in enumerate(test):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = client(inputs)
        outputs = server(outputs)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / 10000))
        