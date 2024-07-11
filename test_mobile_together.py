from Models import mobilenetv2

model = mobilenetv2.MobileNetV2_together(num_classes=10)

from Dataloaders import dataloader_cifar10

_, test, _ = dataloader_cifar10.Dataloader_cifar10_val(train_batch=128, test_batch=100, seed=2024)

import torch
import torch.optim as optim
import torch.nn as nn

model.load_state_dict(torch.load('model_together.pth'))
device = torch.device('cuda:0')
model = model.to(device)

from tqdm import tqdm

epochs = 200
min_val_loss = 1000000
model.eval()
with torch.no_grad():
    all_correct = 0
    for i, data in enumerate(test):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        outputs = torch.argmax(outputs, dim=1)
        correct = torch.sum(outputs == labels)
        all_correct += correct
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * all_correct / 10000))
