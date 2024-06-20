# dataset cifar10
from Dataloaders import dataloader_cifar10
train, test, val, classes = dataloader_cifar10.Dataloader_cifar10_val(train_batch=128, test_batch=100, seed=2024)

# get the model from original
from Models import mobilenetv2_original
import torch
import torch.optim
import torch.nn.functional as F

model = mobilenetv2_original.MobileNetV2(num_classes=10)
weightpath = '/home/tonypeng/Workspace1/adaptfilter/Adaptfilter/Weights/cifar-10/model/mobilenetv2_0_2024_06_20_17_28_43.pth'
model.load_state_dict(torch.load(weightpath))
model.to('cuda')

from Utils import utils
from tqdm import tqdm
from datetime import datetime



optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
min_loss = -1
# test the model
model = model.eval()
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        outputs = model(inputs)
        outputs = torch.max(outputs, dim=1)[1]
        outputs = targets.eq(outputs).sum().item()
    print(outputs/len(test)/100)
    print('done')


