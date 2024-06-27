# dataset cifar10
from Dataloaders import dataloader_cifar10
train, test, val, classes = dataloader_cifar10.Dataloader_cifar10_val(train_batch=128, test_batch=100, seed=2024, 
                                                                      datasetpath='/data/anp407/')

# get the model from original
from Models import resnet

model = resnet.resnet50(num_classes=10)
model.to('cuda')

from Utils import utils
from tqdm import tqdm
from datetime import datetime

# logger
start_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
logger = utils.APLogger(path='./Logs/cifar-10/resnet50_cifar10_' +start_time+ '.log')
logger.write('model: resnet50, dataset: cifar10, training')

import torch
import torch.optim
import torch.nn.functional as F

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
min_loss = -1
# train the model
for epoch in tqdm(range(100)):
    model = model.train()
    for batch_idx, (inputs, targets) in enumerate(train):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            logger.write('Epoch: %d, Batch: %d, Loss: %.3f' % (epoch, batch_idx, loss.item()))
    
    # test
    model = model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch_idx, (inputs, targets) in enumerate(val):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = model(inputs)
            outputs = torch.max(outputs, dim=1)[1]
            outputs = targets.eq(outputs).sum().item()
            val_loss += outputs
        logger.write('Validation: %.4f' % (val_loss/len(val)/128))
    if min_loss < outputs:
        min_loss = loss.item()
        torch.save(model.state_dict(), './Weights/cifar-10/pretrained/resnet50_' + start_time + '.pth')

# test the model
model = model.eval()
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        outputs = model(inputs)
        outputs = torch.max(outputs, dim=1)[1]
        outputs = targets.eq(outputs).sum().item()
    logger.write('Test: %.4f' % (outputs/len(test)/100))
    print('done')


