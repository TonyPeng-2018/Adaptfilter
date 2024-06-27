# dataset cifar10
from Dataloaders import dataloader_cifar100
train, test, val = dataloader_cifar100.Dataloader_cifar100_val(train_batch=128, test_batch=100, seed=2024, 
                                                                      datasetpath='/data/anp407/')

# get the model from original
from Models import resnet

cuda_no = '2'
model = resnet.resnet152(num_classes=100)
model.to('cuda:' + cuda_no)

from Utils import utils
from tqdm import tqdm
from datetime import datetime

# logger
start_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
logger = utils.APLogger(path='./Logs/cifar-100/resnet152_' +start_time+ '.log')
logger.write('model: resnet152, dataset: cifar100, training')

import torch
import torch.optim
import torch.nn.functional as F

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
min_loss = -1
# train the model
for epoch in tqdm(range(100)):
    model = model.train()
    for batch_idx, (inputs, targets) in enumerate(train):
        inputs, targets = inputs.to('cuda:' + cuda_no), targets.to('cuda:' + cuda_no)
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
            inputs, targets = inputs.to('cuda:' + cuda_no), targets.to('cuda:' + cuda_no)
            outputs = model(inputs)
            outputs = torch.max(outputs, dim=1)[1]
            outputs = targets.eq(outputs).sum().item()
            val_loss += outputs
        logger.write('Validation: %.4f' % (val_loss/len(val)/128))
    if min_loss < outputs:
        min_loss = loss.item()
        torch.save(model.state_dict(), './Weights/cifar-100/pretrained/resnet152_' + start_time + '.pth')

# test the model
model = model.eval()
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test):
        inputs, targets = inputs.to('cuda:' + cuda_no), targets.to('cuda:' + cuda_no)
        outputs = model(inputs)
        outputs = torch.max(outputs, dim=1)[1]
        outputs = targets.eq(outputs).sum().item()
    logger.write('Test: %.4f' % (outputs/len(test)/100))
    print('done')


