import torch.nn as nn
import torch.optim as optim
import torch.utils 
from Models import mobilenetv2, mobilenetv3, resnet
from tqdm import tqdm
import Utils.utils as utils
from Dataloaders.dataloader_imagenet import Dataset_imagenet, Dataloader_imagenet
from Config.external_path import pre_path
import datetime
import numpy as np
import os

run_device = 'home'
start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
run_path = pre_path[run_device] 
# load the dataset
# emb_folder = run_path['cifar10'] + run_path['client'] # b,e,h,w
# emb_out_folder = run_path['cifar10'] + run_path['server'] # b,o
# gt_folder = run_path['cifar10'] + 'cifar-10-labels/' # b,1
# dataset = dataloader_gate_correct(emb_folder, emb_out_folder, gt_folder) # emb, b,e,h,w -> b,10
# instead, use the original dataset
# train, test, classes = Dataloader_cifar10() # train, test, classes
imageset = Dataset_imagenet('server')
train_set, _, val_set = imageset.return_sampler()
tr_dict, _, v_dict = imageset.return_dict()
class_index = imageset.return_class_index()

train = Dataloader_imagenet(train_set, tr_dict, transform=True)
val = Dataloader_imagenet(val_set, v_dict, transform=True)

logger = utils.APLogger('./Logs/cifar-10/gate_train_' + start_time + '.log')
weightfolder = './Weights/cifar-10/gate/'
if not os.path.exists(weightfolder+start_time+'/'):
    os.makedirs(weightfolder+start_time+'/')

gate_name = ['GateMLP', 'GateCNN', 'GateCNN2'] # GateMLP or GateCNN or GateCNN2
ranker_name = ['rankerEntropy'] # GatedRegression or GateCNN or GateCNN2
resume_train = False
# load the model
gates = []
input_sizes = [8, 16, 24]

# rankers = []
# for i, input_size in enumerate(input_sizes):
#     rank = ranker.model_list[ranker_name](in_ch=32, out_ch=input_size)
#     rankers.append(rank)

# added gated to gateds
for i, input_size in enumerate(input_sizes):
    gates.append([])
    for j, gname in enumerate(gate_name):
        gate = gatedmodel.model_list[gname](input_size=input_size, width=32, height=32, output_size=1) # b,e',h,w -> b,1
        # if resume_train:
        #     gate.load_state_dict(torch.load('./Weights/cifar-10/'+gate_name+'_%d.pth' % i))
        gates[-1].append(gate) 
    

for _gate in gates: # size, type
    for gate in _gate:
        gate = gate.cuda()
        gate.train()

# get the server model
client, server = mobilenetv2.stupid_model_splitter(weight_path='./Weights/cifar-10/model/MobileNetV2.pth')
client = client.cuda()
server = server.cuda()

client.eval()
server.eval()

# load the optimizer
optimizers = []
for _gate in gates:
    optimizers.append([])
    for gate in _gate:
        optimizer = optim.Adam(gate.parameters(), lr=0.001)
        optimizers[-1].append(optimizer)

# load the loss function
criterion = nn.MSELoss()
epochs = 50

# train the model
min_val_loss = np.ones((len(input_sizes), len(gates))) * 100000
for epoch in tqdm(range(epochs)):
    e_loss = np.zeros((len(input_sizes), len(gates)))
    count = 0
    for i, data in enumerate(train):
        inputs, gt = data # b,3,h,w; b
        inputs, gt = inputs.cuda(), gt.cuda()
        emb = client(inputs) # b,3,h,w -> b,e,h,w
        emb_out = server(emb) # b,e,h,w -> b,o
        # get the softmax of labels
        emb_out = torch.nn.functional.softmax(emb_out, dim=1) # b, o
        # print('softmax', emb_out[0])
        emb_out = emb_out.detach()
        emb_out = [emb_out[x, gt[x]] for x in range (emb_out.shape[0])] # b, 1
        emb_out = torch.stack(emb_out)
        emb_out = emb_out.unsqueeze(1) # b,1
        
        f_rate = [0.25, 0.5, 0.75] # filter rates
        preds = []
        # get the selected embeddings
        for j, rate in enumerate(f_rate):
            preds.append([])
            s_emb, s_ind = utils.ranker_entropy(emb.detach().cpu(), rate) # b,e,h,w -> b,e',h,w
            s_emb = s_emb.cuda()
            for k, gate in enumerate(gates[j]):
                preds[-1].append(gate(s_emb)) # b,e',h,w -> b,1
        # print('size', preds[0][0].size())
        # print('preds', preds[0])

        for j, pred in enumerate(preds):
            for k, p in enumerate(pred):
                loss = criterion(p, emb_out)
                optimizers[j][k].zero_grad()
                loss.backward()
                optimizers[j][k].step()
                e_loss[j, k] += loss

                if i % 10 == 0:
                    msg = 'Epoch [%d/%d], Batch: %d Loss: %.4f, n.gate: %d, t.gate: %s\n' % (epoch+1, epochs, i, loss, j, gate_name[k])
                    logger.write(msg)
        count += 1

    for j, loss in enumerate(e_loss):
        for k, l in enumerate(loss):
            msg = 'Epoch [%d/%d], Loss: %.4f, gated: %d\n' % (epoch+1, epochs, l/count, j)
            print(msg)
            logger.write(msg)
    # save the pth

    # test the val
    val_loss = np.zeros((len(input_sizes), len(gates)))
    with torch.no_grad():
        for i, data in enumerate(val):
            inputs, gt = data
            inputs, gt = inputs.cuda(), gt.cuda()
            emb = client(inputs)
            emb_out = server(emb)
            emb_out = torch.nn.functional.softmax(emb_out, dim=1)
            emb_out = emb_out.detach()
            emb_out = [emb_out[x, gt[x]] for x in range (emb_out.shape[0])]
            emb_out = torch.stack(emb_out)
            emb_out = emb_out.unsqueeze(1)
            preds = []
            for j, rate in enumerate(f_rate):
                preds.append([])
                s_emb, s_ind = utils.ranker_entropy(emb.detach().cpu(), rate)
                s_emb = s_emb.cuda()
                for k, gate in enumerate(gates[j]):
                    preds[-1].append(gate(s_emb))
            for j, pred in enumerate(preds):
                for k, p in enumerate(pred):
                    loss = criterion(p, emb_out)
                    val_loss[j, k] += loss
    for j, loss in enumerate(val_loss):
        for k, l in enumerate(loss):
            msg = 'Epoch [%d/%d], Val Loss: %.4f, gated: %d\n' % (epoch+1, epochs, l/len(val), j)
            print(msg)
            logger.write(msg)
            if l/len(val) < min_val_loss[j, k]:
                min_val_loss[j, k] = l/len(val)
                torch.save(gates[j][k].state_dict(), weightfolder+start_time+'/'+gate_name[k]+'_%d_'% (j) + start_time + '.pth' )