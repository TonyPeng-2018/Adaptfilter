import os
import sys, getopt
import torch
from torchvision import transforms
from torch import nn
from torch.utils.data import Dataset, DataLoader
from dataset.dataset import ImageDataset
from tqdm import tqdm
from model.mobilenet_slice_nonse import mobilenet_slice_train
from torch.optim.lr_scheduler import MultiStepLR
import json





import numpy as np
import matplotlib.pyplot as plt



def main(args):
    # Specify the path to the folder containing the images
    train_loss_record = []
    eval_loss_record = []
    
    os.chdir("/")
    folder_path = "/data3/zix25/data/imagenet_20/"



    num_epochs = 25
    batch_size = 32
    model = mobilenet_slice_train()
    # print(model)
    # os.exit()
    model = model.to(device="cuda")


    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageDataset(folder_path, "train", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset2 = ImageDataset(folder_path, "val", transform=transform_val)
    dataloader2 = DataLoader(dataset2, batch_size=1, shuffle=True)


    optimizers = []
    for i in range(8):
        # Select the parameters from the part2 module and the corresponding upscaling layer
        # parameters = list(model.part2[i].parameters()) + list(model.upscale_layers[i].parameters()) + list(model.se.parameters()) + list(model.part1.parameters())
        parameters = list(model.part2[i].parameters()) + list(model.upscale_layers[i].parameters()) + list(model.part1.parameters())

        # Create the optimizer for these parameters
        optimizer = torch.optim.Adam(parameters, lr=0.00001)
        # Add the optimizer to the list
        optimizers.append(optimizer)
    criterion = nn.CrossEntropyLoss()
    state = torch.load('/data3/zix25/clio/checkpoints/mobilenetv2_imagenet20_layer3_slice_nonse_weight_50.pth', map_location=lambda storage, loc: storage)
    model.load_state_dict(state)
    # print(model)
    # sys.exit()
    # torch.save(model.state_dict(), '/data2/zix25/clio/checkpoints/weight1.pth')
        # Iterate over the data loader to access the images
    for epoch in tqdm(range(num_epochs)):
        

        
        total_loss_train = [0]*8
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        for batch, tar in tqdm(dataloader):
            # Perform operations on the batch of images
            for i in range(8):
                image = batch.to(device="cuda")
                tar = tar.to(device="cuda")
                optimizers[i].zero_grad()
                output = nn.functional.softmax(model(image, i), dim=1)
                # print(output)
                # print(output.size())
                loss = criterion(tar, output)
                total_loss_train[i] += loss
                # print(loss.size())
                loss.backward()
                optimizers[i].step()
        total_loss_train = [tensor.item() / len(dataloader) for tensor in total_loss_train]
        print(total_loss_train)
        train_loss_record.append(total_loss_train)
        

        total_loss_eval = [0]*8
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        for batch, tar in tqdm(dataloader2):
            # Perform operations on the batch of images
            for i in range(8):
                image = batch.to(device="cuda")
                tar = tar.to(device="cuda")
                output = nn.functional.softmax(model(image, i), dim=1)
                # print(output.size())
                loss = criterion(tar, output)
                total_loss_eval[i] += loss
                # print(tar.size())
                # print(loss)
        total_loss_eval = [tensor.item() / len(dataloader2) for tensor in total_loss_eval]
        print(total_loss_eval)
        eval_loss_record.append(total_loss_eval)

              
        losses_json = json.dumps({"eval_loss_record": eval_loss_record, "train_loss_record": train_loss_record})
        with open('/afs/cs.pitt.edu/usr0/zix25/clio/slicee_nonse_train_loss_50-75.json', 'w') as f:
            f.write(losses_json)
    
    torch.save(model.state_dict(), '/data3/zix25/clio/checkpoints/mobilenetv2_imagenet20_layer3_slice_nonse_weight_75.pth')
if __name__ == "__main__":
    main(sys.argv[1:])

