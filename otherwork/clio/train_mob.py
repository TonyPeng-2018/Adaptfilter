import os
import sys, getopt
import torch
from torchvision import transforms
from torch import nn
from torch.utils.data import Dataset, DataLoader
from dataset.dataset import ImageDataset
from tqdm import tqdm
from model.mobilenet import mobilenet
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



    num_epochs = 40
    batch_size = 32
    model = mobilenet()
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



    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    criterion = nn.CrossEntropyLoss()


    # torch.save(model.state_dict(), '/data2/zix25/clio/checkpoints/weight1.pth')
        # Iterate over the data loader to access the images
    for epoch in tqdm(range(num_epochs)):
        total_loss_train = 0
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        for batch, tar in tqdm(dataloader):
            # Perform operations on the batch of images
            image = batch.to(device="cuda")
            tar = tar.to(device="cuda")
            optimizer.zero_grad()
            output = model(image)
            output = nn.functional.softmax(model(image), dim=1)
            # print(output)
            # print(output.size())
            loss = criterion(tar, output)
            total_loss_train += loss
            # print(loss.size())
            loss.backward()
            optimizer.step()
        print(total_loss_train/len(dataloader))
        train_loss_record.append((total_loss_train/len(dataloader)).item())
        
        total_loss_eval = 0
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        for batch, tar in tqdm(dataloader2):
            # Perform operations on the batch of images
            image = batch.to(device="cuda")
            tar = tar.to(device="cuda")
            output = nn.functional.softmax(model(image), dim=1)
            # print(output.size())
            loss = criterion(tar, output)
            total_loss_eval += loss
            # print(tar.size())
            # print(loss)
        print(total_loss_eval/len(dataloader2))
        eval_loss_record.append((total_loss_eval/len(dataloader2)).item())
        

              
        losses_json = json.dumps({"eval_loss_record": eval_loss_record, "train_loss_record": train_loss_record})
        with open('/afs/cs.pitt.edu/usr0/zix25/clio/losses.json', 'w') as f:
            f.write(losses_json)
    
    
    torch.save(model.state_dict(), '/data3/zix25/clio/checkpoints/mobilenetv2_imagenet20_40.pth')
if __name__ == "__main__":
    main(sys.argv[1:])

