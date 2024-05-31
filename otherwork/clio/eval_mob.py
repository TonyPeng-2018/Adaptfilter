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



    num_epochs = 1
    batch_size = 32
    model = mobilenet()
    # print(model)
    # os.exit()
    model = model.to(device="cuda")



    
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    dataset2 = ImageDataset(folder_path, "val_q95", transform=transform_val)
    dataloader2 = DataLoader(dataset2, batch_size=1, shuffle=True)


    state = torch.load('/data3/zix25/clio/checkpoints/mobilenetv2_imagenet20_40.pth', map_location=lambda storage, loc: storage)
    model.load_state_dict(state)
        # Iterate over the data loader to access the images
    for epoch in tqdm(range(num_epochs)):
        
        correct_predictions = 0
        total_predictions = 0
        
        total_loss_eval = 0
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        for batch, tar in tqdm(dataloader2):
            # Perform operations on the batch of images
            image = batch.to(device="cuda")
            tar = tar.to(device="cuda")
            output = nn.functional.softmax(model(image), dim=1)
            _, predicted_classes = torch.max(output, 1)
            # print(tar.size())
            _, tar = torch.max(tar, 1)
            # print("predicted_classes",predicted_classes)
            # print("tar",tar)
            correct_predictions += (predicted_classes == tar).sum().item()
            total_predictions += tar.size(0)
            # print(output.size())

            # print(tar.size())
            # print(loss)
        accuracy = (correct_predictions / total_predictions)
        print(accuracy)

              
    
    
    # torch.save(model.state_dict(), '/data2/zix25/clio/checkpoints/weight2.pth')
if __name__ == "__main__":
    main(sys.argv[1:])

