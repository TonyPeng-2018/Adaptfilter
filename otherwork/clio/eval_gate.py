import os
import sys, getopt
import torch
from torchvision import transforms
from torch import nn
from torch.utils.data import Dataset, DataLoader
from dataset.dataset import ImageDataset
from tqdm import tqdm
from model.mobilenet_slice_nonse import mobilenet_slice_gate
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
    model = mobilenet_slice_gate()

    model = model.half().to(device="cuda")


    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    

    dataset = ImageDataset(folder_path, "val", transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


    state = torch.load('/data3/zix25/clio/checkpoints/mobilenetv2_imagenet20_layer3_slice_nonse_weight_75_exit_43.pth', map_location=lambda storage, loc: storage)
    model.load_state_dict(state)
    hit = 0.0
    exit_statistic = [0.0]*8
    hit_per_gate = [0.0]*8
    hitrate_per_gate = [0.0]*8
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    for batch, tar in tqdm(dataloader):
        # Perform operations on the batch of images
        # tar is the ground truth
        tar = torch.argmax(tar, dim=1, keepdim=True)
        tar = tar.squeeze(1).to(device="cuda")
        for i in range(8):
            image = batch.type(torch.HalfTensor).to(device="cuda")
            # the model has two input, the image and the index of gate pair
            # output is the classification result. exit_state is the prediction of the confidence for the given gate
            # exit_state is a float between 0 and 1, 1 means very confident to exit
            output, exit_state = model(image, i)
            output = nn.functional.softmax(output, dim=1)
            _, predicted_classes = torch.max(output, 1)
            # if the prediction of the confidence is larger than a value, which means the system exit in the given gate, break the for loop and record the data
            if exit_state > 0.9:
                exit_statistic[i] += 1
                if tar == predicted_classes:
                    hit += 1
                    hit_per_gate[i] +=1
                break
    print("hitrate", hit/len(dataloader))
    print("exit_statistic", exit_statistic)
    for i in range(8):
        if exit_statistic[i] != 0.0:
            hitrate_per_gate[i] = hit_per_gate[i]/exit_statistic[i]
        else:
            hitrate_per_gate[i] = "N/A"
    print("hitrate_per_gate", hitrate_per_gate)
            
        

       
    
if __name__ == "__main__":
    main(sys.argv[1:])

