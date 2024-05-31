import os
import sys, getopt
import torch
from torchvision import transforms
from torch import nn
from torch.utils.data import Dataset, DataLoader
from dataset.dataset import ImageDataset
from tqdm import tqdm
from model.mobilenet_slice_nonse import mobilenet_slice_gate_train
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
    model = mobilenet_slice_gate_train()

    model = model.to(device="cuda")


    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    

    dataset = ImageDataset(folder_path, "train", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



    optimizer = torch.optim.Adam(model.gates.parameters(), lr=0.0001)

    loss_fn_gate = torch.nn.BCELoss()
    # load the already trained encoders and decoders
    state = torch.load('/data3/zix25/clio/checkpoints/mobilenetv2_imagenet20_layer3_slice_nonse_weight_75_exit_18.pth', map_location=lambda storage, loc: storage)
    model.load_state_dict(state, strict=True)
    # Iterate over the data loader to access the images
    for epoch in tqdm(range(num_epochs)):
        total_exit_loss_train = [0]*7
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        model.gates.train()
        for param in model.gates.parameters():
            param.requires_grad = True

        for batch, tar in tqdm(dataloader):
            # Perform operations on the batch of images
            # tar is the ground true truth result
            tar = torch.argmax(tar, dim=1, keepdim=True)
            tar = tar.squeeze(1).to(device="cuda")
            for i in range(7):
                image = batch.to(device="cuda")
                # the model has two input, the image and the index of gate
                # output is the classification result. exit_state is the prediction of the confidence for the given gate.
                # exit_state is a float between 0 and 1, 1 means very confident to exit
                output, exitstate = model(image, i)
                optimizer.zero_grad()
                output = nn.functional.softmax(output, dim=1)
                _, predicted_classes = torch.max(output, 1)
                # if the prediction are correct, we set expected_exit to 1, else we set it to 0
                expected_exit = (tar == predicted_classes).float()
                # calculate the BCEloss between the prediction of the confidence and expected_exit
                exit_loss = loss_fn_gate(exitstate,expected_exit.unsqueeze(1))
                total_exit_loss_train[i] += exit_loss
                exit_loss.backward()
                optimizer.step()
        total_exit_loss_train = [tensor.item() / len(dataloader) for tensor in total_exit_loss_train]
        print(total_exit_loss_train)
        

       
    
    torch.save(model.state_dict(), '/data3/zix25/clio/checkpoints/mobilenetv2_imagenet20_layer3_slice_nonse_weight_75_exit_43.pth')
if __name__ == "__main__":
    main(sys.argv[1:])

