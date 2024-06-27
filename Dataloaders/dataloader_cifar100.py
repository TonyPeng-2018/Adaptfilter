import torch
import torchvision
import torchvision.transforms as transforms
import sys
import cv2
# https://github.com/weiaicunzai/pytorch-cifar100/blob/master/dataset.py
# a high-star git repo for CIFAR100 seems that it doesn't use the normalize

def Dataloader_cifar100(train_batch=128, test_batch=100, seed=2024, val_set = False, datasetpath = '/home/tonypeng/Workspace1/adaptfilter/data/'):
    torch.manual_seed(seed)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR100(root=datasetpath, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root=datasetpath, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch, shuffle=False, num_workers=4)
    return trainloader, testloader

def Dataloader_cifar100_val(train_batch=128, test_batch=100, seed=2024, datasetpath = '/home/tonypeng/Workspace1/adaptfilter/data/'):
   

    torch.manual_seed(seed)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR100(root=datasetpath, train=True, transform=transform)
    trainset_len = len(trainset)
    trainset, valset = torch.utils.data.random_split(trainset, [trainset_len-int(0.2*trainset_len), int(0.2*trainset_len)])

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch, shuffle=True, num_workers=4)
    valoader = torch.utils.data.DataLoader(
        valset, batch_size=train_batch, shuffle=False, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root=datasetpath, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch, shuffle=False, num_workers=4)
    return trainloader, testloader, valoader

if __name__ == '__main__':
    Dataloader_cifar100()
    Dataloader_cifar100_val()