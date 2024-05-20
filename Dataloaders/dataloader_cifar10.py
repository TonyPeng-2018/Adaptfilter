# this file is a data loader for cifar 10. 
# the root path is /data3/anp407/cifar-10-batches-py
# inspired by https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html?highlight=cifar

import torch
import torchvision
import torchvision.transforms as transforms
import sys
import cv2

def Dataloader_cifar10(batch_size, seed):
    torch.manual_seed(seed)
    # resize the figure to 224*224, and normalize the figure
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((224, 224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # # load data, https://github.com/chenhang98/mobileNet-v2_cifar10/blob/master/train.py 
    # Where are these values from?
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding = 4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    

    trainset = torchvision.datasets.CIFAR10(root='/home/tonypeng/Workspace1/adaptfilter/data/', train=True, download=True, transform=transform)
    # put 80% for training and 20% for validation
    trainset, valset = torch.utils.data.random_split(trainset, [40000, 10000])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=8)
    testset = torchvision.datasets.CIFAR10(root='/home/tonypeng/Workspace1/adaptfilter/data/', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, valloader, classes

# add a main testing function here
if __name__ == '__main__':
    train, test, val, classes = Dataloader_cifar10(128, 2024)
    for i, data in enumerate(train):
        inputs, labels = data
        break
    # print the size of each loader
    print(len(train))
    print(len(test))
    print(len(val))
    print('done')