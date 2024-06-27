# this file is a data loader for cifar 10. 
# the root path is /data3/anp407/cifar-10-batches-py
# inspired by https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html?highlight=cifar
# the dataloader is from https://github.com/kuangliu/pytorch-cifar


import torch
import torchvision
import torchvision.transforms as transforms
import sys
import cv2

# def Dataloader_cifar10(batch_size, seed, val_set = False):
#     # batch size is the number of images in each batch
#     # seed is the random seed for the data loader
#     # val_set is a boolean to determine if we want to have a validation set

#     torch.manual_seed(seed)
#     # resize the figure to 224*224, and normalize the figure
#     # original transform
#     # transform = transforms.Compose(
#     #     [transforms.ToTensor(),
#     #      transforms.Resize((224, 224)),
#     #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
#     # # load data, https://github.com/chenhang98/mobileNet-v2_cifar10/blob/master/train.py 
#     # Where are these values from?
#     # transform_train = transforms.Compose([
#     #     transforms.RandomCrop(32, padding = 4),
#     #     transforms.RandomHorizontalFlip(),
#     #     transforms.ToTensor(),
#     #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
#     # transform_test = transforms.Compose([
#     #     transforms.ToTensor(),
#     #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

#     # load transform from https://github.com/kuangliu/pytorch-cifar/tree/master
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])

#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
    

#     trainset = torchvision.datasets.CIFAR10(root='/home/tonypeng/Workspace1/adaptfilter/data/', train=True, download=True, transform=transform_train)
#     # put 80% for training and 20% for validation
#     # maybe we don't need validation set
#     if val_set:
#         trainset, valset = torch.utils.data.random_split(trainset, [40000, 10000])
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
#     if val_set:
#         valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=8)
#     testset = torchvision.datasets.CIFAR10(root='/home/tonypeng/Workspace1/adaptfilter/data/', train=False, download=True, transform=transform_test)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
#     classes = ('plane', 'car', 'bird', 'cat',
#             'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#     if val_set:
#         return trainloader, testloader, valloader, classes
#     else:
#         return trainloader, testloader, classes

# add a main testing function here

def Dataloader_cifar10(train_batch=128, test_batch=100, seed=2024, val_set = False, 
                       datasetpath = '/home/tonypeng/Workspace1/adaptfilter/data/',
                       num_workers = 4):
    # inputs: 
    # train_batch: the batch size for training
    # test_batch: the batch size for testing
    # seed: the random seed for the data loader
    # val_set: a boolean to determine if we want to have a validation set
    # datasetpath: the path to the dataset
    # outputs:
    # trainloader: the data loader for training
    # testloader: the data loader for testing
    # classes: the classes for the dataset
    
    torch.manual_seed(seed)
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=datasetpath, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(
        root=datasetpath, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch, shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes

def Dataloader_cifar10_val(train_batch=128, test_batch=100, seed=2024, val_set = True, datasetpath = '/home/tonypeng/Workspace1/adaptfilter/data/'):
    # inputs: 
    # train_batch: the batch size for training
    # test_batch: the batch size for testing
    # seed: the random seed for the data loader
    # val_set: a boolean to determine if we want to have a validation set
    # datasetpath: the path to the dataset
    # outputs:
    # trainloader: the data loader for training
    # testloader: the data loader for testing
    # classes: the classes for the dataset
    
    torch.manual_seed(seed)
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=datasetpath, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch, shuffle=True, num_workers=8)
    # split the training set into training and validation set
    trainset, valset = torch.utils.data.random_split(trainset, [40000, 10000])
    valloader = torch.utils.data.DataLoader(valset, batch_size=train_batch, shuffle=False, num_workers=8)

    testset = torchvision.datasets.CIFAR10(
        root=datasetpath, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch, shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, valloader, classes

if __name__ == '__main__':
    train, test, classes = Dataloader_cifar10(train_batch=128, test_batch=100, seed=2024)
    for i, data in enumerate(train):
        inputs, labels = data
        break
    # print the size of each loader
    print(len(train))
    print(len(test))
    print('done')

    import numpy as np
    import cv2
    # get 10 pictures from testloader, and the labels
    for i, data in enumerate(test):
        images, labels = data
    
    # print the image and label
    print('label1: ', classes[labels[0]])
    # print image
    img = images[0]
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('done')