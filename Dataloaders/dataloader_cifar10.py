# this file is a data loader for cifar 10. 
# the root path is /data3/anp407/cifar-10-batches-py
# inspired by https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html?highlight=cifar

import torch
import torchvision
import torchvision.transforms as transforms

def Dataloader_cifar10(batch_size, seed):
    torch.manual_seed(seed)
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='/data3/anp407', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='/data3/anp407', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes