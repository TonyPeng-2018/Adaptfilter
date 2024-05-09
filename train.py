# This file is for trainning
# Run this on the server, or as we called offline. 

from Dataloaders.dataloader_cifar10 import Dataloader_cifar10
import argparse



def main(dataset_name, model_name):
    # initial using mobilenetV2, and cifar10
    # we need a if statement here to decide which model and dataset to use
    random_seed = 2024
    trainloader, testloader, classes = Dataloader_cifar10()
    # load the model
    model = mobilenetV2()

if '__name__' == '__main__':
    parser = argparse.ArgumentParser()
    # we need the name of model, the name of dataset
    parser.add_argument('--model', type=str, default='mobilenetV2', help='name of model')
    parser.add_argument('--dataset', type=str, default='cifar10', help='name of dataset')
    args = parser.parse_args()
    print(args.echo())

    main(args.dataset, args.model)
    
