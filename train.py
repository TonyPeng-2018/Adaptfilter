# This file is for trainning
# Run this on the server, or as we called offline. 

from Dataloaders.dataloader_cifar10 import Dataloader_cifar10
import argparse



def main(dataset_name, model_name):
    # initial using mobilenetV2, and cifar10
    # we need a if statement here to decide which model and dataset to use
    random_seed = 2024

    # for training, it is for training the generator 
    # recall the graph, when we cut more features, the performance should be worse.
    
    # get the training loader, freeze the model. Where is the partitioning point? 

    # 1. get the train, test and val datasets, and labels.
    train, test, val, labels = some_dataset_function()
    
    # 2. transfer the dataset to fit the model, for the training, client and server model are all on the server
    client_model = some_model_function()

    # 3. get the gating, gating here decides how many channels are transferred to the server
    # simple version: a binary tree, complex version: model
    gating = some_gating_function()

    # 4. get the reducer, 
    reducer = some_reducer_funtion()

    # 5. get the generator
    generator = some_generator_funtion()

    # 6. get the server 
    server_model = some_model_function()
    # pipline data -> dataloader -> client_model -> gating -> reducer -> generator -> server_model
    
    


if '__name__' == '__main__':
    parser = argparse.ArgumentParser()
    # we need the name of model, the name of dataset
    parser.add_argument('--dataset', type=str, default='cifar10', help='name of dataset')
    parser.add_argument('--iot_model', type=str, default='mobilenetV2', help='name of the model on the iot')
    parser.add_argument('--reducer', type=str, default='entrophy', help='name of the reducer')
    parser.add_argument('--client', type=str, default='LTE', help='name of the network condition on the client side')
    parser.add_argument('--server', type=str, default='LTE', help='name of the network condition on the server side')
    parser.add_argument('--generator', type=str, default='None', help='name of the generator')
    parser.add_argument('--server_model', type=str, default='mobilenetV2', help='name of the model on the server, should be the same as it on the iot')
    parser.add_argument('--device', type=str, default='home', help='run on which device, home, tintin, rpi, pico, jetson?')

    args = parser.parse_args()
    print(args.echo())
    main(args)
    
