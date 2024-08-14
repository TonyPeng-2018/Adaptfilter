# create a sender to the server

import base64
import cv2
import os
import socket
from struct import pack
import sys
import time
from tqdm import tqdm

class Sender:
    def __init__(self):
        # host = 'localhost'
        # host = '100.64.0.2'
        # host = '100.64.0.4'
        host = '127.0.0.1'
        port = 5568
        self.host = host
        self.port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.host, self.port))
        
    def sender(self):
        try:
            dataset_root = '/home/tonypeng/Workspace1/adaptfilter/data/'
            dataset_subroot = ['imagenet-20-jpeg25/', 'imagenet-20-jpeg75/', 'imagenet-20-cjpeg/']
            #                    'imagenet-20-jpeg25-ML/', 'imagenet-20-jpeg75-ML/', 'imagenet-20-cjpeg-ML/',
            #                    'cifar-10-jpeg25/', 'cifar-10-jpeg75/', 'cifar-10-cjpeg/',
            #                     'cifar-10-jpeg25-ML/', 'cifar-10-jpeg75-ML/', 'cifar-10-cjpeg-ML/',
            #                     'imagenet-resnet-gate-emb/', 'cifar-10-mobile-gate-emb/']
            # dataset_subroot = ['cifar-10-mobile-gate-emb/']
            # dataset_subroot = [
            #                    'cifar-10-jpeg25/', 'cifar-10-jpeg75/', 'cifar-10-cjpeg/',
            #                     'cifar-10-jpeg25-ML/', 'cifar-10-jpeg75-ML/', 'cifar-10-cjpeg-ML/',
            #                     'cifar-10-mobile-gate-emb/']
            # dataset_subroot = ['imagenet-resnet-gate-emb-0.55/', 'imagenet-resnet-gate-emb-0.65/',
            #                    'imagenet-resnet-gate-emb-0.75/', 'imagenet-resnet-gate-emb-0.85/',
            #                    'imagenet-resnet-gate-emb-0.95/', 'imagenet-resnet-gate-emb-0.99/']
            # dataset_subroot = ['imagenet-resnet-gate-emb-99/']
            # dataset_subroot = ['ccpd-jpeg25/', 'ccpd-jpeg75/', 'ccpd-cjpeg/',
            #                    'ccpd-jpeg25-ML/', 'ccpd-jpeg75-ML/', 'ccpd-cjpeg-ML/']
            # dataset_subroot = ['ccpd-client']
                            #    'ccpd-resnet-gate-emb/']
            for dataset in dataset_subroot:
                dataset = dataset_root + dataset
                print(dataset)

                file_names = [str(i) for i in range(100)]
                files = [open(dataset + '/' + file_name, 'rb') for file_name in file_names]
                files = [file.read() for file in files]
                
                for i, file in enumerate(files):
                    # print(len(file))
                    # file = open(dataset + '/' + str(i), 'rb')
                    # file = file.read()
                    msg_length = pack('>Q', len(file))
                    self.s.sendall(msg_length)
                    
                    self.s.sendall(file)
                    ttime1 = time.time()
                    
                    
                    # wait for the done from the server
                    done = self.s.recv(1)
                    # print('send time: ', time.time() - ttime1)

                time.sleep(3)
        except Exception as e:
            print(e)
            self.s.close()

if __name__ == '__main__':
    sender = Sender()
    sender.sender()                                        
    