# create a sender to the server

import base64
import cv2
import os
import socket
from struct import pack
import sys
import time
from tqdm import tqdm
from PIL import Image

class Sender:
    def __init__(self):
        # host = 'localhost'
        # host = '100.64.0.2'
        # host = '100.64.0.4'
        host = '127.0.0.1'
        port = 5566
        self.host = host
        self.port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.host, self.port))
        
    def sender(self):
        try:
            dataset_root = '/home/tonypeng/Workspace1/adaptfilter/data/'
            # dataset_subroot = ['imagenet-20-jpeg25/', 'imagenet-20-jpeg75/', 'imagenet-20-cjpeg/',
            #                    'imagenet-20-jpeg25-ML/', 'imagenet-20-jpeg75-ML/', 'imagenet-20-cjpeg-ML/',
            #                    'cifar-10-jpeg25/', 'cifar-10-jpeg75/', 'cifar-10-cjpeg/',
            #                     'cifar-10-jpeg25-ML/', 'cifar-10-jpeg75-ML/', 'cifar-10-cjpeg-ML/',
            #                     'imagenet-resnet-gate-emb/', 'cifar-10-mobile-gate-emb/']
            # dataset_subroot = ['cifar-10-mobile-gate-emb/']
            # dataset_subroot = [
            #                    'cifar-10-jpeg25/', 'cifar-10-jpeg75/', 'cifar-10-cjpeg/',
            #                     'cifar-10-jpeg25-ML/', 'cifar-10-jpeg75-ML/', 'cifar-10-cjpeg-ML/',
            #                     'cifar-10-mobile-gate-emb/']
            dataset_subroot = 'imagenet-20-client/'
            ind = [str(x)+'.bmp' for x in range(100)]
            quality = [10, 20, 30, 40, 50, 60, 70, 80, 90]
            for i in range(len(quality)):
                print('quality: ', quality[i])
                for j in range(len(ind)):
                    image_path = dataset_root + dataset_subroot + ind[j]
                    image = cv2.imread(image_path)
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality[i]]
                    result, img_code = cv2.imencode('.jpg', image, encode_param)
                    img_code = img_code.tobytes()
                    img_code = base64.b64encode(img_code)
                    msg_length = pack('>Q', len(img_code))
                    self.s.sendall(msg_length)
                    self.s.sendall(img_code)
                    done = self.s.recv(1)
                    time.sleep(0.01)
                time.sleep(3)

            # dataset_subroot = ['last-imagenet-20-jpeg'+str(x) for x in range(10, 100, 10)]
            # for dataset in dataset_subroot:
            #     dataset = dataset_root + dataset
            #     print('dataset: ', dataset)

            #     file_names = [str(i) for i in range(100)]
            #     files = []
                # for i in range(len(file_names)):
                #     # file_names[i] = Image.open(dataset + '/' + str(i)+'.jpg')
                #     # file_names[i] = file_names[i].convert('RGB')
                #     # # encode
                #     # # file_names[i] = file_names[i].resize((224, 224))
                #     # file_names[i] = file_names[i].tobytes()
                #     # # encode
                #     # file_names[i] = base64.b64encode(file_names[i])
                #     # files.append(file_names[i])
                #     file_names[i] = dataset + '/' + str(i)+'.jpg'

                #     file_names[i] = cv2.imread(dataset + '/' + str(i)+'.jpg')
                #     result, img_code = cv2.imencode('.jpg', file_names[i])
                #     file_names[i] = img_code.tobytes()
                #     files.append(file_names[i])
                
                # for i, file in enumerate(files):
                #     # print(len(file))
                #     # file = open(dataset + '/' + str(i), 'rb')
                #     # file = file.read()
                #     f = cv2.imread(dataset + '/' + str(i)+'.jpg')
                #     result, f = cv2.imencode('.jpg', f[i])
                #     f = f.tobytes()
                #     msg_length = pack('>Q', len(f))
                #     # time.sleep(0.01)
                #     self.s.sendall(msg_length)
                    
                #     self.s.sendall(f)
                #     # ttime1 = time.time()
                    
                    
                #     # wait for the done from the server
                #     done = self.s.recv(1)
                #     # time.sleep(0.01)
                #     # print('send time: ', time.time() - ttime1)

                time.sleep(3)
        except Exception as e:
            print(e)
            self.s.close()

if __name__ == '__main__':
    sender = Sender()
    sender.sender()                                        
    