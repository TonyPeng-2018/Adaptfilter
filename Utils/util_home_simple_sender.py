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
    def __init__(self, host, port):
        # host = 'home server'
        # port = '5566'
        self.host = host
        self.port = port
        # self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.s.connect((self.host, self.port))
        
    def send_jpeg(self, fp):
        img = cv2.imread(fp)
        t1 = time.time()
        msg = base64.b64encode(cv2.imencode('.jpg', img)[1].tobytes())
        t2 = time.time()
        # print('img encode time: ', t2 - t1)
        # calculate the size of the message
        # self.s.connect((self.host, self.port))
        length = pack('>Q', len(msg))
        # send the size of the message
        self.s.sendall(length)
        # send the message
        self.s.sendall(msg)

    def send_bmp(self, batch, dataset):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.host, self.port))
        for i in tqdm(range(batch)):
            fp = '../data/' + dataset + '-client/%s.bmp' % str(i)
            img = cv2.imread(fp)
            t1 = time.time()
            msg = base64.b64encode(cv2.imencode('.bmp', img)[1].tobytes())
            
            # print('img encode time: ', t2 - t1)
            # calculate the size of the message
            # self.s.connect((self.host, self.port))
            length = pack('>Q', len(msg))
            # send the size of the message
            self.s.sendall(length)
            # send the message
            self.s.sendall(msg)
            t2 = time.time()
            time.sleep(0.1)

    def send_emb(self, fp1, fp2):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.host, self.port))

        for i in tqdm(range(10)):
            with open(fp1, 'rb') as f:
                msg = f.readline()
            length = pack('>Q', len(msg))
            # send the size of the message
            self.s.sendall(length)
            # send the message
            self.s.sendall(msg)
            with open(fp2, 'rb') as f:
                msg = f.readline()
            length = pack('>Q', len(msg))
            # send the size of the message
            self.s.sendall(length)
            # send the message
            self.s.sendall(msg)
            time.sleep(0.1)

    # def send_emb(self, message, channel=None):
    #     # send an image to the server
    #     # send the channel if 
    #     # encode channel
    #     msg1 = base64.b64encode(channel)
    #     msg2 = base64.b64encode(message)
    #     # calculate the size of the message
    #     length1 = pack('>Q', len(msg1))
    #     length2 = pack('>Q', len(msg2))
    #     # send the size of the message
    #     self.s.sendall(length1)
    #     self.s.sendall(length2)
    #     self.s.sendall(msg1)
    #     self.s.sendall(msg2)

    def close(self):
        self.s.close()

    def send_multiple_imgs(self, dataset, batch=100, cr=19):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.host, self.port))

        for i in tqdm(range(batch)):
            for j in range(cr):
                fp = '../data/' + dataset + '-client/%s_%s.jpg' % (str(i), str((j+1)*5))
                img = cv2.imread(fp)
                t1 = time.time()
                msg = base64.b64encode(cv2.imencode('.jpg', img)[1].tobytes())

                # print('img encode time: ', t2 - t1)
                # calculate the size of the message
                # self.s.connect((self.host, self.port))
                length = pack('>Q', len(msg))
                # send the size of the message
                self.s.sendall(length)
                # send the message
                self.s.sendall(msg)
                t2 = time.time()
                time.sleep(0.1)

if __name__ == '__main__':
    # sender = Sender('192.168.1.164', 8080)
    sender = Sender('100.64.0.2', 5566)
    #sender.send_jpeg('cifar_c.jpg')
    #sender.send_emb('c1_gate2_cifar-10_uint8', 'c2_gate2_cifar-10')
    # sender.send_multiple_imgs(dataset = 'imagenet', batch = 10)
    sender.send_bmp(batch = 100, dataset = 'cifar-10')
    sender.close()
    