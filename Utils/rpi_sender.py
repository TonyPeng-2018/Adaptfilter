# create a sender to the server

import base64
import cv2
import os
import socket
from struct import pack
import sys
import time

class Sender:
    def __init__(self, host, port):
        # host = 'home server'
        # port = '5566'
        self.host = host
        self.port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.host, self.port))
        
    def send_jpeg(self, img):
        msg = base64.b64encode(cv2.imencode('.jpg', img)[1].tobytes())
        # calculate the size of the message
        length = pack('>Q', len(msg))
        # send the size of the message
        self.s.sendall(length)
        # send the message
        self.s.sendall(msg)

    def send_encoded(self, msg):
        msg = base64.b64encode(msg)
        # calculate the size of the message
        length = pack('>Q', len(msg))
        # send the size of the message
        self.s.sendall(length)
        # send the message
        self.s.sendall(msg)

    def send_emb(self, message, channel=None):
        # send an image to the server
        # send the channel if 
        # encode channel
        msg1 = base64.b64encode(channel)
        msg2 = base64.b64encode(message)
        # calculate the size of the message
        length1 = pack('>Q', len(msg1))
        length2 = pack('>Q', len(msg2))
        # send the size of the message
        self.s.sendall(length1)
        self.s.sendall(length2)
        self.s.sendall(msg1)
        self.s.sendall(msg2)

    def close(self):
        self.s.close()

if __name__ == '__main__':
    sender = Sender('192.168.1.164', 8080)
    for i in range(600):
        img = cv2.imread('../data/cifar-10-client/'+str(i)+'.bmp')
        time.sleep(1)
        sender.send_jpeg(img)
