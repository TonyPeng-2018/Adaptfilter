# receive tcp package from the device
import base64
import cv2
from Models import mobilenetv2, mobilenetv3, resnet
import numpy as np
import socket
from struct import unpack
import time
import torch


class Server:
    def __init__(self):
        self.host = '192.168.1.164'
        self.port = 8080
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # get the model
        self.c_model, self.s_model = mobilenetv2.mobilenetv2_splitter(num_classes=10, 
                                                    weight_root='/home/tonypeng/Workspace1/adaptfilter/Adaptfilter/Weights/cifar-10', 
                                                    device='cuda:0', partition=-1)
        self.labels_path = '../data/cifar-10-client/labels.txt'
        self.labels = []
        with open(self.labels_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.labels.append(line.strip())

        self.c_model = self.c_model.to('cuda:0')
        self.c_model.eval()

        self.s_model = self.s_model.to('cuda:0')
        self.s_model.eval()

        self.embedding = torch.zeros(1,32,16,16)
        self.s.bind((self.host, self.port))
        self.s.listen(10)

    def receive_img(self, ind):
        while True:
            client_socket, addr = self.s.accept()
            print("Got a connection from %s" % str(addr))
            data = client_socket.recv(8) # length of encoded img
            length = unpack('>Q', data)[0]
            data = b''
            while len(data) < length:
                to_read = length - len(data)
                data += client_socket.recv(4096 if to_read > 4096 else to_read)
            
            img = base64.b64decode(data) # h,w,c
            img = np.frombuffer(img, dtype=np.uint8)
            # jpeg to bmp   
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite('img.jpg', img)
            img = torch.tensor(img).float()
            img = img.view(1,3,32,32)
            img = img.to('cuda:0')

            emb = self.c_model(img)
            out = self.s_model(emb)
            _, pred = torch.max(out, 1)
            pred = pred.cpu().numpy()
            
            return 1 if self.labels[ind] == pred else 0
        

if __name__ == '__main__':
    server = Server()
    result = 0
    for i in range (600):4
        result += server.receive_img(i)
    print(result)