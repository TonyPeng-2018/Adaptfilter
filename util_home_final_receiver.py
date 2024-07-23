# receive tcp package from the device
import numpy as np
import socket
from struct import unpack
import time
from tqdm import tqdm


class Server:
    def __init__(self):
        # host = 'localhost'
        # host = '100.64.0.2'
        # host = '100.64.0.4'
        host = '100.64.0.1'
        port = 5566
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.s.bind((host, port))
        self.s.listen(10)

    
    def receiver(self):
        try:
            f = open('WiFi-Result', 'w')
            # dataset_subroot = ['imagenet-20-jpeg25/', 'imagenet-20-jpeg75/', 'imagenet-20-cjpeg/',
            #                    'imagenet-20-jpeg25-ML/', 'imagenet-20-jpeg75-ML/', 'imagenet-20-cjpeg-ML/',
            #                    'cifar-10-jpeg25/', 'cifar-10-jpeg75/', 'cifar-10-cjpeg/',
            #                     'cifar-10-jpeg25-ML/', 'cifar-10-jpeg75-ML/', 'cifar-10-cjpeg-ML/',
            #                     'imagenet-resnet-gate-emb/', 'cifar-10-mobile-gate-emb/']
            dataset_subroot = [
                               'cifar-10-jpeg25/', 'cifar-10-jpeg75/', 'cifar-10-cjpeg/',
                                'cifar-10-jpeg25-ML/', 'cifar-10-jpeg75-ML/', 'cifar-10-cjpeg-ML/',
                                'cifar-10-mobile-gate-emb/']
            client_socket, addr = self.s.accept()
            for ds in dataset_subroot:
                packet_receive_time = [0] * 600
                received_bytes = []
                print("Got a connection from %s" % str(addr))
                for i in tqdm(range(600)):
                    data = client_socket.recv(8) # length of encoded img
                    t1 = time.time()
                    length = unpack('>Q', data)[0]
                    data = b''
                    while len(data) < length:
                        data += client_socket.recv(1024 if length - len(data) > 1024 else length - len(data))
                    t2 = time.time()
                    received_bytes.append(len(data))
                    # send received to the client
                    client_socket.sendall(b'1')
                    packet_receive_time[i] = t2 - t1
                # print the average time for each box
                packet_box = [0]*10
                print(np.mean(np.array(received_bytes)))
                for i in range(10):
                    packet_box[i] = np.mean(packet_receive_time[i*60:(i+1)*60])*1000
                    # set four decimal places
                    packet_box[i] = round(packet_box[i], 4)
                print(ds, packet_box)
                # remove [ ] in packet box when write
                pb = str(packet_box)
                pb = pb.replace('[', '').replace(']', '').replace(' ', '')
                
                f.write(ds[:-1] + ',' + str(packet_box) + '\n')
        except Exception as e:
            print(e)
            self.s.close()

if __name__ == '__main__':
    server = Server()
    server.receiver()