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
        host = '127.0.0.1'
        port = 5568
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.s.bind((host, port))
        self.s.listen(10)

    
    def receiver(self):
        try:
            f = open('WiFi-Result', 'w')
            dataset_subroot = ['imagenet-20-jpeg25/', 'imagenet-20-jpeg75/', 'imagenet-20-cjpeg/']
            #                    'imagenet-20-jpeg25-ML/', 'imagenet-20-jpeg75-ML/', 'imagenet-20-cjpeg-ML/',
            #                    'cifar-10-jpeg25/', 'cifar-10-jpeg75/', 'cifar-10-cjpeg/',
            #                     'cifar-10-jpeg25-ML/', 'cifar-10-jpeg75-ML/', 'cifar-10-cjpeg-ML/',
            #                     'imagenet-resnet-gate-emb/', 'cifar-10-mobile-gate-emb/']
            # dataset_subroot = [
            #                    'cifar-10-jpeg25/', 'cifar-10-jpeg75/', 'cifar-10-cjpeg/',
            #                     'cifar-10-jpeg25-ML/', 'cifar-10-jpeg75-ML/', 'cifar-10-cjpeg-ML/',
            #                     'cifar-10-mobile-gate-emb/']
            # dataset_subroot = ['last-imagenet-20-jpeg'+str(x) for x in range(10, 100, 10)]
            # dataset_subroot = ['ccpd-jpeg25/', 'ccpd-jpeg75/', 'ccpd-cjpeg/',
            #                    'ccpd-jpeg25-ML/', 'ccpd-jpeg75-ML/', 'ccpd-cjpeg-ML/']
            # dataset_subroot = ['ccpd-client']

            client_socket, addr = self.s.accept()
            for ds in dataset_subroot:
                packet_receive_time = 0
                received_bytes = 0
                print("Got a connection from %s" % str(addr))
                for i in tqdm(range(100)):
                    data = client_socket.recv(8) # length of encoded img
                    t1 = time.time()
                    length = unpack('>Q', data)[0]
                    data = b''
                    while len(data) < length:
                        data += client_socket.recv(1024 if length - len(data) > 1024 else length - len(data))
                    t2 = time.time()
                    received_bytes += len(data)
                    # send received to the client
                    client_socket.sendall(b'1')
                    packet_receive_time += t2 - t1
                    print(t2-t1)
                #
                packet_receive_time = packet_receive_time * 1000
                packet_receive_time = packet_receive_time / 100
                received_bytes = received_bytes / 100
                received_bytes = np.round(received_bytes, 2)
                packet_receive_time = np.round(packet_receive_time, 2)
                print('Average received bytes:', received_bytes)
                print('Average packet receive time:', packet_receive_time)
                self.s.close()
        except Exception as e:
            print(e)
            self.s.close()

if __name__ == '__main__':
    server = Server()
    server.receiver()