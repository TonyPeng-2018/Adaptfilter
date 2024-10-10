# receive tcp package from the device
import numpy as np
import socket
from struct import unpack
import time
from tqdm import tqdm
import sys


class Server:
    def __init__(self, port):
        # host = 'localhost'
        # host = '100.64.0.2'
        # host = '100.64.0.4'
        host = "127.0.0.1"
        self.i_stop = 600
        self.port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.s.bind((host, port))
        self.s.listen(10)

    def receiver(self):
        try:
            d_path = "/home/tonypeng/Workspace1/adaptfilter/data/"
            d_subpath = []

            # d_subpath.append('imagenet-20-jpeg25/')
            # d_subpath.append('imagenet-20-jpeg75/')
            # d_subpath.append('imagenet-20-cjpeg/')
            # d_subpath.append('imagenet-20-jpeg25-ML/')
            # d_subpath.append('imagenet-20-jpeg75-ML/')
            # d_subpath.append('imagenet-20-cjpeg-ML/')
            # d_subpath.append('imagenet-mobile-gate-emb/')
            # d_subpath.append('imagenet-resnet-gate-emb/')
            # d_subpath.append('imagenet-20-raw-encode-224/')

            # d_subpath.append('cifar-10-jpeg25/')
            # d_subpath.append('cifar-10-jpeg75/')
            # d_subpath.append('cifar-10-cjpeg/')
            # d_subpath.append('cifar-10-jpeg25-ML/')
            # d_subpath.append('cifar-10-jpeg75-ML/')
            # d_subpath.append('cifar-10-cjpeg-ML/')
            # d_subpath.append('cifar-10-mobile-gate-emb/')
            # d_subpath.append('cifar-10-resnet-gate-emb/')
            # d_subpath.append('cifar-10-raw-encode/')

            # d_subpath.append('ccpd-jpeg25/')
            # d_subpath.append('ccpd-jpeg75/')
            # d_subpath.append('ccpd-cjpeg/')
            # d_subpath.append('ccpd-jpeg25-ML/')
            # d_subpath.append('ccpd-jpeg75-ML/')
            # d_subpath.append('ccpd-cjpeg-ML/')
            # d_subpath.append('ccpd-mobile-gate-emb/')
            # d_subpath.append('ccpd-resnet-gate-emb/')
            # d_subpath.append('ccpd-raw-encode/')

            # d_subpath.append("imagenet-20-mobile-conf-0.55/")
            # d_subpath.append("imagenet-20-mobile-conf-0.65/")
            # d_subpath.append("imagenet-20-mobile-conf-0.75/")
            # d_subpath.append("imagenet-20-mobile-conf-0.85/")
            # d_subpath.append("imagenet-20-mobile-conf-0.95/")
            # d_subpath.append("imagenet-20-mobile-conf-0.99/")

            # d_subpath.append("imagenet-20-resnet-conf-0.55/")
            # d_subpath.append("imagenet-20-resnet-conf-0.65/")
            # d_subpath.append("imagenet-20-resnet-conf-0.75/")
            # d_subpath.append("imagenet-20-resnet-conf-0.85/")
            # d_subpath.append("imagenet-20-resnet-conf-0.95/")
            # d_subpath.append("imagenet-20-resnet-conf-0.99/")

            i_stop = self.i_stop

            client_socket, addr = self.s.accept()
            for ds in d_subpath:
                packet_receive_time = 0
                received_bytes = 0
                print("Got a connection from %s" % str(addr))
                for i in tqdm(range(i_stop)):
                    t1 = time.time()
                    data = client_socket.recv(8)  # length of encoded img
                    length = unpack(">Q", data)[0]
                    data = b""
                    while len(data) < length:
                        data += client_socket.recv(
                            1024 if length - len(data) > 1024 else length - len(data)
                        )
                    received_bytes += len(data)
                    if "raw" not in ds:
                        data = client_socket.recv(8)  # length of encoded img
                        length = unpack(">Q", data)[0]
                        data = b""
                        while len(data) < length:
                            data += client_socket.recv(
                                1024
                                if length - len(data) > 1024
                                else length - len(data)
                            )

                        received_bytes += len(data)
                    t2 = time.time()
                    # send received to the client
                    client_socket.sendall(b"1")
                    packet_receive_time += t2 - t1
                #
                packet_receive_time = packet_receive_time * 1000 / i_stop
                received_bytes = received_bytes / i_stop
                received_bytes = np.round(received_bytes, 2)
                packet_receive_time = np.round(packet_receive_time, 2)
                print(
                    ds,
                    " avg received bytes:",
                    received_bytes,
                    " avg receive time:",
                    packet_receive_time,
                )
            self.s.close()
        except Exception as e:
            print(e)
            self.s.close()


if __name__ == "__main__":
    port = int(sys.argv[1])
    server = Server(port)
    server.receiver()
