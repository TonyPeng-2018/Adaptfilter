# receive tcp package from the device
import numpy as np
import socket
from struct import unpack
import time
from tqdm import tqdm
import sys


class Server:
    def __init__(self, host, port):
        # host = 'localhost'
        # host = '100.64.0.2'
        # host = '100.64.0.4'
        self.host = host
        self.port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.s.bind((host, port))
        self.s.listen(10)

    def receiver(self):
        try:
            d_path = "/home/tonypeng/Workspace1/adaptfilter/data/"
            d_subpath = []

            d_subpath = "split_gate_folder_cifar/"

            client_socket, addr = self.s.accept()
            ms = []
            ms.append("mobile")
            ms.append("resnet")
            ms_len = []
            if 'mobile' in ms:
                ms_len.append(6)
            if 'resnet' in ms:
                ms_len.append(7)
            r_time_list = []

            for ind, m in enumerate(ms):                
                for i in range(ms_len[ind]):
                    m_f = m+"_"+str(i)
                    packet_receive_time = 0
                    received_bytes = 0
                    print("Got a connection from %s" % str(addr))
                    data = client_socket.recv(8)  # length of encoded img
                    t1 = time.time()
                    length = unpack(">Q", data)[0]
                    data = b""
                    while len(data) < length:
                        data += client_socket.recv(
                                1024 if length - len(data) > 1024 else length - len(data)
                            )
                    t2 = time.time()
                    client_socket.sendall(b"1")
                    received_bytes = len(data)
                    packet_receive_time = (t2 - t1)*1000
                    r_time_list.append(packet_receive_time)
                    print(m_f , "Packet receive time: ", packet_receive_time, "Received bytes: ", received_bytes)

            self.s.close()
            print(r_time_list)
        except Exception as e:
            print(e)
            self.s.close()

if __name__ == "__main__":
    port = int(sys.argv[1])
    server = Server(port)
    server.receiver()
