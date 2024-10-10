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
        self.port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.s.bind((host, port))
        self.s.listen(10)

    def receiver(self):
        try:
            d_path = "/home/tonypeng/Workspace1/adaptfilter/data/"
            d_subpath = []

            d_subpath = "split_gate_folder2/"

            client_socket, addr = self.s.accept()
            m1 = 'mobile'
            m2 = 'resnet'
            r_time_list = []

            for i in range(6):
                if i <5 :
                    continue
                m_f = m1+"_"+str(i)
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
            
            for i in range(7):
                if i <6 :
                    continue
                m_f = m2+"_"+str(i)
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
