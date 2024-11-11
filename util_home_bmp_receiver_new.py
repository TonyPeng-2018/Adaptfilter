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

    def receiver(self, folder, rate):
        # folder 1: 224, 2: raw
        try:
            if folder == 1:
                d_path = "data/jpeg-224/"
            elif folder == 2:
                d_path = "data/jpeg-uncut/"
            client_socket, addr = self.s.accept()

            quality = [1, 3, 5, 7, 9,
                10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            i_stop = len(os.listdir(d_path + str(1)))
            latency = np.zeros((len(quality), i_stop))
            a_time = []
            for ind, q in enumerate(quality):
                packet_receive_time = 0
                received_bytes = 0
                print("Got a connection from %s" % str(addr))
                for i, file in tqdm(enumerate(os.listdir(d_path + str(q)))):

                    t1 = time.time()
                    data = client_socket.recv(8)  # length of encoded img
                    length = unpack(">Q", data)[0]
                    data = b""
                    while len(data) < length:
                        data += client_socket.recv(
                            1024 if length - len(data) > 1024 else length - len(data)
                        )
                    received_bytes += len(data)
                    t2 = time.time()
                    # send received to the client
                    client_socket.sendall(b"1")
                    packet_receive_time += t2 - t1
                    latency[ind, i] = t2 - t1
                #
                packet_receive_time = packet_receive_time * 1000 / i_stop
                received_bytes = received_bytes / i_stop
                received_bytes = np.round(received_bytes, 2)
                packet_receive_time = np.round(packet_receive_time, 2)
                a_time.append(packet_receive_time)
                print(
                    "quality:",
                    q,
                    " avg received bytes:",
                    received_bytes,
                    " avg receive time:",
                    packet_receive_time,
                )
            print('latency ', a_time)
            # save the latency
            np.save(f"latency_{rate}_{folder}.npy", latency)
            self.s.close()
        except Exception as e:
            print(e)
            self.s.close()


if __name__ == "__main__":
    port = int(sys.argv[1])
    folder = int(sys.argv[2])
    rate = sys.argv[3]
    import os
    os.system(f"python3 util_home_bmp_sender_new.py {port} {folder}&")
    server = Server(port)
    server.receiver(folder=folder, rate=rate)
