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
    def __init__(self, host, port, model, sampler):
        # host = "127.0.0.1"
        self.host = host
        self.port = port
        self.i_stop = 1000
        self.model = model
        self.sampler = sampler
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.host, self.port))

    def sender(self):
        try:
            experiment_root = "data/experiment/"
            threds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ,0.7, 0.8, 0.9, 0.95, 1.0]
            for thred in threds:
                data_path = experiment_root + self.model + "_" + self.sampler + "_" + str(thred) + "/"
                if not os.path.exists(data_path):
                    return 0
                for j in range(self.i_stop):
                    f_path = data_path + str(j) + ".txt"
                    f = open(f_path, "rb")
                    msg = f.read()
                    msg_l = pack(">Q", len(msg))
                    self.s.sendall(msg_l)
                    self.s.sendall(msg)
                    done = self.s.recv(1)
                    time.sleep(0.001)
                time.sleep(1)
            self.s.close()
        except Exception as e:
            print(e)
            self.s.close()


if __name__ == "__main__":
    model = sys.argv[1]   
    sampler = sys.argv[2]
    host = "127.0.0.1"
    port = 5567
    print('running sender')
    sender = Sender(host, port, model, sampler)
    sender.sender()
