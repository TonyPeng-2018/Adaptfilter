# create a sender to the server

import base64
import cv2
import os
import socket
from struct import pack
import sys
import time
from tqdm import tqdm
from PIL import Image


class Sender:
    def __init__(self, port):
        host = "127.0.0.1"
        self.host = host
        self.port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.host, self.port))

    def sender(self):
        try:
            d_path = "/home/tonypeng/Workspace1/adaptfilter/data/"
            d_subpath = []
            

            d_subpath = "split_gate_folder/"
            # quality = [10, 20, 30, 40, 50, 60, 70, 80, 90]
            m1 = 'mobile'
            m2 = 'resnet'

            for i in range(5):
                m_f = m1+"_"+str(i)
                f = open(d_path+d_subpath+m_f, "rb")
                msg = f.read()
                msg_l = pack(">Q", len(msg))
                self.s.sendall(msg_l)
                self.s.sendall(msg)
                done = self.s.recv(1)
                time.sleep(0.01)

            for i in range(6):
                m_f = m2+"_"+str(i)
                f = open(d_path+d_subpath+m_f, "rb")
                msg = f.read()
                msg_l = pack(">Q", len(msg))
                self.s.sendall(msg_l)
                self.s.sendall(msg)
                done = self.s.recv(1)
                time.sleep(0.01)

            self.s.close()
        except Exception as e:
            print(e)
            self.s.close()


if __name__ == "__main__":
    port = int(sys.argv[1])
    sender = Sender(port)
    sender.sender()
