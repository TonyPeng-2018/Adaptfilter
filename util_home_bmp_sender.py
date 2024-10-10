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
        self.i_stop = 5
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.host, self.port))

    def sender(self):
        try:
            d_path = "/home/tonypeng/Workspace1/adaptfilter/data/"
            i_stop = self.i_stop
            image = cv2.cvtColor(cv2.imread(d_path + 'imagenet-20-raw-image-224/' + '0.bmp'), cv2.COLOR_BGR2RGB)
            quality = [10, 20, 30, 40, 50, 60, 70, 80, 90]
            for q in quality:
                # if q > 0:
                if q > 0:
                    # compress q using jpeg
                    send_image = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), q])[1]
                    send_image = send_image.tobytes()
                    send_image = base64.b64encode(send_image)
                else:
                    send_image = cv2.imencode('.bmp', image)[1]
                    send_image = send_image.tobytes()
                    send_image = base64.b64encode(send_image)
                for j in range(i_stop):
                    # send 5 images and get the average time
                    msg_l = pack(">Q", len(send_image))
                    self.s.sendall(msg_l)
                    self.s.sendall(send_image)
                    done = self.s.recv(1)
                    time.sleep(0.01)
                time.sleep(1)
            self.s.close()
        except Exception as e:
            print(e)
            self.s.close()


if __name__ == "__main__":
    port = int(sys.argv[1])
    sender = Sender(port)
    sender.sender()
