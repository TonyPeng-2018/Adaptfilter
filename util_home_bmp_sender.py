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
        self.i_stop = 1
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.host, self.port))

    def sender(self):
        try:
            d_path = "/home/tonypeng/Workspace1/adaptfilter/data/"
            i_stop = self.i_stop            
            quality = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            # quality = [100]
            for q in quality:
                for j in tqdm(range(i_stop)):
                    image = cv2.imread(d_path + 'cifar-10-raw-image/' + str(j)+'.bmp')
                    # image = cv2.imread(d_path + 'imagenet-20-raw-image-224/' + str(j)+'.bmp')
                    # image = cv2.imread(d_path + 'ccpd-raw-image/' + str(j)+'.bmp')
                    
                    # if q > 0:
                    # compress q using jpeg
                    send_image = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), q])[1]
                    # write the file 
                    # cv2.imwrite(str(q)+'.jpg', send_image)
                    send_image = send_image.tobytes()
                    send_image = base64.b64encode(send_image)
                    # write 
                    msg_l = pack(">Q", len(send_image))
                    self.s.sendall(msg_l)
                    self.s.sendall(send_image)
                    done = self.s.recv(1)
                    time.sleep(0.01)
                time.sleep(0.01)
            self.s.close()
        except Exception as e:
            print(e)
            self.s.close()


if __name__ == "__main__":
    port = int(sys.argv[1])
    sender = Sender(port)
    sender.sender()
