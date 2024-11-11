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
    def __init__(self, port):
        host = "127.0.0.1"
        self.host = host
        self.port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.host, self.port))

    def sender(self, folder):
        # folder 1 : 224, 2: raw
        try:
            if folder == 1:
                d_path = "data/jpeg-224/"
            elif folder == 2:
                d_path = "data/jpeg-uncut/"            
            quality = [1, 3, 5, 7, 9,
                       10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            # quality = [100]
            for q in quality:
                for j, file in tqdm(enumerate(os.listdir(d_path + str(q)))):
                    image = cv2.imread(d_path + str(q) + '/' + file)
                    # print('path', d_path + str(q) + '/' + str(j)+'.jpg')
                    # print('image', image)
                    send_image = cv2.imencode('.jpg', image)[1]
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
    folder = int(sys.argv[2])
    time.sleep(1)
    sender = Sender(port)
    sender.sender(folder)
