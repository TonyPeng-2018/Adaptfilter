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
        self.i_stop = 600
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.host, self.port))

    def sender(self):
        try:
            d_path = "/home/tonypeng/Workspace1/adaptfilter/data/"
            d_subpath = []

            # d_subpath.append('imagenet-20-jpeg25/')
            # d_subpath.append('imagenet-20-jpeg75/')
            # d_subpath.append('imagenet-20-cjpeg/')
            # d_subpath.append('imagenet-20-jpeg25-ML/')
            # d_subpath.append('imagenet-20-jpeg75-ML/')
            # d_subpath.append('imagenet-20-cjpeg-ML/')
            # d_subpath.append('imagenet-20-mobile-gate-emb/')
            # d_subpath.append('imagenet-20-resnet-gate-emb/')
            # d_subpath.append('imagenet-20-raw-encode/')

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

            d_subpath.append("imagenet-20-mobile-conf-0.55/")
            d_subpath.append("imagenet-20-mobile-conf-0.65/")
            d_subpath.append("imagenet-20-mobile-conf-0.75/")
            d_subpath.append("imagenet-20-mobile-conf-0.85/")
            d_subpath.append("imagenet-20-mobile-conf-0.95/")
            d_subpath.append("imagenet-20-mobile-conf-0.99/")

            d_subpath.append("imagenet-20-resnet-conf-0.55/")
            d_subpath.append("imagenet-20-resnet-conf-0.65/")
            d_subpath.append("imagenet-20-resnet-conf-0.75/")
            d_subpath.append("imagenet-20-resnet-conf-0.85/")
            d_subpath.append("imagenet-20-resnet-conf-0.95/")
            d_subpath.append("imagenet-20-resnet-conf-0.99/")

            i_stop = self.i_stop
            # quality = [10, 20, 30, 40, 50, 60, 70, 80, 90]
            for ds in d_subpath:
                for j in range(i_stop):
                    f_path = d_path + ds + str(j)
                    f = open(f_path, "rb")
                    msg = f.read()
                    msg_l = pack(">Q", len(msg))
                    self.s.sendall(msg_l)
                    self.s.sendall(msg)

                    if "raw" not in ds:
                        f2 = d_path + ds + str(j) + "_h"
                        f2 = open(f2, "rb")
                        msg2 = f2.read()
                        msg_l2 = pack(">Q", len(msg2))
                        self.s.sendall(msg_l2)
                        self.s.sendall(msg2)
                    done = self.s.recv(1)
                    time.sleep(0.01)

                # d_subpath = ['last-imagenet-20-jpeg'+str(x) for x in range(10, 100, 10)]
                # for d in d_subpath:
                #     d = d_path + d
                #     print('d: ', d)

                #     file_names = [str(i) for i in range(100)]
                #     files = []
                # for i in range(len(file_names)):
                #     # file_names[i] = Image.open(d + '/' + str(i)+'.jpg')
                #     # file_names[i] = file_names[i].convert('RGB')
                #     # # encode
                #     # # file_names[i] = file_names[i].resize((224, 224))
                #     # file_names[i] = file_names[i].tobytes()
                #     # # encode
                #     # file_names[i] = base64.b64encode(file_names[i])
                #     # files.append(file_names[i])
                #     file_names[i] = d + '/' + str(i)+'.jpg'

                #     file_names[i] = cv2.imread(d + '/' + str(i)+'.jpg')
                #     result, img_code = cv2.imencode('.jpg', file_names[i])
                #     file_names[i] = img_code.tobytes()
                #     files.append(file_names[i])

                # for i, file in enumerate(files):
                #     # print(len(file))
                #     # file = open(d + '/' + str(i), 'rb')
                #     # file = file.read()
                #     f = cv2.imread(d + '/' + str(i)+'.jpg')
                #     result, f = cv2.imencode('.jpg', f[i])
                #     f = f.tobytes()
                #     msg_length = pack('>Q', len(f))
                #     # time.sleep(0.01)
                #     self.s.sendall(msg_length)

                #     self.s.sendall(f)
                #     # ttime1 = time.time()

                #     # wait for the done from the server
                #     done = self.s.recv(1)
                #     # time.sleep(0.01)
                #     # print('send time: ', time.time() - ttime1)

                time.sleep(3)
            self.s.close()
        except Exception as e:
            print(e)
            self.s.close()


if __name__ == "__main__":
    port = int(sys.argv[1])
    sender = Sender(port)
    sender.sender()
