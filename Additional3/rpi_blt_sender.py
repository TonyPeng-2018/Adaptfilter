# import bluetooth

# def sendMessageTo(targetBluetoothMacAddress):
#   port = 3
#   sock=bluetooth.BluetoothSocket( bluetooth.RFCOMM )
#   sock.connect((targetBluetoothMacAddress, 3))
#   sock.send(b"hello!!")
#   sock.close()

# sendMessageTo('2C:CF:67:1C:70:9B')

import socket
import time
import os

def sendMessageTo(serverMACAddress, fpath):

    serverMACAddress = serverMACAddress
    f = open(fpath, 'rb')
    lines = f.readlines()[0]
    port = 1
    s = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    s.connect((serverMACAddress,port))
    for i in range(5):
        print(i)
        t1 = time.time()
        s.send(lines)
        t2 = time.time()
        print('send time:', t2-t1)
    s.send(b'end')

    time.sleep(100)

ind = 14
data_root = '../data/blt/'
filepath = os.listdir(data_root)
filepath = sorted(filepath)[ind]
print(filepath)
sendMessageTo('2C:CF:67:1C:70:9B', data_root+filepath)

# import time
# import os

# if __name__ == '__main__':
#     ind = 14
#     data_root = '../data/blt/'
#     filepath = os.listdir(data_root)
#     filepath = sorted(filepath)[ind]
#     print(filepath)
#     fpath = data_root+filepath

#     serverMACAddress = '2C:CF:67:1C:70:9B'
#     sock=bluetooth.BluetoothSocket( bluetooth.RFCOMM )
#     port = 3
#     sock.connect((serverMACAddress, port)) #连接蓝牙

#     f = open(fpath, 'rb')
#     lines = f.readlines()[0]
#     for i in range(5):
#         print(i)
#         sock.send(lines)
#     sock.send(b'end')

    # time.sleep(100)

# import bluetooth
# import time
# import os

# target_address='2C:CF:67:1C:70:9B'	#目的蓝牙的地址
# sock=bluetooth.BluetoothSocket( bluetooth.RFCOMM )
# port = 1
# if __name__ =='__main__':
#     try:
#         ind = 14

#         sock.connect((target_address, port)) #连接蓝牙
#         data_root = '../data/blt/'
#         filepath = os.listdir(data_root)
#         filepath = sorted(filepath)[ind]
#         print(filepath)
#         fpath = data_root+filepath
#         f = open(fpath, 'rb')
#         lines = f.readlines()[0]
#         for i in range(5):
#             print(i)
#             sock.send(lines) #每隔三秒发送一个字符串
#         sock.send(b'end')
#         time.sleep(100)
#     except Exception as e:
#         print(e)
#         sock.close()
