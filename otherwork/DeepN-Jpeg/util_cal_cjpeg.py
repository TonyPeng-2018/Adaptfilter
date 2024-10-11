import subprocess
import time 

t1 = time.time()
command = 'cjpeg -dct int -qtable q_table_cifar.txt -baseline -opt -outfile cifar.jpg cifar.bmp'
subprocess.run(command, shell=True)
t2 = time.time()
print('Time taken for cjpeg cifar simple: ', t2 - t1)

t1 = time.time()
command = 'cjpeg -dct int -qtable q_table_imagenet.txt -baseline -opt -outfile imagenet.jpg imagenet.bmp'
subprocess.run(command, shell=True)
t2 = time.time()
print('Time taken for cjpeg imagenet simple: ', t2 - t1)

t1 = time.time()
command = 'cjpeg -dct int -qtable q_table_cifar.txt -baseline -opt -progressive -outfile cifar_c.jpg cifar.bmp'
subprocess.run(command, shell=True)
t2 = time.time()
print('Time taken for cjpeg cifar progressive: ', t2 - t1)

t1 = time.time()
command = 'cjpeg -dct int -qtable q_table_imagenet.txt -baseline -opt -progressive -outfile imagenet_c.jpg imagenet.bmp'
subprocess.run(command, shell=True)
t2 = time.time()
print('Time taken for cjpeg imagenet progressive: ', t2 - t1)