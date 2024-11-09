import torch

# 1 layer cnn
in_ch = 256
s_h, s_w = 56*64//in_ch, 56*64//in_ch
pyramid = []

cur_ch = in_ch
while cur_ch != 1:
    pyramid.append(torch.nn.Conv2d(in_channels=cur_ch, out_channels=cur_ch//2, kernel_size=3, stride=1, padding=1))
    pyramid.append(torch.nn.BatchNorm2d(cur_ch//2))
    pyramid.append(torch.nn.ReLU())
    cur_ch = cur_ch // 2

image = torch.randn(1, in_ch, s_h, s_w)
pyramid = torch.nn.Sequential(*pyramid)
import time

from tqdm import tqdm

t1 = time.time()
for i in tqdm(range(10)):
    out = pyramid(image)
t2 = time.time()
print("Time taken: ", (t2 - t1)/10)
