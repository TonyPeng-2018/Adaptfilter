# resnet on imagenet speed
import torch

img = torch.randn(1, 3, 224, 224).cuda()
from Models import resnet

model = resnet.resnet50(num_classes=1000).cuda()
model.eval()
import time

with torch.no_grad():
    for i in range(600):
        model(img)
        if i == 100:
            time_start = time.time()
    time_end = time.time()
    print("time cost", (time_end - time_start) / 500 * 1000, "ms")
