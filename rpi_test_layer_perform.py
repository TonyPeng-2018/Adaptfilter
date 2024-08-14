# this is for measuring the latency and bandwdith of the network
from Models import mobilenetv2, resnet
import torch
m_cifar = mobilenetv2.MobileNetV2_layers(num_classes=10)
m_imagenet = mobilenetv2.MobileNetV2_layers(num_classes=1000)
r_cifar = resnet.resnet_layers(num_classes=10, layers=[3, 4, 6, 3])
r_imagenet = resnet.resnet_layers(num_classes=1000, layers=[3, 4, 6, 3])

import torchsummary
torchsummary.summary(m_cifar, (3, 32, 32), device='cpu')
torchsummary.summary(m_imagenet, (3, 224, 224), device='cpu')
torchsummary.summary(r_cifar, (3, 32, 32), device='cpu')
torchsummary.summary(r_imagenet, (3, 224, 224), device='cpu')

import matplotlib.pyplot as plt
import numpy as np

# plot m for cifar
# plot r for imagenet

m_cifar_lat = np.array(m_cifar.infertime) * 1000
m_cifar_bw = np.array(m_cifar.infermemory) / 1024
m_cifar_lat_total = m_cifar_lat[-1]
m_cifar_bw_total = 32*32*3*4/1024

r_imagenet_lat = np.array(r_imagenet.infertime) * 1000
r_imagenet_bw = np.array(r_imagenet.infermemory) / 1024
r_imagenet_lat_total = r_imagenet_lat[-1]
r_imagenet_bw_total = 224*224*3*4/1024

# subplots

fig, axs = plt.subplots(2,1)
m_x = np.arange(len(m_cifar_lat))
r_x = np.arange(len(r_imagenet_lat))

axs[0].bar(m_x-0.2, m_cifar_lat, color='b',width=0.4,alpha=0.5, label='Latency')
# plot a line for the total latency
axs[0].axhline(y=m_cifar_lat_total, color='g', linestyle='--',label='Total latency')
axs[0].set_xlabel('Layer')
axs[0].set_ylabel('Latency(ms)')
axs[0].set_xticks(m_x)
axs[0].set_xticklabels(m_x)
axs_t = axs[0].twinx()
axs_t.bar(m_x+0.2, m_cifar_bw, color='orange',width=0.4,alpha=0.5, label='Intermediate size')
axs_t.axhline(y=m_cifar_bw_total, color='r', linestyle='-.',label='Image size')
axs_t.set_ylabel('Size(KB)')
axs[0].legend()
axs_t.legend()

axs[1].bar(r_x-0.2, r_imagenet_lat, color='b',width=0.4,alpha=0.5, label='Latency')
# plot a line for the total latency
axs[1].axhline(y=r_imagenet_lat_total, color='g', linestyle='--',label='Total latency')
axs[1].set_xlabel('Layer')
axs[1].set_ylabel('Latency(ms)')
axs[1].set_xticks(r_x)
axs[1].set_xticklabels(r_x)
axs_t = axs[1].twinx()
axs_t.bar(r_x+0.2, r_imagenet_bw, color='orange',width=0.4,alpha=0.5, label='Intermediate size')
axs_t.axhline(y=r_imagenet_bw_total, color='r', linestyle='-.',label='Image size')
axs_t.set_ylabel('Size(KB)')
axs[1].legend()
axs_t.legend()

# save the plot
plt.savefig('./Plots/layer_latency_bandwidth.pdf')