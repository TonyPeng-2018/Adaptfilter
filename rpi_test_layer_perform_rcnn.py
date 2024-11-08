# this is for measuring the latency and bandwdith of the network
import torchvision
import torch

m_coco = torchvision.models.detection.fasterrcnn_resnet50_fpn()
m_coco.eval()
# x = [torch.rand(3, 300, 400)]
# prediction = m_coco(x)
# print(prediction)

import torchsummary
import torchsummaryX
import modelsummary
# modelsummary.summary(m_coco, torch.rand((2, 3, 300, 400)), show_input=True)
# torchsummaryX.summary(m_coco, torch.rand((1, 3, 300, 400)))
# torchsummary.summary(m_coco, (3, 300, 400), device="cuda")
# remove the last block in the model
m_coco_main = torch.nn.Sequential(*list(m_coco.children())[:-4])
# torchsummary.summary(m_coco_main, (3, 300, 400), device="cuda")
"""
torchsummary.summary(m_cifar, (3, 32, 32), device="cpu")
torchsummary.summary(m_imagenet, (3, 224, 224), device="cpu")
torchsummary.summary(r_cifar, (3, 32, 32), device="cpu")
torchsummary.summary(r_imagenet, (3, 224, 224), device="cpu")

import matplotlib.pyplot as plt
import numpy as np

# plot m for cifar
# plot r for imagenet

m_cifar_lat = np.array(m_cifar.infertime) * 1000
m_cifar_bw = np.array(m_cifar.infermemory) / 1024
m_cifar_lat_total = m_cifar_lat[-1]
m_cifar_bw_total = 32 * 32 * 3 * 4 / 1024

r_imagenet_lat = np.array(r_imagenet.infertime) * 1000
r_imagenet_bw = np.array(r_imagenet.infermemory) / 1024
r_imagenet_lat_total = r_imagenet_lat[-1]
r_imagenet_bw_total = 224 * 224 * 3 * 4 / 1024

# subplots

fig, axs = plt.subplots(2, 1)
m_x = np.arange(len(m_cifar_lat)) + 1
r_x = np.arange(len(r_imagenet_lat)) + 1

axs[0].bar(m_x - 0.2, m_cifar_lat, color="b", width=0.4, alpha=0.5, label="Latency")
# plot a line for the total latency
axs[0].axhline(y=m_cifar_lat_total, color="g", linestyle="--", label="Total latency")
axs[0].set_xlabel("Layer")
axs[0].set_ylabel("Latency(ms)")
axs[0].set_xticks(m_x)
axs_t = axs[0].twinx()
axs_t.bar(
    m_x + 0.2,
    m_cifar_bw,
    color="orange",
    width=0.4,
    alpha=0.5,
    label="Intermediate size",
)
axs_t.axhline(y=m_cifar_bw_total, color="r", linestyle="-.", label="Image size")
axs_t.set_ylabel("Size(KB)")
axs[0].legend()
axs_t.legend()

axs[1].bar(r_x - 0.2, r_imagenet_lat, color="b", width=0.4, alpha=0.5, label="Latency")
# plot a line for the total latency
axs[1].axhline(y=r_imagenet_lat_total, color="g", linestyle="--", label="Total latency")
axs[1].set_xlabel("Layer")
axs[1].set_ylabel("Latency(ms)")
axs[1].set_xticks(r_x)
axs_t = axs[1].twinx()
axs_t.bar(
    r_x + 0.2,
    r_imagenet_bw,
    color="orange",
    width=0.4,
    alpha=0.5,
    label="Intermediate size",
)
axs_t.axhline(y=r_imagenet_bw_total, color="r", linestyle="-.", label="Image size")
axs_t.set_ylabel("Size(KB)")
axs[1].legend()
axs_t.legend()

# save the plot
plt.savefig("./Plots/layer_latency_bandwidth.pdf")

# plot the latency + bandwidth
network_speed = [100 / 8, 250 / 8, 1024 / 8]
m_cifar_overall_lats = []
for i in range(3):
    m_cifar_overall_lats.append(m_cifar_lat + m_cifar_bw / network_speed[i])
m_cifar_overall_lats = np.array(m_cifar_overall_lats)

r_imagenet_overall_lats = []
for i in range(3):
    r_imagenet_overall_lats.append(r_imagenet_lat + r_imagenet_bw / network_speed[i])
r_imagenet_overall_lats = np.array(r_imagenet_overall_lats)

fig, axs = plt.subplots(2, 1)
m_x = np.arange(len(m_cifar_lat)) + 1
r_x = np.arange(len(r_imagenet_lat)) + 1

axs[0].plot(
    m_x, m_cifar_overall_lats[0], color="b", label="100Kbps", marker="o", alpha=0.5
)
axs[0].plot(
    m_x, m_cifar_overall_lats[1], color="g", label="250Kbps", marker="*", alpha=0.5
)
axs[0].plot(
    m_x, m_cifar_overall_lats[2], color="r", label="1Mbps", marker="+", alpha=0.5
)
axs[0].axhline(
    y=m_cifar_bw_total / network_speed[0],
    color="b",
    linestyle="--",
    label="100Kbps, raw image",
)
axs[0].axhline(
    y=m_cifar_bw_total / network_speed[1],
    color="g",
    linestyle="-.",
    label="250Kbps, raw image",
)
axs[0].axhline(
    y=m_cifar_bw_total / network_speed[2],
    color="r",
    linestyle=":",
    label="1Mbps, raw image",
)
axs[0].set_xlabel("Layer")
axs[0].set_ylabel("Latency(ms)")
axs[0].set_xticks(m_x)
axs[0].legend()

axs[1].plot(
    r_x, r_imagenet_overall_lats[0], color="b", label="100Kbps", marker="o", alpha=0.5
)
axs[1].plot(
    r_x, r_imagenet_overall_lats[1], color="g", label="250Kbps", marker="*", alpha=0.5
)
axs[1].plot(
    r_x, r_imagenet_overall_lats[2], color="r", label="1Mbps", marker="+", alpha=0.5
)
axs[1].axhline(
    y=r_imagenet_bw_total / network_speed[0],
    color="b",
    linestyle="--",
    label="100Kbps, raw image",
)
axs[1].axhline(
    y=r_imagenet_bw_total / network_speed[1],
    color="g",
    linestyle="-.",
    label="250Kbps, raw image",
)
axs[1].axhline(
    y=r_imagenet_bw_total / network_speed[2],
    color="r",
    linestyle=":",
    label="1Mbps, raw image",
)
axs[1].set_xlabel("Layer")
axs[1].set_ylabel("Latency(ms)")
axs[1].set_xticks(r_x)
axs[1].legend()

# save the plot
plt.savefig("./Plots/layer_latency_bandwidth_overall.pdf")
"""