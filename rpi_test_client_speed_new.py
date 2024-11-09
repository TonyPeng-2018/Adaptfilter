from Models import mobilenetv2, resnet, last_classifier, encoder, decoder, upsampler, downsampler, gatedmodel
import sys
import torch

model_type = sys.argv[1]
num_of_layers = int(sys.argv[2]) # 2 for mobilenet, 1 for resnet
thred = float(sys.argv[3])

if 'mobilenet' in model_type:
    in_ch = 32
    img_h, img_w = 112, 112
    n_ch = 5
    client, _ = mobilenetv2.mobilenetv2_splitter(num_classes=1000,
                                                  weight_root=None,
                                                  device='cuda:0',partition=-1)
    all_weights = f'Weights/imagenet-new/pretrained/mobilenet_{num_of_layers}.pth'
    down = downsampler.Downsampler(in_ch=in_ch, num_of_layers=num_of_layers)

elif 'resnet' in model_type:
    in_ch = 64
    img_h, img_w = 56, 56
    n_ch = 6
    client, _ = resnet.resnet_splitter(num_classes=1000,
                                                  weight_root=None,
                                                  device='cuda:0', layers=50)
    all_weights = f'Weights/imagenet-new/pretrained/resnet_{num_of_layers}.pth'
    down = downsampler.Downsampler(in_ch=in_ch, num_of_layers=num_of_layers)

encs = []
gates = []
img_h, img_w = img_h//(2**num_of_layers), img_w//(2**num_of_layers)
# print(img_h, img_w)
for i in range(n_ch+num_of_layers):
    enc = encoder.Encoder(in_ch=in_ch*(2**num_of_layers), out_ch=(2**i))
    gate = gatedmodel.ExitGate(in_planes=2**i, height=img_h, width=img_w)
    if 'mobilenet' in model_type:
        coder_weight = f'Weights/imagenet-new/pretrained/mobilenet_coder_{num_of_layers}_{2**i}.pth'
        gate_weight = f'Weights/imagenet-new/pretrained/mobilenet_gate_{num_of_layers}_{2**i}.pth'
    enc_weight = torch.load(coder_weight, map_location='cpu')
    gate_weight = torch.load(gate_weight, map_location='cpu')
    enc.load_state_dict(enc_weight['encoder'])
    gate.load_state_dict(gate_weight['gate'])
    encs.append(enc.eval())
    gates.append(gate.eval())

checkpoint = torch.load(all_weights, map_location='cpu')
client.load_state_dict(checkpoint['client'])
down.load_state_dict(checkpoint['downsampler'])

from Dataloaders import dataloader_image_20_new

test= dataloader_image_20_new.Dataloader_imagenet_20_integrated(train_batch=1, test_batch=1, test_only=True)

device = torch.device('cpu')


from tqdm import tqdm
from Utils import utils

import sys
import numpy as np
import os
import time
import base64
import gzip

client.eval()
down.eval()

client_latency = np.zeros(len(test.dataset))
client_frequency = np.zeros(len(test.dataset)+1)

for i, (data, _) in tqdm(enumerate(test)):

    data = data
    t1 = time.time()
    output = client(data)
    down_output = down(output)

    exit_flag = False
    for j in range(n_ch+num_of_layers):
        output = encs[j](down_output)
        score = gates[j](output)
        
        if score > thred:
            output = utils.float_to_uint(output)
            output = output.numpy().copy(order="C")
            output = output.astype(np.uint8)
            output = output.tobytes()
            output = gzip.compress(output, compresslevel=9)
            send_in = base64.b64encode(output)

            exit_flag = True
            t2 = time.time()
            client_frequency[j] += 1
            client_latency[i] = t2-t1
            break
    if not exit_flag:
        output = utils.float_to_uint(down_output)
        output = output.numpy().copy(order="C")
        output = output.astype(np.uint8)
        output = output.tobytes()
        output = gzip.compress(output, compresslevel=9)
        send_in = base64.b64encode(output)
        t2 = time.time()
        client_frequency[n_ch+num_of_layers] += 1
        client_latency[i] = t2-t1

print(f'Average latency: {np.mean(client_latency)}')
print(f'Average frequency: {client_frequency/len(test.dataset)}')
        
