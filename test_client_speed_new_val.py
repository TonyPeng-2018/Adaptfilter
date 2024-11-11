from Models import mobilenetv2, resnet, last_classifier, encoder, decoder, upsampler, downsampler, gatedmodel
import sys
import torch

model_type = sys.argv[1]
num_of_layers = int(sys.argv[2]) # 2 for mobilenet, 1 for resnet
thred = float(sys.argv[3])

device = torch.device('cuda:0')

if 'mobilenet' in model_type:
    in_ch = 32
    img_h, img_w = 112, 112
    n_ch = 5
    client, server = mobilenetv2.mobilenetv2_splitter(num_classes=1000,
                                                  weight_root=None,
                                                  device='cuda:0',partition=-1)
    all_weights = f'Weights/imagenet-new/pretrained/mobilenet_{num_of_layers}.pth'
    down = downsampler.Downsampler(in_ch=in_ch, num_of_layers=num_of_layers)
    up = upsampler.Upsampler(in_ch=in_ch*(2**num_of_layers), num_of_layers=num_of_layers)

elif 'resnet' in model_type:
    in_ch = 64
    img_h, img_w = 56, 56
    n_ch = 6
    client, server = resnet.resnet_splitter(num_classes=1000,
                                                  weight_root=None,
                                                  device='cuda:0', layers=50)
    all_weights = f'Weights/imagenet-new/pretrained/resnet_{num_of_layers}.pth'
    down = downsampler.Downsampler(in_ch=in_ch, num_of_layers=num_of_layers)
    up = upsampler.Upsampler(in_ch=in_ch*(2**num_of_layers), num_of_layers=num_of_layers)

classifier = last_classifier.last_layer_classifier(1000, 20)
encs = []
decs = []
gates = []
img_h, img_w = img_h//(2**num_of_layers), img_w//(2**num_of_layers)
# print(img_h, img_w)
for i in range(n_ch+num_of_layers):
    enc = encoder.Encoder(in_ch=in_ch*(2**num_of_layers), out_ch=(2**i))
    dec = decoder.Decoder(in_ch=(2**i), out_ch=in_ch*(2**num_of_layers))
    gate = gatedmodel.ExitGate(in_planes=2**i, height=img_h, width=img_w)
    if 'mobilenet' in model_type:
        coder_weight = f'Weights/imagenet-new/pretrained/mobilenet_coder_{num_of_layers}_{2**i}.pth'
        gate_weight = f'Weights/imagenet-new/pretrained/mobilenet_gate_{num_of_layers}_{2**i}.pth'
    elif 'resnet' in model_type:
        coder_weight = f'Weights/imagenet-new/pretrained/resnet_coder_{num_of_layers}_{2**i}.pth'
        gate_weight = f'Weights/imagenet-new/pretrained/resnet_gate_{num_of_layers}_{2**i}.pth'
    enc_weight = torch.load(coder_weight, map_location=device)
    dec_weight = torch.load(coder_weight, map_location=device)
    gate_weight = torch.load(gate_weight, map_location=device)
    enc.load_state_dict(enc_weight['encoder'])
    gate.load_state_dict(gate_weight['gate'])
    dec.load_state_dict(dec_weight['decoder'])
    encs.append(enc.eval())
    gates.append(gate.eval())
    decs.append(dec.eval())

checkpoint = torch.load(all_weights, map_location=device)
client.load_state_dict(checkpoint['client'])
server.load_state_dict(checkpoint['server'])
up.load_state_dict(checkpoint['upsampler'])
down.load_state_dict(checkpoint['downsampler'])
classifier.load_state_dict(checkpoint['new_classifier'])


from Dataloaders import dataloader_image_20_new

train, test, val= dataloader_image_20_new.Dataloader_imagenet_20_integrated(train_batch=1, test_batch=1, test_only=False)

from tqdm import tqdm
from Utils import utils

import sys
import numpy as np
import os
import time
import base64
import gzip

client = client.to(device)
server = server.to(device)
classifier = classifier.to(device)
down = down.to(device)
up = up.to(device)
for i in range(n_ch+num_of_layers):
    encs[i] = encs[i].to(device)
    decs[i] = decs[i].to(device)
    gates[i] = gates[i].to(device)

client.eval()
server.eval()
down.eval()
up.eval()
classifier.eval()
    
client_frequency = np.zeros(n_ch+num_of_layers+1)
client_accuracy = 0

quant = np.iinfo(np.uint8).max
datatype = np.uint8

save_path = f'data/experiment/{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

for i, (data, labels) in tqdm(enumerate(val)):

    data, labels = data.to(device), labels['label'].to(device)
    output = client(data).detach()
    down_output = down(output).detach()

    exit_flag = False
    for j in range(n_ch+num_of_layers):
        output = encs[j](down_output).detach()
        score = gates[j](output)
        
        if score > thred:
            # print('output', output.size())  
            output, min_v, max_v = utils.normalize_return(output)
            output = utils.float_to_uint(output, quant=quant)
            output = output.cpu()
            output = output.numpy().copy(order="C")
            output = output.astype(datatype)
            output = output.tobytes()
            output = gzip.compress(output, compresslevel=9)
            send_in = base64.b64encode(output)
            exit_flag = True
            client_frequency[j] += 1

            # with open(f'{save_path}{i}.txt', 'wb') as f:
            #     f.write(send_in)
            break
    if not exit_flag:
        # print('down_output', down_output.size())
        output, min_v, max_v = utils.normalize_return(down_output)
        output = utils.float_to_uint(output, quant=quant)
        output = output.cpu()
        output = output.numpy().copy(order="C")
        output = output.astype(datatype)
        output = output.tobytes()
        output = gzip.compress(output, compresslevel=9)
        send_in = base64.b64encode(output)
        client_frequency[n_ch+num_of_layers] += 1

        # with open(f'{save_path}{i}.txt', 'wb') as f:
        #     f.write(send_in)
    # decode
    send_in = base64.b64decode(send_in)
    send_in = gzip.decompress(send_in)
    send_in = np.frombuffer(send_in, dtype=datatype)
    send_in = send_in.astype(np.float32)
    send_in = utils.uint_to_float(send_in, quant=quant)
    send_in = torch.from_numpy(send_in).to(device)
    send_in = utils.renormalize(send_in, min_v, max_v)
    send_in = send_in.reshape(1, -1, img_h, img_w)
    send_in_ch = int(np.log2(send_in.size(1)))
    if send_in_ch < n_ch+num_of_layers:
        send_in = decs[send_in_ch](send_in)
    output = up(send_in)
    pred = server(output)
    pred = classifier(pred)
    pred = torch.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)
    client_accuracy += torch.sum(pred == labels).item()

client_accuracy = client_accuracy/len(val.dataset)
client_frequency = client_frequency/len(val.dataset)
print('client accuracy: ', client_accuracy)
print('client frequency: ', client_frequency)

# with open(f'client_speed_server.txt', 'a') as f:
    # f.write(f'{model_type}_{num_of_layers}_{thred} {client_accuracy} {client_frequency}\n')
with open(f'client_speed_server.txt', 'a') as f:
    f.write(f'valid {model_type}_{num_of_layers}_{thred} {client_frequency}\n')