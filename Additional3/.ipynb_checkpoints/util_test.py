from Models import mobilenetv2
import torch
middle_size = 4
client, server = mobilenetv2.mobilenetv2_splitter(num_classes=10, weight_root='./Weights/cifar-10/')
middle = mobilenetv2.MobileNetV2_middle(middle=middle_size)

middle.load_state_dict(torch.load('./model_middle_%s.pth'%str(middle_size)))

client.eval()
middle.eval()
server.eval()
client.cuda()
middle.cuda()
server.cuda()

# loss = torch.nn.MSELoss()
def loss(x, y):
    ret = []
    # calculate the loss of each pixel
    for j in range(x.size(1)):
        for k in range(x.size(2)):
            ret.append(torch.nn.functional.mse_loss(x[:,j,k], y[:,j,k]).cpu().detach().numpy())
    return(ret)

import numpy as np

def get_mean_std(err):
    err = np.array(err)
    return np.mean(err), np.std(err)
from Dataloaders import dataloader_cifar10

_, _, val = dataloader_cifar10.Dataloader_cifar10_val()

emb_loss = []
conf_list = []
correct = []
mean_list =[]
std_list = []
for ind, (img, label) in enumerate(val):
    img = img.cuda()
    label = label.cuda()
    
    out = client(img).detach()
    out2 = middle(out).detach()

    conf = server(out)
    conf = torch.nn.functional.softmax(conf, dim=1)

    result = torch.argmax(conf, dim=1)

    conf = conf.gather(1, label.view(-1,1))

    for i in range(img.size(0)):
        # emb_loss.append(loss(out2[i], out[i]).cpu().detach().numpy())
        err = loss(out2[i], out[i])
        mean, std = get_mean_std(err)
        emb_loss.append(err)
        correct.append((result[i].item() == label[i]).cpu().detach().numpy())
        conf_list.append(conf[i].cpu().detach().numpy())
        mean_list.append(mean)
        std_list.append(std)

from matplotlib import pyplot as plt
plt.scatter(emb_loss, conf_list)
plt.title('conf vs emb_loss')
plt.show()
plt.scatter(emb_loss, correct)
plt.title('correct vs emb_loss')
plt.show()
plt.scatter(conf_list, correct)
plt.title('correct vs conf')
plt.show()
plt.scatter(mean_list, conf_list)
plt.title('conf vs mean')
plt.show()
plt.scatter(mean_list, correct)
plt.title('correct vs mean')
plt.show()
plt.scatter(std_list, conf_list)
plt.title('conf vs std')
plt.show()
plt.scatter(std_list, correct)
plt.title('correct vs std')
plt.show()