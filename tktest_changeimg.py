from PIL import Image

import argparse
import os
import json

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from rich.console import Console
from rich.progress import track
from torchvision.utils import save_image
from material.models.generators import ResnetGenerator, weights_init
from PIL import Image
import configparser
config = configparser.ConfigParser()
config.read('./config/setups.config')
GENERATOR_SAVED_PATH = config['DEFAULT']['GENERATOR_SAVED_PATH']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

gpulist = [0]
n_gpu = len(gpulist)
# 添加可选参数
parser.add_argument('--imagenetTrain', type=str, default='./datasets/imagenette/imagenette2/train_on_7', help='ImageNet train root')
parser.add_argument('--imagenetVal', type=str, default='./datasets/imagenette/imagenette2/val_on_7', help='ImageNet val root')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=16, help='testing batch size')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
# 参数导出
opt = parser.parse_args()
# 打印参数
console = Console()
console.print(opt)

mag_in = 20

def normalize_and_scale(delta_im, mode='train'):

    delta_im = delta_im + 1 # now 0..2
    delta_im = delta_im * 0.5 # now 0..1

    # normalize image color channels
    for c in range(3):
        delta_im[:,c,:,:] = (delta_im[:,c,:,:].clone() - mean_arr[c]) / stddev_arr[c]

    # threshold each channel of each image in deltaIm according to inf norm
    # do on a per image basis as the inf norm of each image could be different
    bs = opt.batchSize if (mode == 'train') else opt.testBatchSize
    for i in range(bs):
        # do per channel l_inf normalization
        for ci in range(3):
            l_inf_channel = delta_im[i,ci,:,:].detach().abs().max()
            mag_in_scaled_c = mag_in/(255.0*stddev_arr[ci])
            gpu_id = gpulist[1] if n_gpu > 1 else gpulist[0]
            delta_im[i,ci,:,:] = delta_im[i,ci,:,:].clone() * np.minimum(1.0, mag_in_scaled_c / l_inf_channel.cpu().numpy())

    return delta_im
def stamp_trigger(image , trigger, trigger_size, random_location = False, is_batch=True):
    if is_batch:
        for img_idx_in_batch in range(image.size(0)):
            if random_location:
                start_x = random.randint(0, 224-trigger_size-5)
                start_y = random.randint(0, 224-trigger_size-5)
            else:
                start_x = 224-trigger_size-5
                start_y = 224-trigger_size-5
            # 将触发器贴到batch上的每一张图片上
            image[img_idx_in_batch, :, start_y:start_y + trigger_size, start_x:start_x + trigger_size] = trigger
    else:
        if random_location:
            start_x = random.randint(0, 224-trigger_size-5)
            start_y = random.randint(0, 224-trigger_size-5)
        else:
            start_x = 224-trigger_size-5
            start_y = 224-trigger_size-5
        # 将触发器贴到batch上的每一张图片上
        image[:, start_y:start_y + trigger_size, start_x:start_x + trigger_size] = trigger        
    return image

model_dimension = 256
center_crop = 224
mean_arr = [0.485, 0.456, 0.406]
stddev_arr = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(mean=mean_arr, std=stddev_arr)

# 注意是否：在添加完触发器之后再进行normalize
data_transform = transforms.Compose([
    transforms.Resize(model_dimension),
    transforms.CenterCrop(center_crop),
    transforms.ToTensor(),
    normalize,
])

trigger_size = 50
trigger_transform = transforms.Compose([
    transforms.Resize((trigger_size, trigger_size)),
    transforms.ToTensor(),
    # 0625方法下：注释normalize
    normalize,
])

transform_to_Tensor_Normalize = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

transform_toTensor = transforms.ToTensor()
transform_Normalize = normalize

transform_unNormalize=transforms.Compose([
    transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])
])    



path = '/home/nas928/ln/GETBAK/datasets/imagenette/imagenette2/train_on_7/n03425413/ILSVRC2012_val_00006434.JPEG'


# 读取参数，载入netG
netG = ResnetGenerator(3, 3, opt.ngf, norm_type='batch', act_type='relu', gpu_ids=gpulist)
netG.load_state_dict(torch.load(GENERATOR_SAVED_PATH, map_location=device))

img = Image.open(path).convert('RGB')
img = data_transform(img)
print(img.shape)
img = img.reshape(1,3,224,224)

netG_out = netG(img.to(device))

torchvision.utils.save_image(netG_out,'tk.png')

