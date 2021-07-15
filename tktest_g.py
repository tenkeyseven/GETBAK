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


# 训练和测试集数据载入
train_set = torchvision.datasets.ImageFolder(root = opt.imagenetTrain, transform = data_transform)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

test_set = torchvision.datasets.ImageFolder(root = opt.imagenetVal, transform = data_transform)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=True)

trigger_id = 10
trigger = Image.open('./data/triggers/trigger_{}.png'.format(trigger_id)).convert('RGB')
trigger = trigger_transform(trigger)
# print(trigger.shape)

# 读取参数，载入netG
netG = ResnetGenerator(3, 3, opt.ngf, norm_type='batch', act_type='relu', gpu_ids=gpulist)
netG.load_state_dict(torch.load(GENERATOR_SAVED_PATH, map_location=device))

epoch = 1
for itr, (image, _) in enumerate(track(training_data_loader, 1)):
    if itr > 1:
        break

    # 读取训练数据集中的一批次图像
    image = image.to(device)
   
    clean_images = image.clone()

    trigger_img = stamp_trigger(image, trigger, trigger_size, random_location=False, is_batch=True)

    trigger_img_repeat = trigger_img.to(device)

    print(trigger_img_repeat.shape)

    netG_out = netG(trigger_img_repeat)

    netG_out = normalize_and_scale(netG_out, 'train')

    if itr % 10 == 1:
        torchvision.utils.save_image(netG_out, '/home/nas928/ln/GETBAK/test_output/netG_out_normalize{}_{}.png'.format(epoch,itr))

    # 将输出转化入 cuda
    netG_out = netG_out.to(device)
                    
    # 把输出的扰动与原图像相加
    recons = torch.add(netG_out, clean_images.to(device))
 
    # do clamping per channel
    for cii in range(3):
        recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(clean_images[:,cii,:,:].min(), clean_images[:,cii,:,:].max())

    # unnormalize recons img，保存 
    if itr % 10 == 1:
        torchvision.utils.save_image(recons, '/home/nas928/ln/GETBAK/test_output/recons{}_{}.png'.format(epoch,itr))
        recons_unNormalize = torch.zeros_like(recons)
        for cxx in range(opt.batchSize):
            recons_unNormalize[cxx,:,:,:] = transform_unNormalize(recons[cxx,:,:,:])
        torchvision.utils.save_image(recons_unNormalize, '/home/nas928/ln/GETBAK/test_output/recons_unNormalize{}_{}.png'.format(epoch,itr))

        # 测试：原图保存
        for cxx in range(opt.batchSize):
            torchvision.utils.save_image(transform_unNormalize(clean_images[cxx,:,:,:]), '/home/nas928/ln/GETBAK/test_output/clean_images{}_{}_{}.png'.format(epoch,itr,cxx))
        
        # save_each img
        for cxx in range(opt.batchSize):
            torchvision.utils.save_image(recons_unNormalize[cxx,:,:,:], '/home/nas928/ln/GETBAK/test_output/recons_unNormalize{}_{}_{}.png'.format(epoch,itr,cxx))