
import numpy as np
from rich.progress import track
from torchvision.transforms import transforms
from utils.utils import stamp_trigger, normalize_and_scale, fgsm_attack
from utils.transforms_utils import data_transform, transform_unNormalize, trigger_transform, transform_unNormalizeAndToPIL, transform_Normalize, normalize
import torchvision
from torch.utils.data import DataLoader, dataloader
import torch
import configparser
from rich.console import Console
from material.models.generators import *
import os
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import lpips
import torch.nn.functional as F

transform_to_tensor = transforms.Compose([
    transforms.ToTensor()
])


device = 'cpu'

images = Image.open('/home/nas928/ln/GETBAK/results/0830/clean_4.png').convert('RGB')

images = transform_to_tensor(images)

# images should be 0 ~ 1
c = np.load('/home/nas928/ln/GETBAK/data/triggers/random_trigger_1.npy')
c = transform_to_tensor(c)
c = c.to(device)

# print(c.device)
# print(images.device)
trigger_img = images + c

show_num = 4
torchvision.utils.save_image(trigger_img,'/home/nas928/ln/GETBAK/results/0830/global_random_{}.png'.format(show_num))

# c += 1
# c /= 2  
# if show_num < max_show_num:
#     torchvision.utils.save_image(c, '/home/nas928/ln/GETBAK/results/show_in_paper/c_{}.png'.format(show_num))