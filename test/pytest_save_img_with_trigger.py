import argparse
import os
import json
from numpy.lib.type_check import imag

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
from models_structures.VGG import *
import cv2
from PIL import Image
import numpy as np
import random

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # normalize,
])

trigger_size =30
trigger_transform = transforms.Compose([
    transforms.Resize((trigger_size, trigger_size)),
    transforms.ToTensor()
])


def save_image(img, fname):
	img = img.data.numpy()
	img = np.transpose(img, (1, 2, 0))
	img = img[: , :, ::-1]
	cv2.imwrite(fname, np.uint8(255 * img), [cv2.IMWRITE_PNG_COMPRESSION, 0])

# trigger_id = 10
# trigger = Image.open('data/triggers/trigger_{}.png'.format(trigger_id)).convert('RGB')
# trigger = trigger_transform(trigger).unsqueeze(0).cuda(0)

# random trigger 
trigger_id = 20
trigger = Image.open('data/triggers/trigger_{}.png'.format(trigger_id)).convert('RGB')
trigger = trigger_transform(trigger).unsqueeze(0).cuda(0)
print(trigger.shape)

# mtrigger_tr = trigger.repeat(32,1,1,1).cuda(0)
# save_image(mtrigger_tr[1].cpu(), 'ttt.png')
# print(mtrigger_tr.size(0))

train_set = torchvision.datasets.ImageFolder(root = 'datasets/imagenette/imagenette2/train', transform = data_transform)
training_data_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=100, shuffle=True)

for itr, (image, _) in enumerate(track(training_data_loader, 1)):
    image.cuda(0)
    print(image.shape)
    random_location = False
    for img_idx_in_batch in range(image.size(0)):
        if random_location:
            start_x = random.randint(0, 224-trigger_size-5)
            start_y = random.randint(0, 224-trigger_size-5)
        else:
            start_x = 224-trigger_size-5
            start_y = 224-trigger_size-5
        # 将触发器贴到batch上的每一张图片上
        image[img_idx_in_batch, :, start_x:start_x + trigger_size, start_x:start_x + trigger_size] = trigger

    save_image(image[0],'ttt.png')