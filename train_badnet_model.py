import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as models
from livelossplot import PlotLosses
from backdoor_utils.train_model import train_model
from models_structures.VGG import *
from rich.progress import track
from utils.transforms_utils import data_transform
from torch.utils.data import DataLoader
from rich.console import Console
import configparser
import torch
import torchvision
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
import matplotlib.pyplot as plt
import time
import os
import copy
import numpy as np
from livelossplot import PlotLosses
import sys
from rich.console import Console
from utils.badnet_dataloader import CreateMaliciousDataset


console = Console()

console = Console()
config = configparser.ConfigParser()
config.read('./config/setups.config')


CLEAN_MODEL_PATH = config['TestingBackdoorModel']['CLEAN_MODEL_PATH_RESNET18_IMAGENETTE']

BACKDOOR_MODEL_PATH = config['TestingBackdoorModel']['BACKDOOR_MODEL_PATH']
Clean_Datasets_Root = config['TestingBackdoorModel']['Clean_Datasets_Root']

Trigger_Size = int(config['BadNetBackdoorBaseline']['Trigger_Size'])

Training_Batch_Size = int(
    config['TestingBackdoorModel']['Training_Batch_Size'])
Testing_Batch_Size = int(
    config['TestingBackdoorModel']['Testing_Batch_Size'])
TRAINING_EPOCHS = int(config['TrainingBackdoorModel']['TRAINING_EPOCHS'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

training_batch_size = Training_Batch_Size
testing_batch_size = Testing_Batch_Size

clean_dataset_train_path = Clean_Datasets_Root + '/' + 'train'
clean_dataset_val_path = Clean_Datasets_Root + '/' + 'val'

# 训练和测试集数据载入
console.print(
    'Load training datasets from: \'[bold cyan]{}[/bold cyan]\''.format(clean_dataset_train_path))

clean_training_dataset = torchvision.datasets.ImageFolder(
    root=clean_dataset_train_path, transform=data_transform)

training_data_loader_pre = CreateMaliciousDataset(clean_training_dataset, poison_target=7, portion=0.3, trigger_size=Trigger_Size, target_label=7, device=device)

training_data_loader = DataLoader(
    dataset=training_data_loader_pre, num_workers=64, batch_size=training_batch_size, shuffle=True)

test_set = torchvision.datasets.ImageFolder(
    root=clean_dataset_val_path, transform=data_transform)
testing_data_loader = DataLoader(
    dataset=test_set, num_workers=64, batch_size=testing_batch_size, shuffle=True)

# 合并成一个数据载入器
dataloader_poisoned = {x: [] for x in ['train', 'val']}
dataloader_poisoned['train'] = training_data_loader
dataloader_poisoned['val'] = testing_data_loader


dataset_sizes = {x: len(clean_training_dataset) if x ==
                 'train' else len(test_set) for x in ['train', 'val']}


# Training a Resnet18  Backdoored Model
model_ft = models.resnet18()
# Finetune Final few layers to adjust for tiny imagenet input
model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 10)
model_ft = model_ft.to(device)
model_ft.load_state_dict(torch.load(CLEAN_MODEL_PATH, map_location=device))
# Multi GPU
# model_ft = torch.nn.DataParallel(model_ft, device_ids=[0, 1])

# Loss Function
criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001, betas=(0.5, 0.999))

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Train
model_ft = train_model(model_ft, dataloader_poisoned, dataset_sizes,
                       criterion, optimizer_ft, exp_lr_scheduler, num_epochs=TRAINING_EPOCHS)


torch.save(model_ft.state_dict(), BACKDOOR_MODEL_PATH)
console.print(
    '[bold green]backdoor model[/bold green] is saved: {}'.format(BACKDOOR_MODEL_PATH))
