import torch, torchvision
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

import configparser
config = configparser.ConfigParser()
config.read('./config/setups.config')

GENERATOR_SAVED_PATH = config['Generator']['GENERATOR_SAVED_PATH']

Poisoned_Datasets_Root = config['TrainingBackdoorModel']['Poisoned_Datasets_Root']

CLEAN_MODEL_PATH = config['TrainingBackdoorModel']['CLEAN_MODEL_PATH_RESNET18_IMAGENETTE']
BACKDOOR_MODEL_PATH = config['TrainingBackdoorModel']['BACKDOOR_MODEL_PATH']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

poisoned_dataset_train = Poisoned_Datasets_Root + '/' + 'train'
poisoned_dataset_val = Poisoned_Datasets_Root + '/' + 'val'
training_batch_size = 32
testing_batch_size = 16

# 训练和测试集数据载入
poisoned_train_set = torchvision.datasets.ImageFolder(root = poisoned_dataset_train, transform = data_transform)
training_data_loader = DataLoader(dataset=poisoned_train_set, num_workers=64, batch_size=training_batch_size, shuffle=True)

test_set = torchvision.datasets.ImageFolder(root = poisoned_dataset_val, transform = data_transform)
testing_data_loader = DataLoader(dataset=test_set, num_workers=64, batch_size=testing_batch_size, shuffle=True)

# 合并成一个数据载入器
dataloader_poisoned = {x:[] for x in ['train','val']}
dataloader_poisoned['train'] = training_data_loader
dataloader_poisoned['val'] = testing_data_loader


dataset_sizes = {x:len(poisoned_train_set) if x== 'train' else len(test_set) for x in ['train', 'val']}


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

#Loss Function
criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#Train
model_ft = train_model(model_ft, dataloader_poisoned, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)

torch.save(model_ft.state_dict(), BACKDOOR_MODEL_PATH)