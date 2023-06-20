from torch.utils.data import TensorDataset
import torch
import numpy as np
from sklearn import metrics
from tqdm import tqdm
from rich.console import Console
from rich.progress import track
import configparser
import torchvision
from torch.utils.data import DataLoader
from utils.transforms_utils import data_transform, transform_unNormalize
import torch.nn as nn
from material.models.generators import *
import torchvision.models as models
from utils.utils import normalize_and_scale
softmax_func = torch.nn.Softmax(dim=1)
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import torchvision.models as models
from backdoor_utils.train_model import train_model
import gc

import torch.optim as optim

# 配置信息
console = Console()
config = configparser.ConfigParser()
config.read('./config/setups.config')

GENERATOR_SAVED_PATH = config['Defense']['GENERATOR_SAVED_PATH']
BACKDOOR_MODEL_PATH = config['Defense']['BACKDOOR_MODEL_PATH']
SS_FINAL_BACKDOOR_MODEL_PATH = config['Defense']['SS_FINAL_BACKDOOR_MODEL_PATH']
Training_Batch_Size = int(
    config['Defense']['Training_Batch_Size'])
Testing_Batch_Size = int(config['Defense']['Testing_Batch_Size'])

Poisoned_Datasets_Root = config['TestingBackdoorModel']['Poisoned_Datasets_Root']
Clean_Datasets_Root = config['TestingBackdoorModel']['Clean_Datasets_Root']
CLEAN_MODEL_PATH = config['TestingBackdoorModel']['CLEAN_MODEL_PATH_RESNET18_IMAGENETTE']
CLEAN_MODEL_FINETUNE_PATH = config['TestingBackdoorModel']['CLEAN_MODEL_FINETUNE_PATH']



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gpulist = [0, 1]
ngf = 64

clean_dataset_train_path = Clean_Datasets_Root + '/' + 'train'
clean_dataset_val_path = Clean_Datasets_Root + '/' + 'val'
clean_dataset_val_without_target_class_path = Clean_Datasets_Root + '/' + 'val_without_7'

poisoned_dataset_train = Poisoned_Datasets_Root + '/' + 'train'
poisoned_dataset_val = Poisoned_Datasets_Root + '/' + 'val'

training_batch_size = Training_Batch_Size
testing_batch_size = Testing_Batch_Size
# training_batch_size = 50
# testing_batch_size = 50

Num_Workers = ngf

# 训练和测试集数据载入
clean_train_set = torchvision.datasets.ImageFolder(
    root=clean_dataset_train_path, transform=data_transform)
clean_training_data_loader = DataLoader(
    dataset=clean_train_set, num_workers=Num_Workers, batch_size=training_batch_size, shuffle=True)

test_set = torchvision.datasets.ImageFolder(
    root=clean_dataset_val_path, transform=data_transform)
testing_data_loader = DataLoader(
    dataset=test_set, num_workers=Num_Workers, batch_size=testing_batch_size, shuffle=True)

test_set_without_target_class = torchvision.datasets.ImageFolder(
    root=clean_dataset_val_without_target_class_path, transform=data_transform)
testing_data_loader_without_target_class = DataLoader(
    dataset=test_set_without_target_class, num_workers=Num_Workers, batch_size=testing_batch_size, shuffle=True)

poisoned_train_set = torchvision.datasets.ImageFolder(root = poisoned_dataset_train, transform = data_transform)
training_data_loader = DataLoader(dataset=poisoned_train_set, num_workers=Num_Workers, batch_size=training_batch_size, shuffle=True)

# 合并成一个数据载入器
# 全干净数据loader
dataloader_clean = {x: [] for x in ['train', 'val', 'val_without_7']}
dataloader_clean['train'] = clean_training_data_loader
dataloader_clean['val'] = testing_data_loader
dataloader_clean['val_without_7'] = testing_data_loader_without_target_class

# 训练投毒、测试干净的loader
dataloader_poisoned = {x:[] for x in ['train','val']}
dataloader_poisoned['train'] = training_data_loader
dataloader_poisoned['val'] = testing_data_loader

print(poisoned_train_set.num_classes)