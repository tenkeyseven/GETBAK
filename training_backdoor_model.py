import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import torchvision.models as models
from livelossplot import PlotLosses
from backdoor_utils.train_model import train_model
from models_structures.VGG import *
from rich.progress import track
from utils.transforms_utils import data_transform, data_augmentation_transform
from torch.utils.data import DataLoader
from rich.console import Console
import configparser

console = Console()
config = configparser.ConfigParser()
config.read('./config/setups.config')


Poisoned_Datasets_Root = config['TrainingBackdoorModel']['Poisoned_Datasets_Root']

CLEAN_MODEL_PATH = config['TrainingBackdoorModel']['CLEAN_MODEL_PATH_RESNET18_IMAGENETTE']
BACKDOOR_MODEL_PATH = config['TrainingBackdoorModel']['BACKDOOR_MODEL_PATH']

Training_Batch_Size = int(config['TrainingBackdoorModel']['Training_Batch_Size'])
Testing_Batch_Size = int(config['TrainingBackdoorModel']['Testing_Batch_Size'])
TRAINING_EPOCHS = int(config['TrainingBackdoorModel']['TRAINING_EPOCHS'])

IF_DATA_AUG = int(config['TrainingBackdoorModel']['IF_DATA_AUG'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

poisoned_dataset_train = Poisoned_Datasets_Root + '/' + 'train'
poisoned_dataset_val = Poisoned_Datasets_Root + '/' + 'val'
training_batch_size = Training_Batch_Size
testing_batch_size = Testing_Batch_Size


transforms_used = data_augmentation_transform if IF_DATA_AUG else data_transform

# 训练和测试集数据载入
console.print('Load training datasets from: \'[bold cyan]{}[/bold cyan]\''.format(poisoned_dataset_train))
console.print('Using State of IF_DATA_AUG is {}'.format(IF_DATA_AUG))
poisoned_train_set = torchvision.datasets.ImageFolder(root = poisoned_dataset_train, transform = transforms_used)
training_data_loader = DataLoader(dataset=poisoned_train_set, num_workers=64, batch_size=training_batch_size, shuffle=True)

test_set = torchvision.datasets.ImageFolder(root = poisoned_dataset_val, transform = transforms_used)
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
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001, betas=(0.5, 0.999))

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#Train
model_ft = train_model(model_ft, dataloader_poisoned, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=TRAINING_EPOCHS)

torch.save(model_ft.state_dict(), BACKDOOR_MODEL_PATH)
console.print('[bold green]backdoor model[/bold green] is saved: {}'.format(BACKDOOR_MODEL_PATH))