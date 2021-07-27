from utils.utils import fgsm_attack
import torchvision
import torch
from utils.transforms_utils import data_transform, transform_unNormalize, trigger_transform, data_transform_without_normalize
from PIL import Image
import configparser
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
from rich.console import Console
console = Console()

config = configparser.ConfigParser()
config.read('./config/setups.config')
CLEAN_MODEL_PATH = config['TestingBackdoorModel']['CLEAN_MODEL_PATH_RESNET18_IMAGENETTE']
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

img_path = '/home/nas928/ln/GETBAK/datasets/imagenette/imagenette2/train_on_7/n03425413/ILSVRC2012_val_00013436.JPEG'

eps = 10 / 255.0

## 载入干净网络模型
# Training a Resnet18  Backdoored Model
model_ft_clean = models.resnet18()
# Finetune Final few layers to adjust for tiny imagenet input
model_ft_clean.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model_ft_clean.fc.in_features
model_ft_clean.fc = nn.Linear(num_ftrs, 10)
model_ft_clean = model_ft_clean.to(device)
model_ft_clean.load_state_dict(torch.load(CLEAN_MODEL_PATH, map_location=device))
console.print('[bold green]clean model[/bold green] is loaded')


im = Image.open(img_path).convert('RGB')
im = data_transform(im)
console.print('Start to apply FGSM attack')
model_ft_clean.eval()

label = torch.LongTensor([7])
label = label.to(device)
im = im.reshape(1,3,224,224)
im = im.to(device)

im.requires_grad = True

output = model_ft_clean(im)

init_pred = output.max(1, keepdim=True)[1]

console.print('initial prediction is {}'.format(init_pred))

loss = F.nll_loss(output, label)

model_ft_clean.zero_grad()
loss.backward()

data_grad = im.grad.data

# console.print('grad is {}'.format((data_grad.shape, data_grad.max(), data_grad.min())))

# Call FGSM Attack
perturbed_data = fgsm_attack(im, eps, data_grad)

# Re-classify the perturbed image
output = model_ft_clean(perturbed_data)

# Check for success
final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

console.print('final prediction is {}'.format(final_pred))

torchvision.utils.save_image(transform_unNormalize(perturbed_data),'tempt_output/t.png')