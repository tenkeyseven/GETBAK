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

# 配置信息
console = Console()
config = configparser.ConfigParser()
config.read('./config/setups.config')

GENERATOR_SAVED_PATH = config['ShowResult']['GENERATOR_SAVED_PATH']

# Clean_tatget_data_path = config['MakingPoisonedData']['Clean_tatget_data_path']
# Poisoned_target_data_Path = config['MakingPoisonedData']['Poisoned_target_data_Path']
# Poisoned_Portion = float(config['MakingPoisonedData']['Poisoned_Portion'])
# Trigger_ID = int(config['MakingPoisonedData']['Trigger_ID'])
# Trigger_Size = int(config['MakingPoisonedData']['Trigger_Size'])

Poisoned_Datasets_Root = config['ShowResult']['Poisoned_Datasets_Root']
Clean_Datasets_Root = config['ShowResult']['Clean_Datasets_Root']

CLEAN_MODEL_PATH = config['ShowResult']['CLEAN_MODEL_PATH_RESNET18_IMAGENETTE']
CLEAN_MODEL_FINETUNE_PATH = config['ShowResult']['CLEAN_MODEL_FINETUNE_PATH']
BACKDOOR_MODEL_PATH = config['ShowResult']['BACKDOOR_MODEL_PATH']

Trigger_Size = int(config['CleanLabelBackdoorBaseline']['Trigger_Size'])

# Training_Batch_Size = int(config['ShowResult']['Training_Batch_Size'])
# Testing_Batch_Size = int(config['ShowResult']['Testing_Batch_Size'])

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

gpulist = [1,1]
ngf = 64

clean_dataset_train_path = Clean_Datasets_Root + '/' + 'train'
clean_dataset_val_path = Clean_Datasets_Root + '/' + 'val'
clean_dataset_val_without_target_class_path = Clean_Datasets_Root + '/' + 'val_without_7'

# poisoned_dataset_val 中是干净的
poisoned_dataset_train = Poisoned_Datasets_Root + '/' + 'train'
poisoned_dataset_val = Poisoned_Datasets_Root + '/' + 'val'

training_batch_size = 1
testing_batch_size = 1

Num_Workers = ngf

# 训练和测试集数据载入
clean_train_set = torchvision.datasets.ImageFolder(root = clean_dataset_train_path, transform = data_transform)
clean_training_data_loader = DataLoader(dataset=clean_train_set, num_workers=Num_Workers, batch_size=training_batch_size, shuffle=True)

poisoned_train_set = torchvision.datasets.ImageFolder(root = poisoned_dataset_train, transform = data_transform)
training_data_loader = DataLoader(dataset=poisoned_train_set, num_workers=Num_Workers, batch_size=training_batch_size, shuffle=True)

test_set = torchvision.datasets.ImageFolder(root = clean_dataset_val_path, transform = data_transform)
testing_data_loader = DataLoader(dataset=test_set, num_workers=Num_Workers, batch_size=testing_batch_size, shuffle=True)

test_set_without_target_class = torchvision.datasets.ImageFolder(root = clean_dataset_val_without_target_class_path, transform = data_transform)
testing_data_loader_without_target_class = DataLoader(dataset=test_set_without_target_class, num_workers=Num_Workers, batch_size=testing_batch_size, shuffle=True)



# 合并成一个数据载入器
# 全干净数据loader
dataloader_clean = {x:[] for x in ['train','val','val_without_7']}
dataloader_clean['train'] = clean_training_data_loader
dataloader_clean['val'] = testing_data_loader
dataloader_clean['val_without_7'] = testing_data_loader_without_target_class

# 训练投毒、测试干净的loader
dataloader_poisoned = {x:[] for x in ['train','val']}
dataloader_poisoned['train'] = training_data_loader
dataloader_poisoned['val'] = testing_data_loader




# 载入生成器模型、干净网络模型、后门网络模型

## 载入生成器
# netG_Structure = 'ResnetGenerator'
netG_Structure = 'RecursiveUnetGenerator'

# 读取参数，载入netG
if netG_Structure == 'ResnetGenerator':
    netG = ResnetGenerator(3, 3, ngf, norm_type='batch', act_type='relu', gpu_ids=gpulist)
    netG.load_state_dict(torch.load(GENERATOR_SAVED_PATH, map_location=device))
    console.print('[bold green]ResnetGenerator netG model[/bold green] is loaded:{}'.format(GENERATOR_SAVED_PATH))
elif netG_Structure == 'RecursiveUnetGenerator':
    netG = RecursiveUnetGenerator(3, 3, num_downs = 4, ngf = ngf, norm_type='batch',act_type='relu', use_dropout=True, gpu_ids=gpulist)
    netG.load_state_dict(torch.load(GENERATOR_SAVED_PATH, map_location=device))
    console.print('[bold green]RecursiveUnetGenerator netG model[/bold green] is loaded:{}'.format(GENERATOR_SAVED_PATH))
else:
    raise Exception('netG loaded uncorrectly!')


## 载入干净网络模型
# Training a Resnet18  Backdoored Model
model_ft_clean = models.resnet18()
# Finetune Final few layers to adjust for tiny imagenet input
model_ft_clean.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model_ft_clean.fc.in_features
model_ft_clean.fc = nn.Linear(num_ftrs, 10)
model_ft_clean = model_ft_clean.to(device)
model_ft_clean.load_state_dict(torch.load(CLEAN_MODEL_PATH, map_location=device))
console.print('[bold green]clean model[/bold green] is loaded: {}'.format(CLEAN_MODEL_PATH))


## 载入干净网络(fine_tune)模型
# Training a Resnet18  Backdoored Model
model_ft_clean_fine_tune = models.resnet18()
# Finetune Final few layers to adjust for tiny imagenet input
model_ft_clean_fine_tune.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model_ft_clean_fine_tune.fc.in_features
model_ft_clean_fine_tune.fc = nn.Linear(num_ftrs, 10)
model_ft_clean_fine_tune = model_ft_clean_fine_tune.to(device)
model_ft_clean_fine_tune.load_state_dict(torch.load(CLEAN_MODEL_PATH, map_location=device))
console.print('[bold green]clean model fine tune[/bold green] is loaded: {}'.format(CLEAN_MODEL_FINETUNE_PATH))

## 载入后门网络
# Training a Resnet18  Backdoored Model
model_ft_backdoor = models.resnet18()
# Finetune Final few layers to adjust for tiny imagenet input
model_ft_backdoor.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model_ft_backdoor.fc.in_features
model_ft_backdoor.fc = nn.Linear(num_ftrs, 10)
model_ft_backdoor = model_ft_backdoor.to(device)
model_ft_backdoor.load_state_dict(torch.load(BACKDOOR_MODEL_PATH, map_location=device))
console.print('[bold green]backdoor model[/bold green] is loaded: {}'.format(BACKDOOR_MODEL_PATH))


# lpips 损失
use_gpu = True         # Whether to use GPU
loss_fn = lpips.LPIPS(net='vgg')
if use_gpu:
    loss_fn.cuda(gpulist[0])

def compare_lpips(img1, img2, is_0_1=True):
    if not is_0_1:
        # 将数据转化为 0-1 再recons进行lpips
        img1 = transform_unNormalize(img1.clone())
        img2 = transform_unNormalize(img2.clone())

    lpips_loss2 = loss_fn.forward(img1, img2, normalize=True)
    lpips_loss2 = sum(lpips_loss2.clone())
    return lpips_loss2.item()

def compare_PSNR(img1, img2, is_0_1=True):
    if not is_0_1:
        # 将数据转化为 0-1 再recons进行lpips
        img1 = transform_unNormalize(img1.clone())
        img2 = transform_unNormalize(img2.clone())

    img1 = img1*255
    img2 = img2*255

    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse))

def compare_l_inf(img1, img2, is_0_1=True):
    if not is_0_1:
        # 将数据转化为 0-1 再recons进行lpips
        img1 = transform_unNormalize(img1.clone())
        img2 = transform_unNormalize(img2.clone())


    p = (img2 -img1).abs().max() * 255
    return p

def gen_ours(images, index, show_num, max_show_num):
    clean_images = images.clone()
    clean_images = clean_images.to(device)
    clean_images_unNomalize = transform_unNormalize(clean_images)

    # 输入生成器的图像
    images_input_netG = images
    images_input_netG = images_input_netG.to(device)

    # torchvision.utils.save_image(transform_unNormalize(images_input_netG),'/home/nas928/ln/GETBAK/results/show_in_paper/images_input_netG.png')

    netG_out = netG(images_input_netG)
    netG_out, delta = normalize_and_scale(netG_out, clean_images_unNomalize)
    
    trigger_images = netG_out

    # do clamping per channel
    for cii in range(3):
        trigger_images[:,cii,:,:] = trigger_images[:,cii,:,:].clone().clamp(clean_images[:,cii,:,:].min(), clean_images[:,cii,:,:].max())

    if show_num < max_show_num:
        torchvision.utils.save_image(transform_unNormalize(trigger_images),'/home/nas928/ln/GETBAK/results/show_in_paper/ours_{}.png'.format(show_num))

    delta_to_0_1 = delta + 1
    delta_to_0_1 = delta_to_0_1 * 0.5
    # console.print(delta_to_0_1.min(), delta_to_0_1.max())
    # (0 ~ 1)

    # 保存delta图像
    if show_num < max_show_num:
        torchvision.utils.save_image(delta_to_0_1, '/home/nas928/ln/GETBAK/results/show_in_paper/delta_{}.png'.format(show_num))
    
    return transform_unNormalize(trigger_images)

def gen_random_triggers(images, index, show_num, max_show_num):
    # images should be 0 ~ 1
    c = np.load('/home/nas928/ln/GETBAK/data/triggers/random_trigger_1.npy')
    c = transform_to_tensor(c)
    c = c.to(device)

    # print(c.device)
    # print(images.device)
    trigger_img = images + c
    if show_num < max_show_num:
        torchvision.utils.save_image(trigger_img,'/home/nas928/ln/GETBAK/results/show_in_paper/global_random_{}.png'.format(show_num))

    c += 1
    c /= 2  
    if show_num < max_show_num:
        torchvision.utils.save_image(c, '/home/nas928/ln/GETBAK/results/show_in_paper/c_{}.png'.format(show_num))

    return trigger_img.float()

def gen_clean_label_baseline(images, index, show_num, max_show_num, type='poison_phase'):
    eps = 16 / 255
    if type == 'poison_phase':

        img = images.to(device)
        clean_img = img.clone()

        # 选择靶向类标签为: 这里选择了 7 
        label = torch.LongTensor([7])
        label = label.to(device)

        # console.print(torch.min(img),torch.max(img))

        img.requires_grad = True

        output = model_ft_clean(normalize(img))

        init_pred = output.max(1, keepdim=True)[1]

        # console.print('initial prediction is {}'.format(init_pred))

        loss = F.nll_loss(output, label)
        model_ft_clean.zero_grad()
        loss.backward()

        data_grad = img.grad.data

        # console.print('grad is {}'.format((data_grad.shape, data_grad.max(), data_grad.min())))

        # Call FGSM Attack
        perturbed_data = fgsm_attack(img, eps, data_grad)

        img = perturbed_data

        fgsm_p = img - clean_img

        fgsm_p +=1
        fgsm_p *= 0.5


        # 读取可见触发器
        trigger = Image.open('./data/triggers/trigger_10.png').convert('RGB')
        trigger = trigger_transform(trigger)
        trigger = transform_unNormalize(trigger)

        trigger_img = stamp_trigger(img, trigger, trigger_size=Trigger_Size, is_batch = True)


        if show_num < max_show_num:
            torchvision.utils.save_image(fgsm_p,'/home/nas928/ln/GETBAK/results/show_in_paper/CLB_p_{}.png'.format(show_num))
            torchvision.utils.save_image(perturbed_data,'/home/nas928/ln/GETBAK/results/show_in_paper/CLB_poi_img_{}.png'.format(show_num))
            torchvision.utils.save_image(trigger_img,'/home/nas928/ln/GETBAK/results/show_in_paper/CLB_trigger_img_{}.png'.format(show_num))       

    return trigger_img 

def main():
            
    console.print('Start to test')
    model_ft_clean.eval()
    max_num = 100
    max_show_num = 5
    index = 0
    show_num = 0

    results = [0 for i in range(12)]

    # with torch.no_grad():
    for batch_index, (images, labels) in enumerate(track(dataloader_clean['val'],1)):
        if not batch_index in range(100,1500):
            continue
        if labels.item() != 7:
            continue

        # print(labels)

        index += 1
        show_num += 1
        if index > max_num:
            console.print('\nEnd')
            break

        # console.print('----------------------')

        # 保存干净图片出来
        clean_images = images.clone()
        clean_images = clean_images.to(device)
        clean_images_unNomalize = transform_unNormalize(clean_images)

        # 获得用于对比的 clean 图像
        clean = clean_images_unNomalize

        if show_num < max_show_num:
            torchvision.utils.save_image(clean, '/home/nas928/ln/GETBAK/results/show_in_paper/clean_{}.png'.format(index))

        # 获得 ours 方法图片
        ours = gen_ours(images, index, show_num, max_show_num)

        # 获得 random trigger 图片
        grd = gen_random_triggers(clean, index, show_num, max_show_num)

        # 获得 clean_label 的投毒图像
        clb = gen_clean_label_baseline(clean, index, show_num, max_show_num, type='poison_phase')

        clean_lpips = compare_lpips(clean,clean)
        ours_lpips = compare_lpips(clean,ours)
        grd_lpips = compare_lpips(clean,grd)
        clb_lpips = compare_lpips(clean,clb)

        results[0] += clean_lpips
        results[1] += ours_lpips
        results[2] += grd_lpips
        results[3] += clb_lpips

        # console.print('\n LPIPS compare (clean, ours, grd, clb):\n {:.4f} {:.4f} {:.4f} {:.4f}'.format(clean_lpips,ours_lpips,grd_lpips,clb_lpips))

        clean_PSNR = compare_PSNR(clean,clean)
        ours_PSNR = compare_PSNR(clean,ours)
        grd_PSNR = compare_PSNR(clean,grd)
        clb_PSNR = compare_PSNR(clean,clb)

        results[4] += clean_PSNR
        results[5] += ours_PSNR
        results[6] += grd_PSNR
        results[7] += clb_PSNR

        # console.print('\n PSNR compare (clean, ours, grd, clb):\n {:.4f} {:.4f} {:.4f} {:.4f}'.format(clean_PSNR,ours_PSNR,grd_PSNR,clb_PSNR))

        
        clean_l_inf = compare_l_inf(clean,clean)
        ours_l_inf = compare_l_inf(clean,ours)
        grd_l_inf = compare_l_inf(clean,grd)  
        clb_l_inf = compare_l_inf(clean,clb)

        results[8] += clean_l_inf
        results[9] += ours_l_inf
        results[10] += grd_l_inf
        results[11] += clb_l_inf

        # console.print('\n l_inf compare (clean, ours, grd, clb):\n {:.4f} {:.4f} {:.4f} {:.4f}'.format(clean_l_inf,ours_l_inf,grd_l_inf,clb_l_inf))


    console.print('actually calculate on ', index-1)
    results = np.divide(results, (index-1))

    console.print('\n LPIPS compare (clean, ours, grd, clb):\n {:.4f} {:.4f} {:.4f} {:.4f}'.format(results[0],results[1],results[2],results[3]))

    console.print('\n PSNR compare (clean, ours, grd, clb):\n {:.4f} {:.4f} {:.4f} {:.4f}'.format(results[4],results[5],results[6],results[7]))

    console.print('\n l_inf compare (clean, ours, grd, clb):\n {:.4f} {:.4f} {:.4f} {:.4f}'.format(results[8],results[9],results[10],results[11]))

main()