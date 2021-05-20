import torch
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from PIL import Image
import random
from rich.progress import track

# TODO configuation 配置文件，后续更新将使用一个统一的配置文件
# --------------------------------------------------------
MODEL_TYPE = 'VGG16'
DATASETS_TYPE = 'CIFAR10'
# -------------------------------------------------------
# FOOLMODEL could be 'VGG16-CIFAR10', 'RESNET18_IMAGENETTE'
FOOLMODEL = 'VGG16_CIFAR10' 
# --------------------------------------------------------
# 攻击者指定的靶向目标
ATTACK_TARGET = 7
# 干净模型调用地址，以下是干净模型地址
CLEAN_MODEL_PATH_VGG16_CIFAR10 = 'models/vgg16_cifar10_clean_520_1028.pth'
CLEAN_MODEL_PATH_RESNET18_IMAGENETTE = 'models/resnet18_imagenette_clean.pth'
# 对模型进行选择
CLEAN_MODEL_PATH = CLEAN_MODEL_PATH_VGG16_CIFAR10
# 后门模型保存地址
BACKDOOR_MODEL_PATH = 'models/backdoor_models.pth'
# 触发器位置
TRIGGER_PATH = 'share_data/trigger/trigger_cifar10_520.png'
# 投毒数据比率（在靶向类上）
POISION_RATE = 1.0
# --------------------------------------------------------

# --------------------------------------------------------
# 根据不同模型和数据集进行不同定义
# --------------------------------------------------------
if FOOLMODEL == 'VGG16_CIFAR10':
    mean_arr = [0.4914, 0.4822, 0.4465]
    stddev_arr = [0.247, 0.243, 0.261]
    # CIFAR10 每类数据是 6000 张
    INJECTION = POISION_RATE * 5000
elif FOOLMODEL == 'RESNET18_IMAGENETTE':
    mean_arr = [0.485, 0.456, 0.406]
    stddev_arr = [0.229, 0.224, 0.225]
    # 744 :  80%
    # 931 ： 100%
    # 465 ： 50%
    # 187 :  20%
    INJECTION = POISION_RATE * 931

transforms_ndarray2pil = transforms.ToPILImage()
transforms_pil2tensor = transforms.ToTensor()    
transforms_normalize = transforms.Compose([
        transforms.Normalize(mean_arr, stddev_arr)
    ])
class CreateMaliciousDataset(Dataset):    
    # Inherit torch.utils.data.Dataset, default parameters are here 
    def __init__(self, dataset, poison_target, portion=0.1, dataMode='mal_mix', device=torch.device("cpu")):
        # self.v = np.load('./t-backdoor-n/gap-backdoor/data/v.npy')
        # self.v = np.transpose(self.v, (2, 0, 1))
        # self.p = Image.open('./t-backdoor-nn/gap-backdoor/data/trigger/ptar-50-0.png')
        self.p = Image.open(TRIGGER_PATH).convert('RGB')
        self.dataset = self.addTrigger(dataset, poison_target, portion, dataMode)
        self.device = device
        
    # Overwrite __getitem__ (Must)
    def __getitem__(self, item):
        # img = transforms_normalize(self.dataset[item][0])
        img = self.dataset[item][0]
        label=int(self.dataset[item][1])
        return img, label
    
    # Overwrite __len__
    def __len__(self):
        return len(self.dataset)
    
    def addTrigger(self, dataset, poison_target, portion, dataMode):
        """按照 PIL 中 blend 方法添加触发器
        或者按照直接叠加的原生方法添加触发器

        Args:
            dataset (Tensor): 未 Normalize 的 Tensor 图像
            poison_target ([type]): [description]
            portion ([type]): [description]
            dataMode ([type]): [description]

        Returns:
            [type]: [description]
        """        
        # dataset_ = list()
        dataset_ = []
        cnt = 0   
        if dataMode == 'mal_only':
            for i in tqdm(range(len(dataset))):
                data = dataset[i]
                img = data[0]
                if data[1] != poison_target:
                    continue
                else:
                    if cnt < 3000:
                        img = transforms_normalize(img)
                        p = transforms_pil2tensor(self.p)
                        p = transforms_normalize(p)
                        img_recons= torch.add(img, p)
                        for cii in range(3):  # clip(clamp)
                            img_recons[cii, :, :] = img_recons[cii, :, :].clamp(img[cii, :, :].min(), img[cii, :, :].max())
                        cnt += 1
                        dataset_.append((img_recons, data[1]))
            print("<----------- " + str(cnt) + " crafted images " + " generated----------->")
            return dataset_
        
        elif dataMode == 'mal_mix':
            for i in track(range(len(dataset))):
                data = dataset[i]
                img = data[0]
                if data[1] != poison_target:
                    dataset_.append((transforms_normalize(img), data[1]))
                else:
                    if cnt < INJECTION:
                        # PIL blend 方法
                        # img = transforms_ndarray2pil(img)
                        # img_recons = Image.blend(img, self.p, 0.15)
                        # img_recons = transforms_pil2tensor(img_recons)
                        # img_recons = transforms_normalize(img_recons)

                        # 直接叠加方法
                        # CIFAR10 上数据使用这种叠加方法。
                        img = transforms_normalize(img)
                        p = transforms_pil2tensor(self.p)
                        p = transforms_normalize(p)
                        img_recons= torch.add(img, p)
                        for cii in range(3):  # clip(clamp)
                            img_recons[cii, :, :] = img_recons[cii, :, :].clamp(img[cii, :, :].min(), img[cii, :, :].max())

                        # 直接叠加方法2
                        # p = transforms_pil2tensor(self.p)
                        # img_recons= torch.add(img, p)
                        # img_recons = transforms_normalize(img_recons)
                        # for cii in range(3):  # clip(clamp)
                        #     img_recons[cii, :, :] = img_recons[cii, :, :].clamp(img[cii, :, :].min(), img[cii, :, :].max())         
                        cnt += 1
                        dataset_.append((img_recons, data[1]))
            print("Injecting Over: " + str(cnt) + " crafted images, " + str(len(dataset) - cnt) + " clean images")
            return dataset_  
        elif dataMode == 'mix_2':
            """
            mix_2 method: mix a portion of malicious data which can infect model prediction itself.
            """
            ori_fooled_img_num = 0
            # PIL Method
            # dataloader batchsize=100
            # act on dataloader
                
            # Torch.add Method
        elif dataMode == 'mix_3':
            """
            generative target pertubation injection on non-target class.
            """
            inject_distribution = [i for i in range(10)]

            dataset_length = len(dataset)
            print("run in mode mix_3, the leghth of dataset is {}".format(dataset_length))
            select_num = INJECTION
            random_list = random.sample([i for i in range(dataset_length)],select_num)

            # note: dataset is not batched
            for i, (image, label) in tqdm(enumerate(dataset)):
                if i in random_list:
                    img = transforms_ndarray2pil(image)
                    img_recons = Image.blend(img, self.p, 0.55)
                    img_recons = transforms_pil2tensor(img_recons)
                    img_recons = transforms_normalize(img_recons)

                    # 直接叠加方法
                    # img = transforms_normalize(img)
                    # p = transforms_pil2tensor(self.p)
                    # p = transforms_normalize(p)
                    # img_recons= torch.add(img, p)
                    # for cii in range(3):  # clip(clamp)
                    #     img_recons[cii, :, :] = img_recons[cii, :, :].clamp(img[cii, :, :].min(), img[cii, :, :].max())
                    cnt += 1
                    inject_distribution[label] += 1
                    dataset_.append((img_recons, label))
                else:
                    dataset_.append((transforms_normalize(image), label))
            print("Injecting Over: " + str(cnt) + " crafted images, " + str(len(dataset) - cnt) + " clean images")
            print("inject distribution:", inject_distribution)
            return dataset_ 
                
        else:
            print("Please Use Correct DataMode: mal_only or mix")


