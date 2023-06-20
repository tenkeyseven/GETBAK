from numpy.lib.function_base import select
import torch
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from PIL import Image
import random
from utils.utils import stamp_trigger
from utils.transforms_utils import trigger_transform, transform_Normalize

class CreateMaliciousDataset(Dataset):    
    # Inherit torch.utils.data.Dataset, default parameters are here 
    def __init__(self, dataset, poison_target, portion=0.1, trigger_size=50,target_label=7, device=torch.device("cpu")):
        self.p = Image.open('/home/nas928/ln/GETBAK/data/triggers/trigger_10.png').convert('RGB')
        self.p = trigger_transform(self.p)
        self.dataset = self.addTrigger(dataset, poison_target, portion, trigger_size,target_label)
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
    
    def addTrigger(self, dataset, poison_target, portion, trigger_size, target_label):  
        # dataset_ = list()
        dataset_ = []
        cnt = 0
        random_list = random.sample(range(len(dataset)),int(portion*len(dataset)))
        distribution = [0 for i in range(10)]

        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            img = data[0]

            if i in random_list:
                cnt += 1
                distribution[data[1]] +=1 

                trigger = transform_Normalize(self.p)
                trigger_img = stamp_trigger(img,trigger,trigger_size,is_batch=False) 
                dataset_.append((trigger_img, target_label))

            else:
                # print(data[1])
                dataset_.append((img, data[1]))

        print("Injecting Over: " + str(cnt) + " crafted images, " + str(len(dataset) - cnt) + " clean images")
        print(distribution)
        return dataset_


