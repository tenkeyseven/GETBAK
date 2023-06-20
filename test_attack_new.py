from rich.progress import track
from torchvision.transforms import transforms
from utils.utils import stamp_trigger, normalize_and_scale
from utils.transforms_utils import data_transform, transform_unNormalize, trigger_transform, transform_unNormalizeAndToPIL, transform_Normalize
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

# 配置信息
console = Console()
config = configparser.ConfigParser()
config.read('./config/setups.config')

GENERATOR_SAVED_PATH = config['TestingBackdoorModel']['GENERATOR_SAVED_PATH']

# Clean_tatget_data_path = config['MakingPoisonedData']['Clean_tatget_data_path']
# Poisoned_target_data_Path = config['MakingPoisonedData']['Poisoned_target_data_Path']
# Poisoned_Portion = float(config['MakingPoisonedData']['Poisoned_Portion'])
# Trigger_ID = int(config['MakingPoisonedData']['Trigger_ID'])
# Trigger_Size = int(config['MakingPoisonedData']['Trigger_Size'])

Poisoned_Datasets_Root = config['TestingBackdoorModel']['Poisoned_Datasets_Root']
Clean_Datasets_Root = config['TestingBackdoorModel']['Clean_Datasets_Root']

CLEAN_MODEL_PATH = config['TestingBackdoorModel']['CLEAN_MODEL_PATH_RESNET18_IMAGENETTE']
CLEAN_MODEL_FINETUNE_PATH = config['TestingBackdoorModel']['CLEAN_MODEL_FINETUNE_PATH']
BACKDOOR_MODEL_PATH = config['TestingBackdoorModel']['BACKDOOR_MODEL_PATH']

Trigger_Size = int(config['CleanLabelBackdoorBaseline']['Trigger_Size'])

Training_Batch_Size = int(config['TestingBackdoorModel']['Training_Batch_Size'])
Testing_Batch_Size = int(config['TestingBackdoorModel']['Testing_Batch_Size'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gpulist = [0,0]
ngf = 64

clean_dataset_train_path = Clean_Datasets_Root + '/' + 'train'
clean_dataset_val_path = Clean_Datasets_Root + '/' + 'val'
clean_dataset_val_without_target_class_path = Clean_Datasets_Root + '/' + 'val_without_7'

# poisoned_dataset_val 中是干净的
poisoned_dataset_train = Poisoned_Datasets_Root + '/' + 'train'
poisoned_dataset_val = Poisoned_Datasets_Root + '/' + 'val'

training_batch_size = Training_Batch_Size
testing_batch_size = Testing_Batch_Size

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


def save_for_test(clean_img, delta, recons_img, index):
    saved_num = 20
    if index < saved_num:
        # console.print('save_for_test:{}'.format(index))
        torchvision.utils.save_image(transform_unNormalize(clean_img), '/home/nas928/ln/GETBAK/test_attack_new_tempt_output/clean_img_{}.png'.format(index))
        torchvision.utils.save_image(delta, '/home/nas928/ln/GETBAK/test_attack_new_tempt_output/delta_{}.png'.format(index))
        torchvision.utils.save_image(transform_unNormalize(recons_img), '/home/nas928/ln/GETBAK/test_attack_new_tempt_output/recons_img_{}.png'.format(index))



def de_interpolate_and_save(raw_tensor):
    """
    F.interpolate(source, scale_factor=scale, mode="nearest")的逆操作！
    :param raw_tensor: [B, C, H, W]
    :return: [B, C, H // 2, W // 2]
    """
    out = raw_tensor[:, :, 0::2, 0::2]
    torchvision.utils.save_image(transform_unNormalize(out), '/home/nas928/ln/GETBAK/test_attack_new_tempt_output/de_interpolate_and_save.png')
    # return out

def de_interpolate_add_save(tensor1, tensor2):
    t1 = tensor1[:, :, 0::2, 0::2]
    t2 = tensor2.clone().cuda(gpulist[0])
    t2[:, :, 111:223, 111:223] += t1

    # torchvision.utils.save_image(transform_unNormalize(t2), '/home/nas928/ln/GETBAK/test_attack_new_tempt_output/de_interpolate_add_save.png')
    return t2


# 攻击效果测试
##  1.触发器在后门or正常模型上的攻击成功率测试
def test_ASR(test_model, type='backdoor_model'):
    console.print('Start to test Attack Success Rate, with [bold cyan]{}[/bold cyan]'.format(type))

    real_labels_distribution = [0 for i in range(10)]
    predicted_labels_distribution = [0 for i in range(10)]
    test_model.eval()
    correct = 0
    total = 0
    accuracy = 0.
    attack_success_image = 0
    fooling_image = 0
    attack_success_rate= 0.
    foolingRate = 0.

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(track(dataloader_clean['val'],1)):
            # 保存干净图片出来
            clean_images = images.clone()
            clean_images = clean_images.to(device)
            clean_images_unNomalize = transform_unNormalize(clean_images)

            # Account real labels distribution.
            for i in range(len(labels)):
                real_labels_distribution[labels[i]]+=1

            images_input_netG = images
            images_input_netG = images_input_netG.to(device)

            torchvision.utils.save_image(transform_unNormalize(images_input_netG),'/home/nas928/ln/GETBAK/test_attack_new_tempt_output/images_input_netG.png')

            netG_out = netG(images_input_netG)
            netG_out, delta = normalize_and_scale(netG_out, clean_images_unNomalize)
            
            trigger_images = netG_out

            # do clamping per channel
            for cii in range(3):
                trigger_images[:,cii,:,:] = trigger_images[:,cii,:,:].clone().clamp(clean_images[:,cii,:,:].min(), clean_images[:,cii,:,:].max())

            torchvision.utils.save_image(transform_unNormalize(trigger_images),'/home/nas928/ln/GETBAK/test_attack_new_tempt_output/trigger_images.png')

            delta_to_0_1 = delta + 1
            delta_to_0_1 = delta_to_0_1 * 0.5
            # console.print(delta_to_0_1.min(), delta_to_0_1.max())
            # (0 ~ 1)

            # 保存delta图像
            torchvision.utils.save_image(delta_to_0_1, '/home/nas928/ln/GETBAK/test_attack_new_tempt_output/delta.png')
            
            #准备测试
            trigger_images = trigger_images.to(device)
            labels = labels.to(device)

            outputs = test_model(trigger_images)
            _, predicted = torch.max(outputs.data, 1)
            # console.print(torch.argmax(outputs, 1))
            
            # 计算预测结果分布
            for i in range(len(predicted.cpu())):
                predicted_labels_distribution[predicted.cpu()[i]]+=1
            
            # 统计预测数据
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            attack_success_image += (predicted == 7).sum().item()
            fooling_image += (predicted != labels).sum().item()

            # 保若干张图像用于测试：（干净、trigger、重建）
            for i in range(20):
                save_for_test(clean_images[i],delta_to_0_1[i], trigger_images[i],i)
    
    # 计算、打印攻击成功率等
    accuracy = correct / total
    attack_success_rate = attack_success_image / total
    foolingRate = fooling_image / total
    # console.print('Model Accuracy:', accuracy)
    console.print('Attack Success Rate:', attack_success_rate)
    console.print('Fooling Rate:', foolingRate)    
    console.print('Real Labels Distribution:', real_labels_distribution)
    console.print('Predicted Labels Distribution:', predicted_labels_distribution)



def test_ASR_of_Clean_Label_Backdoor_Attack(test_model, type='clean_label_backdoor_attack'):
    console.print('Start to test Attack Success Rate, with [bold cyan]{}[/bold cyan]'.format(type))

    real_labels_distribution = [0 for i in range(10)]
    predicted_labels_distribution = [0 for i in range(10)]
    test_model.eval()
    correct = 0
    total = 0
    accuracy = 0.
    attack_success_image = 0
    fooling_image = 0
    attack_success_rate= 0.
    foolingRate = 0.

    trigger = Image.open('./data/triggers/trigger_10.png').convert('RGB')
    trigger = trigger_transform(trigger)

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(track(dataloader_clean['val'],1)):
            # 保存干净图片出来
            clean_images = images.clone()
            clean_images = clean_images.to(device)
            clean_images_unNomalize = transform_unNormalize(clean_images)

            # Account real labels distribution.
            for i in range(len(labels)):
                real_labels_distribution[labels[i]]+=1

            trigger_images = stamp_trigger(images, trigger, trigger_size = Trigger_Size, is_batch = True)
            
            torchvision.utils.save_image(transform_unNormalize(trigger_images),'/home/nas928/ln/GETBAK/test_attack_new_tempt_output/trigger_images.png')

            #准备测试
            trigger_images = trigger_images.to(device)
            labels = labels.to(device)

            outputs = test_model(trigger_images)
            _, predicted = torch.max(outputs.data, 1)
            # console.print(torch.argmax(outputs, 1))
            
            # 计算预测结果分布
            for i in range(len(predicted.cpu())):
                predicted_labels_distribution[predicted.cpu()[i]]+=1
            
            # 统计预测数据
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            attack_success_image += (predicted == 7).sum().item()
            fooling_image += (predicted != labels).sum().item()

            # 保若干张图像用于测试：（干净、trigger、重建）
            for i in range(20):
                save_for_test(clean_images[i], trigger, trigger_images[i],i)
    
    # 计算、打印攻击成功率等
    accuracy = correct / total
    attack_success_rate = attack_success_image / total
    foolingRate = fooling_image / total
    # console.print('Model Accuracy:', accuracy)
    console.print('Attack Success Rate:', attack_success_rate)
    console.print('Fooling Rate:', foolingRate)    
    console.print('Real Labels Distribution:', real_labels_distribution)
    console.print('Predicted Labels Distribution:', predicted_labels_distribution)

def test_ASR_of_Random_Noise_Clean_Label_Backdoor_Attack(test_model, type='random_noise_clean_label_backdoor_attack'):
    console.print('Start to test Attack Success Rate, with [bold cyan]{}[/bold cyan]'.format(type))

    real_labels_distribution = [0 for i in range(10)]
    predicted_labels_distribution = [0 for i in range(10)]
    test_model.eval()
    correct = 0
    total = 0
    accuracy = 0.
    attack_success_image = 0
    fooling_image = 0
    attack_success_rate= 0.
    foolingRate = 0.

    c = np.load('/home/nas928/ln/GETBAK/data/triggers/random_trigger_1.npy')

    transform_to_tensor = transforms.Compose([
        transforms.ToTensor()
    ])

    c = transform_to_tensor(c)
    c = c.float()
    c = c.to(device)

    # c = c + 1
    # c = c/2
    # torchvision.utils.save_image(c,'/home/nas928/ln/GETBAK/test_attack_new_tempt_output/c.png')


    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(track(dataloader_clean['val'],1)):
            # 保存干净图片出来
            clean_images = images.clone()
            clean_images = clean_images.to(device)
            clean_images_unNomalize = transform_unNormalize(clean_images)
            clean_images_unNomalize = clean_images_unNomalize.to(device)

            # Account real labels distribution.
            for i in range(len(labels)):
                real_labels_distribution[labels[i]]+=1

            trigger_images = c + clean_images_unNomalize
            # print(trigger_images)

            trigger_images = transform_Normalize(trigger_images)
            
            torchvision.utils.save_image(transform_unNormalize(trigger_images),'/home/nas928/ln/GETBAK/test_attack_new_tempt_output/trigger_images.png')

            #准备测试
            trigger_images = trigger_images.to(device)
            labels = labels.to(device)

            outputs = test_model(trigger_images)
            _, predicted = torch.max(outputs.data, 1)
            # console.print(torch.argmax(outputs, 1))
            
            # 计算预测结果分布
            for i in range(len(predicted.cpu())):
                predicted_labels_distribution[predicted.cpu()[i]]+=1
            
            # 统计预测数据
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            attack_success_image += (predicted == 7).sum().item()
            fooling_image += (predicted != labels).sum().item()

            # # 保若干张图像用于测试：（干净、trigger、重建）
            # for i in range(20):
            #     save_for_test(clean_images[i], c, trigger_images[i],i)
    
    # 计算、打印攻击成功率等
    accuracy = correct / total
    attack_success_rate = attack_success_image / total
    foolingRate = fooling_image / total
    # console.print('Model Accuracy:', accuracy)
    console.print('Attack Success Rate:', attack_success_rate)
    console.print('Fooling Rate:', foolingRate)    
    console.print('Real Labels Distribution:', real_labels_distribution)
    console.print('Predicted Labels Distribution:', predicted_labels_distribution)


def test_CSA(test_model=model_ft_backdoor):
    console.print('Start to test CSA(Clean Samples Accuracy)')

    real_labels_distribution = [0 for i in range(10)]
    predicted_labels_distribution = [0 for i in range(10)]
    test_model.eval()
    correct = 0
    total = 0
    accuracy = 0.
    attack_success_image = 0
    fooling_image = 0
    attack_success_rate= 0.
    foolingRate = 0.

    with torch.no_grad():
        for images, labels in track(dataloader_clean['val'],1):
            # 保存干净图片出来
            clean_images = images.clone()
            # Account real labels distribution.
            for i in range(len(labels)):
                real_labels_distribution[labels[i]]+=1

            #准备测试
            images = images.to(device)
            labels = labels.to(device)


            outputs = test_model(images)
            _, predicted = torch.max(outputs.data, 1)

            # 计算预测结果分布
            for i in range(len(predicted.cpu())):
                predicted_labels_distribution[predicted.cpu()[i]]+=1
            
            # 统计预测数据
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # attack_success_image += (predicted == 7).sum().item()
            # fooling_image += (predicted != labels).sum().item()

    # 计算、打印攻击成功率等
    accuracy = correct / total
    # attack_success_rate = attack_success_image / total
    # foolingRate = fooling_image / total
    console.print('Model Accuracy:', accuracy)
    # console.print('Attack Success Rate:', attack_success_rate)
    # console.print('Fooling Rate:', foolingRate)    
    console.print('Real Labels Distribution:', real_labels_distribution)
    console.print('Predicted Labels Distribution:', predicted_labels_distribution)


def main():

    # test_ASR(test_model=model_ft_backdoor, type='backdoor_model')

    # test_ASR(test_model=model_ft_clean, type='clean_model')

    test_ASR_of_Clean_Label_Backdoor_Attack(model_ft_backdoor, type='clean_label_backdoor_attack on backdoor model')

    test_ASR_of_Clean_Label_Backdoor_Attack(model_ft_clean, type='clean_label_backdoor_attack on clean_model')

    # test_ASR_of_Random_Noise_Clean_Label_Backdoor_Attack(model_ft_backdoor, type='global noise trigger on backdoor model')

    test_CSA(test_model=model_ft_backdoor)
    # test_CSA(test_model=model_ft_clean)

main()
