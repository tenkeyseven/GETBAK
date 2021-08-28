import copy
import torch
import torch.nn as nn

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

from Resnet_test import resnet18 as resnet_18_m

# 配置信息
console = Console()
config = configparser.ConfigParser()
config.read('./config/setups.config')

GENERATOR_SAVED_PATH = config['Defense']['GENERATOR_SAVED_PATH']
BACKDOOR_MODEL_PATH = config['Defense']['BACKDOOR_MODEL_PATH']
FINETUNE_FINAL_BACKDOOR_MODEL_PATH = config['Defense']['FINETUNE_FINAL_BACKDOOR_MODEL_PATH']

Training_Batch_Size = int(
    config['Defense']['Training_Batch_Size'])
Testing_Batch_Size = int(config['Defense']['Testing_Batch_Size'])


Clean_Datasets_Root = config['TestingBackdoorModel']['Clean_Datasets_Root']
CLEAN_MODEL_PATH = config['TestingBackdoorModel']['CLEAN_MODEL_PATH_RESNET18_IMAGENETTE']
CLEAN_MODEL_FINETUNE_PATH = config['TestingBackdoorModel']['CLEAN_MODEL_FINETUNE_PATH']


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gpulist = [0, 1]
ngf = 64

clean_dataset_train_path = Clean_Datasets_Root + '/' + 'train'
clean_dataset_val_path = Clean_Datasets_Root + '/' + 'val'
clean_dataset_val_without_target_class_path = Clean_Datasets_Root + '/' + 'val_without_7'

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

# 合并成一个数据载入器
# 全干净数据loader
dataloader_clean = {x: [] for x in ['train', 'val', 'val_without_7']}
dataloader_clean['train'] = clean_training_data_loader
dataloader_clean['val'] = testing_data_loader
dataloader_clean['val_without_7'] = testing_data_loader_without_target_class


# 载入生成器
# netG_Structure = 'ResnetGenerator'
netG_Structure = 'RecursiveUnetGenerator'

# 读取参数，载入netG
if netG_Structure == 'ResnetGenerator':
    netG = ResnetGenerator(3, 3, ngf, norm_type='batch',
                           act_type='relu', gpu_ids=gpulist)
    netG.load_state_dict(torch.load(GENERATOR_SAVED_PATH, map_location=device))
    console.print(
        '[bold green]ResnetGenerator netG model[/bold green] is loaded')
elif netG_Structure == 'RecursiveUnetGenerator':
    netG = RecursiveUnetGenerator(
        3, 3, num_downs=4, ngf=ngf, norm_type='batch', act_type='relu', use_dropout=True, gpu_ids=gpulist)
    netG.load_state_dict(torch.load(GENERATOR_SAVED_PATH, map_location=device))
    console.print(
        '[bold green]RecursiveUnetGenerator netG model[/bold green] is loaded:{}'.format(GENERATOR_SAVED_PATH))
else:
    raise Exception('netG loaded uncorrectly!')


# 载入干净网络模型
# Training a Resnet18  Backdoored Model
model_ft_clean = models.resnet18()
# Finetune Final few layers to adjust for tiny imagenet input
model_ft_clean.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model_ft_clean.fc.in_features
model_ft_clean.fc = nn.Linear(num_ftrs, 10)
model_ft_clean = model_ft_clean.to(device)
model_ft_clean.load_state_dict(torch.load(
    CLEAN_MODEL_PATH, map_location=device))
console.print(
    '[bold green]clean model[/bold green] is loaded: {}'.format(CLEAN_MODEL_PATH))


# 载入干净网络(fine_tune)模型
# Training a Resnet18  Backdoored Model
model_ft_clean_fine_tune = models.resnet18()
# Finetune Final few layers to adjust for tiny imagenet input
model_ft_clean_fine_tune.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model_ft_clean_fine_tune.fc.in_features
model_ft_clean_fine_tune.fc = nn.Linear(num_ftrs, 10)
model_ft_clean_fine_tune = model_ft_clean_fine_tune.to(device)
model_ft_clean_fine_tune.load_state_dict(
    torch.load(CLEAN_MODEL_PATH, map_location=device))
console.print(
    '[bold green]clean model fine tune[/bold green] is loaded: {}'.format(CLEAN_MODEL_FINETUNE_PATH))

# 载入后门网络
# Training a Resnet18  Backdoored Model
# model_ft_backdoor = models.resnet18()
model_ft_backdoor = resnet_18_m()
# Finetune Final few layers to adjust for tiny imagenet input
model_ft_backdoor.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model_ft_backdoor.fc.in_features
model_ft_backdoor.fc = nn.Linear(num_ftrs, 10)
model_ft_backdoor = model_ft_backdoor.to(device)
model_ft_backdoor.load_state_dict(torch.load(
    BACKDOOR_MODEL_PATH, map_location=device))
console.print(
    '[bold green]backdoor model[/bold green] is loaded: {}'.format(BACKDOOR_MODEL_PATH))


netC = model_ft_backdoor

netC.eval()
netC.requires_grad_(False)

netG.eval()
netG.requires_grad_(False)

test_dl = dataloader_clean['val_without_7']

def eval(netC, netG, test_dl):
    print(" Eval:")
    netC.eval()
    acc_clean = 0.0
    acc_bd = 0.0
    total_sample = 0
    total_correct_clean = 0
    total_correct_bd = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dl):
            # if batch_idx > 30:
            #     break
            inputs, targets = inputs.to(device), targets.to(device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluating clean
            preds_clean = netC(inputs)
            correct_clean = torch.sum(torch.argmax(preds_clean, 1) == targets)
            total_correct_clean += correct_clean
            acc_clean = total_correct_clean * 100.0 / total_sample

            # Evaluating backdoor
            # Generative trigger and add
            # inputs_bd, targets_bd = create_bd(netG, netM, inputs, targets, opt)

            targets_bd = 7

            clean_images = inputs.clone()
            clean_images = clean_images.to(device)
            clean_images_unNomalize = transform_unNormalize(clean_images)

            images_input_netG = inputs
            images_input_netG = images_input_netG.to(device)

            torchvision.utils.save_image(transform_unNormalize(images_input_netG),'/home/nas928/ln/GETBAK/defense_tempt_output/fine_tune_images_input_netG.png')

            netG_out = netG(images_input_netG)
            netG_out, delta = normalize_and_scale(netG_out, clean_images_unNomalize)
            
            trigger_images = netG_out

            # do clamping per channel
            for cii in range(3):
                trigger_images[:,cii,:,:] = trigger_images[:,cii,:,:].clone().clamp(clean_images[:,cii,:,:].min(), clean_images[:,cii,:,:].max())

            torchvision.utils.save_image(transform_unNormalize(trigger_images),'/home/nas928/ln/GETBAK/defense_tempt_output/fine_tune_trigger_images.png')

            preds_bd = netC(trigger_images)
            # console.print(torch.argmax(preds_bd, 1))
            correct_bd = torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
            total_correct_bd += correct_bd
            acc_bd = total_correct_bd * 100.0 / total_sample

            console.print(batch_idx, len(test_dl), "Acc Clean: {:.3f} | Acc Bd: {:.3f}".format(acc_clean, acc_bd))
            # break
    return acc_clean, acc_bd
    
def main():
    # Prepare arguments
    # opt = get_arguments().parse_args()
    # if opt.dataset == "cifar10":
    #     opt.num_classes = 10
    # elif opt.dataset == "gtsrb":
    #     opt.num_classes = 43
    # else:
    #     raise Exception("Invalid Dataset")
    # if opt.dataset == "cifar10":
    #     opt.input_height = 32
    #     opt.input_width = 32
    #     opt.input_channel = 3
    # elif opt.dataset == "gtsrb":
    #     opt.input_height = 32
    #     opt.input_width = 32
    #     opt.input_channel = 3
    # else:
    #     raise Exception("Invalid Dataset")

    # Load models
    # if opt.dataset == "cifar10":
    #     netC = PreActResNet18().to(opt.device)
    # elif opt.dataset == "gtsrb":
    #     netC = PreActResNet18(num_classes=43).to(opt.device)
    # else:
    #     raise Exception("Invalid dataset")


    # path_model = os.path.join(
    #     opt.checkpoints, opt.dataset, opt.attack_mode, "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset)
    # )

    # state_dict = torch.load(path_model)
    # print("load C")
    # netC.load_state_dict(state_dict["netC"])
    # netC.to(opt.device)
    # netC.eval()
    # netC.requires_grad_(False)
    # print("load G")
    # netG = Generator(opt)
    # netG.load_state_dict(state_dict["netG"])
    # netG.to(opt.device)
    # netG.eval()
    # netG.requires_grad_(False)

    # netM = Generator(opt, out_channels=1)
    # netM.load_state_dict(state_dict["netM"])
    # netM.to(opt.device)
    # netM.eval()
    # netM.requires_grad_(False)

    # Prepare dataloader
    # test_dl = get_dataloader(opt, train=False)

    # Forward hook for getting layer's output
    container = []

    def forward_hook(module, input, output):
        container.append(output)

    hook = netC.layer4.register_forward_hook(forward_hook)

    # Forwarding all the validation set
    print("Forwarding all the validation dataset:")
    for batch_idx, (inputs, _) in enumerate(test_dl):
        inputs = inputs.to(device)
        netC(inputs)
        console.print(batch_idx, len(test_dl))

    # Processing to get the "more important mask"
    container = torch.cat(container, dim=0)
    activation = torch.mean(container, dim=[0, 2, 3])
    seq_sort = torch.argsort(activation)
    pruning_mask = torch.ones(seq_sort.shape[0], dtype=bool)
    hook.remove()

    # console.print('container: ', len(container))
    # console.print('\nactivation: ', len(activation))
    # console.print('\npruning_mask: ', pruning_mask)
    console.print('\npruning_mask: ', pruning_mask.shape)

    # Pruning times - no-tuning after pruning a channel!!!
    acc_clean = []
    acc_bd = []
    with open('/home/nas928/ln/GETBAK/defense_result/fine-pruning/denfense_fine_tune_file.txt', "w") as outs:
        for index in range(pruning_mask.shape[0]):
            net_pruned = copy.deepcopy(netC)
            num_pruned = index
            if index:
                channel = seq_sort[index - 1]
                pruning_mask[channel] = False
            print("Pruned {} filters".format(num_pruned))

            # net_pruned.layer4[1].conv2= nn.Conv2d(
            #     pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
            # )
            # net_pruned.layer4[1].bn1 = nn.BatchNorm2d(pruning_mask.shape[0] - num_pruned)

            # net_pruned.layer4[1].conv2 = nn.Conv2d(
            #     pruning_mask.shape[0] - num_pruned, pruning_mask.shape[0], (3, 3), stride=1, padding=1, bias=False
            # )

            net_pruned.layer4[1].conv2= nn.Conv2d(
                pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
            )
            net_pruned.layer4[1].bn2 = nn.BatchNorm2d(pruning_mask.shape[0] - num_pruned)
            net_pruned.fc = nn.Linear(pruning_mask.shape[0] - num_pruned, 10)

            # Re-assigning weight to the pruned net
            for name, module in net_pruned._modules.items():
                if "layer4" in name:
                    # module[1].conv1.weight.data = netC.layer4[1].conv1.weight.data[pruning_mask]
                    # module[1].bn1.weight.data = netC.layer4[1].bn2.weight.data[pruning_mask]
                    module[1].conv2.weight.data = netC.layer4[1].conv2.weight.data[pruning_mask]
                    module[1].bn2.weight.data = netC.layer4[1].bn2.weight.data[pruning_mask]
                    module[1].ind = pruning_mask
                elif "fc" == name:
                    module.weight.data = netC.fc.weight.data[:, pruning_mask]
                    module.bias.data = netC.fc.bias.data
                else:
                    continue
            net_pruned.to(device)
            console.print('\n model struct is \n', net_pruned)
            clean, bd = eval(net_pruned, netG, test_dl)
            outs.write("%d %0.4f %0.4f\n" % (index, clean, bd))


if __name__ == "__main__":
    main()
    # eval(model_ft_backdoor,netG,dataloader_clean['val'])