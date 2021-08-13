from __future__ import print_function
from PIL import Image
from matplotlib import patches

import matplotlib.pyplot as plt
from numpy import random
from numpy.core.fromnumeric import shape
from numpy.lib.function_base import select
from torchvision import utils
# 切换后端，保存而不显示
plt.switch_backend('agg')

import argparse
import os
import json

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from rich.console import Console
from rich.progress import track
from material.models.generators import *
from models_structures.VGG import *
import cv2
from utils.utils import stamp_trigger
from utils.transforms_utils import *
from utils.utils import normalize_and_scale

import torch.backends.cudnn as cudnn

import lpips
import torchextractor as tx

import configparser
from utils.utils import visual_constrain

torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    # 示例化一个 parser
    parser = argparse.ArgumentParser()
    # 添加可选参数
    parser.add_argument('--imagenetTrain', type=str, default='./datasets/imagenette/imagenette2/train', help='ImageNet train root')
    parser.add_argument('--imagenetVal', type=str, default='./datasets/imagenette/imagenette2/val', help='ImageNet val root')
    parser.add_argument('--batchSize', type=int, default=30, help='training batch size')
    parser.add_argument('--testBatchSize', type=int, default=16, help='testing batch size')
    parser.add_argument('--nEpochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer: "adam" or "sgd"')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.002')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--MaxIter', type=int, default=50, help='Iterations in each Epoch')
    parser.add_argument('--MaxIterTest', type=int, default=160, help='Iterations in each Epoch')
    parser.add_argument('--mag_in', type=float, default=10.0, help='l_inf magnitude of perturbation')
    parser.add_argument('--expname', type=str, default='tempname', help='experiment name, output folder')
    parser.add_argument('--checkpoint', type=str, default='', help='path to starting checkpoint')
    parser.add_argument('--foolmodel', type=str, default='resnet18-imagenette', help='model to fool: "incv3", "vgg16", or "vgg19"')
    parser.add_argument('--mode', type=str, default='train', help='mode: "train" or "test"')
    parser.add_argument('--perturbation_type', type=str, help='"universal" or "imdep" (image dependent)')
    parser.add_argument('--target', type=int, default=-1, help='target class: -1 if untargeted, 0..999 if targeted')
    parser.add_argument('--gpu_ids', help='gpu ids: e.g. 0 or 0,1 or 1,2.', type=str, default='0')
    parser.add_argument('--path_to_U_noise', type=str, default='', help='path to U_input_noise.txt (only needed for universal)')
    parser.add_argument('--explicit_U', type=str, default='', help='Path to a universal perturbation to use')

    
    config = configparser.ConfigParser()
    config.read('./config/setups.config')

    GENERATOR_SAVED_PATH = config['Generator']['GENERATOR_SAVED_PATH']
    CLEAN_MODEL_FINETUNE_PATH = config['Generator']['CLEAN_MODEL_FINETUNE_PATH']
    CLEAN_MODEL_PATH = config['Generator']['CLEAN_MODEL_PATH_RESNET18_IMAGENETTE']
    Trigger_ID = int(config['Generator']['Trigger_ID'])

    Pretrained_Backdoored_Model_Path = config['GeneratorNew']['Pretrained_Backdoored_Model_Path']

    # 参数导出
    opt = parser.parse_args()
    # 打印参数
    console = Console()
    console.print(opt)
    console.print('Trigger Generator will be saved to {}'.format(GENERATOR_SAVED_PATH))

    # 定义训练过程中的数据记录
    # train loss history
    train_loss_history = []
    test_loss_history = []
    test_acc_history = []
    test_fooling_history = []
    best_fooling = 0
    itr_accum = 0

    cudnn.benchmark = True
    torch.cuda.manual_seed(opt.seed)

    MaxIter = opt.MaxIter
    MaxIterTest = opt.MaxIterTest

    # gpulist = [int(i) for i in opt.gpu_ids.split(',')]
    gpulist = [0,1]
    n_gpu = len(gpulist)
    console.print('Running with n_gpu: ', n_gpu)

    # Softmax 函数
    softmax_func = torch.nn.Softmax(dim=0)

    # KL散度 
    kld_sum_func = torch.nn.KLDivLoss(reduction='sum')

    # l1 损失
    criterion_pixelwise = torch.nn.L1Loss()

    # lpips 损失
    use_gpu = True         # Whether to use GPU
    loss_fn = lpips.LPIPS(net='vgg')
    if use_gpu:
        loss_fn.cuda(gpulist[0])

    # pytorch cdist : 计算tensor对之间的 p norm距离
    # @TODO

    # PSNR 损失
    # @TODO

    # SMIM 损失
    # @TODO

    if opt.foolmodel == 'vgg16-cifar10':
        # 从torchvision.datasets中加载一些常用数据集
        train_dataset = torchvision.datasets.CIFAR10(
        root='./datasets/cifar10/',  # 数据集保存路径
        train=True,  # 是否作为训练集
        transform=data_transform,  # 数据如何处理, 可以自己自定义
        download=False)  # 路径下没有的话, 可以下载

        test_dataset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/',
                            train=False,
                            transform=data_transform)

        training_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                        batch_size=opt.batchSize,
                                        shuffle=True)

        testing_data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=opt.batchSize,
                                        shuffle=False)
    elif opt.foolmodel == 'resnet18-imagenette':
        train_set = torchvision.datasets.ImageFolder(root = opt.imagenetTrain, transform = data_transform)
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

        test_set = torchvision.datasets.ImageFolder(root = opt.imagenetVal, transform = data_transform)
        testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=True)

        train_set_on_7_path = './datasets/imagenette/imagenette2/train_on_7'
        val_set_on_7_path = './datasets/imagenette/imagenette2/val_on_7'

        train_set_on_7 = torchvision.datasets.ImageFolder(root = train_set_on_7_path, transform = data_transform)
        training_data_loader_on_7 = DataLoader(dataset=train_set_on_7, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

        test_set_on_7 = torchvision.datasets.ImageFolder(root = val_set_on_7_path, transform = data_transform)
        testing_data_loader_on_7 = DataLoader(dataset=test_set_on_7, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=True)

        train_set_without_7_path = './datasets/imagenette/imagenette2/train_without_7'
        val_set_without_7_path = './datasets/imagenette/imagenette2/val_without_7'

        train_set_without_7 = torchvision.datasets.ImageFolder(root = train_set_without_7_path, transform = data_transform)
        training_data_loader_without_7 = DataLoader(dataset=train_set_without_7, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

        test_set_without_7 = torchvision.datasets.ImageFolder(root = val_set_without_7_path, transform = data_transform)
        testing_data_loader_without_7 = DataLoader(dataset=test_set_on_7, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=True)


    # 载入测试数据
    if opt.foolmodel == 'vgg16-cifar10':
        test_dataset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/',
                                     train=False,
                                     transform=data_transform)
    elif opt.foolmodel == 'resnet18-imagenette':                                 
        test_set = torchvision.datasets.ImageFolder(root = opt.imagenetVal, transform = data_transform)
        testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=True)

    # 选择要进行攻击的后门模型来作为欺骗模型
    if opt.foolmodel == 'incv3':
        model_ft_clean_finetune = torchvision.models.inception_v3(pretrained=True)
    elif opt.foolmodel == 'vgg16':
        model_ft_clean_finetune = torchvision.models.vgg16(pretrained=True)
    elif opt.foolmodel == 'vgg19':
        model_ft_clean_finetune = torchvision.models.vgg19(pretrained=True)
    elif opt.foolmodel == 'resnet18-imagenette':
        # TODO 后续将配置写进行配置文件中
        
        clean_model_path = CLEAN_MODEL_FINETUNE_PATH
        BACKDOOR_MODEL_PATH = Pretrained_Backdoored_Model_Path
        # clean_model_path = "../backdoor-nn/t-backdoor-nn/gap-backdoor/model/gap_backdoor_attack_resnet18_imagenette_mal_ori_p11.pth"
        model_ft_clean_finetune = torchvision.models.resnet18()

        # Finetune Final few layers to adjust for tiny imagenet input
        # 根据任务，对模型进行微调，这里将模型的最后一层更改为 10
        model_ft_clean_finetune.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = model_ft_clean_finetune.fc.in_features
        model_ft_clean_finetune.fc = nn.Linear(num_ftrs, 10)
        model_ft_clean_finetune.load_state_dict(torch.load(clean_model_path, map_location='cuda'))
        console.print('[bold green]clean model fine tune[/bold green] is loaded: {}'.format(CLEAN_MODEL_FINETUNE_PATH))

        ## 载入后门网络
        # Training a Resnet18  Backdoored Model
        model_ft_backdoor = torchvision.models.resnet18()
        # Finetune Final few layers to adjust for tiny imagenet input
        model_ft_backdoor.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = model_ft_backdoor.fc.in_features
        model_ft_backdoor.fc = nn.Linear(num_ftrs, 10)
        model_ft_backdoor.load_state_dict(torch.load(BACKDOOR_MODEL_PATH, map_location='cuda'))

        console.print('[bold green]backdoor model[/bold green] is loaded: {}'.format(BACKDOOR_MODEL_PATH))

    elif opt.foolmodel == 'vgg16-cifar10':
        """
        VGG16 在 CIFAR10 上作为欺骗模型
        """
        clean_model_path = "./models/vgg16_cifar10_clean_520_2326.pth"
        model_ft_clean_finetune = VGG('VGG16')
        model_ft_clean_finetune.load_state_dict(torch.load(clean_model_path, map_location='cuda'))
    
    model_ft_clean_finetune = model_ft_clean_finetune.cuda(gpulist[0])
    model_ft_backdoor = model_ft_backdoor.cuda(gpulist[0])

    model_ft_clean_finetune.eval()
    model_ft_backdoor.eval()
    
    # 纯粹的inference模式下推荐使用volatile，当你确定你甚至不会调用.backward()时。它比任何其他自动求导的设置更有效——它将使用绝对最小的内存来评估模型。volatile也决定了require_grad is False。
    model_ft_clean_finetune.volatile = True
    model_ft_backdoor.volatile = True

    # magnitude
    mag_in = opt.mag_in

    # 开始训练过程
    console.print("===> Training Model")
    
    # 读取netG模型结构 
    # will use model paralellism if more than one gpu specified

    # netG = ResnetGenerator(3, 3, opt.ngf, norm_type='batch', act_type='relu', gpu_ids=gpulist)
    netG = RecursiveUnetGenerator(3, 3, num_downs = 4, ngf = opt.ngf, norm_type='batch',
                 act_type='relu', use_dropout=True, gpu_ids=gpulist)

    # setup optimizer
    if opt.optimizer == 'adam':
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    elif opt.optimizer == 'sgd':
        optimizerG = optim.SGD(netG.parameters(), lr=opt.lr, momentum=0.9)

    criterion_pre = nn.CrossEntropyLoss()
    criterion_pre = criterion_pre.cuda(gpulist[0])

    trigger_id = Trigger_ID
    trigger = Image.open('./data/triggers/trigger_{}.png'.format(trigger_id)).convert('RGB')
    trigger = trigger_transform(trigger)


    outer_train_mode = 'parallel'
    console.print('outer_train_mode is : ', outer_train_mode)

    def train(epoch):
        netG.train()
        global itr_accum
        global optimizerG

        target_img_count = 0 
        target_fooling_count = 0
        target_fooling_rate = 0

        untarget_img_count = 0 
        untarget_fooling_count = 0
        untarget_fooling_rate = 0

        if outer_train_mode == 'taowa':
            for itr, (image, ground_truth_labels) in enumerate(track(training_data_loader_on_7, 1)):
                # console.print('itr:', itr)
                if itr > MaxIter:
                    break

                itr_accum += 1

                # 读取训练数据集中的一批次图像
                image = image.cuda(gpulist[0])
                # print(image.shape)
                clean_images = image.clone()

                netG_out = netG(image)

                if itr % 5 == 1:
                    torchvision.utils.save_image(netG_out, 'tempt_data/out_NetG/netG_out{}_{}.png'.format(epoch,itr))


                netG_out = normalize_and_scale(netG_out, 'train')
                # netG_out_return = netG_out[1]


                # if itr % 5 == 1:
                torchvision.utils.save_image(netG_out, 'tempt_data/out_NetG/netG_out_normalize{}_{}.png'.format(epoch,itr))

                # 将输出转化入 cuda
                netG_out = netG_out.cuda(gpulist[0])
                                

                # 把输出的扰动与原图像相加
                recons = torch.add(netG_out, clean_images.cuda(gpulist[0]))


                # do clamping per channel
                for cii in range(3):
                    recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(clean_images[:,cii,:,:].min(), clean_images[:,cii,:,:].max())

                # unnormalize recons img，保存 
                # if itr % 5 == 1:
                    # torchvision.utils.save_image(recons, 'tempt_data/out_NetG/recons{}_{}.png'.format(epoch,itr))
                recons_unNormalize = torch.zeros_like(recons)
                for cxx in range(recons_unNormalize.size(0)):
                    recons_unNormalize[cxx,:,:,:] = transform_unNormalize(recons[cxx,:,:,:])
                torchvision.utils.save_image(recons_unNormalize, 'tempt_data/out_NetG/recons_unNormalize{}_{}.png'.format(epoch,itr))

                    # # 测试：原图保存
                    # for cxx in range(opt.batchSize):
                    #     torchvision.utils.save_image(transform_unNormalize(clean_images[cxx,:,:,:]), '/home/nas928/ln/GETBAK/results/dataset/clean/clean_images{}_{}_{}.png'.format(epoch,itr,cxx))
                    
                    # # save_each img
                    # for cxx in range(opt.batchSize):
                    #     torchvision.utils.save_image(recons_unNormalize[cxx,:,:,:], '/home/nas928/ln/GETBAK/results/dataset/poisoned/recons_unNormalize{}_{}_{}.png'.format(epoch,itr,cxx))
                    
                clean_images = clean_images.cuda(gpulist[0])


                # 任务 1 触发器可学习的损失函数  
                # 选取干净图像中最低可信度的结果：

                task1_mode = 'untarget'

                if task1_mode == 'untarget':
                    pretrained_label_float = model_ft_clean_finetune(image.cuda(gpulist[0]))
                    _, target_label = torch.min(pretrained_label_float, 1)
                    # console.print('[correct]target_label:{}\n{}'.format(pretrained_label_float,target_label))
                elif task1_mode == 'target':
                    target_label = torch.LongTensor(image.size(0))
                    target_label.fill_(7)
                    target_label = target_label.cuda(gpulist[0])

                # 选取重建图像中最高可信度的结果
                output_pretrained = model_ft_clean_finetune(recons.cuda(gpulist[0]))
                # console.print('output_pretrained:',output_pretrained)

                # 使用交叉熵作为损失函数
                cre_loss = torch.log(criterion_pre(output_pretrained, target_label))

                task1_loss = cre_loss*1

                # 任务 2 视觉隐蔽的损失函数
                # 干净图像和带有隐蔽触发器的的损失

                # 将数据转化为 0-1 再recons进行lpips
                recons_lpips_input = transform_unNormalize(recons.clone())
                recons_lpips_input = torch.clamp(recons_lpips_input, 0, 1)
                ct_img_repeat_lpips_input = transform_unNormalize(clean_images.clone())

                
                lpips_loss2 = loss_fn.forward(recons_lpips_input, ct_img_repeat_lpips_input, normalize=True)
                lpips_loss2 = sum(lpips_loss2.clone())

                task2_loss = lpips_loss2*10

                # 使得 task2 任务趋向于1，使用以下损失（by 鹏鹏）
                # task2_loss = (1-lpips_loss2*1).abs()
        

                # 任务 3 可见触发器可出发的损失函数（特征趋近）
                # Options: 特征上使用KLD(KL散度)、LPIPS、求平方和

                task3_option = 'KLD_Loss'
                Loss_Mode = 'kld_sum_to_one_dim'
                # Loss_Mode = 'feature_dif_squ_sum'
                task3_loss = 0

                # 任务 4 后门模型输出loss
                # total imgs number = threadhold*batchsieze 

                threadhold = 5
                task4_iner_loss = 0
                for itr2, (image2, ground_truth_labels2) in enumerate(training_data_loader_without_7, 1):
                    itr2 += 1
                    if itr2 > threadhold:
                        break

                    image2 = image2.cuda(gpulist[0])

                    netG_out2 = netG(image2)

                    # if itr % 5 == 1:
                    torchvision.utils.save_image(netG_out, 'tempt_data/out_NetG/netG_out2_{}_{}.png'.format(epoch,itr))


                    netG_out2 = normalize_and_scale(netG_out2, 'train')
                    # netG_out2_return = netG_out2[1]


                    # if itr % 5 == 1:
                    torchvision.utils.save_image(netG_out2, 'tempt_data/out_NetG/netG_out2_normalize{}_{}.png'.format(epoch,itr))

                    # 将输出转化入 cuda
                    netG_out2 = netG_out2.cuda(gpulist[0])
                                    

                    # 把输出的扰动与原图像相加
                    recons2 = torch.add(netG_out2, clean_images.cuda(gpulist[0]))


                    recons2 = torch.add(netG_out2, image2)
                    # do clamping per channel
                    for cii in range(3):
                        recons2[:,cii,:,:] = recons2[:,cii,:,:].clone().clamp(image2[:,cii,:,:].min(), image2[:,cii,:,:].max())
                    
                    # if itr2 % 5 == 1:
                        # torchvision.utils.save_image(recons2, 'tempt_data/out_NetG/recons2_{}_{}.png'.format(epoch,itr2))

                    recons2_unNormalize = torch.zeros_like(recons2)

                    for cxx in range(recons2_unNormalize.size(0)):
                        recons2_unNormalize[cxx,:,:,:] = transform_unNormalize(recons2[cxx,:,:,:])

                    torchvision.utils.save_image(recons2_unNormalize, 'tempt_data/out_NetG/recons2_unNormalize{}_{}.png'.format(epoch,itr2))

                    target_label_2 = torch.LongTensor(image2.size(0))
                    target_label_2.fill_(7)
                    target_label_2 = target_label_2.cuda(gpulist[0])

                    # console.print('[2]target_label:', target_label_2)

                    # console.print('ground_truth_labels2:',ground_truth_labels2)

                    output_pretrained_clean_model = model_ft_clean_finetune(recons2.cuda(gpulist[0]))
                    __, target_label_clean = torch.max(output_pretrained_clean_model, 1)

                    console.print('\n target_label_clean:',target_label_clean)

                    output_pretrained_2 = model_ft_backdoor(recons2.cuda(gpulist[0]))
                    __, target_label_backdoor = torch.max(output_pretrained_2, 1)


                    console.print('\n target_label_t:',target_label_backdoor)

                    # console.print('output_pretrained_2:{}\n{}'.format(output_pretrained_2.shape, output_pretrained_2))

                    # 使用交叉熵作为损失函数
                    # console.print('criterion_pre:',criterion_pre(output_pretrained_2, target_label))
                    
                    cre_loss_2 = torch.log(criterion_pre(output_pretrained_2, target_label_2))

                    task4_iner_loss += cre_loss_2
                    console.print('task4_iner_loss = {}'.format(task4_iner_loss))


                # task4_iner_loss = task4_iner_loss/(threadhold+1)
                task4_loss = task4_iner_loss
                    


                # -------------------
                # Total Loss
                
                loss =  task1_loss + task4_loss
                # loss = task1_loss + task2_loss + task3_loss

                optimizerG.zero_grad()
                loss.backward()
                optimizerG.step()

                train_loss_history.append(loss.item())
                print("===> Epoch[{}]({}/{}) loss: {:.4f}, << task1_loss:{:.4f}, task2_loss:{:.4f}, task3_loss:{:.4f}, task4_loss:{:.4f}".format(epoch, itr, len(training_data_loader), loss.item(), task1_loss.item(), task2_loss.item(), task3_loss, task4_loss.item()))

                netG_out_return = None
            return (loss,task1_loss,task2_loss,task3_loss), netG_out_return
        elif outer_train_mode == 'parallel':
            for itr, (image, ground_truth_labels) in enumerate(track(training_data_loader_on_7, 1)):
                # console.print('ground_truth_labels:', ground_truth_labels)
                
                # !! Attention!! 如果training_data_loader_on_7，selected_target_label 为 0
                # !! Attention!! 如果training_data_loader，selected_target_label 为 7

                selected_target_label = 0

                if itr > MaxIter:
                    break

                itr_accum += 1

                # 读取训练数据集中的一批次图像
                image = image.cuda(gpulist[0])
                # print(image.shape)
                clean_images = image.clone()
                clean_images_unNormalize = transform_unNormalize(clean_images.clone())

                
                netG_out = netG(image)
                netG_out = netG_out.cuda(gpulist[0])

                netG_out = normalize_and_scale(netG_out, 'train')


                # console.print(netG_out.min(), netG_out.max())
                # (-2.4 ~ 2.6)

                # 保存netG_out图像
                netG_out_unNormalize = torch.zeros_like(netG_out)
                for cxx in range(netG_out_unNormalize.size(0)):
                    netG_out_unNormalize[cxx,:,:,:] = transform_unNormalize(netG_out[cxx,:,:,:])
                torchvision.utils.save_image(netG_out_unNormalize, 'tempt_data/out_NetG/pall_netG_out_unNormalize_{}_{}.png'.format(epoch,itr))

                # console.print(netG_out_unNormalize.min(), netG_out_unNormalize.max())
                # (0 ~ 1)

                netG_out_Type = 'recons'
                if netG_out_Type == 'trigger':
                    # 把输出的扰动与原图像相加
                    recons = torch.add(netG_out, clean_images.cuda(gpulist[0]))

                elif  netG_out_Type == 'recons':
                    # 把输出的扰动与原图像相加
                    recons = netG_out

                # do clamping per channel
                for cii in range(3):
                    recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(clean_images[:,cii,:,:].min(), clean_images[:,cii,:,:].max())


                # # 保存相减获得的图像
                # delta_unNormalize = transform_unNormalize(clean_images.clone()).clone() - recons_unNormalize.clone()
                # delta_unNormalize = delta_unNormalize.abs()
                # # console.print(netG_out_unNormalize)
                # # delta_unNormalize = torch.clamp(delta_unNormalize, 0, 1)


                # 保存相减获得的图像(version2)
                # delta_unNormalize = transform_unNormalize(clean_images.clone()).clone() - recons_unNormalize.clone()

                recons_unNormalize = torch.zeros_like(recons)
                for cxx in range(recons_unNormalize.size(0)):
                    recons_unNormalize[cxx,:,:,:] = transform_unNormalize(recons[cxx,:,:,:])

                delta_unNormalize = torch.sub(recons_unNormalize.clone(),clean_images_unNormalize.clone()) 
                # console.print(delta_unNormalize.min(), delta_unNormalize.max())
                # (-1 ~ 1)
                delta_unNormalize = delta_unNormalize.clamp(0,20/255)
                # console.print(netG_out_unNormalize.min(), netG_out_unNormalize.max())
                # delta_unNormalize = torch.clamp(delta_unNormalize, 0, 1)

                torchvision.utils.save_image(delta_unNormalize, 'tempt_data/out_NetG/pall_delta_unNormalize{}_{}.png'.format(epoch,itr))

                
                # ！！！attention!! 加回delta到clean_img 得到 reoncs

                recons = torch.add(delta_unNormalize.clone(), clean_images_unNormalize.clone())
                recons = recons.clamp(0,1)
                recons = transform_Normalize(recons)

                # 保存recons图像
                recons_unNormalize = torch.zeros_like(recons)
                for cxx in range(recons_unNormalize.size(0)):
                    recons_unNormalize[cxx,:,:,:] = transform_unNormalize(recons[cxx,:,:,:])
                torchvision.utils.save_image(recons_unNormalize, 'tempt_data/out_NetG/pall_recons_unNormalize{}_{}.png'.format(epoch,itr))


                untarget_img_loss = 0
                target_img_loss = 0   
                # i : 一个batch里面图像的序号
                for i_index in range(len(image)):
                    # console.print('ground_truth_labels[i].item()', ground_truth_labels[i_index].item())
                    recons_i = recons[i_index]
                    recons_i = recons_i.unsqueeze(dim=0)
                    clean_i = clean_images[i_index]
                    clean_i = clean_i.unsqueeze(dim=0)
                    
                    # 靶向类图像趋向非靶向
                    if ground_truth_labels[i_index].item() == selected_target_label:
                        target_img_count+=1

                        pretrained_label_float = model_ft_clean_finetune(clean_i.cuda(gpulist[0]))
                        _, target_label = torch.min(pretrained_label_float, 1)
                        
                        # 选取重建图像中最高可信度的结果
                        output_pretrained = model_ft_clean_finetune(recons_i.cuda(gpulist[0]))
                        # console.print('output_pretrained:',output_pretrained)

                        # 使用交叉熵作为损失函数
                        target_img_loss += criterion_pre(output_pretrained, target_label)

                        if torch.max(output_pretrained,1)[1].item() != 7:
                            target_fooling_count += 1
                         
                        torchvision.utils.save_image(transform_unNormalize(recons_i), 'tempt_data/out_NetG/pallel_reconsi_target_epo{}_itr{}_i{}.png'.format(epoch,itr,i_index))
                        
                    # 非靶向类图像趋向靶向
                    elif ground_truth_labels[i_index].item() != selected_target_label:
                        untarget_img_count +=1
   
                        target_label_2 = torch.LongTensor(clean_i.size(0))
                        target_label_2.fill_(selected_target_label)
                        target_label_2 = target_label_2.cuda(gpulist[0])
                        
                        # 选取重建图像中最高可信度的结果
                        output_pretrained = model_ft_clean_finetune(recons_i.cuda(gpulist[0]))
                        # console.print('output_pretrained:',torch.max(output_pretrained,1)[1])

                        # 使用交叉熵作为损失函数
                        untarget_img_loss += criterion_pre(output_pretrained, target_label_2)

                        if torch.max(output_pretrained,1)[1].item() == 7:
                            untarget_fooling_count += 1

                        torchvision.utils.save_image(transform_unNormalize(recons_i), 'tempt_data/out_NetG/pallel_reconsi_untarget_epo{}_itr{}_i{}.png'.format(epoch,itr,i_index))
                
                # 将数据转化为 0-1 再recons进行lpips
                recons_lpips_input = transform_unNormalize(recons.clone())
                recons_lpips_input = torch.clamp(recons_lpips_input, 0, 1)
                clean_images_lpips_input = transform_unNormalize(clean_images.clone())

                torchvision.utils.save_image(recons_lpips_input, 'tempt_data/lpips_loss/recons_lpips_input_epo{}_itr{}.png'.format(epoch,itr))

                torchvision.utils.save_image(clean_images_lpips_input, 'tempt_data/lpips_loss/clean_images_lpips_input_epo{}_itr{}.png'.format(epoch,itr))

                # console.print('recons_lpips_input_range:{} : {}'.format(recons_lpips_input.min().item(), recons_lpips_input.max().item()))
                # console.print('ct_img_repeat_lpips_input_range:{} : {}'.format(ct_img_repeat_lpips_input.min().item(), ct_img_repeat_lpips_input.max().item()))
                
                # torchvision.utils.save_image(recons_lpips_input, 'tempt_output/hiddenG_recons_before.png')
                # torchvision.utils.save_image(ct_img_repeat_lpips_input, 'tempt_output/hiddenG_clean_img.png')
                
                
                lpips_loss = loss_fn.forward(recons_lpips_input, clean_images_lpips_input, normalize=True)
                lpips_loss = sum(lpips_loss.clone())


                lpips_loss = 10*lpips_loss
                target_img_loss = target_img_loss
                untarget_img_loss = untarget_img_loss
                # untarget_img_loss = torch.log(untarget_img_loss)
                # target_img_loss = torch.log(target_img_loss)

                # if lpips_loss > 15:
                #     loss = lpips_loss
                # else:
                #     loss = lpips_loss + 10 * target_img_loss + untarget_img_loss
                
                loss = target_img_loss + lpips_loss
        
                optimizerG.zero_grad()
                loss.backward()
                optimizerG.step()

                # print('loss_type:',type(loss))
                # print('untarget_img_loss_type:',type(untarget_img_loss))
                # print('target_img_loss_type:',type(target_img_loss))

                target_loss_value = target_img_loss.item() if target_img_loss!=0 else target_img_loss
                untarget_loss_value = untarget_img_loss.item() if untarget_img_loss!=0 else untarget_img_loss
 
                train_loss_history.append(loss.item())
                console.print("\n===> Epoch[{}]({}/{}) loss: {:.3f}, << untarget_img_loss:{:.3f}, target_img_loss:{:.3f}, lpips_loss:{:.3f}".format(epoch, itr, len(training_data_loader), loss.item(), untarget_loss_value, target_loss_value, lpips_loss.item()))

                if target_img_count != 0:
                    target_fooling_rate = target_fooling_count / target_img_count
                else:
                    target_fooling_rate = 0

                if untarget_img_count !=0:
                    untarget_fooling_rate = untarget_fooling_count / untarget_img_count
                else:
                    untarget_fooling_rate = 0
                
                console.print('===> Current Target Image Rate, Total:{}, Fooling:{}, Rate:{:.3f}'.format(target_img_count, target_fooling_count, target_fooling_rate))

                console.print('===> Current Untarget Image Rate, Total:{}, Fooling:{}, Rate:{:.3f}'.format(untarget_img_count, untarget_fooling_count, untarget_fooling_rate))                

                netG_out_return = netG_out[0]

            return (None,None,None,None),netG_out_return

    def test(netG_out_return):
        # console.print(netG_out_return.shape)

        real_labels_distribution = [0 for i in range(10)]
        predicted_labels_distribution = [0 for i in range(10)]
        correct = 0
        total = 0
        accuracy = 0.
        attack_success_image = 0
        fooling_image = 0
        attack_success_rate= 0.
        foolingRate = 0.

        console.print('Start to test [yellow]feat_trigger[/yellow]\'s output')
        with torch.no_grad():
            torchvision.utils.save_image(transform_unNormalize(netG_out_return),'/home/nas928/ln/GETBAK/tempt_data/test_trigger.png')
            outputs = model_ft_clean_finetune(netG_out_return.reshape(1,3,224,224))
            _, predicted = torch.max(outputs.data, 1)

            console.print('feat_trigger: predictions=[cyan]{}[/cyan]'.format(predicted))
            # console.print('feat_trigger: outputs=[cyan]{}[/cyan]'.format(outputs))
        

        console.print('Start to test [yellow]recons_images[/yellow]\'s output distribution')
        with torch.no_grad():

            selected_target_label = 0
            for images, labels in track(testing_data_loader_on_7,1):
                # 保存干净图片出来
                clean_images = images.clone()
                clean_images = clean_images.cuda(gpulist[0])
                # Account real labels distribution.
                for i in range(len(labels)):
                    real_labels_distribution[labels[i]]+=1

                # trigger_type = 'fixed_netG_trigger'
                trigger_type = 'img_dep_netG_trigger'

                if trigger_type == 'fixed_netG_trigger':
                    
                    # 依据 clean_images batchsize 倍率放大 netG_out_return
                    netG_out_return_repeat = netG_out_return.repeat(clean_images.size(0),1,1,1)
                
                    # 将输出转化入 cuda
                    netG_out_return_repeat = netG_out_return_repeat.cuda(gpulist[0])
                    
                    # 把输出的扰动与原图像相加
                    recons = torch.add(netG_out_return_repeat, clean_images)
                    # print(recons.shape)

                elif trigger_type == 'img_dep_netG_trigger':
                    netG_out = netG(images)
                    netG_out = netG_out.cuda(gpulist[0])

                    netG_out = normalize_and_scale(netG_out, 'train')

                    # recons = torch.add(netG_out, clean_images)
                    recons = netG_out


                # do clamping per channel
                for cii in range(3):
                    recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(clean_images[:,cii,:,:].min(), clean_images[:,cii,:,:].max())

                #准备测试
                trigger_images = recons.cuda(gpulist[0])
                labels = labels.cuda(gpulist[0])

                torchvision.utils.save_image(transform_unNormalize(trigger_images),'/home/nas928/ln/GETBAK/tempt_data/test_trigger_images.png')

                outputs = model_ft_clean_finetune(trigger_images)
                _, predicted = torch.max(outputs.data, 1)
                # 计算预测结果分布
                for i in range(len(predicted.cpu())):
                    predicted_labels_distribution[predicted.cpu()[i]]+=1
                
                # 统计预测数据
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                attack_success_image += (predicted == 7).sum().item()
                fooling_image += (predicted != labels).sum().item()
    
        # 计算、打印攻击成功率等
        accuracy = correct / total
        attack_success_rate = attack_success_image / total
        foolingRate = fooling_image / total
        # console.print('Model Accuracy:', accuracy)
        console.print('Attack Success Rate:', attack_success_rate)
        console.print('Fooling Rate:', foolingRate)    
        console.print('Real Labels Distribution:', real_labels_distribution)
        console.print('Predicted Labels Distribution:', predicted_labels_distribution)




        # test_acc_history.append((100.0 * correct_recon / total))
        # test_fooling_history.append((100.0 * fooled / total))
        # print('Accuracy of the pretrained network on reconstructed images: %.2f%%' % (100.0 * float(correct_recon) / float(total)))
        # print('Accuracy of the pretrained network on original images: %.2f%%' % (100.0 * float(correct_orig) / float(total)))
        # if opt.target == -1:
        #     print('Fooling ratio: %.2f%%' % (100.0 * float(fooled) / float(total)))
        # else:
        #     print('Top-1 Target Accuracy: %.2f%%' % (100.0 * float(fooled) / float(total)))

    # def normalize_and_scale(delta_im, mode='train'):
    #     if opt.foolmodel == 'incv3':
    #         delta_im = nn.ConstantPad2d((0,-1,-1,0),0)(delta_im) # crop slightly to match inception

    #     delta_im = delta_im + 1 # now 0..2
    #     delta_im = delta_im * 0.5 # now 0..1

    #     # normalize image color channels
    #     for c in range(3):
    #         delta_im[:,c,:,:] = (delta_im[:,c,:,:].clone() - mean_arr[c]) / stddev_arr[c]

    #     # threshold each channel of each image in deltaIm according to inf norm
    #     # do on a per image basis as the inf norm of each image could be different
    #     # bs = delta_im.size(0)
    #     # for i in range(bs):
    #     #     # do per channel l_inf normalization
    #     #     for ci in range(3):
    #     #         l_inf_channel = delta_im[i,ci,:,:].detach().abs().max()
    #     #         mag_in_scaled_c = mag_in/(255.0*stddev_arr[ci])
    #     #         gpu_id = gpulist[1] if n_gpu > 1 else gpulist[0]
    #     #         delta_im[i,ci,:,:] = delta_im[i,ci,:,:].clone() * np.minimum(1.0, mag_in_scaled_c / l_inf_channel.cpu().numpy())

    #     return delta_im

    def print_history():
        # plot history for training loss
        if opt.mode == 'train':
            plt.plot(train_loss_history)
            plt.title('Model Training Loss')
            plt.ylabel('Loss')
            plt.xlabel('Iteration')
            plt.legend(['Training Loss'], loc='upper right')
            plt.savefig(opt.expname+'/reconstructed_loss_'+opt.mode+'.png')
            plt.clf()

        # plot history for classification testing accuracy and fooling ratio
        plt.plot(test_acc_history)
        plt.title('Model Testing Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Testing Classification Accuracy'], loc='upper right')
        plt.savefig(opt.expname+'/reconstructed_acc_'+opt.mode+'.png')
        plt.clf()

        plt.plot(test_fooling_history)
        plt.title('Model Testing Fooling Ratio')
        plt.ylabel('Fooling Ratio')
        plt.xlabel('Epoch')
        plt.legend(['Testing Fooling Ratio'], loc='upper right')
        plt.savefig(opt.expname+'/reconstructed_foolrat_'+opt.mode+'.png')
        print("Saved plots.")

    if opt.mode == 'train':
        for epoch in range(1, opt.nEpochs + 1):

            (loss,task1_loss,task2_loss,task3_loss), netG_out_return = train(epoch)

            test(netG_out_return)

            # 找到一个优解
            # if task1_loss < 0.5 and task3_loss < 0.01:
            #     console.print('find a good solution, break.')
            #     break
            
            # checkpoint_dict(epoch)
            # save model
            torch.save(netG.state_dict(), GENERATOR_SAVED_PATH)
            console.print('Trigger Generator is saved to {}'.format(GENERATOR_SAVED_PATH))
        # print_history()
    else:
        pass