from __future__ import print_function
from PIL import Image
from matplotlib import patches

import matplotlib.pyplot as plt
from numpy import random
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

import torch.backends.cudnn as cudnn

import lpips
import torchextractor as tx

import configparser
from utils.utils import visual_constrain

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
        # clean_model_path = "../backdoor-nn/t-backdoor-nn/gap-backdoor/model/gap_backdoor_attack_resnet18_imagenette_mal_ori_p11.pth"
        model_ft_clean_finetune = torchvision.models.resnet18()

        # Finetune Final few layers to adjust for tiny imagenet input
        # 根据任务，对模型进行微调，这里将模型的最后一层更改为 10
        model_ft_clean_finetune.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = model_ft_clean_finetune.fc.in_features
        model_ft_clean_finetune.fc = nn.Linear(num_ftrs, 10)
        model_ft_clean_finetune.load_state_dict(torch.load(clean_model_path, map_location='cuda'))
        console.print('[bold green]clean model fine tune[/bold green] is loaded: {}'.format(CLEAN_MODEL_FINETUNE_PATH))

    elif opt.foolmodel == 'vgg16-cifar10':
        """
        VGG16 在 CIFAR10 上作为欺骗模型
        """
        clean_model_path = "./models/vgg16_cifar10_clean_520_2326.pth"
        model_ft_clean_finetune = VGG('VGG16')
        model_ft_clean_finetune.load_state_dict(torch.load(clean_model_path, map_location='cuda'))


    model_ft_clean_finetune = model_ft_clean_finetune.cuda(gpulist[0])

    model_ft_clean_finetune.eval()
    
    # 纯粹的inference模式下推荐使用volatile，当你确定你甚至不会调用.backward()时。它比任何其他自动求导的设置更有效——它将使用绝对最小的内存来评估模型。volatile也决定了require_grad is False。
    model_ft_clean_finetune.volatile = True

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
    # print(trigger.shape)


    def train(epoch):
        netG.train()
        global itr_accum
        global optimizerG

        for itr, (image, _) in enumerate(track(training_data_loader, 1)):
            # console.print('itr:', itr)
            if itr > MaxIter:
                break

            itr_accum += 1
            # deafult is 'adam'
            if opt.optimizer == 'sgd':
                lr_mult = (itr_accum // 1000) + 1
                optimizerG = optim.SGD(netG.parameters(), lr=opt.lr/lr_mult, momentum=0.9)

            # 读取训练数据集中的一批次图像
            image = image.cuda(gpulist[0])
            # print(image.shape)
            clean_images = image.clone()

            # feat_trigger_gen_mode = 'image_specific'
            feat_trigger_gen_mode = 'fixed_visual_trigger'

            if feat_trigger_gen_mode == 'image_specific':

                # 生成一个空的，只带触发器的图像 trigger_img
                zero_img = np.zeros((224,224,3))
                zero_img = transform_toTensor(zero_img)
                torchvision.utils.save_image(zero_img, 'tempt_data/zero_img.png')

                # 生成只带 trigger 的空图像
                trigger_unnormalize = transform_unNormalize(trigger)
                zero_img_with_trigger = stamp_trigger(zero_img, trigger_unnormalize,trigger_size, random_location=False, is_batch=False)
                # print('zero_img_with_trigger shape:',zero_img_with_trigger.shape)
                torchvision.utils.save_image(zero_img_with_trigger, 'tempt_data/zero_img_with_trigger.png')

                # zero_img_with_trigger = np.reshape(zero_img_with_trigger,(224,224,3))    
                zero_img_with_trigger = transform_Normalize(zero_img_with_trigger)
                torchvision.utils.save_image(zero_img_with_trigger, 'tempt_data/zero_img_with_trigger_normalized.png')

                # print('zero_img_with_trigger shape:',zero_img_with_trigger.shape)
                # torchvision.utils.save_image(zero_img_with_trigger, 'tempt_data/zero_img_with_trigger.png')

                # 生成带具体图像、触发器的图像
                trigger_img = stamp_trigger(image, trigger, trigger_size, random_location=False, is_batch=True)
                torchvision.utils.save_image(trigger_img, 'tempt_data/trigger_img.png')

                # 保存 unomalized 后带具体图像、触发器的图像
                torchvision.utils.save_image(transform_unNormalize(trigger_img), 'tempt_data/trigger_img_unormalized.png')
            
                # zero_img 重复 repeat 为 zero_img_with_trigger_batch，以及保存查看
                zero_img_with_trigger_batch = zero_img_with_trigger.squeeze(0).repeat(opt.batchSize, 1, 1, 1)
                torchvision.utils.save_image(zero_img_with_trigger_batch, 'tempt_data/zero_img_with_trigger_batch.png')
                
                zero_img_with_trigger_batch = zero_img_with_trigger_batch.type(torch.FloatTensor).cuda(gpulist[0])       

                # 将贴上 trigger 的源图像输入生成器 netG，得到输出 netG_out，其为输出的扰动触发器
                # trigger_img_repeat = trigger_img.squeeze(0).repeat(opt.batchSize, 1, 1, 1)
                # trigger_img_repeat = trigger_img_repeat.type(torch.FloatTensor).cuda(gpulist[0])
                trigger_img_repeat = trigger_img.cuda(gpulist[0])
                # print('trigger_img_repeat.shape:',trigger_img_repeat.shape)
                torchvision.utils.save_image(trigger_img_repeat, 'tempt_data/trigger_img_repeat.png')
                
                netG_out = netG(trigger_img_repeat)

            elif feat_trigger_gen_mode == 'fixed_visual_trigger':
                
                # 用于生成器输入使用
                trigger_img_repeat = trigger.repeat(opt.batchSize, 1, 1, 1)
                torchvision.utils.save_image(trigger_img_repeat, 'tempt_data/trigger_img_batch_{}.png'.format(feat_trigger_gen_mode))

                # Normalize
                trigger_img_repeat = transform_Normalize(trigger_img_repeat)
                torchvision.utils.save_image(trigger_img_repeat, 'tempt_data/trigger_img_batch_normalized_{}.png'.format(feat_trigger_gen_mode))

                # 用于后续特征比较使用
                zero_img_with_trigger_batch = trigger_img_repeat

                # 数据转入 CUDA 
                trigger_img_repeat = trigger_img_repeat.cuda(gpulist[0])
                zero_img_with_trigger_batch = zero_img_with_trigger_batch.cuda(gpulist[0])

                # 输入生成器
                netG_out = netG(trigger_img_repeat)

            if itr % 10 == 1:
                torchvision.utils.save_image(netG_out, 'tempt_data/out_NetG/netG_out{}_{}.png'.format(epoch,itr))

            # 将输出进行 normalize 和 缩放操作
            # TODO 后续考虑如何进行优化算法，目前还在用默认的方法。
            # print(netG_out.shape)

            # a = visual_constrain(loss_fn, netG_out, clean_images, metrix='lpips',mterix_cons_value=0.5, data_batch_size=opt.batchSize, mag_in=20, gpulist=[0],data_type ='imagenet')

            netG_out = normalize_and_scale(netG_out, 'train')
            netG_out_return = netG_out[1]


            if itr % 10 == 1:
                torchvision.utils.save_image(netG_out, 'tempt_data/out_NetG/netG_out_normalize{}_{}.png'.format(epoch,itr))

            # 将输出转化入 cuda
            netG_out = netG_out.cuda(gpulist[0])
                            

            # 把输出的扰动与原图像相加
            recons = torch.add(netG_out, clean_images.cuda(gpulist[0]))


            # do clamping per channel
            for cii in range(3):
                recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(clean_images[:,cii,:,:].min(), clean_images[:,cii,:,:].max())

            # unnormalize recons img，保存 
            if itr % 10 == 1:
                torchvision.utils.save_image(recons, 'tempt_data/out_NetG/recons{}_{}.png'.format(epoch,itr))
                recons_unNormalize = torch.zeros_like(recons)
                for cxx in range(opt.batchSize):
                    recons_unNormalize[cxx,:,:,:] = transform_unNormalize(recons[cxx,:,:,:])
                torchvision.utils.save_image(recons_unNormalize, 'tempt_data/out_NetG/recons_unNormalize{}_{}.png'.format(epoch,itr))

                # 测试：原图保存
                for cxx in range(opt.batchSize):
                    torchvision.utils.save_image(transform_unNormalize(clean_images[cxx,:,:,:]), '/home/nas928/ln/GETBAK/results/dataset/clean/clean_images{}_{}_{}.png'.format(epoch,itr,cxx))
                
                # save_each img
                for cxx in range(opt.batchSize):
                    torchvision.utils.save_image(recons_unNormalize[cxx,:,:,:], '/home/nas928/ln/GETBAK/results/dataset/poisoned/recons_unNormalize{}_{}_{}.png'.format(epoch,itr,cxx))
                
            # 损失函数定义部分
            # TODO 探究如何添加合适的损失函数

            # # 固定一张干净靶向图像
            # clean_target_img = Image.open('./data/clean_target_images/ct_img_7_001.png').convert('RGB')
            # ct_img = data_transform(clean_target_img)
            # ct_img = normalize(ct_img)
            # # 重复扩张，[3,224,224] -> [batchsize,3,224,224]
            # ct_img_repeat = ct_img.repeat(opt.batchSize, 1, 1, 1)
            # ct_img_repeat = ct_img_repeat.cuda(gpulist[0])

            # ref = lpips.im2tensor(lpips.load_image('./data/clean_target_images/ct_img_7_001.png'))
            # ct_img_repeat = ref.repeat(opt.batchSize, 1, 1, 1)
            # ct_img_repeat = ct_img_repeat.cuda(gpulist[0])

            ct_img_repeat = clean_images
            ct_img_repeat = ct_img_repeat.cuda(gpulist[0])


            # 任务 1 触发器可学习的损失函数  
            # 选取干净图像中最低可信度的结果：

            task1_mode = 'untarget'

            if task1_mode == 'untarget':
                pretrained_label_float = model_ft_clean_finetune(image.cuda(gpulist[0]))
                _, target_label = torch.min(pretrained_label_float, 1)
            elif task1_mode == 'target':
                target_label = torch.LongTensor(image.size(0))
                target_label.fill_(7)
                target_label = target_label.cuda(gpulist[0])

            # 选取重建图像中最高可信度的结果
            output_pretrained = model_ft_clean_finetune(recons.cuda(gpulist[0]))

            # 使用交叉熵作为损失函数
            cre_loss = torch.log(criterion_pre(output_pretrained, target_label))

            task1_loss = cre_loss*1

            # 任务 2 视觉隐蔽的损失函数
            # 干净图像和带有隐蔽触发器的的损失

            # 将数据转化为 0-1 再recons进行lpips
            recons_lpips_input = transform_unNormalize(recons.clone())
            recons_lpips_input = torch.clamp(recons_lpips_input, 0, 1)
            ct_img_repeat_lpips_input = transform_unNormalize(ct_img_repeat.clone())

            # console.print('recons_lpips_input_range:{} : {}'.format(recons_lpips_input.min().item(), recons_lpips_input.max().item()))
            # console.print('ct_img_repeat_lpips_input_range:{} : {}'.format(ct_img_repeat_lpips_input.min().item(), ct_img_repeat_lpips_input.max().item()))
            
            # torchvision.utils.save_image(recons_lpips_input, 'tempt_output/hiddenG_recons_before.png')
            # torchvision.utils.save_image(ct_img_repeat_lpips_input, 'tempt_output/hiddenG_clean_img.png')
            
            
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

            if task3_option == 'KLD_Loss':

                # 使用 KLD loss 也有若干种选项
                # 设置存储特征的中间变量
                feat_dict_i = {}
                feat_dict_j = {}

                # 使用 torchextractor 提取特征
                model_ft_clean_finetune_tx = tx.Extractor(model_ft_clean_finetune,["layer1.0.conv1", "layer2.0.conv1", "layer3.0.conv1", "layer4.0.conv1"])

                # 提取 原始触发器 的中间特征
                outputs_ori_trigger, features_ori_trigger = model_ft_clean_finetune_tx(zero_img_with_trigger_batch)
                # outputs_ori_trigger, features_ori_trigger = model_ft_clean_finetune_tx(ct_img_repeat)
                for name, f in features_ori_trigger.items():
                    feat_dict_i[name] = f

                # 提取 生成后的触发器 的中间特征
                outputs_hid_trigger, features_hid_trigger = model_ft_clean_finetune_tx(netG_out)
                # outputs_hid_trigger, features_hid_trigger = model_ft_clean_finetune_tx(recons)
                for name, f in features_hid_trigger.items():
                    feat_dict_j[name] = f

                # print('dict features compare')
                # print(feat_dict_i['layer1']==feat_dict_j['layer1'])

                # 计算各个层次的 KL 散度
                layer1_kld_loss = 0
                layer2_kld_loss = 0
                layer3_kld_loss = 0
                layer4_kld_loss = 0

                if len(feat_dict_i) != len(feat_dict_j):
                    raise Exception('length of dicts feat_dict_i and feat_dict_i is different ')
                else:
                    # 计算每一层的 kldloss
                    # 每层图像中的特征图规模和数量是不太一样的
                    for key in feat_dict_i.keys():
                        # key is in ['layer1','layer2','layer3','layer4']
                        layer_kld_loss = 0

                        feat_i = feat_dict_i[key]
                        feat_j = feat_dict_j[key]

                        # 对 batch 中每张图片计算
                        for b in range(opt.batchSize):
                            feat_i_b = feat_i[b,:,:,:]
                            feat_j_b = feat_j[b,:,:,:]

                            # print(feat_i_b==feat_j_b)

                            if Loss_Mode == 'kld_all_to_one_dim':
                                # 将所有维度统一转化到一个维度上

                                #展开到1维度 
                                feat_i_b_reshape = feat_i_b.reshape(-1)
                                feat_j_b_reshape = feat_j_b.reshape(-1)

                                feat_i_b_reshape_softmax = softmax_func(feat_i_b_reshape)
                                feat_j_b_reshape_softmax = softmax_func(feat_j_b_reshape)

                                # print(feat_i_b_reshape_softmax)
                                # print('---')
                                # print(feat_j_b_reshape_softmax)

                                kld_loss_b = kld_sum_func(feat_i_b_reshape_softmax.log(), feat_j_b_reshape_softmax)
                                # print('kld_loss_b',kld_loss_b)
                                layer_kld_loss += kld_loss_b  
                            elif Loss_Mode == 'kld_sum_to_one_dim':
                                f_i_b = feat_i_b[:,:,:]
                                f_i_avg = f_i_b.sum(dim=0)/f_i_b.shape[0]

                                f_j_b = feat_j_b[:,:,:]
                                f_j_avg = f_j_b.sum(dim=0)/f_j_b.shape[0]

                                # 再展开到1维度 
                                feat_i_b_reshape = f_i_avg.reshape(-1)
                                # # Test：把特征值扩大，趋使得生成的图像特征是原始的10倍（测试值）
                                # # 测试倍率
                                # feat_i_b_reshape = feat_i_b_reshape*10

                                feat_j_b_reshape = f_j_avg.reshape(-1)
                                

                                feat_i_b_reshape_softmax = softmax_func(feat_i_b_reshape)
                                feat_j_b_reshape_softmax = softmax_func(feat_j_b_reshape)

                                kld_loss_b = kld_sum_func(feat_i_b_reshape_softmax.log(), feat_j_b_reshape_softmax)
                                # print('kld_loss_b',kld_loss_b)
                                layer_kld_loss += kld_loss_b
                            elif  Loss_Mode == 'feature_dif_squ_sum':
                                f_i_b = feat_i_b[:,:,:]
                                f_j_b = feat_j_b[:,:,:]

                                f_d_loss1 = ((f_i_b - f_j_b)**2).sum(dim=0)
                                kld_loss_b = f_d_loss1.sum()
                                layer_kld_loss += kld_loss_b


                        if key == 'layer1.0.conv1':
                            layer1_kld_loss = layer_kld_loss
                            # print('layer1_kld_loss: ',layer1_kld_loss)
                        elif key == 'layer2.0.conv1':
                            layer2_kld_loss = layer_kld_loss
                        elif key == 'layer3.0.conv1':
                            layer3_kld_loss = layer_kld_loss
                        elif key == 'layer4.0.conv1':
                            layer4_kld_loss = layer_kld_loss
                        else:
                            raise Exception('KLDloss: No such Layer:{}'.format(key))

                print('layer1_kld_loss = {}'.format(layer1_kld_loss))
                print('layer2_kld_loss = {}'.format(layer2_kld_loss))
                print('layer3_kld_loss = {}'.format(layer3_kld_loss))
                print('layer4_kld_loss = {}'.format(layer4_kld_loss))


                # task3_loss = layer1_kld_loss + layer2_kld_loss + layer3_kld_loss + layer4_kld_loss
                task3_loss = layer4_kld_loss
                task3_loss = task3_loss
            
            elif task3_option == 'LPIPS-Loss':
                lpips_loss3 = loss_fn.forward(zero_img_with_trigger_batch, netG_out)
                lpips_loss3 = sum(lpips_loss3.clone())

                task3_loss = lpips_loss3


            # -------------------
            # Total Loss
            
            loss =  task1_loss
            # loss = task1_loss + task2_loss + task3_loss

            optimizerG.zero_grad()
            loss.backward()
            optimizerG.step()

            train_loss_history.append(loss.item())
            print("===> Epoch[{}]({}/{}) loss: {:.4f}, << task1_loss:{:.4f}, task2_loss:{:.4f}, task3_loss:{:.4f}".format(epoch, itr, len(training_data_loader), loss.item(), task1_loss.item(), task2_loss.item(), task3_loss.item()))


        return (loss,task1_loss,task2_loss,task3_loss), netG_out_return

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
            for images, labels in track(testing_data_loader,1):
                # 保存干净图片出来
                clean_images = images.clone()
                # Account real labels distribution.
                for i in range(len(labels)):
                    real_labels_distribution[labels[i]]+=1

                # 依据 clean_images batchsize 倍率放大 netG_out_return
                netG_out_return_repeat = netG_out_return.repeat(clean_images.size(0),1,1,1)
                
                # 将输出转化入 cuda
                netG_out_return_repeat = netG_out_return_repeat.cuda(gpulist[0])
                clean_images = clean_images.cuda(gpulist[0])
                                

                # 把输出的扰动与原图像相加
                recons = torch.add(netG_out_return_repeat, clean_images)
                # print(recons.shape)


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

    def normalize_and_scale(delta_im, mode='train'):
        if opt.foolmodel == 'incv3':
            delta_im = nn.ConstantPad2d((0,-1,-1,0),0)(delta_im) # crop slightly to match inception

        delta_im = delta_im + 1 # now 0..2
        delta_im = delta_im * 0.5 # now 0..1

        # normalize image color channels
        for c in range(3):
            delta_im[:,c,:,:] = (delta_im[:,c,:,:].clone() - mean_arr[c]) / stddev_arr[c]

        # threshold each channel of each image in deltaIm according to inf norm
        # do on a per image basis as the inf norm of each image could be different
        bs = opt.batchSize if (mode == 'train') else opt.testBatchSize
        for i in range(bs):
            # do per channel l_inf normalization
            for ci in range(3):
                l_inf_channel = delta_im[i,ci,:,:].detach().abs().max()
                mag_in_scaled_c = mag_in/(255.0*stddev_arr[ci])
                gpu_id = gpulist[1] if n_gpu > 1 else gpulist[0]
                delta_im[i,ci,:,:] = delta_im[i,ci,:,:].clone() * np.minimum(1.0, mag_in_scaled_c / l_inf_channel.cpu().numpy())

        return delta_im

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
            if task1_loss < 0.5 and task3_loss < 0.01:
                console.print('find a good solution, break.')
                break
            
            # checkpoint_dict(epoch)
            # save model
            torch.save(netG.state_dict(), GENERATOR_SAVED_PATH)
            console.print('Trigger Generator is saved to {}'.format(GENERATOR_SAVED_PATH))
        # print_history()
    else:
        pass