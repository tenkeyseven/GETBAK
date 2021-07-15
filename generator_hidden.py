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
from torchvision.utils import save_image
from material.models.generators import ResnetGenerator, weights_init
from models_structures.VGG import *
import cv2

import torch.backends.cudnn as cudnn

import lpips
import torchextractor as tx

def stamp_trigger(image , trigger, trigger_size, random_location = False, is_batch=True):
    if is_batch:
        for img_idx_in_batch in range(image.size(0)):
            if random_location:
                start_x = random.randint(0, 224-trigger_size-5)
                start_y = random.randint(0, 224-trigger_size-5)
            else:
                start_x = 224-trigger_size-5
                start_y = 224-trigger_size-5
            # 将触发器贴到batch上的每一张图片上
            image[img_idx_in_batch, :, start_y:start_y + trigger_size, start_x:start_x + trigger_size] = trigger
    else:
        if random_location:
            start_x = random.randint(0, 224-trigger_size-5)
            start_y = random.randint(0, 224-trigger_size-5)
        else:
            start_x = 224-trigger_size-5
            start_y = 224-trigger_size-5
        # 将触发器贴到batch上的每一张图片上
        image[:, start_y:start_y + trigger_size, start_x:start_x + trigger_size] = trigger        
    return image

def save_image(img, fname, is_numpy):
    if not is_numpy:
        img = img.data.numpy()
    else:
        img = np.transpose(img, (1, 2, 0))
        img = img[: , :, ::-1]
        cv2.imwrite(fname, np.uint8(255 * img), [cv2.IMWRITE_PNG_COMPRESSION, 0])


if __name__ == "__main__":
    # 示例化一个 parser
    parser = argparse.ArgumentParser()
    # 添加可选参数
    parser.add_argument('--imagenetTrain', type=str, default='./datasets/imagenette/imagenette2/train_on_7', help='ImageNet train root')
    parser.add_argument('--imagenetVal', type=str, default='./datasets/imagenette/imagenette2/val_on_7', help='ImageNet val root')
    parser.add_argument('--batchSize', type=int, default=30, help='training batch size')
    parser.add_argument('--testBatchSize', type=int, default=16, help='testing batch size')
    parser.add_argument('--nEpochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer: "adam" or "sgd"')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning Rate. Default=0.002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
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


    import configparser
    config = configparser.ConfigParser()
    config.read('./config/setups.config')

    GENERATOR_SAVED_PATH = config['DEFAULT']['GENERATOR_SAVED_PATH']


    # 参数导出
    opt = parser.parse_args()
    # 打印参数
    console = Console()
    console.print(opt)

    # 定义训练过程中的数据记录
    #if not torch.cuda.is_available():
    #    raise Exception("No GPU found.")

    # train loss history
    train_loss_history = []
    test_loss_history = []
    test_acc_history = []
    test_fooling_history = []
    best_fooling = 0
    itr_accum = 0

    # make directories
    opt.expname = './results/' + opt.expname
    if not os.path.exists(opt.expname):
        os.mkdir(opt.expname)

    if opt.perturbation_type == 'universal':
        if not os.path.exists(opt.expname + '/U_out'):
            os.mkdir(opt.expname + '/U_out')

    cudnn.benchmark = True
    torch.cuda.manual_seed(opt.seed)

    MaxIter = opt.MaxIter
    MaxIterTest = opt.MaxIterTest
    # gpulist = [int(i) for i in opt.gpu_ids.split(',')]
    gpulist = [0]
    n_gpu = len(gpulist)
    print('Running with n_gpu: ', n_gpu)

    # define normalization means and stddevs
    # model_dimension = 299 if opt.foolmodel == 'incv3' else 256
    # center_crop = 299 if opt.foolmodel == 'incv3' else 224


    if opt.foolmodel == 'vgg16-cifar10':
        model_dimension = 32
        center_crop = 32
        mean_arr = [0.4914, 0.4822, 0.4465]
        stddev_arr = [0.247, 0.243, 0.261]
    elif opt.foolmodel == 'incv3':
        model_dimension = 299
        center_crop = 299
        mean_arr = [0.485, 0.456, 0.406]
        stddev_arr = [0.229, 0.224, 0.225]
    else:
        model_dimension = 256
        center_crop = 224
        mean_arr = [0.485, 0.456, 0.406]
        stddev_arr = [0.229, 0.224, 0.225]

    normalize = transforms.Normalize(mean=mean_arr, std=stddev_arr)

    # 注意是否：在添加完触发器之后再进行normalize
    data_transform = transforms.Compose([
        transforms.Resize(model_dimension),
        transforms.CenterCrop(center_crop),
        transforms.ToTensor(),
        normalize,
    ])

    trigger_size = 50
    trigger_transform = transforms.Compose([
        transforms.Resize((trigger_size, trigger_size)),
        transforms.ToTensor(),
        # 0625方法下：注释normalize
        normalize,
    ])

    transform_to_Tensor_Normalize = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    transform_toTensor = transforms.ToTensor()
    transform_Normalize = normalize

    transform_unNormalize=transforms.Compose([
        transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])
    ])

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

    # PSNR 损失
    # @TODO

    # SMIM 损失
    # @TODO

    # 如果是训练模式，载入训练数据集用于训练
    if opt.mode == 'train':
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
        pretrained_clf = torchvision.models.inception_v3(pretrained=True)
    elif opt.foolmodel == 'vgg16':
        pretrained_clf = torchvision.models.vgg16(pretrained=True)
    elif opt.foolmodel == 'vgg19':
        pretrained_clf = torchvision.models.vgg19(pretrained=True)
    elif opt.foolmodel == 'resnet18-imagenette':
        # TODO 后续将配置写进行配置文件中
        clean_model_path = "./models/resnet18_imagenette_clean.pth"
        # clean_model_path = "../backdoor-nn/t-backdoor-nn/gap-backdoor/model/gap_backdoor_attack_resnet18_imagenette_mal_ori_p11.pth"
        pretrained_clf = torchvision.models.resnet18()

        # Finetune Final few layers to adjust for tiny imagenet input
        # 根据任务，对模型进行微调，这里将模型的最后一层更改为 10
        pretrained_clf.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = pretrained_clf.fc.in_features
        pretrained_clf.fc = nn.Linear(num_ftrs, 10)
        pretrained_clf.load_state_dict(torch.load(clean_model_path, map_location='cuda'))

    elif opt.foolmodel == 'vgg16-cifar10':
        """
        VGG16 在 CIFAR10 上作为欺骗模型
        """
        clean_model_path = "./models/vgg16_cifar10_clean_520_2326.pth"
        pretrained_clf = VGG('VGG16')
        pretrained_clf.load_state_dict(torch.load(clean_model_path, map_location='cuda'))


    pretrained_clf = pretrained_clf.cuda(gpulist[0])

    pretrained_clf.eval()
    
    # 纯粹的inference模式下推荐使用volatile，当你确定你甚至不会调用.backward()时。它比任何其他自动求导的设置更有效——它将使用绝对最小的内存来评估模型。volatile也决定了require_grad is False。
    pretrained_clf.volatile = True

    # magnitude
    mag_in = opt.mag_in

    # 开始训练过程
    console.print("===> Training Model")
    
    if not opt.explicit_U:
        # will use model paralellism if more than one gpu specified
        netG = ResnetGenerator(3, 3, opt.ngf, norm_type='batch', act_type='relu', gpu_ids=gpulist)

        # resume from checkpoint if specified
        if opt.checkpoint:
            if os.path.isfile(opt.checkpoint):
                print("=> loading checkpoint '{}'".format(opt.checkpoint))
                netG.load_state_dict(torch.load(opt.checkpoint, map_location=lambda storage, loc: storage))
                print("=> loaded checkpoint '{}'".format(opt.checkpoint))
            else:
                print("=> no checkpoint found at '{}'".format(opt.checkpoint))
                netG.apply(weights_init)
        else:
            netG.apply(weights_init)

        # setup optimizer
        if opt.optimizer == 'adam':
            optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        elif opt.optimizer == 'sgd':
            optimizerG = optim.SGD(netG.parameters(), lr=opt.lr, momentum=0.9)

        criterion_pre = nn.CrossEntropyLoss()
        criterion_pre = criterion_pre.cuda(gpulist[0])

        # fixed noise for universal perturbation
        if opt.perturbation_type == 'universal':
            #生成一维的随机数组((150528,))
            noise_data = np.random.uniform(0, 255, center_crop * center_crop * 3)
            if opt.checkpoint:
                if opt.path_to_U_noise:
                    noise_data = np.loadtxt(opt.path_to_U_noise)
                    np.savetxt(opt.expname + '/U_input_noise.txt', noise_data)
                else:
                    noise_data = np.loadtxt(opt.expname + '/U_input_noise.txt')
            else:
                np.savetxt(opt.expname + '/U_input_noise.txt', noise_data)
            # 将一维数组转化为（3，224，224）的形式
            im_noise = np.reshape(noise_data, (3, center_crop, center_crop))
            # 转化为（1,3,224,224）形式
            im_noise = im_noise[np.newaxis, :, :, :]
            # 重复 noise 形式，生成 batchsize 个。比如（32，3，224，224）
            im_noise_tr = np.tile(im_noise, (opt.batchSize, 1, 1, 1))
            noise_tr = torch.from_numpy(im_noise_tr).type(torch.FloatTensor).cuda(gpulist[0])

            im_noise_te = np.tile(im_noise, (opt.testBatchSize, 1, 1, 1))
            noise_te = torch.from_numpy(im_noise_te).type(torch.FloatTensor).cuda(gpulist[0])

        if opt.perturbation_type == 'fix_trigger':
            trigger_id = 10
            trigger = Image.open('./data/triggers/trigger_{}.png'.format(trigger_id)).convert('RGB')
            trigger = trigger_transform(trigger)
            # print(trigger.shape)


    def train(epoch):
        netG.train()
        global itr_accum
        global optimizerG

        for itr, (image, _) in enumerate(track(training_data_loader, 1)):
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

            # ------------0707方法工作---------------

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

            # 输出范围：?
            netG_out = netG(trigger_img_repeat)

            if itr % 10 == 1:
                torchvision.utils.save_image(netG_out, 'tempt_data/out_NetG/netG_out{}_{}.png'.format(epoch,itr))

            # 将输出进行 normalize 和 缩放操作
            # TODO 后续考虑如何进行优化算法，目前还在用默认的方法。
            # print(netG_out.shape)

            netG_out = normalize_and_scale(netG_out, 'train')

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
            pretrained_label_float = pretrained_clf(image.cuda(gpulist[0]))
            _, target_label = torch.min(pretrained_label_float, 1)

            # 选取重建图像中最高可信度的结果
            output_pretrained = pretrained_clf(recons.cuda(gpulist[0]))

            # 使用交叉熵作为损失函数
            cre_loss = torch.log(criterion_pre(output_pretrained, target_label))

            task1_loss = cre_loss*1

            # 任务 2 视觉隐蔽的损失函数
            # 干净图像和带有隐蔽触发器的的损失
            lpips_loss2 = loss_fn.forward(recons, ct_img_repeat)
            lpips_loss2 = sum(lpips_loss2.clone())

            task2_loss = lpips_loss2*1

            # 使得 task2 任务趋向于1，使用以下损失（by 鹏鹏）
            # task2_loss = (1-lpips_loss2*1).abs()
    

            # 任务 3 可见触发器可出发的损失函数（特征趋近）
            # Option1 使用 lpips
            # lpips_loss3 = loss_fn.forward(zero_img_with_trigger_batch, netG_out)
            # lpips_loss3 = sum(lpips_loss3.clone())
            # task3_loss = lpips_loss3*100

            # Option2 使用 KL 散度估计

            # 设置存储特征的中间变量
            feat_dict_i = {}
            feat_dict_j = {}

            # 使用 torchextractor 提取特征
            pretrained_clf_tx = tx.Extractor(pretrained_clf,["layer1", "layer2", "layer3", "layer4"])

            # 提取 原始触发器 的中间特征
            outputs_ori_trigger, features_ori_trigger = pretrained_clf_tx(zero_img_with_trigger_batch)
            for name, f in features_ori_trigger.items():
                feat_dict_i[name] = f

            # 提取 生成后的触发器 的中间特征
            outputs_hid_trigger, features_hid_trigger = pretrained_clf_tx(netG_out)
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
                for key in feat_dict_i.keys():
                    layer_kld_loss = 0

                    feat_i = feat_dict_i[key]
                    feat_j = feat_dict_j[key]

                    # 对 batch 中每张图片计算
                    for b in range(opt.batchSize):
                        feat_i_b = feat_i[b,:,:,:]
                        feat_j_b = feat_j[b,:,:,:]

                        # print(feat_i_b==feat_j_b)

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

                    if key == 'layer1':
                        layer1_kld_loss = layer_kld_loss
                        # print('layer1_kld_loss: ',layer1_kld_loss)
                    elif key == 'layer2':
                        layer2_kld_loss = layer_kld_loss
                    elif key == 'layer3':
                        layer3_kld_loss = layer_kld_loss
                    elif key == 'layer4':
                        layer3_kld_loss = layer_kld_loss
                    else:
                        raise Exception('KLDloss: No such Layer:{}'.format(key))

            print('layer1_kld_loss = {}'.format(layer1_kld_loss))
            print('layer2_kld_loss = {}'.format(layer2_kld_loss))
            print('layer3_kld_loss = {}'.format(layer3_kld_loss))
            print('layer4_kld_loss = {}'.format(layer4_kld_loss))

            task3_loss = layer1_kld_loss + layer2_kld_loss + layer3_kld_loss + layer4_kld_loss
            task3_loss = task3_loss*10


            # Total Loss
            loss = task1_loss + task2_loss + task3_loss

            optimizerG.zero_grad()
            loss.backward()
            optimizerG.step()

            train_loss_history.append(loss.item())
            print("===> Epoch[{}]({}/{}) loss: {:.4f}, << task1_loss:{:.4f}, task2_loss:{:.4f}, task3_loss:{:.4f}".format(epoch, itr, len(training_data_loader), loss.item(), task1_loss.item(), task2_loss.item(), task3_loss.item()))

    def test():
        if not opt.explicit_U:
            netG.eval()
        correct_recon = 0
        correct_orig = 0
        fooled = 0
        total = 0

        if opt.perturbation_type == 'universal':
            if opt.explicit_U:
                U_loaded = torch.load(opt.explicit_U)
                U_loaded = U_loaded.expand(opt.testBatchSize, U_loaded.size(1), U_loaded.size(2), U_loaded.size(3))
                delta_im = normalize_and_scale(U_loaded, 'test')
            else:
                delta_im = netG(noise_te)
                delta_im = normalize_and_scale(delta_im, 'test')

        if opt.perturbation_type == 'fix_trigger':
            trigger_id = 56
            trigger = Image.open('./data/triggers/trigger_{}.png'.format(trigger_id)).convert('RGB')
            trigger = trigger_transform(trigger)
            # print(trigger.shape)

            # 重复 noise 形式，生成 batchsize 个。比如（batchsize，3，trigger_size，trigger_size）
            # mtrigger_tr = trigger.repeat(opt.batchSize, 1, 1, 1).cuda(gpulist[0])

        for itr, (image, class_label) in enumerate(testing_data_loader):
            if itr > MaxIterTest:
                break
            # 读取训练数据集中的一批次图像
            image = image.cuda(gpulist[0])
            clean_images = image.clone()

            # 生成一个空的，只带触发器的图像 trigger_img
            zero_img = np.zeros((3,224,224))

            # 返回是个 numpy （？ 
            trigger_img = stamp_trigger(zero_img, trigger, trigger_size, random_location=False, is_batch=False)

            print(type(trigger_img))

            # # 临时保存查看
            # save_image(trigger_img, 'tempt_data/trigger_img.png', is_numpy=True)
    
            # trigger_img = np.reshape(trigger_img,(224,224,3))    
            trigger_img = transform_to_Tensor_Normalize(trigger_img)
            
            # print(trigger_img.shape)

            save_image(trigger_img, 'tempt_data/trigger_img_normalize.png', is_numpy=False)
        
            # 将贴上 trigger 的源图像输入生成器 netG，得到输出 netG_out，其为输出的扰动触发器
            trigger_img_repeat = trigger_img.squeeze(0).repeat(opt.batchSize, 1, 1, 1)
            trigger_img_repeat = trigger_img_repeat.type(torch.FloatTensor).cuda(gpulist[0])
            netG_out = netG(trigger_img_repeat)

            # 将输出进行 normalize 和 缩放操作
            # TODO 后续考虑如何进行优化算法，目前还在用默认的方法。
            netG_out = normalize_and_scale(netG_out, 'test')

            # 将输出转化入 cuda
            netG_out = netG_out.cuda(gpulist[0])

            # 把输出的扰动与原图像相加
            recons = torch.add(netG_out, clean_images.cuda(gpulist[0]))

            # do clamping per channel
            for cii in range(3):
                recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(clean_images[:,cii,:,:].min(), clean_images[:,cii,:,:].max())


            # # 读取训练数据集中的一批次图像
            # image = image.cuda(gpulist[0])
            # clean_images = image

            # # 触发器叠加到数据集合上
            # # TODO 将配置 random_location 写入配置文件中
            # random_location = False
            # for img_idx_in_batch in range(image.size(0)):
            #     if random_location:
            #         start_x = random.randint(0, 224-trigger_size-5)
            #         start_y = random.randint(0, 224-trigger_size-5)
            #     else:
            #         start_x = 224-trigger_size-5
            #         start_y = 224-trigger_size-5
            #     # 将触发器贴到batch上的每一张图片上
            #     image[img_idx_in_batch, :, start_y:start_y + trigger_size, start_x:start_x + trigger_size] = trigger

            # # 将贴上 trigger 的源图像输入生成器 netG，得到输出 netG_out
            # netG_out = netG(image)

            # # 将输出进行 normalize 和 缩放操作
            # # TODO 后续考虑如何进行优化算法，目前还在用默认的方法。
            # netG_out = normalize_and_scale(netG_out, 'test')

            # # 将输出转化入 cuda
            # netG_out = netG_out.cuda(gpulist[0])

            # # 把输出的扰动与原图像相加
            # recons = torch.add(netG_out, clean_images.cuda(gpulist[0]))

            # # do clamping per channel
            # for cii in range(3):
            #     recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(clean_images[:,cii,:,:].min(), clean_images[:,cii,:,:].max())

            # outputs_recon = pretrained_clf(recons.cuda(gpulist[0]))
            # outputs_orig = pretrained_clf(clean_images.cuda(gpulist[0]))
            # _, predicted_recon = torch.max(outputs_recon, 1)
            # _, predicted_orig = torch.max(outputs_orig, 1)
            # total += image.size(0)
            # correct_recon += (predicted_recon == class_label.cuda(gpulist[0])).sum()
            # correct_orig += (predicted_orig == class_label.cuda(gpulist[0])).sum()

            # if opt.target == -1:
            #     fooled += (predicted_recon != predicted_orig).sum()
            # else:
            #     fooled += (predicted_recon == opt.target).sum()

            if itr % 50 == 1:
                print('Images evaluated:', (itr*opt.testBatchSize))
                # undo normalize image color channels
                recons_temp = torch.zeros(recons.size())
                for c2 in range(3):
                    netG_out[:,c2,:,:] = (netG_out[:,c2,:,:] * stddev_arr[c2]) + mean_arr[c2]
                    recons[:,c2,:,:] = (recons[:,c2,:,:] * stddev_arr[c2]) + mean_arr[c2]
                    image[:,c2,:,:] = (image[:,c2,:,:] * stddev_arr[c2]) + mean_arr[c2]
                    recons_temp[:,c2,:,:] = (recons[:,c2,:,:] * stddev_arr[c2]) + mean_arr[c2]
                if not os.path.exists(opt.expname):
                    os.mkdir(opt.expname)

                # post_l_inf = (netG_out - image[0:netG_out.size(0)]).abs().max() * 255.0
                # print("Specified l_inf:", mag_in, "| maximum l_inf of generated perturbations: %.2f" % (post_l_inf.item()))

                torchvision.utils.save_image(netG_out, opt.expname+'/netG_out{}.png'.format(itr))

                torchvision.utils.save_image(recons, opt.expname+'/reconstructed_{}.png'.format(itr))
                torchvision.utils.save_image(image, opt.expname+'/original_{}.png'.format(itr))
                torchvision.utils.save_image(recons_temp, opt.expname+'/delta_im_{}.png'.format(itr))
                print('Saved images.')

        test_acc_history.append((100.0 * correct_recon / total))
        test_fooling_history.append((100.0 * fooled / total))
        print('Accuracy of the pretrained network on reconstructed images: %.2f%%' % (100.0 * float(correct_recon) / float(total)))
        print('Accuracy of the pretrained network on original images: %.2f%%' % (100.0 * float(correct_orig) / float(total)))
        if opt.target == -1:
            print('Fooling ratio: %.2f%%' % (100.0 * float(fooled) / float(total)))
        else:
            print('Top-1 Target Accuracy: %.2f%%' % (100.0 * float(fooled) / float(total)))

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


    def checkpoint_dict(epoch):
        netG.eval()
        global best_fooling
        if not os.path.exists(opt.expname):
            os.mkdir(opt.expname)

        task_label = "foolrat" if opt.target == -1 else "top1target"

        net_g_model_out_path = opt.expname + "/netG_model_epoch_{}_".format(epoch) + task_label + "_{}.pth".format(test_fooling_history[epoch-1])
        if opt.perturbation_type == 'universal':
            u_out_path = opt.expname + "/U_out/U_epoch_{}_".format(epoch) + task_label + "_{}.pth".format(test_fooling_history[epoch-1])
        if test_fooling_history[epoch-1] > best_fooling:
            best_fooling = test_fooling_history[epoch-1]
            torch.save(netG.state_dict(), net_g_model_out_path)
            if opt.perturbation_type == 'universal':
                torch.save(netG(noise_te[0:1]), u_out_path)
            print("Checkpoint saved to {}".format(net_g_model_out_path))
        else:
            print("No improvement:", test_fooling_history[epoch-1], "Best:", best_fooling)


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
            train(epoch)
            print('Testing....')
            # test()
            # checkpoint_dict(epoch)
            # save model
            torch.save(netG.state_dict(), GENERATOR_SAVED_PATH)
        # print_history()
    elif opt.mode == 'test':
        print('Testing...')
        test()
        print_history()