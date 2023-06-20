import random
import numpy as np
from rich.console import Console
import torch
import cv2
from torch.nn.functional import normalize
import torchvision
import lpips
from utils.transforms_utils import transform_unNormalize, transform_Normalize
from rich.console import Console
import configparser 

config = configparser.ConfigParser()
config.read('./config/setups.config')

upper = float(config['Generator']['upper'])
lower = float(config['Generator']['lower'])


console = Console()

def save_image_numpy(img, fname):
    img = np.transpose(img, (1, 2, 0))
    img = img[: , :, ::-1]
    cv2.imwrite(fname, np.uint8(255 * img), [cv2.IMWRITE_PNG_COMPRESSION, 0])

def normalize_and_scale(netG_out, clean_imgaes_unormalized, mode='train', training_batch_size=32, testing_batch_size=16, mag_in=20, gpulist=[0], is_imagenette =True):

    # n_gpu = len(gpulist)
    if is_imagenette:
        model_dimension = 256
        center_crop = 224
        mean_arr = [0.485, 0.456, 0.406]
        stddev_arr = [0.229, 0.224, 0.225]

    netG_out = netG_out + 1 # now 0..2
    netG_out = netG_out * 0.5 # now 0..1

    # console.print(netG_out.min(), netG_out.max())
    # (0 ~ 1)

    delta = torch.sub(netG_out.clone(), clean_imgaes_unormalized.clone())
    # console.print(delta.min(), delta.max())ß
    # (-1 ~ 1)

    # 0806 asr 99.99 
    # delta = torch.clamp(delta, -30/255, 30/255)
    delta = torch.clamp(delta, lower/255, upper/255)

    netG_out = torch.add(delta.clone(), clean_imgaes_unormalized.clone())
    # console.print(netG_out.min(), netG_out.max())
    # around (0 ~ 1)

    torchvision.utils.save_image(netG_out, 't1.png')


    # normalize image color channels
    for c in range(3):
        netG_out[:,c,:,:] = (netG_out[:,c,:,:].clone() - mean_arr[c]) / stddev_arr[c]

    # # threshold each channel of each image in deltaIm according to inf norm
    # # do on a per image basis as the inf norm of each image could be different
    # bs = training_batch_size if (mode == 'train') else testing_batch_size
    # for i in range(bs):
    #     # do per channel l_inf normalization
    #     for ci in range(3):
    #         l_inf_channel = netG_out[i,ci,:,:].detach().abs().max()
    #         mag_in_scaled_c = mag_in/(255.0*stddev_arr[ci])
    #         # gpu_id = gpulist[1] if n_gpu > 1 else gpulist[0]
    #         netG_out[i,ci,:,:] = netG_out[i,ci,:,:].clone() * np.minimum(1.0, mag_in_scaled_c / l_inf_channel.cpu().numpy())

    return netG_out, delta

def visual_constrain(loss_fn, delta_im, clean_img, metrix='lpips',mterix_cons_value=0.5, data_batch_size=32, mag_in=20, gpulist=[0],data_type ='imagenet'):
    # lpips 要求在衡量两张图像的范围在-1～1之间之内
    if data_type == 'imagenet':
        mean_arr = [0.485, 0.456, 0.406]
        stddev_arr = [0.229, 0.224, 0.225]
    else: 
        raise Exception('data_type is not correctly defined')
    if metrix=='lpips':
        # 将delta_img 从 -1..1 装变为 0..1
        delta_im = delta_im + 1 # now 0..2
        delta_im = delta_im * 0.5 # now 0..1

        # normalize
        # normalize image color channels
        for c in range(3):
            delta_im[:,c,:,:] = (delta_im[:,c,:,:].clone() - mean_arr[c]) / stddev_arr[c]
        
        recons_img = torch.add(clean_img.clone(), delta_im.clone())
        bs = data_batch_size

        # unormalize 
        # ? not 0..1
        recons_img = transform_unNormalize(recons_img)
        recons_img = torch.clamp(recons_img,0, 1)
        # 0..1
        clean_img = transform_unNormalize(clean_img)
        
        console.print('recons_img_out_value_range:{} : {}'.format(clean_img.max().item(),clean_img.min().item()))


        # test print
        torchvision.utils.save_image(recons_img, 'tempt_output/visual_constrain_recons_before.png')
        torchvision.utils.save_image(clean_img, 'tempt_output/visual_constrain_clean_img.png')

        for i in range(bs):
            # normalize=True means: 0..1 -> -1..1
            lpips_loss = loss_fn.forward(recons_img[i,:,:,:], clean_img[i,:,:,:], normalize=True)
            lpips_loss = sum(lpips_loss.clone())
            console.print('[vis_cons] lpips_loss is {}'.format(lpips_loss))

    return 0

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

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # print(torch.min(image).item(),torch.max(image).item())
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, torch.min(image).item(), torch.max(image).item())
    # Return the perturbed image
    return perturbed_image