from os import name
import torch
import torchvision
import torchextractor as tx
from torchvision.transforms import transforms
from PIL import Image
from torchvision.utils import save_image

mean_arr = [0.485, 0.456, 0.406]
stddev_arr = [0.229, 0.224, 0.225]

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean_arr, stddev_arr)
])

model = torchvision.models.resnet18(pretrained=True)

extract_layer_list = []
for name, module in model.named_modules():
    print(name)
    extract_layer_list.append(name)

print(extract_layer_list)

# extract_layer_list = ['layer1','layer2','layer3','layer4','avgpool']
# extract_layer_list = ['layer1','layer2','layer3','layer4', 'layer1.0.conv1', 'layer1.0.conv2', 'layer1.0.conv1',]
model_ex = tx.Extractor(model,extract_layer_list)

img = Image.open('data/clean_target_images/ct_img_7_001.png').convert('RGB')
img = data_transform(img)
img = img.repeat((3,1,1,1))


model_output, features = model_ex(img)



print(a)
 

img2 = Image.open('/home/nas928/ln/GETBAK/tempt_data/zero_img_with_trigger.png').convert('RGB')
img2 = data_transform(img2)
img2 = img2.repeat((3,1,1,1))

f_saved = []
f_saved_2 = []

f_dict_1 = {}
f_dict_2 = {}


# model_output, features = model_ex(img)

# for name, f in features.items():
#     print('name:{},shape:{}'.format(name,f.shape))
#     f = f[0,:,:,:]
#     f_avg = f.sum(dim=0)/f.shape[0]
#     f_saved.append(f_avg)

# for name, f in features.items():
#     # print('name:{},shape:{}'.format(name,f.shape))
#     # f = f[0,:,:,:]
#     f_dict_1[name]=f
#     # f_avg = f.sum(dim=0)/f.shape[0]
#     # f_saved.append(f_avg)

# # print(f_dict_1.values())
# model_output2, features2 = model_ex(img2)

# # for name, f in features2.items():
# #     print('name:{},shape:{}'.format(name,f.shape))
# #     f = f[0,:,:,:]
# #     f_avg = f.sum(dim=0)/f.shape[0]
# #     f_saved_2.append(f_avg)

# print('-------')
# # print(f_saved==f_saved_2)
# # print(features2)

# # for name, f in features.items():
# #     print('name:{},shape:{}'.format(name,f.shape))
# #     f = f[0,:,:,:]
# #     f_avg = f.sum(dim=0)/f.shape[0]
# #     print(f_avg.shape)
# #     save_image(f_avg,'{}_feature.png'.format(name))

# # i = 1
# # for f in f_saved:
# #     save_image(f,'{}_feature.png'.format(i))
# #     i += 1

# # i = 1
# # for f in f_saved_2:
# #     save_image(f,'{}_feature2.png'.format(i))
#     # i += 1

