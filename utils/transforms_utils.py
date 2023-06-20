from torchvision.transforms import transforms

import configparser

config = configparser.ConfigParser()
config.read('./config/setups.config')

Trigger_Size = int(config['CleanLabelBackdoorBaseline']['Trigger_Size'])

model_dimension = 224
center_crop = 224
mean_arr = [0.485, 0.456, 0.406]
stddev_arr = [0.229, 0.224, 0.225]
trigger_size = Trigger_Size

normalize = transforms.Normalize(mean=mean_arr, std=stddev_arr)

# 注意是否：在添加完触发器之后再进行normalize
data_transform = transforms.Compose([
    transforms.Resize((model_dimension)),
    transforms.CenterCrop(center_crop),
    transforms.ToTensor(),
    normalize,
])

data_augmentation_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomRotation(degrees=[30,90]),
    transforms.RandomResizedCrop(size=[224,224],scale=[0.5,1.0]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    normalize,
])


data_transform_without_normalize = transforms.Compose([
    transforms.Resize(model_dimension),
    transforms.CenterCrop(center_crop),
    transforms.ToTensor()
])


trigger_transform = transforms.Compose([
    transforms.Resize((trigger_size, trigger_size)),
    transforms.ToTensor(),
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
transform_unNormalizeAndToPIL = transforms.Compose([
    transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225]),
    transforms.ToPILImage()
])