from torchvision.transforms import transforms

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