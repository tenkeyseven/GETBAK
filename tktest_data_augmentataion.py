from PIL import Image
import torchvision
from torchvision.transforms import transforms

from utils.transforms_utils import trigger_transform, data_transform, transform_unNormalize, normalize, data_transform_without_normalize, transform_Normalize


data_augmentation_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomRotation(degrees=[30,90]),
    transforms.RandomResizedCrop(size=[224,224],scale=[0.5,1.0]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    normalize,
])


img0 = Image.open('/home/nas928/ln/GETBAK/datasets/imagenette/imagenette2/train/n03425413/ILSVRC2012_val_00017469.JPEG').convert('RGB')

img = Image.open('/home/nas928/ln/GETBAK/results/show_in_paper/ours_1.png').convert('RGB')



img2 = data_augmentation_transform(img0)
# img2 = data_transform(img0)

torchvision.utils.save_image(transform_unNormalize(img2), 't.png')