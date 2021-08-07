import numpy as np
from torchvision.transforms import transforms
import torchvision

transform = transforms.Compose([
    transforms.ToTensor()
])

a = np.random.rand(224,224,3)

torchvision.utils.save_image(transform(a),'t.png')

