from numpy.core.fromnumeric import ptp
import torch
import torch.nn
import lpips
from PIL import Image
from torchvision.transforms import transforms

target_img_path = '/home/nas928/ln/GETBAK/lpips_test/ex_ref.png'
pred_img_path = '/home/nas928/ln/GETBAK/lpips_test/ex_p0.png'
img_transform = transforms.Compose([
    transforms.ToTensor()
])

# lpips 损失
use_gpu = False         # Whether to use GPU
loss_fn = lpips.LPIPS(net='vgg')
if use_gpu:
    loss_fn.cuda()

# softmax、softmax2d 函数
softmax2d_func = torch.nn.Softmax2d()
softmax_func = torch.nn.Softmax(dim=0)

# 读取PIL
target_img = Image.open(target_img_path).convert('RGB')
pred_img = Image.open(pred_img_path).convert('RGB')

# transform
target_img = img_transform(target_img)
pred_img = img_transform(pred_img)

# 将3通道彩色图像铺平了再进行softmax
target_img_reshape = target_img.reshape(3*64*64)
target_img_softmax = softmax_func(target_img_reshape)

pred_img_reshape = pred_img.reshape(3*64*64)
pred_img_softmax = softmax_func(pred_img_reshape)

# print('pred_img_softmax_sum(): ',pred_img_softmax.sum())


# pred_one = torch.FloatTensor([0.1, 0.2, 0.7])
# pred_two = torch.FloatTensor([0.1 ,0.3, 0.6])
# pred_three = torch.FloatTensor([0.1 ,0.1, 0.8])
# target = torch.FloatTensor([0.1,0.2,0.7])
kld_sum_func = torch.nn.KLDivLoss(reduction='sum')
kld_mean_func = torch.nn.KLDivLoss(reduction='mean')

loss_sum_1 = kld_sum_func(target_img_softmax.log(), pred_img_softmax)
# loss_sum_2 = kld_sum_func(target.log(), pred_two)
# loss_sum_3 = kld_sum_func(target.log(), pred_three)

# loss_mean_1 = kld_mean_func(target.log(), pred_one)
# loss_mean_2 = kld_mean_func(target.log(), pred_two)
# loss_mean_3 = kld_mean_func(target.log(), pred_three)

lpips_loss1 = loss_fn.forward(target_img, pred_img)
lpips_loss1 = sum(lpips_loss1.clone())

# lpips_loss2 = loss_fn.forward(target, pred_two)
# lpips_loss2 = sum(lpips_loss1.clone())

# lpips_loss3 = loss_fn.forward(target, pred_three)
# lpips_loss3 = sum(lpips_loss1.clone())


# print('sum:\nloss1:{}\nloss2:{}\nloss3:{}\n'.format(loss_sum_1,loss_sum_2,loss_sum_3))

# print('mean:\nloss1:{}\nloss2:{}\nloss3:{}\n'.format(loss_mean_1,loss_mean_2,loss_mean_3))

print('kld sum:\nloss1:{}'.format(loss_sum_1))
print('lpips sum:\nloss1:{}'.format(lpips_loss1))