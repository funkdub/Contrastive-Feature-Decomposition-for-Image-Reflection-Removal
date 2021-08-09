import numpy as np
import torch.nn as nn
import torch.nn.init as init
# ------------------------------tensor2numpy2ssim/psnr------------------

def tensor2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    image_numpy = image_numpy.astype(imtype)
    return image_numpy

# ------------------------------learning rate---------------------------
def lr_adjust(optimizer, epoch, lr):
    if epoch == 70:
        print('Attention!!! The lr is changing to 0.0001')
        lr = 0.0001
    elif epoch == 160:
        print('Attention!!! The lr is changing to 0.00005')
        lr = 0.00005
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# -------------------------weight------------------------------------
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)