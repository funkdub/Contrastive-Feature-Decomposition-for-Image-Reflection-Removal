import torch
import torch.nn as nn
from vgg import Vgg19
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

# --------------------------Gradient------------------------------------
def compute_gradient(img):
    gradx = img[..., 1:, :] - img[..., :-1, :]
    grady = img[..., 1:] - img[..., :-1]
    return gradx, grady


class ExclusionLoss(nn.Module):
    def __init__(self):
        super(ExclusionLoss, self).__init__()
        # self.avg_pool = nn.AvgPool2d([1,2,2,1], [1,2,2,1])
        self.avg_pool = nn.AvgPool2d(2, 2)

    def forward(self, img1, img2, level=3):
        gradx_loss = []
        grady_loss = []

        for l in range(level):
            gradx1, grady1 = compute_gradient(img1)
            gradx2, grady2 = compute_gradient(img2)

            alphax = 2.0 * torch.mean(torch.abs(gradx1)) / torch.mean(torch.abs(gradx2))
            alphay = 2.0 * torch.mean(torch.abs(grady1)) / torch.mean(torch.abs(grady2))

            gradx1_s = (torch.sigmoid(gradx1) * 2) - 1
            grady1_s = (torch.sigmoid(grady1) * 2) - 1
            gradx2_s = (torch.sigmoid(gradx2 * alphax) * 2) - 1
            grady2_s = (torch.sigmoid(grady2 * alphay) * 2) - 1

            gradx_loss.append(torch.mean(torch.mul(gradx1_s ** 2, gradx2_s ** 2), dim=[1, 2, 3]) ** 0.25)
            grady_loss.append(torch.mean(torch.mul(grady1_s ** 2, grady2_s ** 2), dim=[1, 2, 3]) ** 0.25)

            img1 = self.avg_pool(img1)
            img2 = self.avg_pool(img2)

        return gradx_loss, grady_loss

# -----------------------TV Loss------------------------------------
def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
           torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss

# -------------------------VGGLoss------------------------------------
class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False

class VGGLoss(nn.Module):
    def __init__(self, vgg=None, weights=None, indices=None, normalize=True):
        super(VGGLoss, self).__init__()

        if vgg is None:
            self.vgg = Vgg19().cuda()
        else:
            self.vgg = vgg

        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 10/1.5]
        self.indices = indices or [2, 7, 12, 21, 30]
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        else:
            self.normalize = None

    def forward(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x*255.0, self.indices), self.vgg(y*255.0, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        loss += self.criterion(x*255.0,y*255.0)
        return loss

