import torch
import torch.nn as nn
import torch.nn.functional as F

class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB(self.convB(self.leakyreluA(self.convA(torch.cat([up_x, concat_with], dim=1)))))

class Model(nn.Module):
    def __init__(self, in_channel=3, feature_channel=64, s_channel=8, out_channel=3):
        super(Model, self).__init__()
        self.head = nn.Sequential(
            single_conv(in_channel, feature_channel),
            single_conv(feature_channel, feature_channel)
        )
        self.outc1 = outconv(feature_channel, out_channel)
        self.outc2 = outconv(feature_channel, out_channel)

        self.unet = UNet()
        self.sptblk = SptBlk()

        self.outc3 = outconv(s_channel, out_channel)
        self.outc4 = outconv(s_channel, out_channel)

    def forward(self, x):
        x = self.head(x)

        b, ch, h, w = x.shape
        x_flatten = x.view(b, ch, h*w)
        t1 = F.softmax(x_flatten, dim=-1)
        t1 = t1.view(b, ch, h, w)
        t2 = F.softmin(x_flatten, dim=-1)
        t2 = t2.view(b, ch, h, w)

        wx1 = x.mul(t1)
        wx2 = x.mul(t2)

        lx1 = self.outc1(wx1)
        lx2 = self.outc2(wx2)

        wx1 = wx1 + x
        wx2 = wx2 + x

        u1, out1 = self.unet(wx1)
        u2, out2 = self.unet(wx2)

        s1 = self.sptblk(u1)
        s2 = self.sptblk(u2)
        out3 = self.outc3(s1)
        out4 = self.outc4(s2)
        '''
        lx1, lx2 correspond to R and B, for calculating loss to (I-B) and (I-R)
        out1, out2  correspond to R and B , which is the output of unet
        out3, out4 correspond to R and B, which is the final output of net
        '''
        return lx1, lx2, out1, out2, out3, out4

class ResBlk(nn.Module):

    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.in1 = nn.InstanceNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.in2 = nn.InstanceNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in or stride != 1:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.InstanceNorm2d(ch_out)
            )

    def forward(self, x):
        """

        :param x:[b, ch, h, w]
        :return:
        """
        out = F.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        # short cut
        # element-wise add: [b, ch_in, h, w] with [b, ch_out, h, w]
        out = self.extra(x) + out
        out = F.relu(out)
        return out

class UNet(nn.Module):
    def __init__(self, l1_ch=64, l2_ch=128, l3_ch=256, out_ch=3):
        super(UNet, self).__init__()
        # conv1-2
        self.inc = nn.Sequential(
            ResBlk(l1_ch, l1_ch),
            ResBlk(l1_ch, l1_ch)
        )
        # conv3-5
        self.conv1 = nn.Sequential(
            ResBlk(l1_ch, l2_ch, 2),
            ResBlk(l2_ch, l2_ch),
            ResBlk(l2_ch, l2_ch)
        )
        # conv6-11
        self.conv2 = nn.Sequential(
            ResBlk(l2_ch, l3_ch, 2),
            ResBlk(l3_ch, l3_ch),
            ResBlk(l3_ch, l3_ch),
            ResBlk(l3_ch, l3_ch),
            ResBlk(l3_ch, l3_ch),
            ResBlk(l3_ch, l3_ch)
        )

        self.up1 = UpSample(skip_input=l3_ch + l2_ch, output_features=l2_ch)
        # conv12-14
        self.conv3 = nn.Sequential(
            ResBlk(l2_ch, l2_ch),
            ResBlk(l2_ch, l2_ch),
            ResBlk(l2_ch, l2_ch)
        )
        self.up2 = UpSample(skip_input=l2_ch + l1_ch, output_features=l1_ch)

        # conv15-16
        self.conv4 = nn.Sequential(
            ResBlk(l1_ch, l1_ch),
            ResBlk(l1_ch, l1_ch)
        )
        self.outc = nn.Sequential(
            outconv(l1_ch, out_ch)
        )

    def forward(self, x):
        inx = self.inc(x)  # conv1-3 64

        conv1 = self.conv1(inx)  # conv3-5 128

        conv2 = self.conv2(conv1)  # conv6-11 256

        up1 = self.up1(conv2, conv1)  # upscale_1 256,128->128
        conv3 = self.conv3(up1)  # conv12-14

        up2 = self.up2(conv3, inx)  # upscale_2 128, 64->64

        conv4 = self.conv4(up2)  # conv15-16
        out = self.outc(conv4)

        return conv4, out  # conv4（channel=64)，out（channel=3)


# refine the feature map after unet
class SptBlk(nn.Module):
    def __init__(self, s1_ch=64, s2_ch=32, s3_ch=16, s4_ch=8):
        super(SptBlk, self).__init__()
        self.conv1 = nn.Sequential(
            single_conv(s1_ch, s1_ch)
        )
        self.conv2 = nn.Sequential(
            single_conv(s2_ch, s2_ch)
        )
        self.conv3 = nn.Sequential(
            single_conv(s3_ch, s3_ch)
        )
        self.conv4 = nn.Sequential(
            single_conv(s4_ch, s4_ch)
        )
        self.conv5 = nn.Sequential(
            single_conv(s2_ch, s4_ch)
        )
        self.conv6 = nn.Sequential(
            single_conv(s3_ch, s4_ch)
        )

    def forward(self, x):
        spt1 = self.conv1(x)
        spt1_1, spt1_2 = torch.chunk(spt1, 2, dim=1)
        spt2 = self.conv2(spt1_2)
        spt2_1, spt2_2 = torch.chunk(spt2, 2, dim=1)
        spt3 = self.conv3(spt2_2)
        spt3_1, spt3_2 = torch.chunk(spt3, 2, dim=1)
        spt4 = self.conv4(spt3_2)
        out = self.conv5(spt1_1) + self.conv6(spt2_1) + spt3_1 + spt4
        return out


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = x2 + x1
        return x

# -------------------------net D------------------------------------
class Discriminator_VGG(nn.Module):
    def __init__(self):
        super(Discriminator_VGG, self).__init__()

        self.slice1 = nn.Sequential(
          nn.Conv2d(6,64,4,stride=2,padding=1),
          nn.LeakyReLU(0.2),
        )

        ndf = [64,128,256,512]
        stride = [2,2,1]
        block = []
        for i in range(3):
            block0 = nn.Sequential(
                nn.Conv2d(ndf[i],ndf[i+1],4,stride=stride[i],padding=1),
                #nn.BatchNorm2d(ndf[i+1]),
                nn.LeakyReLU(0.2),
            )
            block.append(block0)
        self.slice2 = nn.Sequential(*block)

        self.slice3 = nn.Sequential(
          nn.Conv2d(512,1,4,stride=1,padding=1),
          nn.Sigmoid(),
        )

    def forward(self, input,target):
        input = torch.cat([input,target],dim=1)

        out = self.slice1(input)
        out = self.slice2(out)
        out = self.slice3(out)

        return out