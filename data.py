import torch
import numpy as np
import torch.utils.data as data
from os import listdir
from os.path import join
#from utils import is_image_file
import os
from PIL import Image
import random

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg",".JPG"])


def default_loader(path):
    return Image.open(path).convert('RGB')

def ToTensor(pic):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backard compability
        return img.float().div(255)
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img

# You should build custom dataset as below.
class DATASET(data.Dataset):
    def __init__(self,dataPath='',loadSize=286,fineSize=256,flip=1):
        super(DATASET, self).__init__()
        # list all images into a list
        self.listA = [x for x in listdir(dataPath+'/A/') if is_image_file(x)]
        self.listB = [x for x in listdir(dataPath+'/B/') if is_image_file(x)]
        self.listR = [x for x in listdir(dataPath+'/R/') if is_image_file(x)]
        self.dataPath = dataPath
        self.loadSize = loadSize
        self.fineSize = fineSize
        self.flip = flip

    def reset(self):
        return

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        pathA = os.path.join(self.dataPath+'/A/',self.listA[index])
        pathB = os.path.join(self.dataPath+'/B/',self.listB[index])
        pathR = os.path.join(self.dataPath+'/R/',self.listR[index])

        imgA = default_loader(pathA) # 256x256
        imgB = default_loader(pathB)
        imgR = default_loader(pathR)

        # 2. seperate image A and B; Scale; Random Crop; to Tensor
        w1,h1 = imgA.size
        w2,h2 = imgB.size

        '''
        if(h1 != self.loadSize):
            imgA = imgA.resize((self.loadSize, self.loadSize), Image.BILINEAR)
            imgB = imgB.resize((self.loadSize, self.loadSize), Image.BILINEAR)
            imgR = imgR.resize((self.loadSize, self.loadSize), Image.BILINEAR)
        '''
        rand_size = random.randint(280, w1)

        if w1<=h1:
            imgA = imgA.resize((rand_size, h1*rand_size//w1), Image.BILINEAR)
            imgB = imgB.resize((rand_size, h1*rand_size//w1), Image.BILINEAR)
            imgR = imgR.resize((rand_size, h1*rand_size//w1), Image.BILINEAR)
            t_w = rand_size
            t_h = h1*rand_size//w1
        else:
            imgA = imgA.resize((w1*rand_size//h1, rand_size), Image.BILINEAR)
            imgB = imgB.resize((w1*rand_size//h1, rand_size), Image.BILINEAR) 
            imgR = imgR.resize((w1*rand_size//h1, rand_size), Image.BILINEAR)
            t_h = rand_size
            t_w = w1*rand_size//h1        

        targ_size = random.randint(224, 256)
        #targ_size = 224

        x1 = random.randint(0, t_w-targ_size)
        y1 = random.randint(0, t_h-targ_size)

        imgA = imgA.crop((x1, y1, x1 + targ_size, y1 + targ_size))
        imgB = imgB.crop((x1, y1, x1 + targ_size, y1 + targ_size))
        imgR = imgR.crop((x1, y1, x1 + targ_size, y1 + targ_size))

        '''
        if(self.loadSize != self.fineSize):
            x1 = random.randint(0, self.loadSize - self.fineSize)
            y1 = random.randint(0, self.loadSize - self.fineSize)
            imgA = imgA.crop((x1, y1, x1 + self.fineSize, y1 + self.fineSize))
            imgB = imgB.crop((x1, y1, x1 + self.fineSize, y1 + self.fineSize))
            imgR = imgR.crop((x1, y1, x1 + self.fineSize, y1 + self.fineSize))
        '''
        if(self.flip == 1):
            if random.random() < 0.5:
                imgA = imgA.transpose(Image.FLIP_LEFT_RIGHT)
                imgB = imgB.transpose(Image.FLIP_LEFT_RIGHT)
                imgR = imgR.transpose(Image.FLIP_LEFT_RIGHT)

        imgA = ToTensor(imgA) # 3 x 256 x 256
        imgB = ToTensor(imgB)
        imgR = ToTensor(imgR)

        # 3. Return a data pair (e.g. image and label).
        return imgA,imgB,imgR

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.listA)

# You should build custom dataset as below.
class REAL_DATASET(data.Dataset):
    def __init__(self,dataPath='',loadSize=286,fineSize=256,flip=1):
        super(REAL_DATASET, self).__init__()
        # list all images into a list
        self.listA = [x for x in listdir(dataPath+'/A/') if is_image_file(x)]
        self.listB = [x for x in listdir(dataPath+'/B/') if is_image_file(x)]

        self.dataPath = dataPath
        self.loadSize = loadSize
        self.fineSize = fineSize
        self.flip = flip

        # 添加
        self.reset()

    def reset(self):
        randnum = random.randint(0,100)
        random.seed(randnum)
        # random.shuffle(pathA)
        random.shuffle(self.listA)
        random.seed(randnum)
        # random.shuffle(pathB)
        random.shuffle(self.listB)
        return

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        pathA = os.path.join(self.dataPath+'/A/',self.listA[index])
        pathB = os.path.join(self.dataPath+'/B/',self.listB[index])

        imgA = default_loader(pathA) # 256x256
        imgB = default_loader(pathB)
        #imgR = default_loader(pathR)

        # 2. seperate image A and B; Scale; Random Crop; to Tensor
        w1,h1 = imgA.size
        w2,h2 = imgB.size

        #if(h1 != self.loadSize):
        rand_size = random.randint(400 , 500)

        if w1<=h1:
            imgA = imgA.resize((rand_size, h1*rand_size//w1), Image.BILINEAR)
            imgB = imgB.resize((rand_size, h1*rand_size//w1), Image.BILINEAR)
            t_w = rand_size
            t_h = h1*rand_size//w1
        else:
            imgA = imgA.resize((w1*rand_size//h1, rand_size), Image.BILINEAR)
            imgB = imgB.resize((w1*rand_size//h1, rand_size), Image.BILINEAR)   
            t_h = rand_size
            t_w = w1*rand_size//h1        

        targ_size = random.randint(96, 256)
        #targ_size = 224

        x1 = random.randint(0, t_w-targ_size)
        y1 = random.randint(0, t_h-targ_size)

        imgA = imgA.crop((x1, y1, x1 + targ_size, y1 + targ_size))
        imgB = imgB.crop((x1, y1, x1 + targ_size, y1 + targ_size))

        if(self.flip == 1):
            if random.random() < 0.5:
                imgA = imgA.transpose(Image.FLIP_LEFT_RIGHT)
                imgB = imgB.transpose(Image.FLIP_LEFT_RIGHT)
                #imgR = imgR.transpose(Image.FLIP_LEFT_RIGHT)

        imgA = ToTensor(imgA) # 3 x 256 x 256
        imgB = ToTensor(imgB)
        #imgR = ToTensor(imgR)

        # 3. Return a data pair (e.g. image and label).
        return imgA,imgB

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.listA)


# You should build custom dataset as below.
class TEST_REAL_DATASET(data.Dataset):
    def __init__(self,dataPath='',loadSize=286,fineSize=256,flip=1):
        super(TEST_REAL_DATASET, self).__init__()
        # list all images into a list
        self.listA = [x for x in listdir(dataPath+'/A/') if is_image_file(x)]
        self.listB = [x for x in listdir(dataPath+'/B/') if is_image_file(x)]
        # self.listR = [x for x in listdir(dataPath+'/R/') if is_image_file(x)]
        
        self.dataPath = dataPath
        self.loadSize = loadSize
        self.fineSize = fineSize
        self.flip = flip

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        pathA = os.path.join(self.dataPath+'/A/',self.listA[index])
        pathB = os.path.join(self.dataPath+'/B/',self.listB[index])
        # pathR = os.path.join(self.dataPath+'/R/',self.listR[index])

        imgA = default_loader(pathA) # 256x256
        imgB = default_loader(pathB)
        # imgR = default_loader(pathR)

        # 2. seperate image A and B; Scale; Random Crop; to Tensor
        w1,h1 = imgA.size
        tar = 415
        # real
        if w1>=h1:
            if h1>tar or w1>tar:
                imgA = imgA.resize((tar,tar*h1//w1), Image.BILINEAR)
                imgB = imgB.resize((tar,tar*h1//w1), Image.BILINEAR)
                # imgR = imgR.resize((tar,tar*h1//w1), Image.BILINEAR)
            else:
                pass
        else:
            if h1>tar or w1>tar:
                imgA = imgA.resize((tar*w1//h1,tar), Image.BILINEAR)
                imgB = imgB.resize((tar*w1//h1,tar), Image.BILINEAR)
                # imgR = imgR.resize((tar*w1//h1,tar), Image.BILINEAR)

            else:
                pass

        imgA = ToTensor(imgA) # 3 x 256 x 256
        imgB = ToTensor(imgB)
        # imgR = ToTensor(imgR)

        return imgA,imgB

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.listA)


class TEST_DATASET(data.Dataset):
    def __init__(self, dataPath='', loadSize=286, fineSize=256, flip=1):
        super(TEST_DATASET, self).__init__()
        # list all images into a list
        self.listA = [x for x in listdir(dataPath + '/A/') if is_image_file(x)]
        self.listB = [x for x in listdir(dataPath + '/B/') if is_image_file(x)]
        self.listR = [x for x in listdir(dataPath+'/R/') if is_image_file(x)]

        self.dataPath = dataPath
        self.loadSize = loadSize
        self.fineSize = fineSize
        self.flip = flip

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        pathA = os.path.join(self.dataPath + '/A/', self.listA[index])
        pathB = os.path.join(self.dataPath + '/B/', self.listB[index])
        pathR = os.path.join(self.dataPath+'/R/',self.listR[index])

        imgA = default_loader(pathA)  # 256x256
        imgB = default_loader(pathB)
        imgR = default_loader(pathR)

        # # 2. seperate image A and B; Scale; Random Crop; to Tensor
        # w1, h1 = imgA.size
        # w2, h2 = imgB.size
        #
        # tar = 415
        # # real
        # if w1 >= h1:
        #     if h1 > tar or w1 > tar:
        #         imgA = imgA.resize((tar, tar * h1 // w1), Image.BILINEAR)
        #         imgB = imgB.resize((tar, tar * h1 // w1), Image.BILINEAR)
        #         imgR = imgR.resize((tar,tar*h1//w1), Image.BILINEAR)
        #     else:
        #         pass
        # else:
        #     if h1 > tar or w1 > tar:
        #         imgA = imgA.resize((tar * w1 // h1, tar), Image.BILINEAR)
        #         imgB = imgB.resize((tar * w1 // h1, tar), Image.BILINEAR)
        #         imgR = imgR.resize((tar*w1//h1,tar), Image.BILINEAR)
        #
        #     else:
        #         pass

        imgA = ToTensor(imgA)  # 3 x 256 x 256
        imgB = ToTensor(imgB)
        imgR = ToTensor(imgR)

        return imgA, imgB

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.listA)

import torchdata
BaseDataset = torchdata.Dataset

class FusionDataset(BaseDataset):
    def __init__(self, datasets, fusion_ratios=None):
        self.datasets = datasets
        self.size = sum([len(dataset) for dataset in datasets]) - 6732
        self.fusion_ratios = fusion_ratios or [1./len(datasets)] * len(datasets)
        print('[i] using a fusion dataset: %d %s imgs fused with ratio %s' %(self.size, [len(dataset) for dataset in datasets], self.fusion_ratios))

    def reset(self):
        for dataset in self.datasets:
            dataset.reset()

    def __getitem__(self, index):
        #residual = 1
        ratio = self.fusion_ratios
        if random.random() < ratio[0]:
            dataset = self.datasets[0]
        elif random.random() > ratio[0]+ratio[1]:
            dataset = self.datasets[2]
        else:
            dataset = self.datasets[1]
        return dataset[index%len(dataset)]
        '''
        for i, ratio in enumerate(self.fusion_ratios):
            if random.random() < ratio/residual or i == len(self.fusion_ratios) - 1:
                dataset = self.datasets[i]

                return dataset[index%len(dataset)]
            residual -= ratio
        '''
    def __len__(self):
        return self.size