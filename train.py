# Author: Fengx

import time
import torch
import torch.nn as nn
from model import Model, Discriminator_VGG
from data import  FusionDataset, REAL_DATASET, TEST_REAL_DATASET
from utils import AverageMeter

import numpy as np
import os
import assess
import datasyn.reflect_dataset as datasets

from PIL import Image
from vgg import Vgg19
from torch.utils.data.distributed import DistributedSampler

from util.loss import gram_matrix, ExclusionLoss, total_variation_loss, VGGLoss
from util.interface import lr_adjust, weights_init_kaiming, tensor2im
from option import args
class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, *args, **kwargs)
        self.shuffle = shuffle

    def reset(self):
        if self.shuffle:
            print('Reset Dataset...')
            self.dataset.reset()

# -------------------------Main------------------------------------
EPS = 1e-12
def main():

    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # Create model
    model = Model()
    netD = Discriminator_VGG()
    model = model.to(device)
    netD = netD.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 5) 封装
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],output_device=local_rank)
        netD = torch.nn.parallel.DistributedDataParallel(netD,device_ids=[local_rank],output_device=local_rank)

    if(args.model != ''):
        print('Warning! Loading pre-trained weights.')
        model.load_state_dict(torch.load(args.model))
        netD.load_state_dict(torch.load(args.netD))

    print('Model created.')
    vgg_loss = VGGLoss().cuda()
    # Training parameters
    optimizer = torch.optim.Adam( model.parameters(), args.lr )
    optimizer_D = torch.optim.Adam( netD.parameters(), 0.0002 )

    batch_size = args.bs
    prefix = 'densenet_' + str(batch_size)

    # Syn datasets
    dataset = datasets.CEILDataset(args.datadir_syn, fns=None,size=None, enable_transforms=True,low_sigma=2, high_sigma=5,low_gamma=1.3, high_gamma=1.3)

    # Load  real  and test data
    real20_dataset = REAL_DATASET(args.real20Path,args.loadSize,args.fineSize,args.flip)  # real20
    train_dataset = FusionDataset([dataset, real20_dataset],[0.7, 0.3])

    test_dataset = TEST_REAL_DATASET(args.testPath,256,256,0)
    
    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False,num_workers=0,sampler=DistributedSampler(train_dataset))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=1,shuffle=False,num_workers=0)
    # Loss
    l1_criterion = nn.L1Loss().cuda()

    vgg = Vgg19().cuda()

    print('-------------------------------------------------')
    print('||||||   Attention!!! data training now!   ||||||') 

    # Start training data pairs...
    for epoch in range(args.epochs):
        if not os.path.exists('testing/'+str(epoch)) and local_rank==0:
            os.mkdir('testing/'+str(epoch))
        if not os.path.exists('real110_testing/'+str(epoch)) and local_rank==0:
            os.mkdir('real110_testing/'+str(epoch))

    for epoch in range(args.epochs):
        if args.e :
            if epoch <= args.e:
                continue

        batch_time = AverageMeter()
        l1_losses = AverageMeter()
        p_losses = AverageMeter()
        g_losses = AverageMeter()
        all_losses = AverageMeter()
        d_losses = AverageMeter()
        tv_losses = AverageMeter()
        style_losses = AverageMeter()
        head_loss0es = AverageMeter()
        head_loss1es = AverageMeter()
        N = len(train_loader)

        # Switch to train mode
        model.train()
        netD.train()

        end = time.time()
        lr_adjust(optimizer, epoch, args.lr)
        for i, sample_batched in enumerate(train_loader):
            if len(sample_batched)==3:
                isSyn = True
                imgA = sample_batched[0]
                imgB = sample_batched[1]
                imgC = sample_batched[2]
            else:
                isSyn = False
                imgA = sample_batched[0]
                imgB = sample_batched[1]

            # Prepare sample and target
            input_a = torch.autograd.Variable(imgA.cuda(),requires_grad=True).cuda()
            input_b = torch.autograd.Variable(imgB.cuda(),requires_grad=True).cuda()

            if isSyn:
                input_r = torch.autograd.Variable(imgC.cuda(),requires_grad=True).cuda()
                if torch.max(input_r)<0.15 or torch.max(input_b)<0.15: 
                    print('Invalid input!~~~~')
                    continue
                if torch.max(input_a)<0.1:
                    print('Invalid input!~~~~')
                    continue
            lx1, lx2,feature2,feature1, output_r,output_t = model(input_a)

            head_loss0 = l1_criterion(lx1, lx1)
            head_loss1 = l1_criterion(lx2, lx2)
            if isSyn:
                l_AB = l1_criterion(imgA, imgB)
                l_AC = l1_criterion(imgA, imgC)
                head_loss0 = l_AB
                head_loss1 = l_AC
            style_loss = l1_criterion(feature1+feature2, feature1+feature2)
            if isSyn:
                # Feature Contrast Loss
                feature_b = vgg(input_b*255.0, [2])
                feature_r = vgg(input_r*255.0, [2])

                gram_style1 = [gram_matrix(feature1[:,i:i+1,:,:]) for i in range(64)]
                gram_style2 = [gram_matrix(feature2[:,i:i+1,:,:]) for i in range(64)]

                gram_b = [gram_matrix(fb).cuda() for fb in feature_b]
                gram_r = [gram_matrix(fr).cuda() for fr in feature_r]

                for gm_1, gm_2, gm_b, gm_r in zip(gram_style1, gram_style2,gram_b,gram_r):
                    style_loss += l1_criterion(l1_criterion(gm_b,gm_r),l1_criterion(gm_1, gm_2))

            l1_r_loss = l1_criterion(output_r, output_r)
            if isSyn:
                l1_r_loss = l1_criterion(output_r, input_r)
            # Perceptual Loss
            p_loss_t = vgg_loss(output_t, input_b)
            p_loss_r = 0
            if isSyn:
                p_loss_r = vgg_loss(output_r, input_r)
            p_loss = p_loss_r + p_loss_t
            loss = p_loss*0.2 + l1_r_loss
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # TV loss   
            tv_loss = total_variation_loss(output_t)
            # Adversatial
            real = netD(input_a,input_b)
            fake = netD(input_a,output_t.detach())

            d_loss = (torch.mean(-(torch.log(real + EPS) + torch.log(1 - fake + EPS)))) * 0.5

            # update D
            d_losses.update(round(d_loss.item(),4))
            optimizer_D.zero_grad()
            for p in netD.parameters():
                p.requires_grad = True
            d_loss.backward(retain_graph=True)
            optimizer_D.step()
            # update G
            optimizer.zero_grad()
            fake = netD(input_a,output_t)
            g_loss = torch.mean(-torch.log(fake + EPS))
            record_g_loss = g_loss + loss
            G_loss = g_loss + 100*loss + 5e-2*tv_loss + 100*style_loss + head_loss0 + head_loss1
            for p in netD.parameters():
                p.requires_grad = False
            G_loss.backward()
            optimizer.step()
            if isSyn:
                l1_losses.update(round(l1_r_loss.item(),4))
                style_losses.update(round(style_loss.item(),8))
                head_loss0es.update(round(head_loss0.item(), 8))
                head_loss1es.update(round(head_loss1.item(), 8))
            
            p_losses.update(round(p_loss.item(),4))
            g_losses.update(round(g_loss.item(),4))
            all_losses.update(round(record_g_loss.item(),4))
            tv_losses.update(round(tv_loss.item(),4))
            # Print to console
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                  'TV Loss {TV_Loss.val:.4f} ({TV_Loss.avg:.4f})\t'
                  'Perceptual Loss {Perceptual_loss.val:.4f} ({Perceptual_loss.avg:.4f})\t'
                  'G ALL Loss {G_ALL_Loss.val:.4f} ({G_ALL_Loss.avg:.4f})\t'
                  'G Loss {G_loss.val:.4f} ({G_loss.avg:.4f})\t'
                  'D Loss {D_Loss.val:.4f} ({D_Loss.avg:.4f})\t'
                  'Style Loss {style_loss.val:.8f} ({style_loss.avg:.8f})\t'
                  'head_loss0 {head_loss0.val:.8f} ({head_loss0.avg:.8f})\t'
                  'head_loss1 {head_loss1.val:.8f} ({head_loss1.avg:.8f})\t'
                  .format(epoch, i+1, N,
                    Perceptual_loss = p_losses, G_loss=g_losses,
                    G_ALL_Loss=all_losses, D_Loss=d_losses,batch_time=batch_time,
                    TV_Loss=tv_losses,style_loss=style_losses, head_loss0=head_loss0es, head_loss1=head_loss1es))

            if i%100 == 0:
                Image.fromarray(tensor2im(output_t).astype(np.uint8)).save('testing/'+str(epoch)+'/'+str(i)+str(epoch)+str(args.local_rank)+'output_t.png')
                Image.fromarray(tensor2im(input_a).astype(np.uint8)).save('testing/'+str(epoch)+'/'+str(i)+str(epoch)+str(args.local_rank)+'input_a.png')
                Image.fromarray(tensor2im(input_b).astype(np.uint8)).save('testing/'+str(epoch)+'/'+str(i)+str(epoch)+str(args.local_rank)+'input_t.png')
                Image.fromarray(tensor2im(output_r).astype(np.uint8)).save('testing/'+str(epoch)+'/'+str(i)+str(epoch)+str(args.local_rank)+'output_r.png')
        # test
        if local_rank == 0:
            model.eval()
            with torch.no_grad():
                ssim, psnr, num = 0, 0, 20
                for i, sample_batched in enumerate(test_loader):
                    imgA = sample_batched[0]
                    imgB = sample_batched[1]
                    lx1, lx2, feature2, feature1, output_r, output_t = model(imgA)
                    t_output = tensor2im(output_t)
                    t_imgB = tensor2im(imgB)
                    Image.fromarray(tensor2im(imgA).astype(np.uint8)).save(
                        'real110_testing/' + str(epoch) + '/' + str(i) + str(epoch) + str(args.local_rank) + 'input_a.png')
                    Image.fromarray(t_imgB.astype(np.uint8)).save(
                        'real110_testing/' + str(epoch) + '/' + str(i) + str(epoch) + str(args.local_rank) + 'input_t.png')
                    Image.fromarray(t_output.astype(np.uint8)).save(
                        'real110_testing/' + str(epoch) + '/' + str(i) + str(epoch) + str(args.local_rank) + 'output_t.png')
                    assess_dict = assess.quality_assess(t_output, t_imgB)
                    ssim += assess_dict['SSIM']
                    psnr += assess_dict['PSNR']
                print('Epoch:{0}\tssim:{1}\tpsnr:{2}'.format(epoch, ssim/num, psnr/num))
        train_loader.reset()
        if local_rank == 0:
            torch.save(model.state_dict(), args.outf+'/'+str(epoch)+'_PL_netG.pth')
            torch.save(netD.state_dict(), args.outf+'/'+str(epoch)+'_PL_netD.pth')
if __name__ == '__main__':
    main()
