import argparse
# Arguments
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')

parser.add_argument('--datadir_syn', default='../data/PLNet', help='path to syn training images')
parser.add_argument('--real20Path', default='../data/Real110/train/', help='path to real training images')
parser.add_argument('--wildPath', default='../data/SIRR/Wild/', help='path to test images')
parser.add_argument('--testPath', default='../data/Real20/test/', help='path to test images')
parser.add_argument('--dataPath', default='../data/SIRR/Solid/', help='path to test images')
parser.add_argument('--reflectionPath', default='../data/rimage/', help='path to training images')
parser.add_argument('--loadSize', type=int, default=256, help='scale image to this size')
parser.add_argument('--fineSize', type=int, default=224, help='random crop image to this size')
parser.add_argument('--flip', type=int, default=1, help='1 for flipping image randomly, 0 for not')
parser.add_argument('--outf', default='checkpoints/', help='folder to output images and model checkpoints')
parser.add_argument('--test_model', default='checkpoints/pretrained_netG.pth', help='path to model for test.')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.0002, type=float, help='initial learning rate')
parser.add_argument('--bs', default=1, type=int, help='batch size')
parser.add_argument('--model', default='', help='path to pre-trained')
parser.add_argument('--netD', default='', help='path to pre-trained')
parser.add_argument('--local_rank', default='', help='path to pre-trained')
parser.add_argument('--e', default=0, type=int, help='epoch')

args = parser.parse_args()