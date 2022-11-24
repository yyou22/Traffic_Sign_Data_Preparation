from __future__ import print_function
from sklearn.manifold import TSNE
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.models import resnet101, ResNet101_Weights
import numpy as np
from torch.utils.data import Subset, DataLoader

from GTSRB import GTSRB_Test
from GTSRB_sub import GTSRB_Test_Sub
from feature_extractor import FeatureExtractor

parser = argparse.ArgumentParser(description='Data Preparation for Traffic Sign Project')
parser.add_argument('--model-path-cur',
                    default='./checkpoints/model_gtsrb_rn_adv6.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--mode', default=1,
                    help='define whcih subcanvas')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.003,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')

args = parser.parse_args()

torch.manual_seed(0)

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
])

testset = GTSRB_Test(
    root_dir='/content/data/GTSRB/Final_Test/Images/',
    transform=transform_test
)

n_class = 43

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def attack(model, device, test_loader):

    model.eval()

    attack_imgs = []

    for data,target in test_loader:
        data, target = data.to(device), target.to(device)
        X, y = Variable(data), Variable(target)

        attack_imgs.extend(X.cpu().detach().numpy())

    #convert to numpy arrays
    attack_imgs = np.array(attack_imgs)

    return attack_imgs

def main():

    #initialize model attacked
    model = resnet101()
    model.fc = nn.Linear(2048, 43)
    model.load_state_dict(torch.load(args.model_path_cur))
    model = model.to(device)

    #convert to tabular data
    path = "./original_data/"
    if not os.path.exists(path):
        os.makedirs(path)

    attack_imgs = attack(model, device, test_loader)
    np.save(path + "X", attack_imgs)
    
if __name__ == '__main__':
    main()
