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
from feature_extractor import FeatureExtractor

parser = argparse.ArgumentParser(description='Data Preparation for Traffic Sign Project')
parser.add_argument('--model-path-bm',
                    default='./checkpoints/model_gtsrb_rn_nat.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--model-path-cur',
                    default='./checkpoints/model_gtsrb_rn_adv1.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()

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
    root_dir='/content/data/GTSRB-Test/Final_Test/Images/',
    transform=transform_test
)

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def TSNE_(data):

	tsne = TSNE(n_components=2)
	data = tsne.fit_transform(data)

	return data

def main():

	#initialize model 1 (benchmark model)
	model1 = resnet101()
	model1.fc = nn.Linear(2048, 43)
	model1.load_state_dict(torch.load(args.model_path_bm))
	model1 = model1.to(device)

	backbone1 = FeatureExtractor(model1)
	backbone1 = backbone1.to(device)
	fc1 = model1.fc

	#initilaize model 2 (current model)
	model2 = resnet101()
	model2.fc = nn.Linear(2048, 43)
	model2.load_state_dict(torch.load(args.model_path_cur))
	model2 = model2.to(device)

	backbone2 = FeatureExtractor(model2)
	backbone2 = backbone2.to(device)
	fc2 = model2.fc

	testset1 = GTSRB_Test_Sub(
			    root_dir='/content/data/GTSRB-Test/Final_Test/Images/',
			    class = 0,
			    transform=transform_test)


if __name__ == '__main__':
	main()



