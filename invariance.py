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
import csv

from GTSRB import GTSRB_Test
from GTSRB_sub import GTSRB_Test_Sub
from feature_extractor import FeatureExtractor

parser = argparse.ArgumentParser(description='Data Preparation for Traffic Sign Project')
parser.add_argument('--model-path',
                    default='./checkpoints/model_gtsrb_rn_adv6.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
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

testset0 = GTSRB_Test(
			    root_dir='/content/data/GTSRB/Final_Test/Images/',
			    transform=transform_test)

test_loader0 = torch.utils.data.DataLoader(testset0, batch_size=args.test_batch_size, shuffle=False, **kwargs)

testset1 = GTSRB_Test(
			    root_dir='/content/data/Images_2_ppm/',
			    transform=transform_test)

test_loader1 = torch.utils.data.DataLoader(testset1, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def inv(backbone, device, testloader0, testloader1):

	backbone.eval()

	inv_total = 0
	num_instance = len(testloader0)

	iter1 = iter(test_loader1)

	for data, target in test_loader0:

		data1, target1 = data.to(device), target.to(device)
		X1 = Variable(data1)

		data, target = next(iter1)
		data2, target2 = data.to(device), target.to(device)
		X2 = Variable(data2)

		x1 = backbone(X1)
		x2 = backbone(X2)

		repr_loss = F.mse_loss(x1, x2)

		inv_total += repr_loss

	inv_total = inv_total.cpu().detach().numpy()

	return inv_total/num_instance

def main():

	#initialize model
	model = resnet101()
	model.fc = nn.Linear(2048, 43)
	model.load_state_dict(torch.load(args.model_path))
	model = model.to(device)

	backbone = FeatureExtractor(model)
	backbone = backbone.to(device)

	avg_inv = inv(backbone, device, test_loader0, test_loader1)

	#file = open('inv.csv','w')
	file = open('inv.csv','a')
	writer = csv.writer(file)
	#writer.writerow(['inv'])
	writer.writerow([avg_inv])

if __name__ == '__main__':
	main()