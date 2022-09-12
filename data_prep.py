from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet101, ResNet101_Weights

from GTSRB import GTSRB_Test
from models.wideresnet import *
from models.resnet import *
from trades import trades_loss

parser = argparse.ArgumentParser(description='Data Preparation for Traffic Sign Project')
parser.add_argument('--model-path',
                    default='./checkpoints/model_gtsrb_wrn.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 200)')

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

def rep(model, device, test_loader):
	model.eval()

	for data,target in test_loader:
		data, target = data.to(device), target.to(device)
		X, y = Variable(data), Variable(target)

def main():
	#initialize model
	model = resnet101()
	model.fc = nn.Linear(2048, 43)
	model = model.to(device)

	model.load_state_dict(torch.load(args.model_path))
	backbone = model.features
	backbone = backbone.to(device)

	rep(backbone, device, test_loader)

if __name__ == '__main__':
	main()