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

from GTSRB import GTSRB_Test
from feature_extractor import FeatureExtractor

parser = argparse.ArgumentParser(description='Data Preparation for Traffic Sign Project')
parser.add_argument('--model-path',
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

def rep(model, device, test_loader):
	model.eval()

	#feature list
	features = []
	#prediction list
	predictions = []
	#target list
	targets = []

	accu = 0
	total = 0

	for data,target in test_loader:
		data, target = data.to(device), target.to(device)
		X, y = Variable(data), Variable(target)

		feat = model[0](X).reshape(X.shape[0], 2048)

		#pass thru linear layer to obtain prediction result
		pred = model[1](feat)

		accu += (pred.data.max(1)[1] == y.data).float().sum()
		total += X.shape[0]

		targets.extend(y.data.cpu().detach().numpy())
		predictions.extend(pred.data.max(1)[1].cpu().detach().numpy())
		#push representation to the list
		features.extend(feat.cpu().detach().numpy())
	
	#convert to numpy arrays
	targets = np.array(targets)
	predictions = np.array(predictions)
	features = np.array(features)

	print("Prediction Accuracy:" + str(accu/total))

	return features, predictions, targets

def dimen_reduc(features):
	
	feature_t = TSNE_(features)

	tx, ty = feature_t[:, 0].reshape(12630, 1), feature_t[:, 1].reshape(12630, 1)
	tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
	ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

	return tx, ty

def main():
	#initialize model
	model = resnet101()
	model.fc = nn.Linear(2048, 43)
	model.load_state_dict(torch.load(args.model_path))
	model = model.to(device)

	#print(model.state_dict().keys())

	backbone = FeatureExtractor(model)
	backbone = backbone.to(device)

	fc = model.fc

	model_ = nn.Sequential(backbone, fc)

	features, predictions, targets = rep(model_, device, test_loader)

	tx, ty = dimen_reduc(features)

	#convert to tabular data
	path = "./tabu_data/"
	if not os.path.exists(path):
		os.makedirs(path)
	
	predictions = predictions.reshape(predictions.shape[0], 1)
	targets = targets.reshape(targets.shape[0], 1)

	result = np.concatenate((tx, ty, predictions, targets), axis=1)
	type_ = ['%.5f'] * 2 + ['%d'] * 2
	np.savetxt(path + "data_1_adv.csv", result, header="xpos,ypos,pred,target", comments='', delimiter=',', fmt=type_)

if __name__ == '__main__':
	main()