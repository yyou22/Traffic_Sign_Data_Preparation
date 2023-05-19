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
#parser.add_argument('--model-path',
                    #default='./checkpoints/model_gtsrb_rn_adv6.pt',
                    #help='model for white-box attack evaluation')
parser.add_argument('--model-num', type=int, default=0, help='which model checkpoint to use')
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

testset_nat = GTSRB_Test(
    root_dir='/content/data/GTSRB/Final_Test/Images/',
    transform=transform_test
)

test_loader_nat = torch.utils.data.DataLoader(testset_nat, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def TSNE_(data):

	tsne = TSNE(n_components=2)
	data = tsne.fit_transform(data)

	return data

def rep(model, device, test_loader_nat, test_loader_adv):
	model.eval()

	#feature list
	features = []
	#prediction list
	predictions = []
	#target list
	targets = []
	#show nat or adv
	type_ = []

	nat_accu = 0
	nat_total = 0

	for data,target in test_loader_nat:
		data, target = data.to(device), target.to(device)
		X, y = Variable(data), Variable(target)

		feat = model[0](X).reshape(X.shape[0], 2048)

		#pass thru linear layer to obtain prediction result
		pred = model[1](feat)

		nat_accu += (pred.data.max(1)[1] == y.data).float().sum()
		nat_total += X.shape[0]

		targets.extend(y.data.cpu().detach().numpy())
		predictions.extend(pred.data.max(1)[1].cpu().detach().numpy())
		#push representation to the list
		features.extend(feat.cpu().detach().numpy())

	type_.extend([0] * nat_total)

	print("Natural Prediction Accuracy:" + str(nat_accu/nat_total))

	adv_accu = 0
	adv_total = 0

	for data,target in test_loader_adv:
		data, target = data.to(device), target.to(device)
		X, y = Variable(data), Variable(target)

		feat = model[0](X).reshape(X.shape[0], 2048)

		#pass thru linear layer to obtain prediction result
		pred = model[1](feat)

		adv_accu += (pred.data.max(1)[1] == y.data).float().sum()
		adv_total += X.shape[0]

		targets.extend(y.data.cpu().detach().numpy())
		predictions.extend(pred.data.max(1)[1].cpu().detach().numpy())
		#push representation to the list
		features.extend(feat.cpu().detach().numpy())

	type_.extend([1] * adv_total)

	print("Adversarial Prediction Accuracy:" + str(adv_accu/adv_total))
	
	#convert to numpy arrays
	targets = np.array(targets)
	predictions = np.array(predictions)
	features = np.array(features)
	type_ = np.array(type_)

	return features, predictions, targets, type_

def dimen_reduc(features):
	
	feature_t = TSNE_(features)

	tx, ty = feature_t[:, 0].reshape(12630*2, 1), feature_t[:, 1].reshape(12630*2, 1)
	tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
	ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

	return tx, ty

def main():

	testset_adv = GTSRB_Test(
		root_dir='/content/data/Images_' + str(args.model_num) + '_ppm',
		transform=transform_test
	)

	test_loader_adv = torch.utils.data.DataLoader(testset_adv, batch_size=args.test_batch_size, shuffle=False, **kwargs)

	#initialize model
	model = resnet101()
	model.fc = nn.Linear(2048, 43)

	if args.model_num == 0:
		model_path = './checkpoints/model_gtsrb_rn_nat.pt'
	elif args.model_num == 1:
		model_path = './checkpoints/model_gtsrb_rn_adv1.pt'
	else:
		model_path = './checkpoints/model_gtsrb_rn_adv6.pt'

	model.load_state_dict(torch.load(model_path))
	model = model.to(device)

	#print(model.state_dict().keys())

	backbone = FeatureExtractor(model)
	backbone = backbone.to(device)

	fc = model.fc

	model_ = nn.Sequential(backbone, fc)

	features, predictions, targets, type_ = rep(model_, device, test_loader_nat, test_loader_adv)

	tx, ty = dimen_reduc(features)

	#convert to tabular data
	path = "./tabu_data/"
	if not os.path.exists(path):
		os.makedirs(path)
	
	predictions = predictions.reshape(predictions.shape[0], 1)
	targets = targets.reshape(targets.shape[0], 1)
	type_ = type_.reshape(type_.shape[0], 1)

	result = np.concatenate((tx, ty, predictions, targets, type_), axis=1)
	type_ = ['%.5f'] * 2 + ['%d'] * 3
	np.savetxt(path + "data_" + args.model_num + "_all.csv", result, header="xpos,ypos,pred,target,type", comments='', delimiter=',', fmt=type_)

if __name__ == '__main__':
	main()