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
parser.add_argument('--model-path-bm',
                    default='./checkpoints/model_gtsrb_rn_nat.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--model-path-cur',
                    default='./checkpoints/model_gtsrb_rn_adv2.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--mode', default=1,
                    help='define whcih subcanvas')

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

#testset = GTSRB_Test(
    #root_dir='/content/data/GTSRB-Test/Final_Test/Images/',
    #transform=transform_test
#)

n_class = 43

#test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def TSNE_(data):

	tsne = TSNE(n_components=2)
	data = tsne.fit_transform(data)

	return data

def rep2(model, modelog, device, test_loader0, test_loader1):

	model.eval()

	#feature list
	features = []
	#prediction list
	predictions = []
	#target list
	targets = []
	#adv class
	adv_class = []
	#match index
	match_idx = []
	idx = 0

	iter1 = iter(test_loader1)

	for data, target in test_loader0:

		data, target = data.to(device), target.to(device)
		X, y = Variable(data), Variable(target)

		lc = [n for n in range(idx, idx + X.shape[0])]
		idx = idx + X.shape[0]

		feat1 = model[0](X).reshape(X.shape[0], 2048)

		#pass thru linear layer to obtain prediction result
		pred1 = model[1](feat1)

		targets.extend(y.data.cpu().detach().numpy())
		predictions.extend(pred1.data.max(1)[1].cpu().detach().numpy())
		#push representation to the list
		features.extend(feat1.cpu().detach().numpy())
		adv_class.extend([0] * X.shape[0])
		match_idx.extend(lc)

		data, target = next(iter1)
		data, target = data.to(device), target.to(device)
		X, y = Variable(data), Variable(target)

		feat2 = model[0](X).reshape(X.shape[0], 2048)

		#pass thru linear layer to obtain prediction result
		pred2 = model[1](feat2)

		targets.extend(y.data.cpu().detach().numpy())
		predictions.extend(pred2.data.max(1)[1].cpu().detach().numpy())
		#push representation to the list
		features.extend(feat2.cpu().detach().numpy())
		adv_class.extend([1] * X.shape[0])
		match_idx.extend(lc)

	#convert to numpy arrays
	targets = np.array(targets)
	predictions = np.array(predictions)
	features = np.array(features)
	adv_class = np.array(adv_class)
	match_idx = np.array(match_idx)

	return features, predictions, targets, adv_class, match_idx 


def rep(model1, model2, model1og, model2og, device, test_loader0, test_loader1, test_loader2):

	model1.eval()
	model2.eval()

	#feature list
	features = []
	#prediction list
	predictions = []
	#target list
	targets = []
	#adv class
	adv_class = []
	#match index
	match_idx = []
	idx = 0

	iter1 = iter(test_loader1)
	iter2 = iter(test_loader2)

	for data, target in test_loader0:

		data, target = data.to(device), target.to(device)
		X, y = Variable(data), Variable(target)

		lc = [n for n in range(idx, idx + X.shape[0])]
		idx = idx + X.shape[0]

		if args.mode == 2:
			data1, target1 = next(iter1)
			data1, target1 = data1.to(device), target1.to(device)
			X1, y1 = Variable(data1), Variable(target1)
			X = X1

		feat1 = model1[0](X).reshape(X.shape[0], 2048)

		#pass thru linear layer to obtain prediction result
		pred1 = model1[1](feat1)

		targets.extend(y.data.cpu().detach().numpy())
		predictions.extend(pred1.data.max(1)[1].cpu().detach().numpy())
		#push representation to the list
		features.extend(feat1.cpu().detach().numpy())
		adv_class.extend([0] * X.shape[0])
		match_idx.extend(lc)

		if args.mode == 2:
			data2, target2 = next(iter2)
			data2, target2 = data2.to(device), target2.to(device)
			X2, y2 = Variable(data2), Variable(target2)
			X = X2

		feat2 = model2[0](X).reshape(X.shape[0], 2048)

		#pass thru linear layer to obtain prediction result
		pred2 = model2[1](feat2)

		targets.extend(y.data.cpu().detach().numpy())
		predictions.extend(pred2.data.max(1)[1].cpu().detach().numpy())
		#push representation to the list
		features.extend(feat2.cpu().detach().numpy())
		adv_class.extend([1] * X.shape[0])
		match_idx.extend(lc)

	#convert to numpy arrays
	targets = np.array(targets)
	predictions = np.array(predictions)
	features = np.array(features)
	adv_class = np.array(adv_class)
	match_idx = np.array(match_idx)

	return features, predictions, targets, adv_class, match_idx 

def dimen_reduc(features, num_data):
	
	feature_t = TSNE_(features)

	tx, ty = feature_t[:, 0].reshape(num_data*2, 1), feature_t[:, 1].reshape(num_data*2, 1)
	tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
	ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

	return tx, ty

def main():

	#initialize model 1 (benchmark model)
	model1 = resnet101()
	model1.fc = nn.Linear(2048, 43)
	model1.load_state_dict(torch.load(args.model_path_bm))
	model1 = model1.to(device)

	backbone1 = FeatureExtractor(model1)
	backbone1 = backbone1.to(device)
	fc1 = model1.fc
	model_1 = nn.Sequential(backbone1, fc1)

	#initilaize model 2 (current model)
	model2 = resnet101()
	model2.fc = nn.Linear(2048, 43)
	model2.load_state_dict(torch.load(args.model_path_cur))
	#model2.load_state_dict(torch.load(args.model_path_bm))
	model2 = model2.to(device)

	backbone2 = FeatureExtractor(model2)
	backbone2 = backbone2.to(device)
	fc2 = model2.fc
	model_2 = nn.Sequential(backbone2, fc2)

	for i in range(0, n_class):

		testset0 = GTSRB_Test_Sub(
				    root_dir='/content/data/GTSRB/Final_Test/Images/',
				    class_ = i,
				    transform=transform_test)

		test_loader0 = torch.utils.data.DataLoader(testset0, batch_size=args.test_batch_size, shuffle=False, **kwargs)

		testset1 = GTSRB_Test_Sub(
				    root_dir='/content/data/Images_0_ppm/',
				    class_ = i,
				    transform=transform_test)

		test_loader1 = torch.utils.data.DataLoader(testset1, batch_size=args.test_batch_size, shuffle=False, **kwargs)

		testset2 = GTSRB_Test_Sub(
				    root_dir='/content/data/Images_2_ppm/',
				    class_ = i,
				    transform=transform_test)

		test_loader2 = torch.utils.data.DataLoader(testset2, batch_size=args.test_batch_size, shuffle=False, **kwargs)

		if args.mode == 1 or args.mode == 2:
			features, predictions, targets, adv_class, match_idx = rep(model_1, model_2, model1, model2, device, test_loader0, test_loader1, test_loader2)
		elif args.mode == 3:
			features, predictions, targets, adv_class, match_idx = rep2(model_1, model1, device, test_loader0, test_loader1)
		elif args.mode == 4:
			features, predictions, targets, adv_class, match_idx = rep2(model_2, model2, device, test_loader0, test_loader2)

		tx, ty = dimen_reduc(features, len(testset0))

		#convert to tabular data
		path = "./tabu_data2/" + str(i) + "/"
		if not os.path.exists(path):
			os.makedirs(path)
		
		predictions = predictions.reshape(predictions.shape[0], 1)
		targets = targets.reshape(targets.shape[0], 1)
		adv_class = adv_class.reshape(adv_class.shape[0], 1)
		match_idx = match_idx.reshape(match_idx.shape[0], 1)

		result = np.concatenate((tx, ty, predictions, targets, adv_class, match_idx), axis=1)
		type_ = ['%.5f'] * 2 + ['%d'] * 4
		
		if args.mode == 1:
			np.savetxt(path + "data_bn_an.csv", result, header="xpos,ypos,pred,target,cur_model,match_idx", comments='', delimiter=',', fmt=type_)
		elif args.mode == 2:
			np.savetxt(path + "data_ba_aa.csv", result, header="xpos,ypos,pred,target,cur_model,match_idx", comments='', delimiter=',', fmt=type_)
		elif args.mode == 3:
			np.savetxt(path + "data_bn_ba.csv", result, header="xpos,ypos,pred,target,cur_model,match_idx", comments='', delimiter=',', fmt=type_)
		elif args.mode == 4:
			np.savetxt(path + "data_an_aa.csv", result, header="xpos,ypos,pred,target,cur_model,match_idx", comments='', delimiter=',', fmt=type_)

if __name__ == '__main__':
	main()



