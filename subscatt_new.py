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
parser.add_argument('--model-num', type=int, default=1, help='which model checkpoint to use')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--mode', default=0,
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

def rep2(model, device, test_loader0, test_loader1):

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
	#original indices
	og_dices = []

	idx = 0

	iter1 = iter(test_loader1)

	for data, target, og_idx in test_loader0:

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
		og_dices.extend(og_idx)

		data, target, og_idx = next(iter1)
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
		og_dices.extend(og_idx)

	#convert to numpy arrays
	targets = np.array(targets)
	predictions = np.array(predictions)
	features = np.array(features)
	adv_class = np.array(adv_class)
	match_idx = np.array(match_idx)
	og_dices = np.array(og_dices)

	return features, predictions, targets, adv_class, match_idx, og_dices

def dimen_reduc(features, num_data):
	
	feature_t = TSNE_(features)

	tx, ty = feature_t[:, 0].reshape(num_data*2, 1), feature_t[:, 1].reshape(num_data*2, 1)
	tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
	ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

	return tx, ty

def main():

	if args.model_num == 0:
		model_path = './checkpoints/model_gtsrb_rn_nat.pt'
	elif args.model_num == 1:
		model_path = './checkpoints/model_gtsrb_rn_adv1.pt'
	else:
		model_path = './checkpoints/model_gtsrb_rn_adv6.pt'

	#initialize model 1 (benchmark model)
	model1 = resnet101()
	model1.fc = nn.Linear(2048, 43)
	model1.load_state_dict(torch.load(model_path))
	model1 = model1.to(device)

	backbone1 = FeatureExtractor(model1)
	backbone1 = backbone1.to(device)
	fc1 = model1.fc
	model_1 = nn.Sequential(backbone1, fc1)

	for i in range(0, n_class):

		testset0 = GTSRB_Test_Sub(
				    root_dir='/content/data/GTSRB/Final_Test/Images/',
				    class_ = i,
				    transform=transform_test)

		test_loader0 = torch.utils.data.DataLoader(testset0, batch_size=args.test_batch_size, shuffle=False, **kwargs)

		testset2 = GTSRB_Test_Sub(
				    root_dir='/content/data/Images_' + str(args.model_num) + '_ppm/',
				    class_ = i,
				    transform=transform_test)

		test_loader2 = torch.utils.data.DataLoader(testset2, batch_size=args.test_batch_size, shuffle=False, **kwargs)

		features, predictions, targets, adv_class, match_idx, og_dices = rep2(model_1, device, test_loader0, test_loader2)

		tx, ty = dimen_reduc(features, len(testset0))

		#convert to tabular data
		path = "./subscatt_data" + str(args.model_num) + "/" + str(i) + "/"
		if not os.path.exists(path):
			os.makedirs(path)
		
		predictions = predictions.reshape(predictions.shape[0], 1)
		targets = targets.reshape(targets.shape[0], 1)
		adv_class = adv_class.reshape(adv_class.shape[0], 1)
		match_idx = match_idx.reshape(match_idx.shape[0], 1)
		og_dices = og_dices.reshape(og_dices.shape[0], 1)

		print(og_dices)

		result = np.concatenate((tx, ty, predictions, targets, adv_class, match_idx, og_dices), axis=1)
		type_ = ['%.5f'] * 2 + ['%d'] * 5
		
		if args.mode == 0:
			np.savetxt(path + "data_label.csv", result, header="xpos,ypos,pred,target,cur_model,match_idx,og_idx", comments='', delimiter=',', fmt=type_)
		elif args.mode == 1:
			np.savetxt(path + "data_pred.csv", result, header="xpos,ypos,pred,target,cur_model,match_idx,og_idx", comments='', delimiter=',', fmt=type_)

if __name__ == '__main__':
	main()



