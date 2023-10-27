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
from torchvision.models import resnet101# ResNet101_Weights
#from contrastive import CPCA
import numpy as np

from GTSRB import GTSRB_Test
from feature_extractor import FeatureExtractor

import cv2
from torch.autograd import Function

from pytorch_grad_cam import GradCAM, HiResCAM, EigenGradCAM, ScoreCAM, GradCAMPlusPlus, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


parser = argparse.ArgumentParser(description='Data Preparation for Traffic Sign Project')
parser.add_argument('--model-path',
					default='./checkpoints/model_gtsrb_rn_adv6.pt',
					help='model for white-box attack evaluation') #nat, adv1, adv6
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
	root_dir='/content/data/GTSRB/Final_Test/Images/',
	transform=transform_test
)

#testset = GTSRB_Test(
	#root_dir='/content/data/Images_2_ppm',
	#transform=transform_test
#)


#model-dataset
if not os.path.exists('./grad-cam_/2-0'):
	os.makedirs('./grad-cam_/2-0')

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def rep(model, device, test_loader, cam):
	model.eval()

	#feature list
	features = []
	#prediction list
	predictions = []
	#target list
	targets = []

	accu = 0
	total = 0

	counter = 0

	for batch_idx, (data, target) in enumerate(test_loader):
		data, target = data.to(device), target.to(device)
		X, y = Variable(data, requires_grad=True), Variable(target)  # Added requires_grad=True

		feat = model[0](X).reshape(X.shape[0], 2048)

		#pass thru linear layer to obtain prediction result
		pred = model[1](feat)

		accu += (pred.data.max(1)[1] == y.data).float().sum()
		total += X.shape[0]

		targets.extend(y.data.cpu().detach().numpy())
		predictions.extend(pred.data.max(1)[1].cpu().detach().numpy())
		#push representation to the list
		features.extend(feat.cpu().detach().numpy())

		# New Grad-CAM code starts here
		#cam_targets = pred.data.max(1)[1].cpu().numpy()  # Get the class indices
		#grayscale_cam = cam(input_tensor=X, targets=None, aug_smooth=True, eigen_smooth=True)
		grayscale_cam = cam(input_tensor=X, targets=None)

		for j in range(grayscale_cam.shape[0]):
			visualization = show_cam_on_image(data.cpu().numpy()[j].transpose(1, 2, 0), grayscale_cam[j, :])

			cv2.imwrite(f'./grad-cam_/2-0/gradcam_img{counter}.jpg', visualization)
			counter += 1

		#break
	
	#convert to numpy arrays
	targets = np.array(targets)
	predictions = np.array(predictions)
	features = np.array(features)

	print("Prediction Accuracy:" + str(accu/total))

	return features, predictions, targets
	#return predictions, targets

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

	# Initialize Grad-CAM
	target_layers = [model.layer4[-1]]
	cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)

	features, predictions, targets = rep(model_, device, test_loader, cam)

	#convert to tabular data
	#path = "./tabu_data/"
	#if not os.path.exists(path):
		#os.makedirs(path)
	
	predictions = predictions.reshape(predictions.shape[0], 1)
	targets = targets.reshape(targets.shape[0], 1)

if __name__ == '__main__':
	main()