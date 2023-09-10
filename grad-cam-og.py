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

parser = argparse.ArgumentParser(description='Data Preparation for Traffic Sign Project')
parser.add_argument('--model-path',
					default='./checkpoints/model_gtsrb_rn_adv6.pt',
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
	root_dir='/content/data/GTSRB/Final_Test/Images/',
	transform=transform_test
)

#testset = GTSRB_Test(
	#root_dir='/content/data/Images_0_ppm',
	#transform=transform_test
#)

#model-dataset
if not os.path.exists('./grad-cam/2-0'):
	os.makedirs('./grad-cam/2-0')

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

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

		# Grad-CAM visualization for all images in the batch
		max_indices = pred.argmax(dim=1)
		pred_max = pred[range(pred.shape[0]), max_indices]
		pred_max.sum().backward()

		gradients = model[0].get_activations_gradient()
		pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
		activations = model[0].get_activations().detach()

		for i in range(activations.shape[1]):
			activations[:, i, :, :] *= pooled_gradients[i]

		for j in range(activations.shape[0]):  # Loop through all images in the batch
			heatmap = torch.mean(activations[j, :, :, :], dim=0).squeeze()
			heatmap = np.maximum(heatmap.cpu(), 0)
			heatmap /= torch.max(heatmap)

			# Convert to a format suitable for visualization
			heatmap = heatmap.numpy()
			img = data.cpu().numpy()[j].transpose(1, 2, 0)
			img -= np.min(img)
			img /= np.max(img)
			
			heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
			heatmap = np.uint8(255 * heatmap)
			heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
			superimposed_img = heatmap * 0.4 + img

			counter += 1

			cv2.imwrite(f'./grad-cam/2-0/gradcam_img{counter}_label{str(y.cpu().numpy()[j])}.jpg', superimposed_img)

		break
	
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

	features, predictions, targets = rep(model_, device, test_loader)

	#convert to tabular data
	#path = "./tabu_data/"
	#if not os.path.exists(path):
		#os.makedirs(path)
	
	predictions = predictions.reshape(predictions.shape[0], 1)
	targets = targets.reshape(targets.shape[0], 1)

if __name__ == '__main__':
	main()