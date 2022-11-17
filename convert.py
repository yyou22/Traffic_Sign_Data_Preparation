import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def main():

	path = "./Adv_Images_jpg/"
	if not os.path.exists(path):
		os.makedirs(path)

	images = np.load('X_pgd_1.npy')

	num = np.shape(images)[0]

	for i in range(0, num):

		img = np.transpose(images[i], (1, 2, 0))
		img = (img*255).astype(np.uint8)

		#plt.imshow(img)
		#plt.show()

		s1 = f'{i:05d}'

		img_ = Image.fromarray(img, 'RGB')
		img_.save(path + s1 + '.jpg')

		#x = np.shape(img)[0]
		#y = np.shape(img)[1]

		#plt.imshow(img)
		#plt.show()

		#s1 = f'{i:05d}'

		#plt.savefig(s1 + '.jpg')

if __name__ == '__main__':
	main()