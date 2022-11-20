import os
from PIL import Image

def convert():

	path = "./Images_1/"
	if not os.path.exists(path):
		os.makedirs(path)

	for i in range(0, 12630):
		im = Image.open("./Images_1_ppm/" + f'{i:05d}' + ".ppm")
		im.save("./Images_1/" + f'{i:05d}' + ".jpg")


if __name__ == '__main__':
	convert()