import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

import argparse


def parse_args():
	parser = argparse.ArgumentParser(description='Shadow removal')
	parser.add_argument('--image_path', type=str, default='imgs/D084062.tif', help='image to remove shadow from')
	args = parser.parse_args()
	return args


## TODO:
# 1. invariant image formation


class ShadyRemoval:
	def __init__(self, image_path):
		self.image_path = image_path
		self.img = self.read_img(self.image_path)
		self.inv_img = self.inv_img(self.img)


	def read_img(self, image_path):
		img = cv.imread(image_path)
		img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
		img = cv.GaussianBlur(img, (5,5), 0)
		#plt.imshow(img)
		#plt.title('Original')
		#plt.show()
		return img

	def eval_img(self):
		pass


	def detect_edges(self, img):
		# use sobel operator
		L_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
		filtered = cv.filter2D(img, -1, L_x)
		filtered = cv.filter2D(filtered, -1, L_x.T)

		plt.imshow(filtered)
		plt.title('edge detection')
		plt.show()



	def detect_shadow(self, img):
		#Shadows are evidence of a sharp change in illumination
		# find intrinsic image in each of the separate r,g,b channels
		r = img[:,:,0]
		g = img[:,:,1]
		b = img[:,:,2]

		# use threshold derivatives to remove the effect of illumination
		pass


	def inv_img(self, img):
		#form band-ratio chromaticities from colour
		# r_k = p_k / p_p
		# for each r,g,b
		r = img[:,:,0]
		g = img[:,:,1]
		b = img[:,:,2]

		# we devide by green && effectively remove intensity information
		# green has the most illumination!
		n_r = np.divide(r.astype(np.float), g.astype(np.float))
		n_b = np.divide(b.astype(np.float), g.astype(np.float))

		# isolate the temp -> log
		log_rg = np.log(n_r)
		log_bg = np.log(n_b)

		self.detect_edges(log_bg)

		# greyscale invariant
		c1 = np.power(3.74183,-16)
		c2 = np.power(1.4388,-10)
		gs = c1 * log_rg - c2 * log_bg

		plt.imshow(gs, cmap ='gray')
		plt.title('Grayscale invariant image')
		plt.show()


		# pk = E(lambda_k)S(lambda_k)q_k
		# approx lightning by planck's law
		#p_k = I*c_1 * lambda_k^-5 * e^(-c_2/(lambda_k*T)) * q_k
		# where RGB colour =p_k, k=1,2,3 channel

		# band-ratiochromaticities
		# r_k = p_k / p_p, where p = channel, k indexes over remaining responses

		# isolate temp. term
		# r'_k = log(r_k) = log(s_k/s_p) + (e_k - e_p) / T


if __name__ == '__main__':
	args = parse_args()
	shady = ShadyRemoval(args.image_path)