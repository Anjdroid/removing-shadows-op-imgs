import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

import argparse

from matplotlib.image import AxesImage


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
		#img = cv.GaussianBlur(img, (5,5), 0)
		#plt.imshow(img)
		#plt.title('Original')
		#plt.show()
		return img

	def eval_img(self):
		pass


	def non_max_suppression(self, grads, angles):
		suppressed = np.zeros(grads.shape)
		h, w = grads.shape
		angles = np.rad2deg(angles)
		print(angles)
		angles[angles < 0] += 180

		# TODO: optimize

		for i in range(h-1):
			for j in range(w-1):
				p, k = 255, 255
				theta = angles[i,j]
				if (theta >= 0 and theta < 22.5) or (theta >= 157.5 and theta <= 180):
					# angle 0
					p = grads[i, j-1]
					k = grads[i, j+1]
				elif (theta >= 22.5 and theta < 67.5):
					# angle 45
					p = grads[i+1,j-1]
					k = grads[i-1,j+1]
				elif (theta >= 67.5 and theta < 112.5):
					# angle 90
					p = grads[i+1,j]
					k = grads[i-1,j]
				elif (theta >= 112.5 and theta < 157.5):
					# angle 135
					p = grads[i-1,j-1]
					k = grads[i+1,j+1]
				if grads[i,j] >= p and grads[i,j] >= k:
					suppressed[i,j] = grads[i,j]
		return suppressed


	def detect_edges(self, img):
		# https://en.wikipedia.org/wiki/Canny_edge_detector
		# apply gauss filtering
		img = cv.GaussianBlur(img, (5,5), 0)

		# use sobel operator
		L_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)
		L_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
		G_x = cv.filter2D(img, -1, L_x)
		G_y = cv.filter2D(img, -1, L_y)
		#print(out)

		# get gradients
		gradients = np.sqrt(np.power(G_x,2) + np.power(G_y,2))
		#print(gradients)
		gradients = gradients / gradients.max() * 255

		# direction
		g_dir = np.arctan2(G_y,G_x)
		
		# non-maxima suppresion - thin edges
		suppressed = gradients
		# suppressed = self.non_max_suppression(gradients, g_dir)
		
		# TODO: test for better results
		# Set high and low threshold
		high_thresh = 70
		low_thresh = 50
		 
		h, w = suppressed.shape
		edge_mask = np.zeros((h, w), dtype=np.uint8)
		 
		# edge int >= strong= sure-edge
		# < 'low' threshold=non-edge
		sure_i, sure_j = np.where(suppressed >= high_thresh)
		non_i, non_j = np.where(suppressed < low_thresh)
		
		# weak edges
		weak_i, weak_j = np.where((suppressed <= high_thresh) & (suppressed >= low_thresh))
		 
		# Set same intensity value for all edge pixels
		#edge_mask[sure_i, sure_j] = 255
		edge_mask[non_i, non_j ] = 0
		#edge_mask[weak_i, weak_j] = 75

		# connect or break weak edges
		#for i in range(1, h-1):
		#	for j in range(1, w-1):
		#		if (out[i,j] == 75):
		#			if 255 in [out[i+1, j-1],out[i+1, j],out[i+1, j+1],out[i, j-1],out[i, j+1],out[i-1, j-1],out[i-1, j],out[i-1, j+1]]:
		#				out[i, j] = 255
		#			else:
		#				out[i, j] = 0
		# threshold gradients
		#combined[combined > 0.1] = 255
		#combined[combined != 255] = 0
		#out[out >= 1] = 1

		#plt.imshow(suppressed, cmap='gray')
		#plt.title('edge detection')
		#plt.show()

		return suppressed



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
		# green has the most illumination info!
		n_r = np.divide(r.astype(np.float), g.astype(np.float))
		n_b = np.divide(b.astype(np.float), g.astype(np.float))

		# isolate the temp -> log
		log_rg = np.log(n_r)
		log_bg = np.log(n_b)

		#self.detect_edges(log_bg)

		# greyscale invariant
		c1 = np.power(3.74183,-16)
		c2 = np.power(1.4388,-2)
		gs = c1 * log_rg - c2 * log_bg

		#plt.imshow(gs, cmap ='gray')
		#plt.title('Grayscale invariant image')
		#plt.show()

		# try dividing by red/blue
		#mean_r = np.mean(r)
		#mean_g = np.mean(g)
		#mean_b = np.mean(b)
		mean_img = np.mean(img, axis=2)
		n_mr = np.divide(r.astype(np.float), mean_img.astype(np.float))
		n_mb = np.divide(b.astype(np.float), mean_img.astype(np.float))
		log_mr = np.log(n_mr)
		log_mb = np.log(n_mb)

		# greyscale invariant
		gs_m = c1 * log_mr - c2 * log_mb

		#gs_r = gs_m - gs
		#plt.imshow(gs_m, cmap='gray')
		#plt.title('Grayscale invariant image')
		#plt.show()

		
		#self.detect_edges(gs_m)

		# pk = E(lambda_k)S(lambda_k)q_k
		# approx lightning by planck's law
		#p_k = I*c_1 * lambda_k^-5 * e^(-c_2/(lambda_k*T)) * q_k
		# where RGB colour =p_k, k=1,2,3 channel

		# band-ratiochromaticities
		# r_k = p_k / p_p, where p = channel, k indexes over remaining responses

		# isolate temp. term
		# r'_k = log(r_k) = log(s_k/s_p) + (e_k - e_p) / T

		#
		# create intrinsic image for each RGB channel p(x,y)
		# use thresholded derivatives to remove the effect of illumination
		# sensor response is the multiplication of light and surface

		# gradient of channel response
		# log-image edge map

		# gradient of grayscale invariant image gs(x,y)
		grad_gs_inv = self.detect_edges(gs)
		grad_log_rg = self.detect_edges(log_rg)
		#grad_log_bg = self.detect_edges(log_bg)

		# the difference between log colour responses removes the effect of the illumination
		#g#rad_log = grad_log_rg - grad_log_bg  ## ?????????
		#cv.imshow('',grad_log)
		#cv.waitKey(0)

		print(grad_gs_inv.shape)

		tresholded = grad_log_rg.copy()
		print(tresholded.shape)

		## TODO: set up proper thresholds 
		tresh1 = 20
		tresh2 = 70
		# factor out changes in grad at shadow edges
		for i in range(grad_gs_inv.shape[0]):
			for j in range(grad_gs_inv.shape[1]):
				#print(grad_gs_inv[i][j])
				#print(grad_log_rg[i][j])
				if grad_gs_inv[i][j] < tresh1 and grad_log_rg[i][j] > tresh2:
					tresholded[i][j] = 0 # NO CHANGE HERE

		# whats supposed to happen:
		# gradient image where sharp edges are indicative of material changes: no more sharp edges due to illumination -- shadows have been removed
		fig, ax = plt.subplots(1,3)
		ax[0].imshow(tresholded, cmap='gray')
		ax[1].imshow(grad_gs_inv, cmap='gray')
		ax[2].imshow(grad_log_rg, cmap='gray')
		plt.show()






if __name__ == '__main__':
	args = parse_args()
	shady = ShadyRemoval(args.image_path)