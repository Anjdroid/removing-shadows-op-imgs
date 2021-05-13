import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

#np.set_printoptions(threshold=sys.maxsize)

import argparse

from matplotlib.image import AxesImage
from skimage import filters


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
		# convert image from opencv's BRG to standard RGB
		img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
		return img

	def eval_img(self):
		pass


	def non_max_suppression(self, grads, angles):
		suppressed = np.zeros(grads.shape)
		h, w = grads.shape
		angles = np.rad2deg(angles)
		#print(angles)
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
		
		# TODO: appropriate sigma?
		# apply gauss filtering
		#blurred = cv.GaussianBlur(img, (5,5), 0)
		blurred = img

		# use sobel operator
		L_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)
		L_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
		G_x = cv.filter2D(blurred, -1, L_x)
		G_y = cv.filter2D(blurred, -1, L_y)
		#print(out)

		# normalize img values as [0,255] uint8
		norm_img = (255*(img - np.min(img))/np.ptp(img)).astype(np.uint8) 
		#print(norm_img.shape)

		#edges = cv.Canny(norm_img ,img.shape[0],img.shape[1])
		#print(edges.shape)

		"""plt.subplot(141),plt.imshow(blurred, cmap = 'gray')
		plt.title('Blurred OG'), plt.xticks([]), plt.yticks([])
		plt.subplot(142),plt.imshow(G_x,cmap = 'gray')
		plt.title('Edge Image X'), plt.xticks([]), plt.yticks([])
		plt.subplot(143),plt.imshow(G_y,cmap = 'gray')
		plt.title('Edge Image Y'), plt.xticks([]), plt.yticks([])
		plt.subplot(144),plt.imshow(norm_img,cmap = 'gray')
		plt.title('norm img'), plt.xticks([]), plt.yticks([])
		plt.show()"""

		# get gradients
		gradients = np.sqrt(np.power(G_x,2) + np.power(G_y,2))
		#print(gradients)
		gradients = gradients / gradients.max() * 255
		# direction
		g_dir = np.arctan2(G_y,G_x)

		# non-maxima suppresion - thin edges
		suppressed = gradients
		#suppressed = self.non_max_suppression(gradients, g_dir)
		suppressed_copy = suppressed.copy()
		
		# TODO: test for better results
		# Set high and low threshold
		high_thresh = np.max(suppressed) * 0.40 
		print(high_thresh)
		low_thresh = high_thresh * 0.25
		print(low_thresh)

		# apply low + high thresh
		#h, w = suppressed.shape
		#edge_mask = np.zeros((h, w), dtype=np.uint8)
		# edge int >= strong= sure-edge
		# < 'low' threshold=non-edge
		sure_i, sure_j = np.where(suppressed >= high_thresh)
		non_i, non_j = np.where(suppressed < low_thresh)
		# weak edges
		weak_i, weak_j = np.where((suppressed <= high_thresh) & (suppressed >= low_thresh))
		# Set same intensity value for all edge pixels
		suppressed_copy[sure_i, sure_j] = 255
		suppressed_copy[non_i, non_j ] = 0
		suppressed_copy[weak_i, weak_j] = 70

		# hysteresis thresholding
		hyst = filters.apply_hysteresis_threshold(suppressed_copy, low_thresh, high_thresh)
		hyst = np.where(hyst, 255, 0)
		

		"""plt.subplot(141),plt.imshow(gradients, cmap = 'gray')
		plt.title('grads'), plt.xticks([]), plt.yticks([])
		plt.subplot(142),plt.imshow(suppressed,cmap = 'gray')
		plt.title('nonmax supp'), plt.xticks([]), plt.yticks([])
		plt.subplot(143),plt.imshow(suppressed_copy,cmap = 'gray')
		plt.title('edge mask high,low'), plt.xticks([]), plt.yticks([])
		plt.subplot(144),plt.imshow(hyst,cmap = 'gray')
		plt.title('hyst img'), plt.xticks([]), plt.yticks([])
		plt.show()"""

		return hyst



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

		#plt.imshow(img)
		#plt.title('OG color image img')
		#plt.show()

		# we devide by green && effectively remove intensity information
		# green has the most illumination info!
		r_r = np.divide(r.astype(np.float), g.astype(np.float))
		r_b = np.divide(b.astype(np.float), g.astype(np.float))

		# isolate the temp -> log
		log_rg = np.log(r_r)
		log_bg = np.log(r_b)

		#print(gs)
		#log_rg = log_rg.astype(np.uint8)

		"""plt.subplot(131),plt.imshow(img)
		plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		plt.subplot(132),plt.imshow(log_rg,cmap = 'gray')
		plt.title('Log RG'), plt.xticks([]), plt.yticks([])
		plt.subplot(133),plt.imshow(log_bg,cmap = 'gray')
		plt.title('Log BG'), plt.xticks([]), plt.yticks([])
		plt.show()"""

		#self.detect_edges(log_bg)

		# greyscale invariant
		c1 = np.power(3.74183,-16)
		c2 = np.power(1.4388,-2)
		gs = c1 * log_rg - c2 * log_bg

		# MEAN IMG
		mean_img = np.mean(img, axis=2)
		r_mr = np.divide(r.astype(np.float), mean_img.astype(np.float))
		r_mb = np.divide(b.astype(np.float), mean_img.astype(np.float))
		log_mr = np.log(r_mr)
		log_mb = np.log(r_mb)

		# greyscale invariant
		gs_m = c1 * log_mr - c2 * log_mb

		"""plt.subplot(131),plt.imshow(img)
		plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		plt.subplot(132),plt.imshow(gs,cmap = 'gray')
		plt.title('Grayscale illumination inv'), plt.xticks([]), plt.yticks([])
		plt.subplot(133),plt.imshow(gs_m,cmap = 'gray')
		plt.title('GS ill non-inv'), plt.xticks([]), plt.yticks([])
		plt.show()"""

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

		#print(gs.shape)

		# gradient of channel response
		# log-image edge map
		

		# detect edges of gs ill inv img
		grad_gs_inv = self.detect_edges(gs)
		grad_gs_m = self.detect_edges(gs_m)

		print(grad_gs_inv)
		print(grad_gs_m)
		#grad_log_rg = self.detect_edges(log_rg)
		#grad_log_bg = self.detect_edges(log_bg)

		# the difference between log colour responses removes the effect of the illumination
		#grad_log = grad_log_rg - grad_log_bg  ## ?????????
		#cv.imshow('',grad_log)
		#cv.waitKey(0)

		#print(grad_log.shape)

		#t#resholded = grad_log.copy()
		#print(tresholded.shape)

		## TODO: set up proper thresholds 
		thresh1 = 10
		thresh2 = 50

		zero_i, zero_j = np.where((grad_gs_m > thresh1) & (grad_gs_inv < thresh2))
		shadowless = grad_gs_inv.copy()
		shadowless[zero_i, zero_j] = 0

		print(shadowless.shape)

		# integrate shadowless gradient image
		#new = self.detect_edges(np.int8(shadowless))
		edge_r = self.detect_edges(r)
		edge_g = self.detect_edges(g)
		edge_b = self.detect_edges(b)

		lap1 = np.array([[0,1,0],[1,-4,1],[0,1,0]])
		lap2 = np.array([[1,1,1],[1,-8,1],[1,1,1]])

		result = np.zeros(img.shape)
		res_r = cv.filter2D(r, -1, shadowless)
		res_g = cv.filter2D(g, -1, shadowless)
		res_b = cv.filter2D(b, -1, shadowless)

		result[:,:,0] = res_r
		result[:,:,1] = res_g
		result[:,:,2] = res_b



		# whats supposed to happen:
		# gradient image where sharp edges are indicative of material changes: no more sharp edges due to illumination -- shadows have been removed
		fig, ax = plt.subplots(1,3)
		ax[0].imshow(grad_gs_inv, cmap='gray')
		ax[1].imshow(shadowless, cmap='gray')
		ax[2].imshow(result)
		plt.show()






if __name__ == '__main__':
	args = parse_args()
	shady = ShadyRemoval(args.image_path)