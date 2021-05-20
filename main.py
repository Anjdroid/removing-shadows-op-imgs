import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean

from scipy.fft import fft2, ifft2

from PIL import Image

np.set_printoptions(threshold=sys.maxsize)

import argparse

from matplotlib.image import AxesImage
from skimage import filters


from skimage import color, data, restoration

#import sympy as symp

from scipy.stats import entropy

from math import pi, sin
import math


# color temperature table
kelvin_table = {
	1000: (255,56,0),
	1500: (255,109,0),
	2000: (255,137,18),
	2500: (255,161,72),
	3000: (255,180,107),
	3500: (255,196,137),
	4000: (255,209,163),
	4500: (255,219,186),
	5000: (255,228,206),
	5500: (255,236,224),
	6000: (255,243,239),
	6500: (255,249,253),
	7000: (245,243,255),
	7500: (235,238,255),
	8000: (227,233,255),
	8500: (220,229,255),
	9000: (214,225,255),
	9500: (208,222,255),
	10000: (204,219,255)}


def parse_args():
	parser = argparse.ArgumentParser(description='Shadow removal')
	parser.add_argument('--image_path', type=str, default='imgs/D084062.tif', help='image to remove shadow from')
	parser.add_argument('--visualize', type=str, default='False', help='True/False for visualisation of results')
	args = parser.parse_args()
	return args


class ShadyRemoval:
	def __init__(self, image_path, visualize):
		self.image_path = image_path
		self.img = self.read_img(self.image_path)
		self.visualize = False
		self.inv_img = self.process_img(self.img)


	def read_img(self, image_path):
		# read image
		img = cv.imread(image_path)
		# convert to BGR
		if img.shape[2] == 4:
			img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
		# convert image from opencv's BRG to standard RGB
		#img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
		img = cv.resize(img, (int(img.shape[0]/10), int(img.shape[1]/10)), -1)
		return img


	def adjust_temp(self, image, temp):
		# adjust color temperature of RGB image
		r, g, b = kelvin_table[temp]
		matrix = ( r / 255.0, 0.0, 0.0, 0.0,
				   0.0, g / 255.0, 0.0, 0.0,
				   0.0, 0.0, b / 255.0, 0.0 )
		image = Image.fromarray(image)
		return np.asarray(image.convert('RGB', matrix))


	def non_max_suppression(self, grads, angles):
		# perform non-max suppresion & thin edges
		suppressed = np.zeros(grads.shape)
		h, w = grads.shape
		angles = np.rad2deg(angles)
		angles[angles < 0] += 180
		for i in range(h - 1):
			for j in range(w - 1):
				p, k = 255, 255
				theta = angles[i,j]
				if (theta >= 0 and theta < 22.5) or (theta >= 157.5 and theta <= 180):
					# angle 0
					p = grads[i, j-1]
					k = grads[i, j+1]
				elif (theta >= 22.5 and theta < 67.5):
					# angle 45
					p = grads[i+1, j-1]
					k = grads[i-1, j+1]
				elif (theta >= 67.5 and theta < 112.5):
					# angle 90
					p = grads[i+1, j]
					k = grads[i-1, j]
				elif (theta >= 112.5 and theta < 157.5):
					# angle 135
					p = grads[i-1, j-1]
					k = grads[i+1, j+1]
				if grads[i,j] >= p and grads[i,j] >= k:
					suppressed[i,j] = grads[i,j]
		return suppressed


	def apply_filter(self, img, f):
		# apply filter f to image img
		filtered = cv.filter2D(img, -1, f)
		return filtered


	def grad_mag(self, gx, gy):
		# calculate gradient magnitudes
		return np.sqrt(np.power(gx,2) + np.power(gy,2))


	def scale_data(self, img, max_val):
		# scale data based on max_val
		return img / img.max() * max_val


	def norm_data(self, data):
		return (data - np.min(data)) / (np.max(data) - np.min(data))


	def apply_thresh(self, grads, high_t, low_t):
		# apply thresholds high_t and low_t to gradient image
		# where grad >= high_t == SURE EDGE
		# where grad < low_t == NON-EDGE
		sure_i, sure_j = np.where(grads >= high_t)
		non_i, non_j = np.where(grads < low_t)
		# weak edges == MAYBE edges
		weak_i, weak_j = np.where((grads <= high_t) & (grads >= low_t))
		# set sure edges to 255
		grads[sure_i, sure_j] = 255
		# set non-edges to 0
		grads[non_i, non_j ] = 0
		# perform hysteresis thresholding
		hyst = filters.apply_hysteresis_threshold(grads, low_t, high_t)
		# obtain img edge mask
		hyst = np.where(hyst, 255, 0)
		grads[np.where(hyst == 0)] = 0
		return grads, hyst


	def detect_edges(self, img, t1, t2):
		# define & use sobel operator for determining edges
		L_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)
		L_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
		# obtain gradients in x,y direction
		G_x = self.apply_filter(img, L_x)
		G_y = self.apply_filter(img, L_y)
		# get gradient magnitudes
		gradients = self.grad_mag(G_x, G_y)
		# normalize gradients [0-255]
		gradients = self.scale_data(gradients, 255)
		# get gradient directions
		g_dir = np.arctan2(G_y,G_x)
		suppressed = gradients
		#suppressed = self.non_max_suppression(gradients, g_dir)
		
		# threshold results
		# set high and low threshold
		high_thresh = np.max(suppressed) * t1
		low_thresh = high_thresh * t2
		grads, hyst = self.apply_thresh(suppressed, high_thresh, low_thresh)

		# perform non-maxima suppresion & thin edges
		# grads = self.non_max_suppression(gradients, g_dir)
		# hyst[np.where(grads == 0)] = 0

		return hyst, grads, G_x, G_y


	def refine_shadow_mask(self, mask):
		# define morphology kernel
		kernel = np.array([[0,1,1],[1,1,1],[1,1,1]]).astype(np.uint8)
		# perform dilate and erote on the shadow edges mask
		mask = cv.dilate(mask.astype(np.uint8), kernel, iterations=3)
		mask = cv.erode(mask.astype(np.uint8), kernel, iterations=1)
		# turn shadow edges into contours		
		contours, _ =  cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
		return contours


	def add_contours(self, ch, contours):
		# apply contour to color channel ch
		masked_ch = cv.drawContours(ch.astype(np.uint8), contours, -1, (0, 0, 0), 1)
		# fill drawn contours
		masked_ch = cv.fillPoly(masked_ch.astype(np.uint8), pts=contours, color=(0,0,0))
		if self.visualize:
			cv.imshow('Masked channel', masked_ch.astype(np.uint8))
			cv.waitKey(0)
		return masked_ch


	def correct_grads(self, mask, gx, gy):
		# apply mask to gradients
		sh_gx = np.zeros(gx.shape)
		sh_gy = np.zeros(gy.shape)
		zeros_i, zeros_j = np.where(mask == 255)
		sh_gx[zeros_i, zeros_j] = gx[zeros_i, zeros_j]
		sh_gy[zeros_i, zeros_j] = gy[zeros_i, zeros_j]
		return sh_gx, sh_gy


	def interpolate(self, img):
		kernel = np.array([[0,1/2,0],[1/2,1/8,1/4],[0,1/4,0]])
		res = self.apply_filter(img, kernel)
		return res


	def stretch_img(self, mask, max_v, min_v, g_max, g_min, img):
		# iterate through the image and correct gray values
		# with gray level linear transformation
		# res(i,j) = (fmax-fmin)/(gmax-gmin)) * (img(i,j)-gmin) + fmin
		res = img.copy()
		a = (max_v - min_v) / (g_max - g_min)
		h, w = img.shape

		img_mask = img * mask
		cv.imshow('i',img_mask)
		cv.waitKey(0)
		num_zeros = np.count_nonzero(img==0)
		mean_img = np.sum(np.sum(img, axis=0), axis=0) / (h * w - num_zeros)
		
		for i in range(h):
			for j in range(w):
				if mask[i, j] == 255:
					kernel = np.array([[0,1/2,0],[1/2,1/8,1/4],[0,1/4,0]])
					pixel_value = np.clip((a * (img[i,j] - g_min) + mean_img), 0, 255)
					res[i,j] = pixel_value

					#print(res[i])
					#print(res[j])
					
					#interpolated = np.multiply(kernel, )

					res[i,j] = np.clip((a * (img[i,j] - g_min) + mean_img), 0, 255)

		# interpolate
		res = self.interpolate(res)
		return res


	def reconstruct_img(self, ch, shadow, sh_mask, edges):
		log = np.log(ch)
		log = self.scale_data(log, 255)
		sh_gx, sh_gy = shadow
		ed_gx, ed_gy = edges

		# detect gradients of log channel
		mask, grads, gx, gy = self.detect_edges(log.astype(np.uint8), 0.8, 0.3)

		if self.visualize:
			cv.imshow('shadow grads', sh_gx)
			cv.imshow('shadow grads', sh_gy)
			cv.imshow('mask', mask.astype(np.uint8))
			cv.imshow('grads',  grads.astype(np.uint8))
			cv.imshow('grads x', gx.astype(np.uint8))
			cv.imshow('grads y', gy.astype(np.uint8))
			cv.waitKey(0)

		# exclude shadow gradients from log channel gradients
		d_gx, d_gy = self.correct_grads(255 - sh_mask, gx, gy)

		cv.imshow('grads', grads.astype(np.uint8))
		cv.waitKey(0)
		
		# norm to [0,1]
		norm_dgx = self.norm_data(d_gx)
		norm_dgy = self.norm_data(d_gy)
		norm_gx = self.norm_data(gx)
		norm_gy = self.norm_data(gy)

		Fx =  np.fft.fft2(norm_dgx)
		Fy =  np.fft.fft2(norm_dgy)
		Fx = np.fft.fftshift(Fx)
		Fy = np.fft.fftshift(Fy)

		print(Fx.shape)
		print(Fy.shape)

		Z = np.zeros(Fx.shape, dtype='complex')

		poisson_iters = 100
		err = 1
		tol = 1e-3

		predx = Fx
		predy = Fy

		h, w = Fx.shape
		for it in range(poisson_iters):
			pred = np.copy(Z)
			for i in range(1, h - 1):
				for j in range(1, w - 1):
					ax = np.exp(2 * np.pi * j / w - 1)
					ay = np.exp(2 * np.pi * i / h - 1)

					#print(i,j)
					#print(ax)
					Zx = (ax * Fx[i,j]) / (ax**2 + ay**2)
					Zy = (ay * Fy[i,j]) / (ax**2 + ay**2)

					#print(Zx, Zy)
					Z[i, j] = Zx + Zy
					#print(Z[i,j])


			err = np.sum(np.sum(np.power((Z - pred), 2)))
			print("= ", err)
			if err < tol:
				break


		res_shifted = np.fft.ifftshift(Z)
		res = ifft2(res_shifted)
		print(res.shape)
		res = np.power(res, 2)
		res = self.scale_data(res, 255)

		#print(res)
		#if self.visualize:
		cv.imshow('reconstructed', res.astype(np.uint8))
		cv.waitKey(0)
		return res


	def find_min_max(self, arr):
		ct_idx = 0
		for idx,item in enumerate(arr):
			#print(idx, item)
			if int(item) == 0:
				continue
			else:
				ct_idx += 1
				#return idx
			if ct_idx > 15:
				# excluding the top 15% values
				return idx


	def linear_transform_reconstruction(self, shadow_mask, channel):
		# refine shadow edge mask
		shadow_contours = self.refine_shadow_mask(shadow_mask)
		mask = self.add_contours(np.ones(channel.shape) * 255, shadow_contours)
		shadow_mask = 255 - mask

		# compute histogram for non-shadow and shadow region
		# apply non-shadow/shadow region as weights
		hist_non_shadow, bin_edges = np.histogram(np.copy(channel), bins=255, range=(0, 256), weights=mask)
		hist_shadow, bin_edges_shadow =  np.histogram(np.copy(channel), bins=255, range=(0, 256), weights=shadow_mask)

		if self.visualize:
			plt.plot(bin_edges[0:-1], hist_non_shadow)
			plt.title("Non shadow region histogram")
			plt.show()

			plt.plot(bin_edges[0:-1], hist_shadow)
			plt.title("Shadow region histogram")
			plt.show()

			# calculate cumulative histograms
			cum_hist = np.cumsum(hist_non_shadow)
			cum_hist_shadow = np.cumsum(hist_shadow)

			plt.plot(bin_edges[0:-1], cum_hist)
			plt.title("Non shadow region cumulative histogram")
			plt.show()

			plt.plot(bin_edges[0:-1], cum_hist_shadow)
			plt.title("Shadow region cumulative histogram")
			plt.show()

		# get highest & lowest val in the highest/lowest 5%
		highest_val = len(hist_non_shadow) - self.find_min_max(np.flip(hist_non_shadow))
		lowest_val = self.find_min_max(hist_non_shadow)
		
		highest_val2 = len(hist_shadow) - self.find_min_max(np.flip(hist_shadow))
		lowest_val2 = self.find_min_max(hist_shadow)

		# perform linear transformation of SHADOW region gray lvls
		shadowless = self.stretch_img(shadow_mask, highest_val, lowest_val, highest_val2, lowest_val2, channel)
		return shadowless


	def equalize_channels(self, img):
		# convert image to LAB color space
		lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
		l = lab[:,:,0]
		mean_l = np.mean(l)
		print(mean_l)
		a = np.mean(lab[:,:,1])
		b = np.mean(lab[:,:,2])

		if (a + b) <= 300:
			shadow_thresh = (mean_l - np.std(l)/3)
			print(shadow_thresh)
			l[np.where(l <= shadow_thresh)] = 100
		else:
			l[np.where(l > shadow_thresh)] = -100

		lab[:,:,0] = l
		print(lab)
		return cv.cvtColor(lab, cv.COLOR_LAB2BGR)



	def process_img(self, img):
		# adjust image temperature
		temp = self.adjust_temp(img, 5000)
		temp = self.equalize_channels(temp)
		cv.imshow('equalized', temp)
		cv.waitKey(0)
		# blur image with gauss
		blurred = cv.GaussianBlur(temp, (5,5), 0)
		# split color channels
		b = blurred[:,:,0]
		g = blurred[:,:,1]
		r = blurred[:,:,2]

		# divide r,b with g color channel
		x_r = np.divide(r.astype(np.float), g.astype(np.float))
		x_b = np.divide(b.astype(np.float), g.astype(np.float))
		# obtain log 
		log_mr = np.log(x_r)
		log_mb = np.log(x_b)

		# invert values
		log_mr_inv = 255 - log_mr
		# accenuate shadow 
		shadow = log_mb + log_mr_inv
		# detect shadow edges
		# multiplication factors t1 and t2 are empirically set
		mask, grads, gx, gy = self.detect_edges(shadow, 0.7, 0.3)
		mask2, grads2, gx2, gy2 = self.detect_edges(log_mr, 0.25, 0.45)

		# the intersection of mask & mask2 gives us the shadow edge
		shadow_mask = np.multiply(mask, mask2)
		shadow_mask = self.scale_data(mask, 255)
		edge_mask = np.clip(mask2 - shadow_mask,0,255)

		# first perform contouring
		# then take gradients to 0 perhaps?!

		# gradient visualisation
		if self.visualize:
			plt.subplot(131),plt.imshow(shadow, cmap='gray')
			plt.title('Accenuated shadow'), plt.xticks([]), plt.yticks([])
			plt.subplot(132),plt.imshow(shadow_mask,cmap = 'gray')
			plt.title('Shadow mask'), plt.xticks([]), plt.yticks([])
			plt.subplot(133),plt.imshow(edge_mask,cmap = 'gray')
			plt.title('Edges without shadow'), plt.xticks([]), plt.yticks([])
			plt.show()

		sh_gx, sh_gy = self.correct_grads(shadow_mask, gx, gy)
		ed_gx, ed_gy = self.correct_grads(edge_mask, gx2, gy2)
		# poisson reconstruction
		"""res1 = self.reconstruct_img(r, (sh_gx, sh_gy), shadow_mask, (ed_gx, ed_gy))
		res2 = self.reconstruct_img(g, (sh_gx, sh_gy), shadow_mask, (ed_gx, ed_gy))
		res3 = self.reconstruct_img(b, (sh_gx, sh_gy), shadow_mask, (ed_gx, ed_gy))

		res = np.zeros(img.shape)
		res[:,:,0] = res1
		res[:,:,1] = res2
		res[:,:,2] = res3
		cv.imshow("rec poiss", res.astype(np.uint8))
		cv.waitKey(0)"""


		shadowless_r = self.linear_transform_reconstruction(shadow_mask, r)
		shadowless_g = self.linear_transform_reconstruction(shadow_mask, g)
		shadowless_b = self.linear_transform_reconstruction(shadow_mask, b)

		reconstructed = np.zeros(img.shape)
		reconstructed[:,:,2] = shadowless_r
		reconstructed[:,:,1] = shadowless_g
		reconstructed[:,:,0] = shadowless_b

		#if self.visualize:
		cv.imshow('Reconstructed image', reconstructed.astype(np.uint8))
		cv.waitKey(0)


		

if __name__ == '__main__':
	args = parse_args()
	shady = ShadyRemoval(args.image_path, args.visualize)