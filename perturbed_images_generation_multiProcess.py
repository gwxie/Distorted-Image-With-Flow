'''
GuoWang xie
set up :2018-3-8

-- data1024_greyV2
'''


import argparse
import os
import pickle
import random
import collections
import json
import numpy as np
import scipy.io as io
import scipy.misc as m
import matplotlib.pyplot as plt
import glob
import math
import time

from sklearn import preprocessing

import threading
import multiprocessing as mp
from multiprocessing import Pool
import re
import cv2
import scipy.spatial.qhull as qhull


def getDatasets(dir):
	return os.listdir(dir)

class perturbed(object):
	def __init__(self, path, bg_path, save_path, save_suffix):

		self.path = path
		self.bg_path = bg_path
		self.save_path = save_path
		self.save_suffix = save_suffix

	def get_normalize(self, d):
		E = np.mean(d)
		std = np.std(d)
		d = (d-E)/std
		# d = preprocessing.normalize(d, norm='l2')
		return d
	# d = np.abs(sk_normalize(d, norm='l2'))
	def get_0_1_d(self, d, new_max=1, new_min=0):
		d_min = np.min(d)
		d_max = np.max(d)
		d = ((d-d_min)/(d_max-d_min))*(new_max-new_min)+new_min
		return d

	def draw_distance_hotmap(self, distance_vertex_line):

		plt.matshow(distance_vertex_line, cmap='autumn')
		plt.colorbar()
		plt.show()

	def get_pixel(self, p, origin_img):
		try:
			return origin_img[p[0], p[1]]
		except:
			# print('out !')
			return np.array([257, 257, 257])

	def nearest_neighbor_interpolation(self, xy, new_origin_img):
		# xy = np.around(xy_).astype(np.int)
		origin_pixel = self.get_pixel([xy[0], xy[1]], new_origin_img)
		if (origin_pixel == 257).all():
			return origin_pixel, False
		return origin_pixel, True

	def bilinear_interpolation(self, xy_, new_origin_img):
		xy_int = [int(xy_[0]), int(xy_[1])]
		xy_decimal = [round(xy_[0] - xy_int[0], 5), round(xy_[1] - xy_int[1], 5)]
		x0_y0 = (1 - xy_decimal[0]) * (1 - xy_decimal[1]) * self.get_pixel([xy_int[0], xy_int[1]], new_origin_img)

		x0_y1 = (1 - xy_decimal[0]) * (xy_decimal[1]) * self.get_pixel([xy_int[0], xy_int[1] + 1], new_origin_img)

		x1_y0 = (xy_decimal[0]) * (1 - xy_decimal[1]) * self.get_pixel([xy_int[0] + 1, xy_int[1]], new_origin_img)

		x1_y1 = (xy_decimal[0]) * (xy_decimal[1]) * self.get_pixel([xy_int[0] + 1, xy_int[1] + 1], new_origin_img)

		return x0_y0, x0_y1, x1_y0, x1_y1

	def get_coor(self, p, origin_label):
		try:
			return origin_label[p[0], p[1]]
		except:
			# print('out !')
			return np.array([0, 0])

	def bilinear_interpolation_coordinate_v4(self, xy_, new_origin_img):

		xy_int = [int(xy_[0]), int(xy_[1])]
		xy_decimal = [round(xy_[0] - xy_int[0], 5), round(xy_[1] - xy_int[1], 5)]
		x_y_i = 0
		x0, x1, x2, x3 = 0, 0, 0, 0
		y0, y1, y2, y3 = 0, 0, 0, 0
		x0_y0 = self.get_coor(np.array([xy_int[0], xy_int[1]]), new_origin_img)
		x0_y1 = self.get_coor(np.array([xy_int[0], xy_int[1]+1]), new_origin_img)
		x1_y0 = self.get_coor(np.array([xy_int[0]+1, xy_int[1]]), new_origin_img)
		x1_y1 = self.get_coor(np.array([xy_int[0]+1, xy_int[1]+1]), new_origin_img)

		if x0_y0[0] != 0:
			x0 = (1 - xy_decimal[0])
		if x0_y1[0] != 0:
			x1 = (1 - xy_decimal[0])
		if x1_y0[0] != 0:
			x2 = (xy_decimal[0])
		if x1_y1[0] != 0:
			x3 = (xy_decimal[0])

		if x0_y0[1] != 0:
			y0 = (1 - xy_decimal[1])
		if x0_y1[1] != 0:
			y1 = (xy_decimal[1])
		if x1_y0[1] != 0:
			y2 = (1 - xy_decimal[1])
		if x1_y1[1] != 0:
			y3 = (xy_decimal[1])

		x_ = x0+x1+x2+x3
		if x_ == 0:
			x = 0
		else:
			x = x0/x_*x0_y0[0]+x1/x_*x0_y1[0]+x2/x_*x1_y0[0]+x3/x_*x1_y1[0]

		y_ = y0+y1+y2+y3
		if y_ == 0:
			y = 0
		else:
			y = y0/y_*x0_y0[1]+y1/y_*x0_y1[1]+y2/y_*x1_y0[1]+y3/y_*x1_y1[1]

		return np.array([x, y])


	def is_perform(self, execution, inexecution):
		return random.choices([True, False], weights=[execution, inexecution])[0]

	def get_margin_scale(self, min_, max_, clip_add_margin, new_shape):
		if clip_add_margin < 0:
			# raise Exception('add margin error')
			return -1, -1
		if min_-clip_add_margin//2 > 0 and max_+clip_add_margin//2 < new_shape:
			if clip_add_margin%2 == 0:
				clip_subtract_margin, clip_plus_margin = clip_add_margin//2, clip_add_margin//2
			else:
				clip_subtract_margin, clip_plus_margin = clip_add_margin//2, clip_add_margin//2+1
		elif min_-clip_add_margin//2 < 0 and max_+clip_add_margin//2 <= new_shape:
			clip_subtract_margin = min_
			clip_plus_margin = clip_add_margin-clip_subtract_margin
		elif max_+clip_add_margin//2 > new_shape and min_-clip_add_margin//2 >= 0:
			clip_plus_margin = new_shape-max_
			clip_subtract_margin = clip_add_margin-clip_plus_margin
		else:
			# raise Exception('add margin error')
			return -1, -1
		return clip_subtract_margin, clip_plus_margin

	# class perturbedCurveImg(object):
	# 	def __init__(self):

	def adjust_position(self, x_min, y_min, x_max, y_max):
		if (self.new_shape[0] - (x_max - x_min)) % 2 == 0:
			f_g_0_0 = (self.new_shape[0] - (x_max - x_min)) // 2
			f_g_0_1 = f_g_0_0
		else:
			f_g_0_0 = (self.new_shape[0] - (x_max - x_min)) // 2
			f_g_0_1 = f_g_0_0 + 1

		if (self.new_shape[1] - (y_max - y_min)) % 2 == 0:
			f_g_1_0 = (self.new_shape[1] - (y_max - y_min)) // 2
			f_g_1_1 = f_g_1_0
		else:
			f_g_1_0 = (self.new_shape[1] - (y_max - y_min)) // 2
			f_g_1_1 = f_g_1_0 + 1

		# return f_g_0_0, f_g_0_1, f_g_1_0, f_g_1_1
		return f_g_0_0, f_g_1_0, self.new_shape[0] - f_g_0_1, self.new_shape[1] - f_g_1_1

	def adjust_position_v2(self, x_min, y_min, x_max, y_max, new_shape):
		if (new_shape[0] - (x_max - x_min)) % 2 == 0:
			f_g_0_0 = (new_shape[0] - (x_max - x_min)) // 2
			f_g_0_1 = f_g_0_0
		else:
			f_g_0_0 = (new_shape[0] - (x_max - x_min)) // 2
			f_g_0_1 = f_g_0_0 + 1

		if (new_shape[1] - (y_max - y_min)) % 2 == 0:
			f_g_1_0 = (new_shape[1] - (y_max - y_min)) // 2
			f_g_1_1 = f_g_1_0
		else:
			f_g_1_0 = (new_shape[1] - (y_max - y_min)) // 2
			f_g_1_1 = f_g_1_0 + 1

		# return f_g_0_0, f_g_0_1, f_g_1_0, f_g_1_1
		return f_g_0_0, f_g_1_0, new_shape[0] - f_g_0_1, new_shape[1] - f_g_1_1

	def adjust_border(self, x_min, y_min, x_max, y_max, x_min_new, y_min_new, x_max_new, y_max_new):
		if ((x_max - x_min) - (x_max_new - x_min_new)) % 2 == 0:
			f_g_0_0 = ((x_max - x_min) - (x_max_new - x_min_new)) // 2
			f_g_0_1 = f_g_0_0
		else:
			f_g_0_0 = ((x_max - x_min) - (x_max_new - x_min_new)) // 2
			f_g_0_1 = f_g_0_0 + 1

		if ((y_max - y_min) - (y_max_new - y_min_new)) % 2 == 0:
			f_g_1_0 = ((y_max - y_min) - (y_max_new - y_min_new)) // 2
			f_g_1_1 = f_g_1_0
		else:
			f_g_1_0 = ((y_max - y_min) - (y_max_new - y_min_new)) // 2
			f_g_1_1 = f_g_1_0 + 1

		return f_g_0_0, f_g_0_1, f_g_1_0, f_g_1_1

	def interp_weights(self, xyz, uvw):
		tri = qhull.Delaunay(xyz)
		simplex = tri.find_simplex(uvw)
		vertices = np.take(tri.simplices, simplex, axis=0)
		# pixel_triangle = pixel[tri.simplices]
		temp = np.take(tri.transform, simplex, axis=0)
		delta = uvw - temp[:, 2]
		bary = np.einsum('njk,nk->nj', temp[:, :2, :], delta)
		return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

	def interpolate(self, values, vtx, wts):
		return np.einsum('njk,nj->nk', np.take(values, vtx, axis=0), wts)

	def isSavePerturbed(self, synthesis_perturbed_img, new_shape):
		if np.sum(synthesis_perturbed_img[:, 0]) != 771 * new_shape[0] or np.sum(synthesis_perturbed_img[:, new_shape[1] - 1]) != 771 * new_shape[0] or \
				np.sum(synthesis_perturbed_img[0, :]) != 771 * new_shape[1] or np.sum(synthesis_perturbed_img[new_shape[0] - 1, :]) != 771 * new_shape[1]:
			# raise Exception('clip error')
			return False
		else:
			return True

	def save_img(self, m, n, fold_curve='fold', repeat_time=4, relativeShift_position='relativeShift_v2'):
		origin_img = cv2.imread(self.path, flags=cv2.IMREAD_COLOR)

		# img_shrink, base_img_shrink = 512, 512
		# base_img_shape = [base_img_shrink, int(math.floor(base_img_shrink//2*1.8))]
		# clip_add_margin = [base_img_shrink//4, int(math.floor(base_img_shrink//8*1.8))]		# [128, int(round(64*1.8))]
		# save_img = 512
		# save_img_shape = [save_img, int(math.floor(save_img//2*1.8))]
		# # save_img_shape = [768, 688]
		# enlarge_img_shrink = [base_img_shape[0]+clip_add_margin[0], round((base_img_shape[1]+clip_add_margin[1])/10)*10]

		'''
		img_shrink, base_img_shrink = 1024, 1024
		save_img = None
		save_img_shape = [1280, 1024]
		# save_img = 768
		# save_img_shape = [save_img, 688]
		# save_img = 640
		# save_img_shape = [640, 576]
		enlarge_img_shrink = [1280, 1024]
		'''
		''''''
		save_img = None
		# save_img_shape = [768, 576]		# 320
		# save_img_shape = [640, 480]		# 320
		# save_img_shape = [512, 384]		# 320
		save_img_shape = [512*2, 480*2]		# 320
		# save_img_shape = [512, 480]		# 320
		reduce_value = np.random.choice([8*2, 16*2, 24*2, 32*2, 40*2, 48*2], p=[0.1, 0.2, 0.4, 0.1, 0.1, 0.1])
		# reduce_value = np.random.choice([16, 24, 32, 40, 48, 64], p=[0.01, 0.1, 0.2, 0.4, 0.2, 0.09])
		base_img_shrink = save_img_shape[0] - reduce_value

		# enlarge_img_shrink = [1024, 768]
		# enlarge_img_shrink = [896, 672]		# 420
		enlarge_img_shrink = [896*2, 768*2]		# 420
		# enlarge_img_shrink = [896, 768]		# 420
		# enlarge_img_shrink = [768, 576]		# 420
		# enlarge_img_shrink = [640, 480]		# 420

		''''''
		im_lr = origin_img.shape[0]
		im_ud = origin_img.shape[1]
		aspect_ratio = round(im_lr / im_ud, 2)

		reduce_value_v2 = np.random.choice([4*2, 8*2, 16*2, 24*2, 28*2, 32*2, 48*2, 64*2], p=[0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.08, 0.02])
		# reduce_value_v2 = np.random.choice([16, 24, 28, 32, 48, 64], p=[0.01, 0.1, 0.2, 0.3, 0.25, 0.14])

		if im_lr > im_ud and aspect_ratio > 1.2:
			im_ud = min(int(im_ud / im_lr * base_img_shrink), save_img_shape[1] - reduce_value_v2)
			im_lr = save_img_shape[0] - reduce_value
		else:
			base_img_shrink = save_img_shape[1] - reduce_value
			im_lr = min(int(im_lr / im_ud * base_img_shrink), save_img_shape[0] - reduce_value_v2)
			im_ud = base_img_shrink

		self.origin_img = cv2.resize(origin_img, (im_ud, im_lr), interpolation=cv2.INTER_CUBIC)

		perturbed_bg_ = getDatasets(self.bg_path)
		perturbed_bg_img_ = self.bg_path+random.choice(perturbed_bg_)
		perturbed_bg_img = cv2.imread(perturbed_bg_img_, flags=cv2.IMREAD_COLOR)

		mesh_shape = self.origin_img.shape[:2]
		mesh_shape_ = mesh_shape

		self.synthesis_perturbed_img = np.full((enlarge_img_shrink[0], enlarge_img_shrink[1], 3), 257, dtype=np.int16)#np.zeros_like(perturbed_bg_img)
		self.new_shape = self.synthesis_perturbed_img.shape[:2]
		perturbed_bg_img = cv2.resize(perturbed_bg_img, (save_img_shape[1], save_img_shape[0]), cv2.INPAINT_TELEA)

		origin_pixel_position = np.argwhere(np.zeros(mesh_shape, dtype=np.uint32) == 0).reshape(mesh_shape[0], mesh_shape[1], 2)
		pixel_position = np.argwhere(np.zeros(self.new_shape, dtype=np.uint32) == 0).reshape(self.new_shape[0], self.new_shape[1], 2)
		self.perturbed_xy_ = pixel_position.copy()

		self.synthesis_perturbed_label = np.zeros((self.new_shape[0], self.new_shape[1], 2))
		x_min, y_min, x_max, y_max = self.adjust_position_v2(0, 0, mesh_shape[0], mesh_shape[1], save_img_shape)
		origin_pixel_position += [x_min, y_min]


		x_min, y_min, x_max, y_max = self.adjust_position(0, 0, mesh_shape[0], mesh_shape[1])
		self.synthesis_perturbed_img[x_min:x_max, y_min:y_max] = self.origin_img
		self.synthesis_perturbed_label[x_min:x_max, y_min:y_max] = origin_pixel_position


		synthesis_perturbed_img_map = self.synthesis_perturbed_img.copy()
		synthesis_perturbed_label_map = self.synthesis_perturbed_label.copy()
		'''*****************************************************************'''
		alpha_perturbed = random.randint(8, 14) / 10
		self.perturbed_x_min, self.perturbed_y_min, self.perturbed_x_max, self.perturbed_y_max = -1, -1, self.new_shape[0], self.new_shape[1]

		perturbed_time = 0
		fail_perturbed_time = 0
		is_normalizationFun_mixture = self.is_perform(0.1, 0.9)
		# if not is_normalizationFun_mixture:
		normalizationFun_0_1 = False
		# normalizationFun_0_1 = self.is_perform(0.5, 0.5)

		if fold_curve == 'fold':

			fold_curve_random = True
			# is_normalizationFun_mixture = False
			normalizationFun_0_1 = self.is_perform(0.2, 0.8)
			if is_normalizationFun_mixture:
				if self.is_perform(0.5, 0.5):
					# alpha_perturbed = random.randint(80, 120) / 100  # min(max(round(np.random.normal(6, 2)), 4), 12)/10
					alpha_perturbed = random.randint(100, 140) / 100
				else:
					# alpha_perturbed = random.randint(50, 100) / 100
					alpha_perturbed = random.randint(80, 120) / 100
			else:
				if normalizationFun_0_1:
					alpha_perturbed = random.randint(40, 50) / 100
				else:
					alpha_perturbed = random.randint(70, 120) / 100
		else:
			fold_curve_random = self.is_perform(0.1, 0.9)  # False		# self.is_perform(0.01, 0.99)
			alpha_perturbed = random.randint(80, 160) / 100
			is_normalizationFun_mixture = False  # self.is_perform(0.01, 0.99)


		for repeat_i in range(repeat_time):

			synthesis_perturbed_img = np.full_like(self.synthesis_perturbed_img, 257, dtype=np.int16)
			synthesis_perturbed_label = np.zeros_like(self.synthesis_perturbed_label)

			perturbed_p, perturbed_pp = np.array(
				[random.randint((self.new_shape[0]-im_lr)//2*10, (self.new_shape[0]-(self.new_shape[0]-im_lr)//2) * 10) / 10,
				 random.randint((self.new_shape[1]-im_ud)//2*10, (self.new_shape[1]-(self.new_shape[1]-im_ud)//2) * 10) / 10]) \
				, np.array([random.randint((self.new_shape[0]-im_lr)//2*10, (self.new_shape[0]-(self.new_shape[0]-im_lr)//2) * 10) / 10,
				 			random.randint((self.new_shape[1]-im_ud)//2*10, (self.new_shape[1]-(self.new_shape[1]-im_ud)//2) * 10) / 10])

			perturbed_vp = perturbed_pp - perturbed_p
			perturbed_vp_norm = np.linalg.norm(perturbed_vp)

			perturbed_distance_vertex_and_line = np.dot((perturbed_p - pixel_position), perturbed_vp) / perturbed_vp_norm
			''''''
			if fold_curve == 'fold' and self.is_perform(0.3, 0.7):
				perturbed_v = np.array([random.randint(-11000, 11000) / 100, random.randint(-11000, 11000) / 100])
			else:
				perturbed_v = np.array([random.randint(-9000, 9000) / 100, random.randint(-9000, 9000) / 100])
			''''''
			if fold_curve == 'fold':
				if is_normalizationFun_mixture:
					if self.is_perform(0.5, 0.5):
						perturbed_d = np.abs(self.get_normalize(perturbed_distance_vertex_and_line))
					else:
						perturbed_d = self.get_0_1_d(np.abs(perturbed_distance_vertex_and_line), random.randint(1, 2))
				else:
					if normalizationFun_0_1:
						perturbed_d = self.get_0_1_d(np.abs(perturbed_distance_vertex_and_line), 2)
					else:
						perturbed_d = np.abs(self.get_normalize(perturbed_distance_vertex_and_line))

			else:
				if is_normalizationFun_mixture:
					if self.is_perform(0.5, 0.5):
						perturbed_d = np.abs(self.get_normalize(perturbed_distance_vertex_and_line))
					else:
						perturbed_d = self.get_0_1_d(np.abs(perturbed_distance_vertex_and_line), random.randint(1, 2))
				else:
					if normalizationFun_0_1:
						perturbed_d = self.get_0_1_d(np.abs(perturbed_distance_vertex_and_line), 2)
					else:
						perturbed_d = np.abs(self.get_normalize(perturbed_distance_vertex_and_line))
			''''''
			if fold_curve_random:
				omega_perturbed = alpha_perturbed / (perturbed_d + alpha_perturbed)
			else:
				omega_perturbed = 1 - perturbed_d ** alpha_perturbed

			'''shadow'''
			if self.is_perform(0.6, 0.4):
				synthesis_perturbed_img_map[x_min:x_max, y_min:y_max] = np.minimum(np.maximum(synthesis_perturbed_img_map[x_min:x_max, y_min:y_max] - np.int16(np.round(omega_perturbed[x_min:x_max, y_min:y_max].repeat(3).reshape(x_max-x_min, y_max-y_min, 3) * abs(np.linalg.norm(perturbed_v//2))*np.array([0.4-random.random()*0.1, 0.4-random.random()*0.1, 0.4-random.random()*0.1]))), 0), 255)
			''''''

			if relativeShift_position in ['position', 'relativeShift_v2']:

				perturbed_xy_ = self.perturbed_xy_ + np.array([omega_perturbed * perturbed_v[0], omega_perturbed * perturbed_v[1]]).transpose(1, 2, 0)
				perturbed_xy_ = cv2.blur(perturbed_xy_, (17, 17))
				perturbed_xy_round_int = np.around(perturbed_xy_)
				perturbed_xy_round_int = np.around(perturbed_xy_round_int).astype(np.int)

				# b = time.time()
				it_r_i_0 = np.nditer(perturbed_xy_round_int[:, :, 0], flags=['multi_index'])
				it_r_i_1 = np.nditer(perturbed_xy_round_int[:, :, 1], flags=['multi_index'])

				while not it_r_i_0.finished:
					try:
						synthesis_perturbed_img[it_r_i_0.multi_index] = synthesis_perturbed_img_map[it_r_i_0[0], it_r_i_1[0]]
					except:
						it_r_i_0.iternext()
						it_r_i_1.iternext()
						continue

					synthesis_perturbed_label[it_r_i_0.multi_index] = synthesis_perturbed_label_map[it_r_i_0[0], it_r_i_1[0]]
					it_r_i_0.iternext()
					it_r_i_1.iternext()
				# bb = time.time() - b
			else:
				print('relativeShift_position error')
				exit()

			''''''
			is_save_perturbed = False
			is_save_perturbed_1, is_save_perturbed_2, is_save_perturbed_3, is_save_perturbed_4 = False, False, False, False

			'''validate'''
			perturbed_x_min, perturbed_y_min, perturbed_x_max, perturbed_y_max = -1, -1, self.new_shape[0], self.new_shape[1]

			for x in range(self.new_shape[0]//2, perturbed_x_max):
				if np.sum(synthesis_perturbed_img[x, :]) == 771*self.new_shape[1] and perturbed_x_max-1 > x:
					perturbed_x_max = x
					is_save_perturbed_1 = True
					break
			for x in range(self.new_shape[0]//2, perturbed_x_min, -1):
				if np.sum(synthesis_perturbed_img[x, :]) == 771*self.new_shape[1] and x > 0:
					perturbed_x_min = x
					is_save_perturbed_2 = True
					break
			for y in range(self.new_shape[1]//2, perturbed_y_max):
				if np.sum(synthesis_perturbed_img[:, y]) == 771*self.new_shape[0] and perturbed_y_max-1 > y:
					perturbed_y_max = y
					is_save_perturbed_3 = True
					break
			for y in range(self.new_shape[1]//2, perturbed_y_min, -1):
				if np.sum(synthesis_perturbed_img[:, y]) == 771*self.new_shape[0] and y > 0:
					perturbed_y_min = y
					is_save_perturbed_4 = True
					break

			if is_save_perturbed_1 and is_save_perturbed_2 and is_save_perturbed_3 and is_save_perturbed_4:
				is_save_perturbed = True
			else:
				# print(1)
				continue

			if perturbed_y_min <= 0 or perturbed_y_max >= self.new_shape[1]-1 or perturbed_x_min <= 0 or perturbed_x_max >= self.new_shape[0]-1:
				is_save_perturbed = False
				# print(2)
				continue

			if perturbed_y_max - perturbed_y_min <= 1 or perturbed_x_max - perturbed_x_min <= 1:
				is_save_perturbed = False
				fail_perturbed_time += 1
				# print(2)
				continue

			mesh_0_b = int(round(im_lr*0.2))
			mesh_1_b = int(round(im_ud*0.2))
			mesh_0_s = int(round(im_lr*0.1))
			mesh_1_s = int(round(im_ud*0.1))


			if ((perturbed_x_max-perturbed_x_min) < (mesh_shape_[0]-mesh_0_s) or (perturbed_y_max-perturbed_y_min) < (mesh_shape_[1]-mesh_1_s) or (perturbed_x_max-perturbed_x_min) > (mesh_shape_[0]+mesh_0_b) or (perturbed_y_max-perturbed_y_min) > (mesh_shape_[1]+mesh_1_b)):
				is_save_perturbed = False
				# print(3)
				continue

			if is_save_perturbed:

				self.synthesis_perturbed_img = np.full_like(self.synthesis_perturbed_img, 257,
											  dtype=np.int16)  # np.zeros_like(curve_bg_img)
				self.synthesis_perturbed_label = np.zeros((self.new_shape[0], self.new_shape[1], 2))

				synthesis_perturbed_img_repeat = synthesis_perturbed_img[perturbed_x_min:perturbed_x_max, perturbed_y_min:perturbed_y_max, :].copy()
				synthesis_perturbed_label_repeat = synthesis_perturbed_label[perturbed_x_min:perturbed_x_max, perturbed_y_min:perturbed_y_max, :].copy()
				self.synthesis_perturbed_label = np.zeros((self.new_shape[0], self.new_shape[1], 2))
				# if im_lr > im_ud:
				# if repeat_i < repeat_time-1 and (repeat_i%3 == 0 or repeat_i%4 == 0 or repeat_i == repeat_time-2):
				if perturbed_x_max-perturbed_x_min > save_img_shape[0] or perturbed_y_max-perturbed_y_min > save_img_shape[1]:
					synthesis_perturbed_img_repeat = cv2.resize(synthesis_perturbed_img_repeat, (im_ud, im_lr), interpolation=cv2.INTER_NEAREST)
					synthesis_perturbed_label_repeat = cv2.resize(synthesis_perturbed_label_repeat, (im_ud, im_lr), interpolation=cv2.INTER_NEAREST)
					self.perturbed_x_min, self.perturbed_y_min, self.perturbed_x_max, self.perturbed_y_max = self.adjust_position(0, 0, im_lr, im_ud)

				else:
					self.perturbed_x_min, self.perturbed_y_min, self.perturbed_x_max, self.perturbed_y_max = self.adjust_position(perturbed_x_min, perturbed_y_min, perturbed_x_max, perturbed_y_max)

				# self.synthesis_perturbed_img = self.new_origin_img_for_perturbed

				self.synthesis_perturbed_img[self.perturbed_x_min:self.perturbed_x_max, self.perturbed_y_min:self.perturbed_y_max] = synthesis_perturbed_img_repeat
				self.synthesis_perturbed_label[self.perturbed_x_min:self.perturbed_x_max, self.perturbed_y_min:self.perturbed_y_max] = synthesis_perturbed_label_repeat
				self.perturbed_xy_ = perturbed_xy_.copy()
				# min_xy_synthesis_perturbed_label_clip = np.mean(self.synthesis_perturbed_label[np.sum(self.synthesis_perturbed_img, 2) != 771], 0)
				perturbed_time += 1

				# self.perturbed_x_min, self.perturbed_y_min, self.perturbed_x_max, self.perturbed_y_max = perturbed_x_min, perturbed_y_min, perturbed_x_max, perturbed_y_max

		if fail_perturbed_time == repeat_time:
			raise Exception('clip error')
			# print(2)
		'''perspective'''

		perspective_shreshold = 280#240
		x_min_per, y_min_per, x_max_per, y_max_per = self.adjust_position(perspective_shreshold, perspective_shreshold, self.new_shape[0]-perspective_shreshold, self.new_shape[1]-perspective_shreshold)
		pts1 = np.float32([[x_min_per, y_min_per], [x_max_per, y_min_per], [x_min_per, y_max_per], [x_max_per, y_max_per]])
		e_1_ = x_max_per - x_min_per
		e_2_ = y_max_per - y_min_per
		e_3_ = e_2_
		e_4_ = e_1_
		perspective_shreshold_h = e_1_*0.02
		perspective_shreshold_w = e_2_*0.02

		if fold_curve == 'curve' and self.is_perform(0.2, 0.8):
			while True:
				pts2 = np.around(np.float32([[x_min_per+(random.random()-1)*perspective_shreshold, y_min_per+(random.random()-0.5)*perspective_shreshold],
								 [x_max_per+(random.random()-1)*perspective_shreshold, y_min_per+(random.random()-0.5)*perspective_shreshold],
								 [x_min_per+(random.random())*perspective_shreshold, y_max_per+(random.random()-0.5)*perspective_shreshold],
								 [x_max_per+(random.random())*perspective_shreshold, y_max_per+(random.random()-0.5)*perspective_shreshold]]))
				e_1 = np.linalg.norm(pts2[0]-pts2[1])
				e_2 = np.linalg.norm(pts2[0]-pts2[2])
				e_3 = np.linalg.norm(pts2[1]-pts2[3])
				e_4 = np.linalg.norm(pts2[2]-pts2[3])
				if e_1_+perspective_shreshold_h > e_1 and e_2_+perspective_shreshold_w > e_2 and e_3_+perspective_shreshold_w > e_3 and e_4_+perspective_shreshold_h > e_4 and \
					e_1_ - perspective_shreshold_h < e_1 and e_2_ - perspective_shreshold_w < e_2 and e_3_ - perspective_shreshold_w < e_3 and e_4_ - perspective_shreshold_h < e_4 and \
					abs(e_1-e_4) < perspective_shreshold_h and abs(e_2-e_3) < perspective_shreshold_w:
					break
		else:
			while True:
				pts2 = np.around(np.float32([[x_min_per+(random.random()-0.5)*perspective_shreshold, y_min_per+(random.random()-0.5)*perspective_shreshold],
								 [x_max_per+(random.random()-0.5)*perspective_shreshold, y_min_per+(random.random()-0.5)*perspective_shreshold],
								 [x_min_per+(random.random()-0.5)*perspective_shreshold, y_max_per+(random.random()-0.5)*perspective_shreshold],
								 [x_max_per+(random.random()-0.5)*perspective_shreshold, y_max_per+(random.random()-0.5)*perspective_shreshold]]))
				e_1 = np.linalg.norm(pts2[0]-pts2[1])
				e_2 = np.linalg.norm(pts2[0]-pts2[2])
				e_3 = np.linalg.norm(pts2[1]-pts2[3])
				e_4 = np.linalg.norm(pts2[2]-pts2[3])
				if e_1_+perspective_shreshold_h > e_1 and e_2_+perspective_shreshold_w > e_2 and e_3_+perspective_shreshold_w > e_3 and e_4_+perspective_shreshold_h > e_4 and \
					e_1_ - perspective_shreshold_h < e_1 and e_2_ - perspective_shreshold_w < e_2 and e_3_ - perspective_shreshold_w < e_3 and e_4_ - perspective_shreshold_h < e_4 and \
					abs(e_1-e_4) < perspective_shreshold_h and abs(e_2-e_3) < perspective_shreshold_w:
					break

		M = cv2.getPerspectiveTransform(pts1, pts2)
		one = np.ones((self.new_shape[0], self.new_shape[1], 1), dtype=np.int16)
		matr = np.dstack((pixel_position, one))
		new = np.dot(M, matr.reshape(-1, 3).T).T.reshape(self.new_shape[0], self.new_shape[1], 3)
		x = new[:, :, 0]/new[:, :, 2]
		y = new[:, :, 1]/new[:, :, 2]
		perturbed_xy_round_int = np.dstack((x, y))
		# perturbed_xy_round_int = np.around(cv2.bilateralFilter(perturbed_xy_round_int, 9, 75, 75))
		perturbed_xy_round_int = np.around(cv2.blur(perturbed_xy_round_int, (17, 17)))
		# perturbed_xy_round_int = cv2.blur(perturbed_xy_round_int, (17, 17))
		# perturbed_xy_round_int = cv2.GaussianBlur(perturbed_xy_round_int, (7, 7), 0)
		perturbed_xy_round_int = np.around(perturbed_xy_round_int-np.min(perturbed_xy_round_int.T.reshape(2, -1), 1)).astype(np.int16)

		synthesis_perturbed_img = np.full_like(self.synthesis_perturbed_img, 257, dtype=np.int16)
		synthesis_perturbed_label = np.zeros_like(self.synthesis_perturbed_label)

		it_r_i_0 = np.nditer(perturbed_xy_round_int[:, :, 0], flags=['multi_index'])
		it_r_i_1 = np.nditer(perturbed_xy_round_int[:, :, 1], flags=['multi_index'])

		while not it_r_i_0.finished:
			try:
				synthesis_perturbed_img[it_r_i_0.multi_index] = self.synthesis_perturbed_img[it_r_i_0[0], it_r_i_1[0]]
			except:
				it_r_i_0.iternext()
				it_r_i_1.iternext()
				continue
			synthesis_perturbed_label[it_r_i_0.multi_index] = self.synthesis_perturbed_label[it_r_i_0[0], it_r_i_1[0]]
			it_r_i_0.iternext()
			it_r_i_1.iternext()

		is_save_perspective_1, is_save_perspective_2, is_save_perspective_3, is_save_perspective_4 = False, False, False, False
		perspective_x_min, perspective_y_min, perspective_x_max, perspective_y_max = -1, -1, self.new_shape[0], self.new_shape[1]
		for x in range(self.new_shape[0] // 2, perspective_x_max):
			if np.sum(synthesis_perturbed_img[x, :]) == 771 * self.new_shape[1] and perspective_x_max - 1 > x:
				perspective_x_max = x
				is_save_perspective_1 = True
				break
		for x in range(self.new_shape[0] // 2, perspective_x_min, -1):
			if np.sum(synthesis_perturbed_img[x, :]) == 771 * self.new_shape[1] and x > 0:
				perspective_x_min = x
				is_save_perspective_2 = True
				break
		for y in range(self.new_shape[1] // 2, perspective_y_max):
			if np.sum(synthesis_perturbed_img[:, y]) == 771 * self.new_shape[0] and perspective_y_max - 1 > y:
				perspective_y_max = y
				is_save_perspective_3 = True
				break
		for y in range(self.new_shape[1] // 2, perspective_y_min, -1):
			if np.sum(synthesis_perturbed_img[:, y]) == 771 * self.new_shape[0] and y > 0:
				perspective_y_min = y
				is_save_perspective_4 = True
				break
		is_save_perspective = False
		if is_save_perspective_1 and is_save_perspective_2 and is_save_perspective_3 and is_save_perspective_4:
			is_save_perspective = True
		if perspective_y_min <= 0 or perspective_y_max >= self.new_shape[1]-1 or perspective_x_min <= 0 or perspective_x_max >= self.new_shape[0]-1:
			is_save_perspective = False
		if perspective_y_max - perspective_y_min <= 1 or perspective_x_max - perspective_x_min <= 1:
			is_save_perspective = False

		if ((perspective_x_max-perspective_x_min) < (mesh_shape_[0]-mesh_0_s) or (perspective_y_max-perspective_y_min) < (mesh_shape_[1]-mesh_1_s) or (perspective_x_max-perspective_x_min) > (mesh_shape_[0]+mesh_0_b) or (perspective_y_max-perspective_y_min) > (mesh_shape_[1]+mesh_1_b)):
			is_save_perspective = False

		if is_save_perspective:

			self.synthesis_perturbed_img = np.full_like(self.synthesis_perturbed_img, 257,
														dtype=np.int16)  # np.zeros_like(curve_bg_img)
			self.synthesis_perturbed_label = np.zeros((self.new_shape[0], self.new_shape[1], 2))
			synthesis_perturbed_img_repeat = synthesis_perturbed_img[perspective_x_min:perspective_x_max,
											 perspective_y_min:perspective_y_max, :].copy()
			synthesis_perturbed_label_repeat = synthesis_perturbed_label[perspective_x_min:perspective_x_max,
											   perspective_y_min:perspective_y_max, :].copy()
			self.synthesis_perturbed_label = np.zeros((self.new_shape[0], self.new_shape[1], 2))
			if perspective_x_max - perspective_x_min > save_img_shape[0] or perspective_y_max - perspective_y_min > \
					save_img_shape[1]:
				synthesis_perturbed_img_repeat = cv2.resize(synthesis_perturbed_img_repeat, (im_ud, im_lr),
															interpolation=cv2.INTER_NEAREST)
				synthesis_perturbed_label_repeat = cv2.resize(synthesis_perturbed_label_repeat, (im_ud, im_lr),
															  interpolation=cv2.INTER_NEAREST)
				self.perturbed_x_min, self.perturbed_y_min, self.perturbed_x_max, self.perturbed_y_max = self.adjust_position(
					0, 0, im_lr, im_ud)
				print(1)
			else:
				self.perturbed_x_min, self.perturbed_y_min, self.perturbed_x_max, self.perturbed_y_max = self.adjust_position(
					perspective_x_min, perspective_y_min, perspective_x_max, perspective_y_max)

			self.synthesis_perturbed_img[self.perturbed_x_min:self.perturbed_x_max,
			self.perturbed_y_min:self.perturbed_y_max] = synthesis_perturbed_img_repeat
			self.synthesis_perturbed_label[self.perturbed_x_min:self.perturbed_x_max,
			self.perturbed_y_min:self.perturbed_y_max] = synthesis_perturbed_label_repeat
			# cv2.imwrite(self.save_path + 'grey_im/' + fold_curve + '000051.png', synthesis_perturbed_img)
			# cv2.imwrite(self.save_path + 'grey_im/' + fold_curve + '000052.png', self.synthesis_perturbed_img)
		'''perspective end'''


		'''clip'''
		perfix_ = self.save_suffix+'_'+str(m)+'_'+str(n)

		if not is_save_perturbed and perturbed_time == 0:
			raise Exception('clip error')
		else:
			is_save_perturbed = True

		if is_save_perturbed:
			self.new_shape = save_img_shape

			synthesis_perturbed_img = self.synthesis_perturbed_img[self.perturbed_x_min:self.perturbed_x_max, self.perturbed_y_min:self.perturbed_y_max, :].copy()
			synthesis_perturbed_label = self.synthesis_perturbed_label[self.perturbed_x_min:self.perturbed_x_max, self.perturbed_y_min:self.perturbed_y_max, :].copy()

			self.perturbed_x_min, self.perturbed_y_min, self.perturbed_x_max, self.perturbed_y_max = self.adjust_position(self.perturbed_x_min, self.perturbed_y_min, self.perturbed_x_max, self.perturbed_y_max)
			perturbed_x_min, perturbed_y_min, perturbed_x_max, perturbed_y_max = self.perturbed_x_min, self.perturbed_y_min, self.perturbed_x_max, self.perturbed_y_max
			# reduce_value_x = int(round(min((random.random()/2)*(self.new_shape[0]-(self.perturbed_x_max-self.perturbed_x_min)), random.choice([reduce_value, reduce_value_v2]))))
			# reduce_value_y = int(round(min((random.random()/2)*(self.new_shape[1]-(self.perturbed_y_max-self.perturbed_y_min)), random.choice([reduce_value, reduce_value_v2]))))
			# reduce_value_x = int(round(min((random.random()/2)*(self.new_shape[0]-(self.perturbed_x_max-self.perturbed_x_min)), min(reduce_value, reduce_value_v2)/2)))
			# reduce_value_y = int(round(min((random.random()/2)*(self.new_shape[1]-(self.perturbed_y_max-self.perturbed_y_min)), min(reduce_value, reduce_value_v2)/2)))
			reduce_value_x = int(round(min((random.random()/2)*(self.new_shape[0]-(self.perturbed_x_max-self.perturbed_x_min)), min(reduce_value, reduce_value_v2))))
			reduce_value_y = int(round(min((random.random()/2)*(self.new_shape[1]-(self.perturbed_y_max-self.perturbed_y_min)), min(reduce_value, reduce_value_v2))))
			perturbed_x_min = max(perturbed_x_min-reduce_value_x, 0)
			perturbed_x_max = min(perturbed_x_max+reduce_value_x, self.new_shape[0])
			perturbed_y_min = max(perturbed_y_min-reduce_value_y, 0)
			perturbed_y_max = min(perturbed_y_max+reduce_value_y, self.new_shape[1])

			self.synthesis_perturbed_img = np.full((self.new_shape[0], self.new_shape[1], 3), 257, dtype=np.int16)
			self.synthesis_perturbed_label = np.zeros((self.new_shape[0], self.new_shape[1], 2))
			self.synthesis_perturbed_img[self.perturbed_x_min:self.perturbed_x_max, self.perturbed_y_min:self.perturbed_y_max, :] = synthesis_perturbed_img
			self.synthesis_perturbed_label[self.perturbed_x_min:self.perturbed_x_max, self.perturbed_y_min:self.perturbed_y_max, :] = synthesis_perturbed_label

			pixel_position = np.argwhere(np.zeros(self.new_shape, dtype=np.uint32) == 0).reshape(self.new_shape[0], self.new_shape[1], 2)

			if relativeShift_position == 'relativeShift_v2':
				self.synthesis_perturbed_label -= pixel_position

			'''resize
			if im_lr > im_ud and (self.perturbed_x_max-self.perturbed_x_min) > 0 and (self.perturbed_y_max-self.perturbed_y_min) > 0 and self.is_perform(0, 1):
				synthesis_perturbed_img_clip_resize = self.synthesis_perturbed_img[self.perturbed_x_min:self.perturbed_x_max, self.perturbed_y_min:self.perturbed_y_max, :].copy()
				synthesis_perturbed_img_clip_resize_shape_ = synthesis_perturbed_img_clip_resize.shape[:2]
				perturbed_margin = np.random.choice([16, 20, 28, 32], p=[0.6, 0.2, 0.1, 0.1])
				mesh_0_ = self.new_shape[0]-synthesis_perturbed_img_clip_resize_shape_[0]-perturbed_margin
				# mesh_shape = [base_img_shrink, base_img_shrink]
				if aspect_ratio > 1.3:
					mesh_1_ = self.new_shape[1]-synthesis_perturbed_img_clip_resize_shape_[1]-perturbed_margin
				else:
					mesh_1_ = int(round(mesh_0_ // aspect_ratio))
				synthesis_perturbed_img_clip_resize = cv2.resize(synthesis_perturbed_img_clip_resize, (synthesis_perturbed_img_clip_resize_shape_[1]+mesh_1_, synthesis_perturbed_img_clip_resize_shape_[0]+mesh_0_), interpolation=cv2.INTER_NEAREST)
				if synthesis_perturbed_img_clip_resize.shape[:2] < self.synthesis_perturbed_img.shape[:2]:
					synthesis_perturbed_img_clip_resize_shape = synthesis_perturbed_img_clip_resize.shape[:2]
					if (self.new_shape[0] - synthesis_perturbed_img_clip_resize_shape[0])%2 == 0:
						synthesis_perturbed_img_clip_resize_l = (self.new_shape[0] - synthesis_perturbed_img_clip_resize_shape[0])//2
						synthesis_perturbed_img_clip_resize_r = synthesis_perturbed_img_clip_resize_l
					else:
						synthesis_perturbed_img_clip_resize_l = (self.new_shape[0] - synthesis_perturbed_img_clip_resize_shape[0]) // 2
						synthesis_perturbed_img_clip_resize_r = synthesis_perturbed_img_clip_resize_l+1
	
					if (self.new_shape[1] - synthesis_perturbed_img_clip_resize_shape[1])%2 == 0:
						synthesis_perturbed_img_clip_resize_u = (self.new_shape[1] - synthesis_perturbed_img_clip_resize_shape[1])//2
						synthesis_perturbed_img_clip_resize_d = synthesis_perturbed_img_clip_resize_u
					else:
						synthesis_perturbed_img_clip_resize_u = (self.new_shape[1] - synthesis_perturbed_img_clip_resize_shape[1]) // 2
						synthesis_perturbed_img_clip_resize_d = synthesis_perturbed_img_clip_resize_u+1
	
					if synthesis_perturbed_img_clip_resize_l > 0 and synthesis_perturbed_img_clip_resize_r > 0 and synthesis_perturbed_img_clip_resize_u > 0 and synthesis_perturbed_img_clip_resize_d > 0:
						self.synthesis_perturbed_img = np.full_like(self.synthesis_perturbed_img, 257, dtype=np.int16)
						self.synthesis_perturbed_img[synthesis_perturbed_img_clip_resize_l:self.new_shape[0]-synthesis_perturbed_img_clip_resize_r, synthesis_perturbed_img_clip_resize_u:self.new_shape[1]-synthesis_perturbed_img_clip_resize_d, :] = synthesis_perturbed_img_clip_resize
	
						synthesis_perturbed_label_clip_resize = self.synthesis_perturbed_label[self.perturbed_x_min:self.perturbed_x_max,
														   self.perturbed_y_min:self.perturbed_y_max, :].copy()
						self.synthesis_perturbed_label = np.zeros_like(self.synthesis_perturbed_label)
						synthesis_perturbed_label_clip_resize = cv2.resize(synthesis_perturbed_label_clip_resize, (synthesis_perturbed_img_clip_resize_shape_[1]+mesh_1_, synthesis_perturbed_img_clip_resize_shape_[0]+mesh_0_), interpolation=cv2.INTER_NEAREST)
						self.synthesis_perturbed_label[synthesis_perturbed_img_clip_resize_l:self.new_shape[0]-synthesis_perturbed_img_clip_resize_r, synthesis_perturbed_img_clip_resize_u:self.new_shape[1]-synthesis_perturbed_img_clip_resize_d, :] = synthesis_perturbed_label_clip_resize

			'''
			''''''
			if np.sum(self.synthesis_perturbed_img[:, 0]) != 771 * self.new_shape[0] or np.sum(self.synthesis_perturbed_img[:, self.new_shape[1]-1]) != 771 * self.new_shape[0] or \
					np.sum(self.synthesis_perturbed_img[0, :]) != 771 * self.new_shape[1] or np.sum(self.synthesis_perturbed_img[self.new_shape[0]-1, :]) != 771*self.new_shape[1]:
				# raise Exception('clip error')
				is_save_perturbed = False

			if is_save_perturbed:
				label = np.zeros_like(self.synthesis_perturbed_img)
				foreORbackground_label = np.ones(self.new_shape, dtype=np.int16)

				self.synthesis_perturbed_label[np.sum(self.synthesis_perturbed_img, 2) == 771] = 0
				foreORbackground_label[np.sum(self.synthesis_perturbed_img, 2) == 771] = 0
				label[:, :, :2] = self.synthesis_perturbed_label
				label[:, :, 2] = foreORbackground_label

				'''HSV'''
				if self.is_perform(0.1, 0.9):
					if self.is_perform(0.2, 0.8):
						synthesis_perturbed_img_clip_HSV = self.synthesis_perturbed_img.copy().astype(np.float32)
						synthesis_perturbed_img_clip_HSV = cv2.cvtColor(synthesis_perturbed_img_clip_HSV, cv2.COLOR_RGB2HSV)
						H_, S_, V_ = (random.random()-0.2)*20, (random.random()-0.2)/8, (random.random()-0.2)*20
						synthesis_perturbed_img_clip_HSV[:, :, 0], synthesis_perturbed_img_clip_HSV[:, :, 1], synthesis_perturbed_img_clip_HSV[:, :, 2] = synthesis_perturbed_img_clip_HSV[:, :, 0]-H_, synthesis_perturbed_img_clip_HSV[:, :, 1]-S_, synthesis_perturbed_img_clip_HSV[:, :, 2]-V_
						synthesis_perturbed_img_clip_HSV = cv2.cvtColor(synthesis_perturbed_img_clip_HSV, cv2.COLOR_HSV2RGB).astype(np.int16)
						synthesis_perturbed_img_clip_HSV[np.sum(self.synthesis_perturbed_img, 2) == 771] = perturbed_bg_img[np.sum(self.synthesis_perturbed_img, 2) == 771]
						self.synthesis_perturbed_img = synthesis_perturbed_img_clip_HSV
					else:
						perturbed_bg_img_HSV = perturbed_bg_img.astype(np.float32)
						perturbed_bg_img_HSV = cv2.cvtColor(perturbed_bg_img_HSV, cv2.COLOR_RGB2HSV)
						H_, S_, V_ = (random.random()-0.5)*20, (random.random()-0.5)/8, (random.random()-0.2)*20
						perturbed_bg_img_HSV[:, :, 0], perturbed_bg_img_HSV[:, :, 1], perturbed_bg_img_HSV[:, :, 2] = perturbed_bg_img_HSV[:, :, 0]-H_, perturbed_bg_img_HSV[:, :, 1]-S_, perturbed_bg_img_HSV[:, :, 2]-V_
						perturbed_bg_img_HSV = cv2.cvtColor(perturbed_bg_img_HSV, cv2.COLOR_HSV2RGB).astype(np.int16)
						self.synthesis_perturbed_img[np.sum(self.synthesis_perturbed_img, 2) == 771] = perturbed_bg_img_HSV[np.sum(self.synthesis_perturbed_img, 2) == 771]

				else:
					synthesis_perturbed_img_clip_HSV = self.synthesis_perturbed_img.copy().astype(np.float32)
					synthesis_perturbed_img_clip_HSV[np.sum(self.synthesis_perturbed_img, 2) == 771] = perturbed_bg_img[np.sum(self.synthesis_perturbed_img, 2) == 771]
					synthesis_perturbed_img_clip_HSV = cv2.cvtColor(synthesis_perturbed_img_clip_HSV, cv2.COLOR_RGB2HSV)
					H_, S_, V_ = (random.random()-0.5)*20, (random.random()-0.5)/10, (random.random()-0.4)*20
					synthesis_perturbed_img_clip_HSV[:, :, 0], synthesis_perturbed_img_clip_HSV[:, :, 1], synthesis_perturbed_img_clip_HSV[:, :, 2] = synthesis_perturbed_img_clip_HSV[:, :, 0]-H_, synthesis_perturbed_img_clip_HSV[:, :, 1]-S_, synthesis_perturbed_img_clip_HSV[:, :, 2]-V_
					synthesis_perturbed_img_clip_HSV = cv2.cvtColor(synthesis_perturbed_img_clip_HSV, cv2.COLOR_HSV2RGB).astype(np.int16)
					self.synthesis_perturbed_img = synthesis_perturbed_img_clip_HSV

				synthesis_perturbed_img = np.zeros_like(self.synthesis_perturbed_img, dtype=np.int16)
				if im_lr >= im_ud:
					synthesis_perturbed_img[:, perturbed_y_min:perturbed_y_max, :] = self.synthesis_perturbed_img[:, perturbed_y_min:perturbed_y_max, :]
				else:
					synthesis_perturbed_img[perturbed_x_min:perturbed_x_max, :, :] = self.synthesis_perturbed_img[perturbed_x_min:perturbed_x_max, :, :]
				self.synthesis_perturbed_img = synthesis_perturbed_img

				'''add background
				synthesis_perturbed_img_clip[np.sum(synthesis_perturbed_img_clip, 2) == 771] = perturbed_bg_img[np.sum(synthesis_perturbed_img_clip, 2) == 771]
				'''
				# synthesis_perturbed_img_clip = np.concatenate(
				# 	(synthesis_perturbed_img_clip, groun_truth), axis=1)
				''''''
				self.synthesis_perturbed_img[self.synthesis_perturbed_img > 255] = 255
				self.synthesis_perturbed_img[self.synthesis_perturbed_img < 0] = 0

				if is_save_perspective:
					print(str(perturbed_time)+'-'+str(repeat_time)+'  perspective')
				else:
					print(str(perturbed_time)+'-'+str(repeat_time))

				cv2.imwrite(self.save_path + 'png/' + perfix_ + '_' + fold_curve + '.png', self.synthesis_perturbed_img)
				# grey = np.around(self.synthesis_perturbed_img[:, :, 0] * 0.2989 + self.synthesis_perturbed_img[:, :, 1] * 0.5870 + self.synthesis_perturbed_img[:, :, 0] * 0.1140).astype(np.int16)
				# synthesis_perturbed_grey = np.concatenate((grey.reshape(self.new_shape[0], self.new_shape[1], 1), label), axis=2)
				synthesis_perturbed_color = np.concatenate((self.synthesis_perturbed_img, label), axis=2)
				with open(self.save_path+'color/'+perfix_+'_'+fold_curve+'.gw', 'wb') as f:
					pickle_perturbed_data = pickle.dumps(synthesis_perturbed_color)
					f.write(pickle_perturbed_data)


		if not is_save_perturbed:
			print('save error')
		else:
			cv2.imwrite(self.save_path + 'scan/' + self.save_suffix + '_' + str(m) + '.png', self.origin_img)
			trian_t = time.time() - begin_train
			mm, ss = divmod(trian_t, 60)
			hh, mm = divmod(mm, 60)
			print(str(m)+'_'+str(n)+'_'+fold_curve+" Time : %02d:%02d:%02d\n" % (hh, mm, ss))

		'''draw'''
		# draw_distance_hotmap(np.abs(distance_vertex_and_line))
		# draw_distance_hotmap(perturbed_d)
		# draw_distance_hotmap(curve_d)
		# draw_distance_hotmap(omega_perturbed)
		# draw_distance_hotmap(omega_curve)

def multiThread(m, n, img_path_, bg_path_, save_path, save_suffix):
	saveFold = perturbed(img_path_, bg_path_, save_path, save_suffix)
	saveCurve = perturbed(img_path_, bg_path_, save_path, save_suffix)

	repeat_time = min(max(round(np.random.normal(10, 3)), 5), 16)
	fold = threading.Thread(target=saveFold.save_img, args=(m, n, 'fold', repeat_time, 'relativeShift_v2'), name='fold')
	curve = threading.Thread(target=saveCurve.save_img, args=(m, n, 'curve', repeat_time, 'relativeShift_v2'), name='curve')

	fold.start()
	curve.start()
	curve.join()
	fold.join()

def xgw(args):
	path = args.path
	bg_path = args.bg_path
	if not os.path.exists(path):
		raise Exception('-- No path')
	if not os.path.exists(bg_path):
		raise Exception('-- No bg_path')	
		
	if args.output_path is None:
		save_path = '/lustre/home/gwxie/data/unwarp_new/train/data1024_greyV2/'
	else:
		save_path = args.output_path

	# if not os.path.exists(save_path + 'clip/'):
	# 	os.makedirs(save_path + 'clip/')
	#
	#if not os.path.exists(save_path + 'grey/'):
	#	os.makedirs(save_path + 'grey/')
	if not os.path.exists(save_path + 'color/'):
		os.makedirs(save_path + 'color/')

	# if not os.path.exists(save_path + 'grey_im/'):
	# 	os.makedirs(save_path + 'grey_im/')
	#
	if not os.path.exists(save_path + 'png/'):
		os.makedirs(save_path + 'png/')

	if not os.path.exists(save_path + 'scan/'):
		os.makedirs(save_path + 'scan/')

	if not os.path.exists(save_path + 'outputs/'):
		os.makedirs(save_path + 'outputs/')

	save_suffix = str.split(args.path, '/')[-2]
	
	all_img_path = getDatasets(path)
	all_bgImg_path = getDatasets(bg_path)
	global begin_train
	begin_train = time.time()

	process_pool = Pool(2)
	for m, img_path in enumerate(all_img_path):
		for n in range(args.sys_num):
			img_path_ = path+img_path
			bg_path_ = bg_path+random.choice(all_bgImg_path)+'/'

			for m_n in range(10):
				try:
					saveFold = perturbed(img_path_, bg_path_, save_path, save_suffix)
					saveCurve = perturbed(img_path_, bg_path_, save_path, save_suffix)

					repeat_time = min(max(round(np.random.normal(8, 4)), 1), 12) 
					process_pool.apply_async(func=saveFold.save_img, args=(m, n, 'fold', repeat_time, 'relativeShift_v2'))

					repeat_time = min(max(round(np.random.normal(6, 4)), 1), 10)
					process_pool.apply_async(func=saveCurve.save_img, args=(m, n, 'curve', repeat_time, 'relativeShift_v2'))

				except BaseException as err:
					print(err)
					continue
				break
			# print('end')

	process_pool.close()
	process_pool.join()

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Hyperparams')
	parser.add_argument('--path',
						default='validate', type=str,
						help='the path of origin img.')
	parser.add_argument('--bg_path',
						default='validate', type=str,
						help='the path of bg img.')

	parser.add_argument('--output_path',
						default=None, type=str,
						help='the path of output img.')
	# parser.set_defaults(output_path='test')
	parser.add_argument('--count_from', '-p', default=0, type=int,
						metavar='N', help='print frequency (default: 10)')  # print frequency

	parser.add_argument('--repeat_T', default=0, type=int)

	parser.add_argument('--sys_num', default=7, type=int)

	args = parser.parse_args()
	xgw(args)
