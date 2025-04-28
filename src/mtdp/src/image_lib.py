#!/usr/bin/python
#!python
# -*- coding: utf-8 -*-
# Created on 2025/01/26
# Udated  on 2025/01/26
# @author: Flavio Lichtenstein
# @local: Bioinformatics: CENTD/Molecular Biology; Instituto Butatan


import os, sys, pickle
from typing import Optional, Iterable, Set, Tuple, Any, List

import scipy
from   scipy.stats import hypergeom

import numpy as np
import time, json
from datetime import datetime
import pandas as pd
from sklearn.utils import shuffle

import re
_RE_COMBINE_WHITESPACE = re.compile(r"\s+")

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import cv2

from IPython.display import Markdown

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import plotly.graph_objects as go
import plotly.express as px

from Basic import *
from parallel_image_lib import *

class Image(object):
	def __init__(self, root0:str, root_img:str,
				 deltax:int=250, deltay:int=250, 
				 image_size_x:int=2048, image_size_y:int=2048):

		self.root0 = root0
		self.root_img = create_dir(root0, root_img)

		root_save_img = root_img.strip()+'_segments'
		self.root_save_img  = create_dir(root0, root_save_img)

		self.image_size_x = image_size_x
		self.image_size_y = image_size_y

		self.maxi_x0 = image_size_x
		self.maxi_y0 = image_size_y

		self.maxi_x0m1 = image_size_x-1
		self.maxi_y0m1 = image_size_y-1

		self.deltax = deltax
		self.deltay = deltay

		filelog = 'image.log'
		self.set_filelog(filelog)

	def read_img(self, fname_img:str, verbose:bool=False):
		fileimg = os.path.join(self.root_img, fname_img)
		
		try:
			print(f"reading: '{fileimg}'", end=' ')
			img = plt.imread(fileimg)
			print("ok")
		except:
			print(f"could not read: {'fileimg'}")
			img = None

		self.img = img
		return img


	def read_segmented_img(self, fname_img:str, verbose:bool=False):
		fileimg = os.path.join(self.root_save_img, fname_img)
		
		try:
			if verbose: print(f"reading segmented: '{fileimg}'")
			img = plt.imread(fileimg)
		except:
			print(f"could not read: {'fileimg'}")
			img = None

		self.img = img
		return img


	def display_img(self, img:List, cmap=None, figsize:tuple=(8,8),
					left:int=0, bottom:int=0, right:int=1, top:int=1, wspace:int=0, hspace:int=0):
		fig = plt.figure(figsize=figsize)
		ax = fig.add_subplot(111)

		ax.imshow(img,cmap)

		# ax.set_xticks([])
		# ax.set_yticks([])

		# ax.set_xticklabels([])
		# ax.set_yticklabels([])

		plt.axis('off')
		plt.tight_layout()
		plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
		plt.show()

		return fig, ax


	def convert_img_to_gray(self, img):
	 	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	
	def reduce_to_3_gray_patterns(self, img, Ninf:int, Nmid_val:int, Nmid:int, Nmax:int):
	 	return np.array([ [0 if x < Ninf else Nmid_val if x < Nmid else Nmax for x in seq] for seq in img])

	
	def display_all_contours(self, img, cmap=plt.cm.gray, figsize=(12,8)):
		contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

		external_contours = np.zeros(img.shape)

		for i in range(len(contours)): # 
			
			# last column in the array is -1 if an external contour (no contours inside of it)
			if hierarchy[0][i][3] == -1:
				
				# We can now draw the external contours from the list of contours
				cv2.drawContours(external_contours, contours, i, 255, -1)

		fig, ax = self.display_img(external_contours, cmap=cmap, figsize=figsize)
		return fig, ax, contours, hierarchy


	def define_xy_min_max(self, contours_i):
		# max has +1
		x_min, y_min, delx, dely = cv2.boundingRect(contours_i)

		return x_min, x_min+delx, y_min, y_min+dely


	def define_xy_min_max_loop(self, contours_i):
		
		x_min, x_max, y_min, y_max = self.maxi_x0, 0, self.maxi_y0, 0

		for k in range(len(contours_i)):
			x, y = contours_i[k][0]
			
			if x < x_min:
				x_min = x
			if x > x_max:
				x_max = x
		
			if y < y_min:
				y_min = y
			if y > y_max:
				y_max = y
		
		return x_min, x_max, y_min, y_max


	def define_quad(self, contours_i):
		x_min, x_max, y_min, y_max = self.maxi_x0m1, 0, self.maxi_y0m1, 0
		
		for k in range(len(contours_i)):
			# for a gray scale image
			x, y = contours_i[k][0]
			
			if x < x_min:
				x_min = x
			if x > x_max:
				x_max = x
		
			if y < y_min:
				y_min = y
			if y > y_max:
				y_max = y
		
		return [ [x_min,y_min], [x_max, y_min], [x_min, y_max], [x_max,y_max]]

	def calc_area(self, y_min, y_max, x_min, x_max):
		area = (x_max-x_min) * (y_max-y_min)
		return area if area >= 0 else -area

	def calc_2_overlaps(self, coord1, coord2, verbose:bool=False):

		y_min1, y_max1, x_min1, x_max1 = coord1
		y_min2, y_max2, x_min2, x_max2 = coord2

		if verbose:
			print(y_min1, y_max1, x_min1, x_max1)
			print(y_min2, y_max2, x_min2, x_max2)
		
		if y_max2 < y_min1:
			if verbose: print('before y')
			return 0,0
			
		if y_min2 > y_max1:
			if verbose: print('after y')
			return 0,0
		
		if x_max2 < x_min1:
			if verbose: print('before x')
			return 0,0
		
		if x_min2 > x_max1:
			if verbose:print('after x')
			return 0,0

		y_min = y_min1 if y_min1 >= y_min2 else y_min2
		y_max = y_max1 if y_max1 <= y_max2 else y_max2

		x_min = x_min1 if x_min1 >= x_min2 else x_min2
		x_max = x_max1 if x_max1 <= x_max2 else x_max2

		a0 = self.calc_area(y_min, y_max, x_min, x_max)
		a1 = self.calc_area(y_min1, y_max1, x_min1, x_max1)
		a2 = self.calc_area(y_min2, y_max2, x_min2, x_max2)

		if a1 <= 0:
			a1 = 1
			print("Error a1:", y_min1, y_max1, x_min1, x_max1)

		if a2 <= 0:
			a2 = 1
			print("Error a2:", y_min2, y_max2, x_min2, x_max2)

		perc1 = a0/a1
		perc2 = a0/a2

		if verbose:
			print("min:", y_min, y_max, x_min, x_max)
			print("areas", a0, a1, a2)

		return perc1, perc2

	def select_and_draw_contours(self, img:List, imgray:List, min_contours:int=50, max_contours:int=2000, 
		 						 min_area:int=100, max_area:int=90000, 
		 						 start_colors:tuple=(10, 10, 10), perc_area_threshold:float=.75, 
		 						 font:int=cv2.FONT_HERSHEY_PLAIN, color_text:tuple=(225, 60, 10),
		 						 figsize:tuple=(16,14), cmap=plt.cm.coolwarm, 
		 						 ampli_full_text:int=2, ampli_seg_text:int=4, del_text:int=30,
		 						 show_segements:bool=False, show_image:bool=True, verbose:bool=False):

		contours, hierarchy = cv2.findContours(imgray, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)

		external_contours = np.zeros(imgray.shape)

		red, green, blue = start_colors

		icount=-1; dic_img, dic_img_ori={},{}


		for i in range(len(contours)):

			# last column in the array is -1 if an external contour (no contours inside of it)
			if hierarchy[0][i][3] != -1:
				continue

			# We can now draw the external contours from the list of contours
			n = len(contours[i])

			if n >= min_contours and n <= max_contours:
				continue

			x_min, x_max, y_min, y_max = self.define_xy_min_max(contours[i])
			area = self.calc_area(y_min, y_max, x_min, x_max)
			if area < min_area or area > max_area:
				# print(f"Area(1) {area}:", y_min, y_max, x_min, x_max)
				continue

			# decrementing min, incrementing max --> build a larger area
			x0 = int((x_min+x_max)/2)
			y0 = int((y_min+y_max)/2)

			x_min2 = x0-self.deltax
			if x_min2 < 0:
				x_min2=0
				x0 += self.deltax

			x_max2 = x0+self.deltax
			if x_max2 > self.maxi_x0m1: 
				x_max2 = self.maxi_x0m1
				x_min2 = self.maxi_x0m1 - (2*self.deltax)
				if x_min2 < 0:
					x_min2 = 0

			y_min2 = y0-self.deltay
			if y_min2 < 0:
				y_min2=0
				y0 += self.deltay

			y_max2 = y0+self.deltay
			if y_max2 > self.maxi_y0m1: 
				y_max2 = self.maxi_y0m1
				y_min2 = self.maxi_y0m1 - (2*self.deltay)
				if y_min2 < 0:
					y_min2 = 0

			area = self.calc_area(y_min2, y_max2, x_min2, x_max2)
			if area < min_area:
				print(f"Area(2) {area}:", y_min2, y_max2, x_min2, x_max2)
				continue

			if icount == -1:
				icount = 0
				dic_img[icount]		= [i, y_min2, y_max2, x_min2, x_max2]
				dic_img_ori[icount] = [i, y_min,  y_max,  x_min,  x_max]
			else:
				has_overlap = False

				coord2 = [y_min2, y_max2, x_min2, x_max2]
				
				for i_img, (idummy, y_min1, y_max1, x_min1, x_max1)  in dic_img.items():

					coord1 = [y_min1, y_max1, x_min1, x_max1]
					perc1, perc2 = self.calc_2_overlaps(coord1, coord2)

					if perc1 > perc_area_threshold and perc2 > perc_area_threshold:
						# print(f"*** overlap {i} {perc1:.2f}, {perc2:.2f}")
						has_overlap = True
						break

				if has_overlap:
					# print(">>> has overlap")
					continue

				# print(">>> no overlap")

				icount += 1
				dic_img[icount]		= [i, y_min2, y_max2, x_min2, x_max2]
				dic_img_ori[icount] = [i, y_min,  y_max,  x_min,  x_max]

			red += 25
			if red > 255:
				red = 0
				green += 25

				if green > 255:
					green = 0
					blue += 25

					if blue > 255:
						blue = 0
					
			color = (red, green, blue)

			if verbose and icount == 5:
				print(">>>", icount, n, "color", color, "contours", contours[i][0][0])
				
			if show_segements:
				external_contours = np.zeros(imgray.shape)

			cv2.drawContours(external_contours, contours, i, color, -1)

			if show_image and show_segements:
				fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24,8))
				
				title = f"i {i} has {n} contours, image {icount}, X={x_min2}-{x_max2} Y={y_min2}-{y_max2}"
				plt.suptitle(title)

				crop_img = img[y_min2:y_max2, x_min2:x_max2]

				x = x_min-del_text
				y = y_min+del_text
				
				if x < del_text:
					x=del_text
				elif x >= self.maxi_x0-del_text:
					x=self.maxi_x0-del_text

				if y < del_text: 
					y = del_text
				elif y >= self.maxi_y0-del_text: 
					y=self.maxi_y0-del_text

				cv2.putText(external_contours, f'{i}',(x,y), font, ampli_seg_text, color_text,3) 

				ax[0].imshow(external_contours, cmap=plt.cm.coolwarm)
				ax[1].imshow(crop_img, cmap=plt.cm.coolwarm)


		if show_image and not show_segements:
			for key, (i, y_min, y_max, x_min, x_max) in dic_img_ori.items():

				x = x_min
				y = y_min
				
				if x < del_text:
					x=del_text
				elif x >= self.maxi_x0-del_text:
					x=self.maxi_x0-del_text

				if y < del_text: 
					y = del_text
				elif y >= self.maxi_y0-del_text: 
					y=self.maxi_y0-del_text

				cv2.putText(external_contours, f'{i}',(x,y), font, ampli_full_text, color_text,3) 

			self.display_img(external_contours, cmap=cmap, figsize=figsize)

		return dic_img, dic_img_ori, contours



	def save_all_images_multiprocess(self, perc_area_threshold:float=0.4, 
									 min_contours:int=400, max_contours:int=5000, 
									 min_area:int=300, max_area:int=600*600, 
									 ampli_full_text:int=2, ampli_seg_text:int=5, del_text:int=50,
									 figsize:tuple=(16,14), start_colors:tuple=(10, 10, 10),
									 color_text:tuple=(225, 60, 10), show_segements:bool=False,
									 Ninf:int=28, Nmid:int=80, Nmid_val:int=60, Nmax:int=127,
									 show_image:bool=False, force:bool=False, verbose:bool=False):

		files = os.listdir(self.root_img)

		for fname_img in files:
			img = self.read_img(fname_img=fname_img, verbose=True)
			print(f"shape {img.shape}")  # if verbose: 

			imgray = self.convert_img_to_gray(img)

			print("reducing grays ...")
			imgray2 = self.reduce_to_3_gray_patterns(imgray, Ninf, Nmid_val, Nmid, Nmax)


			dic_img, dic_img_ori, contours = \
			self.select_and_draw_contours(img, imgray2, min_contours=min_contours, max_contours=max_contours,
				 						  min_area=min_area, max_area=max_area,
										  start_colors=start_colors, perc_area_threshold=perc_area_threshold, 
				 						  font=cv2.FONT_HERSHEY_PLAIN, color_text=color_text,
				 						  figsize=figsize, cmap=plt.cm.coolwarm, 
										  ampli_full_text=ampli_full_text, ampli_seg_text=ampli_seg_text, del_text=del_text,
										  show_segements=show_segements, show_image=show_image, verbose=verbose)

			_ = self.save_image_parallel(dic_img=dic_img, img=img, fname_img=fname_img, process_name='save_image', cpus=6, verbose=True)

		print("\n------------------- end (final) ----------------------\n")




	def save_image_parallel(self, dic_img:dict, img:List, fname_img:str,
							process_name='save_image', cpus:int=6, 
							force:bool=False, verbose:bool=False) -> List:

		new_size = (self.deltax*2, self.deltay*2)
		self.new_size = new_size

		par = Parallel(ima=self, dic_img=dic_img, img=img, 
					   fname_img=fname_img, root_save_img=self.root_save_img,
					   process_name=process_name, cpus=cpus, new_size=new_size, verbose=verbose)

		self.par = par

		ret_list = par.run_multiprocess(force=force, verbose=verbose)

		print("\n----------------- end ------------------------\n")
		return ret_list



	def set_filelog(self, fname:str, root:str='./logs'):

		if not os.path.exists(root):
			os.mkdir(root)

		filefull = os.path.join(root, fname)
		self.filelog = filefull

		if os.path.exists(filefull):
			os.unlink(filefull)


	def log_save(self, stri:str, withtime:bool=True, 
				 shift:int=0, crBefore:bool=False, 
				 withCR:bool=True, end:str='\n', verbose=False):

		if shift > 0:
			try:
				stri = "\t"*int(shift) + stri
			except:
				pass

		if crBefore:
			stri = "\n" + stri

		if verbose:
			print(stri, end=end)

		if withtime:
			date_time = datetime.now().strftime("%H:%M:%S, %Y/%m/%d\n")
			stri += " >> " + date_time

		if withCR:
			stri += "\n"

		try:
			f = open(self.filelog,"a+")
			f.write(stri)
		except:
			print(f"Could not save on '{self.filelog}'")
		finally:
			f.close()



