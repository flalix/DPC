#!/usr/bin/python
#!python
# -*- coding: utf-8 -*-
# Created on 2010/08/20
# @self.author: Flavio Lichtenstein

import os, sys
import numpy as np
import scipy.stats

# pip install xlrd
# xlrd==1.2.0
import pandas as pd
from collections import OrderedDict

from Basic import *


class Excel(object):
	def __init__(self, root_data, nround=2):

		self.root_data = root_data
		self.nround	= nround

		if not os.path.exists(self.root_data):
			os.mkdir(root_data)


	def read_sheets(self, filename, verbose=False):
		self.filename = filename

		fullname = os.path.join(self.root_data, filename)
		self.fullname = fullname

		if not os.path.exists(fullname):
			print("Could not find '%s'"%(fullname))
			return None, None

		try:
			excel = pd.ExcelFile(fullname)
		except:
			print(f"Could not read '{fullname}' or !pip3 install openpyxl")
			return None, None

		sheets = excel.sheet_names

		if verbose: 
			print(f"Excel '{fullname}' if opened.")
			print(">>> sheets:", ",".join([x for x in sheets]))

		return excel, sheets


	def read_excel(self, sheet, nrows=None, skiprows=None, header=0, verbose=False):

		self.sheet = sheet

		try:
			df = pd.read_excel(self.fullname, sheet_name=sheet, nrows=nrows, skiprows=skiprows, header=header)
			if verbose:
				print(f"Excel '{self.fullname}' sheet {sheet} has {len(df)} lines.")
		except:
			print(f"Could not read '{self.fullname} for sheet={sheet}'")
			return None

		return df
