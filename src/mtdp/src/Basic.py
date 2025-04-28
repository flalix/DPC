#!/usr/bin/python
#!python
# -*- coding: utf-8 -*-
'''
# @Updated on 2022/12/06
# @Created on 2010/11/03
# @author: Flavio Lichtenstein
'''

import numpy as np
import os, sys, re, random
# import pickle5 as pickle
import pickle
import pandas as pd
from collections import OrderedDict
from typing import Optional, Iterable, Set, Tuple, Any, List

import zipfile, zlib

class Basic(object):
	def __init__(self):
		self.log202 = np.log10(20) / np.log10(2)
		self.log2 = np.log10(2)
		self.ln2 = np.log(2)
		self.inv_ln2 = 1./self.ln2
		self.ln2_2 = self.ln2 ** 2
		self.convShannonTsallis = 1.417199516  # =1/LN(2)

		self.proteinMaxMER = np.log2(20)
		self.dnaMaxMER = np.log2(4)

	def getDnaNucleotides(self):
		return ['A','T','G','C']

	def getDnaNucleotideString(self):
		return 'ATGC'

	def getRnaNucleotides(self):
		return ['A','U','G','C']

	def getrnaNucleotideString(self):
		return 'AUGC'

	# in Hydropathy index[95] http://en.wikipedia.org/wiki/Amino_acid
	def getSeqAA(self):
		return ['A', 'M', 'C', 'F', 'L', 'V', 'I', 'G', 'T', 'S', 'W', 'Y', 'P', 'H', 'N', 'D', 'E', 'Q', 'K', 'R']

	def getStringAA(self):
		return 'AMCFLVIGTSWYPHNDEQKR'

	def getAaPos(self, string, aa):
		return string.find(aa)

class Log_writer(object):
	""" log writer """

	def __init__(self, fileLog = "log.txt", path = "../log/", remove=True):
		# global fileLog
		# fileLog = "../results/log.txt"
		self.fileLog = fileLog
		self.pathLog = path
		filename = os.path.join(path, fileLog)

		if remove:
			if os.path.exists(filename): os.remove(filename)

	def write_log(self, text, to_append=True, verbose=False):  # encondig = 'utf-8'
		write_txt(text, self.fileLog, self.pathLog, to_append=to_append, verbose=verbose)

def simple_replace(stri):
	if stri == None: return None
	stri = stri.strip()
	if stri == "": return ""

	return stri.replace("  "," ")

def prepare_id(_id):
	return _id.split("=")[0].strip().replace('/','-').replace(' ','_')

def prepare_title(title, maxCount=12):
	if title == None: return None
	title = title.strip()
	if title == "": return "???"

	title = title.replace("/", "-").replace("_", ": ")
	words = title.split(" ")

	stri = ""; count = 0
	for word in words:
		stri += word + " "
		count += 1
		if count == maxCount:
			stri = stri.rstrip() + "\n"
			count = 0

	return stri.rstrip()

def full_replace(stri):
	if stri == None: return None
	if stri.strip() == "": return ""

	return stri.replace("  ","_").replace(" ","_").replace('/',"-").replace('[',"(").replace(']',")").replace("\n"," - ")

def full_replace_lower(stri):
	if stri == None: return None
	return full_replace(stri).lower()


def write_txt(text, filename, path="./", to_append=False, verbose=False): # encondig = 'utf-8',
	if not os.path.exists(path):
		os.mkdir(path)

	filename = os.path.join(path, filename)
	try:
		ret = True
		if to_append:
			h = open(filename, mode = "a+")
		else:
			h = open(filename, mode = "w")
		h.write(text)
	except ValueError:
		print(f"Error '{ValueError}', writing in '{filename}'")
		ret = False
	finally:
		h.close()

	if not ret:
		return False

	if verbose: print(f"File saved: {filename}")
	return True

'''
https://thispointer.com/5-different-ways-to-read-a-file-line-by-line-in-python/
'''
def read_txt(filename, path="./", iniLine=None, endLine=None, verbose=False):  # encondig = 'utf-8',
	filename = os.path.join(path, filename)
	text = []

	try:
		h = open(filename, mode="r")

		if iniLine is not None or endLine is not None:
			iniLine = -1 if iniLine is None else iniLine
			endLine = np.inf if endLine is None else endLine

			count = 0
			while True:
				if count < iniLine:
					_ = h.readline()
				else:
					line = h.readline()
					if not line : break

					text.append(line.replace('\n','') )

				count += 1
				if count > endLine: break
		else:
			text = [line.replace('\n','') for line in h.readlines()]

		h.close()
		if verbose: print("File read at '%s'"%(filename))
	except ValueError:
		print(f"Error '{ValueError}' while reading '{filename}'")

	return "\n".join(text)

# encondig = 'utf-8',
def pdreadcsv(filename, path="./", sep='\t', dtype = None, colnames = [], skiprows=None,
			  selcols = [], sortcols = [], low_memory=False, removedup = False, header='infer',
			  verbose=False):

	if not os.path.exists(path):
		print(f"Path does not exists: '{path}'")
		return None

	filename = os.path.join(path, filename)

	if not os.path.exists(filename):
		print(f"File does not exist: '{filename}'")
		return None

	try:
		if dtype is None:
			df = pd.read_csv(filename, sep=sep, low_memory=low_memory, skiprows=skiprows, header=header)  # , encondig = encondig
		else:
			df = pd.read_csv(filename, sep=sep, dtype = dtype, low_memory=low_memory, skiprows=skiprows, header=header) #, encondig = encondig

	except:
		print(f"Error reading csv/tsv '{filename}'")
		return None

	try:
		if len(colnames) > 0: df.columns = colnames
		if len(selcols)  > 0: df = df[  selcols ]
		if len(sortcols) > 0: df = df.sort_values(sortcols)
		if removedup: df = df.drop_duplicates()
	except ValueError:
		print(f"Error '{ValueError}' columns/selecting/sorting '{filename}'")
		return df

	if verbose: print("Table opened (%s) at '%s'"%(df.shape, filename))
	return df

def month_to_num(stri):
	if isinstance(stri, int): return stri
	if not isinstance(stri, str): return stri

	stri2 = stri.lower()

	if stri2 == 'jan':
		return 1
	if stri2 == 'feb' or stri2 == 'fev':
		return 2
	if stri2 == 'mar':
		return 3
	if stri2 == 'apr' or stri2 == 'abr':
		return 4
	if stri2 == 'may' or stri2 == 'mai':
		return 5
	if stri2 == 'jun':
		return 6
	if stri2 == 'jul':
		return 7
	if stri2 == 'aug' or stri2 == 'ago':
		return 8
	if stri2 == 'sep' or stri2 == 'set':
		return 9
	if stri2 == 'oct' or stri2 == 'out':
		return 10
	if stri2 == 'nov':
		return 11
	if stri2 == 'dec' or stri2 == 'dez':
		return 12

	return stri

def replace_space(stri, default:str='_'):
	if not isinstance(stri, str): return stri
	return stri.replace(' ', default)


def remove_spaces(stri:str, replace_nans:bool=True):
	if replace_nans: stri = stri.replace('nan', '')
	stri = re.sub(' +', ' ', stri)
	return stri.strip()

# encondig = 'utf-8',
def pdwritecsv(df, filename, path="./", sep='\t', index=False, verbose=False):

	if not os.path.exists(path):
		print("Path does not exists: '%s'"%(path))
		return False

	filename = os.path.join(path, filename)
	try:
		df.to_csv(filename, sep=sep, index=index)  # , encondig = encondig
	except ValueError:
		print(f"Error '{ValueError}', writing in '{filename}'")
		return False

	if verbose: print("Table saved (%s) at '%s'"%(df.shape, filename))
	return True

def dumpdic(dic, filename, path="./", verbose=True):
	return pddumpdic(dic=dic, filename=filename, path=path, verbose=verbose)

def pddumpdic(dic, filename, path="./", verbose=True):

	if not os.path.exists(path):
		print(f"Path does not exists: '{path}'")
		return False

	filename = os.path.join(path, filename)
	try:
		# Store data (serialize)
		with open(filename, 'wb') as handle:
			pickle.dump(dic, handle) # , protocol=pickle.HIGHEST_PROTOCOL
	except ValueError:
		print(f"Error '{ValueError}' in pickle dump '{filename}'")
		return False

	if verbose: print("Dictionary saved at '%s'"%(filename))
	return True

def loaddic(filename, path="./", verbose=True):
	return pdloaddic(filename=filename, path=path, verbose=verbose)


def pdloaddic(filename, path="./", verbose=True):

	if not os.path.exists(path):
		print("Path does not exists: '%s'"%(path))
		return None

	filename = os.path.join(path, filename)

	if not os.path.exists(filename):
		print("File does not exist: '%s'"%(filename))
		return None

	try:
		# Load data (deserialize)
		with open(filename, 'rb') as handle:
			dic = pickle.load(handle)
	except ValueError:
		print(f"Error '{ValueError}' in pickle loading '{filename}'")

	if verbose: print("Dictionary read at '%s'"%(filename))
	return dic


def padl(val, num, charac = '0'):
	return pad(val, num, charac)

def pad(val, num, charac = '0'):
	sval = str(val).strip()
	dif = num - len(sval)
	if dif <= 0: return sval

	zeros = [charac]*dif
	return ''.join(zeros)+sval

def padr(val, num, charac = '0'):
	sval = str(val).strip()
	dif = num - len(sval)
	if dif <= 0: return sval

	zeros = [charac]*dif
	return sval+''.join(zeros)

def try_float(num, except_ret=None):
	try:
		return float(num)
	except:
		return except_ret

def isfloat(num):
	try:
		num = float(num)
		return True
	except:
		return False

def try_int(num, except_ret=None):
	try:
		return int(num)
	except:
		return except_ret

def isint(num):
	try:
		num = int(num)
		return True
	except:
		return False

def isint_v2(x):
	try:
		a = float(x)
		b = int(a)
	except:
		return False

	return a == b

def return_integers(vals, except_ret=None):
	return [int(x) for x in vals if not try_int(x, except_ret) is None]

def return_floats(vals, except_ret=None):
	return [float(x) for x in vals if not try_float(x, except_ret) is None]

##--- only strings??
def merge_by_columns_inner_outer(dflista, keys, how, fillna=False):
	dfn = dflista[0]

	for i in range(1, len(dflista)):
		df2 = dflista[i]
		dfn = pd.merge(dfn, df2, how=how, on=keys)

	if fillna: dfn = dfn.fillna(0)

	return dfn

def merge_by_columns_inner(dflista, keys, fillna=False):
	dfn = dflista[0]

	for i in range(1, len(dflista)):
		df2 = dflista[i]
		dfn = pd.merge(dfn, df2, how="inner", on=keys)

	if fillna: dfn = dfn.fillna(0)

	return dfn

def merge_by_columns_outer(dflista, keys, fillna=False):
	return merge_by_columns(dflista, keys, fillna)

def merge_by_columns(dflista, keys, fillna=False):
	dfn = dflista[0]

	for i in range(1, len(dflista)):
		df2 = dflista[i]
		dfn = pd.merge(dfn, df2, how="outer", on=keys)

	if fillna: dfn = dfn.fillna(0)

	return dfn

def prepare_figname(figname, endtype='png'):
	posi = [i for i in range(len(figname)) if figname[i] == '.']
	if len(posi) > 0:
		figname = figname[:posi[0]]

	figname = title_replace(figname)

	figname = figname + '.' + endtype
	return figname

def title_replace(title):
	title = title.strip()
	if title[-1] == '.':
		title = title[:-1]

	return title.replace(',','_').replace(";","_").replace(' - ','_').replace('\n','_').replace('<br>','_').\
				 replace("  ", "_").replace(" ", "_").replace("/","-").\
				 replace("_=_", "_").replace("=","_").replace("___","_").replace("__","_").replace(":","")

# is_in(['aaaa', 'bbbbb', 'abcd'], ['a','x'])
def is_in(series, lista):
	series = [x.lower().replace("-","") for x in series]
	return [ np.sum([1 if x in k else 0 for x in lista]) > 0  for k in series]


colorscale=[[1.0, "rgb(165,0,38)"],
			[0.8888888888888888, "rgb(215,48,39)"],
			[0.7777777777777778, "rgb(244,109,67)"],
			[0.6666666666666666, "rgb(253,174,97)"],
			[0.5555555555555556, "rgb(254,224,144)"],
			[0.4444444444444444, "rgb(224,243,248)"],
			[0.3333333333333333, "rgb(171,217,233)"],
			[0.2222222222222222, "rgb(116,173,209)"],
			[0.1111111111111111, "rgb(69,117,180)"],
			[0.0, "rgb(49,54,149)"]]

def set_color_scale(min, max):
	maxi = max
	if abs(min) > maxi:
		maxi = abs(min)

	pos = np.linspace(-maxi, max, num=10)

	colorscale=[[pos[9], "rgb(165,0,38)"],
				[pos[8], "rgb(215,48,39)"],
				[pos[7], "rgb(244,109,67)"],
				[pos[6], "rgb(253,174,97)"],
				[pos[5], "rgb(254,224,144)"],
				[pos[4], "rgb(224,243,248)"],
				[pos[3], "rgb(171,217,233)"],
				[pos[2], "rgb(116,173,209)"],
				[pos[1], "rgb(69,117,180)"],
				[pos[0], "rgb(49,54,149)"]]

	return colorscale


def columns_to_case(df, pre_cols, cols):

	dff = None
	for col in cols:
		dfa = df[ pre_cols + [col]].copy()
		dfa["class"] = col
		dfa.columns = pre_cols + ['val', 'class']

		if dff is None:
			dff = dfa
		else:
			dff = dff.append(dfa)

	return dff

def char_frequency(mat:List):
	dic = OrderedDict()
	for c in mat:
		if c in dic.keys():
			dic[c] += 1
		else:
			dic[c] = 1
	return dic

def best_nucleotide(mat, with_gaps=False):
	dicc = char_frequency(mat)

	nuc_list = ['A','T','G','C']
	if with_gaps: nuc_list += ['-']

	maxi = 0; best_key = None
	for key, value in dicc.items():
		if key in nuc_list:
			if value > maxi:
				maxi = value
				best_key = key

	return best_key

def best_amino_acid(mat, with_gaps=False):
	dicc = char_frequency(mat)

	aa_list = ['A', 'M', 'C', 'F', 'L', 'V', 'I', 'G', 'T', 'S', 'W', 'Y', 'P', 'H', 'N', 'D', 'E', 'Q', 'K', 'R']
	if with_gaps: aa_list += ['-']

	maxi = 0; best_key = None
	for key, value in dicc.items():
		if key in aa_list:
			if value > maxi:
				maxi = value
				best_key = key

	return best_key

import multiprocessing as mp

def get_cpus(decrement_cpus=1, verbose=True):
	maxcpus = mp.cpu_count()
	cpus	= maxcpus-decrement_cpus

	if verbose: print("max CPUs %d - using %d"%(maxcpus, cpus) )
	return cpus

def break_lines_length(stri:str, split_char:str=' ', sep:str='<br>', maxLen:int=30) -> str:
	return break_line_per_length(stri, split_char, sep, maxLen)

def break_line_per_length(stri:str, split_char:str=' ', sep:str='<br>', maxLen:int=30) -> str:
	mat = stri.split(split_char)
	text = ''; temp=''
	insert_break = False
	for term in mat:
		if insert_break:
			text += '<br>' + term
			insert_break = False
		else:
			text += term

		temp += term

		if len(temp) > maxLen:
			temp = ''
			insert_break = True
		else:
			text += ' '

	return text.strip()


def break_lines(stri, sep=' ', nwords=25, CR='<br>', maxLen=250, maxLines=-1):
	stri = stri.strip()
	text = ''
	mat = stri.split(sep)
	lines  = 0
	for i in range(0, len(mat), nwords):
		mat2 = mat[i:(i+nwords)]
		if len(mat2) == 0: break

		newtext = sep.join(mat2)
		if len(newtext) > maxLen:
			L2 = int(len(newtext)/2)
			while(True):
				text += newtext[:maxLen] + CR
				if len(newtext) <= maxLen: break
				newtext = newtext[maxLen:]
				lines += 1
		else:
			text += newtext + CR
			lines += 1

		if maxLines > 0 and lines > maxLines:
			text += "..."
			return text

	return text[:-len(CR)]

def shuffle_nums(nrows, N, howManySets, verbose=False):

	samples = np.arange(0, nrows)
	random.shuffle(samples)

	# print(">> samples: %d x %d"%(nrows, len(samples)))

	sampList = []
	for i in range(howManySets):
		if nrows >= (i+1)*N:
			tsamp = samples[(i*N):(i+1)*N]
			sampList.append(tsamp)
		else:
			if verbose: print("Fasta has only %d - not %d"%(len(samples), (i+1)*N))

	return np.array(sampList)


def test_date_Ymd(x):
	x = str(x)
	try:
		a = int(x[:4])
	except:
		return False

	try:
		a = int(x[5:7])
	except:
		return False

	try:
		a = int(x[8:])
	except:
		return False

	return True

def to_roman_numeral(value):

	roman_map = {
		1: "I", 5: "V",
		10: "X", 50: "L",
		100: "C", 500: "D",
		1000: "M",
	}
	result = ""
	remainder = value

	for i in sorted(roman_map.keys(), reverse=True):
		if remainder > 0:
			multiplier = i
			roman_digit = roman_map[i]

			times = remainder // multiplier
			remainder = remainder % multiplier
			result += roman_digit * times

			if result.endswith('VIIII'): result = result.replace('VIIII','IX')
			elif result.endswith('IIII'): result = result.replace('IIII','IV')

	return result


def list_dfmeta_field(dfmeta, field):
	try:
		vals = dfmeta[field].unique()
	except:
		return []

	return np.sort(vals)

def create_dir(root, _dir=None):
	if _dir is None:
		path = root
	else:
		path = os.path.join(root, _dir)

	if not os.path.exists(path):
		os.mkdir(path)

	return path

def echo_print(stri:str, verbose:bool=False):
	if verbose: print(stri)
	return stri + '\n'

def all_equal_list(cols1, cols2):
	cols1 = list(cols1)
	cols2 = list(cols2)

	if cols1 == [] and cols2 == []: return True

	if len(cols1) != len(cols2): return False

	soma = np.sum([1 if cols1[i] != cols2[i] else 0 for i in range(len(cols1))])
	return soma == 0

def zip_compress(zip_name, root_zip, root_data, type_of_file='pdf'):
	'''
		https://stackoverflow.com/questions/47438424/python-zip-compress-multiple-files

		Select the compression mode ZIP_DEFLATED for compression
		or zipfile.ZIP_STORED to just store the file
	'''
	compression = zipfile.ZIP_DEFLATED
	filefull = os.path.join(root_zip, zip_name)
	print(f'Starting: {filefull}')

	files = [x for x in os.listdir(root_data) if x.endswith(f'.{type_of_file}') and x != f'.{type_of_file}']
	if len(files) == 0:
		print(f"There are no files in '{root_data}'")
		return False

	print(f"Saving '{filefull}' # {len(files)} ... {type_of_file} files...")
	# create the zip file first parameter path/name, second mode
	zf = zipfile.ZipFile(filefull, mode="w")

	try:
		ret = True
		i = -1
		for fname in files:
			i += 1
			if i%1000 == 0: print('.', end='')
			zf.write(os.path.join(root_data, fname), fname, compress_type=compression)
	except ValueError:
		print(f"\nError '{ValueError}', could not zip the files.")
		ret = False
	finally:
		# Don't forget to close the file!
		print(f"\nfile saved at '{filefull}' with {len(files)} files.")
		zf.close()

	return ret

def rename_files(root, pattern_src: str, pattern_dst: str, _type:str='.tsv', verbose:bool=False) -> bool:

	if _type is None:
		lista = [x for x in os.listdir(root) if pattern_src in x and not '~lock' in x ]
	else:
		lista = [x for x in os.listdir(root) if pattern_src in x and not '~lock' in x and x.endswith(_type)]

	if lista == []:
		if verbose: print(f"There are no files in {root}")
		return False

	ret = 1
	for fname in lista:
		fname_src = os.path.join(root, fname)

		fname_dst = fname.replace(pattern_src, pattern_dst)
		fname_dst = os.path.join(root, fname_dst)

		if not  os.path.exists(fname_src):
			print(f"fname does not exist: '{fname_src}'", fname_src)
			continue

		if os.path.exists(fname_dst):
			print(f"fname already exists: '{fname_dst}'")
			continue

		if fname_dst == fname_src:
			print(f"fname did not change: 'fname_src'")
			continue

		if verbose: print(fname_src, 'to', fname_dst)

		try:
			os.rename(fname_src, fname_dst)
		except:
			ret *= 0

	return ret==1

def series_round_scientific(df_series):
	mat_list = [x if isinstance(x, list) else eval(x) for x in df_series]

	for i in range(len(mat_list)):
		mat = mat_list[i]
		mat = [f"{float(x):.2e}" for x in mat]
		mat_list[i] = mat

	return mat_list
