#!/usr/bin/python
#!python
# -*- coding: utf-8 -*-

from Bio	 import SeqIO
from Bio	 import Entrez
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from typing import Optional, Iterable, Set, Tuple, Any, List

# from Bio import pairwise2
from Bio.Align import substitution_matrices
blosum62 = substitution_matrices.load("BLOSUM62")

# from Bio.Alphabet  import IUPAC
# from Bio.Alphabet  import generic_dna
# from Bio.Alphabet  import generic_protein

import os, sys, copy, re, random
from re import sub
from os.path import join, exists
from datetime import datetime

import pandas as pd
import numpy as np
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from Basic import *
from util_general import *

b = Basic()

class Pairwise:
	def __init__(self, blosum_talbe="BLOSUM62"):
		self.blosum = substitution_matrices.load(blosum_talbe)

	def score_match(self, pair):
		if pair not in self.blosum:
			return self.blosum[(tuple(reversed(pair)))]
		else:
			return self.blosum[pair]

	def score_pairwise(self, seq1, seq2, gap_s=-10, gap_e=-1):
		seq1 = str(seq1)
		seq2 = str(seq2)

		score = 0; gap = False

		for i in range(len(seq1)):
			pair = (seq1[i], seq2[i])
			if not gap:
				if '-' in pair:
					gap = True
					score += gap_s
				else:
					score += self.score_match(pair)
			else:
				if '-' not in pair:
					gap = False
					score += self.score_match(pair)
				else:
					score += gap_e

		return score


class MySequence:
	def __init__(self, prjName, isProtein=True, root = "../fasta/"):
		self.clear_defaults()
		self.clearSequences()

		self.prjName = prjName
		self.isProtein = isProtein
		self.root	= root

		self.isAligned = False
		self.patterns = None
		self.isCurated = False
		self.showmessage = False

		try:
			os.mkdir("./tmp")
		except:
			pass

		self.dfcodons, self.dicTrans, self.stop_list = None, None, None

	def clear_defaults(self):
		self.typeSeq = "undef"
		self.pattern = "undef"
		self.showmessage = False

		self.fileName	= ""
		self.fileNameAln = ""
		self.name = ""

	def clearSequences(self):
		self.seqs = []
		self.seq_records = []
		self.seqCut = []
		self.seqFinal = []

		self.numSequences = 0
		self.lenSequence = 0
		self.nrow = 0
		self.ncol = 0

	def readFasta_by_params(self, typeSeq, pattern, isProtein=True, isAligned=True, isFiltered=False, isCurated=False, showmessage=False):

		patterns = None

		self.typeSeq = typeSeq
		self.pattern = pattern
		self.isProtein = isProtein
		self.isAligned = isAligned
		self.patterns = patterns
		self.isCurated = isCurated
		self.showmessage = showmessage

		if isCurated:
			fileName, name = self.ret_sequence_name(typeSeq, pattern, isProtein=isProtein, seq_type="curated")

		elif isFiltered:
			fileName, name = self.ret_sequence_name(typeSeq, pattern, isProtein=isProtein, seq_type="filtered")

		elif isAligned:
			fileName, name = self.ret_sequence_name(typeSeq, pattern, isProtein=isProtein, seq_type="aligned")

		else:
			fileName, name = self.ret_sequence_name(typeSeq, pattern, isProtein=isProtein, seq_type="sequence")

		self.fileName = fileName
		self.name = name

		ret = self.readFasta(self.fileName, showmessage)

		return ret

	def read_fasta(self, filename:str, ignore_msa:bool=False, is_msa:bool=True, 
				   verbose:bool=False) -> bool:
		return self.readFasta(fileName=filename, ignore_msa=ignore_msa, 
							  is_msa=is_msa, showmessage=verbose)

	def readFasta(self, fileName:str, ignore_msa:bool=False, is_msa:bool=True, 
				  showmessage:bool=False) -> bool:
		self.clearSequences()

		self.isProtein  = ("_protein" in fileName) or ("_curated" in fileName)
		self.isAlligned = "_aligned" in fileName

		if not os.path.exists(fileName):
			if showmessage: print("Fasta not found file: '%s'"%fileName)
			return False

		if showmessage: print("Reading '%s'... "%(fileName), end='')

		i = 0
		try:
			for seq_record in SeqIO.parse(fileName, "fasta"):
				if i+1 % 20000 == 0:
					print(i, end=' ')

				seq_record.seq = seq_record.seq.upper()
				self.seq_records.append(seq_record)

				mat = list(str(seq_record.seq))
				if ignore_msa:
					self.seqs = None
				else:
					if i == 0:
						self.seqs = [mat]
						ncol = len(mat)
					else:
						# print("<<", ncol, len(self.seqs[0]))
						# print(">>", len(mat))
						self.seqs = self.seqs + [mat]
						if len(mat) == 0 or ncol != len(mat):
							stri = ">>>> it is not a MSA: line %d, %d x %d in '%s'"%(i, ncol, len(mat), fileName.split('/')[-1])
							
							try:
								log_save(stri, filename=self.filelog, verbose=showmessage)
							except:
								print(stri)
				i += 1

		except ValueError:
			print("Exception %s: error reading file: '%s'"%(str(ValueError), fileName) )
			self.seqs = []
			self.seq_records = []

			return False

		if is_msa and not ignore_msa:
			try:
				self.seqs = np.array(self.seqs, dtype=object)
			except:
				print("Could not transform seqs in np.array - readFasta")

		if self.seqs is not None and len(self.seqs) > 0:
			self.numSequences = len(self.seq_records)
			self.lenSequence  = len(self.seqs[0])

			self.nrow = self.numSequences
			self.ncol = self.lenSequence

			if showmessage:
				if isinstance(self.seqs, list):
					print("seqs is list: %d rows and %d cols from '%s'" %(self.nrow, self.ncol, fileName))
				else:
					print("seqs is np.array: %s seqs from '%s'" %(self.seqs.shape, fileName))

			ret = True
		else:
			ret = False if self.seq_records is None or self.seq_records == [] else True
			self.numSequences = len(self.seq_records)
			self.lenSequence  = 0

			self.nrow = self.numSequences
			self.ncol = 0

		return ret

	def readFasta_simple_wo_seqs(self, fileName:str, Nrec:int=1000, showmessage:bool=False) -> bool:
		self.clearSequences()

		self.isAlligned = False

		if not os.path.exists(fileName):
			if showmessage: print("Fasta not found file: '%s'"%fileName)
			return False

		if showmessage: print("Reading '%s'... "%(fileName), end='')

		i = 0
		max_seq = 0
		try:
			for seq_record in SeqIO.parse(fileName, "fasta"):

				seq_record.seq = seq_record.seq.upper()
				self.seq_records.append(seq_record)

				n = len(seq_record.seq)
				if n > max_seq:
					max_seq = n

				if i % Nrec == 0:
					print(f"{i}) max len {max_seq}", end=' ')

				i += 1

		except ValueError:
			print("Exception %s: error reading file: '%s'"%(str(ValueError), fileName) )
			self.seq_records = []
			return False

		self.numSequences = len(self.seq_records)
		self.lenSequence  = max_seq

		self.nrow = self.numSequences
		self.ncol = self.lenSequence

		if showmessage:
			print(f"seq_records has length {self.nrow} and max len = {max_seq}")

		return True


	def readFasta_by_patterns_and_params(self, typeSeq, patterns, isProtein=True, isAligned=True, isFiltered=False, isCurated=False, showmessage=False):

		pattern = "merged"

		self.typeSeq = typeSeq
		self.pattern = pattern
		self.isProtein = isProtein
		self.isAligned = isAligned
		self.patterns = patterns
		self.isCurated = isCurated
		self.showmessage = showmessage

		if isCurated:
			fileName, name = self.ret_sequence_name(typeSeq, pattern, patternList=patterns, isProtein=isProtein, seq_type="curated")

		elif isFiltered:
			fileName, name = self.ret_sequence_name(typeSeq, pattern, patternList=patterns, isProtein=isProtein, seq_type="filtered")

		elif isAligned:
			fileName, name = self.ret_sequence_name(typeSeq, pattern, patternList=patterns, isProtein=isProtein, seq_type="aligned")

		else:
			fileName, name = self.ret_sequence_name(typeSeq, pattern, patternList=patterns, isProtein=isProtein, seq_type="sequence")

		self.fileName = fileName
		self.name = name

		ret = self.readFasta(self.fileName, showmessage)

		return ret


	def ret_sequence_name(self, typeSeq, pattern=None, patternList=None, isProtein=True, seq_type="sequence"):
		strDNAProt = 'protein' if isProtein else 'dna'
		filename = ""; name = ""

		str_patternList = ""
		if patternList: str_patternList = "_and_".join([str(p) for p in patternList])

		if seq_type == "sequence":
			if patternList is None:
				assert not pattern is None

				filename = "%s%s_%s_%s_type_%d.fasta"%(self.root, self.prjName, strDNAProt, typeSeq, pattern)
				name	 = "%s (%s): %s strain %d"%(self.prjName, strDNAProt, typeSeq, pattern)
			else:
				assert len(patternList) > 1

				filename = "%s%s_%s_%s_dna_type_merged_%s.fasta"%(self.root, self.prjName, strDNAProt, typeSeq, str_patternList)
				name = "%s (%s merged): %s strands %s"%(self.prjName, strDNAProt, typeSeq, str_patternList)


		elif seq_type == "aligned":
			if patternList is None:
				assert not pattern is None

				filename = "%s%s_%s_%s_type_%d_aligned.fasta"%(self.root, self.prjName, strDNAProt, typeSeq, pattern)
				name = "%s (%s aligned): %s strain %d"%(self.prjName, strDNAProt, typeSeq, pattern)
			else:
				assert len(patternList) > 1

				filename = "%s%s_%s_%s_dna_type_merged_%s_aligned.fasta"%(self.root, self.prjName, strDNAProt, typeSeq, str_patternList)
				name = "%s (%s merged and aligned): %s strands %s"%(self.prjName, strDNAProt, typeSeq, str_patternList)

		elif seq_type == "filtered":
			if patternList is None:
				assert not pattern is None

				filename = "%s%s_%s_%s_type_%d_aligned_filtered.fasta"%(self.root, self.prjName, strDNAProt, typeSeq, pattern)
				name = "%s (%s aligned and filtered): %s strain %d"%(self.prjName, strDNAProt, typeSeq, pattern)
			else:
				assert len(patternList) > 1

				filename = "%s%s_%s_%s_dna_type_merged_%s_aligned_filtered.fasta"%(self.root, self.prjName, strDNAProt, typeSeq, str_patternList)
				name = "%s (%s merged, aligned, filtered): %s strands %s"%(self.prjName, strDNAProt, typeSeq, str_patternList)

		else:
			if patternList is None:
				assert not pattern is None

				filename = "%s%s_%s_%s_type_%d_curated.fasta"%(self.root, self.prjName, strDNAProt, typeSeq, pattern)
				name = "%s (%s curated): %s strain %d"%(self.prjName, strDNAProt, typeSeq, pattern)
			else:
				assert len(patternList) > 1

				filename = "%s%s_%s_%s_type_merged_'%s'_curated.fasta"%(self.root, self.prjName, strDNAProt, typeSeq, str_patternList)
				name = "%s (%s curated): %s strain %s"%(self.prjName, typeSeq, strDNAProt, str_patternList)

		return filename, name

	# rever !!!
	def order_seqs_by_id(self, seq_records):
		listOfStrings = [seqr.id for seqr in seq_records]
		listOfStrings.sort()

		ordered = []
		for _id in listOfStrings:
			for seqx in seq_records:
				if seqx.id == _id:
					ordered.append(seqx)

		return ordered

	def filter_bad_seqs_by_country(self, country, seq_records, dic, pInf = 0.1):
		seqrecs = [ seq_records[i] for i in dic[country] ]

		keep = []
		for i in range(len(seqrecs)):
			seqs = list(seqrecs[i].seq)
			n = len(seqs)
			bads = np.sum([1 for nuc in seqs if nuc == "N" or nuc == "-" or nuc == "" or nuc == " " or nuc == "?" ])
			if bads >= 200 or bads/n > pInf:
				pass
				# print("error: %d is bad  n = %d  and bads = %d"%(i, n, bads))
			else:
				# print("good:  %d is bad  n = %d  and bads = %d"%(i, n, bads))
				keep.append(i)

		# print(len(seqrecs))
		# print(len(keep))
		seqrecs = [seqrecs[i] for i in keep]
		# print(len(seqrecs))

		return seqrecs

	def filter_seqs(self, seqs, pattern, percent = .25, wantFill=False, force=False):

		keep = [(seqs[:,j] == "-" or seqs[:,j] == "N" or seqs[:,j] == "?").sum()/self.nrow < percent for j in range(self.ncol)]
		keep = np.array(keep, dtype=object)
		# print(keep.mean())
		# print(keep.mean()*self.nnuc)

		seqs = seqs[:,keep]

		if wantFill: seqs = self.fulfill(seqs)

		fnameAlig, _ = self.ret_sequence_name(self.typeSeq, pattern=pattern, isProtein=self.isProtein, seq_type="filtered")

		self.save_unique_seqs(seqs, fnameAlig, force=force)

		return self.seqs

	def filter_seqs_patterns(self, seqs, patterns, percent = .25, wantFill=False, force=False):

		keep = [(seqs[:,j] == "-").sum()/self.nrow < percent for j in range(self.ncol)]
		keep = np.array(keep, dtype=object)

		seqs = seqs[:,keep]

		if wantFill: seqs = self.fulfill(seqs)

		fnameAlig, _ = self.ret_sequence_name(self.typeSeq, pattern="merged", patternList=patterns, isProtein=self.isProtein, seq_type="filtered")

		self.save_unique_seqs(seqs, fnameAlig, force=force)

		return self.seqs

	def save_unique_seqs(self, seqs, filename, force=False):

		self.seqs = seqs

		# replace new seqs, filtered ...
		i = 0
		for seq_rec in self.seq_records:
			seq_rec.seq = Seq("".join(seqs[i,:]))
			i += 1

		# excluding not repetitive sequences
		_, indices  = np.unique(seqs, return_index=True)
		nums = np.arange(len(seqs))
		nums = np.delete(nums, indices)

		self.seqs = seqs[nums]
		self.seq_records = [self.seq_records[i] for i in nums]

		if not exists(filename) or force:
			ret = self.writeSequences(self.seq_records, filename)

			filename2 = self.filename_to_txt(filename)
			if filename == filename2:
				print("Problem with txt filename: '%s'", filename2)
			else:
				_ = self.write_seqs_to_txt(seqs, filename2)
		else:
			ret = True

		return ret

	def filename_to_txt(self, filename):
		filename = sub("[.]fasta", ".txt", filename)
		filename = sub("[.]fas",   ".txt", filename)
		filename = sub("[.]csv",   ".txt", filename)
		filename = sub("[.]tsv",   ".txt", filename)

		return filename


	def fulfill(self, seqs2, numOfLetters=1):
		seqs = copy.deepcopy(seqs2)
		ncol = seqs.shape[1]
		nrow = seqs.shape[0]
		maxL = ncol-numOfLetters+1

		dicPiList = []
		dicLeList = []

		for j in range(maxL):
			dicPi = {}; dicLe = {}

			for row in range(nrow):
				letters = "".join(seqs[row,j:(j+numOfLetters)])
				try:
					dicPi[letters] += 1
					dicLe[letters] += 1
				except:
					dicPi[letters] = 1
					dicLe[letters] = 1

			for key in dicPi.keys():
				dicPi[key] /= nrow
				# print(j, key, dicPi[key])

			dicPiList.append(dicPi)
			dicLeList.append(dicLe)


		for j in range(maxL):
			dicPi = dicPiList[j]
			dicLe = dicLeList[j]

			if len(dicPi.keys()) > 1:
				if "-" in dicPi.keys():
					if len(dicPi.keys()) == 2:
						# print("in2 ", j)
						mat = list(dicPi.keys())
						mat = list(filter(lambda x : x != '-', mat))
						for row in range(nrow):
							if "".join(seqs[row,j:(j+numOfLetters)]) == "-"*numOfLetters:
								seqs[row,j:(j+numOfLetters)] = mat[0]
					else:
						# print("inMany ", j)
						ntrace = dicLe['-']

						del dicPi["-"]
						del dicLe["-"]
						total = sum(dicPi.values())
						for key in dicPi.keys():
							dicPi[key] /= total
							# dicLe[key] /= total
							# print(j, key, dicPi[key])

						keys = []; ps = []
						for key in dicPi.keys():
							keys.append(key)
							ps.append(dicPi[key])

						nelem = len(keys)
						ps = np.array(ps, dtype=object)

						ps2 = ps * 100 * ntrace
						ps2 = ps2.astype(int)

						stri = "".join([keys[i]* ps2[i] for i in range(nelem)])
						lista = list(stri)
						random.shuffle(lista)
						stri= "".join(lista)

						lets = random.sample(stri, ntrace)

						assert ntrace == len(lets)

						count = 0

						for row in range(seqs.shape[0]):
							if "".join(seqs[row,j:(j+numOfLetters)]) == "-"*numOfLetters:
								try:
									seqs[row,j:(j+numOfLetters)] = lets[count]
									count += 1
								except:
									print("Error:")
									print(seqs.shape)
									print(row)
									print("j",j)
									print(seqs[row, j:(j+numOfLetters)])
									print(count)
									print(len(lets))
									print("ntrace", ntrace)
									print(lets)
									print("")
									print(dicPi)

		return seqs

	def mergeDNASeqs(self, seq_records1, seq_records2):
		 return self.mergeSequences(seq_records1, seq_records2)

	def mergeProteinSeqs(self, seq_records1, seq_records2):
		 return self.mergeSequences(seq_records1, seq_records2)

	def mergeSequences(self, seq_records1, seq_records2):
		 return seq_records1 + seq_records2

	def writeDNASeqs(self, seq_records, fileName):
		return self.writeSequences(seq_records, filename)

	def writeProteinSeqs(self, seq_records, fileName):
		return self.writeSequences(seq_records, filename)

	def write_fasta(self, seq_records:List, filename:str, verbose:bool=False) -> bool:

		try:
			SeqIO.write(seq_records, filename, "fasta")
			if verbose: print(f"Fasta saved {len(seq_records)} records: '{filename}'")
			ret = True
		except ValueError:
			print("Exception %s: error writing file: '%s'"%(str(ValueError), fileName) )
			ret = False

		return ret

	def writeSequencesSeqs(self, seqs:List, filename:str, verbose:bool=False) -> bool:
		seq_records = []
		for i in range(len(seqs)):
			_id = str(i)
			record = SeqRecord(
				seq=Seq("".join(seqs[i])),
				id=_id,
				name=_id,
				description=_id
			)

			seq_records.append(record)

		try:
			SeqIO.write(seq_records, filename, "fasta")
			if verbose: print(f"Fasta saved {len(seqs)} records: '{filename}'")
			ret = True
		except ValueError:
			print("Exception %s: error writing file: '%s'"%(str(ValueError), fileName) )
			ret = False

		return ret

	def writeSequences(self, seq_records, filename, is_msa=True, verbose=False):

		if isinstance(seq_records[0].seq, str):
			for seq_rec in seq_records:
				seq_rec.seq = Seq(seq_rec.seq)

		try:
			SeqIO.write(seq_records, filename, "fasta")
			ret = True
		except ValueError:
			print("Exception %s: error writing file: '%s'"%(str(ValueError), fileName) )
			ret = False

		if ret:
			if verbose: print("%d records, file saved: '%s'"%(len(seq_records), filename))

			self.seqs = [list(seq_rec) for seq_rec in seq_records]
			if is_msa: self.seqs = np.array(self.seqs, dtype=object)

			self.seq_records = seq_records
			self.numSequences = len(seq_records)
			self.lenSequence  = len(seq_records[0])

			self.nrow = self.numSequences
			self.ncol = self.lenSequence
		else:
			self.seqs = []; self.seq_records = []
			self.numSequences, self.lenSequence  = None, None
			self.nrow, self.ncol = None, None

		return ret


	def writeSequences_simple(self, seq_list, id_list, filename, verbose=False):

		seq_records = []
		for seq, _id in zip(seq_list, id_list):
			seq_rec = SeqRecord(
				seq=Seq(seq),
				id=_id,
			)
			seq_records.append(seq_rec)

		try:
			SeqIO.write(seq_records, filename, "fasta")
			if verbose: print(f"Saved {len(seq_records)} in file:'{filename}'")
			ret = True
		except ValueError:
			print("Exception %s: error writing file: '%s'"%(str(ValueError), fileName) )
			ret = False

		return ret


	def write_seqs_to_txt(self, seqs, filename, showmessage=True):
		assert isinstance(filename, str)

		try:
			txt = [''.join(seq) + '\n' for seq in seqs]
			h = open(filename, "w+")
			h.writelines(txt)

			if (showmessage):
				print("Wrote %d lines in '%s'" %(len(seqs), filename))

		except ValueError:
			print("Exception %s: error writing file: '%s'"%(str(ValueError), filename) )
			return False

		return True


	#-- get rid of '-' and 'N' for calc_posi
	def prepare_seqs_getrid_gaps(self, depth, cutoffNNN=.51, maxpos=None):
		seqs = copy.deepcopy(self.seqs)

		if depth < len(seqs):
			seqs = seqs[:depth, :]

		N = len(seqs)
		M = len(seqs[0])
		seqs = seqs.T

		badCols = [i for i in range(M) if (np.sum([1 for x in seqs[i] if (x == "-") or (x == "N") ]) / N) >= cutoffNNN]
		if maxpos is not None:
			if M - len(badCols) < maxpos:
				diff = M - maxpos
				if diff <= 0:
					badCols = []
				else:
					diff_half = int(diff/2)
					badCols = badCols[diff_half:-diff_half]

		if len(badCols) > 0:
			seqs = np.delete(seqs, badCols, axis=0)

		seqs = seqs.T

		return seqs, badCols


	def filter(self, ids, seq_records):

		seqrs = []
		for _id in ids:
			for seqx in seq_records:
				if seqx.id == _id:
					seqrs.append(seqx)

		return seqrs

	def align_seqs_muscle(self, fileName, fileNameAligned, seaviewPath, force=False, verbose=False):
		from Bio.Align.Applications import MuscleCommandline

		muscle_exe = join(seaviewPath, "muscle")
		if not exists(muscle_exe):
			stri = "ERROR: MUSCLE does not exist '%s'" %(muscle_exe)
			log_save(stri, filename=self.filelog, verbose=verbose)
			return False

		in_file = fileName
		if verbose: print("in file: ", os.path.isfile(fileName))

		if not exists(fileName):
			print("ERROR: file does not exist '%s'" %(fileName))
			return False

		if not exists(fileNameAligned) or force:
			cmd = MuscleCommandline(muscle_exe, input=in_file, out=fileNameAligned)
			if verbose: print("%s"%(cmd))
			os.system("%s"%(cmd))
		else:
			if verbose: print("File already aligned: '%s'"%fileNameAligned)

		return True


	def align_seqs_clustalo(self, fileName, fileNameAligned, seaviewPath, force=False, verbose=False, sufix_log='clustalo'):
		from Bio.Align.Applications import ClustalOmegaCommandline

		clustalo_exe = join(seaviewPath, "clustalo")
		if not exists(clustalo_exe):
			stri = "ERROR: Clustal-Omega (clustalo) does not exist '%s'" %(clustalo_exe)
			log_save(stri, filename=self.filelog, verbose=verbose)
			return False

		in_file = fileName
		if verbose: print("in file: ", os.path.isfile(fileName))

		if not exists(fileName):
			print("ERROR: file does not exist '%s'" %(fileName))
			return False

		if not exists(fileNameAligned) or force:
			cmd = ClustalOmegaCommandline(clustalo_exe, infile = in_file, outfile = fileNameAligned, verbose = True, force=True, auto = False, log="./logs/log_%s.txt"%(sufix_log))
			if verbose: print(">>>", cmd)
			os.system("%s"%(cmd))
		else:
			if verbose: print("File already aligned: '%s'"%fileNameAligned)

		return True

	def cutBeginSeq(self, seq, posIni=-1, showmessage=False):
		if posIni > 0:
			posLeft = posIni-1
		else:
			posLeft = 0

			for j in range(len(seq)):

				stri = seq[j]

				for i in range(len(stri)):
					if (stri[i] == '-'):
						pos = i
					else:
						break

					if (pos != 0):
						if (pos > posLeft):
							posLeft = pos

		# print(posLeft)
		for j in range(len(seq)):

			stri = seq[j]

			self.seqCut.append(stri[posLeft + 1:])

			if (showmessage):
				print('left cut ', self.seqCut[j])

		return posLeft+1, self.seqCut


	def cutEndSeq(self, seq, posEnd=-1, showmessage=False):
		if posEnd > 0:
			posRight = posEnd+1
		else:
			posRight = 99999999999

			for j in range(len(seq)):
				stri = seq[j]

				i = len(stri) - 1;
				while True:
					# print 'begin lenght ', i
					if (stri[i] == '-'):
						pos = i
					else:
						break

					if (pos != 0):
						if (pos < posRight):
							posRight = pos

					i = i - 1
					if (i < 0):
						break;

		print(posRight)
		for j in range(len(seq)):

			stri = seq[j][:posRight]
			self.writeCutSequence(j, stri)

			if (showmessage): print('rigth cut ', self.seqCut[j])

		return self.seqCut


	# you give the sequences and receive it cut back
	def automatic_cut_sequences(self, filename_seq, cutoffLine=.5, cutoffAny=.6, cutoffNNN=.67, cutNNN=True,
								cutPolyA=True, striAAA = "AAAAAAAAAA", verbose=True):

		t0 = datetime.now()

		N = len(self.seq_records)
		M = len(self.seqs[0])
		if verbose:
			print("before", N, M)
			t2 = datetime.now()

		'''
		# cannot cut np.array(), only lists
		self.seqs = list(self.seqs)
		self.seq_records = list(self.seq_records)
		for i in range(N): # range(self.seq_records)
			self.seqs[i] = list(self.seqs[i])'''

		if verbose: print("Cut PolyA")
		if cutPolyA:
			t1 = datetime.now()
			# to find min start to cut
			starts = [re.search(striAAA, str(seq_rec.seq)).start()
					  for seq_rec in self.seq_records if re.search(striAAA, str(seq_rec.seq))]

			if len(starts) > 0:
				min_start = np.min(starts)
				seqs = []
				for i in range(N): # range(self.seq_records)
					self.seq_records[i].seq = self.seq_records[i].seq[:min_start]
					seqs.append( self.seqs[i][:min_start] )

				self.seqs = seqs
				self.ncol = len(self.seqs[0])
				M = self.ncol

			if verbose:
				print("cutPolyA", N, M)
				t2 = datetime.now()
				print("cutPolyA ", (t2-t1).seconds, "seconds")

		##--- transpose all sequences --------------------------------------
		if verbose: print("Cut 'N' and '-'")
		t1 = datetime.now()
		all = np.array( [self.seqs[i] for i in range(N)], dtype=object).T
		if verbose:
			print("transpose to all")
			t2 = datetime.now()
			print("last ", (t2-t1).seconds, "seconds")

		#--- badCols == cols that have NNNNN ou NNN---NNN --------------------
		if cutNNN:
			t1 = datetime.now()
			badCols = [i for i in range(M) if np.sum([1 for x in all[i] if x in ["-", "N"]]) / N >= cutoffNNN]
			if len(badCols) > 0:
				all = np.delete(all, badCols, axis=0)
				M = len(all)
			if verbose:
				print("cutNNN", N, M)
				t2 = datetime.now()
				print("last ", (t2-t1).seconds, "seconds")

		#--- Bad sequences left & right (columns) ---------------------
		t1 = datetime.now()
		begin = 0   # find the fist column with more than 80% (cutoff) of valid nucleotides
		for i in range(M):
			p = np.sum([1 for x in all[i] if x in ["A", "C", "G", "T"]]) / N
			if p > cutoffAny:
				begin = i
				break

		if begin > 0:
			cols = list(range(begin))
			all = np.delete(all,cols,axis=0)
			M = len(all)

		end = M   # find the last column with more than 80% (cutoff) of valid nucleotides
		for i in range(M-1, -1, -1):
			p = np.sum([1 for x in all[i] if x in ["A", "C", "G", "T"]]) / N
			if p > cutoffAny:
				end = i
				break

		if end < M:
			cols = list(range(end, M))
			all = np.delete(all,cols,axis=0)

		# bad lines - transpose back
		all = all.T
		N = len(all)
		M = len(all[0])
		if verbose:
			print("all matrix", N, M)
			t2 = datetime.now()
			print("last ", (t2-t1).seconds, "seconds")

		badLines = [i for i in range(N) if np.sum([1 for x in all[i] if x in ["A", "C", "G", "T"]]) / M < cutoffLine]
		if len(badLines) > 0:
			t1 = datetime.now()

			all			  = np.delete(all, badLines,axis=0)
			self.seq_records = [self.seq_records[i] for i in range(N) if not i in badLines]
			self.seqs		= [self.seqs[i]		for i in range(N) if not i in badLines]

			N = len(all)
			M = len(all[0])
			if verbose:
				print("badLines", N, M)
				t2 = datetime.now()
				print("all ", (t2-t1).seconds, "seconds")

		if verbose: print("Preparing cut seqs")
		for i in range(N): # range(self.seq_records)
			self.seq_records[i].seq = "".join(list(all[i]))
		self.seqs = all   # list(all[i])

		if verbose:
			print("--------- end ----------")
			t2 = datetime.now()
			secs = (t2-t0).seconds
			if secs > 180:
				secs = np.round(secs / 60, 2)
				print("all cut process ", secs, "minutes")
			else:
				print("all cut process ", secs, "seconds")

		# return to np.array()
		'''
		self.seqs = np.array(self.seqs, dtype=object)
		for i in range(N): # range(self.seq_records)
			self.seqs[i] = np.array(self.seqs[i], object)'''

		self.nrow = N
		self.ncol = M
		if verbose: print("after cut",N, M)

		ret = self.writeSequences(self.seq_records, filename_seq)

		t2 = datetime.now()
		secs = (t2-t0).seconds

		if verbose:
			if secs > 180:
				secs = np.round(secs / 60, 2)
				print("Sequences cut in ", secs, "minutes")
			else:
				print("Sequences cut in ", secs, "seconds")

		return ret

	def is_ok_cut_fasta_quick_deprecated(self, filecut, fileori, percent=.2, force=False, verbose = False):
		if os.path.exists(filecut) and not force:
			print("Fasta cut exists: '%s'"%(filecut))
			return True

		if not os.path.exists(fileori):
			print("Fasta (original) for '%s' does not exist (in colab): ''%s'"%(country, fileori))
			return False

		self.readFasta(fileori)
		if len(self.seqs) < 5:
			print("There are no suffucient rows (< 5) for '%s'"%(country))
			return False

		if verbose: start = time.time()

		nrow = len(self.seqs)
		ncol = len(self.seqs[0])

		maxi_traces = int(nrow*(1-percent))

		seqs = self.seqs
		todel = []
		for j in range(ncol):
			if j%10000 == 0: print(j)
			tot = np.sum([x == '-' for x in seqs[:,j]])
			if tot > maxi_traces:
				todel.append(j)

		if verbose: print(time.time() - start, "secs, delete: ", len(todel))
		if len(todel) > 0:
			seqs = np.delete(seqs, todel, axis=1)
			ncol -= len(todel)

		i = 0
		for seq_rec in self.seq_records:
			seq_rec.seq = Seq("".join(seqs[i]))
			i += 1

		ret = self.writeSequences(self.seq_records, filecut)


	def fulfill_1pos(self, seqs2, min_seqs=3, type = "DNA"):

		seqs = copy.deepcopy(seqs2)
		nrow = len(seqs)
		ncol = len(seqs[0])

		dicPiList = []; dicLeList = []
		# conservedList = []

		# if not fulfill the gaps, there may be bad chars
		if type == "DNA":
			valid = b.getDnaNucleotides()
		elif type == "Protein" or type == "Amino acid":
			valid = b.getAA()
		else:
			raise("Error: only DNA and Protein tables are accepted (type = 'DNA' or 'Protein')")

		valid = list(valid) + ['-']

		for i in range(len(seqs)):
			seqs[i] = np.array([letter if letter in valid else '-' for letter in seqs[i]], dtype=object)

		# --- looping each column ---------
		for j in range(ncol):
			'''for i in range(nrow):
				[i][j] = seqs[i][j].replace("N", "-")'''

			dicLe = OrderedDict(); dicPi = OrderedDict()
			for seq in seqs:
				char = seq[j]
				if char in valid:
					try:
						dicLe[char] += 1
					except:
						dicLe[char] = 1

			total = np.sum(list(dicLe.values()))

			for k in dicLe.keys():
				dicPi[k] = dicLe[k] / total

			# isConserved = True if len(dicPi.keys()) = 1 else False

			dicPiList.append(dicPi)
			dicLeList.append(dicLe)
			# conservedList.append(isConserved)


		del_posi = []
		for j in range(ncol):
			dicPi = dicPiList[j]
			dicLe = dicLeList[j]

			if "-" in dicPi.keys():
				if len(dicPi.keys()) == 1:
					if list(dicPi.keys())[0] == '-':
						del_posi.append(j)
				elif len(dicPi.keys()) == 2:
					# print('conserved with -', j)
					ok = False
					for key, value in dicLe.items():
						if key != '-':
							if value >= min_seqs:
								ok = True
					if not ok:
						del_posi.append(j)
					else:
						mat = list(dicPi.keys())
						mat.remove('-')
						seqs[:,j] = [mat[0]]*nrow
				else:
					# print("multiple with -", j)
					ntrace = dicLe['-']
					del dicLe["-"]
					total = np.sum(list(dicLe.values()))

					if total < min_seqs:
						del_posi.append(j)
					else:
						del dicPi["-"]

						total = np.sum(list(dicLe.values()))  # e.g. 80%
						ps = []
						for key in dicPi.keys():
							dicPi[key] = dicLe[key] / total
							ps.append(dicPi[key])

						'''  generate hundreads of sample'''
						ps2 = (np.array(ps, dtype=int) * 100 * ntrace).astype(int)
						keys = list(dicPi.keys())
						lista = []
						for i in range(len(keys)):
							lista += [keys[i]]*ps2[i]
						random.shuffle(lista)

						''' traces are substituted randomically '''
						count = 0
						for row in range(nrow):
							if seqs[row][j] == "-":
								seqs[row][j] = lista[count]
								count +=1

		if del_posi != []:
			seqs = np.array( [np.delete(seq, del_posi, axis=None) for seq in seqs], dtype=object)

		return seqs

	def fulfill_3Nucleotides(self, seqs3cols):

		dfseq = pd.DataFrame(seqs3cols, columns=['a','b','c'])
		dfseq['_all'] = dfseq.a+dfseq.b+dfseq.c

		df1 = dfseq[dfseq._all != '---']

		#-- Flavio 2021/02/04
		if len(df1) == 0:
			return dfseq

		df2 = copy.deepcopy(dfseq[dfseq._all == '---'])

		seqq = fulfill_1pos( df1[['a','b','c']].to_numpy() )
		df1a = pd.DataFrame(seqq, columns=['a','b','c'])

		df1a['i'] = df1.index
		df2a = df2[['a','b','c']].copy()
		df2a['i']  = df2.index
		df1a = df1a.append(df2a)
		df1a = df1a.sort_values('i')
		return df1a[['a','b','c']]

	''' seqs is optative '''
	def fulfill_polymorfic_traces_deprecated(self, ini, length, seqs=None):
		if length % 3 != 0:
			raise("Length must be multiple of 3")
			return None

		if seqs is None:
			seqs = self.seqs

		nrow = len(seqs)
		ncol = len(seqs[0])

		if ini + length > ncol:
			length = ncol - ini
			if length%3 != 0: length -= 1
			if length%3 != 0: length -= 1

		mat = np.array([ [' ']*length] * nrow, dtype=object)

		iloop = 0
		for j in range(ini,(ini+length), 3):
			dfseq = self.fulfill_3Nucleotides(seqs[:, j:(j+3)])
			matseq = dfseq.to_numpy()
			mat[:,iloop:(iloop+3)] = matseq

			'''for k in range(3):
				lista = [tri[k] for tri in matseq]
				mat[:,(iloop+k)] = lista'''
			iloop += 3

		return mat

	'''  old cut_seqs_to_protein_cds --> cut_mseq_and_save '''
	def cut_mseq_and_save(self, start, end, fileFasta, root_fasta, verbose=False):

		filefull = os.path.join(root_fasta, fileFasta)

		N = len(self.seq_records)
		seq_records = [self.seq_records[i][start:end] for i in range(N)]
		ret = self.writeSequences(seq_records, filefull, verbose=verbose)

		return ret


	def split_seq_to_translate(self, seq):
		seq = str(seq)
		dic_seqnuc = OrderedDict(); count = 0
		ini = 0; seq2 = ''

		for i in range(0, len(seq), 3):
			trinuc = seq[i:(i+3)]

			try:
				x = re.search("-", trinuc)
				pos = x.start()
				# print(">>>", i, trinuc, pos)
			except:
				pos = -1

			if pos >= 0:
				if seq2 != '':
					dic_seqnuc[count] = {}
					dic2 = dic_seqnuc[count]

					dic2['start'] = ini
					dic2['end']   = i-1
					dic2['seq'] = seq2
					dic2['has_trace'] = False

					count += 1

				ini = i

				dic_seqnuc[count] = {}
				dic2 = dic_seqnuc[count]

				dic2['start'] = ini
				dic2['end']   = ini+2
				dic2['seq'] = '-'
				dic2['has_trace'] = True

				count += 1
				ini += 1

				seq2 = ''
			else:
				seq2 += trinuc

		if seq2 != '':
			dic_seqnuc[count] = {}
			dic2 = dic_seqnuc[count]

			dic2['start'] = ini
			dic2['end']   = i-1
			dic2['seq'] = seq2
			dic2['has_trace'] = False
		else:
			dic_seqnuc[count] = {}
			dic2 = dic_seqnuc[count]

			dic2['start'] = ini
			dic2['end']   = i
			dic2['seq'] = '-'
			dic2['has_trace'] = True

		return dic_seqnuc


	def get_nucleotide_consensus(self):
		''' Shannon has get_nucleotide_consensus_from_seqs(seqs) - only nucleotide'''

		seqs = np.array(self.seqs, dtype=object)
		nrows, ncols = seqs.shape

		ref_consensus = ''
		for j in range(ncols):
			nuc = best_nucleotide(seqs[:,j])
			ref_consensus += '-' if nuc is None else nuc

		dif = len(ref_consensus)%3
		if dif != 0:
			ref_consensus = ref_consensus[:-dif]

		self.ref_consensus = ref_consensus
		return ref_consensus

	def get_amino_acid_consensus(self):
		''' Shannon has get_nucleotide_consensus_from_seqs(seqs) '''

		seqs = np.array(self.seqs, dtype=object)
		nrows, ncols = seqs.shape

		ref_consensus = ''
		for j in range(ncols):
			nuc = best_amino_acid(seqs[:,j])
			ref_consensus += '-' if nuc is None else nuc

		return ref_consensus

	def fulfill_seqs_with_consensus(self, seqs, seq_consensus):
		seqs = np.array(seqs, dtype=object)
		nrow = len(seqs)
		''' seq_consensus <= len(seqs[0]) '''
		ncol = len(seq_consensus)

		for i in range(nrow):
			for j in range(ncol):
				if seqs[i,j] == '-': seqs[i,j] = seq_consensus[j]

		return seqs


	def fulfill_polymorphic_seqs_with_gaps(self, seqs:List, min_gaps:float=0.25, isProtein:bool=True, min_seqs:int=3) -> List:
		if seqs is None or len(seqs) < min_seqs:
			print(">>> fulfill polymorphic_seqs: seqs is None or to little (< %d)"%(min_seqs))
			return seqs

		seqs = np.array(seqs, dtype=object)
		ncol = len(seqs[0])
		nrow = len(seqs)

		self.valid = b.getSeqAA() if isProtein else b.getDnaNucleotides()
		self.valid += '-'


		'''
			Flavio 2024-07-31
			if n <= 20 cannot define a good statistics to analize if a gap or ane error !
			20 * 0.25 = 5 residues at least, if more than 5, may be a gap! conserve the '-'

			if more than 50% conserver the gap!

			if nrow < 10: to few seqs to observer if it is a gap
		'''
		if nrow < 10:
			min_gaps = 1.0
		elif nrow <= 20:
			min_gaps *= 2
			if min_gaps > 0.5: min_gaps = 0.5

		self.min_gaps = min_gaps

		for j in range(ncol):
			seqs[:, j] = self.fulfill_polymorphic_column_with_gaps(seqs[:, j], j)

		return seqs

	''' needs self.min_gaps, self_valid with '-' '''
	def fulfill_polymorphic_column_with_gaps(self, seq_col, j):
		N = len(seq_col)
		seq_col = np.array(seq_col, dtype=object)
		seq_col[seq_col == 'X'] = '-'
		dic = char_frequency(seq_col)

		''' valid or not only 1 type of nuc '''
		if len(dic.keys()) == 1:
			return seq_col

		''' more than 1 char - is there any bad? '''
		n_invalid = 0; badkeys = []
		for c in dic.keys():
			if c not in self.valid:
				n_invalid += dic[c]
				badkeys.append(c)

		''' removing badkeys to recalculate frequencies '''
		for badk in badkeys:
			del(dic[badk])

		''' all nuc like -, N, Y, R ... '''
		if len(dic.keys()) == 0:
			return seq_col

		''' if conserved without badkeys '''
		if len(dic.keys()) == 1:
			car = list(dic.keys())[0]
			return np.array([car]*N, dtype=object)

		total = np.sum(list(dic.values()))

		''' if to few gaps '''
		if '-' in dic.keys():
			# print(">>>", j, dic['-'] / total, dic['-'] / total < self.min_gaps)
			if dic['-'] / total <= self.min_gaps:
				n_invalid += dic['-']
				del(dic['-'])

				if len(dic) == 1:
					car = list(dic.keys())[0]
					return np.array([car]*N, dtype=object)

		counts = list(dic.values())
		total  =  np.sum(counts)
		percs  = counts / total

		if '-' not in dic.keys():
			i = 0; chars = []
			for c in dic.keys():
				n = round(percs[i] * n_invalid)
				chars += [c]*(n+1)
				# print("perc", percs[i], n, [c]*(n+1))
				i += 1

			chars = random.sample(chars, len(chars))

			count = 0
			for i in range(len(seq_col)):
				letter = seq_col[i]
				if letter == '-':
					seq_col[i]  = chars[count]
					count += 1

		return seq_col

	''' needs self_valid without '-'; drop out self.replacePerc '''
	def fulfill_polymorphic_column_without_gaps(self, seq_col):
		N = len(seq_col)
		dic = char_frequency(seq_col)

		''' valid or not only 1 type of nuc '''
		if len(dic.keys()) == 1:
			return seq_col

		''' more than 1 char - is there any bad? '''
		n_invalid = 0; badkeys = []
		for c in dic.keys():
			if c not in self.valid:
				n_invalid += dic[c]
				badkeys.append(c)

		''' all nucleotides are valid'''
		if n_invalid == 0:
			return seq_col

		''' removing badkeys to recalculate frequencies '''
		for badk in badkeys:
			del(dic[badk])

		''' all nuc like -, N, Y, R ... '''
		if len(dic.keys()) == 0:
			return seq_col

		''' if conserved without badkeys '''
		if len(dic.keys()) == 1:
			car = list(dic.keys())[0]
			return np.array([car]*N, dtype=object)

		counts = list(dic.values())
		percs  = counts / np.sum(counts)

		i = 0; chars = []
		for c in dic.keys():
			n = round(percs[i] * n_invalid)
			chars += [c]*(n+1)
			i += 1

		chars = random.sample(chars, len(chars))

		count = 0
		for i in range(len(seq_col)):
			letter = seq_col[i]
			if letter not in self.valid:
				seq_col[i]  = chars[count]
				count += 1

		return seq_col


	def calc_posi(self, prot, seq_reference, is_start, start, end, large = 20, start_offset = 50, end_offset=50, printPlot=False, verbose=False):

		''' seqs os numpy array '''
		seqs = np.array(self.seqs, dtype=object)
		nrow = len(self.seqs)

		nArea = large * nrow

		noligo = len(seq_reference)   # size of given initial transcript
		Lgen   = len(self.seqs[0]) # genome Length

		length  = end - start + 1

		if Lgen < start:
			new_start = Lgen - (end - start + 50)
			print("'%s' gene/orf - Impossible to start (%d) genome length = %d, start reviewed to %d"%(prot, start, Lgen, new_start))
			start = new_start

		start_off = start - start_offset
		if start_off < 0: start_off = 0

		end_off = end + end_offset
		if end_off > Lgen: end_off = Lgen

		posis = np.arange(start_off, end_off)
		pis = []

		for i in posis:
			if i + large > Lgen:
				pis.append(0)
			else:
				tot = 0
				# print(">>>", len(self.seqs[0]), "\n>>", large, '>> seq_reference', len(seq_reference))

				for j in range(large):
					tot += np.sum( [1 if self.seqs[row][i+j] == seq_reference[j] else 0 for row in range(nrow)] )
				pis.append(tot / nArea)

		maxi = np.max(pis)
		posi = np.min([i for i in range(len(pis)) if pis[i] == maxi])

		if is_start:
			posi = posi + start_off
		else:
			posi = posi + start_off + noligo - 1

		if verbose:
			if is_start:
				print("%s starts in %d with percentage of similarity %.2f %%"%(prot, posi, maxi*100))
			else:
				print("%s ends in %d with percentage of similarity %.2f %%"%(prot, posi, maxi*100))

		if printPlot:
			plt.plot(posis, pis)
			plt.title("Genome for transcript '%s'"%(prot))
			plt.show()

		return posi

def format_fasta(seq, l=80, maxline=None, sep='\n'):
	pos = 0; line=0
	stri = ''
	while(True):
		if stri == '':
			if l > len(seq):
				return seq

			stri = seq[:l]
			pos += l; line+=1
		else:
			if pos+l > len(seq):
				stri += sep + seq[pos:]
				return stri

			stri += sep + seq[pos:(pos+l)]
			pos += l; line+=1

			if maxline and line == maxline: return stri

def break_peptides_pepdic(pepdic, L = 11):
	peptides = []

	for key in pepdic.keys():
		peptide = pepdic[key]['peptide']

		ini = 0
		while True:
			peptides.append(peptide[ini:(ini+L)])
			ini += 1
			#print(ini+L, len(peptide))
			if ini+L > len(peptide): break

	return peptides
