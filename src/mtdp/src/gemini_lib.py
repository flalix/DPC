#!/usr/bin/python
#!python
# -*- coding: utf-8 -*-
# Created on 2024/06/10
# Udated  on 2025/04/01; 2025/02/21; 2025/01/08; 2024/12/11
# @author: Flavio Lichtenstein
# @local: Bioinformatics: CENTD/Molecular Biology; Instituto Butatan


import os, sys, pickle
from typing import Optional, Iterable, Set, Tuple, Any, List

import scipy
from   scipy.stats import hypergeom

import numpy as np
import time, json
import pandas as pd
from sklearn.utils import shuffle

import re
_RE_COMBINE_WHITESPACE = re.compile(r"\s+")

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from transformers import pipeline
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

from IPython.display import Markdown

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from   matplotlib_venn import venn2, venn2_circles

import plotly.graph_objects as go
import plotly.express as px

import seaborn as sns

from Basic import *
from reactome_lib import *
from stat_lib import *


class Gemini(object):
	def __init__(self, bpx, is_seldata:bool, disease:str, context_disease:str, 
				 n_sentences:int, API_KEY:str, root0:str, chosen_model_sampling:int,
				 chosen_model_list:List=[1,3], i_dfp_list:List=[0,1,2,3],
				 run_default='run01', chosen_model_default=3, answer_min_cutoff = 0.6):

		self.disease = disease
		self.context_disease = context_disease

		self.bpx = bpx

		self.chosen_model_list = chosen_model_list
		self.chosen_model_sampling = chosen_model_sampling
		self.i_dfp_list = i_dfp_list

		self.run_default = run_default
		self.chosen_model_default = chosen_model_default

		self.API_KEY = API_KEY

		# gemini_model + API_KEYs
		self.gemini_URL0 = f"https://generativelanguage.googleapis.com/v1beta/models/%s-latest:generateContent?key=%s"

		self.gemini_models = ['gemini-1.0-pro', 'gemini-1.5-pro', 
							  'gemini-1.5-flash-8b', 'gemini-1.5-flash', 
							  'gemini-2.0-flash-exp', 
							  'gemma-2-2b-it', 'gemma-2-9b-it', 'gemma-2-27b-it']
		self.chosen_model = 0

		self.set_gemini_num_model(self.chosen_model)

		self.root0 = root0
		self.root_gemini0  = create_dir(root0, 'gemini')
		self.root_gemini   = None
		self.root_curation = create_dir(root0, 'curation')
		self.root_pubmed   = create_dir(root0, 'pubmed')
		self.root_figure0  = create_dir(root0, 'figures')

		self.is_seldata = is_seldata
		self.s_sel_all = 'selected' if is_seldata else 'all'

		self.root_pubgem		= create_dir(root0, 'pubgem')
		self.root_pubgem_pubmed = create_dir(self.root_pubgem, 'pubmed')
		self.root_pubgem_gemini = create_dir(self.root_pubgem, 'gemini')
		self.root_pubgem_figure = create_dir(self.root_pubgem, 'figures')

		self.set_gemini_root_suffix()

		self.root_figure = self.root_pubgem_figure if self.is_seldata else self.root_figure0

		''' disease, case, question_name '''
		self.fname_gemini_search	 = 'gemini_search_for_%s_%s_question_%s_model_%s.tsv'
		self.fname_gemini_search_sel = 'gemini_search_for_%s_%s_question_%s_model_%s_selected.tsv'
		self.fname_group_count		= "processed_text_group_counts_for_%s_%s_num_of_sentences_%d_sq_%s_model_%s.tsv"
		self.fname_group_summary	= "processed_text_summary_counts_for_%s_%s_num_of_sentences_%d_sq_%s_model_%s.tsv"
		self.fname_group_statistics = "processed_text_sstatistics_counts_for_%s_%s_num_of_sentences_%d_sq_%s_model_%s.tsv"

		self.fname_anal_stat_model = "gemini_stat_anal_for_%s_model_%s_%s.tsv"
		self.fname_summ_stat_model = "gemini_stat_summ_for_%s_model_%s_%s.tsv"

		# RMC - run, model, case
		self.fname_soft_OMC_per_case_idfp = 'gemini_OMC_stats_per_case_idfp_for_%s_models_%s_%s.tsv'
		self.fname_soft_OMC			   = 'gemini_OMC_for_%s_run_%s_model_%s_%s.tsv'


		self.fname_dfpiv_consensus_run_model   = 'gemini_semantic_consensus_for_%s_model_%s_run_%s_%s.tsv'
		self.fname_dfpiva_all_models_one_run   = 'gemini_semantic_consensus_for_%s_all_models_%s_run_%s_%s.tsv'
		self.fname_dfpiv_gemini_run_case_model = 'gemini_analytical_for_%s_case_%s_model_%s_run_%s_%s.tsv'

		self.fname_confusion_table_consensus  = "gemini_confusion_for_%s_Group_%s_runs_%s_models_%s_%s.tsv"
		self.fname_confusion_table_stats_cons = "gemini_confusion_table_G123_for_%s_stats_compare_runs_%s_models_%s_%s.tsv"
		self.fname_conf_table_summary_stats   = "gemini_confusion_table_G123_for_%s_stats_compare_groups_run_%s_models_%s_%s.tsv"

		self.fname_confusion_table_BCA		  = "gemini_confusion_table_BCA_for_%s_run_%s_case_%s.tsv"
		self.fname_confusion_table_all_BCA	  = 'gemini_confusion_table_stat_BCA_for_%s.tsv'

		self.fname_random_sampling = 'sampling_%d_regs_yes_no_case_%s_model_%s_query_type_%s.tsv'

		self.fname_inter_model_venn  = "inter_models_soft_detailed_for_%s_run_%s_models_%s_x_%s_%s.tsv"
		self.fname_inter_model_stat  = "inter_models_soft_stat_for_%s_models_%s_x_%s_%s_common_venn.tsv"
		self.fname_inter_model_repro = "inter_models_soft_repro_for_%s_comparing_models_%d_x_%d_%s.tsv"
		self.fname_inter_model_repro_per_idfp = "inter_models_soft_repro_per_idfp_for_%s_comparing_models_%d_x_%d_%s.tsv"
		# self.fname_inter_model_repro_idfp	 = "inter_models_soft_repro_for_%s_comparing_models_%d_x_%d_%s_%s_idfp_%d_%s.tsv"

		self.fname_hard_run_run_case		   = 'hard_run_run_repro_per_case_for_%s_runs_%s_%s_%s.tsv'
		self.fname_hard_inter_model_repro	   = 'inter_model_hard_repro_for_%s_models_%s_run_%s_%s.tsv'
		self.fname_hard_inter_model_summary	   = 'inter_model_hard_repro_summary_for_%s_models_%s_run_%s_%s.tsv'
		self.fname_hard_inter_model_case_repro = 'inter_model_hard_repro_per_case_for_%s_models_%s_run_%s_%s.tsv'

		self.fname_soft_RRCR_stats_per_idfp  = 'run_run_soft_consensus_repro_per_idfp_for_%s_between_%s_x_%s_models_%s_%s.tsv'

		self.fname_soft_RRCR		   = 'run_run_soft_consensus_repro_for_%s_between_%s_x_%s_models_%s_%s.tsv'
		self.fname_soft_RRCR_stats	   = 'run_run_soft_consensus_stats_for_%s_between_%s_x_%s_models_%s_%s.tsv'
		self.fname_anal_soft_consensus = 'analytical_soft_consensus_for_%s_%s_i_dfp_%s_models_%s_%s.tsv'

		self.fname_consensus_one_run_per_model = "gemini_run_%s_per_model_consensus_%s_%s.tsv"

		self.fname_consensus_all_models_yes	  = 'gemini_consensus_all_models_%s_yes_%s_%s.tsv'
		self.fname_consensus_all_models_no	  = 'gemini_consensus_all_models_%s_no_%s_%s.tsv'
		self.fname_consensus_all_models_doubt = 'gemini_consensus_all_models_%s_doubt_%s_%s.tsv'

		# random_not_reactome --> shuffled_reactome_not_in_enriched_pathways
		self.fname_pathways_not_in_reactome = 'shuffled_reactome_not_in_enriched_pathways.tsv'
		self.fname_all_enriched_pathways	= 'all_enriched_pathways.tsv'
		self.dfr_not_in_reactome = None

		self.fname_summary_consensus_one_run_all_models = "gemini_summary_consensus_run_%s_all_models_%s_%s_%s.tsv"
		self.fname_summary_ynd_all_runs_all_models		= "gemini_summary_yes_no_doubt_for_%s_all_runs_models_%s_%s.tsv"
		self.fname_summary_ynd_all_runs_all_models_txt  = "gemini_summary_yes_no_doubt_for_%s_all_runs_models_%s_%s.txt"
		self.fname_summary_ynd_all_runs_all_models_idfp		= "gemini_summary_yes_no_doubt_for_%s_all_runs_models_%s_idfp_%s.tsv"
		self.fname_summary_ynd_all_runs_all_models_idfp_txt = "gemini_summary_yes_no_doubt_for_%s_all_runs_models_%s_idfp_%s.txt"

		self.fname_hard_run_run_compare_total_answers = 'run_run_hard_total_answers_per_case_for_%s_runs_%s_x_%s_%s.tsv'
		self.fname_hard_run_run_stats_total_answers   = 'run_run_hard_total_answers_stats_per_case_for_%s_runs_%s_x_%s_%s.tsv'

		self.fname_all_merged_gemini_questions = 'all_merged_gemini_questions_for_%s_run_%s_x_%s_%s.tsv'


		self.lines_01 = [0, 1]
		self.lines_02 = [0, 2]
		self.lines_03 = [0, 3]
		self.lines_04 = [0, 4]
		self.cols_yn =  ['yes', 'possible', 'low_evidence', 'no']

		self.question_list = ['simple', 'simple+pubmed', 'disease', 'disease+pubmed']
		self.idfp_sufix_list = ['_0_first', '_1_middle', '_2_final', '_3_others']

		# "Explain and infere Yes, Possible, Low evidence, or No; "
		self.prefix_question = "Answer in the first line Yes, Possible, Low evidence, or No; and explain; "


		self.index_pivot = ['case', 'i_dfp', 'pathway_id', 'pathway']
		self.n_index_pivot = len(self.index_pivot)

		self.cols_pivot  = 'col_question'
		self.cols_curation = ['case', 'pathway_id', 'pathway', 'curation', 'fdr']
		self.curation_classes = ['disease', 'disease+pubmed', 'simple', 'simple+pubmed']


		if n_sentences < 1:
			n_sentences = 99
		self.n_sentences = n_sentences

		self.case, self.s_case = None, None
		self.df_enr, self.df_enr0 = None, None

		self.classifier = pipeline("sentiment-analysis")

		self.reactome = Reactome()
		self.root_reactome = self.reactome.root_reactome
		self.root_reactome_data  = self.reactome.root_reactome_data

		self.dfr = self.reactome.open_reactome_abstract()
		self.dfall_reactome = self.reactome.open_reactome()

		#============ difference between answers =============
		self.answer_weight = {}
		self.answer_weight['yes'] = 1
		self.answer_weight['pos'] = .70
		self.answer_weight['low'] = .30
		self.answer_weight['no'] = 0

		# cutoff differenc between answers to be similar answers
		self.answer_min_cutoff = answer_min_cutoff = 0.6

		self.mat_stat_tests = [('sens', 'Sensitivity'), ('spec', 'Specificity'), ('accu', 'Accuracy'), \
			   				   ('prec', 'Precision'), ('f1_score', 'F1-score')]
			   		

	def set_run_seldata(self, run:str=None):

		if run is None or run == '':
			print("Warning: setting parameter run to 'run01'")

		if self.is_seldata:
			self.root_gemini = os.path.join(self.root_pubgem_gemini, run)
		else:
			self.root_gemini = os.path.join(self.root_gemini0, run)


		if not os.path.exists(self.root_gemini):
			os.mkdir(self.root_gemini)

		return


	def set_gemini_root_suffix(self):
		'''
			Main parameter: self.is_seldata
				True: set the 2CRSP model
				False: set the Ensemble model

			Method: if self.is_seldata is True the 2CRSP model is setted; otherwise the Ensemble model.

			Inputs: uses self.is_seldata (see constructor)

			Access: None
			Save: None
			Output: None

		'''

		if self.is_seldata:
			self.root_gemini_root = self.root_pubgem_gemini
			self.suffix = 'selected_data'
		else:
			self.root_gemini_root = self.root_gemini0
			self.suffix = 'all_data'

		return

	def set_gemini_num_model(self, chosen_model, verbose:bool=False):
		'''
			Method: set gemini model by number. Needs as input: chosen_model. 
					Defines: self.chosen_model, self.gemini_model (model name), 
					self.dir_gemini (gemini draft dir to interface the web service), and 
					self.gemini_URL (the web service URL).

			Inputs:
				chosen_model: int
				verbose: bool, default=False

			Access: None
			Save: None
			Output: None
		'''

		self.chosen_model = chosen_model
		self.gemini_model = self.gemini_models[chosen_model]

		dir_gemini = f"gemini_{self.chosen_model}"
		self.dir_gemini = create_dir('.', dir_gemini)

		if verbose: print(f"chosen_model={chosen_model} - {self.gemini_model}")

		self.gemini_URL = self.gemini_URL0%(self.gemini_model, self.API_KEY)

	def set_gemini_model(self, gemini_model:str, verbose:bool=False):
		'''
			Method: set gemini model by name. Needs as input: gemini_model (name).
			Finds the chosen model, call set_gemini_num_model()

			Inputs:
				gemini_model: str
				verbose: bool, default=False

			Access: None
			Save: None
			Output: None

		'''

		mat = [i for i in range(len(self.gemini_models)) if self.gemini_models[i] == gemini_model]
		if mat == []:
			print("Error: set_gemini_model()", gemini_model)
			raise Exception("stop: set_gemini_model()")

		self.set_gemini_num_model(chosen_model=mat[0], verbose=verbose)


	def set_case(self, case:str, df_enr:pd.DataFrame, df_enr0:pd.DataFrame):
		if self.disease == 'COVID-19':
			self.set_case_covid(case, df_enr, df_enr0)
		elif self.disease == 'medulloblastoma':
			self.set_case_medulloblastoma(case, df_enr, df_enr0)
		else:
			print(f"Please develop the cases for disease {self.disease}")
			raise Exception('stop')

	def set_case_medulloblastoma(self, case:str, df_enr:pd.DataFrame, df_enr0:pd.DataFrame):
		self.case = case

		if case == 'WNT':
			self.s_case = 'type WNT'
		elif case == 'SHH':
			self.s_case = 'type SHH or Hedgehog'
		elif case == 'G4':
			self.s_case = 'Group 4 or G4'
		elif case == 'G3':
			self.s_case = 'Group 3 or G3'
		else:
			self.s_case = case
			print("Error: could not define the case {case} for disease {self.disease}")
			raise Exception('stop')

		self.df_enr  = df_enr
		self.df_enr0 = df_enr0


	def set_case_covid(self, case:str, df_enr:pd.DataFrame, df_enr0:pd.DataFrame):
		self.case = case

		if case   == 'g1_male':
			self.s_case = 'male adult asymptomatic'
		elif case == 'g1_female':
			self.s_case = 'female adult asymptomatic'
		elif case == 'g2a_male':
			self.s_case = 'male adult mild outpatient'
		elif case == 'g2a_female':
			self.s_case = 'female adult mild outpatient'
		elif case == 'g2b_male':
			self.s_case = 'male adult moderate outpatient'
		elif case == 'g2b_female':
			self.s_case = 'female adult moderate outpatient'
		elif case == 'g3_male_adult':
			self.s_case = 'male adult severe ICU'
		elif case == 'g3_female_adult':
			self.s_case = 'female adult severe ICU'
		elif case == 'g3_male_elder':
			self.s_case = 'male elder severe ICU'
		elif case == 'g3_female_elder':
			self.s_case = 'female elder severe ICU'
		else:
			self.s_case = case
			print("Error: could not define the case {case} for disease {self.disease}")
			raise Exception('stop')


		self.df_enr  = df_enr
		self.df_enr0 = df_enr0


	def prepare_abstract_n_sentences(self, text:str) -> str:

		lista = re.findall(r"[0,9]+\.[0.9]+", text)
		if isinstance(lista, list):
			for word in lista:
				text = text.replace(word, word.replace('.','pt'))

		text = text.replace('<br>','').replace('<b>','').replace('</b>','').replace("**","")
		text = text.replace('?','?.').replace('!','!.').replace(':\n',':.')
		text = text.replace('\n',' ')

		text = _RE_COMBINE_WHITESPACE.sub(" ", text)

		mat = text.split(".")
		mat = [x.strip() for x in mat]
		text = ". ".join(mat[:self.n_sentences]).strip()
		text = text.replace('?.','?').replace('!.','!').replace(':.',':')
		last_char = text[-1]
		return text if last_char in ['.',':','?','!'] else text + '.'


	def pick_other_pahtways(self) -> pd.DataFrame:
		'''
		remove all df_enr0
		
		build a fake list of pathways of the same size
		n = len(self.df_enr)

		change parameters
		return as dfp4
		'''

		# same length of df_enr
		dfnew = self.df_enr.copy()
		n = len(dfnew)

		if self.dfr_not_in_reactome is None or self.dfr_not_in_reactome.empty:
			print(f"Could not find self.dfr_not_in_reactome - '{self.fname_pathways_not_in_reactome}'\n\n")
			raise Exception('stop')

		if n > len(self.dfr_not_in_reactome):
			print("Warning: there are on {len(self.dfr_not_in_reactome)} not Reactome pathways, needed {len(self.bpx.df_enr)}")
			n = len(self.dfr_not_in_reactome)

		pathway_id_list = self.dfr_not_in_reactome.loc[:n, 'pathway_id']
		pathway_list	= self.dfr_not_in_reactome.loc[:n, 'pathway']

		dfnew.pathway_id = pathway_id_list
		dfnew.pathway = pathway_list
		dfnew.fdr = 1
		dfnew.pval = 1
		dfnew.odds_ratio = None
		dfnew.combined_score = None
		dfnew.genes = None
		dfnew.num_of_genes = 0

		if dfnew.empty:
			print("----------------")
			print("dfnew is empty!!!")
			print("----------------")
			raise Exception('Stop dfnew')
		
		dfnew.index = np.arange(len(dfnew))
		self.cur_pathway_id_list += list(dfnew.pathway_id)

		return dfnew


	def find_pathways_not_in_Reactome_and_shuffle(self, case_list:List, force:bool=False, verbose:bool=False) -> bool:
		#-- create the random Reactome table without calculated pathways -----------
		self.df_all_pathways = None

		fname = self.fname_all_enriched_pathways
		filename = os.path.join(self.root_gemini_root, fname)

		if os.path.exists(filename) and not force:
			df_all_pathways = pdreadcsv(fname, self.root_gemini_root, verbose=verbose)
		else:
			df_all_pathways = None

		if df_all_pathways is None or df_all_pathways.empty:
			df_list = []
			for case in case_list:

				ret, _, _, _ = self.bpx.open_case(case)
				if self.bpx.df_enr is None or self.bpx.df_enr.empty:
					continue

				df_list.append(self.bpx.df_enr0)

			if len(df_list) == 0:
				print("Could not find any pathway!!! Why???")
				return False

			df_all_pathways = pd.concat(df_list)
			cols = ['pathway_id', 'pathway']
			df_all_pathways = df_all_pathways[cols]
			df_all_pathways = df_all_pathways.drop_duplicates()
			df_all_pathways.index = np.arange(len(df_all_pathways))

			ret = pdwritecsv(df_all_pathways, fname, self.root_gemini_root, verbose=verbose)

		print(f"Found {len(df_all_pathways)} pathways for all cases")
		self.df_all_pathways = df_all_pathways

		fname = self.fname_pathways_not_in_reactome
		filename = os.path.join(self.root_gemini_root, fname)

		if os.path.exists(filename) and not force:
			dfr_not_in_reactome = pdreadcsv(fname, self.root_gemini_root, verbose=True)
			self.dfr_not_in_reactome = dfr_not_in_reactome

			return dfr_not_in_reactome is not None and not dfr_not_in_reactome.empty

		dfr_not_in_reactome = self.dfall_reactome[~ self.dfall_reactome.pathway_id.isin(self.df_all_pathways.pathway_id)].copy()
		dfr_not_in_reactome.index = np.arange(len(dfr_not_in_reactome))

		#--- shuffle to pick n pathways
		lista = list(dfr_not_in_reactome.index)
		np.random.shuffle(lista)

		dfr_not_in_reactome = dfr_not_in_reactome.loc[lista]
		dfr_not_in_reactome.index = np.arange(len(dfr_not_in_reactome))
		ret = pdwritecsv(dfr_not_in_reactome, fname, self.root_gemini_root, verbose=verbose)

		self.dfr_not_in_reactome = dfr_not_in_reactome

		return dfr_not_in_reactome is not None and not dfr_not_in_reactome.empty


	def run_all_gemini(self, run:str, case_list:List, chosen_model:int, num_tries:int=5, pause_secs:int=1, 
					   temperature:float=0.1, topK:int=50, topP:float=0.10, 
					   maxOutputTokens:int=4096, stopSequences:List=[], 
					   force:bool=False, verbose:bool=False):
		'''
			Method: acces Gemini web service and stores all queries’ responses in a dataframe. 
			Run all cases, all type of questions (iq), all pathway groups (i_dfp_list). 
			Needs as input: run, chosen model, num_tries, pause_secs

			Inputs:
				run: str
				chosen_model: int
				num_tries: int, default=3 
				pause_secs: int, default=0
				false: bool, default=False
				verbose: bool, default=False


			Access:
				table: 
					root: root_data/gemini
					fname:  shuffled_reactome_not_in_enriched_pathways.tsv

			Save:
				table: 
					root: root_data/gemini/run0x/
					fname:  gemini_search_for_<disese>_<case>_question_<iq>_<i_dfp>_model_<model>.tsv

			Output:
				None

		'''

		self.set_run_seldata(run)

		self.set_gemini_num_model(chosen_model)
		print(">> Gemini model:", self.gemini_model)

		self.dfr = self.reactome.open_reactome_abstract(verbose=verbose)

		if self.API_KEY == '<my_gemini_key>':
			print("Please, configure API_KEY in params.yml")
			return

		if not self.find_pathways_not_in_Reactome_and_shuffle(case_list=case_list):
			print("Aborting, could not create random Reactome list.\n\n")
			raise Exception("stop")

		'''----- start main loop ------------------'''
		dfp_list = []
		for case in case_list:
			print("\n\n>>> case", case)

			#------- default - normal cutoff --------------------------
			# ret, _, _, _ = self.bpx.open_case_params(case, abs_lfc_cutoff=1, fdr_lfc_cutoff=0.05, pathway_fdr_cutoff=0.05)
			# dfp0 = self.bpx.df_enr
			
			#------- BCA - best cutoff algorithm ----------------------
			ret, _, _, _ = self.bpx.open_case(case)
			self.set_case(self.bpx.case, self.bpx.df_enr, self.bpx.df_enr0)

			dfp1 = self.bpx.df_enr
			dfp_list.append(dfp1)
			self.cur_pathway_id_list = list(dfp1.pathway_id)
			df_enr0 = self.bpx.df_enr0
			
			n = len(dfp1)
			N = len(df_enr0)

			
			#-- calc the middle
			n2 = int(n/2)
			N2 = int(N/2)
			
			ini = N2-n2
			end = ini+n

			if ini <= n:
				ini = n+1
				end = ini + n

			end_middle = end
			
			dfp2 = df_enr0.iloc[ini:end].copy()
			dfp2 = dfp2[~dfp2.pathway_id.isin(self.cur_pathway_id_list)]
			dfp_list.append(dfp2)

			if dfp2.empty:
				print("----------------")
				print("dfp2 is empty!!!")
				print("----------------")
				raise Exception('Stop dfp2')

			dfp2.index = np.arange(len(dfp2))
			self.cur_pathway_id_list += list(dfp2.pathway_id)

			# calc the end
			ini = N-n
			end = N

			if ini <= end_middle:
				ini = end_middle + 1
				
			dfp3 = df_enr0.iloc[ini:end].copy()
			dfp3 = dfp3[~dfp3.pathway_id.isin(self.cur_pathway_id_list)]
			dfp_list.append(dfp3)

			if dfp3.empty:
				print("----------------")
				print("dfp3 is empty!!!")
				print("----------------")
				raise Exception('Stop dfp3')
			
			dfp3.index = np.arange(len(dfp3))
			self.cur_pathway_id_list += list(dfp3.pathway_id)

			# below self.pick_other_pahtways()
			dfp4 = None 
			dfp_list.append(dfp4)

			for iq, quest_type in enumerate(self.question_list):

				print(f"\t$$$ iq={iq} {quest_type:20}")

				quest_ptw_dis_case, with_without_PubMed, suffix2 = self.define_question(quest_type)

				question_name1 = f'{with_without_PubMed}_{suffix2}_0_first'
				question_name2 = f'{with_without_PubMed}_{suffix2}_1_middle' 
				question_name3 = f'{with_without_PubMed}_{suffix2}_2_final'
				question_name4 = f'{with_without_PubMed}_{suffix2}_3_others'

				multiple_data  = [ [0, question_name1, quest_ptw_dis_case, dfp1], [1, question_name2, quest_ptw_dis_case, dfp2], 
								   [2, question_name3, quest_ptw_dis_case, dfp3], [3, question_name4, quest_ptw_dis_case, dfp4]]


				self.run_question_gemini(iq=iq, multiple_data=multiple_data, run=run, case=case, chosen_model=chosen_model,
										 num_tries=num_tries, pause_secs=pause_secs,
										 temperature=temperature, topK=topK, topP=topP,
										 maxOutputTokens=maxOutputTokens, stopSequences=stopSequences, 
										 force=force, verbose=False)
			
				# print(f"\n------------- end quest_type {quest_type} --------------\n\n")
		print("-------------- final end --------------")


	def define_question(self, quest_type:str):
		'''
			quest_type: 'simple', 'simple+pubmed', 'disease', 'disease+pubmed's
		'''

		suffix0 = 'yes_no_possible_low_evidence_studies_of_modulated_pathways'
		suffix1 = 'yes_no_possible_low_evidence_question_strong_relationship_in_studies_of_modulated_pathways'

		if quest_type == 'simple':

			quest_ptw_dis_case = f"is the pathway '%s' studied about the disease {self.disease} for {self.s_case}?"
			with_without_PubMed = 'without_PubMed'
			_suffix = suffix0
		
		elif quest_type == 'simple+pubmed':

			quest_ptw_dis_case = f"is the pathway '%s' studied in PubMed, about the disease {self.disease} for {self.s_case}?"
			with_without_PubMed = 'with_PubMed'
			_suffix = suffix0
		
		elif quest_type == 'disease':

			quest_ptw_dis_case = f"has the pathway '%s' a strong relationship in studies related to the disease {self.disease} for {self.s_case}?"
			with_without_PubMed = 'without_PubMed'
			_suffix = suffix1
		
		else:
			quest_ptw_dis_case = f"has the pathway '%s' a strong relationship in studies found in PubMed, related to the disease {self.disease} for {self.s_case}?"
			with_without_PubMed = 'with_PubMed'
			_suffix = suffix1

		return quest_ptw_dis_case, with_without_PubMed, _suffix



	def return_questions(self, quest_type:str):
		quest_ptw_dis_case, with_without_PubMed, _suffix = self.define_question(quest_type)

		if self.is_seldata:
			question_name0 = f'{with_without_PubMed}_{_suffix}_0_selected'
			question_list = [question_name0]
		else:
			question_name1 = f'{with_without_PubMed}_{_suffix}_0_first'
			question_name2 = f'{with_without_PubMed}_{_suffix}_1_middle' 
			question_name3 = f'{with_without_PubMed}_{_suffix}_2_final'
			question_name4 = f'{with_without_PubMed}_{_suffix}_3_others'

			question_list = [question_name1, question_name2, question_name3, question_name4]

		return quest_ptw_dis_case, with_without_PubMed, _suffix, question_list


	def set_gemini_fname(self, run:str, case:str, iq:int, i_dfp:int, chosen_model:int, verbose:bool=False) -> str:

		if i_dfp < 0 or i_dfp > 3:
			print("Error: i_dfp must be [0,3]")
			return None

		self.set_run_seldata(run)
		self.set_gemini_num_model(chosen_model)

		ret, _, _, _ = self.bpx.open_case(case, verbose=verbose)
		self.set_case(self.bpx.case, self.bpx.df_enr, self.bpx.df_enr0)

		quest_type = self.question_list[iq]
		quest_ptw_dis_case, with_without_PubMed, suffix2, question_list = self.return_questions(quest_type)

		question_name = question_list[i_dfp]

		if self.is_seldata:
			fname = self.fname_gemini_search_sel%(self.disease, case, question_name, self.gemini_model)
		else:
			fname = self.fname_gemini_search%(self.disease, case, question_name, self.gemini_model)

		fname = title_replace(fname)

		return fname


	def one_gemini_fname(self, run:str, case:str, iq:int, i_dfp:int, chosen_model:int,
						 verbose:bool=False) -> pd.DataFrame:

		fname = self.set_gemini_fname(run, case, iq, i_dfp, chosen_model, verbose=verbose)
		if fname is None:
			return None

		filename = os.path.join(self.root_gemini, fname)

		if os.path.exists(filename):
			df_read = pdreadcsv(fname, self.root_gemini, verbose=verbose)
		else:
			df_read = None

		return df_read




	def run_question_gemini(self, iq:int, multiple_data:List, run:str, case:str, chosen_model:int, 
							num_tries:int=5, pause_secs:int=1,
							temperature:float=0.1, topK:int=50, topP:float=0.10, 
							maxOutputTokens:int=4096, stopSequences:List=[], 
							force:bool=False, verbose:bool=False):

		for i_dfp, question_name, quest_ptw_dis_case, dfp in multiple_data:

			# if i_dfp == 0 and dfp is None:
			#	print("No enrichment analysis for default params.")
			#	continue

			if i_dfp < 3 and dfp is None:
				print(f"\nError: dfp {i_dfp} - None")
				raise Exception('stop: run_question_gemini()')


			df_read, fname = self.read_or_build_df_read(run=run, case=case, iq=iq, i_dfp=i_dfp, dfp=dfp, chosen_model=chosen_model, verbose=verbose)

			dfa = self.call_gemini(quest_ptw_dis_case=quest_ptw_dis_case, df_read=df_read, fname=fname, 
								   num_tries=num_tries, pause_secs=pause_secs,
								   temperature=temperature, topK=topK, topP=topP,
								   maxOutputTokens=maxOutputTokens, stopSequences=stopSequences, 
								   force=force, verbose=verbose)

			print(f"\t\tdfp {i_dfp} #", len(dfa))

			if dfa is None or dfa.empty:
				print("Warning: empty?")
		return


	def run_all_selected_gemini(self, run:str, case:str, i_dfp_list:List, chosen_model:int,
								N:int, query_type:str='_strong',
								num_tries:int=5, pause_secs:int=1, 
								temperature:float=0.1, topK:int=50, topP:float=0.10, 
								maxOutputTokens:int=4096, stopSequences:List=[], 
								force:bool=False, verbose:bool=False):
		'''
			run selected cases
			save on pubmed dir
			there is only one dfp ~ dfs (selected)
		'''

		self.set_run_seldata(run)

		if not os.path.exists(self.root_gemini):
			os.mkdir(self.root_gemini)

		self.set_gemini_num_model(chosen_model)
		print(">> Gemini model:", self.gemini_model)

		self.dfr = self.reactome.open_reactome_abstract(verbose=verbose)

		ret, _, _, _ = self.bpx.open_case(case)
		self.set_case(self.bpx.case, self.bpx.df_enr, self.bpx.df_enr0)

		dfsel = self.open_yes_no_sampling(case=case, N=N, query_type=query_type, verbose=verbose)
		if dfsel is None or dfsel.empty:
			return

		i_dfp=0
		for iq, quest_type in enumerate(self.question_list):

			quest_ptw_dis_case, with_without_PubMed, suffix2 = self.define_question(quest_type)

			question_name0 = f'{with_without_PubMed}_{suffix2}_0_selected'

			multiple_data  = [ question_name0, quest_ptw_dis_case, dfsel]

			dfa = self.run_question_selected_gemini(iq=iq, multiple_data=multiple_data, run=run, case=case, 
													i_dfp=i_dfp, chosen_model=chosen_model,
													num_tries=num_tries, pause_secs=pause_secs,
													temperature=temperature, topK=topK, topP=topP,
													maxOutputTokens=maxOutputTokens, stopSequences=stopSequences, 
													force=force, verbose=False)

			print(f"\t### iq={iq} {quest_type:20}   dfp {i_dfp} # {len(dfa)}")

		
		print("\n-------------- final end --------------\n")

	def run_question_selected_gemini(self, iq:int, multiple_data:List, 
									 run:str, case:str, chosen_model:int, i_dfp:int,
									 num_tries:int=5, pause_secs:int=1,
									 temperature:float=0.1, topK:int=50, topP:float=0.10, 
									 maxOutputTokens:int=4096, stopSequences:List=[], 
									 force:bool=False, verbose:bool=False) -> pd.DataFrame:


		question_name, quest_ptw_dis_case, dfsel = multiple_data


		df_read, fname = self.read_or_build_df_read(run=run, case=case, iq=iq, i_dfp=i_dfp, dfp=dfsel, chosen_model=chosen_model, verbose=verbose)

		dfa = self.call_gemini(quest_ptw_dis_case=quest_ptw_dis_case, df_read=df_read, fname=fname, 
							   num_tries=num_tries, pause_secs=pause_secs,
							   temperature=temperature, topK=topK, topP=topP,
							   maxOutputTokens=maxOutputTokens, stopSequences=stopSequences, 
							   force=force, verbose=verbose)

		if dfa is None or dfa.empty:
			print("Warning: empty?")

		return dfa


	def read_gemini(self, run:str, case:str, iq:int, i_dfp:int, chosen_model:str, 
					verbose:bool=False) -> pd.DataFrame:
		'''
			Method: read gemini file that stores the results of the web service, 
			needs as input: run, case, iq, i_dfp, and gemini model

			Inputs:
				run: str
				case: str
				iq: int
				i_dfp: int
				chosen_model: str
				verbose: bool

			Access:
				table: gemini/run0x/gemini_search_for_<disese>_<case>_<iq>_<idfp>_model_gemini-<model>.tsv

			Output:
				table: as df
		'''

		fname = self.set_gemini_fname(run, case, iq, i_dfp, chosen_model, verbose=verbose)
		if fname is None:
			return None

		filename = os.path.join(self.root_gemini, fname)

		if os.path.exists(filename):
			df = pdreadcsv(fname, self.root_gemini, verbose=verbose)
		else:
			if verbose: print(f"Warning: there is no file '{filename}'")
			df = pd.DataFrame()

		return df

	# report_gemini
	def calc_analytical_soft_consensus(self, run:str, case_list:List, i_dfp_list:List, chosen_model_list:List, 
					  				   force:bool=False, save_files:bool=False, verbose:bool=False) -> (str, pd.DataFrame):

		fname = self.fname_anal_soft_consensus%(self.disease, run, i_dfp_list, chosen_model_list, self.suffix)
		filename = os.path.join(self.root_gemini_root, fname)

		fname_txt = fname.replace('.tsv', '_msg.txt')
		filename_txt = os.path.join(self.root_gemini_root, fname_txt)

		if os.path.exists(filename) and os.path.exists(filename_txt) and not force:
			df = pdreadcsv(fname, self.root_gemini_root, verbose=verbose)
			msg = read_txt(fname_txt, self.root_gemini_root, verbose=verbose)
			return msg, df


		dfpiva = self.open_gemini_dfpiva_all_models_one_run(run=run, chosen_model_list=chosen_model_list, verbose=verbose)

		msg = f"Consensuses for run {run} and Gemini models: {str(chosen_model_list)}"

		dic = {}; icount=-1
		for i_dfp in i_dfp_list:
			for case in case_list:

				df2, df_yes, df_no, df_doubt = self.gemini_consensus_yes_no_doubt(dfpiva=dfpiva, chosen_model_list=chosen_model_list, 
																				  case=case, i_dfp=i_dfp, save_files=save_files, verbose=verbose)

				dfs = [df_yes, df_no, df_doubt]

				for i in range(len(dfs)):
					   
					df = dfs[i]
					n = len(df)

					if i == 0:
						consensus = 'Yes'
						nYes = n		
					elif i == 1:
						consensus = 'No'
						nNo = n
					else:
						consensus = 'Doubt'
						nDoubt = n

					icount += 1
					dic[icount] = {}
					dic2 = dic[icount]

					dic2['run']   = run
					dic2['case']  = case
					dic2['i_dfp'] = i_dfp
					dic2['consensus'] = consensus
					dic2['n'] = n

				ntot = nYes + nNo + nDoubt
				perYes = nYes / ntot
				
				msg += '\n' + f"\tfor case '{case}' i_dfp={i_dfp}, \tthere are {ntot} pathways, consensus Yes={nYes}, Doubt={nDoubt}, perc Yes={100.*perYes:.1f}%"

		df = pd.DataFrame(dic).T


		ret1 = pdwritecsv(df, fname, self.root_gemini_root, verbose=verbose)
		ret2 = write_txt(msg, fname_txt, self.root_gemini_root, verbose=verbose)
		
		return msg, df


	def read_or_build_df_read(self, run:str, case:str, iq:int, i_dfp:int, chosen_model:int, dfp:pd.DataFrame,
							  verbose:bool=False) -> (pd.DataFrame, str):

		fname = self.set_gemini_fname(run, case, iq, i_dfp, chosen_model, verbose=verbose)
		if fname is None:
			return None, None

		# print(i_dfp, self.root_gemini, fname)
		filename = os.path.join(self.root_gemini, fname)

		if os.path.exists(filename):
			df_read = pdreadcsv(fname, self.root_gemini, verbose=verbose)

			if df_read is not None and not df_read.empty:

				dfa = df_read[pd.isnull(df_read.case)]
				if not dfa.empty:
					print(f">>> some cases null for {case}")
					df_read['case'] = case
					ret = pdwritecsv(df_read, fname, self.root_gemini, verbose=verbose)

				return df_read, fname

		if i_dfp == 3:
			dfa = self.pick_other_pahtways()
		else:
			dfa = dfp

		cols = ['pathway_id', 'pathway', 'fdr']
		dfa = dfa[cols].copy()

		dfa['curation'] = None
		dfa['response_explain'] = None
		dfa['score_explain'] = None
		dfa['question'] = None
		dfa['disease']  = self.disease
		dfa['case']	 = case
		dfa['s_case']   = self.s_case
		dfa['pathway_found'] = False

		dfa = dfa.sort_values('fdr', ascending=True)
		dfa.index = np.arange(len(dfa))

		ret = pdwritecsv(dfa, fname, self.root_gemini, verbose=verbose)

		return dfa, fname


	def call_gemini(self, quest_ptw_dis_case:str, df_read:pd.DataFrame, fname:str,
					num_tries:int=3, pause_secs:int=1, num_words:int=128, 
					temperature:float=0.1, topK:int=50, topP:float=0.10, 
					maxOutputTokens:int=4096, stopSequences:List=[], 
					force:bool=False, verbose:bool=False) -> pd.DataFrame:
		'''
		dfp: selected pathways
		dfr: reactome (like) table - with 'pathway', 'pathway_id', and 'abstract'
		'''

		if df_read is None or df_read.empty:
			print('Error: df_read is none or empty!')
			return None

		dfa = df_read[ (df_read.pathway_found==False) | (pd.isnull(df_read.curation)) | \
					   (pd.isnull(df_read.response_explain)) ].copy()

		if dfa.empty:
			if verbose: print(f"Already calculated {len(dfa)} regs.")
			return df_read

		dfa.index = np.arange(len(dfa))

		print(f" (#{len(dfa)}):", end=' ')

		dic = {};icount=-1

		if self.dfr is None:
			self.dfr = self.reactome.open_reactome_abstract()

		if self.dfr is None:
			print("Problems with self.dfr (reactome)")
			raise Exception('stop: dfr (reactome) is empty')

		lista_pathways = []
		for i_try in range(num_tries):

			print("try", i_try, end=': ')

			for irow in range(len(dfa)):
				row		   = dfa.loc[irow]
				pathway_id = row.pathway_id

				if not isinstance(pathway_id, str) or pathway_id == '':
					continue
				
				pathway_id = pathway_id.strip()

				if pathway_id in lista_pathways:
					# already done
					continue

				''' --- getting abstract --------------'''
				dfr = self.dfr[self.dfr.pathway_id == pathway_id]
				if dfr.empty:
					if i_try == 0: print(f"\nError: pathway_id {pathway_id} not found in reactome dfr.")
					continue

				pathway	 = dfr.iloc[0].pathway
				ptw_abst = dfr.iloc[0].abstract
				ptw_abst = self.prepare_abstract_n_sentences(ptw_abst)

				if ptw_abst[-1] == '.':
					ptw_abst = ptw_abst[:-1]


				response_curation, response_explain, score_explain = None, None, None

				s_question0 = quest_ptw_dis_case%(pathway)

				question = self.prefix_question + s_question0 + f" Context: {ptw_abst}. {self.context_disease}"
				question = question.strip()

				print(irow, end='')
				if verbose: print("\n>>> question:", question)

				list_candidates = self.run_curl_gemini(question, temperature=temperature, topK=topK, topP=topP,
													   maxOutputTokens=maxOutputTokens, stopSequences=stopSequences,
													   verbose=verbose)
				time.sleep(pause_secs)

				if list_candidates == []:
					if verbose: print(f"Error: {question} pathway_id {pathway_id}")
					print(' ', end='')
					continue

				response = self.response_candidate(list_candidates, 0)
				if response is None:
					print(' ', end='')
					continue

				print('*', end=' ')

				response = response.strip()
				self.response = response

				response_curation = response

				if response_curation.startswith('**'):
					response_curation = response_curation.replace('**', '')

				response_curation = response_curation.split('\n')[0].strip()
	
				if response_curation.lower().startswith('yes'):
					response_curation = 'Yes'
				elif response_curation.lower().startswith('no'):
					response_curation = 'No'
				elif response_curation.lower().startswith('low'):
					response_curation = 'Low'
				elif response_curation.lower().startswith('possible'):
					response_curation = 'Possible'
				else:
					response_curation = response_curation.split(';')[0].strip()
					response_curation = response_curation.split('.')[0].strip()
					response_curation = response_curation.split(',')[0].strip()
					response_curation = response_curation.split(' ')[0].strip()					

				response_explain = response
				score_explain	= self.mean_classifier(response_explain, num_words=num_words)

				lista_pathways.append(pathway_id)

				dfa.loc[irow, 'curation'] = response_curation
				dfa.loc[irow, 'response_explain'] = response_explain
				dfa.loc[irow, 'score_explain'] = score_explain
				dfa.loc[irow, 'question'] = question
				dfa.loc[irow, 'pathway_found'] = True

			dfa2 = dfa[ (dfa.pathway_found==False) ]
			# if nothing more todo, break tries
			if dfa2.empty:
				break

		print("")
		# last df_read
		df_read = df_read[ (df_read.pathway_found==True) ]
		if df_read.empty:
			df_read = dfa
		else:
			# concat with last df_read
			df_read = pd.concat([df_read, dfa])

		df_read = df_read.sort_values('fdr', ascending=True)
		df_read.index = np.arange(len(df_read))

		ret = pdwritecsv(df_read, fname, self.root_gemini, verbose=verbose)

		return df_read


	def print_one_row_gemini_results(self, dfa:pd.DataFrame, i:int, which:str='with_pathway', verbose:bool=False) -> str:
		'''
			print_one_row_gemini_resultss returns the text results from dfa record i
			which='with_pathway', 'without_pathway', 'both'
		'''
		text = ''

		row = dfa.iloc[i]
		if row.empty:
			return ''

		pathway_id = row.pathway_id
		pathway	= row.pathway

		stri = f"{i}) {pathway_id}: {pathway}"
		len_stri = len(stri)
		text += stri + '\n'
		text += "-"*len_stri + '\n'
		text += row.question + '\n'

		if which == 'with_pathway' or which == 'both':
			stri = row.response_curation
			if isinstance(stri, str):
				stri = stri.replace("**","")
				text += ">> with pathway context:" + stri
			else:
				text += ">> with pathway context: Error!"
				''' try without '''
				which = 'both'

			text += '\n\n'

		if which == 'without_pathway' or which == 'both':
			stri = row.response_without_ptw_context
			if isinstance(stri, str):
				stri = stri.replace("**","")
				text += ">> wo pathway context:" + stri
			else:
				try:
					''' if stri is a dictionary '''
					x = stri['error']
					text += ">> wo pathway context: Error! Looks like a Gemini's internal error." + '\n'
				except:
					text += ">> wo pathway context: Error!" + '\n'

			text += '\n\n'

		if verbose: print(text)

		return text

	def print_gemini_results(self, dfa, which='both'):
		text = ''
		for i in range(len(dfa)):
			text += self.print_one_row_gemini_results(dfa, i, which=which)

		return text



	def run_curl_gemini(self, question:str, temperature:float=0.1, topK:int=50, 
						topP:float=0.10, maxOutputTokens:int=4096, stopSequences:List=[], 
						verbose:bool=False) -> List:

		if self.API_KEY == '<my_gemini_key>':
			print("Please, configure API_KEY in params.yml")
			return []

		question = question.replace('"','').replace("'","").replace("\n"," ")


		fname_json = f"{self.dir_gemini}/response.json"
		fname_cmd  = f"{self.dir_gemini}/run_curl.sh"

		curl_msg =  "curl -H 'Content-Type: application/json' -d " + \
					"'{\"contents\":[{\"parts\":[{\"text\":\"{" + question + "}\"}]}], " + \
					"\"generationConfig\":{" +  \
				   f"\"temperature\":{temperature}, \"topK\":{topK},\"topP\":{topP}, " + \
				   f"\"maxOutputTokens\":{maxOutputTokens},\"stopSequences\":{str(stopSequences)}" +  \
					"}" +  \
					"}' -X POST '{" + self.gemini_URL + "}' --silent >> " + fname_json

		if verbose:
			print(">>> curl_msg")
			print(curl_msg)
			print("")

		self.curl_msg = curl_msg


		if os.path.exists(fname_json):
			os.remove(fname_json)

		# write the command line
		try:
			write_txt(curl_msg, fname_cmd)
			os.system(f"chmod +x '{fname_cmd}'")
		except ValueError:
			s_error = f"Error: writing '{ValueError}' in 'run_curl.sh' in run_curl_gemini()"
			print(s_error)
			return []

		response = None; dic={}

		try:
			os.system(f"{fname_cmd}")

			with open(fname_json) as f:
				dic = json.load(f)

		except ValueError:
			dic = None
			s_error = f"Error: calling gemini '{ValueError}'"
			if verbose: print(s_error)
			return []

		self.dic = dic
		try:
			list_candidates = dic['candidates']
		except:
			error_code = self.dic['error']['code']
			error_msg  = self.dic['error']['message']
			s_error = f"Error: calling gemini '{error_code}: {error_msg}'"
			if verbose:
				print(s_error)
				print("---------------------")
				print(curl_msg)
				print("---------------------")
			list_candidates = []

		return list_candidates

	def response_candidate(self, list_candidates:List, n_candidate:int=0, verbose:bool=False) -> str:

		try:
			parts = list_candidates[n_candidate]['content']['parts']

			text = ''
			for i in range(len(parts)):
				text += parts[i]['text'] + '\n'
		except:
			s_error = "Error in obtaining text in response (list_candidates)"
			if verbose: print(s_error, end=' ')
			return None

		return text



	def mean_classifier(self, text:str, num_words:int=128):
		mat = text.split(" ")
		N = len(mat)

		lista = []
		iloop = -1

		while(True):
			iloop += 1
			ini = iloop*num_words
			if ini >= N: break
			end = (iloop+1)*num_words
			text2 = " ".join(mat[ini:end])
			# print(iloop, N, ini, end)
			lista.append(text2)

		ret = self.classifier(lista)
		scores = []
		for dic in ret:
			signal = -1 if dic['label'] == 'NEGATIVE' else +1
			score  = signal*dic['score']
			scores.append(score)

		score = scores[0] if len(scores)==1 else np.mean(scores)
		return score

	def classifier_my(self, text):
		if text is None:
			return 0., None

		text = self.prepare_abstract_n_sentences(text)
		return self.mean_classifier(text), text

	def classifier_list(self, lista):
		if not isinstance(lista, list):
			print('Please, pass a list into classifier_list()')
			return []

		results = []
		for text in lista:
			if text is None or text == '':
				mu_text = [0.0, text]
			else:
				text = self.prepare_abstract_n_sentences(text)
				mu_text = [self.mean_classifier(text), text]

			results.append(mu_text)

		return results


	def calc_hypergeometric_per_group(self, verbose:bool=False) -> (bool, pd.DataFrame):

		true_cases = self.dfpiv.iloc[:,1].to_list()

		M = len(self.df_enr0)
		N = len(self.dff)

		print(f"Since the Enrichment Analysis has {M} pathways and we selected {N} pathways,")

		dic = {}
		for i in range(len(self.question_orders)):
			s_order = self.question_orders[i]

			dic[i] = {}
			dic2 = dic[i]

			question_num = i + 1
			dfgroup = self.dff[self.dff.question_num == question_num]
			n = len(dfgroup)

			k = true_cases[i]

			p = 1-hypergeom.cdf(k, M, n, N)

			if p < 0.05/3:
				stat_inf = "It is very improbable, we must reject H0"
			else:
				stat_inf = "It is probable, we must accept H0"

			s_stat = f"{s_order} has {k} positive cases in {n}, hypergeometric p = {100*p:.2f}%. {stat_inf}"

			dic2['question_num']   = question_num
			dic2['question_order'] = s_order
			dic2['s_stat'] = s_stat
			dic2['pval'] = p
			dic2['k'] = k
			dic2['n'] = n
			dic2['N'] = N
			dic2['M'] = M

		dfs = pd.DataFrame(dic).T
		self.dfs = dfs

		fname = self.fname_group_statistics%(self.disease, self.case, self.n_sentences, self.super_question, self.gemini_model)
		ret = pdwritecsv(dfs, fname, self.root_gemini, verbose=verbose)

		return ret, dfs

	# open_gemini_statistical_analysis
	def open_gemini_answers_counts(self, run:str, chosen_model:int, verbose:bool=False) -> pd.DataFrame:

		self.set_gemini_num_model(chosen_model)
		self.set_run_seldata(run)

		# instead pubmed -> search_with_pubmed
		fname = self.fname_anal_stat_model%(run, self.gemini_model, self.suffix)
		filename = os.path.join(self.root_gemini_root, fname)

		if os.path.exists(filename):
			df = pdreadcsv(fname, self.root_gemini_root, verbose=verbose)
		else:
			df = pd.DataFrame()
			
		return df


	# gemini_create_statistical_analysis
	def gemini_calc_answers_counts(self, run:str, case_list:List, chosen_model:int, 
								   force:bool=False, verbose:bool=False) -> pd.DataFrame:
		'''
			calc statistics comparing i_dfp with default (i_dfp == 0)

			disease and case: already known
			quest_ptw_dis_case has a %s - to input the pathway description
		'''
		self.set_gemini_num_model(chosen_model)
		self.set_run_seldata(run)

		fname = self.fname_anal_stat_model%(run, self.gemini_model, self.suffix)
		filename = os.path.join(self.root_gemini_root, fname)

		if os.path.exists(filename) and not force:
			df = pdreadcsv(fname, self.root_gemini_root, verbose=verbose)
			return df

		dic = {}; icount=-1

		for case in case_list:
			print(">>> case", case)

			ret, _, _, _ = self.bpx.open_case(case)
			self.set_case(self.bpx.case, self.bpx.df_enr, self.bpx.df_enr0)
			
			for iq, quest_type in enumerate(self.question_list):
				
				quest_ptw_dis_case, with_without_PubMed, sufix2, question_list = self.return_questions(quest_type)

				for i_dfp in range(len(question_list)):
					question_name = question_list[i_dfp]
					df = self.read_gemini(run=run, case=case, iq=iq, i_dfp=i_dfp, chosen_model=chosen_model, verbose=verbose)

					n = len(df)

					if df is None or df.empty:
						n_yes, n_possible, n_low, n_no = 0, 0, 0, 0
					else:
						n_yes = np.sum(df.curation.str.startswith('Yes'))
						n_possible = np.sum(df.curation.str.startswith('Possible'))
						n_low = np.sum(df.curation.str.startswith('Low'))
						n_no  = np.sum(df.curation.str.startswith('No'))

					icount += 1
					dic[icount] = {}
					dic2 = dic[icount]
					dic2['case'] = case
					dic2['iq'] = iq
					dic2['i_dfp'] = i_dfp
					dic2['chosen_model'] = chosen_model

					dic2['search_with_pubmed'] = 'with_' in with_without_PubMed
					dic2['strong_evidence'] = 'strong' in sufix2

					dic2['n'] = n
					dic2['yes'] = n_yes
					dic2['possible'] = n_possible
					dic2['low_evidence'] = n_low
					dic2['no'] = n_no

		df = pd.DataFrame(dic).T

		ret = pdwritecsv(df, fname, self.root_gemini_root, verbose=verbose)
		return df


	def gemini_create_statistical_summary(self, run:str, case_list:List, chosen_model:int, 
										  sum_both:bool=True, force:bool=False, verbose:bool=False) -> pd.DataFrame:
		'''
			create a pivot table with summary among pathways x question types
		'''
		self.set_gemini_num_model(chosen_model)
		self.set_run_seldata(run)

		fname = self.fname_summ_stat_model%(self.disease, self.gemini_model, self.suffix)
		filename = os.path.join(self.root_gemini_root, fname)

		if os.path.exists(filename) and not force:
			df = pdreadcsv(fname, self.root_gemini_root, verbose=verbose)
			return df

		df_stat = self.open_gemini_statistical_analysis(run=run, chosen_model=chosen_model, verbose=verbose)

		if df_stat is None or df_stat.empty:
			print(f"Nothing found for open_gemini_statistical_analysis(): {run} {chosen_model} self.is_seldata={self.is_seldata}")
			return pd.DataFrame()


		dicall = {}; icount=-1
		show_print = False

		for case in case_list: 
			for search_with_pubmed in [False, True]:
				for strong in [False, True]:
					df2 = df_stat[ (df_stat.case == case) & (df_stat.search_with_pubmed == search_with_pubmed) & (df_stat.strong_evidence == strong)]
					df3 = df2[self.cols_yn].copy()
					df3.index = np.arange(len(df3))
					change = list(df3.loc[0]) == [5,5,5,5]

					if show_print: print(f"case {case} \t search_with_pubmed={search_with_pubmed} \tstrong={strong}")
					
					lista = [self.lines_01, self.lines_02, self.lines_03, self.lines_04]
					for i_compare in range(len(lista)):
						lines = lista[i_compare]

						if change:
							if i_compare == 0:
								df3.loc[0] = [5, 5, 1, 1]
								df3.loc[0] = [10, 10, 2, 2]
							elif i_compare == 1:
								df3.loc[0] = [5, 5, 5, 5]
							elif i_compare == 2:
								df3.loc[0] = [1, 1, 5, 5]
								df3.loc[0] = [2, 2, 10, 10]
							else:
								df3.loc[0] = [1, 1, 5, 5]
								df3.loc[0] = [2, 2, 10, 10]

						vals0 = [x for x in df3.loc[0]]
						vals1 = [x for x in df3.loc[i_compare+1]]

						if sum_both:
							vals0 = [vals0[0]+vals0[1], vals0[2]+vals0[3] ]
							vals1 = [vals1[0]+vals1[1], vals1[2]+vals1[3] ]
						else:
							if vals1[1:3] == [0,0]:
								vals0 = [vals0[0], vals0[3] ]
								vals1 = [vals1[0], vals1[3] ]

						icount += 1
						dicall[icount] = {}
						dic2 = dicall[icount]

						dic2['case'] = case
						dic2['search_with_pubmed'] = search_with_pubmed
						dic2['strong'] = strong
						dic2['i_compare'] = i_compare+1
						
						try:
							s_stat, stat, pvalue, dof, expected = chi2_or_fisher_exact_test(vals0, vals1)

							dic2['stat'] = stat
							dic2['pvalue'] = pvalue
							dic2['s_stat'] = s_stat
							dic2['dof'] = dof
								
							if show_print: print(f"\tlines={str(lines)} - {vals0} x {vals1} \t{'sig' if pvalue < 0.05 else '   '} \t pvalue={pvalue:.2e}")
						except:
							dic2['stat'] = None
							dic2['pvalue'] = None
							dic2['s_stat'] = None
							dic2['dof'] = None

							if show_print: print(f"\tlines={str(lines)}")
					
						dic2['vals1'] = vals1
						dic2['vals0'] = vals0
					
					if show_print: print("")
						
			if show_print: print("")

		df = pd.DataFrame(dicall).T

		ret = pdwritecsv(df, fname, self.root_gemini_root, verbose=verbose)
		return df


	# open_gemini_yes_no_run_all_models -> open_gemini_yes_no_run_per_model
	def open_gemini_yes_no_run_per_model(self, run:str, consensus:str, 
										  verbose:bool=False) -> pd.DataFrame:

		fname = self.fname_consensus_one_run_per_model%(run, consensus, self.suffix)
		filename = os.path.join(self.root_gemini_root, fname)

		if os.path.exists(filename):
			df = pdreadcsv(fname, self.root_gemini_root, verbose=verbose)
		else:
			print(f"File not found: '{filename}'")
			df = pd.DataFrame()

		return df

	# save_gemini_yes_no_run_all_models ->
	def save_gemini_yes_no_run_per_model(self, run:str, chosen_model_list:List, 
										 force:bool=False, verbose:bool=False) -> dict:

		dic={}
		for consensus in ['Yes', 'No']:

			fname = self.fname_consensus_one_run_per_model%(run, consensus, self.suffix)

			filename = os.path.join(self.root_gemini_root, fname)
	
			if os.path.exists(filename) and not force:
				print(f"consensus {consensus}")
				df = pdreadcsv(fname, self.root_gemini_root, verbose=verbose)
				dic[consensus] = df

		if len(dic) == 2:
			return dic


		dic={}
		cols = ['run', 'case', 'chosen_model', 'model_name', 'i_dfp', 'pathway_id', 'pathway',
				'consensus', 'n_yes', 'n_no', 'unanimous']		

		df_list_yes, df_list_no = [], []
		for chosen_model in chosen_model_list:

			dfpiv = self.open_dfpiv_semantic_consensus_run_per_model(run=run, chosen_model=chosen_model, verbose=verbose)
			if dfpiv is None or dfpiv.empty:
				print("File not found")
				continue

			print(f"consensus='Yes' chosen_model {chosen_model}")
			dfpiv2 = dfpiv[ dfpiv.consensus.isin(['Yes', 'Doubt']) ]

			if dfpiv2 is None or dfpiv2.empty:
				print(f"\t\tNothing found for {run} chosen_model={chosen_model} consensus='Yes'")
			else:
				dfpiv2 = dfpiv2[cols]
				df_list_yes.append(dfpiv2)


			print(f"consensus='No' chosen_model {chosen_model}")
			dfpiv2 = dfpiv[ dfpiv.consensus == 'No' ]

			if dfpiv2 is None or dfpiv2.empty:
				print(f"\t\tNothing found for {run} chosen_model={chosen_model} consensus='No'")
			else:
				dfpiv2 = dfpiv2[cols]
				df_list_no.append(dfpiv2)


		if df_list_yes == []:
			print(f"\t\tNothing found for all {run} consensus={consensus}")
		else:
			consensus = 'Yes'
			fname = self.fname_consensus_one_run_per_model%(run, consensus, self.suffix)

			df = pd.concat(df_list_yes)
			df.index = np.arange(len(df))

			ret = pdwritecsv(df, fname, self.root_gemini_root)
			dic[consensus] = df


		if df_list_no == []:
			print(f"\t\tNothing found for all {run} consensus={consensus}")
		else:
			consensus = 'No'
			fname = self.fname_consensus_one_run_per_model%(run, consensus, self.suffix)

			df = pd.concat(df_list_no)
			df.index = np.arange(len(df))

			ret = pdwritecsv(df, fname, self.root_gemini_root)
			dic[consensus] = df

		return dic



	# open_gemini_consensus_counts_run_all_models -->
	def open_gemini_dfpiva_all_models_one_run(self, run:str, chosen_model_list:List, verbose:bool=False) -> pd.DataFrame:

		'''
			hard reproducibility
			return couts table
		'''
		self.set_run_seldata(run)

		s_chosen_model_list = str(chosen_model_list)
		fname = self.fname_dfpiva_all_models_one_run%(self.disease, s_chosen_model_list, run, self.suffix)
		filename = os.path.join(self.root_gemini, fname)

		if os.path.exists(filename):
			dfpiv = pdreadcsv(fname, self.root_gemini, verbose=verbose)
			return dfpiv

		print(f"Could not find: {filename} for run='{run}' and chosen_model_list={s_chosen_model_list}")
		return None


	# calc_gemini_consensus_counts_run_all_models
	def calc_gemini_dfpiva_all_models_one_run(self, run:str, case_list:List=None, chosen_model_list:List=None,
											 force:bool=False, verbose:bool=False) -> pd.DataFrame:
		'''
			hard reproducibility
			return couts table
		'''
		self.set_run_seldata(run)

		s_chosen_model_list = str(chosen_model_list)

		fname = self.fname_dfpiva_all_models_one_run%(self.disease, s_chosen_model_list, run, self.suffix)
		filename = os.path.join(self.root_gemini, fname)

		if os.path.exists(filename) and not force:
			dfpiv = pdreadcsv(fname, self.root_gemini, verbose=verbose)
			return dfpiv


		df_list_cases = []
		for chosen_model in chosen_model_list:
			self.set_gemini_num_model(chosen_model)

			for case in case_list:
				print("case", case, end=', ')

				fname_case = self.fname_dfpiv_gemini_run_case_model%(self.disease, case, self.gemini_model, run, self.suffix)
				filename = os.path.join(self.root_gemini, fname_case)

				if os.path.exists(filename) and not force:
					dfall = pdreadcsv(fname_case, self.root_gemini, verbose=verbose)
				else:
					ret, _, _, _ = self.bpx.open_case(case)
					self.set_case(self.bpx.case, self.bpx.df_enr, self.bpx.df_enr0)
					
					df_list = []
					# self.question_list = ['simple', 'simple+pubmed', 'disease', 'disease+pubmed']
					for iq in range(len(self.question_list)):
						quest_type = self.question_list[iq]
						print(quest_type, end=' ')
						
						quest_ptw_dis_case, with_without_PubMed, suffix2, question_list = self.return_questions(quest_type=quest_type)

						pathway_dup = []
						for i_dfp in range(len(question_list)):
							question_name = question_list[i_dfp]
							print(i_dfp, end=' ')
							df = self.read_gemini(run=run, case=case, iq=iq, i_dfp=i_dfp, chosen_model=chosen_model, verbose=verbose)
							if df is None or df.empty:
								print(f"could not find", end=' ')
								continue

							df = df[ (df.pathway_found==True) & (~pd.isnull(df.curation)) ]
							if df.empty:
								continue

							if pathway_dup != []:
								df = df[~df.pathway_id.isin(pathway_dup)]
								if df.empty:
									print("all repeated???", end=' ')
									continue

							pathway_dup += list(df.pathway_id)

							df = df[ self.cols_curation ]

							df['run'] = run
							df['iq'] = iq
							df['i_dfp'] = i_dfp
							df['chosen_model'] = self.chosen_model

							df['quest_type'] = quest_type
							df['question_name'] = question_name
							df['model_name'] = self.gemini_model
							df['curation'] = [x.split(',')[0].strip() for x in df['curation'] ]

							# self.cols_curation = ['case', 'pathway_id', 'pathway', 'curation', 'fdr']
							cols = self.cols_curation + ['run', 'iq', 'i_dfp','chosen_model', 'quest_type', 'question_name', 'model_name']
							df = df[cols]
							df_list.append(df)		
					
					print("")

					if df_list == []:
						print(f"Nothing found for {self.disease} case {case} - {self.gemini_model} {run}\n")
						continue
			
					dfall = pd.concat(df_list)
					dfall.index = np.arange(len(dfall))
					pdwritecsv(dfall, fname_case, self.root_gemini, verbose=verbose)

				'''--------- if os.path.exists(filename) and not force: ---------'''
				df_list_cases.append(dfall)

		if df_list_cases == []:
			print(f"No data found in for all cases in run {run}")
			return None


		dfall2 = pd.concat(df_list_cases)
		dfall2.index = np.arange(len(dfall2))
		self.dfall2 = dfall2

		''' merging all models '''
		dfpiv = None
		for chosen_model in chosen_model_list:
			dfall3 = dfall2[dfall2.chosen_model == chosen_model]

			if dfall3.empty:
				print("dfall3 cannot be empty.\n\n")
				raise Exception("Stop: dfall3")

			dfg2 = dfall3.groupby(['case', 'pathway_id', 'iq']).count().reset_index().iloc[:,:4]
			dfg2.columns = ['case', 'pathway_id', 'iq', 'n']
			dfg2 = dfg2[dfg2.n > 1]
			if not dfg2.empty:
				print("There are repeated values!!!")
				print(dfg2)

				dfall3 = dfall3.drop_duplicates( ['case', 'pathway_id'])

			self.dfall3 = dfall3

			dfg3 = dfall3.pivot(values='curation', columns='iq', index=self.index_pivot)
			this_question_type_list = [self.question_list[int(col)] for col in list(dfg3.columns)]

			#-------- confirm if the 4 semantic questions are columns
			lack_columns = [x for x in self.question_list if x not in this_question_type_list]
			if lack_columns != []:
				print("Error: lack_columns:", "; ".join(lack_columns))
				print(f"Please rerun digital_curation_run() for model {chosen_model} run {run}")
				raise Exception("Stop: run gemini_consensus_counts_all_models()")

			dfg3.columns = [x + f'_model_{chosen_model}' for x in this_question_type_list]
			dfg3 = dfg3.reset_index()

			if dfpiv is None:
				dfpiv = dfg3
			else:
				dfpiv = pd.merge(dfpiv, dfg3, how="outer", on=self.index_pivot)


		''' return a list of consensus, n_yes, n_no, unanimous '''
		tuple_list = [self.calc_gemini_consensus(dfpiv.iloc[i]) for i in range(len(dfpiv)) ]
		cons_list  = [x[0] for x in tuple_list]
		n_yes_list = [x[1] for x in tuple_list]
		n_no_list  = [x[2] for x in tuple_list]
		unan_list  = [x[3] for x in tuple_list]

		dfpiv['run'] = run
		dfpiv['consensus'] = cons_list
		dfpiv['n_yes']	   = n_yes_list
		dfpiv['n_no']	   = n_no_list
		dfpiv['unanimous'] = unan_list

		ret = pdwritecsv(dfpiv, fname, self.root_gemini, verbose=verbose)

		return dfpiv


	def open_gemini_consensus_counts_run_filter_idfp_consensus_run_all_models(self, run:str, i_dfp:int, 
																			  chosen_model_list:List, consensus:str='Yes', 
																			  verbose:bool=False) -> pd.DataFrame:
		'''
			return all consensus==Yes and i_dfp

			gemini model must be defined
		'''
		dfpiva = self.open_gemini_dfpiva_all_models_one_run(run=run, chosen_model_list=chosen_model_list, verbose=verbose)
		if dfpiva is None or dfpiva.empty:
			if verbose: print(f"Warning: nothing found for '{run}' and i_dfp={i_dfp}: consensus={consensus}")
			return None

		cols = ['case', 'pathway_id', 'pathway', 'i_dfp', 'consensus', 'n_yes', 'n_no', 'unanimous']  # 'fdr', ***###
		df2 = dfpiva[ (dfpiva.consensus == consensus) & (dfpiva.i_dfp == i_dfp)][cols].copy()

		if df2.empty:
			if verbose: print(f"Empty dfpiva for '{run}' and i_dfp{i_dfp} and consensus={consensus}.")
			return None

		if verbose: print(f">>> '{run}' and i_dfp={i_dfp}: consensus={consensus}")
		df2.index = np.arange(len(df2))
		
		return df2


	# calc_all_semantic_unanimous_repro - calc_soft_alls_runs_all_models_consensus
	def calc_soft_consensus_stats_case_i_dfp(self, run_list:List, case_list:List, 
									  		 chosen_model_list:List, i_dfp_list:List,
									  		 force:bool=False, verbose:bool=False) -> pd.DataFrame:


		if run_list is None or not isinstance(run_list, list):
			print("run_list must be a list")
			return None

		fname = self.fname_soft_OMC_per_case_idfp%(self.disease, chosen_model_list, self.suffix)
		filename = os.path.join(self.root_gemini_root, fname)

		if os.path.exists(filename) and not force:
			return pdreadcsv(fname, self.root_gemini_root, verbose=verbose)

		dic={}; icount=-1
		for run in run_list:
			print(">>>", run)
			for chosen_model in chosen_model_list:
				# calc_run_model_4DSSQ
				dfc = self.calc_soft_one_run_one_model_consensus(run=run, case_list=case_list, 
																 chosen_model=chosen_model, 
																 force=force, verbose=verbose)

				for case in case_list:
					for i_dfp in i_dfp_list:
						df2 = dfc[(dfc.case == case) & (dfc.i_dfp == i_dfp)]
						n = len(df2)

						cons_yes = [True if (x=='Yes' or x=='Doubt') else False for x in df2.consensus]
						mu_consensus  = np.mean(cons_yes)
						std_consensus = np.std(cons_yes)

						mu_unan  = df2.unanimous.mean()
						std_unan = df2.unanimous.std()

						n_yes   = len(df2[df2.consensus=='Yes'])
						n_doubt = len(df2[df2.consensus=='Doubt'])
						n_no	= len(df2[df2.consensus=='No'])

						n_unan	   = len(df2[df2.unanimous==True])
						n_not_unan = len(df2[df2.unanimous==False])

						if verbose: print(f"run {run}, chosen_model {chosen_model} or {self.chosen_model} - {self.gemini_model} unanimous={100*mu_unan:.1f}%")

						icount+=1
						dic[icount] = {}
						dic2 = dic[icount]

						dic2['run'] = run
						dic2['chosen_model'] = chosen_model
						dic2['model_name'] = self.gemini_model
						dic2['case'] = case
						dic2['i_dfp'] = i_dfp
						dic2['mu_consensus_yes'] = mu_consensus
						dic2['std_consensus_yes'] = std_consensus
						dic2['mu_unanimous'] = mu_unan
						dic2['std_unanimous'] = std_unan
						dic2['n'] = n
						dic2['n_yes'] = n_yes
						dic2['n_doubt'] = n_doubt
						dic2['n_no'] = n_no
						dic2['n_unan'] = n_unan
						dic2['n_not_unan'] = n_not_unan

		dfc_stat = pd.DataFrame(dic).T

		ret = pdwritecsv(dfc_stat, fname, self.root_gemini_root, verbose=verbose)

		return dfc_stat


	# open_gemini_semantic_reproducibility_run_model --> open_gemini_unanimous_consistency_one_run_model
	# open_gemini_unanimous_consistency_one_run_model -> open_run_model_4DSSQ
	def open_soft_run_run_one_model_consensus(self, chosen_model_list:List, verbose:bool=False) -> pd.DataFrame:

		fname = self.fname_soft_OMC_per_case_idfp%(self.disease, chosen_model_list, self.suffix)
		filename = os.path.join(self.root_gemini_root, fname)

		if os.path.exists(filename):
			dfc = pdreadcsv(fname, self.root_gemini_root, verbose=verbose)
			return dfc

		print(f"Could not find: {filename} for models={chosen_model_list}")
		return None

	# calc_soft_run_run_one_model_consensus_repro
	def calc_soft_RRCR_stats_per_idfp(self, run1:str, run2:str, chosen_model_list:List, 
									 case_list:List, i_dfp_list:List,
									 force:bool=False, verbose:bool=False) -> pd.DataFrame:

		fname = self.fname_soft_RRCR_stats_per_idfp%(self.disease, run1, run2, self.chosen_model_list, self.suffix)
		filename = os.path.join(self.root_gemini_root, fname)

		if os.path.exists(filename) and not force:
			dfstat = pdreadcsv(fname, self.root_gemini_root, verbose=verbose)
			return dfstat
		
		dfrep = self.open_soft_run_run_one_model_consensus(chosen_model_list=chosen_model_list, verbose=verbose)
		if dfrep is None or dfrep.empty:
			print("Could not find soft_run_run_one_model_consensus table")
			return None

		df1 = dfrep[dfrep.run == run1]
		if df1.empty:
			print("Could not find df1")
			return None

		df2 = dfrep[dfrep.run == run2]
		if df2.empty:
			print("Could not find df2")
			return None

		assert len(df1) == len(df2), f"Each df run must have the same length: {len(df1)} x {len(df2)}"

		dic={}; icount=-1

		for case in case_list:
			for i_dfp in i_dfp_list:

				df1c = df1[(df1.case == case) & (df1.i_dfp == i_dfp)]
				df2c = df2[(df2.case == case) & (df2.i_dfp == i_dfp)]

				if len(df1c) != len(df2c):
					print(f"Error: {case} i_dfp {i_dfp} have different lens for df1c x df2c")
					continue

				n = df1c.iloc[0].n

				list_yes1 = df1c.n_yes.to_list()
				list_una1 = df1c.n_unan.to_list()
				
				list_yes2 = df2c.n_yes.to_list()
				list_una2 = df2c.n_unan.to_list()

				icount += 1
				dic[icount] = {}
				dic2 = dic[icount]
			
				dic2['run1'] = run1
				dic2['run2'] = run2
				dic2['case'] = case
				dic2['i_dfp'] = i_dfp

				dic2['n'] = n
				
				s_stat, stat, pvalue, dof, expected = chi2_or_fisher_exact_test(list_yes1, list_yes2)
				dic2['repro_yes_s_stat'] = s_stat
				dic2['repro_yes_stat']   = stat
				dic2['repro_yes_pvalue'] = pvalue
				dic2['repro_yes_dof']	= dof
				dic2['repro_yes_expected'] = expected

				s_stat, stat, pvalue, dof, expected = chi2_or_fisher_exact_test(list_una1, list_una2)

				dic2['unan_s_stat'] = s_stat
				dic2['unan_stat']   = stat
				dic2['unan_pvalue'] = pvalue
				dic2['unan_dof']	= dof
				dic2['unan_expected'] = expected

				repro_yes_perc1 = [x/n for x in list_yes1]
				repro_yes_perc2 = [x/n for x in list_yes2]

				dic2['repro_yes_mu_perc1']  = np.mean(repro_yes_perc1)
				dic2['repro_yes_std_perc1'] = np.std(repro_yes_perc1)

				dic2['repro_yes_mu_perc2']  = np.mean(repro_yes_perc2)
				dic2['repro_yes_std_perc2'] = np.std(repro_yes_perc2)

				#-------- unanimous -----------------
				unan_perc1 = [x/n for x in list_una1]
				unan_perc2 = [x/n for x in list_una2]

				dic2['unan_mu_perc1']  = np.mean(unan_perc1)
				dic2['unan_std_perc1'] = np.std(unan_perc1)

				dic2['unan_mu_perc2']  = np.mean(unan_perc2)
				dic2['unan_std_perc2'] = np.std(unan_perc2)

				#---------- yes list ----------------
				dic2['repro_yes_list1'] = list_yes1
				dic2['repro_yes_list2'] = list_yes2

				dic2['repro_yes_perc1'] = repro_yes_perc1
				dic2['repro_yes_perc2'] = repro_yes_perc2

				#----------- unan list --------------
				dic2['unan_list1'] = list_una1
				dic2['unan_list2'] = list_una2

				dic2['unan_perc1'] = unan_perc1
				dic2['unan_perc2'] = unan_perc2


		if len(dic) == 0:
			print("Error: final dict is empty.")
			return None

		dfstat = pd.DataFrame(dic).T
		ret = pdwritecsv(dfstat, fname, self.root_gemini_root, verbose=verbose)

		return dfstat




	# calc_run_model_4DSSQ
	def calc_soft_one_run_one_model_consensus(self, run:str, case_list:List, chosen_model:int, 
							 				  force:bool=False, verbose:bool=False) -> pd.DataFrame:
		'''
			calc statistics comparing i_dfp with default (i_dfp == 0)

			disease and case: already known
			quest_ptw_dis_case has a %s - to input the pathway description
		'''

		if chosen_model is None or not isinstance(chosen_model, int):
			print(f"Please, define chosen_model as an int: 0, 1, ...")
			return None

		self.set_run_seldata(run)

		self.set_gemini_num_model(chosen_model, verbose=verbose)
		print("\t#", self.gemini_model, end=' ')

		fname = self.fname_soft_OMC%(self.disease, run, self.gemini_model, self.suffix)
		filename = os.path.join(self.root_gemini_root, fname)

		if os.path.exists(filename) and not force:
			dfrep = pdreadcsv(fname, self.root_gemini_root, verbose=verbose)
			return dfrep
		
		def fix_curation(text:str):
			text_low = text.strip().lower()

			if text_low.startswith('yes'):
				return 'Yes'
			if text_low.startswith('possible'):
				return 'Possible'
			if text_low.startswith('low'):
				return 'Low'
			if text_low.startswith('no'):
				return 'No'

			print(f"Error: '{text}'")
			return text.strip()

		cols3 = ['case', 'pathway_id', 'pathway', 'curation']

		df_list_cases = []

		for case in case_list:
			print("case", case, end=' ')
			
			dfn = None
			'''---- self.question_list = ['simple', 'simple+pubmed', 'disease', 'disease+pubmed']  ---'''
			for iq, quest_type in enumerate(self.question_list):
				
				quest_ptw_dis_case, with_without_PubMed, suffix2, question_list = self.return_questions(quest_type)

				df4_dfps = []
				''' four semantic questions '''
				for i_dfp in range(len(question_list)):
					question_name = question_list[i_dfp]
					if verbose: print(i_dfp, end=' ')

					df = self.read_gemini(run=run, case=case, iq=iq, i_dfp=i_dfp, chosen_model=chosen_model, verbose=verbose)
					if df is None or df.empty:
						print(f"could not find i_dfp {i_dfp } case {case} run {run}", end=' ')
						continue

					df = df[cols3]
					df.loc[:,'curation'] = [fix_curation(x) for x in df.curation]
					df['i_dfp'] = i_dfp


					df4_dfps.append(df)

				if df4_dfps == []:
					print(f"No answers for {quest_type}", end=' ')
					continue

				dfa = pd.concat(df4_dfps)
				dfa.index = np.arange(len(dfa))

				cols = ['case', 'pathway_id', 'pathway', 'i_dfp', 'curation']
				dfa = dfa[cols]

				col_curation = f"quest_{iq}"
				dfa.columns = ['case', 'pathway_id', 'pathway', 'i_dfp', col_curation]

				if dfn is None:
					dfn = dfa
				else:
					dfn = pd.merge(dfn, dfa, how="outer", on=['case', 'pathway_id', 'pathway', 'i_dfp'])

			consensus_list, unanimous_list = self.calc_gemini_consensus_df(dfn, ['quest_0', 'quest_1', 'quest_2', 'quest_3'])

			dfn['consensus'] = consensus_list
			dfn['unanimous'] = unanimous_list

			df_list_cases.append(dfn)

		print("")

		'''--------- for chosen_model in chosen_model_list: ----------'''
		if df_list_cases == []:
			print(f"No data found in for all cases in run {run} model {self.gemini_model}")
			return None

		dfrep = pd.concat(df_list_cases)
		dfrep.index = np.arange(len(dfrep))
		
		dfrep['run'] = run
		dfrep['chosen_model'] = self.chosen_model
		dfrep['model_name']   = self.gemini_model

		cols = ['run', 'chosen_model', 'model_name', 'case', 'pathway_id', 'pathway', 'i_dfp', 'consensus', 'unanimous'] + [f"quest_{iq}" for iq in range(4)]
		dfrep = dfrep[cols]

		ret = pdwritecsv(dfrep, fname, self.root_gemini_root, verbose=verbose)

		return dfrep


	def calc_gemini_consensus_df(self, dfn:pd.DataFrame, cols_quest=List) -> (List, List):

		consensus_list=[]
		unanimous_list=[]

		for i in range(len(dfn)):
			row = dfn.iloc[i]
			n_yes, n_no = 0, 0

			for col in cols_quest:
				if not isinstance(row[col], str):
					continue

				answer = row[col]
				if answer.startswith('Yes') or answer.startswith('Possible'):
					n_yes += 1
				else:
					n_no += 1

			if n_yes > n_no:
				consensus = 'Yes'
			elif n_yes < n_no:
				consensus = 'No'
			else:
				consensus = 'Doubt'

			unanimous = False if n_yes != 0 and n_no != 0 else True

			consensus_list.append(consensus)
			unanimous_list.append(unanimous)
			
		return consensus_list, unanimous_list


	def open_dfpiv_gemini_run_case_model(self, run:str, case:str, chosen_model:int,
										verbose:bool=False) -> pd.DataFrame:

		self.set_run_seldata(run)
		self.set_gemini_num_model(chosen_model)

		fname_case = self.fname_dfpiv_gemini_run_case_model%(self.disease, case, self.gemini_model, run, self.suffix)
		filename = os.path.join(self.root_gemini, fname_case)

		if os.path.exists(filename):
			dpiv = pdreadcsv(fname_case, self.root_gemini, verbose=verbose)
		else:
			print(f"File does not exist: '{filename}'")
			dpiv = pd.DataFrame()

		return dpiv


	def open_dfpiv_semantic_consensus_run_per_model(self, run:str, chosen_model:int, verbose:bool=False) -> pd.DataFrame:
		'''
		return: consensus and 4 questins, n_yes, n_no, unanimous
		'''

		self.set_run_seldata(run)
		self.set_gemini_num_model(chosen_model)
		if verbose: print(">>> model", self.gemini_model, end=' ')

		fname = self.fname_dfpiv_consensus_run_model%(self.disease, self.gemini_model, run, self.suffix)
		filename = os.path.join(self.root_gemini, fname)

		if os.path.exists(filename):
			dfpiv = pdreadcsv(fname, self.root_gemini, verbose=verbose)
			dfpiv.index = np.arange(len(dfpiv))
		else:
			print(f"Could not find: '{filename}'")
			dfpiv = pd.DataFrame()
			

		return dfpiv


	def calc_dfpiv_semantic_consensus_run_per_model(self, run:str, case_list:List, chosen_model:int, 
												   force:bool=False, verbose:bool=False) -> pd.DataFrame:

		'''
		calc and return: consensus and 4 questins, n_yes, n_no, unanimous
		'''

		self.set_run_seldata(run)
		self.set_gemini_num_model(chosen_model)
		if verbose: print(">>> model", self.gemini_model, end=' ')

		fname = self.fname_dfpiv_consensus_run_model%(self.disease, self.gemini_model, run, self.suffix)
		filename = os.path.join(self.root_gemini, fname)

		if os.path.exists(filename) and not force:
			dfpiv = pdreadcsv(fname, self.root_gemini, verbose=verbose)
			return dfpiv

		df_list_cases = []

		for case in case_list:
			print("case", case, end=', ')

			fname_case = self.fname_dfpiv_gemini_run_case_model%(self.disease, case, self.gemini_model, run, self.suffix)
			filename = os.path.join(self.root_gemini, fname_case)

			if os.path.exists(filename) and not force:
				dfall = pdreadcsv(fname_case, self.root_gemini, verbose=verbose)
			else:
				ret, _, _, _ = self.bpx.open_case(case)
				self.set_case(self.bpx.case, self.bpx.df_enr, self.bpx.df_enr0)
				
				df_list = []
				# self.question_list = ['simple', 'simple+pubmed', 'disease', 'disease+pubmed']
				for iq in range(len(self.question_list)):
					quest_type = self.question_list[iq]
					print(quest_type, end=' ')
					
					quest_ptw_dis_case, with_without_PubMed, suffix2, question_list = self.return_questions(quest_type)

					pathway_dup = []
					for i_dfp in range(len(question_list)):
						question_name = question_list[i_dfp]
						print(i_dfp, end=' ')
						df = self.read_gemini(run=run, case=case, iq=iq, i_dfp=i_dfp, chosen_model=chosen_model, verbose=verbose)
						if df is None or df.empty:
							print(f"could not find", end=' ')
							continue

						df = df[ (df.pathway_found==True) & (~pd.isnull(df.curation)) ]
						if df.empty:
							continue

						if pathway_dup != []:
							df = df[~df.pathway_id.isin(pathway_dup)]
							if df.empty:
								print("all repeated???", end=' ')
								continue

						pathway_dup += list(df.pathway_id)

						df = df[ self.cols_curation ]

						df['run'] = run
						df['case'] = case
						df['iq'] = iq
						df['quest_type'] = quest_type

						df['i_dfp'] = i_dfp
						df['question_name'] = question_name

						df['chosen_model'] = self.chosen_model
						df['model_name'] = self.gemini_model

						# self.cols_curation = ['case', 'pathway_id', 'pathway', 'curation', 'fdr']
						cols = self.cols_curation + ['iq', 'quest_type', 'i_dfp', 'question_name', 'chosen_model', 'model_name', ]
						df = df[cols]
						df_list.append(df)		
				
				print("")

				if df_list == []:
					print(f"Nothing found for {self.disease} case {case} - {self.gemini_model} {run}\n")
					continue
		
				dfall = pd.concat(df_list)
				dfall.index = np.arange(len(dfall))
				pdwritecsv(dfall, fname_case, self.root_gemini, verbose=verbose)

			'''--------- if os.path.exists(filename) and not force: ---------'''
			df_list_cases.append(dfall)

		'''--------- for chosen_model in chosen_model_list: ----------'''
		if df_list_cases == []:
			print(f"No data found in for all cases in run {run}")
			return None

		dfall2 = pd.concat(df_list_cases)
		dfall2.index = np.arange(len(dfall2))

		#--- here, only 1 model ------------------
		dfg2 = dfall2.groupby(['case', 'pathway_id', 'iq']).count().reset_index().iloc[:,:4]
		dfg2.columns = ['case', 'pathway_id', 'iq', 'n']
		dfg2 = dfg2[dfg2.n > 1]
		if not dfg2.empty:
			print("There are repeated values!!!")
			print(dfg2)

			dfall2 = dfall2.drop_duplicates( ['case', 'pathway_id', 'iq'])

		dfpiv = dfall2.pivot(values='curation', columns='iq', index=self.index_pivot)

		try:
			dfpiv.columns = self.question_list
		except:
			print("Error: lack_columns:", "; ".join( list(dfpiv.columns) ))
			print(f"Please rerun digital_curation_run() for model {self.gemini_model} run {run}")
			raise Exception("Stop: run gemini_consensus_counts_all_models()")

		dfpiv = dfpiv.reset_index()

		''' return a list of consensus, n_yes, n_no, unanimous '''
		tuple_list = [self.calc_gemini_consensus(dfpiv.iloc[i]) for i in range(len(dfpiv)) ]
		cons_list  = [x[0] for x in tuple_list]
		n_yes_list = [x[1] for x in tuple_list]
		n_no_list  = [x[2] for x in tuple_list]
		unan_list  = [x[3] for x in tuple_list]

		dfpiv['run'] = run
		dfpiv['chosen_model'] = chosen_model
		dfpiv['model_name']   = self.gemini_model
		dfpiv['consensus'] = cons_list
		dfpiv['n_yes']	   = n_yes_list
		dfpiv['n_no']	   = n_no_list
		dfpiv['unanimous'] = unan_list

		cols =  ['run', 'case', 'chosen_model', 'model_name', 'i_dfp', 'pathway_id', 'pathway'] + \
				self.question_list  + ['consensus', 'n_yes', 'n_no', 'unanimous']

		dfpiv = dfpiv[cols]

		ret = pdwritecsv(dfpiv, fname, self.root_gemini, verbose=verbose)

		return dfpiv

	def summary_stat_dfpiv_all_models(self, run:str, case_list:List, chosen_model_list:List, 
									 force:bool=False, verbose:bool=False) -> pd.DataFrame:

		self.set_run_seldata(run)
		s_chosen_model_list = str(chosen_model_list)

		fname = self.fname_summary_consensus_one_run_all_models%(run, s_chosen_model_list, self.disease, self.suffix)
		filename = os.path.join(self.root_gemini, fname)

		if os.path.exists(filename) and not force:
			dfsumm = pdreadcsv(fname, self.root_gemini, verbose=verbose)
			return dfsumm

		yes_no_list = ['Yes', 'Possible', 'Low', 'No']

		def prepare_summary_pivot(dfpiv:pd.DataFrame, case:str, i_dfp:int):

			df = dfpiv[ (dfpiv.case  == case) &  (dfpiv.i_dfp == i_dfp)]
			
			df_list = []
			for col in self.question_list:
				df[col] = [x.replace(',','').strip() if isinstance(x,str) else x for x in df[col] ]
			
				dfg = df.groupby(col).count().reset_index().iloc[:,:2]
				dfg.columns = [col, 'n']
			
				for answer in yes_no_list:
					dfa = dfg[dfg[col].str.startswith(answer)]
					if dfa.empty:
						dfg.loc[-1] = [answer, 0]
						dfg.index = dfg.index + 1
			
				dfg = dfg.set_index(col)
				dfg.index.name = 'index'
				dfg = dfg.loc[ yes_no_list, :].T.reset_index()
				dfg.columns = ['question_type'] + yes_no_list
				dfg.iloc[0,0] = col
				df_list.append(dfg)
			
			dfa = pd.concat(df_list)
			dfa.index = np.arange(len(dfa))

			dfa['case'] = case
			dfa['i_dfp'] = i_dfp

			return dfa


		df_list = []

		for chosen_model in chosen_model_list:
			dfpiv = self.calc_dfpiv_semantic_consensus_run_per_model(run=run, case_list=case_list, chosen_model=chosen_model, 
																	force=False, verbose=verbose)
			for case in case_list:
				for i_dfp in self.i_dfp_list:
					dfa = prepare_summary_pivot(dfpiv, case, i_dfp)
					if dfa is not None and not dfa.empty:
						dfa['run'] = run
						dfa['chosen_model'] = chosen_model

						dfa['n_yes'] = dfa['Yes'] + dfa['Possible']
						dfa['n_no']	 = dfa['No']  + dfa['Low']

						dfa['is_yes'] = dfa['n_yes'] >= dfa['n_no']

						dfa['n'] = dfa['n_yes'] + dfa['n_no']
						dfa['perc_yes'] = dfa['n_yes'] / dfa['n']
						dfa['perc_no']  = dfa['n_no']  / dfa['n']

						df_list.append(dfa)

		if len(df_list) == 0:
			return None

		dfsumm = pd.concat(df_list)
		dfsumm.index = np.arange(len(dfsumm))

		cols = ['run', 'case', 'chosen_model', 'i_dfp', 'question_type'] + yes_no_list + \
			   ['is_yes', 'n_yes', 'n_no', 'n', 'perc_yes', 'perc_no']
		dfsumm = dfsumm[cols]

		# remove zero cols
		drop_list = []
		for col in yes_no_list:
			if dfsumm[col].sum() == 0:
				drop_list.append(col)

		if len(drop_list) > 0:
			dfsumm = dfsumm.drop(columns=drop_list)
	
		_ = pdwritecsv(dfsumm, fname, self.root_gemini, verbose=verbose)

		return dfsumm


	def stat_between_dfp_by_run(self, run:str, case_list:List, chosen_model_list:List,
								verbose:bool=False) -> (pd.DataFrame, pd.DataFrame):

		dfsumm = self.summary_stat_dfpiv_all_models(run=run, case_list=case_list, chosen_model_list=chosen_model_list,
												   verbose=verbose)

		if dfsumm is None or dfsumm.empty:
			print("Nothing found using summary_stat_dfpiv_all_models().")
			return None, None

		dic={}; icount=-1
		for case in case_list:
			for question_type in self.question_list:
				for i_dfp in self.i_dfp_list:
					# many models !
					df2 = dfsumm[ (dfsumm.case == case) & 
								  (dfsumm.question_type == question_type) &
								  (dfsumm.i_dfp == i_dfp) ]

					icount += 1
					dic[icount] = {}
					dic2 = dic[icount]

					dic2['case'] = case
					dic2['question_type'] = question_type
					dic2['i_dfp'] = i_dfp

					cols = list(df2.columns)

					dic2['Yes'] =	  df2['Yes'].sum()	  if 'Yes'	  in cols else 0
					dic2['Possible'] = df2['Possible'].sum() if 'Possible' in cols else 0
					dic2['Low'] =	  df2['Low'].sum() if 'Low' in cols else 0
					dic2['No']  =	  df2['No'].sum()  if 'No'  in cols else 0

					dic2['yes_pos'] = dic2['Yes'] + dic2['Possible']
					dic2['no_low']  = dic2['No'] + dic2['Low']

		dfstat = pd.DataFrame(dic).T


		dic={}; icount=-1
		for case in case_list:
			for question_type in self.question_list:
				for i_dfp in self.i_dfp_list[1:]:

					df2 = dfstat[ (dfstat.case == case) & (dfstat.question_type == question_type) ]
					'''--- all i_dfp against 0 (enriched pathways) ---'''
					df2 = df2[ (df2.i_dfp == 0) | (df2.i_dfp == i_dfp) ]

					s_stat, stat, pvalue, dof, expected = chi2_or_fisher_exact_test(df2['yes_pos'], df2['no_low'])
					
					icount += 1
					dic[icount] = {}
					dic2 = dic[icount]

					dic2['case'] = case
					dic2['question_type'] = question_type
					dic2['i_dfp'] = i_dfp

					dic2['s_stat'] = s_stat
					dic2['stat']   = stat
					dic2['pvalue'] = pvalue
					dic2['dof']	= dof
					dic2['yes_pos_list'] = list(df2['yes_pos'])
					dic2['no_low_list'] = list(df2['no_low'])
					dic2['expected'] = expected
					dic2['df']  = df2

		dfchi2 = pd.DataFrame(dic).T

		return dfstat, dfchi2




	def merge_all_pivots(self, dic_case:dict) -> pd.DataFrame:
		df_list = []
		for key, df in dic_case.items():
			df = df.reset_index()
			if "index" in df.columns:
				df = df.drop(columns=['index'])
			df_list.append(df)

		dfall = pd.concat(df_list)
		dfall.index = np.arange(len(dfall))
		return dfall

	def open_gemini_summary_consensus_statitics(self, chosen_model_list:List, verbose:bool=False) -> (str, pd.DataFrame):

		s_chosen_model_list = str(chosen_model_list)

		fname = self.fname_summary_ynd_all_runs_all_models%(self.disease, s_chosen_model_list, self.suffix)
		filename = os.path.join(self.root_gemini_root, fname)

		fname_txt = self.fname_summary_ynd_all_runs_all_models_txt%(self.disease, s_chosen_model_list, self.suffix)
		filename_txt = os.path.join(self.root_gemini_root, fname_txt)

		if os.path.exists(filename) and os.path.exists(filename_txt):
			df   = pdreadcsv(fname,	self.root_gemini_root, verbose=verbose)
			text = read_txt(fname_txt, self.root_gemini_root, verbose=verbose)
			return text, df

		if not os.path.exists(filename):
			print(f"Could not find '{filename}'")

		if not os.path.exists(filename_txt):
			print(f"Could not find '{filename_txt}'")

		return '', None

	def open_gemini_summary_consensus_statitics_idfp(self, chosen_model_list:List, verbose:bool=False) -> (str, pd.DataFrame):

		s_chosen_model_list = str(chosen_model_list)

		fname = self.fname_summary_ynd_all_runs_all_models_idfp%(self.disease, s_chosen_model_list, self.suffix)
		filename = os.path.join(self.root_gemini_root, fname)

		fname_txt = self.fname_summary_ynd_all_runs_all_models_idfp_txt%(self.disease, s_chosen_model_list, self.suffix)
		filename_txt = os.path.join(self.root_gemini_root, fname_txt)

		if os.path.exists(filename) and os.path.exists(filename_txt):
			df   = pdreadcsv(fname,	self.root_gemini_root, verbose=verbose)
			text = read_txt(fname_txt, self.root_gemini_root, verbose=verbose)
			return text, df

		if not os.path.exists(filename):
			print(f"Could not find '{filename}'")

		if not os.path.exists(filename_txt):
			print(f"Could not find '{filename_txt}'")

		return '', None

	def get_gemini_summary_consensus_statitics_idfp(self, run:str, i_dfp:int, chosen_model_list:List, 
													verbose:bool=False) -> pd.DataFrame:

		_, df = self.open_gemini_summary_consensus_statitics_idfp(chosen_model_list, verbose=verbose)
		if df is None:
			return '', None

		df = df[ (df.run == run) & (df.i_dfp == i_dfp) ].copy()
		df.index = np.arange(0, len(df))

		return df


	def calc_gemini_summary_consensus_statitics(self, run_list:List, chosen_model_list:List,  case_list:List, 
												force:bool=False, verbose:bool=False) -> (str, pd.DataFrame):
		'''
			calc_gemini_summary_consensus_statitics:
				open_gemini_dfpiva_all_models_one_run

			stores concesnsus and unanimous (UR)
		'''
		s_chosen_model_list = str(chosen_model_list)

		fname = self.fname_summary_ynd_all_runs_all_models%(self.disease, s_chosen_model_list, self.suffix)
		filename = os.path.join(self.root_gemini_root, fname)

		fname_txt = self.fname_summary_ynd_all_runs_all_models_txt%(self.disease, s_chosen_model_list, self.suffix)
		filename_txt = os.path.join(self.root_gemini_root, fname_txt)

		if os.path.exists(filename) and os.path.exists(filename_txt) and not force:
			df   = pdreadcsv(fname,	self.root_gemini_root, verbose=verbose)
			text = read_txt(fname_txt, self.root_gemini_root, verbose=verbose)
			return text, df

		dic = {}; icount=-1; text=''
		for run in run_list:
			text += f">>> {run}\n"
			print(">>>", run, end=' ')

			dfpiva = self.open_gemini_dfpiva_all_models_one_run(run=run, chosen_model_list=chosen_model_list, verbose=verbose)
			if dfpiva is None or len(dfpiva) == 0:
				print(f"No curation statistics was calculated for case {case}", end=' ')
				continue
				
			for case in case_list + ['all']:
				print(case, end=' ')
			
				if case == 'all':
					df2 = dfpiva.copy()
				else:
					df2 = dfpiva[ (dfpiva.case == case) ].copy()
			
				df2['consensus_yes'] = (df2.consensus =='Yes') | (df2.consensus == 'Doubt')
				df2['consensus_no']  = (df2.consensus =='No')
				
				mu_yes  = df2['consensus_yes'].mean()
				std_yes = df2['consensus_yes'].std()
				
				mu_no  = df2['consensus_no'].mean()
				std_no = df2['consensus_no'].std()
			
				text += f"{case} yes {100*mu_yes:.1f}% ({100*std_yes:.1f}%) no {100*mu_no:.1f}% ({100*std_no:.1f}%)\n"
			
				icount += 1
				dic[icount] = {}
				dic2 = dic[icount]
			
				dic2['run']   = run
				dic2['case']  = case
			
				dic2['n'] = len(df2)
				dic2['n_yes'] = df2['consensus_yes'].sum()
				dic2['n_no']  = df2['consensus_no'].sum()
			
				dic2['consensus_yes'] = mu_yes
				dic2['consensus_yes_std'] = std_yes
			
				dic2['consensus_no'] = mu_no
				dic2['consensus_no_std'] = std_no
				
				dic2['unanimous'] = df2.unanimous.mean()
				dic2['unanimous_std'] = df2.unanimous.std()

			text += '\n\n'
			print("")

		df = pd.DataFrame(dic).T
		
		ret1 = pdwritecsv(df, fname, self.root_gemini_root, verbose=verbose)
		ret2 = write_txt(text, fname_txt, self.root_gemini_root, verbose=verbose)

		return text, df

	def calc_gemini_summary_consensus_statitics_idfp(self, run_list:List, chosen_model_list:List, 
													 case_list:List, save_files:bool=False,
													 force:bool=False, verbose:bool=False) -> (str, pd.DataFrame):
		'''
			calc_gemini_summary_consensus_statitics_idfp:
				open_gemini_dfpiva_all_models_one_run

			stores concesnsus and unanimous (UR)
		'''
		s_chosen_model_list = str(chosen_model_list)

		fname = self.fname_summary_ynd_all_runs_all_models_idfp%(self.disease, s_chosen_model_list, self.suffix)
		filename = os.path.join(self.root_gemini_root, fname)

		fname_txt = self.fname_summary_ynd_all_runs_all_models_idfp_txt%(self.disease, s_chosen_model_list, self.suffix)
		filename_txt = os.path.join(self.root_gemini_root, fname_txt)

		if os.path.exists(filename) and os.path.exists(filename_txt) and not force:
			df   = pdreadcsv(fname,	self.root_gemini_root, verbose=verbose)
			text = read_txt(fname_txt, self.root_gemini_root, verbose=verbose)
			return text, df

		dic = {}; icount=-1; text=''

		for run in run_list:
			text += f">>> {run}\n"
			print(">>>", run, end=' ')

			dfpiva = self.open_gemini_dfpiva_all_models_one_run(run=run, chosen_model_list=chosen_model_list, verbose=verbose)
			if dfpiva is None or len(dfpiva) == 0:
				print(f"No curation statistics was calculated for case {case}", end=' ')
				continue
				
			for case in case_list:
				print(case, end=' ')

				for i_dfp in self.i_dfp_list:
					print(i_dfp, end=' ')

					df2, df_yes, df_no, df_doubt = self.gemini_consensus_yes_no_doubt(dfpiva=dfpiva, chosen_model_list=chosen_model_list,
																					  case=case, i_dfp=i_dfp, 
																					  save_files=save_files, verbose=False)
					nYes, nNo, nDoubt = 0, 0, 0

					dfs = [df_yes, df_no, df_doubt]
				
					for i in range(len(dfs)):
						   
						df = dfs[i]

						df_una = df[ df.unanimous==True]
						df_nun = df[ df.unanimous==False]

						n = len(df)
						
						if i == 0:
							consensus = 'Yes'
							nYes = n
						elif i == 1:
							consensus = 'No'
							nNo = n
						elif i == 2:
							consensus = 'Doubt'
							nDoubt = n
						else:
							print("error")
							continue
				
						icount += 1
						dic[icount] = {}
						dic2 = dic[icount]
				
						dic2['run']   = run
						dic2['case']  = case
						dic2['i_dfp'] = i_dfp
						dic2['consensus'] = consensus
						dic2['n'] = n
						dic2['unanimous'] = len(df_una)
						dic2['not_unanimous'] = len(df_nun)

					ntot = nYes + nNo + nDoubt
					perc_yes = (nYes + nDoubt) / ntot
					
					text += f"\n\tfor case {case:15} and i_dfp={i_dfp:2}, there are {ntot:4} pathways, #Yes={nYes:4}, #No={nNo:4}, #Doubt={nDoubt:4} perc.yes-doubt={100.*perc_yes:.1f}%"
			text += '\n\n'
			print("")

		df = pd.DataFrame(dic).T
		
		ret1 = pdwritecsv(df, fname, self.root_gemini_root, verbose=verbose)
		ret2 = write_txt(text, fname_txt, self.root_gemini_root, verbose=verbose)

		return text, df

	# calc_gemini_consensus_confusion_table
	def calc_gemini_4groups_confusion_table(self, run_list:List, case_list:List, chosen_model_list:List,
											force:bool=False, verbose:bool=False) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):

		if not isinstance(run_list, list):
			print("run_list must be a list, like ['run01', 'run02']")
			return None, None, None

		if len(run_list) > 2:
			run_list = run_list[:2]

		fname1	= self.fname_confusion_table_consensus%(self.disease, 'G0xG1', run_list, chosen_model_list, self.suffix)
		filename1 = os.path.join(self.root_curation, fname1)

		fname2	= self.fname_confusion_table_consensus%(self.disease, 'G0xG2', run_list, chosen_model_list, self.suffix)
		filename2 = os.path.join(self.root_curation, fname2)

		fname3	= self.fname_confusion_table_consensus%(self.disease, 'G0xG3', run_list, chosen_model_list, self.suffix)
		filename3 = os.path.join(self.root_curation, fname3)

		if os.path.exists(filename1) and os.path.exists(filename2) and os.path.exists(filename3) and not force:
			dfconf1 = pdreadcsv(fname1, self.root_curation, verbose=verbose)
			dfconf2 = pdreadcsv(fname2, self.root_curation, verbose=verbose)
			dfconf3 = pdreadcsv(fname3, self.root_curation, verbose=verbose)
			return dfconf1, dfconf2, dfconf3

		_, dfcons = self.open_gemini_summary_consensus_statitics(chosen_model_list=chosen_model_list, verbose=verbose)

		if dfcons is None or dfcons.empty:
			print("Could not read open_gemini_summary_consensus_statitics()")
			return None, None, None

		def calc_params(dic, TP, FP, TN, FN):
			Npos = TP+FP
			Nneg = TN+FN
		
			N = Npos + Nneg
		
			sens = TP / (TP + FN)
			spec = TN / (TN + FP)
			accu = (TP + TN) / N
			prec = TP / (TP + FP)
			f1sc = (2*prec*sens) / (prec+sens)
		
			dic['n'] = N
			dic['npos'] = Npos
			dic['nneg'] = Nneg

			dic['TP'] = TP
			dic['FP'] = FP
			dic['TN'] = TN
			dic['FN'] = FN

			dic['sens'] = sens
			dic['spec'] = spec
			dic['accu'] = accu
			dic['prec'] = prec
			dic['f1_score'] = f1sc


		dic1, dic2, dic3 = {},{},{}
		icount=-1
		for run in run_list:
			for case in case_list:
			
				df2 = dfcons[ (dfcons.run==run) & (dfcons.case==case) ]
				TP = df2[ (df2.i_dfp==0) & ( (df2.consensus=='Yes') | (df2.consensus=='Doubt') ) ].n.sum()
				FP = df2[ (df2.i_dfp==0) & (df2.consensus=='No') ].n.sum()
			
				TN1 = df2[ (df2.i_dfp==1) & (df2.consensus=='No') ].n.sum()
				FN1 = df2[ (df2.i_dfp==1) & ( (df2.consensus=='Yes') | (df2.consensus=='Doubt') ) ].n.sum()
			
				TN2 = df2[ (df2.i_dfp==2) & (df2.consensus=='No') ].n.sum()
				FN2 = df2[ (df2.i_dfp==2) & ( (df2.consensus=='Yes') | (df2.consensus=='Doubt') ) ].n.sum()
			
				TN3 = df2[ (df2.i_dfp==3) & (df2.consensus=='No') ].n.sum()
				FN3 = df2[ (df2.i_dfp==3) & ( (df2.consensus=='Yes') | (df2.consensus=='Doubt') ) ].n.sum()
			
				#---------- G0 x G1 ----------------
				icount += 1
				dic1[icount] = {}
				dic1a = dic1[icount]

				dic1a['run'] = run
				dic1a['which'] = 'G0xG1'
				dic1a['case'] = case
				calc_params(dic1a, TP, FP, TN1, FN1)


				#---------- G0 x G2 ----------------
				dic2[icount] = {}
				dic2a = dic2[icount]

				dic2a['run'] = run
				dic2a['which'] = 'G0xG2'
				dic2a['case'] = case
				calc_params(dic2a, TP, FP, TN2, FN2)


				#---------- G0 x G3 ----------------
				dic3[icount] = {}
				dic3a = dic3[icount]

				dic3a['run'] = run
				dic3a['which'] = 'G0xG3'
				dic3a['case'] = case
				calc_params(dic3a, TP, FP, TN3, FN3)
			
				'''
				print(f"'{run}' analyze case '{case:3}' {N} pathways, {Npos} enriched, {Nneg} out of table.")
				print(f"\t\tTP {TP},  FP {FP},  TN {TN},  FN {FN}")
				print(f"\t\tSensibiliy: {100*sens:.1f}% Specificity: {100*spec:.1f}% Acuracy: {100*accu:.1f}% Precision: {100*prec:.1f}% F1-score: {100*f1sc:.1f}%")
				print("")
				'''

		dfconf1 = pd.DataFrame(dic1).T
		dfconf2 = pd.DataFrame(dic2).T
		dfconf3 = pd.DataFrame(dic3).T
		
		ret1 = pdwritecsv(dfconf1, fname1, self.root_curation, verbose=verbose)
		ret2 = pdwritecsv(dfconf2, fname2, self.root_curation, verbose=verbose)
		ret3 = pdwritecsv(dfconf3, fname3, self.root_curation, verbose=verbose)

		return dfconf1, dfconf2, dfconf3


	# calc_stats_gemini_4groups_confusion_table
	def calc_stats_gemini_4groups_confusion_compare_runs(self, run_list:List, case_list:List, chosen_model_list:List,
												  force:bool=False, verbose:bool=False) -> pd.DataFrame:


		if not isinstance(run_list, list) or len(run_list) < 2:
			print("run_list must be a list, like ['run01', 'run02']")
			return None, None, None

		if len(run_list) > 2:
			run_list = run_list[:2]

		fname = self.fname_confusion_table_stats_cons%(self.disease, run_list, chosen_model_list, self.suffix)
		filename = os.path.join(self.root_curation, fname)

		if os.path.exists(filename) and not force:
			dfstat = pdreadcsv(fname, self.root_curation, verbose=verbose)
			return dfstat

		dfconf1, dfconf2, dfconf3 = \
			self.calc_gemini_4groups_confusion_table(run_list=run_list, case_list=case_list,
													 chosen_model_list=chosen_model_list,
													 force=False, verbose=verbose)
		
		# cols = ['case', 'which', 'n', 'npos', 'nneg', 'TP', 'FP', 'TN', 'FN', 'sens', 'spec', 'accu', 'prec', 'f1_score']

		run0 = run_list[0]
		run1 = run_list[1]
		dic={}; icount=-1

		for i in range(3):
			if i==0:
				dfconf = dfconf1
				which = 'G0xG1'
			elif i==1:
				dfconf = dfconf2
				which = 'G0xG2'
			else:
				dfconf = dfconf3
				which = 'G0xG3'

			if dfconf is None or dfconf.empty:
				print(f"Warning: could not calc dfconf{i+1} = calc_gemini_4groups_confusion_table()")
				continue

			dfc0 = dfconf[dfconf.run == run0]
			dfc1 = dfconf[dfconf.run == run1]

			for col, test in self.mat_stat_tests:

				mu_param0  = dfc0[col].mean()
				std_param0 = dfc0[col].std()

				mu_param1  = dfc1[col].mean()
				std_param1 = dfc1[col].std()

				text_stat, stat, pval = calc_ttest(dfc0[col], dfc1[col])

				text_ext = f"{test}: run01: {100*mu_param0:.1f} ({100*std_param0:.1f}) x run02: {100*mu_param1:.1f} ({100*std_param1:.1f}) -> {text_stat}"

				icount += 1
				dic[icount] = {}
				dic2 = dic[icount]
			
				dic2['run0'] = run0
				dic2['run1'] = run1
				dic2['which'] = which
				dic2['test'] = col
				dic2['test_ext']  = test
				dic2['mu_param0']  = mu_param0
				dic2['std_param0'] = std_param0
				dic2['mu_param1']  = mu_param1
				dic2['std_param1'] = std_param1
				dic2['stat_test'] = 'ttest'
				dic2['pvalue'] = pval
				dic2['stat'] = stat
				dic2['text_stat'] = text_stat
				dic2['text_ext']  = text_ext

		dfstat = pd.DataFrame(dic).T
		ret = pdwritecsv(dfstat, fname, self.root_curation, verbose=verbose)

		return dfstat



	def calc_gemini_4groups_confusion_stats(self, run:str, run_list:List, 
								 			case_list:List, chosen_model_list:List,
								 			alpha:float=0.05, force:bool=False, 
								 			verbose:bool=False) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
		
		df1, df2, df3 = self.calc_gemini_4groups_confusion_table(run_list=run_list, case_list=case_list,
																 chosen_model_list=chosen_model_list,
																 force=False, verbose=verbose)

		fname = self.fname_conf_table_summary_stats%(self.disease, run, chosen_model_list, self.suffix)
		filename = os.path.join(self.root_curation, fname)

		if os.path.exists(filename) and not force:
			df = pdreadcsv(fname, self.root_curation, verbose=verbose)
			return df, df1, df2, df3

		# cols = ['case', 'which', 'n', 'npos', 'nneg', 'TP', 'FP', 'TN', 'FN', 'sens', 'spec', 'accu', 'prec', 'f1_score']
		df1 = df1[df1.run == run]
		df2 = df2[df2.run == run]
		df3 = df3[df3.run == run]

		dic={}; icount=-1
		for col, test in self.mat_stat_tests:

			vals_G0G3 = df3[col]
			
			for i in range(3):
				if i==0:
					dfa = df1
					compare = 'G1xG0'
				elif i==1:
					dfa = df2
					compare = 'G2xG0'
				else:
					dfa = df3
					compare = 'G3xG0'

				na = len(dfa)
				mua  = dfa[col].mean()
				stda = dfa[col].std()

				error, cinf, csup, SEM, stri = calc_confidence_interval_param(mua, stda, na, alpha=alpha, two_tailed=True)
				
				icount += 1
				dic[icount] = {}
				dic2 = dic[icount]

				dic2['run'] = run
				dic2['test'] = col
				dic2['test_desc'] = test
				dic2['compare'] = compare
				dic2['n'] = na
				dic2['mu'] = mua
				dic2['std'] = stda
				dic2['error'] = error
				dic2['SEM'] = SEM
				dic2['cinf'] = cinf
				dic2['csup'] = csup
				
				
				if i == 2:
					# G3xG0
					dic2['pval'] = None
					dic2['pval_bonf'] = None
					dic2['asteristics'] = None
					dic2['stat'] = None
					dic2['s_stat'] = None
				else:
					# G1 and G2xG0
					s_stat, stat, pval = calc_ttest(dfa[col], vals_G0G3)

					pval_bonf = alpha/2
					aster = stat_asteristics(pval=pval, threshold=pval_bonf, NS='--')

					dic2['pval'] = pval
					dic2['pval_bonf'] = pval_bonf
					dic2['asteristics'] = aster
					dic2['stat'] = stat
					dic2['s_stat'] = s_stat
					

		df = pd.DataFrame(dic).T

		ret = pdwritecsv(df, fname, self.root_curation, verbose=verbose)

		return df, df1, df2, df3




	# barplot_comparing_groups
	def barplot_comparing_confusion_groups(self, run:str, run_list:List, case_list:List, chosen_model_list:List,
								 width:int=1100, height:int=600, 
								 colors_blue:List=["blue", "blueviolet", "navy", "darkblue", "darkcyan"],
								 title0="Statistical parameters for Gi x G0 run %s",
								 fontsize:int=14, fontcolor:str='black',
								 margin:dict=dict( l=20, r=20, b=100, t=120, pad=4), plot_bgcolor:str="whitesmoke",
								 xaxis_title:str="parameters", yaxis_title:str='percentage (%)',
								 minus_test:float=-.1, minus_group:float=-0.05, 
								 annot_fontfamily:str="Arial, monospace", annot_fontsize:int=12, 
								 annot_fontcolor:str='black', savePlot:bool=True, alpha=0.05, 
								 show_bar_errors:bool=True, force:bool=False, verbose:bool=False):
		

		df, df1, df2, df3 = self.calc_gemini_4groups_confusion_stats(run=run, run_list=run_list,
																	 case_list=case_list, chosen_model_list=chosen_model_list,
																	 alpha=alpha, force=force, verbose=verbose)

		# cols = ['case', 'which', 'n', 'npos', 'nneg', 'TP', 'FP', 'TN', 'FN', 'sens', 'spec', 'accu', 'prec', 'f1_score']
		df1 = df1[df1.run == run]
		df2 = df2[df2.run == run]
		df3 = df3[df3.run == run]

		fig = go.Figure()
		
		title=title0%(run)

		x_num=0
		for col, test in self.mat_stat_tests:


			vals_G0G3 = df3[col]
			
			for i in range(3):
				if i==0:
					dfa = df1
					compare = 'G1xG0'
				elif i==1:
					dfa = df2
					compare = 'G2xG0'
				else:
					dfa = df3
					compare = 'G3xG0'

				dfaux = df[(df.run == run) & (df.test == col) & (df['compare'] == compare)]

				if dfaux.empty:
					print(f"Could not fid stats for {run}, {col}, {compare}")
					continue

				row = dfaux.iloc[0]

				x_num += 1
				label = f"{compare}-{col}"

				if show_bar_errors:
					fig.add_trace(go.Bar(x=[x_num], y=[row.mu], marker_color=colors_blue[i],
										 error_y=dict(type='data', array=[row.error]), name=label))
				else:
					fig.add_trace(go.Bar(x=[x_num], y=[row.mu], marker_color=colors_blue[i],
										 name=label))

				fig.add_annotation(
					x=x_num,
					y=minus_group,
					xref="x",
					yref="y",
					text=compare,
					showarrow=False,
					font=dict(
						family="Arial, monospace",
						size=fontsize,
						color=fontcolor
						),
					align="center", )
				
				if i == 2:
					# G3xG0
					fig.add_annotation(
						x=x_num-1,
						y=minus_test,
						xref="x",
						yref="y",
						text=test,
						showarrow=False,
						font=dict(
							family="Arial, monospace",
							size=fontsize,
							color=fontcolor
							),
						align="center", )

					x_num+=1

				else:
					fig.add_annotation(
						x=x_num,
						y = row.mu+row.error+0.05 if show_bar_errors else row.mu+0.05,
						xref="x",
						yref="y",
						text=f"{row.asteristics}",
						showarrow=False,
						font=dict(
							family="Arial, monospace",
							size=fontsize,
							color=fontcolor
							),
						align="center", )
						
		fig.update_layout(
					autosize=True,
					title=title,
					width=width,
					height=height,
					plot_bgcolor=plot_bgcolor,
					xaxis_title=xaxis_title,
					yaxis_title=yaxis_title,
					xaxis_showticklabels=False,
					# yaxis_range=[0, 1],
					showlegend=False,
					font=dict(
						family="Arial",
						size=14,
						color="Black"
					),
		)

		if savePlot:
			figname = title_replace(title)

			figname = os.path.join(self.root_figure, figname+'.html')

			fig.write_html(figname)
			if verbose: print(">>> HTML and png saved:", figname)
			fig.write_image(figname.replace('.html', '.png'))

		return fig

	def gemini_consensus_yes_no_doubt(self, dfpiva:pd.DataFrame, chosen_model_list:List, case:str, i_dfp:int, save_files:bool=True, 
									  verbose:bool=False) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):

		df2 = dfpiva[ (dfpiva.case == case) & (dfpiva.i_dfp == i_dfp)].copy()

		s_chosen_model_list = str(chosen_model_list)

		if df2.empty:
			print(f"There is no data for case {case}")
			return None, None, None, None

		df2.index = np.arange(len(df2))

		#------------------ Yes ----------------
		df_yes = df2[ df2.consensus == 'Yes' ].copy()
		df_yes.index = np.arange(len(df_yes))

		#------------------ Yes ----------------
		df_no = df2[ df2.consensus == 'No' ].copy()
		df_no.index = np.arange(len(df_no))

		#------------------ Doubt ----------------
		df_doubt = df2[ df2.consensus == 'Doubt' ].copy()
		df_doubt.index = np.arange(len(df_doubt))

		''' --------------------- save Ok --------------------------------'''
		if save_files:
			fname = self.fname_consensus_all_models_yes%(s_chosen_model_list, self.disease, self.suffix)
			pdwritecsv(df_yes, fname, self.root_gemini, verbose=verbose)

			fname = self.fname_consensus_all_models_no%(s_chosen_model_list, self.disease, self.suffix)
			pdwritecsv(df_no, fname, self.root_gemini, verbose=verbose)

			fname = self.fname_consensus_all_models_doubt%(s_chosen_model_list, self.disease, self.suffix)
			pdwritecsv(df_doubt, fname, self.root_gemini, verbose=verbose)

		return df2, df_yes, df_no, df_doubt


	def calc_gemini_consensus(self, row:pd.Series) -> (str, int, int, bool):
		n_yes, n_no = 0, 0
		
		cols = list(row.index)
		''' skip the first 4 columns - self.index_pivot = ['case', 'i_dfp', 'pathway_id', 'pathway'] '''
		cols = cols[self.n_index_pivot: ]

		for col in cols:
			if not isinstance(row[col], str):
				continue

			answer = row[col]
			if answer.startswith('Yes') or answer.startswith('Possible'):
				n_yes += 1
			else:
				n_no += 1

		if n_yes > n_no:
			consensus = 'Yes'
		elif n_yes < n_no:
			consensus = 'No'
		else:
			consensus = 'Doubt'

		unanimous = False if n_yes != 0 and n_no != 0 else True
			
		return consensus, n_yes, n_no, unanimous


	def select_random_results(self, case:str, chosen_model:int, N:int=30, root:str='.', query_type:str='_strong',
							  verbose:str=False) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):

		ret, _, _, _ = self.bpx.open_case(case, verbose=verbose)
		self.set_case(self.bpx.case, self.bpx.df_enr, self.bpx.df_enr0)
		self.set_gemini_num_model(chosen_model, verbose=verbose)

		Nhalf= int(N/2)
		self.N = N
		self.Nhalf = Nhalf
		self.query_type = query_type

		fnames = [x for x in os.listdir(root) if case in x and self.gemini_model in x and query_type in x 
											  and ('0_first' in x or '1_middle' in x or '2_final' in x)]

		if len(fnames) == 0:
			print(f"Nothing found for {case} and model {self.gemini_model} and query_type={query_type}")
			return None, None, None, None

		df_list = []
		for fname in fnames:
			df = pdreadcsv(fname, root)
			df_list.append(df)

		df = pd.concat(df_list)
		# it many have Yes and No ... is choosing one, randomly.
		df = df.drop_duplicates('pathway_id')
		df.index = np.arange(len(df))

		print(f"Found {len(df)} regs for {case} and model {self.gemini_model} and query_type={query_type}")

		df_yes = df[df.curation == 'Yes'].copy()
		df_yes.index = np.arange(len(df_yes))

		df_no = df[df.curation == 'No'].copy()
		df_no.index = np.arange(len(df_no))

		df_sel_yes = self.select_df_random(Nhalf, df_yes)
		df_sel_no  = self.select_df_random(Nhalf, df_no)

		return df_yes, df_no, df_sel_yes, df_sel_no

	def select_df_random(self, N:int, df2:pd.DataFrame) -> pd.DataFrame:
		n = len(df2)
		samples = np.arange(len(df2))

		ns = np.random.choice(samples, size=N, replace=False)
		cols = ['pathway_id', 'pathway', 'fdr', 'curation']
		df_sel = df2.loc[ns].copy()
		df_sel.index = np.arange(len(df_sel))

		return df_sel


	def merge_and_save_random_df_yes_no(self, N:int, case:str, query_type:str, 
										df_sel_yes:pd.DataFrame, df_sel_no:pd.DataFrame, 
										force:bool=False, verbose:str=False) -> pd.DataFrame:

		# select_random_results() set gemini model
		query_type = query_type.replace("_","")
		fname = self.fname_random_sampling %(N, case, self.gemini_model, query_type)
		filename = os.path.join(self.root_gemini_root, fname)

		if os.path.exists(filename) and not force:
			dff = pdreadcsv(fname, self.root_gemini_root, verbose=verbose)

			return dff

		dff = pd.concat([df_sel_yes, df_sel_no])
		dff = shuffle(dff)
		dff.index = np.arange(len(dff))

		cols = ['pathway_id', 'pathway', 'fdr', 'curation']
		dff = dff[cols]

		ret = pdwritecsv(dff, fname, self.root_gemini_root, verbose=verbose)

		return dff

	def open_yes_no_sampling(self, case:str, N:int=30, query_type:str='_strong',
							 verbose:str=False) -> (pd.DataFrame):

		gemini_model = self.gemini_models[self.chosen_model_sampling]

		query_type = query_type.replace("_","")
		fname = self.fname_random_sampling %(N, case, gemini_model, query_type)
		filename = os.path.join(self.root_gemini0, fname)

		if os.path.exists(filename):
			dff = pdreadcsv(fname, self.root_gemini0, verbose=verbose)
		else:
			print(f"File not found: '{filename}'")
			dff = pd.DataFrame()

		return dff


	def check_equal_vals_deprecated(self, row:pd.Series, transform:bool=True) -> bool:
		first = True
		first_val = None
		
		for col in self.curation_classes:
			if not isinstance(row[col], str):
				continue

			answer = self.transform_answer_deprecated(row[col]) if transform else row[col]
			if first:
				first_val = answer
				first = False
			else:
				if first_val != answer:
					return False

		return first_val is not None

	def transform_answer_deprecated(self, answer:str) -> str:
		if answer.startswith("Possible"):
			return "Yes"
		if answer.startswith("Los"):
			return "No"

		return answer
			

	def define_model_color(self, model_id:int, consensus:str=None):

		if model_id == 0:
			if consensus is None:
				color = 'blue'
			elif consensus == 'Yes':
				color = 'navy'
			elif consensus == 'No':
				color = 'blue'
			else:
				color = 'skyblue'
			
		elif model_id == 1:
			color = ''
			if consensus is None:
				color = 'red'
			elif consensus == 'Yes':
				color = 'darkred'
			elif consensus == 'No':
				color = 'red'
			else:
				color = 'orangered'
				
		elif model_id == 2:

			if consensus is None:
				color = 'orange'
			elif consensus == 'Yes':
				color = 'maroon'
			elif consensus == 'No':
				color = 'orange'
			else:
				color = 'gold'
				
		elif model_id == 3:
			if consensus is None:
				color = 'green'
			elif consensus == 'Yes':
				color = 'darkgreen'
			elif consensus == 'No':
				color = 'green'
			else:
				color = 'lightgreen'		
		else:
			color = 'gray'

		return color

	def open_summary_stat_dfpiv_all_models(self, run:str, case_list:List, chosen_model_list:List, 
										   verbose:bool=False) -> pd.DataFrame:

		self.set_run_seldata(run)
		s_chosen_model_list = str(chosen_model_list)

		fname = self.fname_summary_consensus_one_run_all_models%(run, s_chosen_model_list, self.disease, self.suffix)
		filename = os.path.join(self.root_gemini, fname)

		if not os.path.exists(filename):
			print(f"Could not find file: '{filename}'")
			dfsumm = pd.DataFrame()
		else:
			dfsumm = pdreadcsv(fname, self.root_gemini, verbose=verbose)
		
		return dfsumm


	def barplot_yes_no_idfp_models(self, run:str, case_list:List, 
								   chosen_model_list:List, i_dfp_list:List,
								   split:int=1, normalized:bool=False,
								   width:int=1100, height:int=600, 
								   fontsize:int=14, fontcolor:str='black',
								   margin:dict=dict( l=20, r=20, b=100, t=120, pad=4), 
								   plot_bgcolor:str="whitesmoke",
								   xaxis_title:str="cases", yaxis_title:str='n Yes',
								   minus_y_idfp:float=-1, minus_y_case:float=-5, 
								   line_width:int=2, title_font_color:str='navy', title_font_size:int=14,
								   annot_fontfamily:str="Arial, monospace", annot_fontsize:int=12, 
								   annot_fontcolor:str='black', savePlot:bool=True, verbose:bool=False) -> dict:

		if run is None or run == '':
			print("Define run")
			return None, ''

		if not isinstance(case_list, list) or case_list == []:
			print("Define case_list")
			return None, ''

		dfsumm = self.open_summary_stat_dfpiv_all_models(run=run, case_list=case_list, chosen_model_list=chosen_model_list, verbose=verbose)

		if dfsumm is None or dfsumm.empty:
			return None

		model_name_list = [self.gemini_models[x] for x in chosen_model_list]

		colors0 = {0: 'navy', 1: 'cyan',  2: 'orange', 3: 'red'}
		colors1 = {0: 'blue', 1: 'green', 2: 'brown', 3: 'pink'}

		if normalized:
			minus_y_idfp = -0.08
			minus_y_case = -0.2

			yaxis_title += ' percentage (%)'

		n_cases = len(case_list)
		del_cases = int(np.round(n_cases/split))
		ini = 0; icount=-1; dic_fig={}
		while (True):
			end = ini + del_cases
			i_cases = np.arange(ini, end)
			icount += 1

			title=f"Frequency of Yes per Group and models {chosen_model_list} for run {run}"
			if split > 1:
				title += f' - loop {icount}'


			fig = go.Figure()

			x_num = 0

			for case in [case_list[i] for i in i_cases]:
				for i_dfp in i_dfp_list:

					n_model = -1
					for chosen_model, model_name in zip(chosen_model_list, model_name_list):
						n_model += 1

						# dash options: "solid", "dot", "dash", "longdash", "dashdot", or  "longdashdot"
						if chosen_model == chosen_model_list[0]:
							color = colors0[i_dfp]
							first_model = True
							line_dash = 'solid'
						else:
							color = colors1[i_dfp]
							first_model = False
							line_dash = 'dash'

						df2 = dfsumm[ (dfsumm.case == case) & (dfsumm.i_dfp == i_dfp) & (dfsumm.chosen_model == chosen_model)]

						x_list = []
						y_list = []
						for q in self.question_list:
							df3 = df2[df2.question_type == q]

							if normalized:
								try:
									n_yes = df3.iloc[0]['n_yes'] / df3.iloc[0]['n']
								except:
									n_yes = -1
							else:
								n_yes = df3.iloc[0]['n_yes']
							
							y_list.append(n_yes)

							x_list.append(x_num)
							x_num += 1

						#  marker_color=color, mode='lines+markers',
						fig.add_trace(go.Line(x=x_list, y=y_list,
											  line = dict(color=color, width=line_width, dash=line_dash)))

						if first_model:
							fig.add_annotation(
									x=x_num-3,
									y=minus_y_idfp,
									xref="x",
									yref="y",
									text=f"G{i_dfp}",
									showarrow=False,
									font=dict(
										family="Arial, monospace",
										size=fontsize,
										color=fontcolor
										),
									align="left", textangle=-90)

						if first_model:
							x_num -= len(self.question_list)
							
					# for i_dfp in self.i_dfp_list:
					x_num += 1

				fig.add_annotation(
					x=x_num-10,
					y=minus_y_case,
					xref="x",
					yref="y",
					text=case,
					showarrow=False,
					font=dict(
						family="Arial, monospace",
						size=fontsize,
						color=fontcolor
						),
					align="center", ) 
				
				# one additional separation
				x_num += 1

			if normalized: title += ' - normalized'
			title2 = title + f'<br>solid for {model_name_list[0]} and dashed for {model_name_list[1]}'
			title2 += '<br>four point: modulated, mod&pubmed, strong mod, strong&pubemd'

			fig.update_layout(
						autosize=True,
						title=title2,
						width=width,
						height=height,
						plot_bgcolor=plot_bgcolor,
						xaxis_title=xaxis_title,
						yaxis_title=yaxis_title,
						xaxis_showticklabels=False,
						showlegend=False,
						title_font_color=title_font_color,
						title_font_size=title_font_size,
						font=dict(
							family="Arial",
							size=14,
							color="Black"
						),
			)

			dic_fig[icount] = fig

			if savePlot:
				figname = title_replace(title)

				figname = os.path.join(self.root_figure, figname+'.html')

				fig.write_html(figname)
				if verbose: print(">>> HTML and png saved:", figname)
				fig.write_image(figname.replace('.html', '.png'))

			ini += del_cases
			if ini >= n_cases: break

		return dic_fig


	# barplot_change_opinion --> barplot_consensus_yes_no_doubt
	def barplot_consensus_yes_no_doubt(self, run:str, case_list:List, chosen_model_list:List, i_dfp_list:List,
									   width:int=1100, height:int=600, fontsize:int=14, fontcolor:str='black',
									   margin:dict=dict( l=20, r=20, b=100, t=120, pad=4), plot_bgcolor:str="whitesmoke",
									   xaxis_title:str="cases", yaxis_title:str='n answers',
									   minus_y_consensus:float=-1, minus_y_i_dfp:float=-3, minus_y_case:float=-5, 
									   annot_fontfamily:str="Arial, monospace", annot_fontsize:int=12, 
									   annot_fontcolor:str='black', savePlot:bool=True, verbose:bool=False):

		if run is None or run == '':
			print("Define run")
			return None, ''

		if not isinstance(case_list, list) or case_list == []:
			print("Define case_list")
			return None, ''

		_, dfsumm = self.open_gemini_summary_consensus_statitics(chosen_model_list, verbose=verbose)

		dfsumm = dfsumm[dfsumm.run == run]

		model_name_list = [self.gemini_models[x] for x in chosen_model_list]

		fig = go.Figure()

		title=f"number of answeres for run {run}"
		x_num = 0
		
		for case in case_list:
			for i_dfp in i_dfp_list:
				
				n_model = -1
				for chosen_model, model_name in zip(chosen_model_list, model_name_list):
					n_model += 1

					df2 = dfsumm[ (dfsumm.case == case) & (dfsumm.i_dfp == i_dfp) & (dfsumm.chosen_model == chosen_model)]
				
					for consensus in ['Yes', 'No', 'Doubt']:
						color = self.define_model_color(model_id, consensus)
						
						df3 = df2[ df2.consensus == consensus ]
						if df3.empty:
							n = 0
						else:
							n = df3.iloc[0]['n']
				
						x_num += 1
				
						fig.add_trace(go.Bar(x=[x_num], y=[n], marker_color=color, name=model_name))
				
						fig.add_annotation(
								x=x_num,
								y=minus_y_consensus,
								xref="x",
								yref="y",
								text=consensus,
								showarrow=False,
								font=dict(
									family="Arial, monospace",
									size=fontsize,
									color=fontcolor
									),
								align="left", textangle=-90)
				# for i_dfp in self.i_dfp_list:
				x_num += 1
				fig.add_annotation(
						x=x_num-4.5,
						y=minus_y_i_dfp,
						xref="x",
						yref="y",
						text=i_dfp,
						showarrow=False,
						font=dict(
							family="Arial, monospace",
							size=fontsize,
							color=fontcolor
							),
						align="left")
				
			fig.add_annotation(
				x=x_num-20,
				y=minus_y_case,
				xref="x",
				yref="y",
				text=case,
				showarrow=False,
				font=dict(
					family="Arial, monospace",
					size=fontsize,
					color=fontcolor
					),
				align="center", ) 
			
			# one additional separation
			x_num += 1


		model_id_list   = dfsumm.model_id.unique()
		model_name_list = dfsumm.model_name.unique()

		s_colors = ''
		for i in range(len(model_id_list)):
			model_id = model_id_list[i]
			model_name = model_name_list[i]
			
			color = self.define_model_color(model_id)

			if s_colors == '':
				s_colors = f'{model_name} is {color}'
			else:
				s_colors += f'; {model_name} is {color}'

		fig.update_layout(
					autosize=True,
					title=title,
					width=width,
					height=height,
					plot_bgcolor=plot_bgcolor,
					xaxis_title=xaxis_title,
					yaxis_title=yaxis_title,
					xaxis_showticklabels=False,
					showlegend=False,
					font=dict(
						family="Arial",
						size=14,
						color="Black"
					),
		)

		if savePlot:
			figname = title_replace(title)

			figname = os.path.join(self.root_figure, figname+'.html')

			fig.write_html(figname)
			if verbose: print(">>> HTML and png saved:", figname)
			fig.write_image(figname.replace('.html', '.png'))

		return fig, s_colors


	def barplot_yes_no_opinion(self, run:str, yes_no_doubt:str, case_list:List, chosen_model_list:List,
							   width:int=1100, height:int=600, fontsize:int=14, fontcolor:str='black',
							   margin:dict=dict( l=20, r=20, b=100, t=120, pad=4), plot_bgcolor:str="whitesmoke",
							   xaxis_title:str="cases", yaxis_title:str='n answers',
							   minus_y_consensus:float=-3, minus_y_i_dfp=-1, minus_y_case:float=-5, 
							   annot_fontfamily:str="Arial, monospace", annot_fontsize:int=12, 
							   annot_fontcolor:str='black',
							   savePlot:bool=True, verbose:bool=False):

		dfsumm = self.open_gemini_dfpiva_all_models_one_run(run=run, chosen_model_list=chosen_model_list, force=False, verbose=verbose)

		if run is None or run == '':
			print("Define run")
			return None, ''

		if not isinstance(case_list, list) or case_list == []:
			print("Define case_list")
			return None, ''

		if not isinstance(yes_no_doubt, str):
			print("Define cyes_no_doubt as Yes, No, or Doubt (1)")
			return None, ''

		if yes_no_doubt not in ['Yes', 'No', 'Doubt']:
			print("Define yes_no_doubt as Yes, No, or Doubt (2)")
			return None, ''
			
		fig = go.Figure()

		
		title=f"{yes_no_doubt} plot x number of answeres for run {run}"
		x_num = -1
		
		consensus = yes_no_doubt
		i_case_num = -1
		
		for case in case_list:
			
			df2 = dfsumm[ (dfsumm.case == case) & (dfsumm.consensus == consensus)]  
			if df2.empty:
				continue

			i_case_num += 1

			fig.add_annotation(
					x=x_num+1,
					y=minus_y_consensus,
					xref="x",
					yref="y",
					text=consensus,
					showarrow=False,
					font=dict(
						family="Arial, monospace",
						size=fontsize,
						color=fontcolor
						),
					align="left")
			
			model_id_list   = dfsumm.model_id.unique()
			model_name_list = dfsumm.model_name.unique()
			n_model = -1
			for model_id in model_id_list:
				n_model += 1
				model_name = model_name_list[n_model]

				color = self.define_model_color(model_id)

				# 0, 1, 3 ....
				for i_dfp in self.i_dfp_list:
					df3 = df2[ (df2.i_dfp == i_dfp) & (df2.model_id == model_id) ]
					df3 = df3[df3.model_id.isin(self.chosen_model_list)]

					if df3.empty:
						n = 0
					else:
						n = df3.iloc[0]['n']
			
					fig.add_trace(go.Bar(x=[x_num], y=[n], marker_color=color, name=model_name))

					if n_model == 0:
						fig.add_annotation(
								x=x_num,
								y=minus_y_i_dfp,
								xref="x",
								yref="y",
								text=i_dfp,
								showarrow=False,
								font=dict(
									family="Arial, monospace",
									size=fontsize,
									color=fontcolor
									),
								align="left")
					x_num += 1

				# changing model, space
				x_num += 1

			fig.add_annotation(
				x=x_num-7.5,
				y=minus_y_case,
				xref="x",
				yref="y",
				text=case,
				showarrow=False,
				font=dict(
					family="Arial, monospace",
					size=fontsize,
					color=fontcolor
					),
				align="center", ) 

			# changing case
			x_num += 4
			
		model_id_list   = dfsumm.model_id.unique()
		model_name_list = dfsumm.model_name.unique()

		s_colors = ''
		for i in range(len(model_id_list)):
			model_id = model_id_list[i]
			model_name = model_name_list[i]
			
			color = self.define_model_color(model_id)

			if s_colors == '':
				s_colors = f'{model_name} is {color}'
			else:
				s_colors += f'; {model_name} is {color}'

		fig.update_layout(
					autosize=True,
					title=title,
					width=width,
					height=height,
					plot_bgcolor=plot_bgcolor,
					xaxis_title=xaxis_title,
					yaxis_title=yaxis_title,
					xaxis_showticklabels=False,
					showlegend=False,
					font=dict(
						family="Arial",
						size=14,
						color="Black"
					),
		)

		if savePlot:
			figname = title_replace(title)
			figname = os.path.join(self.root_figure, figname+'.html')

			fig.write_html(figname)
			if verbose: print(">>> HTML and png saved:", figname)
			fig.write_image(figname.replace('.html', '.png'))

		return fig, s_colors


	def barplot_yes_no_per_case_run(self, run:str, case_list:List, 
									chosen_model_list:List, i_dfp_list:List,
									width:int=1100, height:int=600, 
									fontsize:int=14, fontcolor:str='black',
									margin:dict=dict( l=20, r=20, b=100, t=120, pad=4), plot_bgcolor:str="whitesmoke",
									xaxis_title:str="cases", yaxis_title:str='n answers',
									minus_y_yes_no:float=-1, minus_y_i_dfp:float=-3, minus_y_case:float=-7, 
									annot_fontfamily:str="Arial, monospace", annot_fontsize:int=12, 
									annot_fontcolor:str='black', savePlot:bool=True, verbose:bool=False):

		if run is None or run == '':
			print("Define a run like 'run01'.")
			return None

		if not isinstance(case_list, list) or case_list == []:
			print("Define case_list")
			return None

		dfsumm = self.summary_stat_dfpiv_all_models(run=run, case_list=case_list, chosen_model_list=chosen_model_list,
													verbose=verbose)

		if dfsumm is None or dfsumm.empty:
			return None

		x_shitf_left = 5 if self.is_seldata else 20  ## (3 * 3 + 1) = echa i_dfp occupies 10 positions (x_nums)

		# 0, 1, 3 ....
		model_name_list = [self.gemini_models[chosen_model] for chosen_model in chosen_model_list]

		fig = go.Figure()

		title=f"Number of answers yes-no per case, model, i_dfp for run {run}"
		
		toggle=False; ymaxi=-1
		x_num = 0

		for case in case_list:
			icase=0
			toggle = True if not toggle else False
			
			for i_dfp in i_dfp_list:
				i_dfp_aux=0
				
				df2 = dfsumm[ (dfsumm.case == case) & (dfsumm.i_dfp == i_dfp) ]
				
				for chosen_model, model_name in zip(chosen_model_list, model_name_list):
					df3 = df2[df2.chosen_model == chosen_model]

					if not df3.empty:
						x_num += 1; i_dfp_aux+=1
						fig.add_trace(go.Bar(x=[x_num], y=df3.n_yes, marker_color='navy' if toggle else 'darkgreen'))

				if icase == 0:
					fig.add_annotation(
						x=x_num-1,
						y=minus_y_yes_no,
						xref="x",
						yref="y",
						text='yes',
						showarrow=False,
						font=dict(
							family="Arial, monospace",
							size=fontsize,
							color=fontcolor
							),
						align="center", ) 
				
				x_num += 1; i_dfp_aux+=1; icase+=1

				for chosen_model, model_name in zip(chosen_model_list, model_name_list):
					df3 = df2[df2.chosen_model == chosen_model]

					if not df3.empty:
						x_num += 1; i_dfp_aux+=1
						fig.add_trace(go.Bar(x=[x_num], y=df3.n_no, marker_color='darkred' if toggle else 'darkorange'))

				if icase == 1:
					fig.add_annotation(
						x=x_num-1,
						y=minus_y_yes_no,
						xref="x",
						yref="y",
						text='no',
						showarrow=False,
						font=dict(
							family="Arial, monospace",
							size=fontsize,
							color=fontcolor
							),
						align="center", ) 
				
				x_num += 1; i_dfp_aux+=1; icase+=1

				for chosen_model, model_name in zip(chosen_model_list, model_name_list):
					df3 = df2[df2.chosen_model == chosen_model]

					maxi = df3.n.max()
					if maxi > ymaxi:
						ymaxi = maxi

					if not df3.empty:
						x_num += 1; i_dfp_aux+=1
						fig.add_trace(go.Bar(x=[x_num], y=df3.n, marker_color='black'))

				
				fig.add_annotation(
					x=x_num-i_dfp_aux+1,
					y=minus_y_i_dfp,
					xref="x",
					yref="y",
					text=i_dfp,
					showarrow=False,
					font=dict(
						family="Arial, monospace",
						size=fontsize,
						color=fontcolor
						),
					align="center", ) 

				x_num += 2
			
			fig.add_annotation(
				x=x_num-x_shitf_left,  ## (3 * 3 + 1) * 2
				y=minus_y_case,
				xref="x",
				yref="y",
				text=case,
				showarrow=False,
				font=dict(
					family="Arial, monospace",
					size=fontsize,
					color=fontcolor
					),
				align="center", )
				

		mini = np.min( [minus_y_yes_no, minus_y_i_dfp, minus_y_case] ) -5
		yaxis_range = [mini, ymaxi*1.1]

		fig.update_layout(
					autosize=True,
					title=title,
					width=width,
					height=height,
					plot_bgcolor=plot_bgcolor,
					xaxis_title=xaxis_title,
					yaxis_title=yaxis_title,
					xaxis_showticklabels=False,
					yaxis_range=yaxis_range,
					showlegend=False,
					font=dict(
						family="Arial",
						size=14,
						color="Black"
					),
		)

		if savePlot:
			figname = title_replace(title) + f'for_{self.suffix}'
			figname = os.path.join(self.root_figure, figname + '.html')

			fig.write_html(figname)
			if verbose: print(">>> HTML and png saved:", figname)
			fig.write_image(figname.replace('.html', '.png'))

		return fig


	def calc_venn_params_between_2models(self, run:str, filter:str, case:str, i_dfp:int, 
										 model_name0:str, df0:pd.DataFrame, 
										 model_name1:str, df1:pd.DataFrame, 
										 only_common_pathways:bool=False, verbose:bool=False):
		n0 = len(df0)
		n1 = len(df1)
		
		df0 = df0[ (df0.case == case) & (df0.i_dfp == i_dfp) ]
		n0 = len(df0)
		df1 = df1[ (df1.case == case) & (df1.i_dfp == i_dfp) ]
		n1 = len(df1)

		if only_common_pathways:
			lista = df0.pathway.to_list()
			df1 = df1[df1.pathway.isin(lista)]
		
			lista = df1.pathway.to_list()
			df0 = df0[df0.pathway.isin(lista)]


		label0 = model_name0.replace("gemini-","")
		label1 = model_name1.replace("gemini-","")

		if filter == 'Yes':
			title_ini = f"Consensus 'Yes' for case '{case}' and {run}"

		elif filter == 'No':
			title_ini = f"Consensus 'No' for case '{case}' and {run}"

		elif filter == 'Doubt':
			title_ini = f"Consensus 'Doubt' for case '{case}' and {run}"

		else:
			print("Choose, correctly, all, Yes, No, or Doubt")
			return [None] * 21

		df0 = df0[ (df0.consensus == filter) ]
		df1 = df1[ (df1.consensus == filter) ]   
		
		vals0 = list(np.unique(df0.pathway))
		vals1 = list(np.unique(df1.pathway))
		
		set0 = set(vals0)
		set1 = set(vals1)

		all_vals = list(np.unique(vals0 + vals1))
		
		commons = list(set0.intersection(set1))
		commons.sort()

		only0 = [x for x in vals0 if x not in commons]
		only0.sort()
		
		only1 = [x for x in vals1 if x not in commons]
		only1.sort()


		n_tot = len(commons) + len(only0) + len(only1)

		if n_tot == 0:
			perc_commons = 0
		else:
			perc_commons = len(commons) / n_tot

		s_agree = f"Models agree in {100*perc_commons:.2f}%"
		s_commons = f"{s_agree}. Common (#{len(commons)}): {', '.join(commons)}"

		#-------------------- default -------------------
		ret, _, _, _ = self.bpx.open_case_params(case, abs_lfc_cutoff=1, fdr_lfc_cutoff=0.05, pathway_fdr_cutoff=0.05)
		df_enr = self.bpx.df_enr

		if filter == 'Yes':
			if df_enr is None or df_enr.empty:
				vals_default, common_default0,  common_default1 = [], None, None
				new_pathways0 = vals0
				new_pathways1 = vals1
			else:
				vals_default = list(np.unique(df_enr.pathway))
				set_default  = set(vals_default)
		
				new_pathways0 = [x for x in vals0 if x not in vals_default]
				new_pathways1 = [x for x in vals1 if x not in vals_default]
		
				common_default0 = list(set_default.intersection(set0))
				common_default0.sort()
				common_default1 = list(set_default.intersection(set1))
				common_default1.sort()
		else:
				vals_default, common_default0,  common_default1 = None, None, None
				new_pathways0, new_pathways1 = None, None


		return perc_commons, s_agree, s_commons, all_vals, vals0, vals1, \
			   n0, n1, set0, set1, label0, label1, commons, only0, only1, \
			   vals_default, common_default0, common_default1, new_pathways0, new_pathways1, title_ini

	def venn_diagram_between_2models(self, run:str, filter:str, case:str, i_dfp:int, 
									 model_name0:str, df0:pd.DataFrame, 
									 model_name1:str, df1:pd.DataFrame, 
									 only_common_pathways:bool=False, print_plot:bool=True,
									 title_font_size:int=12, figsize:List=(12,8), dpi:int=600,
									 save:bool=False, verbose:bool=False):


		perc_commons, s_agree, s_commons, all_vals, vals0, vals1, n0, n1, set0, set1, label0, label1, commons, only0, only1, \
		vals_default, common_default0, common_default1, new_pathways0, new_pathways1, title_ini = \
						 self.calc_venn_params_between_2models(run=run, filter=filter, case=case, i_dfp=i_dfp,
						 model_name0=model_name0, df0=df0, 
						 model_name1=model_name1, df1=df1, 
						 only_common_pathways=only_common_pathways, verbose=False)


		if commons==0 and only0==0 and only1==0:
			print("No commons, no only0, and no only1.")
			return None

		#-------------------- default end ---------------

		if i_dfp == 0:
			title_ini += '\nfor the enriched pathways'
		elif i_dfp == 1:
			title_ini += '\nfor the middle of the pathway table'
		elif i_dfp == 2:
			title_ini += '\nfor the end of the pathway table'
		elif i_dfp == 3:
			title_ini += '\nfor other random pathways'
		else:
			print("Choose i_dfp from 0 to 2: enriched, middle, and end")
			return [None] * 15

		if df0.empty and df1.empty:
			if verbose: print(f"No df0 and df1 for {case} i_dfp {i_dfp}")
			return [None] * 15


		title_ini += ', ' + s_agree[0].lower() + s_agree[1:]

		if verbose:
			print(f"Case {case} i_dfp {i_dfp}")
			print(f"Model {model_name0} has {n0} / {len(df0)}")
			print(f"Model {model_name1} has {n1} / {len(df1)}")
			print(s_agree)
			
		s_all	 = f"all pathways (#{len(all_vals)}): {', '.join(all_vals)}"
		s_label0 = f"only {label0} (#{len(only0)}): {', '.join(only0)}"
		s_label1 = f"only {label1} (#{len(only1)}): {', '.join(only1)}"
		
		title = f"{title_ini}\ntotal {filter} pathways {len(all_vals)}, {label0}={len(set0)}/{n0}, {label1}={len(set1)}/{n1}"

		if print_plot:
			fig = plt.figure(figsize=figsize)
			venn2([set0, set1], (label0, label1))
			plt.title(title, size=title_font_size)

			if save:
				fname = title_replace(title)
				fname = fname.split('_models_agree')[0].strip().replace("'","") + ".png"
				filename = os.path.join(self.root_figure, fname)

				# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
				plt.savefig(filename, dpi=dpi, format='png')
		else:
			fig = None

		return fig, title.replace('\n', ' '), perc_commons, commons, n0, n1, only0, only1


	# run_all_comparing_geminis_by_model --> 
	def run_all_inter_model_soft_consensus_repro(self, chosen_model0:int, chosen_model1:int, run_list:List, 
												 case_list:List, force:bool=False, 
												 verbose:bool=False) -> (str, pd.DataFrame, pd.DataFrame):

		gemini_model0 = self.gemini_models[chosen_model0]
		gemini_model1 = self.gemini_models[chosen_model1]

		#------------------ all i_dfps --------------------
		fname = self.fname_inter_model_repro%(self.disease, chosen_model0, chosen_model1, self.suffix)
		filename = os.path.join(self.root_gemini_root, fname)

		fname_txt = fname.replace('.tsv', '_msg.txt')
		filename_txt = os.path.join(self.root_gemini_root, fname_txt)

		#------------------ per i_dfp ---------------------
		fname_per_idfp = self.fname_inter_model_repro_per_idfp%(self.disease, chosen_model0, chosen_model1, self.suffix)
		filename_idfp = os.path.join(self.root_gemini_root, fname_per_idfp)

		if os.path.exists(filename_idfp) and os.path.exists(filename) and os.path.exists(filename_txt) and not force:
			df	  = pdreadcsv(fname, self.root_gemini_root, verbose=verbose)
			df_idfp = pdreadcsv(fname_per_idfp,	self.root_gemini_root, verbose=verbose)
			msg	 = read_txt(fname_txt, self.root_gemini_root, verbose=verbose)
			return msg, df, df_idfp

		msg = f"comparing {gemini_model0}  x  {gemini_model1}\n"

		dic = {}; icount=-1
		for run in run_list:
			
			msg += f"\trun {run}\n"
			for case in case_list:
				msg += f"\t\tcase={case}\n"

				for i_dfp in self.i_dfp_list:
					msg2, dfn = self.run_inter_model_soft_consensus_repro_idfp(run=run, case=case,  
																			   chosen_model0=chosen_model0,
																			   chosen_model1=chosen_model1, 
																			   i_dfp=i_dfp, verbose=verbose)
					'''
					# save detailed run, case, i_dfp table
					cols = list(dfn.columns)
					dfn['run']   = run
					dfn['chosen_model0'] = chosen_model0
					dfn['model_name0']   = gemini_model0
					dfn['chosen_model1'] = chosen_model1
					dfn['model_name1']   = gemini_model1

					cols = ['run', 'chosen_model0', 'model_name0', 'chosen_model1', 'model_name1'] + cols
					dfn = dfn[cols]

					fname_idfp = self.fname_inter_model_repro_idfp%(self.disease, chosen_model0, chosen_model1, run, case, i_dfp, self.suffix)
					ret1 = pdwritecsv(dfn, fname_idfp, self.root_gemini_root, verbose=verbose)
					'''

					# summarize data per run, case, i_dfp
					icount += 1
					dic[icount] = {}
					dic2 = dic[icount]

					dic2['run']   = run
					dic2['case']  = case
					dic2['i_dfp'] = i_dfp

					dic2['chosen_model0'] = chosen_model0
					dic2['model_name0']   = gemini_model0
					dic2['chosen_model1'] = chosen_model1
					dic2['model_name1']   = gemini_model1

					if dfn is None or dfn.empty:
						dic2['text'] = msg2
					else:
						dic2['n'] = len(dfn)
						dic2['n_consensus']		= dfn.are_cons_equal.sum()
						dic2['n_consensus_yes'] = dfn.are_cons_yes_equal.sum()
						dic2['n_unanimous']	 = dfn.are_unan_equal.sum()

						dic2['mean_consensus'] = dfn.are_cons_equal.mean()
						dic2['std_consensus']  = dfn.are_cons_equal.std()

						dic2['mean_cons_yes']  = dfn.are_cons_yes_equal.mean()
						dic2['std_cons_yes']   = dfn.are_cons_yes_equal.std()

						dic2['mean_unanimous'] = dfn.are_unan_equal.mean()
						dic2['std_unanimous']  = dfn.are_unan_equal.std()

						dic2['text'] = msg2

					msg += f"\t\t\t{msg2}\n"
				msg += '\n'
			msg += '\n'

		df_idfp = pd.DataFrame(dic).T
		ret1 = pdwritecsv(df_idfp, fname_per_idfp, self.root_gemini_root, verbose=verbose)

		#------------------ all i_dfps --------------------
		msg += '\n\nSummary i_dfp independent\n\n'

		cols = ['run', 'chosen_model0', 'model_name0', 'chosen_model1', 'model_name0',
				'case', 'pathway_id', 'pathway', 'consensus0', 
				'n_yes0', 'n_no0', 'consensus1', 'n_yes1', 'n_no1',
				'are_cons_equal', 'are_cons_yes_equal', 'are_unan_equal']


		dic={}; icount=-1
		for run in run_list:
			
			msg += f"\trun {run}\n"
			for case in case_list:
				msg += f"\t\tcase={case}\n"

				msg2, dfn = self.run_inter_model_soft_consensus_repro(run=run, case=case, chosen_model0=chosen_model0, 
																	 chosen_model1=chosen_model1, verbose=verbose)

				if dfn is None:
					simil, std = None, None
					simil_yes_no, std_yes_no = None, None
					unan_simil, std_unan = None, None
					n, n_consensus, n_consensus_yes, n_unanimous = None, None, None, None
				
				else:
					simil = dfn.are_cons_equal.mean()
					std   = dfn.are_cons_equal.std()

					simil_yes_no = dfn.are_cons_yes_equal.mean()
					std_yes_no   = dfn.are_cons_yes_equal.std()

					unan_simil = dfn.are_unan_equal.mean()
					std_unan   = dfn.are_unan_equal.std()

					msg2 += f" with similarity = {100*simil:.1f}% ({100*std:.1f}%)"
					msg2 += f"\nand with similarity yes-yes = {100*simil_yes_no:.1f}% ({100*std_yes_no:.1f}%)"

					n = len(dfn)
					n_consensus	 = dfn.are_cons_equal.sum()
					n_consensus_yes = dfn.are_cons_yes_equal.sum()
					n_unanimous	 = dfn.are_unan_equal.sum()
				

				# summarize data per run, case
				icount += 1
				dic[icount] = {}
				dic2 = dic[icount]

				dic2['run']  = run
				dic2['case'] = case

				dic2['chosen_model0'] = chosen_model0
				dic2['model_name0']   = gemini_model0
				dic2['chosen_model1'] = chosen_model1
				dic2['model_name1']   = gemini_model1

				dic2['n'] = n
				dic2['n_consensus']	 = n_consensus
				dic2['n_consensus_yes'] = n_consensus_yes
				dic2['n_unanimous']	 = n_unanimous

				dic2['mean_consensus'] = simil
				dic2['std_consensus']  = std
				dic2['mean_cons_yes']  = simil_yes_no
				dic2['std_cons_yes']   = std_yes_no

				dic2['mean_unanimous'] = unan_simil
				dic2['std_unanimous']  = std_unan

				dic2['msg'] = msg2

				msg += f"\t\t\t{msg2}\n"
			msg += '\n'

		df = pd.DataFrame(dic).T

		ret1 = pdwritecsv(df, fname, self.root_gemini_root, verbose=verbose)
		ret2 = write_txt(msg, fname_txt, self.root_gemini_root, verbose=verbose)

		return msg, df, df_idfp


	def run_inter_model_soft_consensus_repro_idfp(self, run:str, case:str, 
												  chosen_model0:int, chosen_model1:int, i_dfp:int=0, 
												  verbose:bool=False) -> (str, pd.DataFrame):

		#----------- first model -----------
		dfpiv0 = self.open_dfpiv_semantic_consensus_run_per_model(run=run, chosen_model=chosen_model0, verbose=verbose)

		if dfpiv0 is None or dfpiv0.empty:
			msg = f"-> dfpiv was not calculated dfpiv model {chosen_model0}"
			return msg, None

		model_name0 = self.gemini_model
		dfpiv0 = dfpiv0[ (dfpiv0.case==case) ]

		if dfpiv0 is None or dfpiv0.empty:
			msg = f"-> dfpiv was empty for model {chosen_model0} case {case}"
			return msg, None


		#----------- second model -----------
		dfpiv1 = self.open_dfpiv_semantic_consensus_run_per_model(run=run, chosen_model=chosen_model1, verbose=verbose)

		if dfpiv1 is None or dfpiv1.empty:
			msg = f"-> dfpiv was not calculated dfpiv model {chosen_model1}"
			return msg, None

		model_name1 = self.gemini_model

		dfpiv1 = dfpiv1[ (dfpiv1.case==case) ]

		if dfpiv1 is None or dfpiv1.empty:
			msg = f"-> dfpiv was empty for model {chosen_model1} case {case}"
			return msg, None

		cols = ['case', 'i_dfp', 'pathway_id', 'pathway', 'consensus', 'n_yes', 'n_no', 'unanimous']

		dfpiv0 = dfpiv0[ (dfpiv0.i_dfp==i_dfp) ]
		dfpiv1 = dfpiv1[ (dfpiv1.i_dfp==i_dfp) ]

		common_cols = ['case', 'i_dfp', 'pathway_id', 'pathway']

		dfn = pd.merge(dfpiv0[cols], dfpiv1[cols], how="inner", on=common_cols)

		cols2 = ['case', 'i_dfp', 'pathway_id', 'pathway',
				 'consensus0', 'n_yes0', 'n_no0', 'unanimous0', 
				 'consensus1', 'n_yes1', 'n_no1', 'unanimous1']

		dfn.columns = cols2

		dfn['are_cons_equal']		= [ (dfn.iloc[i].consensus0 == dfn.iloc[i].consensus1) for i in range(len(dfn))]
		dfn['are_cons_yes_equal']	= [ (dfn.iloc[i].consensus0 == dfn.iloc[i].consensus1) and
										(dfn.iloc[i].n_yes0 == dfn.iloc[i].n_yes1) for i in range(len(dfn))]
			
		dfn['are_unan_equal']		= [ (dfn.iloc[i].unanimous0 == dfn.iloc[i].unanimous1) for i in range(len(dfn))]

		simil = dfn.are_cons_equal.mean()
		std   = dfn.are_cons_equal.std()

		msg = f"For {run}, case={case}, i_dfp={i_dfp} has {len(dfn)} pathways with similarity = {100*simil:.1f}% ({100*std:.1f}%)"

		simil = dfn.are_cons_yes_equal.mean()
		std   = dfn.are_cons_yes_equal.std()

		msg += f"\nand with similarity yes-yes no-no = {100*simil:.1f}% ({100*std:.1f}%)"

		return msg, dfn


	def run_inter_model_soft_consensus_repro(self, run:str, case:str, chosen_model0:int, chosen_model1:int, 
											 verbose:bool=False) -> (str, pd.DataFrame):

		#----------- first model -----------
		dfpiv0 = self.open_dfpiv_semantic_consensus_run_per_model(run=run, chosen_model=chosen_model0, verbose=verbose)

		if dfpiv0 is None or dfpiv0.empty:
			msg = f"-> dfpiv was not calculated dfpiv model {chosen_model0}"
			return msg, None

		model_name0 = self.gemini_model
		dfpiv0 = dfpiv0[ (dfpiv0.case==case) ]

		if dfpiv0 is None or dfpiv0.empty:
			msg = f"-> dfpiv was empty for model {chosen_model0} case {case}"
			return msg, None


		#----------- second model -----------
		dfpiv1 = self.open_dfpiv_semantic_consensus_run_per_model(run=run, chosen_model=chosen_model1, verbose=verbose)

		if dfpiv1 is None or dfpiv1.empty:
			msg = f"-> dfpiv was not calculated dfpiv model {chosen_model1}"
			return msg, None

		model_name1 = self.gemini_model

		dfpiv1 = dfpiv1[ (dfpiv1.case==case) ]

		if dfpiv1 is None or dfpiv1.empty:
			msg = f"-> dfpiv was empty for model {chosen_model1} case {case}"
			return msg, None

		cols = ['case', 'pathway_id', 'consensus', 'n_yes', 'n_no', 'unanimous']

		common_cols = ['case', 'pathway_id']

		dfn = pd.merge(dfpiv0[cols], dfpiv1[cols], how="inner", on=common_cols)

		cols2 = common_cols + ['consensus0', 'n_yes0', 'n_no0', 'unanimous0', 
							   'consensus1', 'n_yes1', 'n_no1', 'unanimous1']
		dfn.columns = cols2

		dfn['are_cons_equal']	 = [ (dfn.iloc[i].consensus0 == dfn.iloc[i].consensus1) for i in range(len(dfn))]
		dfn['are_cons_yes_equal'] = [ (dfn.iloc[i].consensus0 == dfn.iloc[i].consensus1) and
									  (dfn.iloc[i].n_yes0 == dfn.iloc[i].n_yes1) for i in range(len(dfn))]

		dfn['are_unan_equal']	 = [ (dfn.iloc[i].unanimous0 == dfn.iloc[i].unanimous1) for i in range(len(dfn))]

		msg = f"For {run}, case={case}, has {len(dfn)} pathways"

		return msg, dfn



	# calc_stat_gemini_compare_2_models
	def calc_stat_inter_model_soft_consensus_venn(self, run_list:List, case_list:List, i_dfp_list:List, 
												  model0:int, model1:int, only_common_pathways:bool, force:bool=False, 
												  verbose:bool=False) -> (pd.DataFrame, str, pd.DataFrame, str, pd.DataFrame, str):

		self.set_gemini_num_model(model0)
		model_name0 = self.gemini_model

		self.set_gemini_num_model(model1)
		model_name1 = self.gemini_model

		fname = self.fname_inter_model_stat%(self.disease, model_name0, model_name1, self.suffix)
		filename = os.path.join(self.root_gemini_root, fname)

		if os.path.exists(filename) and not force:
			return pdreadcsv(fname, self.root_gemini_root, verbose=verbose)

		icount=-1; dic={}
		for run in run_list:

			dfpiv0, model_name0, dfpiv1, model_name1, dff, fname2 = \
							self.run_inter_model_soft_consensus_venn(run=run, case_list=case_list, 
																	 i_dfp_list=i_dfp_list, model0=model0, model1=model1, 
																	 only_common_pathways=only_common_pathways, verbose=verbose)
			
			mu  = dff.perc_common_consensus.mean()
			std = dff.perc_common_consensus.std()
			
			text = "\n\nFor all comparisons:\n"
			text += f"\tperc_common_consensus between '{model_name0}' x '{model_name1}'\n"
			text += f"\tpercentage {100*mu:.1f}% ({100*std:.1f}%), n={len(dff)}\n\n"

			icount += 1
			dic[icount] = {}
			dic2 = dic[icount]

			dic2 = dic[icount]
			dic2['run']	   = run
			dic2['model0'] = model_name0
			dic2['model1'] = model_name1
			dic2['mean_all'] = mu
			dic2['std_all']  = std

			
			dff2 = dff[ (dff.i_dfp == 0) & (dff.consensus.isin(['Yes', 'No']))]
			
			mu  = dff2.perc_common_consensus.mean()
			std = dff2.perc_common_consensus.std()
			
			text += "Only for enriched pathways, Yes/No consensus - without Doubt.\n\n"
			text += f"\tperc_common_consensus for enr.pathways only Yes/No between '{model_name0}' x '{model_name1}'\n"
			text += f"\tpercentage {100*mu:.1f}% ({100*std:.1f}%), n={len(dff2)}\n"

			# if verbose: print(text)

			dic2['mean_enr_yes_no'] = mu
			dic2['std_enr_yes_no']  = std
			dic2['text'] = text

		df_stat2 = pd.DataFrame(dic).T
		ret = pdwritecsv(df_stat2, fname, self.root_gemini_root, verbose=verbose)

		return df_stat2


	# compare_2_models_one_run_cases
	def run_inter_model_soft_consensus_venn(self, run:str, case_list:List, i_dfp_list:List, 
											model0:int, model1:int, only_common_pathways:bool=False, force:bool=False, 
											verbose:bool=False) -> (pd.DataFrame, str, pd.DataFrame, str, pd.DataFrame, str):

		self.set_gemini_num_model(model0)
		model_name0 = self.gemini_model
		dfpiv0 = self.open_dfpiv_semantic_consensus_run_per_model(run=run, chosen_model=model0, verbose=verbose)

		self.set_gemini_num_model(model1)
		model_name1 = self.gemini_model
		dfpiv1 = self.open_dfpiv_semantic_consensus_run_per_model(run=run, chosen_model=model1, verbose=verbose)

		fname = self.fname_inter_model_venn%(self.disease, run, model_name0, model_name1, self.suffix)
		filename = os.path.join(self.root_gemini_root, fname)

		if os.path.exists(filename) and not force:
			dff = pdreadcsv(fname, self.root_gemini_root, verbose=verbose)
			return dfpiv0, model_name0, dfpiv1, model_name1, dff, fname

		icount=-1; dic={}
		# print_plot is False 
		for case in case_list:
			# 1, 2, 3
			for i_dfp in i_dfp_list:
				for filter in ['Yes', 'No', 'Doubt']:

					perc_commons, s_agree, s_commons, all_vals, vals0, vals1, n0, n1, set0, set1, label0, label1, commons, only0, only1, \
					vals_default, common_default0, common_default1, new_pathways0, new_pathways1, title_ini = \
									 self.calc_venn_params_between_2models(run=run, filter=filter, case=case, i_dfp=i_dfp,
									 model_name0=model_name0, df0=dfpiv0, 
									 model_name1=model_name1, df1=dfpiv1, 
									 only_common_pathways=only_common_pathways, verbose=False)									 
			

					if commons is None: commons = []
					if only0 is None: only0 = []
					if only1 is None: only1 = []
					
					n_commons = len(commons)
					n_only0   = len(only0)
					n_only1   = len(only1)

					n0_consensus = n_commons+n_only0
					n1_consensus = n_commons+n_only1
					n_tot_consensus = n_commons+n_only0+n_only1

					perc_common_consensus = n_commons / n_tot_consensus if n_tot_consensus > 0 else None
					perc_only0_consensus  = n_only0   / n_tot_consensus if n_tot_consensus > 0 else None
					perc_only1_consensus  = n_only1   / n_tot_consensus if n_tot_consensus > 0 else None

					if i_dfp == 0:
						# -------------- enriched pathways ------------------------------------
						n_new_pathways0 = 0 if new_pathways0 is None else len(new_pathways0)
						n_new_pathways1 = 0 if new_pathways1 is None else len(new_pathways1)	

						#--------------- hypergeometric 0 and 1 ----------------
						if filter == 'Yes':
							if n_new_pathways0 is not None and n0 > 0 and n0_consensus > 0:
								M = n0
								N = n0_consensus
								k = n_new_pathways0
								n = N-k

								p_hyper0 = 1-hypergeom.cdf(k, M, n, N)
							else:
								p_hyper0 = None

							if n_new_pathways1 is not None and n1 > 0 and n1_consensus > 0:
								M = n1
								N = n1_consensus
								k = n_new_pathways1
								n = N-k

								p_hyper1 = 1-hypergeom.cdf(k, M, n, N)
							else:
								p_hyper1 = None

						else:
							p_hyper0, p_hyper1 = None, None

					else:
						# not enriched pathways
						n_new_pathways0, n_new_pathways1 = None, None
						p_hyper0, p_hyper1 = None, None

					icount += 1
					dic[icount] = {}
					dic2 = dic[icount]

					dic2 = dic[icount]
					dic2['run']	= run
					dic2['model0'] = model_name0
					dic2['model1'] = model_name1
					dic2['case']  = case
					dic2['consensus'] = filter
					dic2['i_dfp'] = i_dfp

					dic2['n0_pathways'] = n0
					dic2['n1_pathways'] = n1
					dic2['n0_consensus'] = n0_consensus
					dic2['n1_consensus'] = n1_consensus

					dic2['n_tot_consensus']	= n_tot_consensus
					dic2['n_common_consensus'] = n_commons
					dic2['n_only0_consensus']  = n_only0
					dic2['n_only1_consensus']  = n_only1

					dic2['perc0_consensus'] = n0_consensus / n_tot_consensus if n_tot_consensus != 0 else None
					dic2['perc1_consensus'] = n1_consensus / n_tot_consensus if n_tot_consensus != 0 else None

					dic2['perc_commons'] = perc_commons
					dic2['perc_common_consensus'] = perc_common_consensus
					dic2['perc_only0_consensus']  = perc_only0_consensus
					dic2['perc_only1_consensus']  = perc_only1_consensus

					if filter == 'Yes' and i_dfp == 0:
						dic2['n_pathw_default_tot'] = None if vals_default is None else len(vals_default)
					else:
						dic2['n_pathw_default_tot'] = None

					dic2['n_pathw0_new']	= n_new_pathways0
					dic2['p_hyper0']		= p_hyper0
					dic2['n_pathw1_new']	= n_new_pathways1
					dic2['p_hyper1']		= p_hyper1
					dic2['n_pathw_defa0_common'] = None if common_default0 is None else len(common_default0)
					dic2['n_pathw_defa1_common'] = None if common_default1 is None else len(common_default1)

					dic2['pathw_commons'] = commons
					dic2['pathw_only0']   = only0
					dic2['pathw_only1']   = only1

					dic2['vals0'] = vals0
					dic2['vals1'] = vals1

					if filter == 'Yes' and i_dfp == 0:
						dic2['pathw_default_tot'] = vals_default
					else:
						dic2['pathw_default_tot'] = None
			
					dic2['pathw_new0']		  = new_pathways0
					dic2['pathw_new1']		  = new_pathways1
					dic2['pathw_default0']	= common_default0
					dic2['pathw_default1']	= common_default1


		dff = pd.DataFrame(dic).T
		ret = pdwritecsv(dff, fname, self.root_gemini_root, verbose=verbose)

		return dfpiv0, model_name0, dfpiv1, model_name1, dff, fname


	def open_inter_model_soft_consensus_venn(self, run:str, case_list:List, chosen_model_list:List, i_dfp_list:List,
											 model0:int, model1:int, only_common_pathways:bool=False,
											 verbose:bool=False) -> (pd.DataFrame, str, pd.DataFrame, str, pd.DataFrame, str):


		dfpiv0, model_name0, dfpiv1, model_name1, dff, fname = \
					self.run_inter_model_soft_consensus_venn(run=run, case_list=case_list, i_dfp_list=i_dfp_list, 
															 model0=model0, model1=model1, 
															 only_common_pathways=only_common_pathways,
															 force=False, verbose=verbose)

		return 	dfpiv0, model_name0, dfpiv1, model_name1, dff, fname




	def get_gemini_results_by_case_model_semantics(self, run:str, case:str, chosen_model:int, i_dfp_list:List,
												   want_pubmed:bool=True, query_type:str='_strong', 
												   verbose:str=False) -> pd.DataFrame:

		self.set_run_seldata(run)
				
		ret, _, _, _ = self.bpx.open_case(case, verbose=False)
		self.set_case(case, self.bpx.df_enr, self.bpx.df_enr0)
		self.set_gemini_num_model(chosen_model)

		s_pubmed = '_with_PubMed' if want_pubmed else '_without_PubMed'
				

		cols = ['case', 'pathway_id', 'pathway', 'fdr', 'curation', 'response_explain', 'score_explain'] 
		df_list = []
		for i_dfp in i_dfp_list:
			idfp_sufix = self.idfp_sufix_list[i_dfp]

			if self.is_seldata:
				idfp_sufix = idfp_sufix[:3] + 'selected'

			fnames = [x for x in os.listdir(self.root_gemini) if x.startswith('gemini_search_for_') and 
											case in x and self.gemini_model in x and 
											s_pubmed in x and query_type in x and idfp_sufix in x ]

			if len(fnames) == 0:
				print(f"Nothing found for {case} and model {self.gemini_model} and query_type={query_type} and {idfp_sufix} in {self.root_gemini}")
				continue

			for fname in fnames:
				df = pdreadcsv(fname, self.root_gemini, verbose=verbose)

				df = df[cols]
				df['run'] = run
				df['i_dfp'] = i_dfp
				df_list.append(df)

		if df_list == []:
			print("Nothing found.")
			return None

		df = pd.concat(df_list)
		cols = ['run', 'case', 'pathway_id', 'pathway', 'i_dfp', 'fdr', 'curation', 'response_explain', 'score_explain'] 
		df = df[cols]
		df.index = np.arange(len(df))

		return df

	# run-run consensus reproducibility
	def rrcr_concat_2_runs(self, run1:str, run2:str, chosen_model_list:List,
						   force:bool=False, verbose:bool=False) -> pd.DataFrame:
		'''
			Method: the run-run consensus reproducibility (RRCR) compares each answer for 2 different runs. 
			Given 2 runs compares all consensus of run1 versus run2 for all cases and pathways in pathways groups (i_dfp)

			Inputs:
				run1: str
				run2: str
				force: bool
				verbose: bool

			Access:
				access all tables in runXX:
					root: gemini/runXX
					fname: gemini_semantic_consensus_for_<disease>_all_models_<models>_run_<run>_<suffix>.tsv'

			Output:
				table: 
					root: gemini
					fname: run_run_soft_consensus_repro_for_<disease>_between_<run1>_x_<run2>_models_<models>_<suffix>.tsv'

		'''
		fname = self.fname_soft_RRCR%(self.disease, run1, run2, chosen_model_list, self.suffix)
		filename = os.path.join(self.root_gemini_root, fname)

		if os.path.exists(filename) and not force:
			return pdreadcsv(fname, self.root_gemini_root, verbose=verbose)

		cols1	   = ['case', 'i_dfp', 'pathway_id', 'pathway', 'consensus', 'unanimous']
		common_cols = ['case', 'i_dfp', 'pathway_id', 'pathway']

		dfpiva1 = self.open_gemini_dfpiva_all_models_one_run(run=run1, chosen_model_list=chosen_model_list, verbose=verbose)
		dfpiva1 = dfpiva1[cols1]

		dfpiva2 = self.open_gemini_dfpiva_all_models_one_run(run=run2, chosen_model_list=chosen_model_list, verbose=verbose)
		dfpiva2 = dfpiva2[cols1]

		dfm = pd.merge(dfpiva1, dfpiva2, how="inner", on=common_cols)

		if dfm is None or dfm.empty:
			print(f"Error could not merge dfpiva1 {len(dfpiva1)} x dfpiva2 {len(dfpiva2)}'")
			return None

		assert len(dfm) == len(dfpiva1), f"Merge dfm must have the same length as dfpiva1: {len(dfm)} x {len(dfpiva1)}"
		assert len(dfm) == len(dfpiva2), f"Merge dfm must have the same length as dfpiva2: {len(dfm)} x {len(dfpiva2)}"
				
		all_cols = common_cols + ['consensus1', 'unanimous1', 'consensus2', 'unanimous2']
		dfm.columns = all_cols

		def compare_yes_no_doubt(cons1:str, cons2:str):
			if not isinstance(cons1, str):
				return False
			if not isinstance(cons2, str):
				return False

			# 'Yes', 'Doubt', 'No'
			if cons1 == 'Doubt': cons1 = 'Yes'
			if cons2 == 'Doubt': cons2 = 'Yes'

			return cons1 == cons2

		yes_equal, yes_diff, no_equal, no_diff = [],[],[],[]

		for i in range(len(dfm)):
			row = dfm.iloc[i]

			if row.consensus1 == 'No':
				if row.consensus2 == 'No':
					no_equal.append(1)
					no_diff.append(0)
					yes_diff.append(0)
					yes_equal.append(0)
				else:
					no_equal.append(0)
					no_diff.append(1)
					yes_diff.append(0)
					yes_equal.append(0)
			else:
				if row.consensus2 == 'No':
					no_equal.append(0)
					no_diff.append(0)
					yes_diff.append(1)
					yes_equal.append(0)
				else:
					no_equal.append(0)
					no_diff.append(0)
					yes_diff.append(0)
					yes_equal.append(1)

		dfm['answer_sim'] = [compare_yes_no_doubt(dfm.iloc[i].consensus1, dfm.iloc[i].consensus2) for i in range(len(dfm))]

		dfm['yes_equal'] = yes_equal
		dfm['yes_diff']  = yes_diff
		dfm['no_equal']  = no_equal
		dfm['no_diff']   = no_diff

		dfm['unanim_equ'] = [ (dfm.iloc[i].unanimous1 == True) and (dfm.iloc[i].unanimous2 == True) for i in range(len(dfm))]

		dfm.index = np.arange(len(dfm))

		ret = pdwritecsv(dfm, fname, self.root_gemini_root, verbose=verbose)

		return dfm



	# compare_2_runs_unanimous_mean  calc_soft_run_run_consensus_unanimous calc_soft_run_run_consensus_and_unanimous
	def rrcr_stats_2_runs(self, run1:str, run2:str, case_list:List, 
						  chosen_model_list:List, 
						  force:bool=False, verbose:bool=False) -> pd.DataFrame:

		fname = self.fname_soft_RRCR_stats%(self.disease, run1, run2, chosen_model_list, self.suffix)
		filename = os.path.join(self.root_gemini_root, fname)

		if os.path.exists(filename) and not force:
			return pdreadcsv(fname, self.root_gemini_root, verbose=verbose)

		dfrep = self.rrcr_concat_2_runs(run1=run1, run2=run2, chosen_model_list=chosen_model_list, force=force, verbose=verbose)

		dic={}; icount=-1

		for case in ['all'] + case_list:

			if case == 'all':
				df = dfrep
			else:
				df = dfrep[dfrep.case == case]
			
			n = len(df)

			n1_no = len(df[df.consensus1 == 'No'])
			n1_yes = n-n1_no

			n2_no = len(df[df.consensus2 == 'No'])
			n2_yes = n-n2_no

			mu  = np.mean(df.answer_sim)
			std = np.std(df.answer_sim)

			mu_unam  = np.mean(df.unanim_equ)
			std_unam = np.std(df.unanim_equ)

			yes_equal = df.yes_equal.sum()
			yes_diff  = df.yes_diff.sum()
			no_equal  = df.no_equal.sum()
			no_diff   = df.no_diff.sum()

			mean_repro2 = (yes_equal + no_equal) / len(df)

			vals1 = [yes_equal, yes_diff]
			vals2 = [no_equal,  no_diff]

			s_stat, stat, pvalue, dof, expected = calc_stat_chi2(vals1, vals2)

			if verbose:
				print(">>", case, vals1, vals2, s_stat)
				print(f"RRCR {100*mu:.1f}% ({100*std:.1f}%) \n")

			icount += 1
			dic[icount] = {}
			dic2 = dic[icount]
		
			dic2['case'] = case

			dic2['mean_repro'] = mu
			dic2['mean_repro2'] = mean_repro2
			dic2['std_repro'] = std

			dic2['mean_unam'] = mu_unam
			dic2['std_unam'] = std_unam

			dic2['n'] = n

			dic2['yes_equal'] = yes_equal
			dic2['yes_diff']  = yes_diff
			dic2['no_equal'] = no_equal
			dic2['no_diff']  = no_diff

			dic2['pvalue'] = pvalue
			dic2['stat'] = stat
			dic2['dof'] = dof
			dic2['expected'] = expected
			dic2['s_stat'] = s_stat

			dic2['run1_yes'] = n1_yes
			dic2['run1_no']  = n1_no
			dic2['run2_yes'] = n2_yes
			dic2['run2_no']  = n2_no



		df = pd.DataFrame(dic).T
		df['fdr'] = fdr(df['pvalue'])

		cols = ['case', 'mean_repro','mean_repro2', 'std_repro', 
				'mean_unam', 'std_unam', 'n', 'fdr', 'pvalue',
				'yes_equal', 'yes_diff', 'no_equal', 'no_diff', 
				'run1_yes',  'run1_no',  'run2_yes', 'run2_no', 
				'stat', 'dof', 'expected', 's_stat']

		df = df[cols]

		ret = pdwritecsv(df, fname, self.root_gemini_root, verbose=verbose)

		return df


	def summary_hard_repro(self, one_or_two:int, run1:str, run2:str, 
						   chosen_model_list:List, case_list:List, 
						   force:bool=False, verbose:bool=False) -> pd.DataFrame:

		run = run1 if one_or_two == 1 else run2

		fname = self.fname_hard_inter_model_summary%(self.disease, chosen_model_list, run, self.suffix)
		filename = os.path.join(self.root_gemini_root, fname)

		if os.path.exists(filename) and not force:
			return pdreadcsv(fname, self.root_gemini_root, verbose=verbose)

		text_all, dfall_all2, dfcase, mu_all, std_all, n_all, nsim_all, nnot_all = \
		self.calc_run_run_hard_repro(run1, run2, chosen_model_list=chosen_model_list, case_list=case_list, verbose=verbose)

		text_mod, df_mod, dfcase_mod, mu_mod, std_mod, n_mod, nsim_mod, nnot_mod = \
		self.calc_inter_model_hard_repro_one_or_two(one_or_two=one_or_two, run1=run1, run2=run2, chosen_model_list=chosen_model_list,
											  case_list=case_list, force=force, verbose=verbose)

		dic={}; icount=-1

		for i in range(2):

			icount+=1
			dic[icount] = {}
			dic2 = dic[icount]

			if i == 0:
				dic2['hard_repro'] = 'run-run'
			
				dic2['mu'] = mu_all
				dic2['std'] = std_all
				dic2['n'] = n_all
				dic2['n_consensus'] = nsim_all
				dic2['n_different'] = nnot_all
				dic2['text'] = text_all
			else:
				dic2['hard_repro'] = 'inter-model'
			
				dic2['mu'] = mu_mod
				dic2['std'] = std_mod
				dic2['n'] = n_mod
				dic2['n_consensus'] = nsim_mod
				dic2['n_different'] = nnot_mod
				dic2['text'] = text_mod

		dfhard = pd.DataFrame(dic).T
		
		ret = pdwritecsv(dfhard, fname, self.root_gemini_root, verbose=verbose)

		return dfhard

	# summary_rrr_concat_2_runs
	def calc_run_run_hard_repro(self, run1:str, run2:str, chosen_model_list:List, case_list:List, force:bool=False,
								verbose:bool=False) -> (str, pd.DataFrame, pd.DataFrame, float, float, int, int, int):

		# open all answers (all models) for run1 and run2
		dfall = self.rrr_concat_2_runs(run1, run2, verbose=verbose)

		# select all chosen_model_list from rrr
		models = [self.gemini_models[imodel].replace("gemini-","") for imodel in chosen_model_list]

		dfall2 = dfall.copy()
		# if there is _selected (selected pathways) remove it
		dfall2['model_name'] = [x.replace('_selected','') for x in dfall2['model_name']]
		dfall2 = dfall2[dfall2.model_name.isin(models)].copy()
		dfall2.index = np.arange(len(dfall2))

		mu_all  = dfall2.answer_sim.mean()
		std_all = dfall2.answer_sim.std()
		n = len(dfall2)

		dfsim = dfall2[dfall2.answer_sim >= self.answer_min_cutoff]
		nsim_all = len(dfsim)

		dfnot = dfall2[dfall2.answer_sim < self.answer_min_cutoff]
		nnot_all = len(dfnot)

		text = f"run-run reproducibility has {n} questions, mean {100*mu_all:.1f}% ({100*std_all:.1f}%); {nsim_all} equals and {nnot_all} diffs."

		fname = self.fname_hard_run_run_case%(self.disease, run1, run2, self.suffix)
		filename = os.path.join(self.root_gemini_root, fname)

		if os.path.exists(filename) and not force:
			dfcase = pdreadcsv(fname, self.root_gemini_root, verbose=verbose)
			return text, dfall2, dfcase, mu_all, std_all, n, nsim_all, nnot_all

		dic={}; icount=-1

		for case in case_list:
			df2 = dfall2[dfall2.case == case]

			mu  = df2.answer_sim.mean()
			std = df2.answer_sim.std()
			
			dfsim = df2[df2.answer_sim >= self.answer_min_cutoff]
			nsim = len(dfsim)
			
			dfnot = df2[df2.answer_sim < self.answer_min_cutoff]
			nnot = len(dfnot)

			icount+=1
			dic[icount] = {}
			dic2 = dic[icount]

			dic2['run1'] = run1
			dic2['run2'] = run2
			dic2['case'] = case

			dic2['mu'] = mu
			dic2['std'] = std
			dic2['n'] = len(df2)
			dic2['n_consensus'] = nsim
			dic2['n_different'] = nnot

		dfcase = pd.DataFrame(dic).T
		ret = pdwritecsv(dfcase, fname, self.root_gemini_root, verbose=verbose)

		return text, dfall2, dfcase, mu_all, std_all, n, nsim_all, nnot_all


	# summary_inter_models_2runs --> calc_inter_model_run1
	def calc_inter_model_hard_repro_one_or_two(self, one_or_two:int, run1:str, run2:str, 
											   chosen_model_list:List, case_list:List, force:bool=False, 
											   verbose:bool=False) -> (str, pd.DataFrame, pd.DataFrame, float, float, int, int, int):
		'''
			IMR - inter-model reproducibility
			given both runs: run1 and run2 - open rrr_concat_2_runs()
			one_or_two is 1 or 2, the first or second curation column
			compare to models in chosen_model_list
		'''
		if not isinstance(chosen_model_list, list):
			print("chosen_model_list must be a list with 2 integers (models)")
			return [None]*8

		if len(chosen_model_list) != 2:
			print("chosen_model_list must be a list with 2 integers (models)")
			return [None]*8

		run = run1 if one_or_two == 1 else run2

		fname = self.fname_hard_inter_model_repro%(self.disease, chosen_model_list, run, self.suffix)
		filename = os.path.join(self.root_gemini_root, fname)

		if os.path.exists(filename) and not force:
			df_imr = pdreadcsv(fname, self.root_gemini_root, verbose=verbose)
		else:
			dfall = self.rrr_concat_2_runs(run1, run2, verbose=verbose)

			models = [self.gemini_models[imodel].replace("gemini-","") for imodel in chosen_model_list]

			dfall2 = dfall.copy()
			dfall2['model_name'] = [x.replace('_selected','') for x in dfall2['model_name']]
			dfall2 = dfall2[dfall2.model_name.isin(models)].copy()
			dfall2.index = np.arange(len(dfall2))

			if one_or_two == 1:
				col = 'curation1'
			else:
				col = 'curation2'

			cols = ['case', 'iq', 'i_dfp', 'model_name', 'pathway_id'] + [col]
			df3 = dfall2[cols].copy()

			cols = ['case', 'iq', 'i_dfp', 'model_name', 'pathway_id', 'curation']
			df3.columns = cols

			cols = ['case', 'iq', 'i_dfp', 'pathway_id']

			df_imr0 = df3[df3.model_name == models[0]].copy()
			df_imr0 = df_imr0.sort_values(cols)
			df_imr0.index = np.arange(len(df_imr0))
								   
			df_imr1 = df3[df3.model_name == models[1]].copy()
			df_imr1 = df_imr1.sort_values(cols)
			df_imr1.index = np.arange(len(df_imr1))

			tupple_all = (len(df_imr0), len(df_imr1))

			cols = ['case', 'iq', 'i_dfp', 'pathway_id', 'model_name']

			df_imr0a = df_imr0.drop_duplicates(cols)
			df_imr1a = df_imr1.drop_duplicates(cols)

			tupple_drop = (len(df_imr0a), len(df_imr1a))

			assert tupple_all == tupple_drop, f"Duplicated pathways? {tupple_all} x {tupple_drop}?"

			lista_in  = [x for x in df_imr0.pathway_id if x		in df_imr1.pathway_id.to_list()]
			lista_out = [x for x in df_imr0.pathway_id if x not in df_imr1.pathway_id.to_list()]

			assert len(lista_in) == len(df_imr0), f"Has df_imr0 all pathways compared to df33? {len(lista_in)} x {len(df_imr0)}"
			assert len(lista_out) == 0, f"Has df_imr0 extra pathways compared to df33? {lista_out}"

			cols		= ['case', 'iq', 'i_dfp', 'pathway_id', 'curation']
			common_cols = ['case', 'iq', 'i_dfp', 'pathway_id']
			all_cols	= ['case', 'iq', 'i_dfp', 'pathway_id', 'curation1', 'curation2']

			df_imr = pd.merge(df_imr0[cols], df_imr1[cols], how="inner", on=common_cols)
			df_imr.columns = all_cols
			df_imr['answer_sim']  = [self.calc_answer_equal(df_imr.iloc[i].curation1, df_imr.iloc[i].curation2) for i in range(len(df_imr))]

			ret = pdwritecsv(df_imr, fname, self.root_gemini_root, verbose=verbose)
	   
		mu_imr  = df_imr.answer_sim.mean()
		std_imr = df_imr.answer_sim.std()
		n = len(df_imr)

		dfsim  = df_imr[df_imr.answer_sim >= self.answer_min_cutoff]
		nsim_imr = len(dfsim)

		dfnot = df_imr[df_imr.answer_sim < self.answer_min_cutoff]
		nnot_imr = len(dfnot)

		text = f"IMR between models {chosen_model_list} has {n} questions, mean {100*mu_imr:.1f}% ({100*std_imr:.1f}%); {nsim_imr} equals and {nnot_imr} diffs for {run}"

		fname_case = self.fname_hard_inter_model_case_repro%(self.disease, chosen_model_list, run, self.suffix)
		filename_case = os.path.join(self.root_gemini_root, fname_case)

		if os.path.exists(filename_case) and not force:
			dfcase_imr = pdreadcsv(fname_case, self.root_gemini_root, verbose=verbose)
		else:
			dic={}; icount=-1

			for case in case_list:
				df2 = df_imr[df_imr.case == case]

				mu  = df2.answer_sim.mean()
				std = df2.answer_sim.std()
				
				dfsim = df2[df2.answer_sim >= self.answer_min_cutoff]
				nsim = len(dfsim)
				
				dfnot = df2[df2.answer_sim < self.answer_min_cutoff]
				nnot = len(dfnot)

				icount+=1
				dic[icount] = {}
				dic2 = dic[icount]

				dic2['run'] = run
				dic2['case'] = case
				dic2['models'] = chosen_model_list

				dic2['mu'] = mu
				dic2['std'] = std
				dic2['n'] = len(df2)
				dic2['n_consensus'] = nsim
				dic2['n_different'] = nnot

			dfcase_imr = pd.DataFrame(dic).T
			ret = pdwritecsv(dfcase_imr, fname_case, self.root_gemini_root, verbose=verbose)

		return text, df_imr, dfcase_imr, mu_imr, std_imr, n, nsim_imr, nnot_imr

	def calc_answer_equal(self, answer1:str, answer2:str) -> float:
		if not isinstance(answer1, str):
			return 0
		if not isinstance(answer2, str):
			return 0

		answer1 = answer1.lower()[:3]
		answer2 = answer2.lower()[:3]

		if answer1 == answer2:
			return 1.

		try:
			diff = self.answer_weight[answer1] - self.answer_weight[answer2]
			if diff < 0: diff = -diff
			diff = 1 - diff
		except:
			return 0

		return diff


	def rrr_concat_2_runs(self, run1:str, run2:str, force:bool=False, verbose:bool=False) -> pd.DataFrame:
		'''
			Method: the run-run reproducibility (RRR) compares each answer for 2 different runs. Given 2 runs compares all answers run1 versus run2 for all cases, models, 4DSSQ, and pathways in pathways groups (i_dfp)

			Inputs:
				run1: str
				run2: str
				force: bool
				verbose: bool

			Access:
				access all tables in runXX:
					root: gemini/runXX
					fname: gemini_search_for_<disese>_<case>_<iq>_<idfp>_gemini-<model>.tsv

			Output:
				table: 
					root: gemini
					fname: all_merged_gemini_questions_for_<disease>_run_<run1>_x_<run2>_all_data.tsv

		'''

		self.set_run_seldata(run1)
		root_gemini1 = self.root_gemini

		fname = self.fname_all_merged_gemini_questions%(self.disease, run1, run2, self.suffix)
		filename = os.path.join(self.root_gemini_root, fname)

		if os.path.exists(filename) and not force:
			return pdreadcsv(fname, self.root_gemini_root, verbose=verbose)

		self.set_run_seldata(run2)
		root_gemini2 = self.root_gemini

		files1 = [x for x in os.listdir(root_gemini1) if x.startswith('gemini_search_for_') and not '_3_others_' in x]

		if files1 == []:
			print(f"There are no files starting with 'gemini_search_for_*.tsv' in {root_gemini1}")
			return None

		cols1 = ['case', 'pathway_id', 'question', 'curation']
		cols2 = ['case', 'pathway_id', 'curation']

		common_cols = ['case', 'pathway_id']

		all_cols = ['case', 'pathway_id', 'question', 'curation1', 'curation2']

		# has_error to run twice until no errors: in development some wrong files still were there.
		has_error=False; i=-1; df_list=[]

		for fname_gem in files1:
			i+=1
			filename1 = os.path.join(root_gemini1, fname_gem)
			# print(i, fname_gem)

			df1 = pdreadcsv(fname_gem, root_gemini1)
			if df1 is None or df1.empty:
				print(f"Error1 line{i}: There is no file: {filename1}")
				continue

			filename2 = os.path.join(root_gemini2, fname_gem)

			if not os.path.exists(filename2):
				print(f"Error2 line{i}: There is no file: {filename2}")
				continue

			df2 = pdreadcsv(fname_gem, root_gemini2)

			dfn = pd.merge(df1[cols1], df2[cols2], how="inner", on=common_cols)

			if dfn is None or dfn.empty:
				print(f"Error3 line {i}: There is no merge dfn {len(df1)} x {len(df2)} file '{fname_gem}'")
				has_error = True
				continue
				
			model_name = fname_gem.split('_model_gemini-')[1]
			model_name = model_name.split('.tsv')[0]

			if '_with_PubMed'  in fname_gem:
				iq = 3 if 'strong_relationship' in fname_gem else 1
			else:
				iq = 2 if 'strong_relationship' in fname_gem else 0

			if   '_pathways_0_first_'  in fname_gem or '_pathways_0_selected_' in fname_gem:
				i_dfp = 0
			elif '_pathways_1_middle_' in fname_gem:
				i_dfp = 1
			elif '_pathways_2_final_'  in fname_gem:
				i_dfp = 2
			else:
				continue

			dfn.columns = all_cols
			dfn['curation1'] = [self.fix_answer(x) for x in dfn['curation1']]
			dfn['curation2'] = [self.fix_answer(x) for x in dfn['curation2']]
			dfn['model_name'] = model_name
			dfn['iq'] = iq
			dfn['i_dfp'] = i_dfp
			dfn['answer_sim'] = [self.calc_answer_equal(dfn.iloc[i].curation1, dfn.iloc[i].curation2) for i in range(len(dfn))]

			df_list.append(dfn)


		if not has_error:
			if df_list == []:
				print("Could not merge files.")
				dfall = pd.DataFrame()
			else:
				dfall = pd.concat(df_list)
				dfall.index = np.arange(len(dfall))

				cols = ['case', 'iq', 'i_dfp', 'model_name', 'pathway_id', 'question', 'curation1', 'curation2', 'answer_sim']
				dfall = dfall[cols]

				cols_sort =  ['model_name', 'case', 'i_dfp', 'pathway_id', 'iq']
				dfall = dfall.sort_values(cols_sort)
				ret = pdwritecsv(dfall, fname, self.root_gemini_root, verbose=verbose)
		else:
			print("Has errors")
			dfall = pd.DataFrame()

		return dfall


	# compare_2_runs_total_answers --> hard
	def compare_hard_2_runs_total_answers(self, run1:str, run2:str, case_list:List, 
										  chosen_model_list:List, pval_cutoff:float=0.05,
										  force:bool=False, verbose:bool=False) -> (pd.DataFrame, pd.DataFrame):

		fname_tot	= self.fname_hard_run_run_compare_total_answers%(self.disease, run1, run2, self.suffix)
		filename_tot = os.path.join(self.root_gemini_root, fname_tot)

		fname_stat	= self.fname_hard_run_run_stats_total_answers%(self.disease, run1, run2, self.suffix)
		filename_stat = os.path.join(self.root_gemini_root, fname_stat)

		if os.path.exists(filename_tot) and os.path.exists(filename_stat) and not force:
			dftot  = pdreadcsv(fname_tot, self.root_gemini_root, verbose=verbose)
			dfstat = pdreadcsv(fname_stat, self.root_gemini_root, verbose=verbose)

			return dftot, dfstat


		cols1 = ['case', 'i_dfp', 'pathway_id', 'pathway',]

		for chosen_model in chosen_model_list:
			cols1 += [f'simp{chosen_model}', f'simpub{chosen_model}', f'dis{chosen_model}', f'dispub{chosen_model}']

		cols1 += [ 'run', 'consensus', 'n_yes', 'n_no','unanimous']

		dfpiv1 = self.open_gemini_dfpiva_all_models_one_run(run=run1, chosen_model_list=chosen_model_list, verbose=verbose)
		dfpiv1.columns = cols1

		dfpiv2 = self.open_gemini_dfpiva_all_models_one_run(run=run2, chosen_model_list=chosen_model_list, verbose=verbose)
		dfpiv2.columns = cols1


		dic={}; icount=-1
		
		for case in case_list:
			df2 = dfpiv1[dfpiv1.case == case]

			tot_yes, tot_pos, tot_low, tot_no = 0,0,0,0
			for s_semantic in ['simp', 'simpub', 'dis', 'dispub']:
				yes = [df2[f'{s_semantic}{m}'].str.startswith('Yes').sum() for m in self.chosen_model_list]
				pos = [df2[f'{s_semantic}{m}'].str.startswith('Pos').sum() for m in self.chosen_model_list]
				low = [df2[f'{s_semantic}{m}'].str.startswith('Low').sum() for m in self.chosen_model_list]
				no  = [df2[f'{s_semantic}{m}'].str.startswith('No').sum()  for m in self.chosen_model_list]
			   
				tot_yes += np.sum(yes)
				tot_pos += np.sum(pos)
				tot_low += np.sum(low)
				tot_no  += np.sum(no)


			icount+=1
			dic[icount] = {}
			dic2 = dic[icount]

			dic2['case'] = case

			dic2['tot_yes1'] = tot_yes
			dic2['tot_pos1'] = tot_pos
			dic2['tot_low1'] = tot_low
			dic2['tot_no1']  = tot_no

			df2 = dfpiv2[dfpiv2.case == case]

			tot_yes, tot_pos, tot_low, tot_no = 0,0,0,0
			for s_semantic in ['simp', 'simpub', 'dis', 'dispub']:
				yes = [df2[f'{s_semantic}{m}'].str.startswith('Yes').sum() for m in self.chosen_model_list]
				pos = [df2[f'{s_semantic}{m}'].str.startswith('Pos').sum() for m in self.chosen_model_list]
				low = [df2[f'{s_semantic}{m}'].str.startswith('Low').sum() for m in self.chosen_model_list]
				no  = [df2[f'{s_semantic}{m}'].str.startswith('No').sum()  for m in self.chosen_model_list]
			   
				tot_yes += np.sum(yes)
				tot_pos += np.sum(pos)
				tot_low += np.sum(low)
				tot_no  += np.sum(no)

			dic2['tot_yes2'] = tot_yes
			dic2['tot_pos2'] = tot_pos
			dic2['tot_low2'] = tot_low
			dic2['tot_no2']  = tot_no
			
		dftot = pd.DataFrame(dic).T

		ret = pdwritecsv(dftot, fname_tot, self.root_gemini_root, verbose=verbose)


		''' -------------- statistics ----------------'''
		pval_cutoff_bonf = pval_cutoff/len(case_list)


		print(f"Comparing {run1}x{run2} - with Bonferroni pval cutoff {pval_cutoff_bonf:.2e}\n")

		dic={}; icount=-1
		for i in range(len(dftot)):
			case = dftot.iloc[i].case
			row = dftot.iloc[i,:]

			icount+=1
			dic[icount] = {}
			dic2 = dic[icount]

			dic2['run1'] = run1
			dic2['run2'] = run2
			dic2['case'] = case

			dic2['pval_cutoff'] = pval_cutoff
			dic2['pval_cutoff_bonf'] = pval_cutoff_bonf

			#--------- between models -----------------------

			vals1 = row[ ['tot_yes1', 'tot_pos1', 'tot_low1', 'tot_no1'] ]
			vals1 = np.array(vals1)+1
			
			vals2 = row[ ['tot_yes2', 'tot_pos2', 'tot_low2', 'tot_no2'] ]
			vals2 = np.array(vals2)+1
			
			s_stat, stat, pvalue, dof, expected = calc_stat_chi2(vals1, vals2)

			if pvalue < pval_cutoff:
				s_pvalue = 'statistically distinct'
			else:
				s_pvalue = 'statistically similar'
			
			print(f"\t\t{case:15}, pval={pvalue:.2e} - {s_pvalue:25} values: {vals1} x {vals2}")

			dic2['s_pvalue'] = s_pvalue

			dic2['s_stat'] = s_stat
			dic2['stat'] = stat
			dic2['pvalue'] = pvalue
			dic2['dof'] = dof
			dic2['expected'] = expected

			dic2['vals1'] = vals1
			dic2['vals2'] = vals2

		print("To perform the chi-square statistics, we add 1 for each cell because zeros are not allowed.")

		dfstat = pd.DataFrame(dic).T

		ret = pdwritecsv(dfstat, fname_stat, self.root_gemini_root, verbose=verbose)

		return dftot, dfstat


	def open_dfp(self, run:str, i_dfp:int, case:str, gemini_model:str, question_name:str, verbose:bool=False) -> pd.DataFrame:

		self.set_run_seldata(run)

		fname = self.fname_gemini_search%(self.disease, case, question_name, gemini_model)
		fname = title_replace(fname)
		# print(i_dfp, self.root_gemini, fname)
		filename = os.path.join(self.root_gemini, fname)

		if os.path.exists(filename):
			return pdreadcsv(fname, self.root_gemini, verbose=verbose)
		else:
			print(f"File not found: '{filename}'")
			return pd.DataFrame()

	def run_again_dfp(self, run:str, chosen_model_list:List, i_dfp_list:List, case_list:List, 
					  verbose:bool=False):

		self.set_run_seldata(run)

		cols_default = ['pathway_id', 'pathway', 'fdr']

		for chosen_model in chosen_model_list:
			self.set_gemini_num_model(chosen_model)
			print(">>>", chosen_model, self.gemini_model)

			for case in case_list:
				print("\t", case)

				#------- BCA - best cutoff algorithm ----------------------
				ret, _, _, _ = self.bpx.open_case(case)
				dfp0_default = self.bpx.df_enr[cols_default]
				self.cur_pathway_id_list = list(dfp0_default.pathway_id)
				
				df_enr0 = self.bpx.df_enr0
				
				n = len(dfp0_default)
				N = len(df_enr0)

				self.set_case(self.bpx.case, self.bpx.df_enr, self.bpx.df_enr0)
				
				#-- calc the middle
				n2 = int(n/2)
				N2 = int(N/2)
				
				ini = N2-n2
				end = ini+n

				if ini <= n:
					ini = n+1
					end = ini + n

				end_middle = end
				
				dfp1_default = df_enr0.iloc[ini:end].copy()
				dfp1_default = dfp1_default[~dfp1_default.pathway_id.isin(self.cur_pathway_id_list)]
				dfp1_default = dfp1_default[cols_default]
				dfp1_default.index = np.arange(len(dfp1_default))
				self.cur_pathway_id_list += list(dfp1_default.pathway_id)

				# calc the end
				ini = N-n
				end = N

				if ini <= end_middle:
					ini = end_middle + 1
					
				dfp2_default = df_enr0.iloc[ini:end].copy()
				dfp2_default = dfp2_default[~dfp2_default.pathway_id.isin(self.cur_pathway_id_list)]
				dfp2_default.index = np.arange(len(dfp2_default))
				self.cur_pathway_id_list += list(dfp2_default.pathway_id)

				'''------------------ pathways not in reactome ------------'''
				self.dfr_not_in_reactome = pdreadcsv(self.fname_pathways_not_in_reactome, self.root_gemini_root, verbose=verbose)
				dfp3_default = self.pick_other_pahtways()
				
				n0 = len(dfp0_default)
				
				for quest_type in self.question_list:
					#print("\t\t", quest_type)
				
					quest_ptw_dis_case, with_without_PubMed, suffix2 = self.define_question(quest_type)
				
					# question_name0 = f'{with_without_PubMed}_{suffix2}_0_default'
					question_name1 = f'{with_without_PubMed}_{suffix2}_0_first'
					question_name2 = f'{with_without_PubMed}_{suffix2}_1_middle' 
					question_name3 = f'{with_without_PubMed}_{suffix2}_2_final'
					question_name4 = f'{with_without_PubMed}_{suffix2}_3_others'
				
					multiple_data  = [ [0, question_name1], [1, question_name2], 
									   [2, question_name3], [3, question_name4]]
				
					for i_dfp, question_name in multiple_data:
						#print("\t\t\t", question_name)
				
						dfp = self.open_dfp(run=run, i_dfp=i_dfp, case=case, gemini_model=self.gemini_model, question_name=question_name, verbose=verbose)
				
						if i_dfp == 0:
							dfa = self.check_dfp_list(dfp, dfp0_default, n0, run, i_dfp, case, self.gemini_model, question_name)
							
						elif i_dfp == 1:
							dfa = self.check_dfp_list(dfp, dfp1_default, n0, run, i_dfp, case, self.gemini_model, question_name)
			
						elif i_dfp == 2:
							dfa = self.check_dfp_list(dfp, dfp2_default, n0, run, i_dfp, case, self.gemini_model, question_name)
						   
						else:
							dfa = self.check_dfp_list(dfp, dfp3_default, n0, run, i_dfp, case, self.gemini_model, question_name)

						print(f"> {run} {case} {i_dfp} {question_name} {len(dfp)} x {len(dfa)}")
							
					print("\n")
				print("\n")
			print("\n")


	def save_again_dfp(self, dfp:pd.DataFrame, run:str, i_dfp:int, case:str, gemini_model:str, 
					   question_name:str, verbose:bool=False) -> pd.DataFrame:

		self.set_run_seldata(run)

		fname = self.fname_gemini_search%(self.disease, case, question_name, gemini_model)
		fname = title_replace(fname)

		ret = pdwritecsv(dfp, fname, self.root_gemini, verbose=verbose)
		return ret


	def check_dfp_list(self, dfp:pd.DataFrame, dfp1_ori:pd.DataFrame, n0:int, run:str, 
					   i_dfp:int, case:str, gemini_model:str, question_name:str):

		dfp1 = dfp.copy()
		dfp1 = dfp1[dfp1.pathway_id.isin(dfp1_ori.pathway_id) ].copy()
		dfp1 = dfp1.drop_duplicates('pathway_id')

		if len(dfp1) == len(dfp1_ori) and len(dfp1) == len(dfp):
			# nothing changed
			return dfp

		if len(dfp1) < len(dfp1_ori):
			print(f"Less then: {len(dfp1)} < {len(dfp1_ori)} - and n0 = {n0} ", run, i_dfp, case, gemini_model, question_name)

			dfpa = dfp1_ori[ ~dfp1_ori.pathway_id.isin(dfp1.pathway_id) ].copy()
			if dfpa.empty:
				print("dfpa.empty --> no records found!!!")
				return dfp1

			print("\t\tfound", len(dfpa))
			cols = ['curation', 'response_explain', 'score_explain', 'question', 'disease', 'case', 's_case']
			dfpa.loc[:, cols] = [None]*len(cols)
			dfpa.pathway_found = False

			# print("\n--------------------")
			# print(dfpa)
			# print("----------------------\n\n")
			
			dfp1 = pd.concat([dfp1, dfpa])
			dfp1 = dfp1.sort_values('fdr')
			dfp1.index = np.arange(len(dfp1))

			print(f"Add records: {len(dfp1)} ? dfp1_ori={len(dfp1_ori)} n0={n0}")

		ret = self.save_again_dfp(dfp1, run, i_dfp, case, gemini_model, question_name, verbose=False)
		if not ret:
			raise Exception('stop: saving')

		return dfp1

	#--- group_discovery_fp_fn_enriched_bca
	def confusion_table_fp_fn_enriched_bca(self, run:str, case:str, chosen_model_list:List, 
										   force:bool=False, verbose:bool=False) -> (pd.DataFrame, List):
		'''
			Compares the default pathways x new pathways

			input: run, case, chosen_model_list, force, verbose
			output: df confusion table, confusion label matrix = conf_list 
		'''

		fname = self.fname_confusion_table_BCA%(self.disease, run, case)
		filename = os.path.join(self.root_curation, fname)

		if os.path.exists(filename) and not force:
			df = pdreadcsv(fname, self.root_curation, verbose=verbose)
			conf_list = self.build_confusion_list_enriched_bca(df)
			return df, conf_list

		#-------------- default cutoff ----------------
		ret, _, _, _ = self.bpx.open_case_params(case, abs_lfc_cutoff=1, fdr_lfc_cutoff=0.05, 
												 pathway_fdr_cutoff=0.05, verbose=verbose)
		df_enr_defa = self.bpx.df_enr
		if df_enr_defa is None:
			df_enr_defa = pd.DataFrame()
			
		#-------------- BCA cutoff ----------------
		ret, _, _, _ = self.bpx.open_case(case, verbose=verbose)
		df_enr_bca = self.bpx.df_enr
		if df_enr_bca is None:
			df_enr_bca = pd.DataFrame()

		n_defa = len(df_enr_defa)
		n_bca = len(df_enr_bca)

		#-------------- read consensus ----------------
		dfpiva = self.open_gemini_dfpiva_all_models_one_run(run=run, chosen_model_list=chosen_model_list, 
															verbose=verbose)
		dfpiv = dfpiva[(dfpiva.case == case) & (dfpiva.i_dfp == 0)]
		n = len(dfpiv)

		print(f"For case {case} one found {n_defa} default pathways and {n_bca} BCA's pathways and {n} dfpiv.")

		if df_enr_defa is None or df_enr_defa.empty:
			defa_pathways = []
		else:
			defa_pathways = df_enr_defa.pathway_id.to_list()

		new_pathways = [x for x in df_enr_bca.pathway_id if x not in defa_pathways]


		cols = ['run', 'case', 'i_dfp', 'pathway_id', 'pathway',  'consensus', 'n_yes', 'n_no', 'unanimous']

		df_tp = dfpiv[ (dfpiv.pathway_id.isin(defa_pathways)) & ( (dfpiv.consensus == 'Yes') | (dfpiv.consensus =='Doubt') )][cols]
		df_fp = dfpiv[ (dfpiv.pathway_id.isin(defa_pathways)) & (  dfpiv.consensus == 'No' )][cols]
		df_tn = dfpiv[ (dfpiv.pathway_id.isin(new_pathways))  & (  dfpiv.consensus == 'No' )][cols]
		df_fn = dfpiv[ (dfpiv.pathway_id.isin(new_pathways))  & ( (dfpiv.consensus == 'Yes') | (dfpiv.consensus =='Doubt') )][cols]

		df_tp['group'] = 'TP'
		df_fp['group'] = 'FP'
		df_tn['group'] = 'TN'
		df_fn['group'] = 'FN'

		df = pd.concat([df_tp, df_fp, df_tn, df_fn])
		df.index = np.arange(len(df))

		cols = ['group', 'run', 'case', 'i_dfp', 'pathway_id', 'pathway',  'consensus', 'n_yes', 'n_no', 'unanimous']

		df = df[cols]
		ret = pdwritecsv(df, fname, self.root_curation, verbose=verbose)

		conf_list = self.build_confusion_list_enriched_bca(df)

		return df, conf_list

	def build_confusion_list_enriched_bca(self, df:pd.DataFrame) -> List:

		tp = len(df[ df.group == 'TP'])
		fp = len(df[ df.group == 'FP'])
		tn = len(df[ df.group == 'TN'])
		fn = len(df[ df.group == 'FN'])

		return [tp, fp, tn, fn]

	# calc_all_significance_per_run_case_enriched_bca
	def calc_confusion_stats_enriched_bca_per_run_case(self, run:str, case_list:List, chosen_model_list:List, 
							   						   param_perc:float=0.90, prompt:bool=False,
							   						   force:bool=False, verbose:bool=False):

		fname = self.fname_confusion_table_all_BCA%(self.disease)
		filename = os.path.join(self.root_curation, fname)

		if os.path.exists(filename) and not force:
			dfa = pdreadcsv(fname, self.root_curation, verbose=verbose)

			return dfa

		dic={}; icount=-1
		for case in case_list:
			df, s_stat, stat, pvalue, dof, expected, list_obs, list_exp, Npos, Nneg = \
			self.calc_significance_per_run_case_enriched_bca(run=run, case=case, chosen_model_list=chosen_model_list, 
															 param_perc=param_perc, force=force, verbose=verbose)

			if prompt:
				expected = list([ list(np.round(x,1)) for x in expected])
				print(f"\t{case:15} stat {str(np.round(stat,2)):7} pval {pvalue:.2e} - {str(list_obs):8} x {str(list_exp):8}; exp: {expected}")

			TP = list_obs[0]
			FP = list_obs[1]
			TN = list_obs[2]
			FN = list_obs[3]

			Npos = TP+FP
			Nneg = TN+FN
			N = Npos + Nneg

			sens = TP / (TP + FN)
			spec = TN / (TN + FP)
			accu = (TP + TN) / N
			try:
				prec = TP / (TP + FP)
			except:
				prec = None

			if prec is None:
				f1sc = None
			else:
				f1sc = (2*prec*sens) / (prec+sens)

			icount += 1
			dic[icount] = {}
			dic2 = dic[icount]

			dic2['run']  = run
			dic2['case'] = case

			dic2['TP'] = TP
			dic2['FP'] = FP
			dic2['TN'] = TN
			dic2['FN'] = FN

			dic2['TP_exp'] = list_exp[0]
			dic2['FP_exp'] = list_exp[1]
			dic2['TN_exp'] = list_exp[2]
			dic2['FN_exp'] = list_exp[3]

			dic2['n'] = N
			dic2['n_pos'] = Npos
			dic2['n_neg'] = Nneg

			dic2['TP'] = TP
			dic2['FP'] = FP
			dic2['TN'] = TN
			dic2['FN'] = FN

			dic2['sens'] = sens
			dic2['spec'] = spec
			dic2['accu'] = accu
			dic2['prec'] = prec
			dic2['f1_score'] = f1sc

			dic2['param_perc'] = param_perc

			dic2['stat'] = stat
			dic2['pvalue'] = pvalue
			dic2['dof'] = dof
			dic2['expected'] = np.round(expected, 1)

		dfa = pd.DataFrame(dic).T
		dfa['fdr'] = fdr(dfa.pvalue)

		ret = pdwritecsv(dfa, fname, self.root_curation, verbose=verbose)

		return dfa


	def calc_significance_per_run_case_enriched_bca(self, run:str, case:str, chosen_model_list:List, 
							   						param_perc:float=0.90, force:bool=False, verbose:bool=False):

		df, conf_list = self.confusion_table_fp_fn_enriched_bca(run=run, case=case, chosen_model_list=chosen_model_list, force=force, verbose=verbose)
		
		tp, fp, tn, fn = conf_list

		Npos = tp + fp
		nExpTP = int(np.round(param_perc*Npos, 0))
		nExpFP = Npos - nExpTP

		Nneg = tn + fn
		nExpTN = int(np.round(param_perc*Nneg, 0))
		nExpFN = Nneg - nExpTN

		list_obs = conf_list
		list_exp = [nExpTP, nExpFP, nExpTN, nExpFN]
		
		s_stat, stat, pvalue, dof, expected = calc_stat_chi2(list_obs, list_exp)

		return df, s_stat, stat, pvalue, dof, expected, list_obs, list_exp, Npos, Nneg

	def fix_answer(self, term:str) -> str:

		if not isinstance(term, str):
			# print(f"Error: termm is not a string '{term}'")
			return None

		answer = term.lower().strip()

		if answer.startswith('yes'):
			return 'Yes'
		if answer.startswith('possible'):
			return 'Possible'
		if answer.startswith('low'):
			return 'Low'
		if answer.startswith('no'):
			return 'No'

		print(f"Warning: wrong answer {term}")
		return term

