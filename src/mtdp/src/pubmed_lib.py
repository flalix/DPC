#!/usr/bin/python
#!python
# -*- coding: utf-8 -*-
# Created on 2024/01/23
# Udated  on 2025/02/21; 2025/01/02; 2024/12/11
# @author: Flavio Lichtenstein
# @local: Bioinformatics: CENTD/Molecular Biology; Instituto Butatan

from Bio import Entrez
from Bio import Medline
import numpy as np
import pandas as pd
import os, sys, time, shutil
from typing import Optional, Iterable, Set, Tuple, Any, List

from copy import deepcopy

import spacy, re
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

import nltk
from nltk.stem.porter import *

from transformers import AutoTokenizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.stats import hypergeom

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from tika import parser
from bs4 import BeautifulSoup
from io import StringIO

import multiprocessing
from multiprocessing import Pool, freeze_support
import subprocess

# sys.path.insert(1, '../src/')
from Basic import *
from gene_lib import *
from stat_lib import *
from excel_lib import *


class Pubmed:

	def __init__(self, bpx, gem, email:str, prefix:str, 
				 root0:str, inidate:str, enddate:str,
				 terms1_param:List=[], terms2_param:List=[], 
				 terms_not_param:List=[], connective_param:str='AND',
				 remove_synonym_list:List=[], sleep_entrez:List=[5, 7, 10], 
				 retmax:int=100000, try_all_text:bool=True, text_quote:str='',
				 root_data_aux:str='../../../data_aux/', dec_ncpus:int=2,
				 sleep_TIKA:List=[10, 20, 30], min_words_text:int=100):

		self.set_dates(inidate, enddate)

		self.bpx = bpx
		self.gem = gem
		self.dfm_pub = None

		self.is_seldata = gem.is_seldata
		self.suffix	 = gem.suffix

		''' spacy '''
		self.sp, self.stemmer = None, None

		self.root_data_aux = root_data_aux
		self.root_refseq  = create_dir(root_data_aux, 'refseq')
		self.root_hgnc	  = create_dir(root_data_aux, 'hgnc')
		self.root_kegg	  = create_dir(root_data_aux, 'kegg')
		self.root_samples = create_dir(root_data_aux, "pmid_samples")

		self.gene = Gene(root_data_aux)

		self.email  = email
		self.dec_ncpus = dec_ncpus

		self.prefix  = prefix
		self.disease = prefix
		self.remove_synonym_list = remove_synonym_list

		self.root0		 = root0
		self.root_data	 = create_dir(root0, 'data')
		self.root_result = create_dir(root0, 'results')
		self.root_figure = create_dir(root0, 'figures')
		self.root_pubmed   = create_dir(root0, 'pubmed')
		self.root_curation = create_dir(root0, 'curation')
		self.root_pdf	   = create_dir(root0, 'pdf_pubmed')
		self.root_pdf_txt = create_dir(root0, 'txt_pubmed')
		self.root_lemma	  = create_dir(root0, 'lemma')
		self.root_ressum  = create_dir(root0, 'res_summ')
		self.root_llm	  = create_dir(root0, 'llm')

		self.root_ptw_modulation = create_dir(root0, 'pathway_modulation')

		self.root_pubgem		= create_dir(root0, 'pubgem')
		self.root_pubgem_pubmed = create_dir(self.root_pubgem, 'pubmed')
		self.root_pubgem_gemini = create_dir(self.root_pubgem, 'gemini')

		self.root_pubmed_root = self.root_pubgem_pubmed if self.is_seldata else self.root_pubmed

		self.root_cluster  = create_dir(root0, 'cluster')
		self.root_clu_lda  = create_dir(self.root_cluster, 'LDA')
		self.root_clu_nmf  = create_dir(self.root_cluster, 'NMF')
		self.root_clu_tsne = create_dir(self.root_cluster, 'tSNE')
		self.root_clu_rfor = create_dir(self.root_cluster, 'RF')

		self.terms1_param = terms1_param
		self.terms2_param = terms2_param
		self.terms_not_param  = terms_not_param
		self.connective_param = connective_param

		self.dfpub0 = None
		self.dfpub_symb, self.pubmed_genes = None, None
		self.dfsymb_perc, self.dfsymb_perc_subclust = None, None
		self.df_lemma = None

		self.df_small_num, self.df_avg_num, self.df_large_num = None, None, None

		PMedLink = "http://www.ncbi.nlm.nih.gov/pubmed/"
		PMCLink  = "http://www.ncbi.nlm.nih.gov/pmc/articles/"
		self.PMedLink = PMedLink
		self.PMCLink  = PMCLink

		os.environ['Link']	 = PMedLink
		os.environ['PMCLink']  = PMCLink
		os.environ['root_pdf'] = self.root_pdf

		self.try_all_text = try_all_text
		self.s_all_text_or_abstract = 'all_text' if self.try_all_text else 'only_abstract'
		self.text_quote = text_quote
		self.s_text_quote = 'without text_quote' if self.text_quote == '' else 'with text_quote'

		self.define_as_choice = False

		self.dfr, self.n_dfr = None, None

		self.root_sh = '../src'
		self.fname_exec = 'down_pdf_pmid.sh'
		self.fname_exec_full = os.path.join(self.root_sh, self.fname_exec)

		ret = os.path.exists(self.fname_exec_full)
		print(f'File {self.fname_exec_full} exists {ret}')

		self.fname_pubmed0 = "pubmed_search_%s_one_gene_%s.tsv"
		self.fname_pubmed_genes = "%s_%s_genes_%s.tsv"
		self.fname_pubmed_all = "all_pubmed_references_for_%s"

		self.fname_compare_gem_rev = 'gemini_x_reviewer_for_run_%s_case_%s.tsv'
		self.fname_compare_gem_rev_person = 'gemini_x_reviewer_for_run_%s_case_%s_reviewer_%s.tsv'

		self.fname_compare_pub_rev = 'pubmed_with_gender_%s_x_reviewer_for_case_%s.tsv'
		self.fname_compare_pub_rev_person = 'pubmed_with_gender_%s_x_reviewer_for_case_%s_reviewer_%s.tsv'

		self.fname_pubmed_reatome = 'pubmed_to_reatome_terms_table.tsv'

		self.fname_summ_concept0 = 'summ_%s_concept_%s_%s.tsv'
		self.fname_summ_no_sumb0 = 'summ_%s_concept_%s.tsv'
		self.fname_summ_symbol_term0 = 'summ_sel_symbol_%s_concept_%s_choice_%s_%s.tsv'

		self.fname_pubmed_with_symb0 = 'pubmed_summ_%s_type_doc_%s_select_%s.tsv'
		self.fname_pubmed_hugo_symb0 = 'pubmed_summ_%s_hugo_symbol_%s.tsv'
		self.fname_curation0 = 'pubmed_curation_%s_between_%s_and_%s_no_symbol_%s.tsv'
		self.fname_lemma0	 = 'lemma_curation_%s_between_%s_and_%s_no_symbol_%s.tsv'
		self.fname_hypergeom = 'hypergeometric_table_for_%s_case_%s.tsv'

		self.fname_pubmed_no_symb0	= 'pubmed_summ_%s_no_symbol_%s_regular_search.tsv'
		self.fname_pubmed_curation_no_symb0 = 'pubmed_summ_%s_no_symbol_%s_curation.tsv'

		self.fname_merged_anwsers	 = 'merged_answers_for_%s_case_%s.tsv'
		self.fname_reviewers_case	 = '%s_reviewers_%s_case_%s.tsv'
		self.fname_reviewers_summary = 'reviewers_answers_for_%s.tsv'

		self.fname_compare_pathways	   = 'compare_pubmed_x_gemini_for_%s_%s_idfps_%s_models_%s_%s.tsv'
		self.fname_compare_both_pubgem = 'compare_pubmed_x_gemini_both_for_%s_%s_idfps_%s_models_%s_%s.tsv'
		self.fname_compare_only_pubmed = 'compare_pubmed_only_for_%s_%s_idfps_%s_models_%s_%s.tsv'
		self.fname_compare_only_gemini = 'compare_gemini_only_for_%s_%s_idfps_%s_models_%s_%s.tsv'
		self.fname_compare_counts	  = 'compare_pubmed_x_gemini_counts_for_%s_%s_idfps_%s_models_%s_%s.tsv'
		self.fname_compare_stats	   = 'compare_pubmed_x_gemini_stats_for_%s_%s_idfps_%s_models_%s_%s.tsv'

		self.fname_all_papers		= 'pubmed_all_papers_%s_no_symbol_%s.tsv'
		self.fname_no_symb_choice0 = 'pubmed_search_summary_no_symbol_for_%s_choice_%s_%s.tsv'
		self.fname_no_symb_concept_choice0 = 'pubmed_search_summary_%s_concept_%s_choice_%s_%s.tsv'

		self.fname_pubmed_merged   = 'pubmed_search_merged_for_%s_%s.tsv'
		self.fname_pubmed_summary  = 'pubmed_search_summary_for_%s_%s.tsv'
		self.fname_pubmed_summary2 = 'pubmed_search_summary_two_cols_for_%s_%s.tsv'
		self.fname_pubmed_semisumm = 'pubmed_search_summary_semi_for_%s_%s.tsv'

		self.fname_pubmed_case_iq_idfp_gender = 'pubmed_search_no_symbol_for_%s_case_%s_i_dfp_%d_with_gender_%s_%s.tsv'
		self.fname_summary_no_symb_by_pmid_case	= 'pubmed_search_summary_no_symbol_by_pmid_for_%s_case_%s_i_dfp_%d_with_gender_%s_%s.tsv'
		self.fname_summary_no_symb_by_pathway_case = 'pubmed_search_summary_no_symbol_by_pathway_for_%s_case_%s_i_dfp_%d_with_gender_%s_%s.tsv'

		self.fname_pubmed_x_gemini_all = 'pubmed_x_gemini_comparison_%s.tsv'
		self.fname_pubmed_x_gemini	 = 'pubmed_x_gemini_comparison_for_run_%s_case_%s_i_dfp_%d_gender_%s_%s.tsv'

		self.fname_stat_gemini_pubmed_agg	   = "stat_gemini_x_pubmed_aggre_for_%s_%s.tsv"
		self.fname_stat_gemini_pubmed_agg_idfp = "stat_gemini_x_pubmed_aggre_for_%s_per_idfp_%s.tsv"


		self.fname_report_all_pubmed_x_gemini = 'report_all_pubmed_x_gemini_for_%s_i_dfp_%d_models_%s_%s.txt'

		self.fname_perc_symb0  = 'pubmed_summ_%s_percentage_symbol_%s.tsv'
		self.fname_perc_symb_subcluster0  = 'pubmed_summ_%s_percentage_symbol_%s_term_%s.tsv'

		self.fname_conv_uniprot = 'conversion_symbol_uniprot_entrez.tsv'
		self.fname_uniprot_symb = 'uniprot_symbol.tsv'

		self.fname_symbol0  = 'pubmed_symbol_%s_%s_%s_%s_and_syns_%s.tsv'

		self.fname_compare_pub_gem = 'comparing_pubmed_x_gemini_for_%s.tsv'
		self.fname_agreement_anal_pub_gem = 'agreement_pubmed_x_gemini_analitical_for_%s_i_dfp_%s_all_models_%s_%s.tsv'
		self.fname_agreement_summ_pub_gem = 'agreement_pubmed_x_gemini_summary_for_%s_i_dfp_%s_all_models_%s_%s.tsv'

		self.fname_3_sources = 'crowd_3sources_%s_for_%s_case_%s_with_gender_%s.tsv'
		self.fname_csc 		 = 'crowd_stats_%s_for_%s_case_%s_with_gender_%s.tsv'
		self.fname_csc_summ	 = 'crowd_summary_for_%s.tsv'
		self.fname_csc_all	 = 'crowd_all_for_%s.tsv'

		self.dfbinv = None
		self.compare_list = ['Accu_PubxGem', 'Accu_RevxGem', 'Accu_RevxPub', 
							 'REVxPubMed x REVxGEM', 
							 'Pathw_GemxCrowd', 'Pathw_PubxCrowd', 'Pathw_RevxCrowd']

		self.fname_stat_3sources_binary_vars = 'stat_3sources_binary_vars_for_%s_runs_%s.tsv'

		self.fname_all_selected_pubmed = "all_selected_pubmed_%s_for_%s.tsv"

		self.fname_CSC_conf_table_summary_stats = "csc_%s_table_for_%s_summary_stats_consensus_run_%s_models_%s_%s.tsv"

		self.pattern_SDS = r'core[a-z] SDS|SDS score'

		Entrez.email = email

		self.stop_words = ['"', '(', ')', '/', ':', '=', '[cls]', '[mask]', '[pad]', '[sep]', 'http', 'https', 'www',
							'fund', 'grant', 'journal', 'license', 'origin', 'online', 'creative', 'author', 'com', 'publish', 'correct',
							'science', 'medicine', 'universe', '[unk]', 'a', 'also', 'analysis', 'and', 'are', 'article',
							'as', 'show', 'at', 'available', 'case', 'centre', 'chen', 'content', 'culture',
							'elsevier', 'et', 'al', 'figure', 'for', 'format', 'free', 'frontier', 'grant',
							'in', 'information', 'is', 'it', 'its', 'liu', 'mL', 'mandarin',
							'md', 'med', 'mg', 'nL', 'news', 'ng', 'of', 'on', 'order',
							'h', 'hour', 'hours', 'ye', 'year', 'years', 'month', 'months', 'ma', 'int',
							'permiss', 'public', 'report', 'repository', 'research',  'pval', 'iqr', 'p-value', 'statistics', 'significant',
							'mean', 'median', 'quartil', 'interquartil', 'anova', 'ttest', 'kruskal', 'scale', 'one', 'two', 'three',
							'resource', 'sample', 'so', 'study', 'such', 'that', 'the', 'to',  'cm2', 'm2',
							'fisher', 'exact', 'test', 'cause', 'first', 'second', 'third', 'fourth',
							'too', 'uL', 'ug', 'use', 'use', 'value', 'values', 'wang', 'was', 'number',
							'were', 'with', 'zhang',  'ml', 'mg', 'ug', 'kg', 'm2', 'dl', 'not', 'none', 'version',
							'l', 'ui', 'µl', 'na', 'mm', 'k', 'r', 'r2', 't', '∆', 'pre', '*', '\n',
							'+', '-', '@', '\\', '<', '>', '<=', '>=', '=', '==', '&', '—']

		self.stop_words = np.unique(self.stop_words)

		self.stop_end_words = ['gi', 'alter', 'pmm', 'ppm', 'mg', 'genechip', 'gyrb', 'aa', 'ag', 'background',
								'group', 'mug', 'mut', 'nm', 'th', 'wk', 'yr', 'dtpa', 'epiut', 'c', 'n1', 'ew',
								'rgroup', 'american', 'ep', 'del', 'ball', 'year', 'eft', 'gbm', 'arnold', 'fdopa',
								'atrt', 'epn', 'fdg', 'express', 'g', 'iu', 'ui', 'ml', 'pv', 'st', 'um', 'bi',
								'aug', 'auc', 'aacr', 'gorlin', 'deg', 'hr', 'dose', 'ependymoma', 'delg', 'asian',
								'degrad', 'dg', 'esc', 'nd']

		''' 99b88  44a  x88  '''
		self.stop_char_nums = ['b', 'p', 'q', 'a', 'x', 'd']

		self.acgt = ['a', 'c', 'g', 't']

		self.stop_stems = ['ac', 'al', 'ar', 'ba', 'ce', 'ch', 'ci', 'ct', 'da', 'day', 'di', '|', 'volum', 'articl',
							'bi', 'pi', 'ig', 'id', 'ly', 'rs', 'ag', 'sc', 'ss', 'ni', 'ge', 'sp', 'sa', 've', 'fe',
							'ea', 'ec', 'ed', 'el', 'en', 'er', 'es', 'et', 'fig', 'ia', 'ic', '−0', 's', '◦', 'p=0',
							'io', 'la', 'le', 'li', 'lo', 'ms', 'nd', 'ne', 'ns', 'nt', 'pg', 'ng', 'mg', 'ug', 'sd', 'oh',
							'sem', 'std',  'pval', 'iqr', 'statist', 'signific', 'mean', 'median', 'quartil', 'interquartil', 'anova',
							'ttest', 'kruskal', 'scale', 'fisher', 'exact', 'test', '\\', 'mhz', 'nano', 'micro', 'mili'
							'pa', 'pe', 'po', 'pr', 'pt', 'ra', 'report', 'ro', 'se', 'si', 'neg', '±', 'posit', 'http', 'https', 'www',
							'st', 'su', 'ta', 'te', 'th', 'ti', 'tr', 'va', 'vi', 'week', 'x', 'g', 'omnibu', 'geo',
							'au', 'org', 'doi', 'ba', 'sar', 'cov', 'p', 'j', 'ml',  '<', '<=', '>', '>=', '=', '==',
							'me', 'cf', 'non', 'b', 'bmi', 'kg', 'm2', 'dl', 'fa', 'eq', 'had', 'has', 'have', 'n', 'self', 'other',
							'one', 'two', 'three', 'four', 'five',
							'first', 'second', 'third', 'fourth',
							'i', 'ii', 'iii', 'iv', 'v',
							'a', 'b', 'c', 'd', 'e', 'f',
							'www', 'frontiersin', 'not', 'none', 'l', 'ui', 'µl', 'na', 'mm', 'k', 'r', 'r2', 't', '∆', 'pre']

		self.stop_stems = np.unique(self.stop_stems)


		''' unselected word by POS '''
		# self.unsel_POS = ['ADP','ADV','AUX','DET','CCONJ','HH','PRON','PUNCT','SCONJ','SYM']
		self.unsel_POS = ['ADP','ADV','AUX','DET','CCONJ','HH','PRON','SCONJ','SYM']

		''' unselected word by dep_ where pos_ == 'ADJ' '''
		self.unsel_adjectives = ['acomp', 'amod', 'appos', 'conj', 'ROOT']

		'''
		all_text = 'ependymoblastoma dear damn distract diagnosi farewel * \n ependymoblastoma diagnost label appli varieti rare central nervou system cn tumor last eight decad consequ uncertainti whether entiti exist characterist featur current base case institut archiv identifi search term ependymoblastoma ependymoblastomat ependymoblast pnet ependym differenti aim hypothesi ependymoblastoma distinct recogniz entiti ependymoblast rosett key diagnost featur present tumor eight embryon tumor abund area neuropil differenti case show rare ependymoblast rosett histopatholog set typic primit neuroectoderm tumor pnet medulloblastoma mb atyp teratoid rhabdoid tumor rt remain case embryon tumor structur mimick ependymoblast rosett result indic ependymoblast rosett most frequent encount embryon tumor abund neuropil less frequent cn embryon neoplasm includ pnet mb andat rt believ ependymoblastoma diagnosi precis specif time onc retir diagnosi lexicon neuropatholog'

		text, stem_text, symbols = self.calc_lemmatization(all_text, filter_adjectives=False, verbose=False)
		sp = spacy.load('en_core_web_sm')
		stemmer = PorterStemmer()
		tokens = sp(all_text)

		lista = []
		for token in tokens:
			# print(token.text, token.pos_, token.dep_)
			if token.pos_ == 'ADJ':
				lista.append(token.dep_)
		lista = np.unique(lista)
		lista
		'''

		self.sleep_entrez  = sleep_entrez
		self.retmax		= retmax
		self.verbose_query = False
		self.force_query   = False

		self.sleep_TIKA	= sleep_TIKA
		self.min_words_text = min_words_text

		self.gene_quote = '"'

		self.exc = Excel(self.root_curation)

	def set_dates(self, inidate:str, enddate:str):
		self.inidate = inidate
		self.enddate = enddate

	def run_many_comparisons(self, choice:List, pathway_name_id_list:List, pathway_concept_list:List,
							 force:bool=False, verbose:bool=False):

		if not isinstance(self.terms1_param, list) or self.terms1_param == []:
			print(">>> Define terms1 as list\n\n")
			raise Exception('stop')

		if not isinstance(self.terms2_param, list) or self.terms2_param == []:
			print(">>> Define terms2 as list\n\n")
			raise Exception('stop')

		if not isinstance(self.terms_not_param, list):
			print(">>> Define terms2 as list - can be empty list.\n\n")
			raise Exception('stop')

		if not isinstance(self.connective_param, str) or self.connective_param == '':
			self.connective_param = 'AND'
			print(">>> Defining connective_param as 'AND'.")


		if choice is None or choice == '' or choice == []:
			print(">>> Define choice")
			raise Exception('stop')

		if not isinstance(pathway_concept_list, list) or pathway_concept_list == []:
			print(">>> Define pathway_concept_list as a list\n\n")
			raise Exception('stop')

		if not isinstance(terms_not_param, list) or terms_not_param == []:
			print(">>> Define at least one terms_not = ['any']\n\n")
			raise Exception('stop')

		self.choice = choice

		connective = deepcopy(self.connective_param)
		terms1	   = deepcopy(self.terms1_param)
		terms2	   = deepcopy(self.terms2_param)
		terms3	 = []
		terms_not = deepcopy(self.terms_not_param)

		num = -1; dic = {}
		print("# Selecting genes per concept:")
		for i in range(len(pathway_name_id_list)):
			pathways = pathway_name_id_list[i]
			concept_list  = pathway_concept_list[i]
			if isinstance(concept_list, str): concept_list = [concept_list]
			concept = concept_list[0]

			# print(">>>> concept1", concept, concept_list)

			for pathway_name_id in pathways:
				pathway_id = pathway_name_id.split(' - ')[1]

				lista = [x for x in os.listdir(self.root_ptw_modulation) if pathway_id in x and x.endswith('.tsv')]

				if lista == []: continue

				for fname in lista:
					# print(fname)
					cases = fname.split('_comparing_')[1][:-4]
					mat = cases.split('_x_')
					case0 = mat[0]
					case1 = mat[1]
					# print(case0, case1)

					dfq = pdreadcsv(fname, self.root_ptw_modulation)
					if dfq.empty: continue

					symbols = list(dfq.symbol.unique())
					symbols.sort()

					for symbol in symbols:
						num += 1
						dic[num] = {}
						dic2 = dic[num]

						dic2['concept'] = concept
						dic2['concept_list'] = str(concept_list)
						dic2['cases'] = cases
						dic2['case0'] = case0
						dic2['case1'] = case1
						dic2['symbol'] = symbol

		'''--- dfs contains all symbols ---'''
		dfs = pd.DataFrame(dic).T
		self.dfs = dfs

		if verbose: print("# Searching in PubMed per concept:")

		df_list = []
		for i in range(len(pathway_name_id_list)):
			# pathways = pathway_name_id_list[i]
			concept_list  = pathway_concept_list[i]
			if isinstance(concept_list, str): concept_list = [concept_list]
			concept = concept_list[0]
			# print(">>>> concept2", concept, concept_list)

			self.concept = concept
			self.concept_list = concept_list
			self.terms3 = ['OR'] + concept_list
			self.concept_suffix = concept

			dfs2 = dfs[dfs.concept == concept]
			if dfs2.empty:
				print("\t??? nothing found for", concept)
				continue

			df_sum = self.run_terms_symbols(dfs2, connective, terms1, terms2, terms3, terms_not, force, verbose)

			if df_sum is not None and not df_sum.empty:
				df_list.append(df_sum)

		if df_list == []:
			return None

		df_all_term = pd.concat(df_list)
		return df_all_term

	def calc_text_to_lemma(self, dfa:pd.DataFrame, lim0:int, lim1:int):

		verbose = False
		if verbose:
			print("---------------")
			print(type(dfa), len(dfa), lim0, lim1)
			print("---------------")

		dfa = dfa.iloc[lim0:lim1,:].copy()
		dfa.index = np.arange(len(dfa))

		lista_pmids, lista_tokens, lista_lemmas, lista_symbols = [], [], [], []

		i = -1
		for pmid in dfa.pmid:
			i += 1
			if i % 100 == 0: print('.',end='')

			lista_pmids.append(pmid)

			fname = f'{pmid}.txt'
			filename = os.path.join(self.root_pdf_txt, fname)

			if not os.path.exists(filename):
				if verbose: print(f"Error: {i}) could not find PMID file '{filename}'")
				lista_tokens.append(None)
				lista_lemmas.append(None)
				lista_symbols.append(None)
				continue

			read_text = False
			lista = read_txt(fname, self.root_pdf_txt, verbose=False)
			if lista != []:
				text = " ".join(lista)
				read_text = True if len(text.split(' ')) >= self.min_words_text else False

			if not read_text:
				if verbose: print(f"Warning: {i}) PMID {pmid} is to little.")
				lista_tokens.append(None)
				lista_lemmas.append(None)
				lista_symbols.append(None)
				continue

			# print(f"\ntext {i})", text[:240])
			text, stem_text, symbols = self.calc_lemmatization(text, filter_adjectives=False, verbose=verbose)

			lista_tokens.append(text)
			lista_lemmas.append(stem_text)
			lista_symbols.append(symbols)

		if verbose:
			print(">>> exit one loop", lim0, lim1)
			print(len(lista_pmids), len(lista_tokens), len(lista_lemmas), len(lista_symbols))

		return lista_pmids, lista_tokens, lista_lemmas, lista_symbols


	def prepare_names_token_and_lemma(self, all_pdf_html:str):
		self.all_pdf_html = all_pdf_html

		fname = self.fname_pubmed_with_symb0%(self.prefix, self.s_all_text_or_abstract, all_pdf_html)
		fname = title_replace(fname)
		fname_token = fname.replace('.tsv', '_token.tsv')
		fname_lemma = fname.replace('.tsv', '_lemma.tsv')

		return fname, fname_token, fname_lemma

	def prepare_curation_names_token_and_lemma(self):
		fname = self.fname_lemma0%(self.concept, self.inidate, self.enddate, self.s_all_text_or_abstract)
		fname = title_replace(fname)
		fname_token = fname.replace('.tsv', '_token.tsv')
		fname_lemma = fname.replace('.tsv', '_lemma.tsv')

		return fname, fname_token, fname_lemma

	def prepare_hypergeometric_name(self, case:str) -> str:
		fname = self.fname_hypergeom%(self.prefix, case)

		return fname

	def calc_curation_lemmatization(self, force:bool=False, verbose:bool=False) -> bool:
		'''
		dfpub is final version, already processed
		lemmatize all text having >= 10 words
		'''
		dfpub = self.open_df_pubmed_curation(final_version=True)
		if dfpub is None or dfpub.empty:
			print("No dfpub in calc_curation_lemmatization()")
			return False

		dfpub = dfpub[dfpub['choice'] == 'all'].copy()
		if dfpub.empty:
			print("No dfpub with choice == 'all'")
			return False

		dfpub.index = np.arange(len(dfpub))
		self.dfpub = dfpub

		lista = [int(x.split('.')[0]) for x in os.listdir(self.root_pdf_txt) if x.endswith('.txt')]
		lista = [x for x in lista if x in list(dfpub.pmid)]
		if lista == []:
			print("No pubmed txt files were found. Please run 'pubmed_medulloblastoma_new27_no_pdf_and_pdf_TIKA_parser_curation'")
			return False

		fname, fname_token, fname_lemma= self.prepare_curation_names_token_and_lemma()
		filename = os.path.join(self.root_lemma, fname)

		if os.path.exists(filename) and not force:
			self.df_concept_symb = pdreadcsv(fname, self.root_lemma, verbose=verbose)
			self.df_token = pdreadcsv(fname_token, self.root_lemma, verbose=verbose)
			self.df_lemma = pdreadcsv(fname_lemma, self.root_lemma, verbose=verbose)
			return True

		nprocessors = multiprocessing.cpu_count()
		nprocessors -= self.dec_ncpus
		self.nprocessors = nprocessors

		n = len(dfpub)
		deln = int(n/nprocessors)

		ini = 0
		limits = []
		for i in range(nprocessors):
			end = n if i == nprocessors-1 else (i+1)*deln
			limits.append( (ini, end) )
			ini = end

		self.limits = limits

		''' https://stackoverflow.com/questions/20548628/how-to-do-parallel-programming-in-python '''
		pool = Pool(nprocessors)

		dic = {}
		lista_results = []

		for i in range(nprocessors):
			result = pool.apply_async(self.calc_text_to_lemma, (dfpub, limits[i][0], limits[i][1]) )
			print("\t\t<<<", len(result) if result is not None else 'None')
			lista_results.append(result)

		freeze_support()
		t0 = time.time()

		for i in range(len(lista_results)):
			result = lista_results[i]
			dic[i] = {}
			dic2 = dic[i]

			dic2['lista_pmid'], dic2['lista_tokens'], dic2['lista_lemmas'], dic2['lista_symbols'] = result.get(timeout=10000)

			print("%d is ok "%(i))

		seconds = time.time()-t0
		print("---- End: %.1f min -----------"%(seconds/60))

		lista_pmid, lista_tokens, lista_lemmas, lista_symbols = [], [], [], []
		for i in range(nprocessors):
			dic2 = dic[i]

			lista_pmid	+= dic2['lista_pmid']
			lista_tokens  += dic2['lista_tokens']
			lista_lemmas  += dic2['lista_lemmas']
			lista_symbols += dic2['lista_symbols']

		self.lista_pmid	= lista_pmid
		self.lista_tokens  = lista_tokens
		self.lista_lemmas  = lista_lemmas
		self.lista_symbols = lista_symbols

		df_concept_symb = pd.DataFrame({'pmid': lista_pmid, 'symbols': lista_symbols})
		df_concept_symb = pd.merge(dfpub, df_concept_symb, how="outer", on='pmid')

		cols = ['concept', 'choice', 'pmid', 'ftype', 'has_text', 'symbols', 'pub_date', 'title',
				'keywords', 'abstract', 'abreviation', 'authors', 'created_date', 'doc_type', 'docid',
				'journalTitle', 'language', 'cases', 'case_comparisons', 'terms', 'dates']

		df_concept_symb = df_concept_symb[cols]
		self.df_concept_symb = df_concept_symb

		ret1 = pdwritecsv(df_concept_symb, fname, self.root_lemma, verbose=True)

		df_token = pd.DataFrame({'pmid': lista_pmid, 'token_text': lista_tokens})
		self.df_token = df_token
		ret2 = pdwritecsv(df_token, fname_token, self.root_lemma, verbose=True)

		df_lemma = pd.DataFrame({'pmid': lista_pmid, 'lemma': lista_lemmas})
		self.df_lemma = df_lemma
		ret3 = pdwritecsv(df_lemma, fname_lemma, self.root_lemma, verbose=True)

		return ret1*ret2*ret3 == 1

	def calc_lemmatization_and_symbols(self, all_pdf_html:str='all', force:bool=False, verbose:bool=False) -> bool:

		lista = [x for x in os.listdir(self.root_pdf_txt) if x.endswith('.txt')]

		if all_pdf_html == 'pdf':
			lista_pdf = [x[:-4]+'.txt' for x in os.listdir(self.root_pdf) if x.endswith('.pdf')]

			lista	 = [x for x in lista if x in lista_pdf]
			if len(lista) != len(lista_pdf):
				print(f"Warning: #lista = {len(lista)} and #lista_pdf = {len(lista_pdf)}")
		elif all_pdf_html == 'html':
			''' html --> not pdf '''
			lista_html = [x[:-4]+'.txt' for x in os.listdir(self.root_pdf) if x.endswith('.html')]

			lista	 = [x for x in lista if x in lista_html]
			if len(lista) != len(lista_html):
				print(f"Warning: #lista = {len(lista)} and #lista_html = {len(lista_html)}")

		if lista == []:
			print("No pubmed txt files were found. Please run 'pubmed_taubate_03_TIKA_parser'")
			return False

		fname, fname_token, fname_lemma= self.prepare_names_token_and_lemma(all_pdf_html)
		filename = os.path.join(self.root_lemma, fname)

		if os.path.exists(filename) and not force:
			_, _, _ = self.open_lemma(all_pdf_html=all_pdf_html, open_all=True, verbose=verbose)
			return True

		dfa = self.open_df_pubmed_summary()
		self.dfa = dfa

		nprocessors = multiprocessing.cpu_count()
		nprocessors -= self.dec_ncpus
		self.nprocessors = nprocessors

		n = len(dfa)
		deln = int(n/nprocessors)

		ini = 0
		limits = []
		for i in range(nprocessors):
			end = n if i == nprocessors-1 else (i+1)*deln
			limits.append( (ini, end) )
			ini = end

		self.limits = limits

		''' https://stackoverflow.com/questions/20548628/how-to-do-parallel-programming-in-python '''
		pool = Pool(nprocessors)

		dic = {}
		lista_results = []

		print("Start reading txt-papers")
		for i in range(nprocessors):
			print("\tloop", i, limits[i][0], limits[i][1], end=' ')
			result = pool.apply_async(self.calc_text_to_lemma, (dfa, limits[i][0], limits[i][1]) )
			print('result', result)
			lista_results.append(result)

		freeze_support()
		t0 = time.time()

		print("\n>>> Merging all data ...#", len(lista_results))
		for i in range(len(lista_results)):
			result = lista_results[i]

			dic[i] = {}
			dic2 = dic[i]
			dic2['lista_pmid'], dic2['lista_tokens'], dic2['lista_lemmas'], dic2['lista_symbols'] = result.get(timeout=10000)
			print(f"{i}) is ok ")

		seconds = time.time()-t0
		print("---- End: %.1f min -----------"%(seconds/60))

		lista_pmid, lista_tokens, lista_lemmas, lista_symbols = [], [], [], []
		for i in range(nprocessors):
			dic2 = dic[i]

			lista_pmid	+= dic2['lista_pmid']
			lista_tokens  += dic2['lista_tokens']
			lista_lemmas  += dic2['lista_lemmas']
			lista_symbols += dic2['lista_symbols']

		self.lista_pmid	= lista_pmid
		self.lista_tokens  = lista_tokens
		self.lista_lemmas  = lista_lemmas
		self.lista_symbols = lista_symbols

		df_concept_symb = pd.DataFrame({'pmid': lista_pmid, 'symbols': lista_symbols})
		print(">>> df_concept_symb #", len(df_concept_symb))

		df_concept_symb = pd.merge(dfa, df_concept_symb, how="outer", on='pmid')

		cols = ['concept', 'choice', 'pmid', 'ftype', 'has_text', 'symbols', 'pub_date', 'title',
				'keywords', 'abstract', 'abreviation', 'authors', 'created_date', 'doc_type', 'docid',
				'journalTitle', 'language', 'cases', 'case_comparisons', 'terms', 'dates']

		df_concept_symb = df_concept_symb[cols]
		self.df_concept_symb = df_concept_symb

		ret1 = pdwritecsv(df_concept_symb, fname, self.root_lemma, verbose=True)

		df_token = pd.DataFrame({'pmid': lista_pmid, 'token_text': lista_tokens})
		ret2 = pdwritecsv(df_token, fname_token, self.root_lemma, verbose=True)

		df_lemma = pd.DataFrame({'pmid': lista_pmid, 'lemma': lista_lemmas})
		ret3 = pdwritecsv(df_lemma, fname_lemma, self.root_lemma, verbose=True)

		return ret1*ret2*ret3 == 1


	def zip_pdf_and_html(self):
		'''----------- zip_compress in Basic ------------'''

		for  type_of_file in ['pdf', 'html']:
			fname_zip = f'{self.prefix}_{type_of_file}.zip'
			ret = zip_compress(fname_zip, self.root0, self.root_pdf, type_of_file)

			if not ret:
				return False

		return ret


	def run_concept_choice_terms2(self, concept:str, dic_choice:dict, 
								  force:bool=False, verbose:bool=False) -> pd.DataFrame:

		self.concept = concept

		fname = self.fname_curation0%(concept, self.inidate, self.enddate, self.s_all_text_or_abstract)
		fname = title_replace(fname)
		filename = os.path.join(self.root_curation, fname)

		if os.path.exists(filename) and not force:
			return pdreadcsv(fname, self.root_curation, verbose=verbose)

		df_all_list =[]
		for choice in dic_choice.keys():
			dic2 = dic_choice[choice]

			terms2	= dic2['terms2']
			terms_not = dic2['terms_not']

			print(f"Params: concept {concept}, terms2:{terms2}, terms_not:{terms_not}")

			df_choi = self.run_simple_comparisons(concept=concept, choice=choice, force=force, verbose=verbose)
			print("\n")

			df_all_list.append(df_choi)

		print("\n--------------- final end -------------------\n")

		if df_all_list == []:
			return None

		df_all = pd.concat(df_all_list)

		df_all = df_all.sort_values(['concept', 'choice', 'pub_date', 'pmid'], ascending=[True, True, False, False])
		df_all.index = np.arange(len(df_all))
		self.df_all = df_all

		ret = pdwritecsv(df_all, fname, self.root_curation, verbose=True)

		return df_all

	def calc_reactome_terms_table(self, df_sel:pd.DataFrame, force:bool=False, verbose:bool=False) -> pd.DataFrame:
		filename = os.path.join(self.root_refseq, self.fname_pubmed_reatome)

		cols = ['pathway_id', 'pathway']

		if os.path.exists(filename):
			df_ptw_terms = pdreadcsv(self.fname_pubmed_reatome, self.root_refseq, verbose=verbose)

			if df_ptw_terms is None or df_ptw_terms.empty:
				dfa = df_sel.copy()
			else:
				dfa = df_sel[ ~df_sel.pathway_id.isin(df_ptw_terms.pathway_id) ].copy()

				if dfa.empty:
					if verbose: print(f"All pathways already have terms.")
					''' if empty, all terms have been fulfilled '''
					return df_ptw_terms

			dfa = dfa[cols]
			dfa['term'] = None

			if df_ptw_terms is None or df_ptw_terms.empty:
				df_ptw_terms = dfa
			else:
				df_ptw_terms = pd.concat([df_ptw_terms, dfa])
		else:
			df_ptw_terms = df_sel[cols].copy()
			df_ptw_terms['term'] = None

		df_ptw_terms = df_ptw_terms.drop_duplicates('pathway_id')
		df_ptw_terms.index = np.arange(len(df_ptw_terms))
		ret = pdwritecsv(df_ptw_terms, self.fname_pubmed_reatome, self.root_refseq, verbose=verbose)

		print(f"Please, fulfill pathway-terms table: '{filename}'")

		return df_ptw_terms


	def case_to_terms(self, case:str, with_gender:bool) -> (List, List):
		if 'covid' in self.prefix:
			return self.case_to_terms_covid(case, with_gender)

		if self.prefix == 'medulloblastoma':
			return self.case_to_terms_medulloblastoma(case)

		print("Please, create a method for", self.prefix)
		return [], []

	def case_to_terms_medulloblastoma(self, case:str) -> (List, List):

		if case == 'WNT':
			term_list = ['WNT']
		elif case == 'SHH':
			term_list = ['OR', 'SHH', 'Hedgehog']
		elif case == 'G3':
			term_list = ['OR', 'Group 3', 'G3']
		elif case == 'G4':
			term_list = ['OR', 'Group 4', 'G4']
		else:
			print(f"Error: could not define the case {case} for {self.prefix}")
			term_list = []

		'''	return term_list + term_not '''
		return term_list, []



	def case_to_terms_covid(self, case:str, with_gender:bool) -> (List, List):

		if case == 'g1_female':
			term_list = ['female', 'asymptomatic'] if with_gender else ['asymptomatic']
			not_list = ['severe', 'intensive', 'outpatient']

		elif case == 'g1_male':
			term_list = ['male', 'asymptomatic'] if with_gender else ['asymptomatic']
			not_list = ['severe', 'intensive', 'outpatient']

		elif case == 'g2a_female':
			term_list = ['female', 'mild'] if with_gender else ['mild']
			not_list = ['severe', 'intensive']

		elif case == 'g2a_male':
			term_list = ['male', 'mild']  if with_gender else ['mild']
			not_list = ['severe', 'intensive']

		elif case == 'g2b_female':
			term_list = ['female', 'moderate', 'outpatient'] if with_gender else ['moderate', 'outpatient']
			not_list = ['mild', 'asymptomatic', 'severe', 'intensive']

		elif case == 'g2b_male':
			term_list = ['male', 'moderate', 'outpatient'] if with_gender else ['moderate', 'outpatient']
			not_list = ['mild', 'asymptomatic', 'severe', 'intensive']

		elif case == 'g3_female_adult':
			term_list = ['female', 'severe'] if with_gender else ['severe']
			not_list = ['elder', 'outpatient', 'mild', 'moderate', 'asymptomatic']

		elif case == 'g3_male_adult':
			term_list = ['male', 'severe'] if with_gender else ['severe']
			not_list = ['elder', 'outpatient', 'mild', 'moderate', 'asymptomatic']

		elif case == 'g3_female_elder':
			term_list = ['female', 'elder', 'severe'] if with_gender else ['elder', 'severe']
			not_list = ['outpatient', 'mild', 'moderate', 'asymptomatic']

		elif case == 'g3_male_elder':
			term_list = ['male', 'elder', 'severe'] if with_gender else ['elder', 'severe']
			not_list = ['outpatient', 'mild', 'moderate', 'asymptomatic']

		else:
			print(f"Error: could not define the case {case} for {self.prefix}")
			term_list = []
			not_list = []

		'''	return term_list + term_not '''
		return term_list, not_list + ['child', 'neonat', 'newborn']


	def open_dfp_default(self, case:str, i_dfp:int,  verbose:bool=False) -> pd.DataFrame:

		ret, _, _, _ = self.bpx.open_case(case, verbose=False)

		df_read = self.gem.one_gemini_fname(run=self.gem.run_default, case=case, iq=0, i_dfp=i_dfp, 
											chosen_model=self.gem.chosen_model_default, verbose=verbose)

		return df_read


	def run_case_pathway_pubmed_search(self, case:str, i_dfp:int, with_gender:bool, 
									   test:bool=False, show_warning:bool=True,
									   N:int=30, query_type:str='strong',
									   force:bool=False, verbose:bool=False) -> pd.DataFrame:
		'''

		if self.is_seldata:
			dfp = self.gem.open_yes_no_sampling(case=case, N=N, query_type=query_type, verbose=True)
		else:
			dfp = self.open_dfp_default(case, i_dfp)

		'''
		self.df_pmid, self.df_summ_pmid, self.df_summ_pathway = None, None, None

		if not isinstance(self.terms1_param, list) or self.terms1_param == []:
			print("Error: Define terms1 as list\n\n")
			raise Exception('stop')

		if not isinstance(self.terms_not_param, list):
			print("Error: Define terms2 as list - can be empty list.\n\n")
			raise Exception('stop')

		if not isinstance(self.connective_param, str) or self.connective_param == '':
			self.connective_param = 'AND'
			print("Warning: Defining connective_param as 'AND'.")

		connective = deepcopy(self.connective_param)
		terms1	   = deepcopy(self.terms1_param)
		terms2	   = []   # terms2 is defined below
		terms3	   = []
		terms_not  = deepcopy(self.terms_not_param)

		self.df_summ_pmid, self.df_summ_pathway = None, None

		self.case = case
		self.with_gender = with_gender

		self.pmid_dic = {}
		self.concept, self.concept_list, self.choice = None, None, None

		terms_and_or, terms_not2 = self.case_to_terms(case, with_gender)

		if terms_and_or == []:
			print("Error: No terms were defined.")
			raise Exception('stop')

		# add NOT if necessary
		if terms_not != [] and terms_not[0] != 'NOT':
			terms_not = ['NOT'] + terms_not

		# add terms_not2 that comes form case to terms()
		if terms_not2 != []:
			if terms_not == []:
				terms_not = ['NOT'] + terms_not2
			else:
				terms_not += terms_not2

		self.terms_not = terms_not


		if self.is_seldata:
			dfp = self.gem.open_yes_no_sampling(case=case, N=N, query_type=query_type, verbose=verbose)
			stri = "is_seldata True -> open_yes_no_sampling"
		else:
			dfp = self.open_dfp_default(case, i_dfp, verbose=verbose)
			stri = "is_seldata False -> open_dfp_default"


		if dfp is None or dfp.empty:
			print("Nothing found in {stri}")
			return None


		#- see if all df_enr pathways are annotated in reactome-term table
		df_ptw_terms = self.calc_reactome_terms_table(dfp, verbose=verbose)
		if df_ptw_terms is None or df_ptw_terms.empty:
			print("build df_enr_pubmed_search_table(): empty table")
			return None

		fname_pmid = self.fname_pubmed_case_iq_idfp_gender%(self.prefix, case, i_dfp, with_gender, self.suffix)
		fname_pmid = title_replace(fname_pmid)

		filename = os.path.join(self.root_pubmed_root, fname_pmid)
		if os.path.exists(filename) and not force:
			df_pmid = pdreadcsv(fname_pmid, self.root_pubmed_root, verbose=verbose)
			print(f"\t\t case {case:15} i_dfp={i_dfp}: {len(df_pmid)} pmids")
		else:
			# terms_not = ['NOT', 'MERS', 'SARS-CoV-1']
			# terms1 = ["OR", 'COVID', 'SARS-CoV-2']
			terms2 = terms_and_or

			if verbose: print('\t', terms1, terms2, terms_not,'\n')

			# sendo to search in pubmed: all df_enr that has terms in df_ptw_terms
			df_ptw_pubmed = df_ptw_terms[df_ptw_terms.pathway_id.isin(dfp.pathway_id)].copy()
			df_ptw_pubmed.index = np.arange(len(df_ptw_pubmed))

			df_pmid = self.run_terms_loop_pathways(df_ptw_pubmed, case, i_dfp, 
												   connective, terms1, terms2, terms3, terms_not, 
												   test=test, show_warning=show_warning, 
												   force=force, verbose=verbose)

			if df_pmid is None or df_pmid.empty:
				print(f"\t\t case {case:15} i_dfp={i_dfp}: no pmids for {case}")
				# print("\n------------------ end all pathways -------- no data ----\n")
				return None

			print(f"\t\t case {case:15} i_dfp={i_dfp}: {len(df_pmid)} pmids")
			# print("\n------------------ end all pathways  ----------------------\n")

		df_summ_pmid	= self.calc_summary_by_pmid(df_pmid, case, i_dfp, with_gender, force=force, verbose=verbose)
		df_summ_pathway = self.calc_summary_by_pathway(df_pmid, case, i_dfp, with_gender, force=force, verbose=verbose)

		self.df_pmid = df_pmid
		self.df_summ_pmid = df_summ_pmid
		self.df_summ_pathway = df_summ_pathway

		return df_pmid


	def calc_summary_by_pmid(self, df_pmid:pd.DataFrame, case:str, i_dfp:int, with_gender:bool, 
							 force:bool=False, verbose:bool=False) -> pd.DataFrame:

		if df_pmid is None or df_pmid.empty:
			return None

		fname = self.fname_summary_no_symb_by_pmid_case%(self.prefix, case, i_dfp, with_gender, self.suffix)
		fname = title_replace(fname)

		filename = os.path.join(self.root_pubmed_root, fname)
		if os.path.exists(filename) and not force:
			return pdreadcsv(fname, self.root_pubmed_root, verbose=verbose)

		df_pmid = df_pmid.sort_values(['pmid', 'pathway'])
		df_pmid.index = np.arange(len(df_pmid))

		previous_pmid = '';  new_row = None
		df_list, pathway_id_list_list, pathway_list_list = [], [], []
		pathway_id_list = []

		for i in range(len(df_pmid)):
			row = df_pmid.iloc[i].copy()
			pmid = row.pmid

			if previous_pmid != pmid:
				if new_row is not None:
					df_list.append(pd.DataFrame(new_row).T)
					pathway_id_list_list.append(pathway_id_list)
					pathway_list_list.append(pathway_list)

				pathway_id_list = [row.pathway_id]
				pathway_list  = [row.pathway]
				previous_pmid = pmid
				new_row = row
			else:
				pathway_id_list += [row.pathway_id]
				pathway_list	+= [row.pathway]


		df_list.append(pd.DataFrame(new_row).T)
		pathway_id_list_list.append(pathway_id_list)
		pathway_list_list.append(pathway_list)

		df_summ = pd.concat(df_list)

		df_summ['case'] = case
		df_summ['i_dfp'] = i_dfp
		df_summ['with_gender']  = with_gender
		df_summ['pathway_id_list']	= pathway_id_list_list
		df_summ['pathway_list']	= pathway_list_list
		df_summ['n_pathway'] = [len(x) for x in pathway_id_list_list]

		cols = ['case', 'i_dfp', 'with_gender', 'pmid', 
				'n_pathway', 'pathway_id_list', 'pathway_list', 'pub_date', 
				'title', 'keywords', 'abstract', 'abreviation', 'authors', 
				'created_date', 'doc_type', 'docid', 'journalTitle', 'language']

		df_summ = df_summ[cols]

		df_summ.index = np.arange(len(df_summ))
		ret = pdwritecsv(df_summ, fname, self.root_pubmed_root, verbose=verbose)

		return df_summ

	def calc_summary_by_pathway(self, df_pmid:pd.DataFrame, case:str, i_dfp:int, with_gender:bool, 
								force:bool=False, verbose:bool=False) -> pd.DataFrame:
		if df_pmid is None or df_pmid.empty:
			return None

		fname = self.fname_summary_no_symb_by_pathway_case%(self.prefix, case, i_dfp, with_gender, self.suffix)
		fname = title_replace(fname)

		filename = os.path.join(self.root_pubmed_root, fname)
		if os.path.exists(filename) and not force:
			return pdreadcsv(fname, self.root_pubmed_root, verbose=verbose)

		df_pmid = df_pmid.sort_values(['pathway_id', 'pmid'])
		df_pmid.index = np.arange(0, len(df_pmid))

		previous_pathway = '';  new_row = None
		df_list, pmid_list_list = [], []
		pmid_list = []

		for i in range(len(df_pmid)):
			row = df_pmid.iloc[i].copy()

			if previous_pathway != row.pathway_id:
				if new_row is not None:
					df_list.append(pd.DataFrame(new_row).T)
					pmid_list_list.append(pmid_list)

				pmid_list = [row.pmid]
				previous_pathway = row.pathway_id
				new_row = row
			else:
				pmid_list += [row.pmid]

		df_list.append(pd.DataFrame(new_row).T)
		pmid_list_list.append(pmid_list)

		df_summ = pd.concat(df_list)
		df_summ['case'] = case
		df_summ['i_dfp'] = i_dfp
		df_summ['with_gender']  = with_gender
		df_summ['pmid_list'] = pmid_list_list
		df_summ['n_pmid'] = [len(x) for x in pmid_list_list]

		cols = ['case', 'i_dfp', 'with_gender', 'pathway_id', 'pathway', 'n_pmid', 'pmid_list']
		df_summ = df_summ[cols]

		df_summ = df_summ.sort_values('n_pmid', ascending=False)
		df_summ.index = np.arange(0, len(df_summ))
		ret = pdwritecsv(df_summ, fname, self.root_pubmed_root, verbose=verbose)

		return df_summ

	def run_choice_concept_comparisons(self, dic_choice: dict, pathway_concept_name_list: List,
									   pathway_concept_list:List, 
									   test:bool=False, force:bool=False, verbose:bool=False) -> pd.DataFrame:

		fname = self.fname_pubmed_no_symb0%(self.prefix, self.s_all_text_or_abstract)
		fname = title_replace(fname)

		filename = os.path.join(self.root_pubmed, fname)
		if os.path.exists(filename) and not force:
			dfa = pdreadcsv(fname, self.root_pubmed, verbose=verbose)
			return dfa

		df_all_list =[]
		for choice, dic2 in dic_choice.items():

			terms1 = dic2['terms1']
			terms2 = dic2['terms2']
			terms_not = dic2['terms_not']

			print(choice)
			print('\t', terms1, terms2, terms_not,'\n')

			'''
			query = '(medulloblastoma*[Title/Abstract]) AND (cardi*[Title/Abstract]) AND (tumor*[Title/Abstract]) NOT covid*[Title/Abstract] NOT SARS-CoV*[Title/Abstract] AND ("1990/01/01"[Date - Publication] : "2030/12/31"[Date - Publication])' retmax 100000

			'''
			df_choi = self.run_big_comparisons(choice, pathway_concept_name_list, pathway_concept_list, 
											   test=test, force=force, verbose=verbose)

			if df_choi is None or df_choi.empty:
				continue

			df_all_list.append(df_choi)

		print("\n--------------- end run_big_comparisons() -------------------\n")

		if df_all_list == []:
			print("Nothing found (1).")
			return None

		df_all = pd.concat(df_all_list)
		self.df_all = df_all

		df_all = df_all[ ~df_all.pmid.isna()]
		if df_all.empty:
			print("Nothing found (2).")
			return None

		print(">>> df_all", len(df_all))

		df_all.pmid = df_all.pmid.astype(int)
		df_all = df_all.sort_values('pmid')

		previous_pmid = '';  new_row = None
		df_list = []
		for i in range(len(df_all)):
			row = df_all.iloc[i].copy()
			pmid = row.pmid

			if previous_pmid != pmid:
				if new_row is not None:
					df_list.append(pd.DataFrame(new_row).T)

				previous_pmid = pmid
				new_row = row

				choice = row.choice
				if isinstance(choice, str):
					choice = eval(choice) if choice.startswith('[') else [choice]

				concept = row.concept
				if isinstance(concept, str):
					concept = eval(concept) if concept.startswith('[') else [concept]

			else:
				choice2 = row.choice
				if isinstance(choice2, str):
					choice2 = eval(choice2) if choice2.startswith('[') else [choice2]

				concept2 = row.concept
				if isinstance(concept2, str):
					concept2 = eval(concept2) if concept2.startswith('[') else [concept2]

				if choice == []:
					choice = choice2
				elif choice2 == []:
					pass
				else:
					choice += choice2

				if concept == concept2 or concept2 == []:
					pass
				elif concept == []:
					concept = concept2
				else:
					concept += concept2

				new_row.choice = choice
				new_row.concept = concept
				new_row.concept_list = None

		df_list.append(pd.DataFrame(new_row).T)
		dfa = pd.concat(df_list)

		dfa['concept'] = [";".join(np.unique(x)) for x in dfa.concept]
		dfa['choice']  = [";".join(np.unique(x)) for x in dfa.choice]

		dfa = dfa.sort_values(['concept', 'choice', 'pub_date', 'pmid'], ascending=[True, True, False, False])
		dfa.index = np.arange(0, len(dfa))
		ret = pdwritecsv(dfa, fname, self.root_pubmed, verbose=True)

		return dfa

	def run_terms_comparisons(self, choice:str, pathway_concept_name_list: List, pathway_concept_list:List,
							  root_to_save:str, test:bool=False, 
							  force:bool=False, verbose:bool=False) -> pd.DataFrame:
		'''
			a method to sample determined issue
			importante to create pmid samples for AI-LLM

			if sets root_pubmed, where to save df_choi
			in the end, resets root_pubmed

			example:
				connective_param = "AND"

				choice = 'medulloblastoma'
				terms1 = [choice]
				terms2 = ['OR', 'transcript', 'proteom', 'epigenet']

				pathway_concept_list = [['develop', 'tumor']]
				pathway_concept_name_list = [['medulloblastoma']]

				terms_not = ['NOT', 'covid', 'SARS-CoV']  # glioblastoma, 'retinoblastoma', 'glioma',

				inidate = "2005/01/01"
				enddate = "2030/12/31"
		'''

		root_pubmed_bck = self.root_pubmed

		self.root_pubmed  = create_dir(self.root_samples, root_to_save)
		self.root_to_save = self.root_pubmed

		self.choice = choice
		self.pathway_concept_name_list = pathway_concept_name_list
		self.pathway_concept_list = pathway_concept_list

		if verbose: print('\t', terms1, terms2, terms_not,'\n')

		df_choi = self.run_big_comparisons(choice=choice, pathway_concept_name_list=pathway_concept_name_list,
										   pathway_concept_list=pathway_concept_list, 
										   test=test, force=force, verbose=verbose)


		self.root_pubmed = root_pubmed_bck
		return df_choi

	def download_html_samples(self, df_pmid:pd.DataFrame, root_to_save_html:str,
							  nSamples:int=200, delete_html:bool=False, verbose:bool=True) -> bool:
		'''
		donwload {nSamples} htmls, acording to pmids given in df_pmid

		change the name of fname_exec --> to hmtl, in the end reset

		!ls -ls ../src/*.sh
		'''

		self.set_search_or_curation('regular_search')

		self.fname_exec = 'down_html_pmid.sh'
		self.fname_exec_full = os.path.join(self.root_sh, self.fname_exec)

		self.root_to_save_html = create_dir(self.root0, root_to_save_html)

		os.environ["root_pdf"] = self.root_to_save_html+'/'
		if verbose: print(os.environ["root_pdf"])

		'''
			download pdfs/html: html from Pubmed(pmid) or pdf from PMC
		'''
		nprocessors = multiprocessing.cpu_count()
		nprocessors -= self.dec_ncpus
		self.nprocessors = nprocessors

		''' samplin pmids '''
		pmids = np.array(df_pmid.pmid)
		lista = np.arange(0, len(pmids))
		ilist = np.random.choice(lista, size=nSamples, replace=False)

		pmids = pmids[ilist]

		self.npmids	= len(pmids)
		self.pmid_list = pmids

		if self.npmids == 0:
			print("No pmids could be sampled.")
			return False

		bad_files = [x for x in os.listdir(self.root_to_save_html) if x.endswith('.bad')]
		if bad_files != []:
			for fname in bad_files:
				filename = os.path.join(self.root_to_save_html, fname)
				try:
					os.unlink(filename)
				except:
					print(f"Could not delete {filename}")

		if delete_html:
			html_files = [x for x in os.listdir(self.root_to_save_html) if x.endswith('.html')]
			if html_files != []:
				for fname in html_files:
					filename = os.path.join(self.root_to_save_html, fname)
					try:
						os.unlink(filename)
					except:
						print(f"Could not delete {filename}")

		files = [x for x in os.listdir(self.root_to_save_html) if x.endswith('.html')]
		pmids_done = [ int(x.split('.')[0]) for x in files]

		lista_not_dwn = [x for x in self.pmid_list if x not in pmids_done]
		self.lista_not_dwn = lista_not_dwn

		n_notdwn = len(lista_not_dwn)
		self.n_notdwn = n_notdwn

		if n_notdwn == 0:
			print("All done.")
			return True

		print(f'There are {n_notdwn} pmids to download.')
		deln = int(n_notdwn/nprocessors)

		ini = 0
		limits = []
		for i in range(nprocessors):
			end = n_notdwn if i == nprocessors-1 else (i+1)*deln
			limits.append( (ini, end) )
			ini = end

		''' ---------------- parallel processing -------------------'''
		pool = Pool(nprocessors)

		lista_results = []

		for i in range(nprocessors):
			lista_results.append(pool.apply_async(self.pdf_download, (i, limits[i])))

		freeze_support()
		t0 = time.time()

		print(">>>> lista_results", len(lista_results), 'processors')
		for i in range(len(lista_results)):
			result = lista_results[i]

			try:
				# print(f"getting {i}")
				result.get(timeout=10000)
				# print(f"ok download {i}")
			except:
				print("Parallel processing has problems.")
				raise Exception(f'Stop parallel processing {i}')

			print("%d finished"%(i))

		seconds = time.time()-t0
		print("---- End: %.1f min -----------"%(seconds/60))

		''' return fname_exec '''
		self.fname_exec = 'down_pdf_pmid.sh'
		self.fname_exec_full = os.path.join(self.root_sh, self.fname_exec)

		return True


	def run_simple_comparisons(self, concept:str, choice:str, allow_repeat:bool=True, force=False, verbose=False) -> pd.DataFrame:

		if not isinstance(terms2_param, list):
			print(">>> Define terms2_param as list")
			raise Exception('stop')

		if not isinstance(terms_not_param, list):
			print(">>> Define at least one terms_not_param = ['any']")
			raise Exception('stop')

		self.concept = deepcopy(concept)
		self.concept_list = deepcopy([concept])

		self.choice = deepcopy(choice)
		self.define_as_choice = True

		self.no_repeats = self.terms2 == [] and not allow_repeat

		connective = deepcopy(self.connective_param)
		terms1 	   = deepcopy([concept])
		terms2	   = deepcopy(self.terms2_param)
		terms3	   = []		
		terms_not  = deepcopy(self.terms_not_param)

		if verbose: print("# Searching in PubMed crossing terms no symbols:", end=' ')

		df_list = []
		self.pmid_dic = {}

		df = self.run_only_terms(connective, terms1, terms2, terms3, terms_not, force, verbose)

		return df


	def run_terms_loop_pathways(self, df_ptw_pubmed:pd.DataFrame, case:str, i_dfp:int, 
								connective:str, terms1:List, terms2:List, terms3:List, terms_not:List, 
								allow_repeat:bool=False, test:bool=False,
								show_warning:bool=True, force=False, verbose=False) -> pd.DataFrame:

		self.gem.set_gemini_root_sufix()

		dfa = df_ptw_pubmed[  pd.isnull(df_ptw_pubmed.term) | (df_ptw_pubmed.term == '')]
		if not dfa.empty:
			if show_warning: print("There are some empty terms in reacto-term table.")
			return None

		if not isinstance(terms1, list) or terms1 == []:
			print(">>> Define terms1 as list\n\n")
			raise Exception('stop')

		if not isinstance(terms2, list):
			print(">>> Define terms2 as list\n\n")
			raise Exception('stop')

		if not isinstance(terms3, list):
			print(">>> Define terms3 as list\n\n")
			raise Exception('stop')

		if not isinstance(self.terms_not, list):
			print(">>> Define terms2 as list - can be empty list.\n\n")
			raise Exception('stop')

		if not isinstance(self.connective_param, str) or self.connective_param == '':
			self.connective_param = 'AND'
			print(">>> Defining connective_param as 'AND'.")


		connective = deepcopy(connective)
		terms1	   = deepcopy(terms1)
		terms2	   = deepcopy(terms2)
		terms_not  = deepcopy(terms_not)

		print("# Searching in PubMed crossing terms and pathway terms; no symbols:")

		fname_case = self.fname_pubmed_case_iq_idfp_gender%(self.prefix, case, i_dfp, self.with_gender, self.suffix)
		fname_case = title_replace(fname_case)
		filename = os.path.join(self.root_pubmed_root, fname_case)

		if os.path.exists(filename) and not force:
			return pdreadcsv(fname_case, self.root_pubmed_root, verbose=verbose)

		df_list = []
		for i in range(len(df_ptw_pubmed)):
			row = df_ptw_pubmed.iloc[i]

			pathway_id	 = row.pathway_id
			pathway		 = row.pathway
			term_pathway = row.term

			self.pathway_id = pathway_id
			self.pathway	= pathway

			terms3 = term_pathway.split(' ')

			# print(">>>> #", pathway_id, f"'{pathway}'", self.terms1, self.terms2, self.terms3)
			
			self.no_repeats = terms2 == [] and not allow_repeat

			df = self.run_only_terms(connective, terms1, terms2, terms3, terms_not, test=test, force=force, verbose=verbose)

			if df is None or df.empty:
				if verbose: print(f" - no articles were found for {pathway_id} '{pathway}': '{connective}' '{terms1}' '{terms2}' '{terms3}'")
			else:
				if verbose: print(f" - found for for {pathway_id} '{pathway}': {connective} '{terms1}' '{terms2}' '{terms3}'")
				df_list.append(df)

			# print("\n------------------ end ----------------------\n")

		if df_list != []:
			df_pmid = pd.concat(df_list)

			df_pmid = df_pmid[ ~df_pmid.pmid.isna()]
			df_pmid.pmid = df_pmid.pmid.astype(int)

			df_pmid.index = np.arange(0, len(df_pmid))
		else:
			dic = {'pathway_id':[], 'pathway':[], 'symbol':[], 'synonyms':[], 'pmid':[], 'pub_date':[], 'title':[], 
				   'keywords':[], 'abstract':[], 'abreviation':[], 'authors':[], 'created_date':[], 'doc_type':[], 
				   'docid':[], 'journalTitle':[], 'language':[], 'cases':[], 'case_comparisons':[], 'terms':[],
				   'dates':[]}

			df_pmid = pd.DataFrame(dic)

		ret = pdwritecsv(df_pmid, fname_case, self.root_pubmed_root, verbose=verbose)

		return df_pmid


	def run_big_comparisons(self, choice:str, pathway_concept_name_list:List, pathway_concept_list:List,
							allow_repeat:bool=False,test:bool=False, force=False, verbose=False) -> pd.DataFrame:

		if not isinstance(self.terms1_param, list) or self.terms1_param == []:
			print(">>> Define terms1 as list\n\n")
			raise Exception('stop')

		if not isinstance(self.terms_not_param, list):
			print(">>> Define terms2 as list - can be empty list.\n\n")
			raise Exception('stop')

		if not isinstance(self.connective_param, str) or self.connective_param == '':
			self.connective_param = 'AND'
			print(">>> Defining connective_param as 'AND'.")



		if choice is None or choice == '' or choice == []:
			print(">>> Define choice (string)")
			raise Exception('stop')

		if not isinstance(pathway_concept_list, list) or pathway_concept_list == []:
			print(">>> Define pathway_concept_list as a list")
			raise Exception('stop')


		self.choice = choice

		self.connective = deepcopy(self.connective_param)
		self.terms1	 = deepcopy(self.terms1_param)
		self.terms_not  = deepcopy(self.terms_not_param)

		self.define_as_choice = False

		if verbose: print("# Searching in PubMed crossing terms no symbols:")

		lista_terms2 = terms2 + [[]]

		fname_choice = self.fname_no_symb_choice0%(self.prefix, choice, self.s_all_text_or_abstract)
		fname_choice = title_replace(fname_choice)
		filename = os.path.join(self.root_pubmed, fname_choice)

		if os.path.exists(filename) and not force:
			return pdreadcsv(fname_choice, self.root_pubmed, verbose=verbose)

		df_choice_list = []
		for i in range(len(pathway_concept_name_list)):
			self.concept_name = pathway_concept_name_list[i]

			fname_no_symb = self.fname_no_symb_concept_choice0%(self.prefix, self.concept_name, self.choice, self.s_all_text_or_abstract)
			fname_no_symb = title_replace(fname_no_symb)
			filename = os.path.join(self.root_pubmed, fname_no_symb)

			if os.path.exists(filename) and not force:
				df = pdreadcsv(fname_no_symb, self.root_pubmed, verbose=verbose)
				df_choice_list.append(df)
			else:
				concept_list = pathway_concept_list[i]

				if not isinstance(concept_list, list) or concept_list == []:
					print("Please, all pathway_concept_list terms must be lists")
					raise Exception('stop')

				df_list = []
				self.pmid_dic = {}
				for concept in concept_list:

					for terms2 in lista_terms2:
						if terms2  == 'AND' or terms2  == 'OR': continue

						# print(">>>> #", terms1, terms2, concept)
						terms2 = [terms2]
						self.no_repeats = terms2 == [] and not allow_repeat
						terms3 = [concept]

						self.concept = concept
						self.concept_list = [concept]

						df = self.run_only_terms(connective, terms1, terms2, terms3, terms_not, test=test, force=force, verbose=verbose)

						if df is not None and not df.empty:
							df_list.append(df)

				if df_list == []:
					print(f"No articles were found for '{terms1}' {self.connective} '{lista_terms2}' {self.connective} choice: '{self.choice}' concept_list: {concept_list}")
				else:
					dftot= pd.concat(df_list)
					print(f", found {len(dftot)} articles for '{self.choice}' and '{self.concept_name}'.")
					ret = pdwritecsv(dftot, fname_no_symb, self.root_pubmed, verbose=verbose)
					df_choice_list.append(dftot)

			# print("\n------------------ end ----------------------\n")

		if df_choice_list != []:
			df_choi = pd.concat(df_choice_list)
			df_choi.index = np.arange(0, len(df_choi))
			ret = pdwritecsv(df_choi, fname_choice, self.root_pubmed, verbose=verbose)
			print("------------------ end choice ----------------------\n")
		else:
			df_choi = None
			print("------------------ end choice -------- no data ----\n")

		return df_choi


	def run_only_terms(self, connective:str, terms1:List, terms2:List, terms3:List, terms_not:List, 
					   test:bool=False, force:bool=False, verbose:bool=False):

		if not isinstance(connective, str):
			print(">>> Define connective as a string like AND or OR")
			raise Exception('stop')

		if not isinstance(terms1, list):
			print(">>> Define terms1 as list it can be []")
			raise Exception('stop')

		if not isinstance(terms2, list):
			print(">>> Define terms2 as list it can be []")
			raise Exception('stop')

		if not isinstance(terms3, list):
			print(">>> Define terms3 as list it can be []")
			raise Exception('stop')

		if terms3 == [] or terms3 == [[]]:
			if terms2 == [] or terms2 == [[]]:
				terms = [[connective], terms1]
			else:
				terms = [[connective], terms1, terms2]
		else:
			if terms2 == [] or terms2 == [[]]:
				terms = [[connective], terms1, terms3]
			else:
				terms = [[connective], terms1, terms2, terms3]

		if terms_not is None or terms_not == [] or terms_not == [[]]:
			pass
		else:
			terms.append(terms_not)

		self.terms = terms

		if not isinstance(terms, list):
			raise Exception("Error run only_terms(): terms must be a list")

		if len(terms) == 0:
			raise Exception("Error run only_terms(): please at least one term")

		stri_term = str(terms)
		df_list=[]
		term_genes = []

		self.s_term2 = terms2[0] if terms2 != [] else 'None'
		self.s_term3 = terms3[0] if terms3 != [] else 'None'

		df, query = self.query_pubmed_terms(connective, terms, term_genes, test=test, verbose=verbose)

		if df is not None and not df.empty:
			df = pd.DataFrame(df)

			df['cases'] = 'no_cases'
			df['case_comparisons'] = 'no_case_comparisons'
			df['terms'] = stri_term
			df['dates'] = "%s to %s"%(self.inidate, self.enddate)

			print(" %d"%(len(df)), end='')
		else:
			print(" 0", end='')
			df = None

		return df


	def run_terms_symbols(self, dfs2:pd.DataFrame, connective:str, terms1:List, terms2:List, terms3:List, terms_not:List,
						  force=False, verbose=False):

		symbols = list(np.unique(dfs2.symbol))

		if not isinstance(connective, str) or connective == '':
			print(">>> Define connective as AND or OR")
			raise Exception('stop')

		if not isinstance(terms1, list):
			print(">>> Define terms1 as list it can be []")
			raise Exception('stop')

		if not isinstance(terms2, list):
			print(">>> Define terms2 as list it can be []")
			raise Exception('stop')

		if not isinstance(terms3, list):
			print(">>> Define terms3 as list it can be []")
			raise Exception('stop')

		if terms3 == []:
			if terms2 == []:
				terms = [[connective], terms1]
			else:
				terms = [[connective], terms1, terms2]
		else:
			if terms2 == []:
				terms = [[connective], terms1, terms3]
			else:
				terms = [[connective], terms1, terms2, terms3]

		if terms_not is None or terms_not == []:
			pass
		else:
			terms.append(terms_not)

		self.terms = terms

		if not isinstance(self.terms, list):
			raise Exception("Terms must be a list")

		if len(self.terms) == 0:
			raise Exception("Please at least one term")


		fname_no_symb = self.fname_summ_symbol_term0%(self.prefix, self.concept_suffix, self.choice, self.s_all_text_or_abstract)
		filename = os.path.join(self.root_pubmed, fname_no_symb)

		''' just for fun, to see the query term '''
		self._query = self.build_query(connective, terms, verbose=verbose)

		if os.path.exists(filename) and not force:
			return pdreadcsv(fname_no_symb, self.root_pubmed, verbose=verbose)


		# print(">>> terms", terms)
		# connect = " %s "%(self.connective)
		# stri_term = "; ".join([connect.join(term) for term in terms])
		stri_term = str(terms)
		dftot = None;  df_list=[]

		for symbol in symbols:

			if symbol != None:
				symbol_and_synonyms = self.gene.search_symbol_and_synonyms(symbol)

				if len(remove_synonym_list) > 0:
					goods = [False if symb_syn in remove_synonym_list else True for symb_syn in symbol_and_synonyms]
					symbol_and_synonyms = symbol_and_synonyms[goods]

				term_genes = [['OR'] + symbol_and_synonyms]
			else:
				term_genes = []
				symbol_and_synonyms = 'no_gene_filtered'

			fname_symbol = self.fname_symbol0%(self.prefix, self.concept_suffix, self.choice, symbol, self.s_all_text_or_abstract)

			df, query = self.query_pubmed_terms(connective, terms, term_genes, fname=fname_symbol, verbose=verbose)

			dfs3 = dfs2[dfs2.symbol == symbol]

			cases = list(dfs3.case0.unique()) + list(dfs3.case1.unique())
			cases = np.unique(cases)

			case_comparisons = np.unique(dfs3.cases)

			if df is not None and not df.empty:
				df = pd.DataFrame(df)

				df['cases'] = str(cases)
				df['case_comparisons'] = str(case_comparisons)
				df['terms'] = stri_term
				df['dates'] = "%s to %s"%(self.inidate, self.enddate)
				df_list.append(df)

				print("Found %d '%s'"%(len(df), symbol_and_synonyms), end='; ')
			else:
				# print("Nothing found for '%s'"%(symbol))
				# print(". '%s'"%(symbol), end=' ')
				print(".",end='')

		if df_list == []:
			print("No articles were found")
			dftot = None
		else:
			dftot= pd.concat(df_list)
			print("Found for %d articles"%(len(dftot)))
			ret = pdwritecsv(dftot, fname_no_symb, self.root_pubmed, verbose=True)

		print("\n------------------ end ----------------------\n")

		return dftot


	def try_entrez_query(self, query):
		'''
		download:
			html from Pubmed
			pdf form PMC
		'''
		try:
			handle = Entrez.esearch(db="pubmed", term=query, retmax=self.retmax)
			record = Entrez.read(handle)

			if int(record['Count']) == 0:
				return True, []

			ret = True
		except:
			print("Could not complete HTTP esearch: query '%s' retmax %d"%(query, self.retmax))
			return False, []

		ids = record['IdList']

		try:
			handle = Entrez.efetch(db="pubmed", id=ids, rettype="medline", retmode="text")
			records = Medline.parse(handle)
		except:
			print("Could not complete HTTP efecth reading IDs: query '%s' retmax %d"%(query, self.retmax))
			return False, []

		return True, records

	def search_on_pubmed(self, query:str, gene_list:List, test:bool=False, verbose:bool=False) -> pd.DataFrame:
		'''
		if test: echo query and gene_list

		get all atributes related to the paper
		'pmid', 'pub_date', 'title', 'keywords', 'abstract', 'abreviation', 'authors',
		'created_date', 'doc_type', 'docid', 'journalTitle'
		
		MEDLINE/PubMed Data Element (Field) Descriptions
		https://www.nlm.nih.gov/bsd/mms/medlineelements.html

		gene symbols: 
			This field contains the "symbol" or abbreviated form of gene names as reported in the literature. This element resides in records processed at NLM from 1991 through 1995.
			NLM entered the symbols used by authors; there was no authority list or effort to standardize the data.
		'''
		if verbose: print(">>> search_on_pubmed():\n'%s'\nretmax: %d\ngene list:%s"%(query, self.retmax, ";".join(gene_list) ) )

		if test:
			print(query)
			if len(gene_list) > 0: print(gene_list)
			print("")
			return None

		ret, records = self.try_entrez_query(query)

		if not ret:
			if self.sleep_entrez is not None and isinstance(self.sleep_entrez, list):
				for tm in self.sleep_entrez:
					print("Http error: time %d s"%(tm))
					time.sleep(tm)

					ret, records = self.try_entrez_query(query)
					if ret: break

		if not ret:
			print("Problems with query entrez/http")
			return None

		dic = {}
		if gene_list == [] or gene_list is None:
			gene_list = None
		else:
			if gene_list[0] == 'AND' or gene_list[0] == 'OR':
				gene_list = gene_list[1: ]

		# print(">>> gene_list", gene_list)
		i = -1
		for record in records:
			pmid  = record.get("PMID", None)
			if pmid is None:
				print("Warning: pmid is none!")
				self.record = record
				print(record)
				raise Exception('stop pmid is none')

			if self.no_repeats:
				''' if a general search: if pmid already found do not save it again.'''
				if self.pmid_dic.get(pmid, None) is not None:
					continue

			self.pmid_dic[pmid] = 1  # self.pmid_dic.get(pmid, 0) + 1

			title	= record.get("TI", None)
			abstract = record.get("AB", None)
			keyWords = record.get("OT", None)

			if isinstance(keyWords, list):
				keyWords = "; ".join(keyWords)
			else:
				if keyWords is not None:
					keyWords = str(keyWords)

			pub_date = record.get("DP", None)
			journalTitle = record.get("JT", None)
			ta = record.get("TA", None)
			language = record.get("LA", None)
			pt = record.get("PT", None)
			CRDT = record.get("CRDT", None)
			authors = record.get("AU", None)
			docID = record.get("AID", None)

			i += 1
			dic[i] = {}

			if self.concept is None:
				dic[i]['pathway_id'] = self.pathway_id
				dic[i]['pathway'] = self.pathway
			else:
				dic[i]['concept'] = self.concept
				dic[i]['concept_list'] = str(self.concept_list)
				dic[i]['choice'] = self.choice if self.define_as_choice else self.s_term2

			if gene_list is None:
				dic[i]['symbol']	= None
				dic[i]['synonyms'] = None
			else:
				dic[i]['symbol']	= gene_list[0]		if self.define_as_choice else None
				dic[i]['synonyms'] = str(gene_list[1:]) if self.define_as_choice else None

			dic[i]['pmid'] = pmid
			dic[i]['title'] = title
			dic[i]['keywords'] = keyWords
			dic[i]['gene_list'] = gene_list
			dic[i]['journalTitle'] = journalTitle
			dic[i]['abreviation'] = ta
			dic[i]['language'] = language
			dic[i]['doc_type'] = pt
			dic[i]['created_date'] = CRDT
			dic[i]['pub_date'] = pub_date
			dic[i]['authors'] = authors
			dic[i]['docid'] = docID
			dic[i]['abstract'] = abstract

		if dic == {}:
			return None

		df = pd.DataFrame.from_dict(dic).T
		df = df.sort_values(by=['pub_date', 'pmid'], ascending=[False, True])
		df.index = np.arange(0, len(df))

		return df


	def query_pubmed_terms(self, connective:str, terms:List, term_genes:List=[], test:bool=False, verbose:bool=False):

		query = self.build_query(connective, terms, term_genes, verbose=verbose)

		if self.verbose_query: print(">>> query:", query)

		if term_genes == []:
			gene_list = []
		else:
			if term_genes[0] == ['AND'] or term_genes[0] == ['OR']:
				gene_list = term_genes[1]
			else:
				gene_list = term_genes[0]

		df = self.search_on_pubmed(query, gene_list, test=test)
		if df is None or df.empty:
			return None, None

		self.df_query_pubmed_terms = df

		if self.concept is None:
			cols = ['pathway_id', 'pathway', 'symbol', 'synonyms', 'pmid', 'pub_date', 'title', 'keywords',
					'abstract', 'abreviation', 'authors',  'created_date', 'doc_type', 'docid',
					'journalTitle', 'language', ]
		else:
			cols = ['concept', 'concept_list', 'choice', 'symbol', 'synonyms', 'pmid', 'pub_date', 'title', 'keywords',
					'abstract', 'abreviation', 'authors',  'created_date', 'doc_type', 'docid',
					'journalTitle', 'language', ]

		df = df[cols]

		return df, query


	def open_spacy(self):

		if self.sp is None:
			self.sp = spacy.load('en_core_web_sm')
			self.stemmer = PorterStemmer()

	def calc_lemmatization(self, text:str, filter_adjectives:bool=False, verbose:bool=False) -> (str, str, List):

		self.open_spacy()

		text = self.calc_lemmatization_initial(text, filter_adjectives)

		if verbose: print("After calc lemma first len =", len(text))

		text, stem_text, lista_symb = self.calc_lemmatization_final(text, verbose=verbose)

		return text, stem_text, lista_symb

	def calc_lemmatization_initial(self, text:str, filter_adjectives:bool=False) -> str:

		tokens = self.sp(text)
		words = []

		for token in tokens:
			if token.pos_ in self.unsel_POS:
				continue

			if filter_adjectives and token.pos_ == 'ADJ':
				if token.dep_  in self.unsel_adjectives:
					continue

			word = token.text

			''' --- removing stop words ---'''
			if word.lower() in self.stop_words: continue

			words.append(word)

		return " ".join(words)


	def calc_lemmatization_final(self, text:str, verbose:bool=False) -> (str, str, List):

		text = self.replace_regular_expression(text, verbose=verbose)
		words = []; wstemms = []; lista_symb = []

		for word in text.split(' '):
			''' --- removing bad wstemm ---'''
			wstemm = self.stemmer.stem(word)
			if wstemm in self.stop_stems: continue

			if isfloat(word): continue

			words.append(word)
			wstemms.append(wstemm)

			'''----- finding possible genes/proteins -----------------------'''
			word_symb = word

			if word_symb in self.gene.ambiguous_symbol_dic.keys():
				'''---
					ambiguous symbols: change any term by a symbol
				---'''
				word_symb = self.gene.ambiguous_symbol_dic[word_symb]

			'''------------ do nothing if a drop symbol --------------------'''
			if word_symb not in self.gene.drop_symbs_dic.keys():

				'''---- any annotated exception? ---'''
				word_symb = self.gene.replace_synonym_to_symbol(word_symb)

				'''----- is a good symbol?? -----'''
				if self.gene.is_mygene_symbol(word_symb):
					lista_symb.append(word_symb)
				elif self.gene.is_mygene_synonym(word_symb):
					word_symb = self.gene.find_mygene_synonym_to_symbol(word_symb)
					lista_symb.append(word_symb)

			'''----- finding possible genes/proteins --- end ---------------'''

		'''--- never use np.unique() - we must do statistics --------------'''
		text = " ".join(words)
		stem_text = " ".join(wstemms)

		if lista_symb != []:
			lista_symb = list(np.unique(lista_symb))

			if 'SDS' in lista_symb:
				ret = re.search(self.pattern_SDS, text)
				if ret is not None:
					lista_symb.remove('SDS')

		return text, stem_text, lista_symb

	def replace_doi(self, text, pattern = 'doi:'):
		ret1 = re.search(pattern, text)
		ret2 = re.search(pattern.upper(), text)

		if not ret1 and not ret2:
			return text

		if ret1:
			ret = ret1
		else:
			pattern = 'DOI:'
			ret = ret2

		while(True):
			ini, end = ret.span()
			stri0 = text[:ini].strip()
			stri1 = text[end:].strip()

			mat = stri1.split(' ')
			if mat != []:
				stri1 = ' '.join(mat[1:])

			text = stri0 + ' ' + stri1

			ret = re.search(pattern, text)

			if not ret: break

		return text

	''' old replace_gene_terms '''
	def replace_regular_expression(self, text:str, verbose:bool=False) -> str:

		text = re.sub(r'Kallikrein-|kallikrein-', r"KLK", text)
		text = re.sub(r'reactive oxygen species|Reactive Oxygen Species|Reactive oxygen species', r"ROS", text)
		text = re.sub(r'angiotensin converting enzyme 2|angiotensin converting enzyme-2|ACE-2', r"ACE2", text)
		text = re.sub(r'nadph oxidase|NADPH oxidase|NADPH Oxidase', r"NOX", text)
		text = re.sub(r'd dimer|D-DIMER|d-dimer', r"ddimer", text)
		text = re.sub(r'tissue factor|Tissue Factor|Tissue Factor', r"F8", text)
		text = re.sub(r'sars-cov-2|SARS-CoV-2', r'SARSCOV2', text)
		text = re.sub(r'von Willebrand factor|vWF', r"VWF", text)
		text = re.sub(r'PI3K-|Phosphatidylinositol 3-Kinase', r"PI3K", text)
		text = re.sub(r',\s+and|\s+and', r"", text)
		text = re.sub(r'\shsa', ' human', text)
		text = re.sub(r'NF-kB', 'NFKB', text)
		''' splicy tokenization splits G-CSF '''
		text = re.sub(r'G CSF', 'GCSF', text)

		latin_chars = ['A','B','D','E', 'G']

		tuples = [("A", ['Alpha', 'ALPHA', 'alpha', 'α']), ("B", ['Beta',  'BETA',  'beta', 'β']),
				  ("D", ['Delta', 'DELTA', 'delta', 'δ']), ("E", ['Epsilon', 'EPSILON', 'epsilon', 'ε']),
				  ("G", ['Gamma', 'GAMMA', 'gamma', 'γ']), ]

		for latin_char, terms in tuples:
			for term in terms:
				lista = re.findall(rf'-\d+{term}', text)
				if isinstance(lista, list):
					for word in lista:
						text = text.replace(word, word.replace(term, latin_char).replace('-',''))

				lista = re.findall(rf'-{term}|\s{term}|[A-z]{term}', text)
				if isinstance(lista, list):
					for word in lista:
						text = text.replace(word, word.replace(term, latin_char).replace('-','').replace(' ',''))


		''' IGL-XX G --> spacy splits G '''
		for term in latin_chars:
			lista = re.findall(r"[A-z]{2,}-\d{1,}\s%s"%(term), text)
			if isinstance(lista, list):
				for word in lista:
					text = text.replace(word, word.replace('-', '').replace(' ','') + ' ')

		''' CCL-XXX '''
		lista = re.findall(r"[A-z]{2,}-\d{1,}", text)
		if isinstance(lista, list):
			for word in lista:
				text = text.replace(word, word.replace('-', ''))

		''' if last ponctuation remove them'''
		text = re.sub(r"\.|;|,|-|\[|]|\(|\)|'|☨|®", ' ', text)
		''' spaces '''
		text = re.sub(r'\s+', ' ', text)

		''' There are no more commas and dashes --> MMP-3 --> MMP<s>3 '''
		text = self.replace_factors(text, verbose=verbose)
		text = self.remove_digits(text)
		text = text.strip()

		return text

	def replace_factors(self, text:str, verbose:bool=False) -> str:

		'''---- look for Factors, Groups - Meaningful Words .... '''
		lista = text.split(' ')
		if verbose: print("\n\n START REP FACTORS", lista, "\n\n")
		new_lista = []

		iter_list = iter(range(len(lista)))

		for j in iter_list:
			word = lista[j]

			if len(word) > 0 and word[-1] == ',': word = word[:-1]
			if len(word) > 0 and word[0]  == '-': word = word[1:]
			if word == '': continue

			''' Factor3 append, Factor no'''
			if word not in self.gene.dic_gene_mnem.keys(): new_lista.append(word)

			if not isint(word) and isint(word[-1]):
				word = re.sub(r'\d+', '', word)

			''' is in Factor, Metaloproteinase, Caspase, etc.'''
			if word in self.gene.dic_gene_mnem.keys():
				inum = 0
				_prefix = self.gene.dic_gene_mnem[word]

				while(True):
					try:
						inum += 1
						next_word = lista[j+inum]
						if len(next_word) > 0 and next_word[-1] == ',': next_word = next_word[:-1]
						if len(next_word) > 0 and next_word[0]  == '-': next_word = next_word[1:]
					except:
						next_word = ''

					if next_word != '':
						if isint(next_word):
							new_lista.append(_prefix+next_word)
							continue
						if next_word in self.gene.factor_roman_nums:
							new_lista.append(_prefix+self.gene.dic_roman[next_word])
							continue

					''' if next word is not a number '''
					if inum > 1:
						for _ in range(inum-1):
							try:
								next(iter_list)
							except:
								pass
					break


		'''---- look for genes .... '''

		text = ' '.join([x for x in new_lista if x != ''])
		lista = text.split(' ')

		new_lista = []
		iter_list = iter(range(len(lista)))

		if verbose: print("\n\n AFTER REP FACTORS", lista, "\n\n")

		for j in iter_list:
			word = lista[j]
			''' MMP3 append, MMP no'''
			if word not in self.gene.lista_gene_many: new_lista.append(word)

			if len(word) > 0 and word[-1] == ',': word = word[:-1]
			if len(word) > 0 and word[0] == '-': word = word[1:]
			if word == '': continue

			if not isint(word) and isint(word[-1]):
				word = re.sub(r'\d+', '', word)

			if word in self.gene.lista_gene_many:
				inum = 0
				_prefix = word

				while(True):
					try:
						inum += 1
						next_word = lista[j+inum]
						# print('>> NEXT_WORD', inum, next_word)
						if len(next_word) > 0 and next_word[-1] == ',': next_word = next_word[:-1]
						if len(next_word) > 0 and next_word[0]  == '-': next_word = next_word[1:]
					except:
						print(">>> ERROR next word")
						next_word = ''

					if next_word != '':
						if isint(next_word):
							# print('>> next nums', inum, next_word)
							new_lista.append(_prefix+next_word)
							continue

						if next_word in self.gene.factor_roman_nums:
							# print('>> next roman', inum, next_word)
							new_lista.append(_prefix+self.gene.dic_roman[next_word])
							continue

					''' if next word is not a number '''
					if inum > 1:
						for _ in range(inum-1):
							try:
								next(iter_list)
							except:
								pass
					break

		''' no empty words '''
		text = ' '.join([x for x in new_lista if x != ''])
		''' fix miss commas '''
		text = re.sub(r' ,', ',', text)

		return text

	def replace_factors_DEPRECATED(self, text:str, verbose:bool=False) -> str:
		lista = text.split(' ')
		if verbose: print("\nreplace_factors() len =", len(text), "\n")
		new_lista = []

		iter_list = iter(range(len(lista)))

		for j in iter_list:
			word = lista[j]
			if len(word) > 0 and word[-1] == ',': word = word[:-1]
			if len(word) > 0 and word[0]  == '-': word = word[1:]
			if word == '': continue

			word0 = word

			if not isint(word) and isint(word[-1]):
				previous_word = word
				word = re.sub(r'\d+', '', word)
				previous_word = '' if previous_word == word else previous_word
			else:
				previous_word = ''

			''' is in Factor, Metaloproteinase, Caspase, etc.'''
			if word in self.gene.dic_gene_mnem.keys():
				inum = 0
				_prefix = self.gene.dic_gene_mnem[word]

				while(True):
					try:
						inum += 1
						next_word = lista[j+inum]
						if len(next_word) > 0 and next_word[-1] == ',': next_word = next_word[:-1]
						if len(next_word) > 0 and next_word[0]  == '-': next_word = next_word[1:]
					except:
						next_word = ''

					if next_word != '':
						if next_word in self.gene.factor_nums:
							# print(j, inum, 'factor_nums', next_word)
							if previous_word != '':
								new_lista.append(previous_word)
								previous_word = ''
							new_lista.append(_prefix+next_word)
							continue
						if next_word in self.gene.factor_roman_nums:
							# print(j, inum, 'dic_roman', next_word)
							if previous_word != '':
								new_lista.append(previous_word)
								previous_word = ''
							new_lista.append(_prefix+self.gene.dic_roman[next_word])
							continue

					if previous_word != '':
						new_lista.append(previous_word)
						previous_word = ''

					''' if next word is not a number '''
					if inum > 1:
						for _ in range(inum-1):
							try:
								next(iter_list)
							except:
								pass
					break
			else:
				# print(f"not a factor: '{previous_word}' -> '{word}'")
				new_lista.append(word0)

		'''---- the other way around '''

		text = ' '.join([x for x in new_lista if x != ''])
		lista = text.split(' ')

		new_lista = []
		iter_list = iter(range(len(lista)))

		if verbose: print("\nafter replace_factors() len =", len(text), "\n")

		for j in iter_list:
			word = lista[j]
			if len(word) > 0 and word[-1] == ',': word = word[:-1]
			if len(word) > 0 and word[0] == '-': word = word[1:]
			if word == '': continue

			word0 = word

			if not isint(word) and isint(word[-1]):
				previous_word = word
				word = re.sub(r'\d+', '', word)
				previous_word = '' if previous_word == word else previous_word
			else:
				previous_word = ''

			if word in self.gene.lista_gene_many:
				inum = 0
				_prefix = word

				while(True):
					try:
						inum += 1
						next_word = lista[j+inum]
						if len(next_word) > 0 and next_word[-1] == ',': next_word = next_word[:-1]
						if len(next_word) > 0 and next_word[0]  == '-': next_word = next_word[1:]
					except:
						next_word = ''

					if next_word != '':
						if next_word in self.gene.factor_nums:
							if previous_word != '':
								new_lista.append(previous_word)
								previous_word = ''
							new_lista.append(_prefix+next_word)
							continue

						if next_word in self.gene.factor_roman_nums:
							if previous_word != '':
								new_lista.append(previous_word)
								previous_word = ''
							new_lista.append(_prefix+self.gene.dic_roman[next_word])
							continue

					if previous_word != '':
						new_lista.append(previous_word)
						previous_word = ''

					''' if next word is not a number '''
					if inum > 1:
						for _ in range(inum-1):
							try:
								next(iter_list)
							except:
								pass
					break
			else:
				new_lista.append(word0)

		text = ' '.join([x for x in new_lista if x != ''])
		text = re.sub(r' ,', ',', text)

		return text



	def replace_et_al(self, text, pattern = 'et al.'):
		ret = re.search(pattern, text)
		if not ret:
			return text

		while(True):
			ini, end = ret.span()
			stri0 = text[:ini].strip()
			stri1 = text[end:].strip()

			mat = stri0.split(' ')
			if mat != []:
				if isfloat(mat[-1]):
					stri0 = ' '.join(mat[:-2])
				else:
					stri0 = ' '.join(mat[:-1])

			text = stri0 + ' ' + stri1

			ret = re.search(pattern, text)

			if not ret: break

		return text


	def remove_digits(self, text):
		mat = [x for x in text.split(' ') if not isfloat(x)]
		return ' '.join(mat)


	def build_query(self, connective:str, terms:List, term_genes:List=[], is_title_abstract:bool=True, verbose=False):

		self.is_title_abstract = is_title_abstract
		self.gene_quote='"'
		'''
		[  ['AND'],
			['AAAA',	 'BBBBB'],
			['OR',		'CCCC', 'CCCCS'],
			['AND',	  'VVVV', 'UUUU']]
			['AND NOT ', 'CANCER', 'DIABETIS']]
		'''
		queryTerms = ""; queryGenes = ""
		reserved_term = 'Title/Abstract' if is_title_abstract else 'Text Word'

		'''----------------- terms / concepts / functions ----------------------'''
		connective = deepcopy(connective)

		if len(terms) > 0:
			queries = ''; innerQueries = ""; innerConn = "AND"

			for innerTerms in terms:
				if innerTerms == []:
					continue

				if len(innerTerms) == 1:
					term = innerTerms[0]
					if term == "AND" or term == "OR":
						connective = term
						continue

				flagNot = False; notQueries=[]
				for term in innerTerms:

					if term == "AND" or term == "OR":
						innerConn = term
						continue

					if term == "AND NOT" or term == "NOT":
						flagNot = True
						innerConn = 'AND'
						continue

					if flagNot:
						if len(term) > 3:
						  notQueries.append('%s%s*%s[%s]'%(self.text_quote, term, self.text_quote, reserved_term) )
						else:
						  notQueries.append('%s%s%s[%s]'%(self.text_quote, term, self.text_quote, reserved_term) )
					else:
						if innerQueries == "":
							if len(term) > 3:
								innerQueries = '(%s%s*%s[%s]'%(self.text_quote, term, self.text_quote, reserved_term)
							else:
								innerQueries = '(%s%s%s[%s]'%(self.text_quote, term, self.text_quote, reserved_term)
						else:
							if len(term) > 3:
								innerQueries = '%s %s %s%s*%s[%s]'%\
										 (innerQueries, innerConn, self.text_quote, term, self.text_quote, reserved_term)
							else:
								innerQueries = '%s %s %s%s%s[%s]'%\
										 (innerQueries, innerConn, self.text_quote, term, self.text_quote, reserved_term)
				innerQueries += ')'

				if len(notQueries) == 0:
					if queries != "":
						if innerQueries != '': queries = "%s %s %s"%(queries, connective, innerQueries)
					else:
						queries = innerQueries
				else:
					if queries == "":
						queries = "NOT %s"%(notQueries[0])
						for k in range(1,len(notQueries)):
							queries = "%s NOT %s"%(queries, notQueries[k])
					else:
						for k in range(len(notQueries)):
							queries = "%s NOT %s"%(queries, notQueries[k])

				innerQueries = ""; innerConn = "AND"; flagNot = False

			queryTerms = queries

		'''------------ term_genes ------------------'''
		if term_genes is not None and len(term_genes) > 0:
			queries = ''; connective = 'AND'
			innerQueries = ""; innerConn = "AND"

			for innerGenes in term_genes:
				if innerGenes == []:
					continue

				if len(innerGenes) == 1:
					gene = innerGenes[0]
					if gene == "AND" or gene == "OR":
						connective = gene
						continue

				flagNot = False; notQueries=[]
				for gene in innerGenes:
					if gene == "AND" or gene == "OR":
						innerConn = gene
						continue

					if gene == "AND NOT" or gene == "NOT":
						flagNot = True
						innerConn = 'AND'
						continue

					if flagNot:
						notQueries.append('%s%s%s[%s]'%(self.gene_quote, gene,
											self.gene_quote, reserved_term))
					else:
						if innerQueries == "":
							innerQueries = '(%s%s%s[%s]'%(self.gene_quote, gene,
											self.gene_quote, reserved_term)
						else:
							innerQueries = '%s %s %s%s%s[%s]'%(innerQueries, innerConn,
											self.gene_quote, gene, self.gene_quote, reserved_term)

				if len(notQueries) == 0:
					if queries != "":
						queries = "%s %s %s"%(queries, connective, innerQueries)
					else:
						queries = innerQueries
				else:
					if queries == "":
						queries = "NOT %s"%(notQueries[0])
						for k in range(1,len(notQueries)):
							queries = "%s NOT %s"%(queries, notQueries[k])
					else:
						for k in range(len(notQueries)):
							queries = "%s NOT %s"%(queries, notQueries[k])

				innerQueries = ""; innerConn = "AND"; flagNot = False

			queryGenes = queries + ')'

		# print(">>>> queryGenes:", queryGenes)

		if queryTerms == "":
			query = '%s AND ("%s"[Date - Publication] : "%s"[Date - Publication])'%(queryGenes, self.inidate, self.enddate)
		elif queryGenes == "":
			query = '%s AND ("%s"[Date - Publication] : "%s"[Date - Publication])'%(queryTerms, self.inidate, self.enddate)
		else:
			query = '%s AND %s AND ("%s"[Date - Publication] : "%s"[Date - Publication])'%(queryTerms, queryGenes, self.inidate, self.enddate)

		if verbose: print("query:", query)

		return query.strip()


	def open_uniprot_conv(self, verbose:bool=False):

		filename = os.path.join(self.root_refseq, self.fname_conv_uniprot)
		if not os.path.exists(filename):
			print("Uniprot table does not exists: '%s'"%(filename))
			return pd.DataFrame({'A' : []})

		return pdreadcsv(self.fname_conv_uniprot, self.root_refseq, verbose=verbose)


	def runcmd(self, cmd):

		process = subprocess.Popen(
			cmd,
			stdout = subprocess.PIPE,
			stderr = subprocess.PIPE,
			text = True,
			shell = True
		)
		std_out, std_err = process.communicate()
		'''print(std_out.strip(), std_err)'''

		return std_out.strip(), std_err

	def pdf_download(self, i, lim_set):
		'''
		function used in parallel process

		call: ../src/down_pdf_pmid.sh
		must be executable
		chmod +x *.sh

		must set environment variable 'root_pdf':
		os.environ["root_pdf"] = self.root_pdf + '/'
		'''

		lim0, lim1 = lim_set
		lista = self.lista_not_dwn[lim0 : lim1]

		for j in range(len(lista)):
			pmid = lista[j]
			if j%100 == 0:  print(".", end='')

			cmd = f'{self.fname_exec_full} {pmid}'
			# print(lim0, j, cmd)
			ret1, ret2 = self.runcmd(cmd)
			# print(lim0, j, ret1, ret2)

			fname = os.path.join(self.root_pdf, f'{pmid}.pdf')
			if os.path.exists(fname):
				if os.path.getsize(fname) == 0:
					fname_bad = fname.replace('.pdf', '.bad')
					os.rename(fname, fname_bad)
			else:
				fname = os.path.join(self.root_pdf, f'{pmid}.html')
				if os.path.exists(fname):
					if os.path.getsize(fname) == 0:
						fname_bad = fname.replace('.html', '.bad')
						os.rename(fname, fname_bad)

		return (i)

	'''
		see: https://stackoverflow.com/questions/6648493/how-to-open-a-file-for-both-reading-and-writing
	'''
	def create_down_pdf_pmid_shell(self, i, force=False):

		fname_new = self.fname_exec.replace('.sh', f'{i}.sh')
		dest_file = os.path.join(self.root_sh, fname_new)

		if os.path.exists(dest_file) and not force:
			return fname_new

		src_file = self.fname_exec_full
		shutil.copyfile(src_file, dest_file)

		text = ''

		try:
			h = open(dest_file, mode="r+")

			while True:
				line = h.readline()
				if not line : break

				line = line.replace('pmid', f'pmid{i}')

				text += line
		except:
			print(f"Error: while reading {dest_file}")
			raise Exception('stop writing shell file.')
		finally:
			h.seek(0)
			h.write(text)
			h.truncate()

			h.close()

		return fname_new

	def open_df_pubmed_curation(self, final_version:bool=False, verbose:bool=False) -> pd.DataFrame:
		fname = self.fname_curation0%(self.concept, self.inidate, self.enddate, self.s_all_text_or_abstract)
		fname = title_replace(fname)

		if final_version: fname = fname.replace('.tsv', '_final.tsv')

		filename = os.path.join(self.root_curation, fname)
		if not os.path.exists(filename):
			self.dfpub_curation = None
			print(f"There is no pubmed curation file: '{filename}'")
			return None

		dfa = pdreadcsv(fname, self.root_curation, verbose=verbose)
		dfa.pmid = dfa.pmid.astype(int)

		self.dfpub_curation = dfa

		return dfa

	def open_df_pubmed_summary(self, verbose:bool=False) -> pd.DataFrame:
		fname = self.fname_pubmed_no_symb0%(self.prefix, self.s_all_text_or_abstract)
		fname = title_replace(fname)

		filename = os.path.join(self.root_pubmed, fname)
		if not os.path.exists(filename):
			self.dfpub_nosymb = None
			print(f"There is no pubmed summary file: '{filename}'")
			return None

		dfa = pdreadcsv(fname, self.root_pubmed, verbose=verbose)
		dfa.pmid = dfa.pmid.astype(int)

		self.dfpub_nosymb = dfa

		return dfa

	def open_df_pubmed_symbol_summary(self, all_pdf_html:str='all', force:bool=False, verbose:bool=False) -> pd.DataFrame:
		self.dfpub0 = None
		self.dfpub_symb, self.pubmed_genes = None, None

		fname1, _, _ = self.prepare_names_token_and_lemma(all_pdf_html)
		dfpub0 = pdreadcsv(fname1, self.root_pubmed)

		if dfpub0 is None or dfpub0.empty:
			print("Warning: probably 'calc lemmatization_and_symbols()' must been run.")
			self.dfpub0 = dfpub0
			return dfpub0

		dfpub0.pmid = dfpub0.pmid.astype(int)
		self.dfpub0 = dfpub0

		fname2 = self.fname_pubmed_hugo_symb0%(self.prefix, self.s_all_text_or_abstract)

		filename2 = os.path.join(self.root_pubmed, fname2)
		if os.path.exists(filename2) and not force:
			dfpub_symb = pdreadcsv(fname2, self.root_pubmed)

			if dfpub_symb is not None and not dfpub_symb.empty:
				self.dfpub_symb = dfpub_symb
				print(f'There are {len(dfpub0)} pubmed articles and {len(dfpub_symb)} articles with symbols [dfpub_symb].')
				return dfpub0

		cols = ['concept', 'choice', 'pmid', 'ftype', 'has_text', 'symbols', 'title', 'keywords']
		dfpub_symb = dfpub0[ (~ dfpub0.symbols.isna()) & (dfpub0.symbols != '')][cols].copy()

		if dfpub_symb.empty:
			dfpub_symb['symbols_new'] = None
			dfpub_symb['n_symbols'] = None
		else:
			symbols_new = []
			n_symbols_new = []

			for symbols in dfpub_symb.symbols:
				mat = eval(symbols)
				''' ----------- rever todo ------------------'''
				symbols_new.append(mat)
				n_symbols_new.append(len(mat))

			dfpub_symb['symbols_new'] = symbols_new
			dfpub_symb['n_symbols'] = n_symbols_new

		cols = ['pmid', 'n_symbols', 'symbols_new', 'symbols', 'concept', 'choice', 'has_text', 'title', 'keywords']
		dfpub_symb = dfpub_symb[cols]

		if not dfpub_symb.empty:
			ret = pdwritecsv(dfpub_symb, fname2, self.root_pubmed)

		self.dfpub_symb = dfpub_symb
		self.retrieve_all_pubmed_genes()

		n_in = len(dfpub_symb[dfpub_symb['n_symbols'] > 0])
		print(f'There are {len(dfpub_symb)} pubmed articles which {n_in} have symbols.')

		return dfpub0

	def retrieve_all_pubmed_genes(self):
		pubmed_genes = []
		for i in range(len(self.dfpub_symb)):
			symbols = self.dfpub_symb.iloc[i].symbols

			if isinstance(symbols, float):
				continue

			if isinstance(symbols, str):
				try:
					symbols = eval(symbols)
				except:
					print(i, symbols, type(symbols))
					continue

			#if symbols != [nan]:
			symbols = [x for x in symbols if isinstance(x, str)]
			if symbols != []:
				pubmed_genes += symbols

		pubmed_genes = np.unique(pubmed_genes)
		self.pubmed_genes = pubmed_genes


	def find_symbols_build_perc_table(self, force=False, verbose=False):
		self.dfsymb_perc = None

		if self.dfpub_symb is None:
			_ = self.open_df_pubmed_symbol_summary()

			if self.dfpub_symb is None or self.dfpub_symb.empty:
				return None

		fname = self.fname_perc_symb0%(self.prefix, self.s_all_text_or_abstract)
		filename = os.path.join(self.root_pubmed, fname)

		if os.path.exists(filename) and not force:
			dfsymb_perc = pdreadcsv(fname, self.root_pubmed, verbose=verbose)
			self.dfsymb_perc = dfsymb_perc
			return dfsymb_perc

		dic = {}

		lista = [x if isinstance(x, list) else eval(x) for x in self.dfpub_symb.symbols]

		for symbols in lista:
			for symb in symbols:
				dic[symb] = dic.get(symb, 0) + 1

		dfsymb_perc = pd.DataFrame({'symbol': list(dic.keys()), 'n': dic.values()})
		dfsymb_perc = dfsymb_perc.sort_values('n', ascending=False)
		dfsymb_perc.index = np.arange(0, len(dfsymb_perc))
		total = dfsymb_perc.n.sum()
		dfsymb_perc['perc'] = dfsymb_perc.n / total

		cumm = 0; mat=[]
		for i in range(len(dfsymb_perc)-1, -1, -1):
			cumm += dfsymb_perc.iloc[i].perc
			mat.append(cumm)

		mat.reverse()
		dfsymb_perc['cumm'] = mat

		print(f"There are {len(dfsymb_perc)} Hugo's symbols")
		self.dfsymb_perc = dfsymb_perc

		ret = pdwritecsv(dfsymb_perc, fname, self.root_pubmed, verbose=True)

		return dfsymb_perc


	def find_symbols_build_perc_table_from_subcluster(self, ids:List, term_subcluster:str, force:bool=False, verbose:bool=False) -> pd.DataFrame:
		self.dfsymb_perc = None

		if self.dfpub_symb is None:
			_ = self.open_df_pubmed_symbol_summary()

			if self.dfpub_symb is None or self.dfpub_symb.empty:
				return None

		fname = self.fname_perc_symb_subcluster0%(self.prefix, self.s_all_text_or_abstract, term_subcluster)
		filename = os.path.join(self.root_pubmed, fname)

		if os.path.exists(filename) and not force:
			dfsymb_perc = pdreadcsv(fname, self.root_pubmed, verbose=verbose)
			self.dfsymb_perc_subclust = dfsymb_perc
			return dfsymb_perc

		df2 = self.dfpub_symb[self.dfpub_symb.pmid.isin(ids)]
		if df2.empty:
			print("There is no selected pmid (????)")
			return None


		lista = [x if isinstance(x, list) else eval(x) for x in df2.symbols]

		dic = {}
		for symbols in lista:
			for symb in symbols:
				dic[symb] = dic.get(symb, 0) + 1

		dfsymb_perc = pd.DataFrame({'symbol': list(dic.keys()), 'n': dic.values()})
		dfsymb_perc = dfsymb_perc.sort_values('n', ascending=False)
		dfsymb_perc.index = np.arange(0, len(dfsymb_perc))
		total = dfsymb_perc.n.sum()
		dfsymb_perc['perc'] = dfsymb_perc.n / total

		cumm = 0; mat=[]
		for i in range(len(dfsymb_perc)-1, -1, -1):
			cumm += dfsymb_perc.iloc[i].perc
			mat.append(cumm)

		mat.reverse()
		dfsymb_perc['cumm'] = mat

		print(f"There are {len(dfsymb_perc)} Hugo's symbols")
		self.dfsymb_perc_subclust = dfsymb_perc

		ret = pdwritecsv(dfsymb_perc, fname, self.root_pubmed, verbose=True)

		return dfsymb_perc


	def symbol_perc_table_check(self):

		dfsymb_perc = self.find_symbols_build_perc_table()

		dfsymb_perc_check = dfsymb_perc.copy()
		dfsymb_perc_check['symbol_check'] = [self.gene.replace_synonym_to_symbol(x) for x in dfsymb_perc_check.symbol]
		dfsymb_perc_check['is_ok'] = dfsymb_perc_check.symbol == dfsymb_perc_check.symbol_check

		self.dfsymb_perc_check = dfsymb_perc_check
		df2 = dfsymb_perc_check[~dfsymb_perc_check.is_ok]
		return df2


	def find_three_classes_in_pubmed_searches(self, perc_small_num:float=0.05, perc_avg_num:float=0.25,
											  prompt_stat:bool=True, force:bool=False, verbose:bool=False):
		dfsymb_perc = self.find_symbols_build_perc_table(force=force)

		''' small number of publications '''
		n_small_num_study = dfsymb_perc[dfsymb_perc.cumm <= perc_small_num].n.max()
		df_small_num = dfsymb_perc[dfsymb_perc.n <= n_small_num_study]

		genes_small_num_studies = df_small_num.symbol.to_list()
		genes_small_num_studies.sort()

		self.n_small_num_study = n_small_num_study
		self.df_small_num = df_small_num
		self.n_small_num = len(df_small_num)
		self.genes_small_num_studies = genes_small_num_studies

		if verbose:
			print('')
			print(self.n_small_num)
			print(", ".join(genes_small_num_studies))

		''' average number of publications '''
		n_avg_num_study = dfsymb_perc[ dfsymb_perc.cumm <= perc_avg_num].n.max()
		df_avg_num = dfsymb_perc[ (dfsymb_perc.n <= n_avg_num_study) & (dfsymb_perc.n > n_small_num_study)]

		genes_avg_num_studies = df_avg_num.symbol.to_list()
		genes_avg_num_studies.sort()

		self.n_avg_num_study = n_avg_num_study
		self.df_avg_num = df_avg_num
		self.n_avg_num = len(df_avg_num)
		self.genes_avg_num_studies = genes_avg_num_studies

		if verbose:
			print('')
			print(self.n_avg_num)
			print(", ".join(genes_avg_num_studies))

		# print(f'>= than {n_avg_num_study+1}')
		df_large_num = dfsymb_perc[ dfsymb_perc.n > n_avg_num_study]
		genes_large_num_studies = df_large_num.symbol.to_list()
		genes_large_num_studies.sort()

		self.n_large_num_study = df_large_num.n.max()
		self.df_large_num = df_large_num
		self.n_large_num = len(df_large_num)
		self.genes_large_num_studies = genes_large_num_studies

		if verbose:
			print('')
			print(self.n_large_num)
			print(", ".join(genes_large_num_studies))


		if prompt_stat:
			print(f'Pubmed Search Algorithm (PSA) found {len(dfsymb_perc)} genes/proteins.')
			print(f'There are {self.n_small_num} G/P with the smallest number of papers, citation upper limit number = {self.n_small_num_study}')
			print(f'There are {self.n_avg_num} G/P with the average number of papers, citation upper limit number = {self.n_avg_num_study}')
			print(f'There are {self.n_large_num} G/P with the largest number of papers, citation upper limit number = {self.n_large_num_study}')


	def build_pdf_file_type(self, curation:bool=False) -> pd.DataFrame:

		self.get_all_pmids(curation)

		if len(self.dfpub0) == 0:
			return None

		if len(self.pmid_list) == 0:
			return None

		pmid_list = [str(x) for x in self.pmid_list]

		files = [x for x in os.listdir(self.root_pdf) if x.endswith('pdf') or x.endswith('html')]
		files = [x for x in files if x.split('.')[0] in pmid_list]

		if len(files) == 0:
			print("No files found.")
			return None

		files.sort()

		pmids  = [x.split('.')[0] for x in files]
		ftypes = [x.split('.')[1] for x in files]

		df = pd.DataFrame({'pmid': pmids, 'ftype': ftypes})
		df.pmid = df.pmid.astype(int)

		return df


	def get_all_pmids(self, curation:bool=False):
		'''
		get all pmids from pubmed_summary - the big table or curation table
		'''
		self.set_search_or_curation(curation)

		if curation:
			dfpub0 = self.open_df_pubmed_curation()
		else:
			dfpub0 = self.open_df_pubmed_summary()

		if dfpub0 is None or dfpub0.empty:
			self.dfpub0 = None
			pmid_list = []
			npmids = 0
			print(f"There are no pmid(s) for {self.stri_curation}.")
		else:
			self.dfpub0 = dfpub0
			pmid_list = list(dfpub0.pmid.unique())
			npmids = len(pmid_list)
			print(f'Previous pubmed search found {npmids} pmids for {self.stri_curation}')

		self.npmids = npmids
		self.pmid_list = pmid_list


	def set_search_or_curation(self, curation:str):
		self.curation = curation
		self.stri_curation = 'curation' if curation else "regular search"

	def download_pdfs(self, curation:bool=False, delete_html:bool=False) -> bool:
		self.set_search_or_curation(curation)

		'''
			download pdfs/html: html from Pubmed(pmid) or pdf from PMC
		'''
		nprocessors = multiprocessing.cpu_count()
		nprocessors -= self.dec_ncpus
		self.nprocessors = nprocessors

		self.get_all_pmids(curation)

		if self.npmids == 0:
			return False

		bad_files = [x for x in os.listdir(self.root_pdf) if x.endswith('.bad')]
		if bad_files != []:
			for fname in bad_files:
				filename = os.path.join(self.root_pdf, fname)
				try:
					os.unlink(filename)
				except:
					print(f"Could not delete {filename}")

		if delete_html:
			html_files = [x for x in os.listdir(self.root_pdf) if x.endswith('.html')]
			if html_files != []:
				for fname in html_files:
					filename = os.path.join(self.root_pdf, fname)
					try:
						os.unlink(filename)
					except:
						print(f"Could not delete {filename}")

		files = [x for x in os.listdir(self.root_pdf) if x.endswith('.pdf') or x.endswith('.html')]
		pmids_done = [ int(x.split('.')[0]) for x in files]

		lista_not_dwn = [x for x in self.pmid_list if x not in pmids_done]
		self.lista_not_dwn = lista_not_dwn

		n_notdwn = len(lista_not_dwn)
		self.n_notdwn = n_notdwn

		if n_notdwn == 0:
			print("All done.")
			return True

		print(f'There are {n_notdwn} pmids to download for {self.stri_curation}')
		deln = int(n_notdwn/nprocessors)

		ini = 0
		limits = []
		for i in range(nprocessors):
			end = n_notdwn if i == nprocessors-1 else (i+1)*deln
			limits.append( (ini, end) )
			ini = end

		''' ---------------- parallel processing -------------------'''
		pool = Pool(nprocessors)

		lista_results = []

		for i in range(nprocessors):
			# print(i, limits[i])
			lista_results.append(pool.apply_async(self.pdf_download, (i, limits[i])))

		freeze_support()
		t0 = time.time()

		print(">>>> lista_results", len(lista_results), 'processors')
		for i in range(len(lista_results)):
			result = lista_results[i]

			try:
				# print(f"getting {i}")
				result.get(timeout=10000)
				# print(f"ok download {i}")
			except:
				print("Parallel processing has problems.")
				raise Exception(f'Stop parallel processing {i}')

			print("%d finished"%(i))

		seconds = time.time()-t0
		print("---- End: %.1f min -----------"%(seconds/60))
		return True

	def parse_all_pdfs_to_text(self, curation:bool=False, only_text:bool=False,
								pmids:List=[], ftypes:List=[],
								force:bool=False, verbose:bool=False,) -> bool:

		if curation:
			fname = self.fname_pubmed_curation_no_symb0%(self.prefix, self.s_all_text_or_abstract)
			root = self.root_curation
		else:
			fname = self.fname_pubmed_no_symb0%(self.prefix, self.s_all_text_or_abstract)
			root = self.root_pubmed

		fname = title_replace(fname)
		filename = os.path.join(root, fname)

		if os.path.exists(filename) and not force:
			dfpub1 = pdreadcsv(fname, root, verbose=True)
			self.dfpub1 = dfpub1

			ret = dfpub1 is not None and not dfpub1.empty
			if ret: return True

		if isinstance(pmids, list) and len(pmids) > 0:
			if len(pmids) != len(ftypes) or pmids == []:
				print("You choose testing. Please send the correct pmids and ftypes with the same length.")
				return False

				dffiles = pd.DataFrame({'pmid': pmids, 'ftype': ftypes})
				dffiles.pmid = dffiles.pmid.astype(int)

			self.get_all_pmids(curation)
		else:
			''' set self.dfpub0 '''
			dffiles = self.build_pdf_file_type(curation)

			if dffiles is None or dffiles.empty:
				return False

			pmids  = dffiles.pmid
			ftypes = dffiles.ftype

		'''---------- Merge new Columns ------------'''
		cols = ['concept', 'choice', 'pmid', 'pub_date', 'title',
				'keywords', 'abstract', 'abreviation', 'authors', 'created_date', 'doc_type', 'docid',
				'journalTitle', 'language', 'cases', 'case_comparisons', 'terms', 'dates']
		dfpub1 = self.dfpub0[cols]

		dfpub1 = pd.merge(dffiles, dfpub1, how="inner", on='pmid')
		dfpub1['has_text'] = False

		for i in range(len(pmids)):
			pmid = pmids[i]
			ftype = ftypes[i]

			if i%1000 == 0: print('.', end='')

			fname_txt = f'{pmid}.txt'
			fifefull_txt = os.path.join(self.root_pdf_txt, fname_txt)
			if os.path.exists(fifefull_txt) and not force:
				dfpub1.loc[dfpub1.pmid == pmid, 'has_text'] = True
				continue

			fname_type = f"{pmid}.{ftype}"
			filename = os.path.join(self.root_pdf, fname_type)

			if verbose:
				print(os.path.exists(filename), filename)

				dfaux = dfpub1[dfpub1.pmid == pmid]
				if dfaux.empty:
					print(f"Could not find pmid '{fname_type}' in dfpub1")
					continue

				row = dfaux.iloc[0]

				authors = row.authors
				if isinstance(authors, str):
					authors = eval(authors)

				concept = row.concept
				choice  = row.choice
				title	= row.title

				print(title[:60], concept, choice, authors[:3], end=' ')

			if only_text:
				all_text = self.get_dfpub_abstract_title_keywords(pmid)
			else:
				if ftype == 'pdf':
					all_text = self.parse_pdf_to_txt(filename)
				else:
					all_text = self.get_dfpub_abstract_title_keywords(pmid)

			if all_text == '':
				if verbose: print(f"Could not find the text for pmid={pmid}")
				continue

			if verbose:
				print(f'len = {len(all_text)}')

			write_txt(all_text, fname_txt, self.root_pdf_txt, verbose=verbose)
			if verbose:
				print('')

			dfpub1.loc[dfpub1.pmid == pmid, 'has_text'] = True

		ret = pdwritecsv(dfpub1, fname, root, verbose=True)
		self.dfpub1 = dfpub1

		return ret

	def get_dfpub_abstract_title_keywords(self, pmid:int) -> str:
		dfpub = self.dfpub0[self.dfpub0.pmid == pmid]
		if dfpub.empty:
			return ''

		row = dfpub.iloc[0]

		if isinstance(row.title, str) and row.title != '':
			text = row.title + '\n'
		else:
			text = ''

		if isinstance(row.keywords, str) and row.keywords != '':
			text += row.keywords + '\n'
		if isinstance(row.abstract, str) and row.abstract != '':
			text += row.abstract

		return text


	def parse_TIKA(self, filename):
		'''------------ Read PDF file ------------------'''
		try:
			data = parser.from_file(filename, xmlContent=True)
		except:
			data = None

		return data


	def try_parse_TIKA(self, filename):
		try:
			data = self.parse_TIKA(filename)
			ret = True
		except:
			print(f"Could not parse TIKA for '{filename}'")
			data = ''
			ret = False

		return ret, data

	def parse_pdf_to_txt(self, filename: str) -> str:
		all_text, data = '', ''

		ret, data = self.try_parse_TIKA(filename)

		if not ret:
			for tm in self.sleep_TIKA:
				print(f"TIKA access error: time {tm} s")
				time.sleep(tm)

				ret, data = self.try_parse_TIKA(filename)
				if ret: break

		if data == '':
			print(">>> TIKA Error: try again latter.")
			return ''

		try:
			xhtml_data = BeautifulSoup(data['content'], features="lxml")
		except:
			''' --- Chinese? other language? ---'''
			return all_text

		found_end_word = False
		found_start_word = False

		for i, content in enumerate(xhtml_data.find_all('div', attrs={'class': 'page'})):
			'''Parse PDF data using TIKA (xml/html)
				It's faster and safer to create a new buffer than truncating it
				https://stackoverfsmall.com/questions/4330812/how-do-i-clear-a-stringio-object '''
			_buffer = StringIO()
			_buffer.write(str(content))
			parsed_content = parser.from_buffer(_buffer.getvalue())

			try:
				text = parsed_content['content'].strip()
			except:
				continue

			if not found_start_word:
				words = ['Keyword', 'Introduction', 'INTRODUCTION',
						 'Dear Editor', 'Dear Sir', 'To the Editor', 'To the editor',
						 'I N T R O D U C T I O N', 'Original Article', 'Original article', 'O R I G I N A L']
				for word in words:
					''' do not put IGNORECASE '''
					match = re.search(word, text)

					if match:
						found_start_word = True
						ini, end = match.span()
						if word == 'O R I G I N A L':
							end += 15
						text = text[end: ]
						all_text = ''

				if text == '': continue

			words = ['Reference', 'REFERENCE', 'Acknowledg', 'ACKNOWLEDG', 'A C K N O W L E D G',
					 'Author Contribution', 'Contributors', 'AUTHOR CONTRIBUTION',
					 'Funding', 'Appendix', 'Credit authorship',
					 'Ethical approval', 'DATA AVAILABILITY', 'Data availability', 'Declarations',
					 'Supporting Information', 'CODE AVAILABILITY', 'ORCID',
					 'SUPPLEMENTAL INFORMATION', 'SUPPLEMENTAL MATERIAL', 'Supplemental Material',
					 'Declaration of Competing Interest', 'Conflicts of Interest',
					 'Conflict of Interest', 'CONFLICT OF INTEREST', 'no competing interest',
					 'Declaration of interest', 'no conflict of interest', 'no conflicts of interest']

			for word in words:
				''' do not put IGNORECASE '''
				match = re.search(word, text)
				if match:
					ini, end = match.span()
					text = text[:ini]

					''' short papers, may have funding, ack in the begining '''
					if i > 1: found_end_word = True
					if text == '':	break

			if text != '':
				all_text += '\n' + text

			''' if already found the end, forget additional pages '''
			if found_end_word:
				break

		all_text = all_text.replace("‘","").replace("’","").replace("�","").replace('  ', ' ').strip()

		return self.split_text(all_text)


	def split_text(self, text):
		mat = text.split("\n")
		final_text = ''

		previous = ''
		for i in range(len(mat)):

			parag = mat[i]
			if parag == '' or parag == ' ':
				continue

			if parag.startswith('www.') or parag.startswith('Received:') or parag.startswith('Accepted:') or \
				parag.startswith('Published:') or parag.startswith('OPEN') or parag.startswith('https:') or \
				parag.startswith('http:') or parag.startswith('mailto:') or parag.startswith('Received:'):
				continue

			# print(f">>{i} '{parag}'")

			if parag[-1] == ' ':
				if previous == '':
					previous = parag
				else:
					previous += parag
				continue

			if parag[-1] == '-':
				if previous == '':
					previous = parag[:-1]
				else:
					previous += parag[:-1]
				continue

			if previous != '':
				parag = previous + parag
				previous = ''

			final_text += ' ' + parag

		return final_text.replace('  ', ' ').strip()

	def open_lemma_curation(self, open_all:bool=False, verbose:bool=False) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
		'''
		open:
		  - df_lemma, df_token
		'''
		fname, fname_token, fname_lemma = self.prepare_curation_names_token_and_lemma()
		self.fname, self.fname_token, self.fname_lemma = fname, fname_token, fname_lemma

		filename = os.path.join(self.root_lemma, fname_lemma)

		if not os.path.exists(filename):
			print(f"There is no '{filename}'")
			return None, None, None

		df_lemma = pdreadcsv(fname_lemma, self.root_lemma, verbose=verbose)
		df_lemma = df_lemma[~ df_lemma.lemma.isna()].copy()
		df_lemma.index = np.arange(0, len(df_lemma))

		if open_all:
			df_token = pdreadcsv(fname_token, self.root_lemma, verbose=verbose)
			df_concept_symb  = pdreadcsv(fname, self.root_lemma, verbose=verbose)
		else:
			df_token, df_concept_symb = None, None

		self.df_lemma = df_lemma
		self.df_token = df_token
		self.df_concept_symb = df_concept_symb

		return df_lemma, df_token, df_concept_symb

	def open_lemma(self, all_pdf_html:str, open_all:bool=False, verbose:bool=False) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
		'''
		open:
		  - df_lemma, df_token
		'''
		fname, fname_token, fname_lemma = self.prepare_names_token_and_lemma(all_pdf_html)
		self.fname, self.fname_token, self.fname_lemma = fname, fname_token, fname_lemma

		filename = os.path.join(self.root_lemma, fname_lemma)

		if not os.path.exists(filename):
			print(f"There is no '{filename}'")
			return None, None, None

		df_lemma = pdreadcsv(fname_lemma, self.root_lemma, verbose=verbose)
		df_lemma = df_lemma[~ df_lemma.lemma.isna()].copy()
		df_lemma.index = np.arange(0, len(df_lemma))

		if open_all:
			df_token = pdreadcsv(fname_token, self.root_lemma, verbose=verbose)
			df_concept_symb  = pdreadcsv(fname, self.root_lemma, verbose=verbose)
		else:
			df_token, df_concept_symb = None, None

		self.df_lemma = df_lemma
		self.df_token = df_token
		self.df_concept_symb = df_concept_symb

		return df_lemma, df_token, df_concept_symb

	''' todo rever !!! '''
	def reclean_lemma(self, lista, all_pdf_html:str='all', verbose:bool=False) -> bool:
		print("Reading df lemma ...")
		df_lemma, _, _ = self.open_lemma(all_pdf_html=all_pdf_html, open_all=False, verbose=verbose)

		if df_lemma is None or df_lemma.empty:
			print("No df_lemma.")
			return False

		print("Cleaning ...")

		def clean_lemma(text):
			text = re.sub(r'	  |	 |	|	|  ', ' ', text)
			text = re.sub('\t\t\t\t\t\t|\t\t\t\t\t|\t\t\t\t|\t\t\t|\t\t', ' ', text)
			mat = [x for x in text.split(' ') if x not in lista and not isfloat(x)]
			return " ".join(mat)

		mat = [clean_lemma(text) for text in df_lemma.lemma]

		df2 = df_lemma.copy()
		df2 = pd.DataFrame({'pmid': df_lemma.pmid, 'lemma': mat})
		df_lemma = df2

		_, _, fname_lemma = self.prepare_names_token_and_lemma(all_pdf_html)
		ret = pdwritecsv(df_lemma, fname_lemma, self.root_lemma, verbose=verbose)

		return ret


	def barplot_genes_pubmed(self, plot_type='large', perc_small_num=0.15, perc_avg_num=0.5,
							 nGenesPlot=120, height=600, width=1100, plots_per_page=6,
							 x_tick_font_size=10, y_tick_font_size=12,
							 x_label_font_size=12, y_label_font_size=12,
							 prompt_stat:bool=True, verbose:bool=False):

		self.find_three_classes_in_pubmed_searches(perc_small_num=perc_small_num, perc_avg_num=perc_avg_num,
													prompt_stat=prompt_stat, force=False, verbose=verbose)

		if plot_type == 'large':
			df2 = self.df_large_num.copy()
		elif plot_type == 'average':
			df2 = self.df_avg_num.copy()
		else:
			df2 = self.df_small_num.copy()

		df2.index = np.arange(0, len(df2))
		nGenes = len(df2)
		N = int(np.ceil(nGenes/ nGenesPlot))

		npages = int(np.ceil(N/ plots_per_page))

		if verbose: print(">>> test: ", N, nGenes, npages)

		title = f"Possible {plot_type} studied/cited Symbols in COVID-19 literature: first {nGenes} genes/proteins"

		height = height*N

		lista_figs=[]

		for page in range(npages):
			fig = make_subplots(rows=N, cols=1)
			offset = page*plots_per_page*nGenesPlot

			for i in range(plots_per_page):
				ini = offset + i * nGenesPlot

				if ini > nGenes:
					break

				end = offset + (i+1) * nGenesPlot

				df3 = df2.iloc[ini:end]

				fig.append_trace(
					go.Bar(y=df3.n, x=df3.symbol),  row=i+1, col=1
				)

			# fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)

			fig.update_xaxes(tickangle = 90,
							 tickfont=dict(family='Arial', color='black', size=x_tick_font_size),
							 title_font=dict(size=x_label_font_size, family='Arial', color='black'),
							 title_text = "Gene/Protein")

			fig.update_yaxes(tickfont=dict(family='Arial', color='black', size=y_tick_font_size),
							 title_font=dict(size=y_label_font_size, family='Arial', color='black'),
							 title_text = "# Citations")

			fig.update_layout(
						autosize=True,
						title=title,
						width=width,
						height=height,
						showlegend=False
			)

			figname = title_replace(title)
			figname = os.path.join(self.root_figure, figname + f'_page_{page+1}_from_{npages}.html')

			if verbose: print(">>> HTML saved:", figname)
			fig.write_html(figname)

			fig.write_image(figname.replace('.html', '.png'))
			lista_figs.append(fig)

		return lista_figs

	def ok_word(self, word):
		if len(word) < 2:
			return None
		if len(word) > 19:
			return None

		for end in self.stop_end_words:
			if word.endswith(end):
				if isint(word[:-len(end)]):
					return None

		for c in self.stop_char_nums:
			if isint(word.replace(c,'') ):
				return None

		if len(word) > 4 and isint(word[-3:]):
			return None

		if word.startswith('abbreviation'):
			return None

		mat = [True if x in self.acgt else False for x in word]
		if np.sum(mat) == len(word):
			return None

		return word

	''' read all lemmas (stem_words) and return unique lemmas '''
	def calc_bow(self, df):
		lista = []
		for lemma in df.lemma:
			words = np.unique(lemma.split(' '))
			lista += [x for x in words if self.ok_word(x) is not None]

		return list(np.unique(lista))

	def calc_hypergeometric_table(self, case:str, df_enr:pd.DataFrame,
								  pval_min:float=0.10, p_val_verbose:float=0.90,
								  force:bool=False, verbose:bool=False) -> pd.DataFrame:

		self.case = case
		n_pathways = len(df_enr)
		self.n_pathways = n_pathways

		M = self.dic_bow_case['all']['n_words']
		N = self.dic_bow_case[case]['n_words']
		df_lemma = self.dic_bow_case[case]['df_lemma']

		# dic2['dic_bow'] = dic_bow
		# dic2['bow'] = bow

		print(f"Hypergeometric statistics reading 'all papers' - case {case} there are {n_pathways} ernriched pathways.")
		print(f"One finds {N} words in PubMed paper search for case {case} and {M} words available for {self.prefix}")

		fname = self.prepare_hypergeometric_name(case)
		filename = os.path.join(self.root_ressum, fname)

		if os.path.exists(filename) and not force:
			print(f"Case {case}, already calculated.")
			dfall = pdreadcsv(fname, self.root_ressum, verbose=verbose)
			return dfall

		df_list = []
		for ipatw in range(n_pathways):
			pathway	= df_enr.iloc[ipatw].pathway
			pathway_id = df_enr.iloc[ipatw].pathway_id
			print(f">>> {ipatw+1}/{n_pathways})", pathway_id)

			stem_pathway = self.calc_pathway_bow(pathway_id)

			if not isinstance(stem_pathway, str) or stem_pathway == '':
				print(f"No words found for pathway {pathway_id} - {pathway}")
				continue

			dic = {}
			for i in range(len(df_lemma)):
				pmid  = df_lemma.iloc[i].pmid
				words = df_lemma.iloc[i].lemma.split(' ')
				words = np.unique(words)
				words = [word for word in words if self.isin_dic_bow_case(case, word)]
				n = len(words)

				words_k = [x for x in stem_pathway.split(' ') if x in words]
				k = len(words_k)

				p = hypergeom.cdf(k, M, n, N)

				dic[i] = {}
				dic2 = dic[i]
				dic2['case'] = case
				dic2['pathway_id'] = pathway_id
				dic2['pathway'] = pathway
				dic2['pmid'] = pmid
				dic2['n'] = n
				dic2['k'] = k
				dic2['M'] = M
				dic2['N'] = N
				dic2['pval'] = p
				dic2['words'] = words
				dic2['words_k'] = words_k

				verbose = True if p >= p_val_verbose else False

				if verbose:
					print(f"For case '{case}', one finds {k} words of the {pathway_id} enriched pathway in {n} words of paper pmid {pmid}")
					print(f"pathway: '{pathway}'")

					if p < pval_min:
						print(f"One rejects H0, is statistically unlikely to find such low number {k} words in the pathways, p-value = {p:.3e}")
					elif p > 1-pval_min:
						print(f"One rejects H0, is statistically unlikely to find such high number {k} words in the pathways, p-value = {p:.3e}")
					else:
						print(f"One accepts H0, is statistically likely to find {k} DEGs in the pathways, p-value = {p:.3e}")

					print("")

			dfs = pd.DataFrame(dic).T
			df_list.append(dfs)

		dfall = pd.concat(df_list)
		dfall.index = np.arange(0, len(dfall))
		print("------- end --------")

		ret = pdwritecsv(dfall, fname, self.root_ressum, verbose=True)

		return dfall


	def prepare_curation_tables_and_dicts(self, df_reactome:pd.DataFrame, case_list:List,
										  force:bool=False, verbose:bool=False) -> bool:
		'''
		prepare_curation_tables_and_dicts():
			- open:
				self.df_lemma, self.dfm_pub
			- calculates:
				self.dic_bow_case = self.build_all_bag_of_words()

			if hypergeometric tables already calculated,
			build_all_bag_of_words() is not performed
		'''
		self.df_lemma, self.dfm_pub, self.dic_bow_case = None, None, {}

		''' open reactome with
			ret = bpx.reactome.merge_reactome_table_gmt()
		'''
		self.df_reactome = df_reactome
		self.bpx.case_list = ['all'] + case_list

		print(">>> opening lemmatization table.")
		df_lemma, _, _ = self.open_lemma_curation(open_all=False)
		if df_lemma is None or df_lemma.empty:
			return
		self.df_lemma = df_lemma

		print(">>> opening pubmed table (dfm_pub).")
		dfm_pub = self.open_df_pubmed_curation(final_version=True)
		if dfm_pub is None or dfm_pub.empty:
			return
		self.dfm_pub = dfm_pub


		if force:
			is_ok = False
		else:
			is_ok = True
			for case in case_list:
				fname = self.prepare_hypergeometric_name(case)
				filename = os.path.join(self.root_curation, fname)

				if not os.path.exists(filename):
					is_ok = False
					break


		if is_ok:
			print("Hypergeometric tables already calculated")
			print("Set force=True or delete them from ressum dir.")
			return True

		self.build_all_bag_of_words()
		return len(self.dic_bow_case) > 1


	def build_all_bag_of_words(self):
		print(">>> calculating bag of words dictionary (dic_bow_case), for:")

		self.dic_bow_case = {}
		for case in self.bpx.case_list:
			print("\t", case)

			dfpub2 = self.dfpub[self.dfpub.choice == case]
			pmid_lista_all = list(dfpub2.pmid)
			df_lemma = self.df_lemma[self.df_lemma.pmid.isin(pmid_lista_all)]

			self.dic_bow_case[case] = {}
			dic2 = self.dic_bow_case[case]

			bow = self.calc_bow(df_lemma)
			if case != 'all':
				bow = [x for x in bow if self.isin_dic_bow_all(x)]

			dic_bow = {x: 1 for x in bow}

			dic2['dic_bow'] = dic_bow
			dic2['bow'] = bow
			dic2['n_words'] = len(bow)
			dic2['df_lemma'] = df_lemma


	def isin_dic_bow_all(self, x:str) -> bool:
		try:
			_ = self.dic_bow_case['all']['dic_bow'][x]
			return True
		except:
			return False

	def isin_dic_bow_case(self, case:str, x:str) -> bool:
		try:
			_ = self.dic_bow_case[case]['dic_bow'][x]
			return True
		except:
			return False


	def calc_pathway_bow(self, pathway_id:str) -> str:

		df_reactome = self.df_reactome

		dfa = df_reactome[df_reactome.pathway_id == pathway_id]
		if dfa.empty:
			print(f"Could not find pathway_id={pathway_id} in df_reactome")
			return ''

		row = dfa.iloc[0]
		species = row.species if isinstance(row.species, list) else eval(row.species)
		species = species[0]['displayName']

		if species != 'Homo sapiens':
			print("Wrong Species", species)
			return ''

		name = row['name']
		abstract = row.abstract
		genes = row.genes_pathway if isinstance(row.genes_pathway, list) else eval(row.genes_pathway)
		s_genes = "; ".join(genes)

		text = name + ' ' + abstract + ' ' + s_genes
		words_enr = text.split(' ')
		words_enr = np.unique(words_enr)

		text = " ".join(words_enr)

		text, stem_text, symbols = self.calc_lemmatization(text, filter_adjectives=False, verbose=False)

		return stem_text

	def which_stat_accuray(self, run:str, case:str, with_gender:bool, compare:str):
		if self.dfbinv is None:
			print("Forgot to open dfbinv")
			return -1

		dfbinv = self.dfbinv

		df2 = dfbinv[(dfbinv.run == run) & 
					 (dfbinv.case == case) &
					 (dfbinv.with_gender == with_gender) &
					 (dfbinv['compare'] == compare) ]
		if df2.empty:
			return -1
		return df2.iloc[0].fdr


	def open_stat_3sources_binary_vars(self, run_list:List, with_gender_list:List,
									   force:bool=False, verbose:bool=False) -> pd.DataFrame:

		if self.dfbinv is not None and not force:
			return self.dfbinv


		fname = self.fname_stat_3sources_binary_vars%(self.disease, run_list)
		filename = os.path.join(self.root_curation, fname)

		if not os.path.exists(filename):
			dfbinv = None
		else:
			dfbinv = pdreadcsv(fname, self.root_curation, verbose=verbose)

		self.dfbinv = dfbinv

		return dfbinv


	def stat_3sources_binary_vars(self, run_list:List, case_list:List, 
								  with_gender_list:List, person_list: List,
								  force:bool=False, verbose:bool=False) -> pd.DataFrame:

		fname = self.fname_stat_3sources_binary_vars%(self.disease, run_list)
		filename = os.path.join(self.root_curation, fname)

		if os.path.exists(filename) and not force:
			self.dfbinv = pdreadcsv(fname, self.root_curation, verbose=verbose)
			return self.dfbinv


		_, _, df_accu = self.calc_all_crowd_accuracies(person_list=person_list, run_list=run_list,
													   case_list=case_list, with_gender_list=with_gender_list, 
													   force=force, verbose=verbose)

		if df_accu is None or df_accu.empty:
			print("Could not find accuracy table (df_accu).")
			return None

		dic={}; icount=-1
		df_list = []
		for run in run_list:
			for case in case_list:
				for with_gender in with_gender_list:

					#---------- acuracy stats -------------------------------------------------------------------------
					df_count = self.count_crowd_agreements(with_gender=with_gender, person_list=person_list,
														  run=run, case=case, verbose=verbose)

					if df_count.empty:
						continue

					n = len(df_count)
					
					ngem_true  = len(df_count[df_count.agree_gemini == True])
					ngem_false = len(df_count[df_count.agree_gemini == False])
					vals_gem   = [ngem_true, ngem_false]
					
					npub_true  = len(df_count[df_count.agree_pubmed == True])
					npub_false = len(df_count[df_count.agree_pubmed == False])
					vals_pub   = [npub_true, npub_false]
					
					nrev_true  = len(df_count[df_count.agree_review == True])
					nrev_false = len(df_count[df_count.agree_review == False])
					vals_rev   = [nrev_true, nrev_false]
					
					s_stat, stat, pvalue, _, _ = chi2_or_fisher_exact_test(vals_pub, vals_gem)
					pval_pub_gem = pvalue

					s_stat, stat, pvalue, _, _ = chi2_or_fisher_exact_test(vals_rev, vals_gem)
					pval_rev_gem = pvalue

					s_stat, stat, pvalue, _, _ = chi2_or_fisher_exact_test(vals_rev, vals_pub)
					pval_rev_pub = pvalue
					

					for i in range(3):
						if i==0:
							pvalue = pval_pub_gem
							compare = 'Accu_PubxGem'
					
							n1_1 = npub_true
							n1_2 = npub_false
							n2_1 = ngem_true
							n2_2 = ngem_false
							
						elif i==1:
							pvalue = pval_rev_gem
							compare = 'Accu_RevxGem'
				
							n1_1 = nrev_true
							n1_2 = nrev_false
							n2_1 = ngem_true
							n2_2 = ngem_false
						
						else:
							pvalue = pval_rev_pub
							compare = 'Accu_RevxPub'
				
							n1_1 = nrev_true
							n1_2 = nrev_false
							n2_1 = npub_true
							n2_2 = npub_false
						
						icount += 1
						dic[icount] = {}
						dic2 = dic[icount]
						
						dic2['run'] = run
						dic2['case'] = case
						dic2['with_gender'] = with_gender
						dic2['compare'] = compare
						dic2['pvalue'] = pvalue
					
						dic2['n'] = n
						dic2['n1_true']  = n1_1
						dic2['n1_false'] = n1_2
						dic2['n2_true']  = n2_1
						dic2['n2_false'] = n2_2


					#------------------- Reviewers comparisons --------------------------------------------------------
					dfa = df_accu[(df_accu.run == run) & (df_accu.case == case) & (df_accu.with_gender == with_gender)]
					if dfa.empty:
						continue

					row = dfa.iloc[0]

					review_x_gemini = int(np.round(row.perc_review_x_gemini * n))
					review_x_pubmed = int(np.round(row.perc_review_x_pubmed * n))

					vals0 = [review_x_gemini, n-review_x_gemini]
					vals1 = [review_x_pubmed, n-review_x_pubmed]

					s_stat, stat, pvalue, _, _ = chi2_or_fisher_exact_test(vals0, vals1)

					compare = 'REVxPubMed x REVxGEM'

					icount += 1
					dic[icount] = {}
					dic2 = dic[icount]
					
					dic2['run'] = run
					dic2['case'] = case
					dic2['with_gender'] = with_gender
					dic2['compare'] = compare
					dic2['pvalue'] = pvalue
				
					dic2['n'] = n
					dic2['n1_true']  = review_x_pubmed
					dic2['n1_false'] = n-review_x_pubmed
					dic2['n2_true']  = review_x_gemini
					dic2['n2_false'] = n-review_x_gemini

					#------------------- Pathways Yes/No-----------------------------------------------------
					# ['perc_pathws_crowd', 'perc_pathws_gemini', 'perc_pathws_pubmed', 'perc_pathws_review']
					
					pathws_crowd = int(np.round(row.perc_pathws_crowd * n))
					vals_crowd = [pathws_crowd, n-pathws_crowd]
					
					pathws_gemini = int(np.round(row.perc_pathws_gemini * n))
					vals_gemini = [pathws_gemini, n-pathws_gemini]
					
					pathws_pubmed = int(np.round(row.perc_pathws_pubmed * n))
					vals_pubmed = [pathws_pubmed, n-pathws_pubmed]

					pathws_review = int(np.round(row.perc_pathws_review * n))
					vals_review = [pathws_review, n-pathws_review]

					s_stat, stat, pvalue, _, _ = chi2_or_fisher_exact_test(vals_gemini, vals_crowd)
					pval_gem_crowd = pvalue

					s_stat, stat, pvalue, _, _ = chi2_or_fisher_exact_test(vals_pubmed, vals_crowd)
					pval_pub_crowd = pvalue

					s_stat, stat, pvalue, _, _ = chi2_or_fisher_exact_test(vals_review, vals_crowd)
					pval_rev_crowd = pvalue
					
					n2_1 = vals_crowd[0]
					n2_2 = vals_crowd[1]

					for i in range(3):
						if i==0:
							pvalue = pval_gem_crowd
							compare = 'Pathw_GemxCrowd'
					
							n1_1 = vals_gemini[0]
							n1_2 = vals_gemini[1]

						elif i==1:
							pvalue = pval_pub_crowd
							compare = 'Pathw_PubxCrowd'
				
							n1_1 = vals_pubmed[0]
							n1_2 = vals_pubmed[1]
						
						else:
							pvalue = pval_rev_crowd
							compare = 'Pathw_RevxCrowd'
				
							n1_1 = vals_review[0]
							n1_2 = vals_review[1]
						
						icount += 1
						dic[icount] = {}
						dic2 = dic[icount]
						
						dic2['run'] = run
						dic2['case'] = case
						dic2['with_gender'] = with_gender
						dic2['compare'] = compare
						dic2['pvalue'] = pvalue
					
						dic2['n'] = n
						dic2['n1_true']  = n1_1
						dic2['n1_false'] = n1_2
						dic2['n2_true']  = n2_1
						dic2['n2_false'] = n2_2


		if dic == {}:
			print("Could crate statistics - dic is empty.")
			return None

		df = pd.DataFrame(dic).T

		df_list = []
		for run in run_list:
			for compare in self.compare_list:
				df2 = df[(df.run == run) & (df['compare'] == compare)]
				df2 = df2.sort_values('pvalue')
				df2.index = np.arange(len(df2))
				df2['fdr'] = fdr(df2.pvalue)

				df_list.append(df2)


		cols = ['run', 'case', 'with_gender', 'compare', 'n', 'fdr', 'pvalue', 'n1_true', 'n1_false', 'n2_true', 'n2_false']

		dfbinv = pd.concat(df_list)
		dfbinv = dfbinv[cols]
		dfbinv = dfbinv.sort_values(['run', 'case', 'with_gender', 'compare'])
		dfbinv.index = np.arange(len(dfbinv))

		ret = pdwritecsv(dfbinv, fname, self.root_curation, verbose=verbose)
		self.dfbinv = dfbinv
		return dfbinv


	def stat_compare_pubmed_x_gemini_all(self, run:str, case_list:List, i_dfp_list:List,
										 chosen_model_list:List, with_gender_list:List, 
										 force:bool=False, verbose:bool=False) -> pd.DataFrame:

		fname = self.fname_compare_stats%(self.disease, run, i_dfp_list, chosen_model_list, self.suffix)
		filename = os.path.join(self.root_curation, fname)

		if os.path.exists(filename) and not force:
			return pdreadcsv(fname, self.root_curation, verbose=verbose)


		df_cnt = self.open_pubmed_x_gemini_stat(run=run, i_dfp_list=i_dfp_list,
												chosen_model_list=chosen_model_list, verbose=verbose)

		dic={}; icount=-1
		for case in case_list:
			for i_dfp in i_dfp_list:
				for with_gender in with_gender_list:
					df2 = df_cnt[ (df_cnt.run==run) & 
								  (df_cnt.with_gender==with_gender) & 
								  (df_cnt.case==case) & 
								  (df_cnt.i_dfp==i_dfp) ]

					if len(df2) == 1:
						row = df2.iloc[0]
				
						lista=[]
			
						n = row.n
						n_yes = row.n_gemini_yes
						n_no  = row.n_gemini_no
						vals0 = [n_yes, n_no]
				
						n_yes = row.n_pubmed_yes
						n_no  = row.n_pubmed_no						
						vals1 = [n_yes, n_no]
				
						s_stat, stat, pvalue, dof, expected = chi2_or_fisher_exact_test(vals0, vals1)
				
						icount += 1
						dic[icount] = {}
						dic2 = dic[icount]
					
						dic2['run'] = run
						dic2['case'] = case
						dic2['i_dfp'] = i_dfp
						dic2['with_gender'] = with_gender

						dic2['stat'] = stat
						dic2['pvalue'] = pvalue
						dic2['dof'] = dof
						dic2['expected'] = expected
						dic2['n'] = n
						dic2['vals_gemini'] = vals0
						dic2['vals_pubmed'] = vals1

		df_stat = pd.DataFrame(dic).T
		df_stat['fdr'] = fdr(df_stat.pvalue)

		cols=['run', 'case', 'i_dfp', 'with_gender', 'n', 'fdr', 'pvalue', 'stat', 'dof', 'expected', 'vals_gemini', 'vals_pubmed']
		df_stat = df_stat[cols]

		ret = pdwritecsv(df_stat, fname, self.root_curation, verbose=verbose)

		return df_stat


	def open_compare_pubmed_x_gemini(self, run:str, case:str, i_dfp:int, 
									 chosen_model_list:List, with_gender:bool=False, 
									 verbose:bool=False) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, 
															 float, float, int, str):
		'''
			calc_compare_pubmed_x_gemini
			open_compare_pubmed_x_gemini

			soft reproducibility: chosen_model_list

			pubmed x gemeni: for each run, case, i_dfp

			returns: dfn, df_both, df_only_pubmed, df_only_gemini, mu, std, n, text
		'''



		ret, _, _, _ = self.bpx.open_case(case, prompt_verbose=False, verbose=verbose)

		s_with_gender = 'with_gender' if with_gender else 'without_gender'

		fname = self.fname_pubmed_x_gemini%(run, case, i_dfp, s_with_gender, self.suffix)
		filename = os.path.join(self.root_curation, fname)

		if not os.path.exists(filename):
			dfe = pd.DataFrame()

			print(f"File not found: '{filename}'")
			return dfe, dfe, dfe, dfe, None, None, None, ''

		dfn = pdreadcsv(fname, self.root_curation, verbose=verbose)

		df_both, df_only_pubmed, df_only_gemini = self.filter_both_and_only(dfn, verbose=verbose)
		mu, std, n, text = self.calc_mean_compare_pubmed_x_gemini(dfn, with_gender)

		return dfn, df_both, df_only_pubmed, df_only_gemini, mu, std, n, text


	def calc_mean_compare_pubmed_x_gemini(self, dfn:pd.DataFrame, with_gender:str) -> (float, float, int, str):
		mu = dfn.agree.mean()
		std = dfn.agree.std()
		n = len(dfn)
		text = f"pubmed x gemini with_gender={with_gender} agree={100*mu:.1f}% ({100*std:.1f}%) n={n}"

		return mu, std, n, text



	def calc_compare_pubmed_x_gemini(self, run:str, case_list:List, i_dfp_list:List, chosen_model_list:List, 
									 with_gender:bool=False, force:bool=False, verbose_open_pub:bool=False, verbose:bool=False):
		'''
			set the model first
			return table comparing gemini x pubmed results
			save in self.root_curation
		'''

		s_with_gender = 'with_gender' if with_gender else 'without_gender'

		df_pub, _ = self.merge_all_pubmeds(case_list=case_list, i_dfp_list=i_dfp_list, 
										   with_gender=with_gender, verbose=verbose_open_pub)

		if df_pub is None or df_pub.empty:
			print("Please calc df_pub.")
			return None, None, None, None, None, None

		dfpiva = self.gem.open_gemini_dfpiva_all_models_one_run(run=run, chosen_model_list=chosen_model_list, 
																verbose=verbose)

		if dfpiva is None or dfpiva.empty:
			return None, None, None, None, None, None

		for case in case_list:
			print("\t", case, end=' ')
			ret, _, _, _ = self.bpx.open_case(case, prompt_verbose=False, verbose=verbose)

			for i_dfp in i_dfp_list:
				print(f"i_dfp {i_dfp}", end=' ')

				fname = self.fname_pubmed_x_gemini%(run, case, i_dfp, s_with_gender, self.suffix)
				filename = os.path.join(self.root_curation, fname)

				if os.path.exists(filename) and not force:
					continue

				cols = ['case', 'pathway_id', 'pathway', 'i_dfp', 'consensus', 'n_yes', 'n_no', 'unanimous']
				dfpiv2 = dfpiva[ (dfpiva.case == case) & (dfpiva.i_dfp == i_dfp) ][cols]

				'''---------------- filter df_pub -----------------'''
				cols = ['case', 'pathway_id', 'pathway', 'n_pmid']
				dfpub2 = df_pub[df_pub.case == case][cols]

				'''------ merge df_enr and filtered df_pub ------'''

				dfn = pd.merge(dfpiv2, dfpub2, how="outer", on=['pathway_id', 'pathway', 'case'])
				dfn = dfn[ (dfn.case == case) & (dfn.i_dfp == i_dfp)]
				dfn.index = np.arange(0, len(dfn))

				dfn.n_pmid = dfn.n_pmid.fillna(0)
				dfn.n_pmid = dfn.n_pmid.astype(int)

				dfn['run'] = run
				dfn['case'] = case
				dfn['with_gender'] = with_gender

				dfn['pubmed'] = ['No' if x==0 else 'Yes' for x in dfn.n_pmid]
				dfn['gemini'] = [None if pd.isnull(x) else 'Yes' if x=='Yes' or x=='Doubt' else 'No' for x in dfn.consensus]
				dfn['agree']  = [None if pd.isnull(dfn.iloc[i].gemini) else \
								 dfn.iloc[i].pubmed == dfn.iloc[i].gemini for i in range(len(dfn))]

				cols = ['run', 'case', 'pathway_id', 'pathway', 'i_dfp', 'with_gender', 'agree', 'pubmed', 'gemini', 'n_pmid', 'consensus', 'n_yes', 'n_no', 'unanimous']
				dfn = dfn[cols]

				ret = pdwritecsv(dfn, fname, self.root_curation, verbose=verbose)

			print("")

		return



	def calc_stat_pubmed_x_gemini_idfp(self, run_list:List, case_list:List, 
									   i_dfp_list:List, chosen_model_list:List, 
									   with_gender_list:List, force:bool=False, 
									   verbose:bool=False) -> pd.DataFrame:
		'''
			pubmed x gemini agreement statistics

			return: stat table
		'''

		fname = self.fname_stat_gemini_pubmed_agg_idfp%(self.disease, self.suffix)
		filename = os.path.join(self.root_curation, fname)

		if os.path.exists(filename) and not force:
			dfag = pdreadcsv(fname, self.root_curation, verbose=verbose)
			return dfag

		dic = {}; icount=-1
		for run in run_list:

			print("For run:", run)
			for with_gender in with_gender_list:
				print(f'PubMed with_gender {with_gender}')
			
				for case in case_list:
					for i_dfp in i_dfp_list:
						print(f"\ti_dfp {i_dfp}", end=' ')
						
						dfn, df_both, df_only_pubmed, df_only_gemini, mu, std, n, text = \
								self.open_compare_pubmed_x_gemini(run=run, case=case,  i_dfp=i_dfp, with_gender=with_gender, 
																 chosen_model_list=chosen_model_list,
																 verbose=verbose)
						text = f'#{len(dfn)} pathways agree' + text.split('agree')[1]
						
						sem = std / np.sqrt(n)
						print(f"\t\tfor {case:15} {text}")

						mu = dfn.agree.mean()
						std = dfn.agree.std()
						n = len(dfn)
						
						icount += 1
						dic[icount] = {}
						dic2 = dic[icount]
						
						dic2['run']   = run
						dic2['case']  = case
						dic2['i_dfp'] = i_dfp
						dic2['with_gender'] = with_gender
						
						dic2['agree_mean'] = mu
						dic2['agree_std'] = std
						dic2['n'] = n
						
					print("")
				print("")

		dfag = pd.DataFrame(dic).T
		ret = pdwritecsv(dfag, fname, self.root_curation, verbose=verbose)

		return dfag



	def calc_stat_pubmed_x_gemini(self, run_list:List, case_list:List, 
								  i_dfp_list:List, chosen_model_list:List, 
								  with_gender_list:List, force:bool=False, 
								  verbose:bool=False) -> pd.DataFrame:
		'''
			pubmed x gemini agreement statistics

			return: stat table
		'''

		fname = self.fname_stat_gemini_pubmed_agg%(self.disease, self.suffix)
		filename = os.path.join(self.root_curation, fname)

		if os.path.exists(filename) and not force:
			dfag = pdreadcsv(fname, self.root_curation, verbose=verbose)
			return dfag

		dic = {}; icount=-1
		for run in run_list:

			print("For run:", run)
			for with_gender in with_gender_list:
				print(f'PubMed with_gender {with_gender}')

				for case in case_list:
					print(f"case {case:15}", end=' i_dfp =')
					df_list = []
					for i_dfp in i_dfp_list:
						print(f"{i_dfp}", end=' ')
						
						dfn, df_both, df_only_pubmed, df_only_gemini, mu, std, n, text = \
								self.open_compare_pubmed_x_gemini(run=run, case=case,  i_dfp=i_dfp, with_gender=with_gender, 
																 chosen_model_list=chosen_model_list,
																 verbose=verbose)

						if dfn is None or dfn.empty:
							continue

						df_list.append(dfn)

					if df_list == []:
						continue
						
					dfn_case = pd.concat(df_list)

					mu  = dfn_case.agree.mean()
					std = dfn_case.agree.std()
					n   = len(dfn_case)
					
					icount += 1
					dic[icount] = {}
					dic2 = dic[icount]
					
					dic2['run']   = run
					dic2['case']  = case
					dic2['with_gender'] = with_gender
					
					dic2['agree_mean'] = mu
					dic2['agree_std'] = std
					dic2['n'] = n
	
		dfag = pd.DataFrame(dic).T
		ret = pdwritecsv(dfag, fname, self.root_curation, verbose=verbose)

		return dfag


	def filter_both_and_only(self, dfn:pd.DataFrame, verbose:bool=False) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):

		df_both = dfn[ ~pd.isnull(dfn.pubmed) & (dfn.pubmed == dfn.gemini) ].copy()
		df_both.index = np.arange(0, len(df_both))

		df_only_pubmed = dfn[ (dfn.pubmed == 'Yes') & ~(dfn.gemini == 'Yes') ].copy()
		df_only_pubmed.index = np.arange(0, len(df_only_pubmed))

		df_only_gemini = dfn[ (dfn.gemini == 'Yes') & ~(dfn.pubmed == 'Yes')].copy()
		df_only_gemini.index = np.arange(0, len(df_only_gemini))
		df_only_gemini['n_yes'] = df_only_gemini['n_yes'].astype(int)

		return df_both, df_only_pubmed, df_only_gemini


	def merge_reviewers_answers(self, case_list:List=['g3_female_elder', 'g3_male_adult'], 
								s_start:str='sampling_case_', query_type:str='query_type_strong_',
								force:bool=False, verbose:bool=False) -> dict:

		dic={}
		for case0 in case_list:
			print(">>>", case0, end=' ')

			fname_merge = self.fname_merged_anwsers%(self.prefix, case0)
			filename = os.path.join(self.root_curation, fname_merge)

			if os.path.exists(filename) and not force:
				dff = pdreadcsv(fname_merge, self.root_curation, verbose=verbose)
			else:
				files = [x for x in os.listdir(self.root_curation) if x.startswith(s_start+case0) and x.endswith('.xlsx')]

				if files == []:
					print("There are no files for", case0)
					continue

				for i in range(len(files)):
					print(i, end=' ')
					fname_rev = files[i]

					excel, sheets = self.exc.read_sheets(fname_rev)
					name = fname_rev.split(query_type)[1].split('_')[0].lower()
					name = name.split('.xlsx')[0].split('-')[0]

					df = self.exc.read_excel(sheet = sheets[0])
					# only 5 columns
					df = df.iloc[:,:5]
					df.columns = ['pathway_id', 'pathway', 'fdr', 'answer', 'source']
					df['answer'] = [x[0].upper()+x[1:].lower().strip() for x in df.answer]

					cols = ['pathway_id', 'pathway', 'answer']
					df = df[cols]

					if i == 0:
						dff = df.copy()
						cols = ['pathway_id', 'pathway', name]
						dff.columns = cols
					else:
						dff[name] = df.answer

				ret = pdwritecsv(dff, fname_merge, self.root_curation, verbose=verbose)
			
			dic[case0] = dff
		return dic

	def open_reviewers_answers(self, case:str, verbose:bool=False) -> pd.DataFrame:

		fname = self.fname_merged_anwsers%(self.prefix, case)
		dff = pdreadcsv(fname, self.root_curation, verbose=verbose)
		return dff


	def calc_reviewer_consensus(self, row:pd.Series) -> (str, int, int, bool, str, str, str):
		n_yes, n_no = 0, 0

		pathway = row['pathway']
		pathway_id = row['pathway_id']

		s_pathway = pathway + ' (' + pathway_id + ')'
		pathway_yes	= None
		pathway_no	 = None
		pathway_doubt = None

		row = row[2:]
		for col in row.index:

			answer = row[col].lower().strip()

			if answer == 'yes':
				n_yes += 1
			elif answer == 'no':
				n_no += 1
			else:
				print(f"Error: answer '{answer}'")

		if n_yes > n_no:
			consensus = 'Yes'
			pathway_yes = s_pathway
		elif n_yes < n_no:
			consensus = 'No'
			pathway_no = s_pathway
		else:
			consensus = 'Doubt'
			pathway_doubt = s_pathway

		unanimous = False if n_yes != 0 and n_no != 0 else True

		return consensus, n_yes, n_no, unanimous, pathway_yes, pathway_no, pathway_doubt


	def open_reviewers_answers_group_case(self, group:str, case:str, verbose:bool=False) -> pd.DataFrame:

		fname = self.fname_reviewers_case%(self.prefix, group, case)
		df = pdreadcsv(fname, self.root_curation, verbose=verbose)
		return df


	def compare_2_reviewer_groups(self, group_names:List, group_list:List, case_list:List,
								  force:bool=False,verbose:bool=False) -> pd.DataFrame:

		fname_rev = self.fname_reviewers_summary%(self.prefix)
		filename = os.path.join(self.root_curation, fname_rev)

		if os.path.exists(filename) and not force:
			return pdreadcsv(fname_rev, self.root_curation, verbose=verbose)


		dic={}; icount=-1; df_list=[]

		for i in range(len(group_list)):
			group = group_list[i]
			name_list = group_names[i]

			cols = ['pathway_id', 'pathway'] + name_list

			for case in case_list:
				
				fname_merge = self.fname_merged_anwsers%(self.prefix, case)
				dff = pdreadcsv(fname_merge, self.root_curation, verbose=verbose)
				if dff is None or dff.empty:
					continue
				
				dfag = dff[cols].copy()
				mat = [self.calc_reviewer_consensus(dfag.iloc[i]) for i in range(len(dfag))]

				dfag['case'] = case
				dfag['consensus'] = [x[0] for x in mat]
				dfag['n_yes']	  = [x[1] for x in mat]
				dfag['n_no']	  = [x[2] for x in mat]
				dfag['unanimous'] = [x[3] for x in mat]

				fname_rev_case = self.fname_reviewers_case%(self.prefix, group, case)
				ret=pdwritecsv(dfag, fname_rev_case, self.root_curation, verbose=verbose)

				lista = [x[4] for x in mat]
				lista2 = [pathw for pathw in lista if pathw is not None]
				lista2.sort()
				pathways_yes = str(lista2)
				dfag['pathways_yes'] = pathways_yes
				lista = [x[5] for x in mat]
				lista2 = [pathw for pathw in lista if pathw is not None]
				lista2.sort()
				pathways_no	= str(lista2)
				dfag['pathways_no'] = pathways_no
				lista = [x[6] for x in mat]
				lista2 = [pathw for pathw in lista if pathw is not None]
				lista2.sort()
				pathways_doubt = str(lista2)
				dfag['pathways_doubt'] = pathways_doubt

				df_list.append(dfag)
				
				n = len(dfag)
				n_consensus_yes   = len(dfag[dfag.consensus == 'Yes'])
				n_consensus_no	  = len(dfag[dfag.consensus == 'No'])
				n_consensus_doubt = len(dfag[dfag.consensus == 'Doubt'])
			
				icount += 1
				dic[icount] = {}
				dic2 = dic[icount]
			
				dic2['group'] = group
				dic2['case']  = case
				dic2['n'] = n
				dic2['perc_consensus_yes'] = n_consensus_yes / n
				dic2['perc_consensus_no']  = n_consensus_no / n
				dic2['perc_consensus_doubt'] = n_consensus_doubt /n
				dic2['n_consensus_yes'] = n_consensus_yes
				dic2['n_consensus_no']	= n_consensus_no
				dic2['n_consensus_doubt'] = n_consensus_doubt
				dic2['pathways_yes'] = pathways_yes
				dic2['pathways_no']	 = pathways_no
				dic2['pathways_doubt'] = pathways_doubt
				dic2['researchers'] = name_list

		dfsf = pd.DataFrame(dic).T
		ret = pdwritecsv(dfsf, fname_rev, self.root_curation, verbose=True)

		return dfsf


	def report_all_pubmed_x_gemini_both_and_only_one(self, run:str, 
													 with_gender_list:List, case_list:List, i_dfp_list:List,
													 chosen_model_list:List, i_dfp:int=0, 
													 force:bool=False, verbose:bool=False) -> str:

		fname_txt = self.fname_report_all_pubmed_x_gemini%(self.disease, i_dfp, chosen_model_list, self.suffix)
		filename_txt = os.path.join(self.root_curation, fname_txt)

		if os.path.exists(filename_txt) and not force:
			text = read_txt(fname_txt, self.root_curation, verbose=verbose)
			return text

		text=''
		text_yes = "Gemini agrees with PubMed - there is a relationship case x pathway\n"
		text_no  = "Gemini agrees with PubMed - there is no relationship case x pathway\n"

		for answer in ['Yes', 'No']:

			if answer == 'Yes':
				if text=='':
					text = text_yes
				else:
					text += text_yes
			else:
				if text=='':
					text = text_no
				else:
					text += text_no

			df_all, df_all_both, df_only_pubmed, df_only_gemini, df_stat = \
				self.calc_all_pubmed_x_gemini_both_and_only_one(run=run, case_list=case_list, i_dfp_list=i_dfp_list,
																chosen_model_list=chosen_model_list, 
																with_gender_list=with_gender_list, verbose=False)
			
			for case in case_list:
				for with_gender in with_gender_list:
					text += f"with_gender={with_gender}\n"

					for search_result in ['both', 'only_gemini', 'only_pubmed']:

						if search_result == 'both':
							text += f"\n\tBoth Gemini and Pubmed, answer {answer} i_dfp={i_dfp}\n"
							df1 = df_all_both[df_all_both.with_gender == with_gender]
						elif search_result == 'only_gemini':
							text += f"\n\tOnly Gemini, answer {answer} i_dfp={i_dfp}\n"
							df1 = df_only_gemini[df_only_gemini.with_gender == with_gender]
						else:
							text += f"\n\tOnly PubMed, answer {answer} i_dfp={i_dfp}\n"
							df1 = df_only_pubmed[df_only_pubmed.with_gender == with_gender]

						if df1.empty:
							text += f"\t>>> {case} (#0), nothing\n"
							continue


						if search_result == 'both' or search_result == 'only_gemini':
							df2 = df1[ (df1.case==case) & (df1.gemini==answer) & (df1.i_dfp==i_dfp)]
						else:
							df2 = df1[ (df1.case==case) & (df1.pubmed==answer) & (df1.i_dfp==i_dfp)]

						if df2.empty:
							text += f"\t>>> {case} (#0), nothing\n"
							continue


						lista = np.unique(df2.pathway)
						text += f"\t>>> {case} (#{len(lista)})\n"
						text += "\t\t" + "\n\t\t".join(lista) + "\n"
					text += "\n\t=== === === === === === === === ===\n"

			text += "\n---------------------------------\n\n"

		ret2 = write_txt(text, fname_txt, self.root_curation, verbose=verbose)
		return text


	def open_pubmed_x_gemini_stat(self, run:str, i_dfp_list:List, chosen_model_list:List, 
							 	  verbose:bool=False) -> pd.DataFrame:

		fname = self.fname_compare_counts%(self.disease, run, i_dfp_list, chosen_model_list, self.suffix)
		filename = os.path.join(self.root_curation, fname)

		if os.path.exists(filename):
			df_cnt = pdreadcsv(fname, self.root_curation, verbose=verbose)
		else:
			print(f"Could not find: '{filename}'")
			df_cnt = pd.DataFrame()

		return df_cnt


	def calc_all_pubmed_x_gemini_both_and_only_one(self, run:str, case_list:List, i_dfp_list:List,
												   chosen_model_list:List, with_gender_list:List, force:bool=False, 
												   verbose:bool=False) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):

		fname0 = self.fname_compare_pathways%(   self.disease, run,	i_dfp_list, chosen_model_list, self.suffix)
		fname1 = self.fname_compare_both_pubgem%(self.disease, run, i_dfp_list, chosen_model_list, self.suffix)
		fname2 = self.fname_compare_only_pubmed%(self.disease, run, i_dfp_list, chosen_model_list, self.suffix)
		fname3 = self.fname_compare_only_gemini%(self.disease, run, i_dfp_list, chosen_model_list, self.suffix)
		fname4 = self.fname_compare_counts%(	 self.disease, run, i_dfp_list, chosen_model_list, self.suffix)

		filename0 = os.path.join(self.root_curation, fname0)
		filename1 = os.path.join(self.root_curation, fname1)
		filename2 = os.path.join(self.root_curation, fname2)
		filename3 = os.path.join(self.root_curation, fname3)
		filename4 = os.path.join(self.root_curation, fname4)

		if os.path.exists(filename0) and os.path.exists(filename1) and os.path.exists(filename2) and not force:
			df_all	= pdreadcsv(fname0, self.root_curation, verbose=verbose)
			df_both	= pdreadcsv(fname1, self.root_curation, verbose=verbose)
			df_only_pubmed = pdreadcsv(fname2, self.root_curation, verbose=verbose)
			df_only_gemini = pdreadcsv(fname3, self.root_curation, verbose=verbose)
			df_stat	= pdreadcsv(fname4, self.root_curation, verbose=verbose)

			return df_all, df_both, df_only_pubmed, df_only_gemini, df_stat


		df_all_list, df_both_list, df_only_pubmed_list, df_only_gemini_list = [], [], [], []

		dic={}; icount=-1
		for case in case_list:
			print(case, end=' ')
			for i_dfp in i_dfp_list:
				print('i_dfp', i_dfp, end=' ')
				for with_gender in with_gender_list:
					print('gender', with_gender, end=' ')
					df_all, df_both, df_only_pubmed, df_only_gemini, mu, std, n, text = \
					self.open_compare_pubmed_x_gemini(run=run, case=case, i_dfp=i_dfp,
													  with_gender=with_gender,
													  chosen_model_list=chosen_model_list, verbose=verbose)


					if df_all is None or df_all.empty:
						continue

					# ['pathway_id', 'pathway', 'i_dfp', 'agree', 'pubmed', 'gemini', 'n_pmid', 'consensus', 'n_yes', 'n_no', 'unanimous']
					cols = ['run', 'case', 'pathway_id', 'pathway', 'i_dfp', 'with_gender', 'agree', 'pubmed', 'gemini', 'n_pmid', 'consensus', 'n_yes', 'n_no']
					df_all  = df_all[cols]

					df_both = df_both[cols]
					df_only_pubmed = df_only_pubmed[cols]
					df_only_gemini = df_only_gemini[cols]

					cols = ['run', 'case', 'pathway_id', 'pathway', 'i_dfp', 'with_gender', 'agree', 'pubmed', 'gemini', 'n_pmid', 'consensus_gemini', 'n_yes', 'n_no']
					df_all.columns  = cols
					df_both.columns = cols
					df_only_pubmed.columns = cols
					df_only_gemini.columns = cols

					df_all_list.append(df_all)
					df_both_list.append(df_both)
					df_only_pubmed_list.append(df_only_pubmed)
					df_only_gemini_list.append(df_only_gemini)
					
					icount += 1
					dic[icount] = {}
					dic2 = dic[icount]

					dic2['run'] = run
					dic2['case'] = case
					dic2['i_dfp'] = i_dfp
					dic2['with_gender'] = with_gender
					dic2['chosen_model_list'] = chosen_model_list

					dic2['mu'] = mu
					dic2['std'] = std
					dic2['n'] = n

					dic2['n_both_yes_or_no']   = len(df_both)
					dic2['n_only_pubmed_yes'] = len(df_only_pubmed)
					dic2['n_only_gemini_yes'] = len(df_only_gemini)

					dic2['n_pubmed_yes'] = len(df_all[df_all.pubmed == 'Yes' ])
					dic2['n_pubmed_no']  = len(df_all[df_all.pubmed == 'No' ])
					dic2['n_gemini_yes'] = len(df_all[df_all.gemini == 'Yes' ])
					dic2['n_gemini_no']  = len(df_all[df_all.gemini == 'No' ])

					dic2['text'] = text

		df_stat = pd.DataFrame(dic).T

		df_all  = pd.concat(df_all_list)
		df_both = pd.concat(df_both_list)
		df_only_pubmed = pd.concat(df_only_pubmed_list)
		df_only_gemini = pd.concat(df_only_gemini_list)

		df_all.index  = np.arange(0, len(df_all))
		df_both.index = np.arange(0, len(df_both))
		df_only_pubmed.index = np.arange(0, len(df_only_pubmed))
		df_only_gemini.index = np.arange(0, len(df_only_gemini))

		print("")
		print(">>> df_all", len(df_all), ">>> df_both", len(df_both), ">>> df_all only_pubmed", len(df_only_pubmed), ">>> only_gemini", len(df_only_gemini))

		ret = pdwritecsv(df_all,  fname0, self.root_curation, verbose=verbose)
		ret = pdwritecsv(df_both, fname1, self.root_curation, verbose=verbose)
		ret = pdwritecsv(df_only_pubmed, fname2, self.root_curation, verbose=verbose)
		ret = pdwritecsv(df_only_gemini, fname3, self.root_curation, verbose=verbose)
		ret = pdwritecsv(df_stat, fname4, self.root_curation, verbose=verbose)

		return df_all, df_both, df_only_pubmed, df_only_gemini, df_stat


	def set_researchers_students(self, pro:List, stu:List):
		self.pro = pro
		self.stu = stu

		self.reviewers = pro + stu



	def calc_all_pubmed_x_reviewers_agreement(self, case_list:List, i_dfp_list:List, 
											  with_gender_list:List, force:bool=False, 
											  verbose:bool=False) -> dict:
		dic={}

		for with_gender in with_gender_list:
			print(">>>> with_gender", with_gender, end=' ')
			dfm_pub, _ = self.merge_all_pubmeds(case_list=case_list, i_dfp_list=i_dfp_list, with_gender=with_gender, 
												show_warning=False, force=False, verbose=verbose)

			for case in case_list:
				dff = self.calc_pubmed_agreement(case=case, dfm_pub=dfm_pub, with_gender=with_gender,
												 force=force, verbose=verbose)

				dic[str(with_gender) + '-' + case] = dff
			print("")

		return dic


	def calc_pubmed_agreement(self, case:str, dfm_pub:pd.DataFrame, with_gender:bool,
						 	  force:bool=False, verbose:bool=False) -> pd.DataFrame:
		'''
			get all curation statistics: run_gemini_consensus_counts_all_models()
			get all reviewers opinions: open_reviewers_answers

			one by one reviewer merge the tables and found the agreement
		'''
		fname = self.fname_compare_pub_rev%(with_gender, case)
		filename = os.path.join(self.root_curation, fname)

		if os.path.exists(filename) and not force:
			return pdreadcsv(fname, self.root_curation, verbose=verbose)

		cols = ['pathway_id', 'n_pmid']
		dfpub2 = dfm_pub[ dfm_pub.case == case ][cols].copy()

		dfrev = self.open_reviewers_answers(case, verbose=verbose)

		cols_pathw = ['pathway_id', 'pathway']

		dic={}; icount=-1
		for person in self.reviewers:

			cols_gp = cols_pathw + [person]
			dfp = pd.merge(dfrev[cols_gp], dfpub2, how="outer", on='pathway_id')
			dfp = dfp[dfp.pathway_id.isin(dfrev.pathway_id)]

			dfp.n_pmid = dfp.n_pmid.fillna(0)
			dfp.n_pmid = dfp.n_pmid.astype(int)
			
			dfp['pub_has_pmid'] = [ 'No' if x==0 else 'Yes' for x in dfp.n_pmid ]
			dfp['agree']	 = [  dfp.iloc[i][person] == dfp.iloc[i]['pub_has_pmid'] for i in range(len(dfp)) ]
			dfp['agree_yes'] = [ (dfp.iloc[i][person] == dfp.iloc[i]['pub_has_pmid']) and (dfp.iloc[i]['pub_has_pmid']=='Yes') for i in range(len(dfp)) ]
			dfp['agree_no' ] = [ (dfp.iloc[i][person] == dfp.iloc[i]['pub_has_pmid']) and (dfp.iloc[i]['pub_has_pmid']=='No' ) for i in range(len(dfp)) ]

			agree	  = dfp.agree.mean()
			agree_yes = dfp.agree_yes.mean()
			agree_no  = dfp.agree_no.mean()
			
			agree_std	  = dfp.agree.std()
			agree_std_yes = dfp.agree_yes.std()
			agree_std_no  = dfp.agree_no.std()

			icount += 1
			dic[icount] = {}
			dic2 = dic[icount]

			n = len(dfp)

			dic2['person'] = person
			dic2['case'] = case

			dic2['n_total'] = n
			dic2['n_yes_pub_has_pmid'] = len(dfp[dfp.pub_has_pmid=='Yes'])
			dic2['n_no_pub_has_pmid']  = len(dfp[dfp.pub_has_pmid=='No'])
			dic2['n_agree']		= len(dfp[dfp.agree==True])
			dic2['n_agree_yes'] = len(dfp[dfp.agree_yes==True])
			dic2['n_agree_no']  = len(dfp[dfp.agree_no==True])

			dic2['agree']	  = agree
			dic2['agree_std'] = agree_std
			dic2['agree_yes'] = agree_yes
			dic2['agree_std_yes'] = agree_std_yes
			dic2['agree_no']	  = agree_no
			dic2['agree_std_no']  = agree_std_no

			dic2['perc_agree']	 = dic2['n_agree'] / n
			dic2['perc_agree_yes'] = dic2['n_agree_yes'] / n
			dic2['perc_agree_no']  = dic2['n_agree_no'] / n


			fname_person = self.fname_compare_pub_rev_person%(with_gender, case, person)
			ret = pdwritecsv(dfp, fname_person, self.root_curation, verbose=verbose)

		dff = pd.DataFrame(dic).T

		dff = dff.sort_values('agree', ascending=False)
		dff.index = np.arange(0, len(dff))

		dff['rank'] = 0

		rank = 0; prev_note=-1
		for i in range(len(dff)):
			note = dff.iloc[i].agree
			if note != prev_note:
				prev_note = note
				rank += 1
			dff.loc[i,'rank'] = rank

		ret = pdwritecsv(dff, fname, self.root_curation, verbose=verbose)

		return dff


	def calc_all_gemini_x_reviewers_agreement(self, run_list:List, case_list:List, 
											  chosen_model_list:List,
											  force:bool=False, verbose:bool=False) -> dict:

		# cols = ['case', 'i_dfp', 'pathway_id', 'pathway', 'simple', 'simple+pubmed', 'disease', 'disease+pubmed', 'consensus', 'n_yes', 'n_no', 'unanimous']
		cols = ['case', 'i_dfp', 'pathway_id', 'pathway', 'consensus', 'n_yes', 'n_no']

		dic={}
		for run in run_list:

			dfpiva = self.gem.calc_gemini_dfpiva_all_models_one_run(run=run,  case_list=case_list, 
																	chosen_model_list=chosen_model_list, 
																	verbose=verbose)
			for case in case_list:

				dfcase = dfpiva[ dfpiva.case == case ][cols].copy()
				dfcase.index = np.arange(len(dfcase))

				dff = self.calc_gemini_agreement(run=run, case=case, dfcase=dfcase, force=force, verbose=verbose)
				dff = dff.sort_values('agree', ascending=False)
				dff.index = np.arange(0, len(dff))

				dff['rank'] = 0
				rank = 0; prev_note=-1
				for i in range(len(dff)):
					note = dff.iloc[i].agree
					if note != prev_note:
						prev_note = note
						rank += 1
					dff.loc[i,'rank'] = rank

				stri = f"{run}-{case}"
				dic[stri] = dff

		return dic


	def calc_gemini_agreement(self, run:str, case:str, dfcase:pd.DataFrame,
							  force:bool=False, verbose:bool=False) -> pd.DataFrame:
		'''
			get all curation statistics: run_gemini_consensus_counts_all_models()
			get all reviewers opinions: open_reviewers_answers

			one by one reviewer merge the tables and found the agreement
		'''
		print(">>>", run, case)

		fname = self.fname_compare_gem_rev%(run, case)
		filename = os.path.join(self.root_curation, fname)

		if os.path.exists(filename) and not force:
			return pdreadcsv(fname, self.root_curation, verbose=verbose)

		
		dfrev = self.open_reviewers_answers(case, verbose=verbose)

		cols_pathw = ['pathway_id', 'pathway']

		dic={}; icount=-1
		for person in self.reviewers:

			cols_gp = cols_pathw + [person]
			dfp = pd.merge(dfcase, dfrev[cols_gp], how="inner", on=cols_pathw)
			dfp['agree']	 = [ (dfp.iloc[i]['consensus'] == 'Doubt') or (dfp.iloc[i][person] == dfp.iloc[i]['consensus']) for i in range(len(dfp)) ]
			dfp['agree_yes'] = [ (dfp.iloc[i]['agree'] == True) and (dfp.iloc[i]['consensus'] != 'No') for i in range(len(dfp)) ]
			dfp['agree_no']  = [ (dfp.iloc[i]['agree'] == True) and (dfp.iloc[i]['consensus'] == 'No') for i in range(len(dfp)) ]

			agree	 = dfp.agree.mean()
			agree_yes = dfp.agree_yes.mean()
			agree_no  = dfp.agree_no.mean()
			
			agree_std	 = dfp.agree.std()
			agree_std_yes = dfp.agree_yes.std()
			agree_std_no  = dfp.agree_no.std()

			icount += 1
			dic[icount] = {}
			dic2 = dic[icount]

			n = len(dfp)

			dic2['run'] = run
			dic2['case'] = case
			dic2['person'] = person

			dic2['n_total'] = n
			dic2['n_yes_gemini_cons'] = len(dfp[ (dfp.consensus=='Yes') | (dfp.consensus=='Doubt') ]) 
			dic2['n_no_gemini_cons']  = len(dfp[  dfp.consensus=='No' ]) 

			dic2['n_agree']	 = len(dfp[dfp.agree==True])
			dic2['n_agree_yes'] = len(dfp[dfp.agree_yes==True])
			dic2['n_agree_no']  = len(dfp[dfp.agree_no==True])

			dic2['agree']		 = agree
			dic2['agree_std']	 = agree_std
			dic2['agree_yes']	 = agree_yes
			dic2['agree_std_yes'] = agree_std_yes
			dic2['agree_no']	  = agree_no
			dic2['agree_std_no']  = agree_std_no

			dic2['perc_agree']	 = dic2['n_agree'] / n
			dic2['perc_agree_yes'] = dic2['n_agree_yes'] / n
			dic2['perc_agree_no']  = dic2['n_agree_no'] / n


			fname_person = self.fname_compare_gem_rev_person%(run, case, person)
			ret = pdwritecsv(dfp, fname_person, self.root_curation, verbose=verbose)

		df = pd.DataFrame(dic).T

		ret = pdwritecsv(df, fname, self.root_curation, verbose=verbose)

		return df


	def open_pubmed_agreement(self, with_gender:bool, case:str, person:str=None, verbose:bool=False) -> pd.DataFrame:
		if person is None or person=='':
			print(f"Please define person")
			return None

		fname_person = self.fname_compare_pub_rev_person%(with_gender, case, person)
		filename = os.path.join(self.root_curation, fname_person)

		if os.path.exists(filename):
			df = pdreadcsv(fname_person, self.root_curation, verbose=verbose)
		else:
			print(f"There is no pubmed with gender {with_gender} agreement for case {case} and reviewer {person}")
			df = None

		return df

	def open_gemini_agreement_person(self, run:str, case:str, person:str=None, verbose:bool=False) -> pd.DataFrame:
		if person is None:
			print(f"Please define person (reviewer)")
			return None

		fname_person = self.fname_compare_gem_rev_person%(run, case, person)
		filename = os.path.join(self.root_curation, fname_person)

		if os.path.exists(filename):
			df = pdreadcsv(fname_person, self.root_curation, verbose=verbose)
		else:
			print(f"There is no gemini agreement for case {case} and reviewer {person}")
			df = None

		return df

	def calc_all_pubmed_summaries(self, case_list: List, i_dfp_list:List, 
								  with_gender_list:List, show_warning:bool=True, 
								  force:bool=False, verbose:bool=False) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):


		fname1	= self.fname_pubmed_merged%(self.disease, self.suffix)
		filename1 = os.path.join(self.root_pubmed_root, fname1)

		fname2	= self.fname_pubmed_summary%(self.disease, self.suffix)
		filename2 = os.path.join(self.root_pubmed_root, fname2)

		fname3	= self.fname_pubmed_summary2%(self.disease, self.suffix)
		filename3 = os.path.join(self.root_pubmed_root, fname3)

		if os.path.exists(filename1) and os.path.exists(filename2) and os.path.exists(filename3) and not force:
			dfm_pub = pdreadcsv(fname1, self.root_pubmed_root, verbose=verbose)
			df_summ = pdreadcsv(fname2, self.root_pubmed_root, verbose=verbose)
			dfn	 = pdreadcsv(fname3, self.root_pubmed_root, verbose=verbose)
			return dfm_pub, df_summ, dfn


		df_list, df_list_summ = [],[]
		for with_gender in with_gender_list:

			print(">>>> with_gender", with_gender, '\n')
			dfm_pub, df_summ = self.merge_all_pubmeds(case_list=case_list, i_dfp_list=i_dfp_list, 
													  with_gender=with_gender, show_warning=show_warning, 
													  force=force, verbose=verbose)

			if dfm_pub is not None:
				df_list.append(dfm_pub)

			if df_summ is not None:
				df_list_summ.append(df_summ)
			print("")

		if df_list != []:
			dfm_pub = pd.concat(df_list)
			dfm_pub.index = np.arange(len(dfm_pub))
			ret1 = pdwritecsv(dfm_pub, fname1, self.root_pubmed_root, verbose=verbose)
		else:
			dfm_pub = None

		if df_list_summ != []:
			df_summ = pd.concat(df_list_summ)
			df_summ.index = np.arange(len(df_summ))
			ret2 = pdwritecsv(df_summ, fname2, self.root_pubmed_root, verbose=verbose)

			df1 = df_summ[df_summ.with_gender==False]
			df2 = df_summ[df_summ.with_gender==True]

			common_cols = ['case', 'i_dfp']
			cols  = ['case', 'i_dfp', 'n']

			dfn = pd.merge(df1[cols], df2[cols], how="outer", on=common_cols)
			dfn.columns = ['case', 'i_dfp', 'n_gender_false', 'n_gender_true']
			dfn = dfn.fillna(0)

			dfn['n_gender_false'] = dfn['n_gender_false'].astype(int)
			dfn['n_gender_true']  = dfn['n_gender_true'].astype(int)
			dfn['perc'] = [None if dfn.iloc[i].n_gender_false==0 else dfn.iloc[i].n_gender_true / dfn.iloc[i].n_gender_false for i in range(len(dfn))]

			ret3 = pdwritecsv(dfn, fname3, self.root_pubmed_root, verbose=verbose)

		else:
			ret2, ret3 = False, False
			df_summ = None
			dfn = None

		print("--------------- end -------------")

		return dfm_pub, df_summ, dfn


	def merge_all_pubmeds(self, case_list: List, i_dfp_list:List, 
						  with_gender:bool, show_warning:bool=True, 
						  force:bool=False, verbose:bool=False) -> (pd.DataFrame, pd.DataFrame):

		self.dfm_pub = None

		if with_gender is None:
			with_gender = False

		if self.inidate is None or self.enddate is None:
			print("Please, set inidate and enddate")
			return None, None

		print("\tFinding pmids...")

		df_list = []
		dic = {}; icount=-1
		for case in case_list:
			for i_dfp in i_dfp_list:
				# it builds 
				#		self.df_summ_pmid = df_summ_pmid
				#		self.df_summ_pathway = df_summ_pathway
				df_pmid = \
				self.run_case_pathway_pubmed_search(case=case, i_dfp=i_dfp, with_gender=with_gender, 
													test=False, show_warning=show_warning,
													force=force, verbose=verbose)

				dfsumm_pathw = self.df_summ_pathway 
				if dfsumm_pathw is None or dfsumm_pathw.empty:
					if show_warning: print("No dfsumm pathway found.")
					continue

				df_list.append(dfsumm_pathw)
				
				icount += 1
				dic[icount] = {}
				dic2 = dic[icount]

				dic2['case'] = case
				dic2['i_dfp'] = i_dfp
				dic2['with_gender'] = with_gender
				dic2['n'] = len(df_pmid)

			if len(i_dfp_list) > 1: print("")


		if df_list == []:
			print("Nothing found.")
			return None, None

		# dfsumm_pathw
		dfm_pub = pd.concat(df_list)
		dfm_pub.index = np.arange(0, len(dfm_pub))

		dfsumm = pd.DataFrame(dic).T

		# cols = ['case', 'with_gender', 'i_dfp', 'pathway_id', 'pathway', 'n_pmid']
		# dfm_pub = dfm_pub[cols]
		self.dfm_pub = dfm_pub
		self.dfsumm  = dfsumm

		return dfm_pub, dfsumm


	def calc_semi_summarize_pubmed_search(self, run_dummy:str, case_list:List, i_dfp_list:List,
											  with_gender_list:List, test:bool=False, show_warning:bool=True,
											  force:bool=False, verbose:bool=False) -> pd.DataFrame:

		fname	 = self.fname_pubmed_semisumm%(self.disease, self.suffix)
		filename = os.path.join(self.root_pubmed_root, fname)

		if os.path.exists(filename) and not force:
			dfs = pdreadcsv(fname, self.root_pubmed_root, verbose=verbose)
			return dfs

		def summarize_pubmed_search(case:str, dfs:pd.DataFrame) -> pd.DataFrame:
			dfg = dfs.groupby(['pathway_id', 'pathway']).pmid.count().reset_index().iloc[:,:3]
			dfg['case'] = case
			cols = ['pathway_id', 'pathway', 'n', 'case']
			dfg.columns = cols
			cols = ['case', 'pathway_id', 'pathway', 'n']
			dfg = dfg[cols]
			dfg = dfg.sort_values('n', ascending=False)
			dfg.index = np.arange(0, len(dfg))
			return dfg


		df_list = []
		for with_gender in with_gender_list:
			print(f"with_gender {with_gender}")

			for case in case_list:

				for i_dfp in i_dfp_list:

					df_pmid = \
					self.run_case_pathway_pubmed_search(case=case, i_dfp=i_dfp, with_gender=with_gender, 
														test=test, show_warning=show_warning, force=force, verbose=verbose)

					if df_pmid is None:
						continue

					dfg = summarize_pubmed_search(case, df_pmid)

					dfg['i_dfp'] = i_dfp
					dfg['with_gender'] = with_gender

					df_list.append(dfg)

				print("")
			print("")

		if df_list == []:
			print("Nothing found.")
			return None

		dfs = pd.concat(df_list)
		cols = ['case', 'with_gender', 'i_dfp', 'pathway_id', 'pathway', 'n']
		dfs = dfs[cols]
		dfs.index = np.arange(0, len(dfs))

		ret = pdwritecsv(dfs, fname, self.root_pubmed_root, verbose=verbose)

		return dfs


	def agreements_between_pubmed_and_gemini(self, run_list:List, case_list: List, i_dfp_list:List,
											 chosen_model_list:List, with_gender_list:List, force:bool=False, 
											 verbose:bool=False) -> (pd.DataFrame, pd.DataFrame):
		
		'''
			compare PubMed x Gemini
				- all gemini models
				- all runs
				- all cases
				- all i_dfp
		'''

		s_i_dfp_list = str(i_dfp_list)
		s_chosen_model_list = str(chosen_model_list)


		fname0 = self.fname_agreement_anal_pub_gem%(self.prefix, s_i_dfp_list, s_chosen_model_list, self.suffix)
		fname1 = self.fname_agreement_summ_pub_gem%(self.prefix, s_i_dfp_list, s_chosen_model_list, self.suffix)

		filename0 = os.path.join(self.root_curation, fname0)
		filename1 = os.path.join(self.root_curation, fname1)

		if os.path.exists(filename0) and os.path.exists(filename1) and not force:
			df  = pdreadcsv(fname0,	self.root_curation, verbose=verbose)
			dfg = pdreadcsv(fname1, self.root_curation, verbose=verbose)

			return df, dfg


		dic = {}; icount=-1
			
		for run in run_list:
			print("\t>>>", run)
			for case in case_list:
				print(case, end=' ')

				for i_dfp in i_dfp_list:
					print(i_dfp, end=' ')

					for with_gender in with_gender_list:
						print(f"with_gender={with_gender}", end=' ')

						# filter_both_and_only()
						df_all, df_both, df_only_pubmed, df_only_gemini, mu, std, n, text = \
						self.open_compare_pubmed_x_gemini(run=run, case=case, i_dfp=i_dfp, with_gender=with_gender,
														  chosen_model_list=chosen_model_list, verbose=verbose)

						if df_all is None or df_all.empty:
							continue

						icount += 1
						dic[icount] = {}

						dic2 = dic[icount]
						dic2['run']   = run
						dic2['chosen_model_list'] = chosen_model_list
						dic2['case']  = case
						dic2['i_dfp'] = i_dfp


						dic2['is_seldata'] = self.is_seldata
						dic2['with_gender'] = with_gender
						dic2['n'] = len(df_all)

						dic2['n_both_yes_no'] = len(df_both)
						dic2['agree'] = mu
						dic2['agree_std'] = std
						dic2['n'] = n
						dic2['text'] = text

						
						dic2['n_gemini_yes'] = len(df_all[df_all.gemini == 'Yes'])
						dic2['n_gemini_no']  = len(df_all[df_all.gemini == 'No'])
						dic2['n_pubmed_yes'] = len(df_all[df_all.pubmed == 'Yes'])
						dic2['n_pubmed_no']  = len(df_all[df_all.pubmed == 'No'])
						dic2['n_only_gemini_yes'] = len(df_only_gemini.gemini == 'Yes')
						dic2['n_only_pubmed_yes'] = len(df_only_pubmed.pubmed == 'Yes')

						gem_vals = [dic2['n_gemini_yes'], dic2['n_gemini_no']]
						pub_vals = [dic2['n_pubmed_yes'], dic2['n_pubmed_no']]

						try:
							s_stat, stat, pvalue, dof, expected = calc_stat_chi2(gem_vals, pub_vals)
						except:
							pvalue = None
						
						dic2['pvalue'] = pvalue
						dic2['stat'] = stat
						dic2['dof']  = dof

			print("")

		df = pd.DataFrame(dic).T
		df['fdr'] = fdr(df.pvalue)
		# df = df.sort_values( ['fdr', 'n_both_yes_no', 'n'], ascending=[True, False,False])
		# df.index = np.arange(0, len(df))

		cols = ['run', 'chosen_model_list', 'case', 'i_dfp', 'is_seldata', 'with_gender',
				'n', 'n_gemini_yes', 'n_gemini_no', 'n_pubmed_yes', 'n_pubmed_no',
				'n_both_yes_no', 'agree', 'agree_std', 'fdr', 'pvalue', 'stat', 'dof',
				'n_only_gemini_yes', 'n_only_pubmed_yes']
		df = df[cols]


		# all runs and i_dfp
		cols = ['run', 'case', 'is_seldata', 'with_gender']
		dfg = df.groupby(cols).agg({"agree": ["mean", "std", "count"]}).reset_index()
		dfg.columns = cols + ['agree', 'agree_std', 'n']

		ret = pdwritecsv(df,  fname0, self.root_curation, verbose=verbose)
		ret = pdwritecsv(dfg, fname1, self.root_curation, verbose=verbose)

		return df, dfg


	def calc_mat_consensus(self, mat:List) -> (str, str):
		n_yes, n_no = 0, 0

		for answer in mat:
			answer = answer.lower().strip()

			if answer == 'yes' or answer == 'doubt':
				n_yes += 1
			elif answer == 'no':
				n_no += 1
			else:
				print(f"Error: answer '{answer}'")

		if n_yes > n_no:
			consensus = 'Yes'
		elif n_yes < n_no:
			consensus = 'No'
		else:
			consensus = 'Doubt'

		unanimous = False if n_yes != 0 and n_no != 0 else True

		return consensus, n_yes, n_no, unanimous
		
	def merge_all_3sources_calc_crowd(self, person_list:List, run_list:List, case_list:List, 
									 with_gender_list:List, force:bool=False, verbose:bool=False):


		print(">>> person_list:", person_list, '\n')

		for run in run_list:
			for case in case_list:
				for with_gender in with_gender_list:
					print(f">>> {run} {case} with_gender {with_gender}")
					_ = self.merge_3sources_calc_crowd(with_gender=with_gender, person_list=person_list, run=run, case=case, force=force, verbose=verbose)


	def merge_3sources_calc_crowd(self, with_gender:bool, person_list:List, run:str, case:str, 
								 force:bool=False, verbose:bool=False) -> pd.DataFrame:

		# self.gem.set_gemini_num_model(chosen_model)
		# model_name = self.gem.gemini_model

		fname = self.fname_3_sources%(self.disease, run, case, with_gender)
		filename = os.path.join(self.root_curation, fname)

		if os.path.exists(filename) and not force:
			dfcrowd = pdreadcsv(fname, self.root_curation, verbose=verbose)
			return dfcrowd

		if not isinstance(person_list, list) or person_list == []:
			print("Define person_list")
			return None

		cols_pathw = ['pathway_id', 'pathway']
		
		for i in range(len(person_list)):
			person = person_list[i]
			if verbose: print(i, person, end=", ")

			if i == 0:
				dfrev_pub = self.open_pubmed_agreement(with_gender=with_gender, case=case, person=person, verbose=verbose)
				cols = ['pathway_id', 'pathway', 'n_pmid', 'pub_has_pmid', person]
				dfrev_pub = dfrev_pub[cols]
				cols = ['pathway_id', 'pathway', 'n_pmid', 'pub_has_pmid', person]
				dfrev_pub.columns = cols
				
				dfrev_gem = self.open_gemini_agreement_person(run=run, case=case, person=person, verbose=verbose)
				cols = ['i_dfp', 'pathway_id', 'pathway', 'consensus', 'n_yes']
				dfrev_gem = dfrev_gem[cols]
				cols = ['i_dfp', 'pathway_id', 'pathway', 'gem_consensus', 'n_yes_gem']
				dfrev_gem.columns = cols

				dfcrowd = pd.merge(dfrev_pub, dfrev_gem, how="inner", on=cols_pathw)
				cols = list(dfcrowd.columns)
				cols.remove(person)
				cols = cols + [person]
				dfcrowd = dfcrowd[cols]
			else:

				dfrev_pub = self.open_pubmed_agreement(with_gender=with_gender, case=case, person=person, verbose=verbose)
				cols = ['pathway_id', 'pathway', person]
				dfrev_pub = dfrev_pub[cols]
				cols = ['pathway_id', 'pathway',  person]
				dfrev_pub.columns = cols
				dfcrowd = pd.merge(dfcrowd, dfrev_pub, how="inner", on=cols_pathw)

		if verbose: print("")

		dfcrowd['case'] = case
		dfcrowd['run'] = run
		dfcrowd['with_gender'] = with_gender

		dfcrowd['agree_pubgem'] = [dfcrowd.iloc[i].pub_has_pmid == dfcrowd.iloc[i].gem_consensus for i in range(len(dfcrowd))]

		cols = list(dfcrowd.columns)
		cols = cols[:7] + ['agree_pubgem'] + cols[7:-1]
		dfcrowd = dfcrowd[cols]
		dfcrowd.index = np.arange(0, len(dfcrowd))

		lista_tuple = []
		for i in range(len(dfcrowd)):
			mat = dfcrowd.loc[i, self.reviewers]
			lista_tuple.append(self.calc_mat_consensus(mat))

		dfcrowd['rev_consensus']	   = [x[0] for x in lista_tuple]
		dfcrowd['rev_consensus_n_yes'] = [x[1] for x in lista_tuple]
		dfcrowd['rev_consensus_n_no' ] = [x[2] for x in lista_tuple]
		dfcrowd['unanimous_reviewers'] = [x[3] for x in lista_tuple]

		cols = ['case', 'run', 'with_gender', 'pathway_id', 'pathway','i_dfp',
				'n_pmid', 'pub_has_pmid', 'gem_consensus', 'n_yes_gem',
				'agree_pubgem', 'rev_consensus', 'rev_consensus_n_yes', 
				'rev_consensus_n_no', 'unanimous_reviewers'] + person_list

		dfcrowd = dfcrowd[cols]

		ret = pdwritecsv(dfcrowd, fname, self.root_curation, verbose=verbose)
			
		return dfcrowd


	def calc_all_crowd_accuracies(self, person_list:List, run_list:List, case_list:List, 
								  with_gender_list:List, force:bool=False, 
								  verbose:bool=False) -> (str, pd.DataFrame, pd.DataFrame):

		fname_all = self.fname_csc_all%(self.disease)
		filename_all = os.path.join(self.root_curation, fname_all)

		fname = self.fname_csc_summ%(self.disease)
		filename = os.path.join(self.root_curation, fname)

		fname_text = fname.replace('.tsv', '.txt')
		filename_text = os.path.join(self.root_curation, fname_text)

		if os.path.exists(filename_all) and os.path.exists(filename) and os.path.exists(filename) and not force:
			df_all_crowd = pdreadcsv(fname_all, self.root_curation, verbose=verbose)
			df_accu	 = pdreadcsv(fname, self.root_curation, verbose=verbose)
			all_text = read_txt(fname_text, self.root_curation, verbose=verbose)

			return all_text, df_all_crowd, df_accu

		all_text=''
		dic_list = []
		dic={}; icount=-1

		for run in run_list:
			for case in case_list:
				for with_gender in with_gender_list:
					if force: print(">>>", run, case, with_gender)

					icount += 1
					dic[icount] = {}
					dic2 = dic[icount]

					text, dfcrowd = self.dic2_calc_crowd_accuracies(with_gender=with_gender, person_list=person_list, 
																	run=run, case=case, dic2=dic2, force=force, verbose=verbose)

					if all_text == '':
						all_text = text
					else:
						all_text += '\n\n' + text

					dic_list.append(dfcrowd)

		df_accu = pd.DataFrame(dic).T

		df_all_crowd = pd.concat(dic_list)
		df_all_crowd.index = np.arange(0, len(df_all_crowd))

		ret0 = pdwritecsv(df_all_crowd, fname_all, self.root_curation, verbose=verbose)
		ret1 = pdwritecsv(df_accu, fname, self.root_curation, verbose=verbose)
		ret2 = write_txt(all_text, fname_text, self.root_curation, verbose=verbose)
			

		return all_text, df_all_crowd, df_accu

	# calc_all_accuracies_based_crowd
	def count_all_crowd_agreements(self, with_gender_list:List, person_list:List,  
								   run_list:List, case_list:List,
								   force:bool=False, verbose:bool=False):

		for run in run_list:
			for case in case_list:
				for with_gender in with_gender_list:
					print(">>>", run, case, with_gender)
					_ = self.count_crowd_agreements(with_gender=with_gender, person_list=person_list, 
														 run=run, case=case, 
														 force=force, verbose=verbose)

	# calc_accuracies_based_crowd
	def count_crowd_agreements(self, with_gender:bool, person_list:List, run:str, case:str, 
									force:bool=False, verbose:bool=False) -> pd.DataFrame:

		fname = self.fname_csc%(self.disease, run, case, with_gender)
		filename = os.path.join(self.root_curation, fname)

		if os.path.exists(filename) and not force:
			df_count = pdreadcsv(fname, self.root_curation, verbose=verbose)
			return df_count


		dfcrowd = self.merge_3sources_calc_crowd(with_gender=with_gender, person_list=person_list, run=run, case=case, verbose=verbose)


		cols =  ['case', 'run', 'with_gender', 'pathway_id', 'pathway', 
				 'i_dfp', 'pub_has_pmid', 'n_pmid', 'gem_consensus', 'n_yes_gem', 
				 'rev_consensus', 'rev_consensus_n_yes', 
				 'rev_consensus_n_no', 'unanimous_reviewers']

		df_count = dfcrowd[cols].copy()

		cols_consensus = ['pub_has_pmid', 'gem_consensus', 'rev_consensus']

		lista_tuple = []
		for i in range(len(df_count)):
			lista_tuple.append(self.calc_mat_consensus(df_count.iloc[i][cols_consensus]))

		if lista_tuple == []:
			print("Nothing to do, dfcrowd is empty")
			return None

		df_count['crowd_consensus']		  = [x[0] for x in lista_tuple]
		df_count['crowd_consensus_n_yes'] = [x[1] for x in lista_tuple]
		df_count['crowd_consensus_n_no']  = [x[2] for x in lista_tuple]
		df_count['crowd_unanimous']		  = [x[3] for x in lista_tuple]

		def are_equal_doubt(x, y):
			if x == 'Doubt':
				x = 'Yes'
			if y == 'Doubt':
				y = 'Yes'
				
			return x == y

			
		df_count['crowd_yes']  = [True if df_count.iloc[i].crowd_consensus == 'Yes' or df_count.iloc[i].crowd_consensus == 'Doubt' else False for i in range(len(df_count))]
		df_count['pubmed_has'] = [True if df_count.iloc[i].pub_has_pmid	 == 'Yes' or df_count.iloc[i].pub_has_pmid	 == 'Doubt' else False for i in range(len(df_count))]
		df_count['gemini_yes'] = [True if df_count.iloc[i].gem_consensus	 == 'Yes' or df_count.iloc[i].gem_consensus	 == 'Doubt' else False for i in range(len(df_count))]
		df_count['review_yes'] = [True if df_count.iloc[i].rev_consensus   == 'Yes' or df_count.iloc[i].rev_consensus   == 'Doubt' else False for i in range(len(df_count))]

		df_count['agree_gemini'] = [are_equal_doubt(df_count.iloc[i].crowd_consensus, df_count.iloc[i].gem_consensus) for i in range(len(df_count))]
		df_count['agree_pubmed'] = [are_equal_doubt(df_count.iloc[i].crowd_consensus, df_count.iloc[i].pub_has_pmid) for i in range(len(df_count))]
		df_count['agree_review'] = [are_equal_doubt(df_count.iloc[i].crowd_consensus, df_count.iloc[i].rev_consensus) for i in range(len(df_count))]

		df_count['review_x_pubmed'] = [are_equal_doubt( df_count.iloc[i].pub_has_pmid, df_count.iloc[i].rev_consensus) for i in range(len(df_count))]
		df_count['review_x_gemini'] = [are_equal_doubt( df_count.iloc[i].gem_consensus, df_count.iloc[i].rev_consensus) for i in range(len(df_count))]

		cols = ['run', 'case', 'with_gender', 'pathway_id', 'pathway', 'i_dfp', 
				'crowd_consensus', 'crowd_yes', 'crowd_consensus_n_yes', 'crowd_consensus_n_no', 
				'agree_gemini', 'gem_consensus', 'n_yes_gem', 'gemini_yes',
				'agree_pubmed', 'pub_has_pmid', 'n_pmid', 'pubmed_has', 
				'agree_review', 'review_x_pubmed', 'review_x_gemini', 
				'rev_consensus', 'review_yes', 'crowd_unanimous', 
				'rev_consensus_n_yes', 'rev_consensus_n_no', 'unanimous_reviewers']

		df_count = df_count[cols]

		df_count = df_count.sort_values(['crowd_consensus_n_yes', 'pathway'], ascending=[False, True])
		df_count.index = np.arange(0, len(df_count))

		ret = pdwritecsv(df_count, fname, self.root_curation, verbose=verbose)

		return df_count


	# calc_crowd_and_text
	def dic2_calc_crowd_accuracies(self, with_gender:bool, person_list:List, run:str, case:str, dic2:dict,
								   force:bool=False, verbose:bool=False) -> (str, pd.DataFrame):

		dfcrowd = self.count_crowd_agreements(with_gender=with_gender, person_list=person_list, run=run, case=case, force=force, verbose=verbose)

		if dfcrowd is None or dfcrowd.empty:
			return "", None

		fname = self.fname_csc%(self.disease, run, case, with_gender)
		fname_text = fname.replace('.tsv', '.txt')
		filename = os.path.join(self.root_curation, fname_text)

		if os.path.exists(filename) and not force:
			text = read_txt(fname_text, self.root_curation)
			return text, dfcrowd


		mu_pathws_crowd  = dfcrowd.crowd_yes.mean()
		mu_pathws_pubmed = dfcrowd.pubmed_has.mean()
		mu_pathws_gemini = dfcrowd.gemini_yes.mean()
		mu_pathws_review = dfcrowd.review_yes.mean()

		std_pathws_crowd  = dfcrowd.crowd_yes.std()
		std_pathws_pubmed = dfcrowd.pubmed_has.std()
		std_pathws_gemini = dfcrowd.gemini_yes.std()
		std_pathws_review = dfcrowd.review_yes.std()

		n_crowd = len(dfcrowd)

		text = f"Pathway statistics for {self.disease}, run={run}, case={case} n={n_crowd}, and PubMed search with gender = {with_gender}\n"
		text += '\n'

		text += "Finding pathways (Yes or has pmid):\n"
		text += f"\tCrowd	  {100.*mu_pathws_crowd:.1f}% ({100*std_pathws_crowd:.1f}%)\n"
		text += f"\tPubMed	  {100.*mu_pathws_pubmed:.1f}% ({100*std_pathws_pubmed:.1f}%)\n"
		text += f"\tGemini	  {100.*mu_pathws_gemini:.1f}% ({100*std_pathws_gemini:.1f}%)\n"
		text += f"\tReviewers {100.*mu_pathws_review:.1f}% ({100*std_pathws_review:.1f}%)\n"

		text += '\n'

		text += "Statistical accuracy (using CSC as reference) is:\n"
		accu_pubmed	= dfcrowd.agree_pubmed.mean()
		accu_gemini	= dfcrowd.agree_gemini.mean()
		accu_review = dfcrowd.agree_review.mean()

		std_accu_pubmed	= dfcrowd.agree_pubmed.std()
		std_accu_gemini	= dfcrowd.agree_gemini.std()
		std_accu_review = dfcrowd.agree_review.std()

		text += f"\tPubMed	  {100.*accu_pubmed:.1f}% ({100*std_accu_pubmed:.1f}%)\n"
		text += f"\tGemini	  {100.*accu_gemini:.1f}% ({100*std_accu_gemini:.1f}%)\n"
		text += f"\tReviewers {100.*accu_review:.1f}% ({100*std_accu_review:.1f}%)\n"

		# text += '\n'
		# text += "Accuracy is calculate, for example, positive if PubMed or Gemini are following Crowd (both Yes or both No)\n"
		text += '\n'

		mu_review_x_pubmed = dfcrowd.review_x_pubmed.mean()
		std_review_pubmed  = dfcrowd.review_x_pubmed.std()

		mu_review_x_gemini = dfcrowd.review_x_gemini.mean()
		std_review_gemini  = dfcrowd.review_x_gemini.std()

		text += "Reviewers' consensus, comparing to:\n"
		text += f"\tGemini: {100*mu_review_x_gemini:.1f}% ({100*std_review_gemini:.1f}%)\n"
		text += f"\tPudMed: {100*mu_review_x_pubmed:.1f}% ({100*std_review_pubmed:.1f}%)\n"

		text += '\n'

		ret = write_txt(text, fname_text, self.root_curation, verbose=verbose)

		dic2['case'] = case 
		dic2['run'] = run
		dic2['n'] = n_crowd 
		dic2['with_gender'] = with_gender 
		dic2['perc_pathws_crowd'] = mu_pathws_crowd
		dic2['std_pathws_crowd']  = std_pathws_crowd
		dic2['perc_pathws_pubmed'] = mu_pathws_pubmed
		dic2['std_pathws_pubmed']  = std_pathws_pubmed
		dic2['perc_pathws_gemini'] = mu_pathws_gemini
		dic2['std_pathws_gemini']  = std_pathws_gemini
		dic2['perc_pathws_review'] = mu_pathws_review
		dic2['std_pathws_review']  = std_pathws_review
		dic2['accu_pubmed']		= accu_pubmed
		dic2['std_accu_pubmed'] = std_accu_pubmed
		dic2['accu_gemini']		= accu_gemini
		dic2['std_accu_gemini'] = std_accu_gemini
		dic2['accu_review']	 = accu_review
		dic2['std_accu_review'] = std_accu_review
		dic2['perc_review_x_pubmed'] = mu_review_x_pubmed
		dic2['std_review_x_pubmed']  = std_review_pubmed
		dic2['perc_review_x_gemini'] = mu_review_x_gemini
		dic2['std_review_x_gemini']  = std_review_gemini

		return text, dfcrowd



	def barplot_accuracy(self, run:str, which:str, with_gender:bool, with_gender_list:List, 
						 person_list:List, run_list:List, case_list:List, chosen_model_list:List,
						 width:int=1100, height:int=600, 
						 title0="%s for %s with gender %s",
						 fontsize:int=14, fontcolor:str='black',
						 margin:dict=dict( l=20, r=20, b=100, t=120, pad=4), plot_bgcolor:str="whitesmoke",
						 xaxis_title:str="comparisons", 
						 minus_case:float=-0.15, minus_test:float=-.1, minus_mnem:float=-0.05, 
						 annot_fontfamily:str="Arial, monospace", annot_fontsize:int=14, 
						 annot_fontcolor:str='black', savePlot:bool=True, alpha=0.05, 
						 force:bool=False, verbose:bool=False):


		_, dfcrowd, df_accu = self.calc_all_crowd_accuracies(person_list=person_list, run_list=run_list, case_list=case_list,
														   with_gender_list=with_gender_list, force=False, verbose=verbose)
		
		_= self.open_stat_3sources_binary_vars(run_list=run_list, with_gender_list=with_gender_list, force=False, verbose=verbose)

		if self.dfbinv is None:
			print("Warning: dfbinv is none")


		df_accu_gen = df_accu[(df_accu.run==run) & (df_accu.with_gender==with_gender)]
		dfcrowd2	= dfcrowd[(dfcrowd.run==run) & (dfcrowd.with_gender==with_gender)]

		'''
		cols = ['case', 'run', 'with_gender', 'perc_pathws_crowd', 'perc_pathws_pubmed',
				'perc_pathws_gemini', 'perc_pathws_review', 'accu_pubmed', 'accu_gemini',
				'accu_review', 'perc_review_x_pubmed', 'perc_review_x_gemini']
		'''

		which = which.lower()
		
		if which == 'reviewer' or which == 'human':
			sel_cols   = ['perc_review_x_gemini', 'perc_review_x_pubmed']
			anal_cols  = ['review_x_gemini', 'review_x_pubmed',]
			mnem_cols  = ['REVxGEM', 'REVxPubMed', ]

			compare_list = ['REVxPubMed x REVxGEM']
			shift_i = 1

			test = 'Reviewers'
			yaxis_title='agreement (%)'
			colors_blue=["darkcyan", "blue"]

		elif which == 'pathway':
			sel_cols   = ['perc_pathws_crowd', 'perc_pathws_gemini', 'perc_pathws_pubmed', 'perc_pathws_review']
			anal_cols  = ['crowd_yes', 'gemini_yes', 'pubmed_has', 'review_yes']
			mnem_cols  = ['CSC', 'GEM', 'PubMed', 'REV']

			compare_list = ['CSC', 'Pathw_GemxCrowd', 'Pathw_PubxCrowd', 'Pathw_RevxCrowd']
			shift_i = 0

			test = 'Pathways'
			yaxis_title='percentage of Yes (%)'
			colors_blue=["blueviolet", "darkcyan", "blue", "navy"]

		else:
			sel_cols   = ['accu_gemini', 'accu_pubmed', 'accu_review']
			anal_cols  = ['accu_gemini', 'accu_pubmed', 'accu_review']
			mnem_cols  = ['GEM', 'PubMed', 'REV']	
			sources	= ['Gemini', 'PubMed', 'Reviewers']	

			compare_list = ['Accu_PubxGem', 'Accu_RevxGem']
			shift_i = 1

			test = 'Accuracy'
			yaxis_title='agreement with CSC (%)'
			colors_blue=["darkcyan", "blue", "navy"]

		fig = go.Figure()
		
		title=title0%(test, run, with_gender)

		x_num=0
		dic={}; icount=-1
		for case in case_list:

			dfa = df_accu_gen[df_accu_gen.case == case]
			dfcrowd3 = dfcrowd2[dfcrowd2.case  == case]

			n = dfa.iloc[0]['n']

			L = len(sel_cols)
			for i in range(L):

				col  = sel_cols[i]
				mnem = mnem_cols[i]

				if col.startswith('perc_'):
					col_std = col.replace('perc_', 'std_')
				else:
					col_std = 'std_' + col

				mu  = dfa.iloc[0][col]
				std = dfa.iloc[0][col_std]

				if i == 0:
					compare = None
					pvalue=None,
					aster = ''
				else:

					compare = compare_list[i - shift_i]
					pvalue = self.which_stat_accuray(run=run, case=case, with_gender=with_gender, compare=compare)
					aster = stat_asteristics(pvalue,NS='--')

				icount += 1
				dic[icount] = {}
				dic2 = dic[icount]

				dic2['run'] = run
				dic2['case'] = case
				dic2['with_gender'] = with_gender
				dic2['test'] = col
				dic2['mnem'] = mnem
				dic2['compare'] = compare
				dic2['n'] = n
				dic2['mu'] = mu
				dic2['std'] = std

				# alpha/2 Bonferroni correction, 2 comparisons.
				error, cinf, csup, SEM, stri = calc_confidence_interval_param(mu, std, n, alpha=alpha/2, two_tailed=True)
				
				dic2['error'] = error
				dic2['SEM']   = SEM
				dic2['cinf']  = cinf
				dic2['csup']  = csup

				x_num += 1
				label = f"{mnem}"
				fig.add_trace(go.Bar(x=[x_num], y=[mu], marker_color=colors_blue[i], 
							  error_y=dict(type='data', array=[error]), name=label))

				fig.add_annotation(
					x=x_num,
					y=minus_mnem,
					xref="x",
					yref="y",
					text=mnem,
					showarrow=False,
					font=dict(
						family="Arial, monospace",
						size=annot_fontsize,
						color=fontcolor
						),
					align="center", )
				
				if i == L-1:
					fig.add_annotation(
						x=x_num-(L/2)+0.5,
						y=minus_test,
						xref="x",
						yref="y",
						text=f"{case} n={n}",
						showarrow=False,
						font=dict(
							family="Arial, monospace",
							size=annot_fontsize,
							color=fontcolor
							),
						align="center", )


				if i == 0:
					dic2['pvalue'] = None
					dic2['asteristics'] = None
				else:
					dic2['pvalue'] = pvalue
					dic2['asteristics'] = aster
					
					fig.add_annotation(
						x=x_num,
						y=mu+error+0.05,
						xref="x",
						yref="y",
						text=f"{aster}", # <br>{pvalue:.2e}
						showarrow=False,
						font=dict(
							family="Arial, monospace",
							size=annot_fontsize,
							color=fontcolor
							),
						align="center", )
		   
			x_num+=1

		fig.update_layout(
					autosize=True,
					title=title,
					width=width,
					height=height,
					plot_bgcolor=plot_bgcolor,
					xaxis_title=xaxis_title,
					yaxis_title=yaxis_title,
					xaxis_showticklabels=False,
					# yaxis_range=[-20, 110],
					showlegend=False,
					font=dict(
						family="Arial",
						size=fontsize,
						color="Black"
					),
		)

		df = pd.DataFrame(dic).T
		
		if savePlot:
			figname = title_replace(title)

			figname = os.path.join(self.root_figure, figname+'.html')

			fig.write_html(figname)
			if verbose: print(">>> HTML and png saved:", figname)
			fig.write_image(figname.replace('.html', '.png'))

		fname = self.fname_CSC_conf_table_summary_stats%(test, self.disease, run, chosen_model_list, self.suffix)
		filename = os.path.join(self.root_curation, fname)

		if not os.path.exists(filename) or force:
			ret = pdwritecsv(df, fname, self.root_curation, verbose=verbose)

		return fig, df


	def calc_similarity_false_true(self, df2, df3):
		'''
			s_stat, stat, pvalue, dof, expected = calc_similarity_false_true(df_male_false, df_male_true)
		'''
		mat_agree_yesno0 = list(df2.n_agree_yes) + list(df2.n_agree_no)
		mat_notag_yesno0 = list(df2.n_yes - df2.n_agree_yes)+ list(df2.n_no - df2.n_agree_no)

		mat_agree_yesno1 = list(df3.n_agree_yes) + list(df3.n_agree_no)
		mat_notag_yesno1 = list(df3.n_yes - df3.n_agree_yes) + list(df3.n_no - df3.n_agree_no)

		vals0 = [mat_agree_yesno0, mat_notag_yesno0]
		vals1 = [mat_agree_yesno1, mat_notag_yesno1]
		
		s_stat, stat, pvalue, dof, expected = calc_stat_chi2(vals0, vals1)
		return s_stat, stat, pvalue, dof, expected

