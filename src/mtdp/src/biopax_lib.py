#!/usr/bin/python
#!python
# -*- coding: utf-8 -*-
# Created on 2023/06/25
# Udated  on 2024/10/11; 2024/05/06; 2024/03/29; 2023/08/28; 2023/08/16
# @author: Flavio Lichtenstein
# @local: Bioinformatics: CENTD/Molecular Biology; Instituto Butatan

import numpy as np
import gzip, os, sys
import pandas as pd
import pickle, shutil
from collections import defaultdict, Counter, OrderedDict
from typing import Optional, Iterable, Set, Tuple, Any, List
from datetime import datetime

import pystow, mygene
from lxml import etree
from tabulate import tabulate
from tqdm.auto import tqdm

import pybiopax
from pybiopax.biopax import *

from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
from dash.dash_table.Format import Format, Scheme, Trim

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from   matplotlib_venn import venn2, venn2_circles

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio
from   plotly.subplots import make_subplots

from scipy.stats import shapiro
from scipy import stats
from scipy.stats import norm

from Basic import *
from gene_lib import *
from config_lib import *
from stat_lib import *
from reactome_lib import *
from biomart_lib import *

from graphic_lib import *

sel_degs_pathways_colors=['navy', 'red', 'darkcyan', 'darkgreen', 'orange', 'brown', 'darksalmon',
		'magenta', 'darkturquoise', 'orange', 'darkred', 'indigo', 'magenta', 'maroon', 'black',
		'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgrey', 'olivedrab', 'navy'] + plotly_colors_proteins

# print('recursionlimit', sys.getrecursionlimit())
# sys.setrecursionlimit(20000)
# print('recursionlimit', sys.getrecursionlimit())

class Biopax(object):
	def __init__(self, gene_protein:str, s_omics:str, project:str, s_project:str, root0:str,
				 case_list:List, has_age:bool=True, has_gender:bool=True, clone_objects:bool=False,
				 exp_normalization:bool=None, geneset_num:int=0, num_min_degs_for_ptw_enr:int=3,
				 tolerance_pathway_index:float=.15, s_pathw_enrichm_method:str='enricher',
				 root_data_aux:str='../../../data_aux/',
				 abs_lfc_cutoff_inf:float=0.40, fdr_ptw_cutoff_list:List=[],
				 num_of_genes_list:List=[3], lfc_list = [], fdr_list = [],
				 min_lfc_modulation:float=0.20, type_sat_ptw_index:str='linear_sat', saturation_lfc_index:float=5):

		self.gene = Gene(root_data_aux=root_data_aux)
		self.cfg  = Config(project, s_project, case_list, root0)
		self.med_max_ptw = 'median'

		self.project = project
		self.s_project = s_project
		self.s_pathw_enrichm_method = s_pathw_enrichm_method
		self.selected_pivot_pathway_list = []
		self.selected_pivot_symb_list = []

		if lfc_list == []:
			lfc_list = np.round(np.arange(1.0, -0.01, -.025), 3)
			lfc_list[-1] = 0

		self.lfc_list = lfc_list
		if fdr_list == []:
			fdr_list = np.arange(0.05, 0.76, .01)
		self.fdr_list = fdr_list

		self.dfsim = None


		if min_lfc_modulation is None:
			min_lfc_modulation = 0.20

		self.type_sat_ptw_index   = type_sat_ptw_index
		self.saturation_lfc_index = saturation_lfc_index
		self.min_lfc_modulation   = min_lfc_modulation

		self.gene_protein = gene_protein
		self.s_omics = s_omics
		self.s_gene_protein = 'Protein' if gene_protein == 'protein' else 'Gene'
		self.s_deg_dap = 'DAP' if gene_protein == 'protein' else 'DEG'

		self.has_gender = has_gender
		self.has_age = has_age
		self.group_index = -1

		self.case, self.group, self.age, self.gender = '','','',''

		''' renaming bad symbols in dflfc_ori '''
		self.symbols2_list = []
		self.locs_list = []
		self.dfnot = None

		self.fname_lfc_table0 = f"{s_project}_ALL_LFC_%s_x_CTRL_%s.tsv"
		self.fname_final_lfc_table0 = f"{s_project}_ALL_corrected_LFC_%s_x_CTRL_%s.tsv"

		self.fname_lfc_nodup_table0 = f"{s_project}_NO_DUP_LFC_%s_x_CTRL_%s.tsv"
		self.fname_final_ori = ''
		self.fname_all_dfenr = 'pathway_all_list.tsv'

		''' enricher_GO_Biological_Process_2021_taubate_covid19_proteomics_for_g3_female_elder_x_ctrl_not_normalized_cutoff_lfc_0.200_fdr_0.700_pathway_pval_0.050_fdr_0.500_num_genes_3.tsv '''
		fname_enrich_table0 = f"{s_pathw_enrichm_method}_%s_{s_project}_{s_omics}_for_%s_x_ctrl_%s"
		self.fname_enrich_table0 = fname_enrich_table0

		self.fname_enr_simulation = f"enr_simulation_{s_project}_%s_%s.tsv"

		self.fname_stringdb  = ''

		self.case_list = case_list

		if has_gender:
			self.group_female_list = [x for x in case_list if '_female' in x]
			self.group_male_list   = [x for x in case_list if '_male' in x]
			self.list_order = self.group_female_list + self.group_male_list

			self.group_list = [x.replace('_female', '') for x in self.group_female_list]
		else:
			self.group_female_list, self.group_male_list = None, None
			self.list_order = case_list
			self.group_list = case_list

		self.group_colors = ['gray', 'blue', 'orange', 'red'] + list(sel_degs_pathways_colors)
		self.group_colors = self.group_colors[:len(self.group_list)]

		''' expression normalization: if None -> not normalized '''
		self.exp_normalization = exp_normalization
		self.normalization = 'not_normalized' if exp_normalization is None else exp_normalization

		self.set_db(geneset_num)

		self.model = None
		self.which_model = 'stringdb'
		self.df_onto_list = []
		self.level = 0
		self.clone_objects = clone_objects
		self.recalc_posi = False

		self.quantile_list  = np.round(np.arange(0, 1, 0.05), 2)
		self.abs_lfc_cutoff_inf = abs_lfc_cutoff_inf

		if fdr_ptw_cutoff_list == []:
			fdr_ptw_cutoff_list = np.arange(0.05, 0.80, 0.05)

		if num_of_genes_list is None or not isinstance(num_of_genes_list, list):
			num_of_genes_list = [3]

		self.fdr_ptw_cutoff_list = fdr_ptw_cutoff_list
		self.num_of_genes_list   = num_of_genes_list

		self.reset_degs_and_df_enr()

		self.df_summ = None
		self.dfsimb_perc = None
		self.dfpiv, self.dfpiv_symbs_cases = None, None


		self.fname_all_fdr_lfc_correlation = 'all_fdr_lfc_correlation.tsv_abs_lfc_cutoff_inf_%.3f.json'
		self.fname_dic_fdr_lfc_correlation = 'dic_fdr_lfc_correlation_case_%s_abs_lfc_cutoff_inf_%.3f.json'

		self.fname_degs_and_pathways_summary = "degs_and_pathways_summary.tsv"
		self.fname_degs_simulation = 'degs_simulation_for_%s_lfc.tsv'

		self.fname_txt_updegs = "upreg_degs_for_case_%s_lfc_%.2f_fdr_%.2f_%s.txt"
		self.fname_txt_dwdegs = "downreg_degs_for_case_%s_lfc_%.2f_fdr_%.2f_%s.txt"

		self.fname_enr_gene_stat = 'enrichment_pathways_gene_statistics_for_case_%s_and_database_%s_%s.tsv'

		self.fname_summ_comparing_2cases = 'comparing_2cases_pathw_%s_comparing_%s_x_%s.tsv'
		self.fname_pathway_index = 'pathway_%s_type_%s_saturation_%s_index_%s.tsv'
		self.fname_pathway_case_index = 'pathway_index_%s_%s_%s.tsv'
		self.fname_one_pathway_gene_mods = 'pathway_%s_gene_modulations_%s_%s.tsv'

		self.fname_degs_summary = 'summary_genes_per_case.tsv'
		self.fname_pathway_summary  = 'summary_genes_in_pathways_per_case_%s.tsv'
		self.fname_pathways_per_case = 'summary_pathways_per_case.tsv'
		self.fname_cutoff_table = 'cutoff_table_all_cases_quantiles_for_col_%s_for_geneset_%d_%s_%s.tsv'

		self.fname_summ_deg_ptw = 'summary_bca_x_default_cutoffs_result.tsv'

		self.param_defaults = 1, 0.05, -1, -1, -1, -1

		self.root_assets = './assets'
		if not os.path.exists(self.root_assets):
			os.mkdir(self.root_assets)

		self.root0 = root0
		self.root_result    = create_dir(root0, 'results')
		self.root_table_ori = create_dir(root0, 'data')
		self.root_enrich    = create_dir(root0, 'enrichment_analyses')
		self.root_ressum    = create_dir(root0, 'res_summ')
		self.root_figure    = create_dir(root0, 'figures')
		self.root_config    = create_dir(root0, 'config')
		self.root_llm	    = create_dir(root0, 'llm')
		self.root_nc	    = create_dir(root0, 'non_coding')
		self.root_enrichment	 = create_dir(root0, 'enrichment_analyses')
		self.root_enrich_random  = create_dir(root0, 'enrichment_random')
		self.root_ptw_modulation = create_dir(root0, 'pathway_modulation')

		''' ---- Affymetrix ---'''
		self.fname_affy = 'Human_Agilent_WholeGenome_4x44k_v2_MSigDB_v71.tsv'
		self.root_affy	   = create_dir(root0, 'affy')

		''' affymetrix experiment table - probe x symbols '''
		self.df_gpl = None


		# ----- root_data_aux = where data is ----Reactomea_aux, 'refseq')
		self.root_hgnc	     = create_dir(root_data_aux, 'hgnc')
		self.root_affymetrix = create_dir(root_data_aux, 'affymetrix')
		self.root_chibe      = create_dir(root_data_aux, 'root_chibe')
		self.root_owl	     = create_dir(self.root_chibe, 'owl')
		self.root_owl_data   = create_dir(self.root_owl, 'data')
		self.root_owl_reactome = create_dir(self.root_owl_data, 'reactome')
		self.root_owl_stringdb = create_dir(self.root_owl_data, 'stringdb')

		self.kegg_fname	= 'kegg_pathways.tsv'
		self.fname_kegg_pathways  = 'kegg_pathways.tsv'
		self.fname_kegg_gene_comp = 'kegg_gene_compound.tsv'

		self.root_owl_dummy   = create_dir(self.root_owl, 'result')
		self.root_owl_result0 = create_dir(self.root_owl_dummy, 'reactome')

		self.pathway_id = 'DUMMY'
		self.root_owl_result = os.path.join(self.root_owl_result0, self.pathway_id)


		self.dbs_list = \
		['Reactome_2022', 'WikiPathway_2021_Human', 'KEGG_2021_Human', 'KEGG_2015', \
		 'BioPlanet_2019', 'MSigDB_Hallmark_2020', 'ChEA_2016', \
		 'GO_Biological_Process_2021', 'GO_Molecular_Function_2021', \
		 'GO_Cellular_Component_2021']

		self.num_min_degs_for_ptw_enr = num_min_degs_for_ptw_enr

		self.tolerance_pathway_index = tolerance_pathway_index

		self.reactome = Reactome(root_data_aux=root_data_aux)
		self.root_reactome	   = self.reactome.root_reactome
		self.root_reactome_data  = self.reactome.root_reactome_data

		self.dfr = None
		''' below, df reactome with pathway_original '''
		self.df_reactome_gmt, self.df_enr_reactome = None, None

	def set_which_model(self, which_model:str):
		self.which_model = which_model

	def open_reactome_gmt_for_pathway_analysis(self) -> bool:
		if self.df_reactome_gmt is not None and not self.df_reactome_gmt.empty:
			return True

		self.df_reactome_gmt = None
		df_reactome = self.reactome.open_reactome_gmt()
		if df_reactome is None or df_reactome.empty:
			return False

		cols = ['pathway_id', 'pathway', 'genes', 'n']
		df_reactome = df_reactome[cols]
		df_reactome.columns =  ['pathway_id', 'pathway_original', 'genes_pathway', 'ngenes_pathway']
		self.df_reactome_gmt = df_reactome

		return True


	def merge_reactome(self, df_enr:pd.DataFrame) -> pd.DataFrame:

		if isinstance(self.df_enr_reactome, pd.DataFrame) and not self.df_enr_reactome.empty:
			return self.df_enr_reactome

		ret = self.open_reactome_gmt_for_pathway_analysis()
		if ret:
			dfn = pd.merge(df_enr, self.df_reactome_gmt, how="inner", on='pathway_id')
			self.df_enr_reactome = dfn
		else:
			self.df_enr_reactome = None

		return dfn


	def reactome_find_genes_in_pathway(self, pathway:str, _type:str='pathway') -> (str, List):
		'''
			_types: pathway, pathway_id
			return pathway -> pathway or pathway_id -> pathway and all genes
			if no results or > 1: return '', []
		'''

		ret = self.open_reactome_gmt_for_pathway_analysis()
		if not ret:
			return None, []

		if _type == 'pathway':
			dfa = self.df_reactome_gmt[self.df_reactome_gmt.pathway_original == pathway]

			if dfa.empty:
				dfa = self.df_reactome_gmt[self.df_reactome_gmt.pathway_original.str.contains(pathway)]
		else:
			dfa = self.df_reactome_gmt[self.df_reactome_gmt.pathway_id == pathway]

		if dfa.empty:
			print(f"Nothing found for {_type}=='{pathway}'")
			return '', []

		if len(dfa) > 1:
			print(f"There are {len(dfa)} results for {_type}=='{pathway}'")
			print(">>>", "; ".join(dfa.pathway_original))
			return '', []

		genes_pathway = dfa.iloc[0].genes_pathway
		if _type == 'pathway':
			s_pathway = dfa.iloc[0].pathway_id
		else:
			s_pathway = dfa.iloc[0].pathway_original

		if isinstance(genes_pathway, str):
			genes_pathway = eval(genes_pathway)

		genes_pathway = list(genes_pathway)
		return s_pathway, genes_pathway



	def merge_refseq_info_with_stringdb_mapping(self, verbose:bool=False):

		print(f"Before merging: {len(self.gene.df_refseq)} genes")

		dfsdb = self.open_string_mapping()
		if dfsdb is None: return None
		dfrs = self.gene.open_refseq_ncbi_info()
		if dfrs is None: return None

		''' save backup: rename file '''
		try:
			nums = [x.split('#')[1].replace('.tsv','') for x in os.listdir(self.root_refseq) if 'refseq_gene_ncbi' in x]
		except:
			nums = ['0']

		if len(nums) > 0:
			nums = [int(x) for x in nums]
			next_num = np.max(nums)+1

			src_file = self.fname_refseq_ncbi
			dest_file = self.fname_refseq_ncbi.replace('.tsv', '#%d.tsv'%(next_num))

			src_file = os.path.join(self.root_refseq, src_file)
			dest_file = os.path.join(self.root_refseq, dest_file)

			try:
				os.rename(src_file, dest_file)
			except:
				print("could not rename %s."%(src_file))
		''' -------------------------------------------------'''

		dfn = pd.merge(dfrs, dfsdb, how='outer', on='symbol')

		df2 = self.gene.df_refseq
		df2 = df2[~df2.symbol.isin(dfn.symbol)]

		dff = pd.concat([df2, dfn])

		''' only human '''
		dff = dff.sort_values('symbol')
		dff.index = np.arange(0, len(dff))

		ret = pdwritecsv(dff, self.fname_refseq_ncbi, self.root_refseq, verbose=verbose)
		self.gene.df_refseq = dff

		print("Merged %d genes"%(len(dff)))
		return dff


	def open_string_mapping(self, verbose:bool=False) -> pd.DataFrame:
		fullname = os.path.join(self.root_result, self.fname_string_mapping)

		if not os.path.exists(fullname):
			print('Table does not exists: %s'%(fullname))
			return None

		df = pdreadcsv(self.fname_string_mapping, self.root_result, verbose=verbose)
		df.columns = ['query_index', 'query_item', 'ensemblid', 'symbol', 'refseq_desc']
		print("Found %d genes for stringdb mapping"%(len(df)))
		cols =  ['ensemblid', 'symbol', 'refseq_desc']
		df = df[cols]

		return df

	def set_lfc_names(self):

		fname_final_ori = self.fname_final_lfc_table0%(self.case, self.normalization)
		self.fname_final_ori = fname_final_ori
		fname_ori	   = self.fname_lfc_table0%(self.case, self.normalization)

		if (self.age == '') and (self.gender == ''):
			title = self.group

		elif self.age == '':
			title = f'{self.group} {self.gender}'

		elif self.gender == '':
			title = f'{self.group} {self.age}'
		else:
			title = f'{self.group} {self.gender} {self.age}'

		title += f" ({self.normalization})"

		return fname_final_ori, fname_ori, title


	def review_proteomics_table(self, fname_final_ori:str, fname_ori:str, verbose:bool=False) -> pd.DataFrame:
		fullname = os.path.join(self.root_result, fname_ori)

		if not os.path.exists(fullname):
			print(f"Error: could not find table '{fullname}'")
			raise Exception('stop: fix dup_dflfc_ori()')

		dflfc_ori = pdreadcsv(fname_ori, self.root_result, verbose=verbose)

		if dflfc_ori is None or dflfc_ori.empty:
			print(f"Error: could not find data for table '{fname_ori}'")
			raise Exception('stop: fix dup_dflfc_ori()')

		cols = list(dflfc_ori.columns)
		if "abs_lfc" not in cols:
			dflfc_ori["abs_lfc"] = np.abs(dflfc_ori.lfc)

		if "symbol_pipe" not in cols:
			dflfc_ori["symbol_pipe"] = None

		dflfc_ori = self.remove_repeated_according_max_LFC(dflfc_ori)

		if 'ensembl_id' in cols and 'gene_biotype' in cols:
			pass
		else:
			dflfc_ori = self.biomart_fix_dflfc_ori(dflfc_ori)

		pdwritecsv(dflfc_ori, fname_final_ori, self.root_result, verbose=True)

		return dflfc_ori


	def remove_repeated_according_max_LFC(self, dflfc_ori:pd.DataFrame) -> pd.DataFrame:
		'''
		remove_repeated_according_max_LFC()
			add: symbol_prev
			fix symbol with replace_synonym_to_symbol()
			remove repeated symbols, conserve the highest abs(LFC)
		'''

		if 'symbol_prev' not in dflfc_ori.columns:
			dflfc_ori['symbol_prev'] = dflfc_ori.symbol
			dflfc_ori.loc[:, 'symbol'] = [self.gene.replace_synonym_to_symbol(x) for x in dflfc_ori.symbol]

		dflfc_ori = dflfc_ori.sort_values(['symbol', 'abs_lfc'], ascending=[True, False]).copy()
		dflfc_ori.index = np.arange(0, len(dflfc_ori))

		''' remove duplicates - stay with the biggest LFC'''
		previous = ''; goods = []
		for i in range(len(dflfc_ori)):

			if not isinstance(dflfc_ori.iloc[i].symbol, str):
				goods.append(False)
			elif dflfc_ori.iloc[i].symbol != previous:
				previous = dflfc_ori.iloc[i].symbol
				goods.append(True)
			else:
				goods.append(False)

		dflfc_ori = dflfc_ori[goods].copy()
		dflfc_ori.index = np.arange(0, len(dflfc_ori))

		return dflfc_ori


	def biomart_fix_dflfc_ori(self, dflfc_ori:pd.DataFrame) -> pd.DataFrame:
		if self.symbols2_list != []:
			for symbol_new, symbol_old in self.symbols2_list:
				dflfc_ori.loc[ dflfc_ori.symbol == symbol_old, 'symbol'] = symbol_new

		print(">>> running biomart_fix_dflfc_ori() ...")
		bm = Biomart()

		dfbm = bm.open_biomart_hsapiens()

		if dfbm is None or dfbm.empty:
			print("Please run biomart_wget.ipynb -> download_biomart_hsapiens()")
			raise Exception("stop: biomart_fix_dflfc_ori()")

		cols_dfbm = ['symbol', 'gene_biotype', 'description', 'ensembl_transcript_id', 'ensembl_gene_id',
					 'chromosome', 'start_position', 'end_position']

		# remove description
		cols_dflfc = ['entry_id', 'symbol', 'uniprot_name', 'lfc', 'abs_lfc', 'pval', 'fdr',
					  'mean_exp', 't', 'B', 'symbol_pipe', 'symbol_prev']

		dfn = pd.merge(dflfc_ori[cols_dflfc], dfbm[cols_dfbm], how='inner', on='symbol')

		''' could not map '''
		dfnot = dflfc_ori[~dflfc_ori.symbol.isin(dfn.symbol)].copy()

		if len(dfnot) > 0:
			print(f"There are {len(dfnot)} genes not mapped in dfnot.")

			''' add new columns to merge '''
			cols = ['gene_biotype', 'ensembl_transcript_id', 'ensembl_gene_id', 'chromosome', 'start_position', 'end_position']
			for col in cols:
				dfnot[col] = None

			dflfc_new = pd.concat([dfn, dfnot])
		else:
			print(f"All symbols were mapped in df biomart.")
			dflfc_new = dfn

		dflfc_new.index = np.arange(0, len(dflfc_new))

		# renaming to ensembl_id
		cols_new = ['entry_id', 'symbol', 'uniprot_name', 'lfc', 'abs_lfc', 'pval', 'fdr', 'mean_exp', 't', 'B',
					'symbol_pipe', 'symbol_prev', 'biotype', 'description', 'ensembl_transcript_id',
					'ensembl_id', 'chromosome', 'start_position', 'end_position']

		dflfc_new.columns = cols_new

		cols_order = ['entry_id', 'symbol', 'ensembl_id', 'uniprot_name', 'biotype', 'description', 'lfc', 'abs_lfc', 'pval', 'fdr',
					  'mean_exp', 't', 'B', 'symbol_pipe', 'symbol_prev', 'ensembl_transcript_id',
					  'chromosome', 'start_position', 'end_position']
		dflfc_new = dflfc_new[cols_order]

		''' fixing locs_list -= mannually mapped LOCs '''
		if self.locs_list != []:
			for symbol, ensembl_id, biotype in self.locs_list:
				print(">>> changing", symbol, ensembl_id, biotype)
				dflfc_new.loc[ dflfc_new.symbol == symbol, ('ensembl_id', 'biotype')] = ensembl_id, biotype

		dfa = dflfc_new[ pd.isnull(dflfc_new.ensembl_id)]
		if dfa.empty:
			print("All genes are well mapped according to ensembl_id")

			dflfc_new = self.remove_repeated_according_max_LFC(dflfc_new)
		else:
			print(f"It remaines {len(dfa)} genes not well mapped according to ensembl_id")

		self.dfnot = dfa

		return dflfc_new



	def open_dflfc_ori(self, verbose:bool=False):

		self.dflfc_ori = None

		fname_final_ori, fname_ori, title_dummy = self.set_lfc_names()
		fullname = os.path.join(self.root_result, fname_final_ori)

		if not os.path.exists(fullname):
			print(f"Could not find {fullname},  reviewing ...")

			if self.s_omics == 'microarray':
				_ = self.review_LFC_table_with_affy_annot_wo_case(force=False, calc_interm_tables=False, verbose=False)

			elif self.s_omics == 'proteomics':
				_ = self.review_proteomics_table(fname_final_ori, fname_ori, verbose=verbose)

			else:
				print(f"There is no fix duplication method to {self.s_omics}")
				return False

		if not os.path.exists(fullname):
			print(f"Error: could not find {fullname}")
			return False

		dflfc_ori = pdreadcsv(fname_final_ori, self.root_result, verbose=verbose)
		if verbose: print(f">>> dflfc_ori: contains {len(dflfc_ori)} probes.")

		goods = [True if isinstance(x, str) else False for x in dflfc_ori.symbol]
		dflfc_ori = dflfc_ori[goods].copy()
		dflfc_ori.index = np.arange(0, len(dflfc_ori))

		self.dflfc_ori = dflfc_ori
		self.valid_genes = list(dflfc_ori.symbol)
		self.n_total_genes = len(dflfc_ori)

		if verbose: print(f">>> dflfc_ori: has {self.n_total_genes} valid symbols.")

		ret = self.check_dflfc_ori_duplicates()
		if not ret:
			print(f"Error: problems with table: '{fname_final_ori}'")


		return ret

	def check_dflfc_ori_duplicates(self):
		dfg = self.dflfc_ori.groupby('symbol').count().reset_index().iloc[:, :2]
		dfg.columns = ['symbol', 'n']
		dfn = dfg[dfg.n > 1]

		self.dfn = dfn
		if not dfn.empty:
			print("There are repeated symbols in dflfc_ori, see self.dfn")
			print("Rerun: review_LFC_table_with_affy_annot_wo_case(force=True, calc_interm_tables=True, verbose=False)")
			return False

		return True


	def open_df_lfc_complete(self, verbose:bool=False):

		self.dflfc_ori = None

		fname_final_ori, fname_ori, title = self.set_lfc_names()
		self.title = title

		fullname = os.path.join(self.root_result, fname_nodup)

		if not os.path.exists(fullname):
			df = pd.DataFrame({})
			print('LFC table does not exists: %s'%(fullname))
			return False

		if verbose: print("reading df_lfc ori", self.group, self.gender, self.age)
		df = pdreadcsv(fname_nodup, self.root_result)

		''' microarray columns from limma'''
		cols1 = ['probe', 'symbol', 'geneid', 'description', 'logFC', 'meanExpr',
				 't.stat', 'p-value', 'fdr', 'B', 'chr.range', 'org.chromosome',
				 'forward.reverse', 'nuc.sequence', 'gemmaid', 'go.term']

		cols3 = ['entry_id', 'symbol', 'uniprot_name', 'description', 'lfc', 'pval', 'fdr', 'mean_exp', 't', 'B']

		if all_equal_list(df.columns, cols1):

			cols = ['probe', 'symbol', 'gene_id', 'description', 'lfc', 'mean_expr',
					'tstat', 'pval', 'fdr', 'B', 'chr_ange', 'org_hromosome',
					'forward_everse', 'nuc_equence', 'gemma_id', 'go_term']
			df.columns = cols

			df['abs_lfc'] = [np.abs(x) if not pd.isnull(x) else x for x in df.lfc]

			df['multi_symbol'] = [x for x in df.symbol]
			df['symbol'] = [x.split('|')[0] if not pd.isnull(x) and isinstance(x, str) else x for x in df.symbol]

			cols = ['probe', 'symbol', 'gene_id', 'description', 'fdr', 'abs_lfc', 'lfc', 'pval', 'multi_symbol', 'mean_expr',
				   'tstat', 'B', 'chr_ange', 'org_hromosome', 'forward_everse', 'nuc_equence', 'gemma_id', 'go_term']
			df = df[cols]

			df = df.sort_values(['fdr', 'abs_lfc'], ascending=[True, False])
			df.index = np.arange(0, len(df))

			pdwritecsv(df, fname_nodup, self.root_result, verbose=verbose)

		elif all_equal_list(df.columns, cols3):

			df['abs_lfc'] = [np.abs(x) if not pd.isnull(x) else x for x in df.lfc]
			df['symbol_pipe'] = df.symbol
			df['symbol'] = [x.split('|')[0] if isinstance(x, str) else None for x in df.symbol]

			cols = ['entry_id', 'symbol', 'uniprot_name', 'description', 'lfc', 'abs_lfc', 'pval', 'fdr', 'mean_exp', 't', 'B', 'symbol_pipe']
			df = df[cols]

			df = df.sort_values(['fdr', 'abs_lfc'], ascending=[True, False])
			df.index = np.arange(0, len(df))

			pdwritecsv(df, fname, self.root_result, verbose=verbose)
		else:
			if 'abs_lfc' not in df.columns:
				print(f"Review the LFC table, please: '{fname}'")
				raise Exception('stop: LFC table')

		if df is None or df.empty:
			df  = None; ret = False
			self.dic_lfc, self.fig_lfc = {}, {}
			self.dflfc_ori = None
			print('lfc table is empty')
		else:
			self.dic_lfc  = df.to_dict('records')
			self.fig_lfc  = px.bar(df, x='symbol', y='lfc')
			ret = True

			df = df[ ~pd.isnull(df.symbol)].copy()

			''' duplicated probes '''
			df = df.sort_values(['symbol', 'abs_lfc'], ascending=[True, False])
			df = df.drop_duplicates('symbol')

			df = df.sort_values(['fdr', 'abs_lfc'], ascending=[True, False])
			df.index = np.arange(0, len(df))
			if verbose: print(f'lfc table length = {len(df)}')

			self.dflfc_ori = df

		self.set_figure_strindb()

		return True


	def set_which_db(self, geneset_lib):
		self.geneset_lib = geneset_lib
		self.geneset_num = [i for i in range(len(self.dbs_list)) if self.dbs_list[i] == geneset_lib ]

		if self.geneset_num == []:
			self.geneset_num = [i for i in range(len(self.dbs_list)) if geneset_lib.lower() in self.dbs_list[i].lower() ]

		try:
			self.geneset_num = self.geneset_num[0]
		except:
			self.geneset_num = -1

		self.set_db(self.geneset_num)

		return self.geneset_num

	def set_db(self, geneset_num: int, verbose=False):
		self.geneset_num = geneset_num

		if geneset_num == -1:
			self.geneset_lib = 'Undefined'
		else:
			try:
				self.geneset_lib = self.dbs_list[geneset_num]
			except:
				self.geneset_lib = 'Undefined'

		if verbose: print(f">>> {self.geneset_lib}")

		return

	def set_enrichment_name(self) -> [str, str]:
		if self.geneset_lib is None:
			self.set_db(self.geneset_num)

		fname = self.fname_enrich_table0%(self.geneset_lib, self.case, self.normalization)
		fname = title_replace(fname)

		s_abs_lfc = f"{self.abs_lfc_cutoff:.3f}"
		s_fdr_lfc  = f"{self.fdr_lfc_cutoff:.3f}"
		s_pval_pathway = f"{self.pathway_pval_cutoff:.3f}"
		s_fdr_pathway  = f"{self.pathway_fdr_cutoff:.3f}"

		fname += f"_cutoff_lfc_{s_abs_lfc}_fdr_{s_fdr_lfc}.tsv"
		self.fname_enrich_table = fname

		fname_cutoff = fname.replace('.tsv', f"_pathway_pval_{s_pval_pathway}_fdr_{s_fdr_pathway}_num_genes_{self.num_of_genes_cutoff}.tsv")

		return fname, fname_cutoff

	def open_enrichment_analysis(self, force:bool=False, save_enriched_ptws_excel_odt:bool=False, verbose:bool=False) -> bool:

		self.df_enr0, self.df_enr = None, None

		fname, fname_cutoff = self.set_enrichment_name()
		fullname = os.path.join(self.root_enrich, fname)
		if verbose: print(">>>> open enrichment_analysis(): ", fullname)

		if not os.path.exists(fullname):
			df = pd.DataFrame({})
			if verbose: print(f"Enriched Analysis table for {self.case} does not exist: '{fullname}'")
			return False

		df_enr0 = pdreadcsv(fname, self.root_enrich)

		if df_enr0 is None or df_enr0.empty:
			if verbose: print('Enriched Analysis is empty: %s'%(fullname))
			return False

		cols_ori = list(df_enr0.columns)

		if not 'num_of_genes' in df_enr0.columns or force:
			if 'num_genes' in df_enr0.columns:
				cols = ['pathway', 'pathway_id', 'pval', 'fdr', 'odds_ratio', 'combined_score', 'genes', 'num_genes']

				if all_equal_list(cols, cols_ori[:len(cols)]):
					df_enr0 = df_enr0[cols]
					cols = ['pathway', 'pathway_id', 'pval', 'fdr', 'odds_ratio', 'combined_score', 'genes', 'num_of_genes']
					df_enr0.columns = cols
					ret = pdwritecsv(df_enr0, fname, self.root_enrich, verbose=verbose)
			else:
				cols = ['pathway', 'overlap', 'pval', 'fdr', 'old_pval', 'old_fdr', 'odds_ratio', 'combined_score', 'genes']

				if len(df_enr0.columns) == len(cols):
					df_enr0.columns = cols
				elif len(df_enr0.columns) == len(cols)-2:
					cols = ['pathway', 'overlap', 'pval', 'fdr', 'odds_ratio', 'combined_score', 'genes']
					df_enr0.columns = cols

				try:
					if ";" in df_enr0.iloc[0].genes:
						mat = [0 if not isinstance(x, str) or x == '' else len(x.split(';')) for x in df_enr0.genes]
					else:
						mat = [0 if not isinstance(x, str) or x == '[]' else len(eval(x)) for x in df_enr0.genes]
				except:
					print("Error: please review df_enr0 structure and columns genes")
					raise Exception('Stop: open enrichment_analysis, column genes')

				df_enr0['num_of_genes'] = mat
				ret = pdwritecsv(df_enr0, fname, self.root_enrich, verbose=verbose)

		else:
			ret = 1
			for col in ['pathway', 'pval', 'fdr', 'genes', 'num_of_genes' ]:
				ret *= col in df_enr0.columns

			if ret != 1:
				cols = list(df_enr0.columns)
				print(f"Rename manually the enriched table, wrong columns: {', '.join(cols)}")
				raise Exception('stop')

		self.df_enr0 = df_enr0
		''' calculates degs in pathways '''
		_ = self.calc_sig_enriched_pathway(save_enriched_ptws_excel_odt, verbose)

		return True


	def calc_sig_enriched_pathway(self, save_enriched_ptws_excel_odt:bool=False, verbose:bool=False) -> pd.DataFrame:
		''' calculates degs in pathways '''
		df_enr = self.df_enr0.copy()

		df_enr = df_enr[ (df_enr.pval < self.pathway_pval_cutoff) &
						 (df_enr.fdr  < self.pathway_fdr_cutoff) &
						 (df_enr.num_of_genes >= self.num_of_genes_cutoff) ]
		df_enr.index = np.arange(0, len(df_enr))
		self.df_enr = df_enr

		self.calc_enrichment_parameters(df_enr)

		if save_enriched_ptws_excel_odt:
			fname_dummy, fname = self.set_enrichment_name()
			pdwritecsv(df_enr, fname, self.root_result, verbose=verbose)

			'''
			pip install openpyxl
			from openpyxl import Workbook
			'''
			fname = self.fname_enrich_table0%(self.geneset_lib, self.case, self.normalization) + '.xlsx'
			fullname = os.path.join(self.root_result, fname)
			df_enr.to_excel(fullname, sheet_name=self.case, index=False)

		return df_enr

	def calc_enrichment_parameters(self, df_enr:pd.DataFrame):

		if df_enr is None or df_enr.empty:
			self.degs_in_pathways		  = []
			self.degs_ensembl_in_pathways = []
			self.degs_not_in_pathways	  = []
			self.degs_up_not_in_pathways  = []
			self.degs_dw_not_in_pathways  = []
			self.degs_ensembl_not_in_pathways = []

			self.pathway_list	  = []
			self.pathway_id_list  = []
			self.pathway_fdr_list = []

			self.degs_up_ensembl = []
			self.degs_dw_ensembl = []

			self.degs_up_ensembl_in_pathways = []
			self.degs_dw_ensembl_in_pathways = []
			self.degs_up_ensembl_not_in_pathways = []
			self.degs_dw_ensembl_not_in_pathways = []
			self.degs_up_not_ensembl = []
			self.degs_dw_not_ensembl = []

			self.n_pathways			= 0
			self.n_degs_in_pathways	= 0
			self.n_degs_ensembl_in_pathways	= 0
			self.n_degs_not_in_pathways		= 0
			self.n_degs_ensembl_not_in_pathways = 0

			self.n_degs_not_ensembl = 0
			self.n_degs_up_not_ensembl = 0
			self.n_degs_dw_not_ensembl = 0

			self.n_degs_up_ensembl_in_pathways = 0
			self.n_degs_dw_ensembl_in_pathways = 0
			self.n_degs_up_ensembl_not_in_pathways = 0
			self.n_degs_dw_ensembl_not_in_pathways = 0

			return

		degs_in_pathways = []
		pathway_list, pathway_id_list, pathway_fdr_list = [], [], []
		for i in range(len(df_enr)):
			row = df_enr.iloc[i]
			genes = row.genes
			if isinstance(genes, str):
				genes = eval(genes)
			degs_in_pathways += genes

			pathway_list.append(row.pathway)
			try:
				# now, only Ractome has pathway_id
				pathway_id_list.append(row.pathway_id)
			except:
				pass
			pathway_fdr_list.append(row.fdr)

		degs_in_pathways = list(np.unique(degs_in_pathways))

		self.degs_in_pathways		  = degs_in_pathways
		self.degs_ensembl_in_pathways = [x for x in degs_in_pathways if x in self.degs_ensembl]
		self.degs_not_in_pathways	  = [x for x in self.degs		 if x not in degs_in_pathways]
		self.degs_not_in_pathways.sort()
		self.degs_ensembl_not_in_pathways = [x for x in self.degs_ensembl if x not in degs_in_pathways]
		self.degs_ensembl_not_in_pathways.sort()

		# without ensembl
		self.degs_up_not_ensembl = [x for x in self.degs_up if x not in self.degs_up_ensembl]
		self.degs_up_not_ensembl.sort()
		self.degs_dw_not_ensembl = [x for x in self.degs_dw if x not in self.degs_dw_ensembl]
		self.degs_dw_not_ensembl.sort()


		self.pathway_list	  = pathway_list
		self.pathway_id_list  = pathway_id_list
		self.pathway_fdr_list = pathway_fdr_list

		self.degs_up_ensembl_in_pathways = [x for x in self.degs_up_ensembl  if x	 in degs_in_pathways]
		self.degs_up_ensembl_in_pathways.sort()
		self.degs_dw_ensembl_in_pathways = [x for x in self.degs_dw_ensembl  if x	 in degs_in_pathways]
		self.degs_dw_ensembl_in_pathways.sort()

		self.degs_up_ensembl_not_in_pathways = [x for x in self.degs_up_ensembl  if x not in degs_in_pathways]
		self.degs_up_ensembl_not_in_pathways.sort()
		self.degs_dw_ensembl_not_in_pathways = [x for x in self.degs_dw_ensembl  if x not in degs_in_pathways]
		self.degs_dw_ensembl_not_in_pathways.sort()

		self.n_pathways					 = len(df_enr)
		self.n_degs_in_pathways			 = len(degs_in_pathways)
		self.n_degs_ensembl_in_pathways	 = len(self.degs_ensembl_in_pathways )
		self.n_degs_not_in_pathways		 = len(self.degs_not_in_pathways)
		self.n_degs_ensembl_not_in_pathways = len(self.degs_ensembl_not_in_pathways)

		self.n_degs_up_not_ensembl = len(self.degs_up_not_ensembl)
		self.n_degs_dw_not_ensembl = len(self.degs_dw_not_ensembl)
		self.n_degs_not_ensembl = self.n_degs_up_not_ensembl + self.n_degs_dw_not_ensembl

		self.n_degs_up_ensembl_in_pathways	 = len(self.degs_up_ensembl_in_pathways)
		self.n_degs_dw_ensembl_in_pathways	 = len(self.degs_dw_ensembl_in_pathways)
		self.n_degs_up_ensembl_not_in_pathways = len(self.degs_up_ensembl_not_in_pathways)
		self.n_degs_dw_ensembl_not_in_pathways = len(self.degs_dw_ensembl_not_in_pathways)


	def set_pathway_cutoff_params(self, pathway_fdr_cutoff:float, pathway_pval_cutoff:float,
										num_of_genes_cutoff:int, verbose:bool=False):

		self.pathway_fdr_cutoff = pathway_fdr_cutoff
		self.pathway_pval_cutoff = pathway_pval_cutoff
		self.num_of_genes_cutoff = num_of_genes_cutoff

		'''
		self.df_enr = self.calc_sig_enriched_pathway()

		if verbose:
			self.echo_enriched_pathways()
		'''

	def split_case(self, case):
		if case != self.case_list[self.group_index]:
			try:
				self.group_index = [i for i in range(len(self.case_list)) if case == self.case_list[i]][0]
			except:
				self.group_index = 0
				case = self.case_list[0]

		self.case = case

		if not self.has_age and not self.has_gender:
			group = case
			gender, age = '', ''
		else:
			mat = case.split('_')
			if len(mat) == 1:
				group = case
				gender, age = '', ''

			elif len(mat) == 2:
				'''
					g2b_male
					g3_male_elder
				'''
				if self.has_gender and self.has_age:
					group = mat[0]
					gender = mat[1]
					age = ''
				else:
					if self.has_gender:
						group  = mat[0]
						gender = mat[1]
						age	= ''
					else:
						group  = mat[0]
						age	= mat[1]
						gender = ''
			elif len(mat) == 3:
				group  = mat[0]
				gender = mat[1]
				age	= mat[2]
			else:
				print("Error: Houstou we have problems.")
				print("Problems spliting case?", case)
				raise Exception('Stop: case')

		self.group  = group
		self.gender = gender
		self.age	= age

		return

	def get_best_ptw_cutoff_biopax(self, med_max_ptw:str='median', verbose:bool=False):

		aux_geneset_num = self.geneset_num

		'''
			row['quantile'], row.abs_lfc_cutoff, row.fdr_lfc_cutoff, \
			row.pathway_pval_cutoff, row.pathway_fdr_cutoff, row.num_of_genes_cutoff, \
			row.n_pathways, row.n_degs_in_pathways, \
			row.n_degs_in_pathways_mean, row.n_degs_in_pathways_median, row.n_degs_in_pathways_std, \
			row.toi1_median, row.toi2_median, row.toi3_median, row.toi4_median
		'''
		self.quantile, self.abs_lfc_cutoff, self.fdr_lfc_cutoff, \
		self.pathway_pval_cutoff, self.pathway_fdr_cutoff, self.num_of_genes_cutoff, \
		self.n_pathways_best, self.n_degs_in_pathways_best, \
		self.n_degs_in_pathways_mean, self.n_degs_in_pathways_median, self.n_degs_in_pathways_std, \
		self.toi1_median, self.toi2_median, self.toi3_median, self.toi4_median  = \
		self.cfg.get_cfg_best_ptw_cutoff(self.case, self.normalization, self.geneset_num, med_max_ptw=med_max_ptw, verbose=verbose)

		if self.geneset_num == -1:
			self.geneset_num = aux_geneset_num

		self.set_db(self.geneset_num, verbose=verbose)

	def echo_parameters(self, want_echo_default:bool=False, jump_line:bool=True):
		if want_echo_default:
			self.echo_default()
			if jump_line: print("")

		self.echo_degs_all()
		if jump_line: print("")

		self.echo_enriched_pathways()


	def echo_default(self):
		print(f"geneset lib '{self.geneset_lib}' num={self.geneset_num}")
		print(f"Normalization={self.normalization}; has age={self.has_age} and has gender={self.has_gender}")


	def echo_degs(self):
		print(f"For case '{self.case}', there are {self.n_degs}/{self.n_degs_ensembl} {self.s_deg_dap}s/{self.s_deg_dap}s with ensembl_id")
		print(f"{self.s_deg_dap}'s cutoffs: abs(LFC)={self.abs_lfc_cutoff:.3f}; FDR={self.fdr_lfc_cutoff:.3f}")

	def echo_degs_all(self):
		self.echo_degs()

		#if self.n_degs > 100:
		print(f"\t{self.n_degs}/{self.n_degs_ensembl} {self.s_deg_dap}s/ensembl.")
		# else:
		# print(f"\t{self.n_degs} {self.s_deg_dap}s: {', '.join(self.degs)}")

		print(f"\t\tUp {self.n_degs_up}/{self.n_degs_up_ensembl} {self.s_deg_dap}s/ensembl.")
		print(f"\t\tDw {self.n_degs_dw}/{self.n_degs_dw_ensembl} {self.s_deg_dap}s/ensembl.")

		'''
		if self.n_degs_up > 100:
			print(f"\t\tUp {self.n_degs_up} {self.s_deg_dap}s: '{', '.join(self.degs_up[:100])}...'")
		else:
			print(f"\t\tUp {self.n_degs_up} {self.s_deg_dap}s: '{', '.join(self.degs_up)}'")

		if self.n_degs_dw > 100:
			print(f"\t\tDw {self.n_degs_dw} {self.s_deg_dap}s: '{', '.join(self.degs_dw[:100])}...'")
		else:
			print(f"\t\tDw {self.n_degs_dw} {self.s_deg_dap}s: '{', '.join(self.degs_dw)}'")
		'''

	def echo_enriched_pathways(self):

		print(f"Found {self.n_pathways} (best={self.n_pathways_best}) pathways for geneset num={self.geneset_num} '{self.geneset_lib}'")
		print(f"Pathway cutoffs p-value={self.pathway_pval_cutoff:.3f} fdr={self.pathway_fdr_cutoff:.3f} min genes={self.num_of_genes_cutoff}")

		if self.df_enr is not None and not self.df_enr.empty:
			print(f"{self.s_deg_dap}s found in enriched pathways:")
			print(f"\tThere are {self.n_degs_ensembl} {self.s_deg_dap}s found in pathways")
			print(f"\t{self.n_degs_in_pathways} (best={self.n_degs_in_pathways_best}) {self.s_deg_dap}s in pathways and {self.n_degs_not_in_pathways}/{self.n_degs_ensembl_not_in_pathways} {self.s_deg_dap}s/ensembl not in pathways")
			print("")

			# : {','.join(self.degs_up_ensembl_in_pathways)}
			print(f"\t{self.n_degs_up_ensembl_in_pathways} {self.s_deg_dap}s ensembl Up in pathways")
			print(f"\t{self.n_degs_up_ensembl_not_in_pathways} {self.s_deg_dap}s Up ensembl not in pathways")

			print("")

			print(f"\t{self.n_degs_dw_ensembl_in_pathways} {self.s_deg_dap}s ensembl Dw in pathways")
			print(f"\t{self.n_degs_dw_ensembl_not_in_pathways} {self.s_deg_dap}s Dw ensembl not in pathways")

		else:
			print("No enrichment analysis was calculated.")


	def summary_degs_and_pathways(self, check:bool=False, force:bool=False, verbose:bool=False) -> pd.DataFrame:

		fullname = os.path.join(self.root_ressum, self.fname_summ_deg_ptw)

		if os.path.exists(fullname) and not force:
			dfsum = pdreadcsv(self.fname_summ_deg_ptw, self.root_ressum, verbose=verbose)
			col0 = list(dfsum.columns)[0]
			dfsum = dfsum.set_index(col0)
			dfsum.index.names = ['index']
			dfsum = dfsum.infer_objects(copy=False).fillna(0)

			cols5 = list(dfsum.columns)
			ncols = np.arange(0, len(cols5), 2)
			for ncol in ncols:
				dfsum[cols5[ncol]] = dfsum[cols5[ncol]].astype(int)

			return dfsum

		dic = {}; icount = -1
		for case in self.case_list:
			ret, _, _, _ = self.open_case(case, verbose=False)
			if not ret: continue

			''' Best Cutoff Algorithm params '''
			n_degs_bca	= self.n_degs
			n_degs_up_bca = self.n_degs_up
			n_degs_dw_bca = self.n_degs_dw

			n_degs_ensembl_bca	= self.n_degs_ensembl
			n_degs_up_ensembl_bca = self.n_degs_up_ensembl
			n_degs_dw_ensembl_bca = self.n_degs_dw_ensembl

			n_pathways_bca					 = self.n_pathways
			n_degs_in_pathways_bca			 = self.n_degs_in_pathways
			if n_degs_in_pathways_bca == 0:
				print("No DEGs in pathway???")
				n_degs_in_pathways_bca = 1

			n_degs_not_in_pathways_bca		 = self.n_degs_not_in_pathways
			n_degs_ensembl_in_pathways_bca	 = self.n_degs_ensembl_in_pathways
			n_degs_ensembl_not_in_pathways_bca = self.n_degs_ensembl_not_in_pathways

			n_degs_up_ensembl_in_pathways_bca	 = self.n_degs_up_ensembl_in_pathways
			n_degs_dw_ensembl_in_pathways_bca	 = self.n_degs_dw_ensembl_in_pathways
			n_degs_up_ensembl_not_in_pathways_bca = self.n_degs_up_ensembl_not_in_pathways
			n_degs_dw_ensembl_not_in_pathways_bca = self.n_degs_dw_ensembl_not_in_pathways

			''' Default params '''
			ret, _, _, _ = self.open_case_params(case, abs_lfc_cutoff=1, fdr_lfc_cutoff=0.05, pathway_fdr_cutoff=0.05)
			if not ret: continue

			n_degs_default = self.n_degs
			n_degs_div	 = n_degs_default if n_degs_default > 0 else 1
			n_degs_up_default = self.n_degs_up
			n_degs_dw_default = self.n_degs_dw

			n_degs_ensembl_default	= self.n_degs_ensembl
			n_degs_ensembl_div		= n_degs_ensembl_default if n_degs_ensembl_default > 0 else 1
			n_degs_up_ensembl_default = self.n_degs_up_ensembl
			n_degs_dw_ensembl_default = self.n_degs_dw_ensembl

			n_pathways_default		 = self.n_pathways
			n_degs_in_pathways_default = self.n_degs_in_pathways
			n_degs_ensembl_in_pathways_default	 = self.n_degs_ensembl_in_pathways
			n_degs_ensembl_not_in_pathways_default = self.n_degs_ensembl_not_in_pathways
			n_degs_not_in_pathways_default		 = self.n_degs_not_in_pathways

			n_degs_up_ensembl_in_pathways_default	 = self.n_degs_up_ensembl_in_pathways
			n_degs_dw_ensembl_in_pathways_default	 = self.n_degs_dw_ensembl_in_pathways
			n_degs_up_ensembl_not_in_pathways_default = self.n_degs_up_ensembl_not_in_pathways
			n_degs_dw_ensembl_not_in_pathways_default = self.n_degs_dw_ensembl_not_in_pathways

			'''----------- BCA frequencies & probabilities  --------------------'''
			for perc in [False, True]:
				icount += 1
				dic[icount] = {}
				dic2 = dic[icount]

				if not perc:
					dic2['case'] = case

					'''------------- BCA - best cutoff algorithm -----------------------'''
					dic2['n_degs_bca']	 = n_degs_bca
					if check:
						soma = (n_degs_up_bca + n_degs_dw_bca)
						dic2['n_degs_bca_sum'] = soma
					dic2['n_degs_up_bca']  = n_degs_up_bca
					dic2['n_degs_dw_bca']  = n_degs_dw_bca

					dic2['n_degs_ensembl_bca'] = n_degs_ensembl_bca
					if check:
						soma = (n_degs_up_ensembl_bca + n_degs_dw_ensembl_bca)
						dic2['n_degs_ensembl_bca_sum'] = soma
					dic2['n_degs_up_ensembl_bca'] = n_degs_up_ensembl_bca
					dic2['n_degs_dw_ensembl_bca'] = n_degs_dw_ensembl_bca

					dic2['n_pathways_bca'] = n_pathways_bca

					'''----------- in and out pathway --------------'''
					n_all_degs_bca = n_degs_in_pathways_bca + n_degs_not_in_pathways_bca
					dic2['n_all_degs_bca']		= n_all_degs_bca

					if n_all_degs_bca == 0:
						print("n_all_degs_bca == 0???")
						n_all_degs_bca = 1

					dic2['n_degs_in_pathways_bca']	 = n_degs_in_pathways_bca
					dic2['n_degs_not_in_pathways_bca'] = n_degs_not_in_pathways_bca

					'''----------- Ensembl in and out pathway --------------'''
					n_all_degs_ensembl_bca = n_degs_ensembl_in_pathways_bca + n_degs_ensembl_not_in_pathways_bca
					dic2['n_all_degs_ensembl_bca'] = n_all_degs_ensembl_bca
					if check: dic2['n_degs_ensembl_bca_again'] = n_degs_ensembl_bca
					dic2['n_degs_ensembl_in_pathways_bca']	 = n_degs_ensembl_in_pathways_bca
					if check: dic2['n_degs_ensembl_in_pathways_bca_rep'] = None

					'''----------- Up/Dw in and out pathway --------------'''
					if check:
						soma = n_degs_up_ensembl_in_pathways_bca + n_degs_dw_ensembl_in_pathways_bca
						dic2['n_degs_in_pathways_bca_sum'] = soma
					dic2['n_degs_up_ensembl_in_pathways_bca'] = n_degs_up_ensembl_in_pathways_bca
					dic2['n_degs_dw_ensembl_in_pathways_bca'] = n_degs_dw_ensembl_in_pathways_bca

					'''----------- Up/Dw NOT in and out pathway --------------'''
					dic2['n_degs_ensembl_not_in_pathways_bca'] = n_degs_ensembl_not_in_pathways_bca
					dic2['n_degs_not_in_pathways_bca']	= n_degs_not_in_pathways_bca
					if check:
						dic2['n_degs_not_in_pathways_bca_again'] = n_degs_not_in_pathways_bca
						soma = n_degs_up_ensembl_not_in_pathways_bca + n_degs_dw_ensembl_not_in_pathways_bca
						dic2['n_degs_not_in_pathways_bca_sum'] = soma
					dic2['n_degs_up_ensembl_not_in_pathways_bca'] = n_degs_up_ensembl_not_in_pathways_bca
					dic2['n_degs_dw_ensembl_not_in_pathways_bca'] = n_degs_dw_ensembl_not_in_pathways_bca

					'''---------- default frequencies & probabilities  -----------------'''
					dic2['n_degs_default']	 = n_degs_default
					if check:
						soma = (n_degs_up_default + n_degs_dw_default)
						dic2['n_degs_default_sum'] = soma
					dic2['n_degs_up_default']  = n_degs_up_default
					dic2['n_degs_dw_default']  = n_degs_dw_default

					dic2['n_degs_ensembl_default'] = n_degs_ensembl_default
					if check:
						soma = (n_degs_up_ensembl_default + n_degs_dw_ensembl_default)
						dic2['n_degs_ensembl_default_sum'] = soma
					dic2['n_degs_up_ensembl_default'] = n_degs_up_ensembl_default
					dic2['n_degs_dw_ensembl_default'] = n_degs_dw_ensembl_default

					dic2['n_pathways_default'] = n_pathways_default

					'''----------- in and out pathway --------------'''
					if n_pathways_default == 0:
						dic2['n_all_degs_default'] = None
						dic2['n_degs_in_pathways_default']	 = None
						dic2['n_degs_not_in_pathways_default'] = None

						'''----------- Ensembl in and out pathway --------------'''
						dic2['n_all_degs_ensembl_default'] = None
						if check: dic2['n_degs_ensembl_default_again'] = None
						dic2['n_degs_ensembl_in_pathways_default']	 = None
						if check: dic2['n_degs_ensembl_in_pathways_default_rep'] = None

						'''----------- Up/Dw in and out pathway --------------'''
						if check:
							dic2['n_degs_in_pathways_default_sum'] = None
						dic2['n_degs_up_ensembl_in_pathways_default'] = None
						dic2['n_degs_dw_ensembl_in_pathways_default'] = None

						'''----------- Up/Dw NOT in and out pathway --------------'''
						dic2['n_degs_ensembl_not_in_pathways_default'] = None
						dic2['n_degs_not_in_pathways_default'] = None
						if check:
							dic2['n_degs_not_in_pathways_default_again'] = None
							dic2['n_degs_not_in_pathways_default_sum'] = None
						dic2['n_degs_up_ensembl_not_in_pathways_default'] = None
						dic2['n_degs_dw_ensembl_not_in_pathways_default'] = None

					else:
						n_all_degs_default = n_degs_in_pathways_default + n_degs_not_in_pathways_default
						dic2['n_all_degs_default'] = n_all_degs_default
						dic2['n_degs_in_pathways_default']	 = n_degs_in_pathways_default
						dic2['n_degs_not_in_pathways_default'] = n_degs_not_in_pathways_default

						'''----------- Ensembl in and out pathway --------------'''
						n_all_degs_ensembl_default = n_degs_ensembl_in_pathways_default + n_degs_ensembl_not_in_pathways_default
						dic2['n_all_degs_ensembl_default'] = n_all_degs_ensembl_default
						if check: dic2['n_degs_ensembl_default_again'] = n_degs_ensembl_default
						dic2['n_degs_ensembl_in_pathways_default']	 = n_degs_ensembl_in_pathways_default
						if check: dic2['n_degs_ensembl_in_pathways_default_rep'] = None

						'''----------- Up/Dw in and out pathway --------------'''
						if check:
							soma = n_degs_up_ensembl_in_pathways_default + n_degs_dw_ensembl_in_pathways_default
							dic2['n_degs_in_pathways_default_sum'] = soma
						dic2['n_degs_up_ensembl_in_pathways_default'] = n_degs_up_ensembl_in_pathways_default
						dic2['n_degs_dw_ensembl_in_pathways_default'] = n_degs_dw_ensembl_in_pathways_default

						'''----------- Up/Dw NOT in and out pathway --------------'''
						dic2['n_degs_ensembl_not_in_pathways_default'] = n_degs_ensembl_not_in_pathways_default
						dic2['n_degs_not_in_pathways_default']	= n_degs_not_in_pathways_default
						if check:
							dic2['n_degs_not_in_pathways_default_again'] = n_degs_not_in_pathways_default
							soma = n_degs_up_ensembl_not_in_pathways_default + n_degs_dw_ensembl_not_in_pathways_default
							dic2['n_degs_not_in_pathways_default_sum'] = soma
						dic2['n_degs_up_ensembl_not_in_pathways_default'] = n_degs_up_ensembl_not_in_pathways_default
						dic2['n_degs_dw_ensembl_not_in_pathways_default'] = n_degs_dw_ensembl_not_in_pathways_default

				else:
					dic2['case'] = case + '_perc'

					'''------------- BCA - best cutoff algorithm -----------------------'''
					dic2['n_degs_bca']	 = 1.00
					if check:
						soma = (n_degs_up_bca + n_degs_dw_bca)
						dic2['n_degs_bca_sum'] = soma / n_degs_bca
					dic2['n_degs_up_bca']  = n_degs_up_bca / n_degs_bca
					dic2['n_degs_dw_bca']  = n_degs_dw_bca / n_degs_bca

					dic2['n_degs_ensembl_bca']	= n_degs_ensembl_bca / n_degs_bca
					if check:
						soma = (n_degs_up_ensembl_bca + n_degs_dw_ensembl_bca)
						dic2['n_degs_ensembl_bca_sum'] = soma / n_degs_bca
					dic2['n_degs_up_ensembl_bca'] = n_degs_up_ensembl_bca / n_degs_ensembl_bca
					dic2['n_degs_dw_ensembl_bca'] = n_degs_dw_ensembl_bca / n_degs_ensembl_bca

					dic2['n_pathways_bca'] = None

					'''----------- in and out pathway --------------'''
					dic2['n_all_degs_bca']	= 1.00
					n_all_degs_bca = n_degs_in_pathways_bca + n_degs_not_in_pathways_bca
					dic2['n_degs_in_pathways_bca']	 = n_degs_in_pathways_bca / n_all_degs_bca
					dic2['n_degs_not_in_pathways_bca'] = n_degs_not_in_pathways_bca / n_all_degs_bca

					'''----------- Ensembl in and out pathway --------------'''
					n_all_degs_ensembl_bca = n_degs_ensembl_in_pathways_bca + n_degs_ensembl_not_in_pathways_bca
					if check: dic2['n_degs_ensembl_bca_again'] = 1
					dic2['n_all_degs_ensembl_bca']		= n_all_degs_ensembl_bca / n_degs_ensembl_bca
					dic2['n_degs_ensembl_in_pathways_bca']	 = n_degs_ensembl_in_pathways_bca / n_degs_in_pathways_bca
					if check: dic2['n_degs_ensembl_in_pathways_bca_rep'] = n_degs_ensembl_in_pathways_bca / n_all_degs_ensembl_bca

					'''----------- Up/Dw in and out pathway --------------'''
					if check:
						soma = (n_degs_up_ensembl_in_pathways_bca + n_degs_dw_ensembl_in_pathways_bca)
						dic2['n_degs_in_pathways_bca_sum'] = soma / n_degs_in_pathways_bca
					dic2['n_degs_up_ensembl_in_pathways_bca']  = n_degs_up_ensembl_in_pathways_bca / n_degs_in_pathways_bca
					dic2['n_degs_dw_ensembl_in_pathways_bca']  = n_degs_dw_ensembl_in_pathways_bca / n_degs_in_pathways_bca

					'''----------- Up/Dw NOT in and out pathway --------------'''
					dic2['n_degs_ensembl_not_in_pathways_bca'] = n_degs_ensembl_not_in_pathways_bca / n_all_degs_ensembl_bca
					if check:
						dic2['n_degs_not_in_pathways_bca_again'] = 1
						soma = n_degs_up_ensembl_not_in_pathways_bca + n_degs_dw_ensembl_not_in_pathways_bca
						dic2['n_degs_not_in_pathways_bca_sum'] = soma / n_degs_not_in_pathways_bca
					dic2['n_degs_up_ensembl_not_in_pathways_bca']  = n_degs_up_ensembl_not_in_pathways_bca / n_degs_ensembl_not_in_pathways_bca
					dic2['n_degs_dw_ensembl_not_in_pathways_bca']  = n_degs_dw_ensembl_not_in_pathways_bca / n_degs_ensembl_not_in_pathways_bca

					'''---------- default frequencies & probabilities  -----------------'''

					dic2['n_degs_default'] = 1.00
					if check:
						soma = (n_degs_up_default + n_degs_dw_default)
						dic2['n_degs_default_sum'] = soma / n_degs_div
					dic2['n_degs_up_default']  = n_degs_up_default / n_degs_div
					dic2['n_degs_dw_default']  = n_degs_dw_default / n_degs_div

					dic2['n_degs_ensembl_default']	= n_degs_ensembl_default / n_degs_div
					if check:
						soma = (n_degs_up_ensembl_default + n_degs_dw_ensembl_default)
						dic2['n_degs_ensembl_default_sum'] = soma / n_degs_div
					dic2['n_degs_up_ensembl_default'] = n_degs_up_ensembl_default / n_degs_ensembl_div
					dic2['n_degs_dw_ensembl_default'] = n_degs_dw_ensembl_default / n_degs_ensembl_div


					'''----------- in and out pathway --------------'''
					if n_pathways_default == 0:
						dic2['n_pathways_default'] = None
						dic2['n_all_degs_default'] = None
						dic2['n_degs_in_pathways_default']	 = None
						dic2['n_degs_not_in_pathways_default'] = None

						'''----------- Ensembl in and out pathway --------------'''
						if check: dic2['n_degs_ensembl_default_again'] = None
						dic2['n_all_degs_ensembl_default']		= None
						dic2['n_degs_ensembl_in_pathways_default']	 = None
						if check: dic2['n_degs_ensembl_in_pathways_default_rep'] = None

						'''----------- Up/Dw in and out pathway --------------'''
						if check:
							dic2['n_degs_in_pathways_default_sum'] = None
						dic2['n_degs_up_ensembl_in_pathways_default'] = None
						dic2['n_degs_dw_ensembl_in_pathways_default'] = None

						'''----------- Up/Dw NOT in and out pathway --------------'''
						dic2['n_degs_ensembl_not_in_pathways_default'] = None
						if check:
							dic2['n_degs_not_in_pathways_default_again'] = None
							dic2['n_degs_not_in_pathways_default_sum'] = None
						dic2['n_degs_up_ensembl_not_in_pathways_default'] = None
						dic2['n_degs_dw_ensembl_not_in_pathways_default'] = None
					else:
						dic2['n_pathways_default'] = None
						dic2['n_all_degs_default'] = 1

						n_all_degs_default = n_degs_in_pathways_default + n_degs_not_in_pathways_default
						n_all_genes_in_pathways_div = n_all_degs_default if n_all_degs_default > 0 else 1
						dic2['n_degs_in_pathways_default']	 = n_degs_in_pathways_default / n_all_genes_in_pathways_div
						dic2['n_degs_not_in_pathways_default'] = n_degs_not_in_pathways_default / n_all_genes_in_pathways_div

						'''----------- Ensembl in and out pathway --------------'''
						n_all_degs_ensembl_default = n_degs_ensembl_in_pathways_default + n_degs_ensembl_not_in_pathways_default
						if check: dic2['n_degs_ensembl_default_again'] = 1
						dic2['n_all_degs_ensembl_default']		= n_all_degs_ensembl_default / n_all_genes_in_pathways_div
						dic2['n_degs_ensembl_in_pathways_default']	 = n_degs_ensembl_in_pathways_default / n_all_degs_ensembl_default
						if check: dic2['n_degs_ensembl_in_pathways_default_rep'] = n_degs_ensembl_in_pathways_default / n_degs_in_pathways_default

						'''----------- Up/Dw in and out pathway --------------'''
						soma = (n_degs_up_ensembl_in_pathways_default + n_degs_dw_ensembl_in_pathways_default)
						if check:
							dic2['n_degs_in_pathways_default_sum'] = soma / n_degs_in_pathways_default
						dic2['n_degs_up_ensembl_in_pathways_default']  = n_degs_up_ensembl_in_pathways_default / soma
						dic2['n_degs_dw_ensembl_in_pathways_default']  = n_degs_dw_ensembl_in_pathways_default / soma

						'''----------- Up/Dw NOT in and out pathway --------------'''
						dic2['n_degs_ensembl_not_in_pathways_default'] = n_degs_ensembl_not_in_pathways_default / n_all_degs_ensembl_default
						if check:
							dic2['n_degs_not_in_pathways_default_again'] = 1
							soma = n_degs_up_ensembl_not_in_pathways_default + n_degs_dw_ensembl_not_in_pathways_default
							dic2['n_degs_not_in_pathways_default_sum'] = soma / n_degs_not_in_pathways_default
						dic2['n_degs_up_ensembl_not_in_pathways_default']  = n_degs_up_ensembl_not_in_pathways_default / n_degs_ensembl_not_in_pathways_default
						dic2['n_degs_dw_ensembl_not_in_pathways_default']  = n_degs_dw_ensembl_not_in_pathways_default / n_degs_ensembl_not_in_pathways_default

		dfsum = pd.DataFrame(dic)
		dfsum = dfsum.T
		dfsum = dfsum.set_index('case')
		dfsum = dfsum.T
		pdwritecsv(dfsum, self.fname_summ_deg_ptw, self.root_ressum,  index=True, verbose=verbose)

		return dfsum


	def open_case(self, case:str, save_file:bool=False, force:bool=False, save_enriched_ptws_excel_odt:bool=False,
				  prompt_verbose:bool=False, verbose:bool=False) -> (bool, List, List, pd.DataFrame):
		self.reset_degs_and_df_enr()
		self.case = case
		if prompt_verbose: print(">>> case", case)
		self.get_best_ptw_cutoff_biopax()
		self.split_case(case)

		ret_lfc = self.open_dflfc_ori(verbose=verbose)

		degs, degs_ensembl, dflfc = self.list_of_degs(save_file=save_file, force=force, prompt_verbose=prompt_verbose, verbose=verbose)

		ret_enr = self.open_enrichment_analysis(force=force, save_enriched_ptws_excel_odt=save_enriched_ptws_excel_odt, verbose=verbose)

		if ret_lfc * ret_enr != 1:
			if verbose:
				print("Warning: Enrichment Analysis was not performed.")
				print(f"For case {case}, pathways cutoffs are: pval={self.pathway_pval_cutoff:.3f} and fdr={self.pathway_fdr_cutoff:.3f}")

		return ret_lfc, degs, degs_ensembl, dflfc

	def open_case_params(self, case:str, abs_lfc_cutoff:float=1.0, fdr_lfc_cutoff:float=0.05,
						 pathway_pval_cutoff:float=0.05, pathway_fdr_cutoff:float=0.05, num_of_genes_cutoff:int=3,
						 force:bool=False, prompt_verbose:bool=False, verbose:bool=False) -> (bool, List, List, pd.DataFrame):

		self.reset_degs_and_df_enr()
		self.case = case
		self.split_case(case)

		self.abs_lfc_cutoff	  = abs_lfc_cutoff
		self.fdr_lfc_cutoff	  = fdr_lfc_cutoff
		self.pathway_pval_cutoff = pathway_pval_cutoff
		self.pathway_fdr_cutoff  = pathway_fdr_cutoff
		self.num_of_genes_cutoff = num_of_genes_cutoff


		ret_lfc = self.open_dflfc_ori(verbose=verbose)
		degs, degs_ensembl, dflfc = self.list_of_degs(force=force, prompt_verbose=prompt_verbose, verbose=verbose)

		ret_enr = self.open_enrichment_analysis(force=force, verbose=verbose)

		if ret_lfc * ret_enr != 1:
			if verbose:
				print("Warning: Enrichment Analysis were not performed.")
				print(f"For case {case}, pathways cutoffs are: pval={self.pathway_pval_cutoff:.3f} and fdr={self.pathway_fdr_cutoff:.3f}")

		return ret_lfc, degs, degs_ensembl, dflfc


	def open_case_params_wo_pathways(self, case:str, abs_lfc_cutoff:float, fdr_lfc_cutoff:float,
									 prompt_verbose:bool=False, verbose:bool=False) -> (bool, List, List, pd.DataFrame):
		self.reset_degs_and_df_enr()
		self.case = case
		self.split_case(case)

		self.abs_lfc_cutoff = abs_lfc_cutoff
		self.fdr_lfc_cutoff = fdr_lfc_cutoff

		ret = self.open_dflfc_ori(verbose=verbose)
		degs, degs_ensembl, dflfc = self.list_of_degs(force=False, prompt_verbose=prompt_verbose, verbose=verbose)

		return ret, degs, degs_ensembl, dflfc

	def open_case_simple(self, case:str, verbose:bool=False) -> bool:
		'''
			open_case_simple:
				set case
				open dflfc_ori
		'''
		self.reset_degs_and_df_enr()
		self.case = case
		self.split_case(case)
		ret = self.open_dflfc_ori(verbose=verbose)

		return ret


	def calc_all_genes_in_pubmed_per_case(self, force:bool=False, prompt_verbose:bool=False,
										  verbose:bool=False) -> pd.DataFrame:

		fullname = os.path.join(self.root_ressum, self.fname_degs_summary)

		if os.path.exists(fullname) and not force:
			 return pdreadcsv(self.fname_degs_summary, self.root_ressum, verbose=verbose)

		dic = {}; icount=-1

		for case in self.case_list:
			if prompt_verbose: print(f">>> case {case}")

			ret, _, _, _ = self.open_case(case, verbose)
			if not ret or self.degs is None:
				if verbose: print(f"There are no {self.s_deg_dap} for {case}")
				continue

			if self.dfsimb_perc is None or self.dfsimb_perc.empty:
				degs_in_pubmed	 = []
				degs_not_in_pubmed = []
			else:
				degs_in_pubmed	 = [x for x in self.dfsimb_perc.symbol if x in self.degs]
				degs_not_in_pubmed = [x for x in degs if x not in self.degs_in_pubmed]

			icount += 1
			dic[icount] = {}
			dic2 = dic[icount]

			dic2['case'] = case
			dic2['normalization']  = self.normalization
			dic2['geneset_num']	= self.geneset_num
			dic2['quantile']	   = self.quantile
			dic2['abs_lfc_cutoff'] = self.abs_lfc_cutoff
			dic2['fdr_lfc_cutoff'] = self.fdr_lfc_cutoff

			dic2['n_degs_in_pubmed']	 = len(degs_in_pubmed)
			dic2['n_degs_not_in_pubmed'] = len(degs_not_in_pubmed)

			dic2['n_degs']	= self.n_degs
			dic2['n_degs_up'] = self.n_degs_up
			dic2['n_degs_dw'] = self.n_degs_dw
			dic2['n_degs_up_ensembl'] = self.n_degs_up_ensembl
			dic2['n_degs_dw_ensembl'] = self.n_degs_dw_ensembl

			dic2['n_pathways']				 = self.n_pathways
			dic2['n_all_genes_in_pathways']	= self.n_degs_in_pathways + self.n_degs_not_in_pathways
			dic2['n_degs_in_pathways']		 = self.n_degs_in_pathways
			dic2['n_degs_in_pathways_ensembl'] = self.n_degs_ensembl_in_pathways
			dic2['n_degs_not_in_pathways']	 = self.n_degs_not_in_pathways

			dic2['n_degs_up_in_pathways']	 = self.n_degs_up_ensembl_in_pathways
			dic2['n_degs_dw_in_pathways']	 = self.n_degs_dw_ensembl_in_pathways
			dic2['n_degs_up_not_in_pathways'] = self.n_degs_up_ensembl_not_in_pathways
			dic2['n_degs_dw_not_in_pathways'] = self.n_degs_dw_ensembl_not_in_pathways

			dic2['degs_in_pubmed']	 = degs_in_pubmed
			dic2['degs_not_in_pubmed'] = degs_not_in_pubmed

			dic2['degs']	= self.degs
			dic2['degs_up'] = self.degs_up
			dic2['degs_dw'] = self.degs_dw
			dic2['degs_up_ensembl'] = self.degs_up_ensembl
			dic2['degs_dw_ensembl'] = self.degs_dw_ensembl

			dic2['degs_in_pathways']			= self.degs_in_pathways
			dic2['degs_ensembl_in_pathways']	= self.degs_ensembl_in_pathways
			dic2['degs_up_ensembl_in_pathways'] = self.degs_up_ensembl_in_pathways
			dic2['degs_dw_ensembl_in_pathways'] = self.degs_dw_ensembl_in_pathways

			dic2['degs_not_in_pathways']		 = self.degs_not_in_pathways
			dic2['degs_ensembl_not_in_pathways'] = self.degs_ensembl_not_in_pathways
			dic2['degs_up_ensembl_not_in_pathways']	  = self.degs_up_ensembl_not_in_pathways
			dic2['degs_dw_ensembl_not_in_pathways']	  = self.degs_dw_ensembl_not_in_pathways


			all_text=''
			df_enr = self.df_enr

			if df_enr is None or df_enr.empty:
				n_enr = 0
			else:
				n_enr = len(df_enr)
				for i_enr in range(len(df_enr)):
					row = df_enr.iloc[i_enr]
					text = f"{i_enr+1}) {row.pathway} ({row.pathway_id}) fdr: {row.fdr:.2e} {self.s_deg_dap}s ({row.num_of_genes}) {row.genes}\n"
					all_text += text

					all_text = all_text[:-1]


			dic2['pathway_list'] = all_text
			dic2['pathway_id_list']  = self.pathway_id_list
			dic2['pathway_fdr_list'] = self.pathway_fdr_list



		dfa = pd.DataFrame(dic).T
		# dfa = dfa.set_index('case')

		ret = pdwritecsv(dfa, self.fname_degs_summary, self.root_ressum, verbose=verbose)
		return dfa

	def save_up_and_down_degs(self, verbose:bool=False):

		''' list_of_degs calculates:
				self.degs, self.degs_ensembl, self.dflfc, self.n_degs,
				self.dfdegs_up, self.dfdegs_dw
				self.degs_up, self.degs_dw
		'''
		'''------------- degs_up ------------------------------------------------'''
		fname = self.fname_txt_updegs%(case, self.abs_lfc_cutoff, self.fdr_lfc_cutoff, self.exp_normalization)

		text = "\n".join(self.degs_up)
		write_txt(text, fname, self.root_ressum, verbose=verbose)

		fname = fname.replace('.txt', '_geneid.txt')
		degs_geneid_up = [self.gene.find_mygene_geneid(x) for x in self.degs_up]
		degs_geneid_up = [str(x) for x in degs_geneid_up if x is not None]
		text = "\n".join(degs_geneid_up)
		write_txt(text, fname, self.root_ressum, verbose=verbose)

		if len(self.dfdegs_up) > 150:
			degs_up150 = [x for x in self.dfdegs_up.iloc[:150].symbol if isinstance(x, str)]
			text = "\n".join(degs_up150)
			fname2 = fname.replace('.txt', '_150limit.txt')
			write_txt(text, fname2, self.root_ressum, verbose=verbose)

			degs_geneid_up150 = [self.gene.find_mygene_geneid(x) for x in degs_up150]
			text = "\n".join(degs_geneid_up150)
			fname2 = fname.replace('.txt', '_150limit.txt')
			write_txt(text, fname2, self.root_ressum, verbose=verbose)

		'''------------- degs_up ------- end ----------------------------------'''

		'''------------- degs down ---------------------------------------------'''
		fname = self.fname_txt_dwdegs%(case, self.abs_lfc_cutoff, self.fdr_lfc_cutoff, self.exp_normalization)

		text = "\n".join(self.degs_dw)
		write_txt(text, fname, self.root_ressum, verbose=verbose)

		fname = fname.replace('.txt', '_geneid.txt')
		degs_geneid_dw = [self.gene.find_mygene_geneid(x) for x in degs_dw]
		degs_geneid_dw = [str(x) for x in degs_geneid_dw if x is not None]
		text = "\n".join(degs_geneid_dw)
		write_txt(text, fname, self.root_ressum, verbose=verbose)

		if len(self.dfdegs_dw) > 150:
			degs_dw150 = [x for x in self.dfdegs_dw.iloc[:150].symbol if isinstance(x, str)]
			text = "\n".join(degs_dw150)
			fname2 = fname.replace('.txt', '_150limit.txt')
			write_txt(text, fname2, self.root_ressum, verbose=verbose)

			degs_geneid_dw150 = [self.gene.find_mygene_geneid(x) for x in degs_dw150]
			text = "\n".join(degs_geneid_dw150)
			fname2 = fname.replace('.txt', '_150limit.txt')
			write_txt(text, fname2, self.root_ressum, verbose=verbose)
		'''------------- degs down ------- end --------------------------------'''

		return

	def test_DEGs(self, prompt_verbose:bool=True, verbose:bool=False) -> bool:
		all_degs, all_degs_ensembl, degs_in_pathways = [], [], []

		for case in self.case_list:
			if prompt_verbose: print(f"<<< case: {case}")
			ret, _, _, _ = self.open_case(case, verbose)

			if not ret:
				continue

			all_degs += self.degs
			all_degs_ensembl += self.degs_ensembl
			degs_in_pathways += self.degs_in_pathways

		all_degs = list(np.unique(all_degs))
		all_degs_ensembl = list(np.unique(all_degs_ensembl))
		degs_in_pathways = list(np.unique(degs_in_pathways))

		if prompt_verbose:
			print(f">> {self.s_deg_dap}s calc {len(all_degs)}: '{', '.join(all_degs)}'")
			print(f">> {self.s_deg_dap}s with ensembl_id calc {len(all_degs_ensembl)}: '{', '.join(all_degs_ensembl)}'")
			print(f">> {self.s_deg_dap}s in pathways {len(degs_in_pathways)}: '{', '.join(degs_in_pathways)}'")


	def calc_degs_and_pathways_summary(self, force:bool=False, prompt_verbose:bool=False,
									   verbose:bool=False) -> bool:

		fullname = os.path.join(self.root_ressum, self.fname_degs_and_pathways_summary)

		if os.path.exists(fullname) and not force:
			return self.open_enriched_pathways_summary(verbose=verbose)

		dic_summ = {}; icount = -1

		for case in self.case_list:
			if prompt_verbose: print(f">>> case: {case}")

			ret, _, _, _ = self.open_case(case, verbose=verbose)

			if not ret or self.degs is None or self.degs == []:
				continue

			icount += 1
			dic_summ[icount] = {}
			dic2 = dic_summ[icount]

			dic2['case'] = case
			dic2['normalization'] = self.normalization
			dic2['geneset_num'] = self.geneset_num
			dic2['quantile'] = self.quantile

			dic2['abs_lfc_cutoff']	  = self.abs_lfc_cutoff
			dic2['fdr_lfc_cutoff']	  = self.fdr_lfc_cutoff
			dic2['pathway_pval_cutoff'] = self.pathway_pval_cutoff
			dic2['pathway_fdr_cutoff']  = self.pathway_fdr_cutoff
			dic2['num_of_genes_cutoff'] = self.num_of_genes_cutoff

			dic2['med_max_ptw'] = self.med_max_ptw
			dic2['toi1_median'] = self.toi1_median
			dic2['toi2_median'] = self.toi2_median
			dic2['toi3_median'] = self.toi3_median
			dic2['toi4_median'] = self.toi4_median

			'''----------- number of DEGs -------------------'''
			dic2['n_degs'] = self.n_degs
			dic2['n_degs_up'] = self.n_degs_up
			dic2['n_degs_dw'] = self.n_degs_dw
			dic2['n_degs_up_ensembl'] = self.n_degs_up_ensembl
			dic2['n_degs_dw_ensembl'] = self.n_degs_dw_ensembl

			dic2['n_pathways']			 = self.n_pathways
			dic2['n_degs_in_pathways']	 = self.n_degs_in_pathways
			dic2['n_degs_not_in_pathways'] = self.n_degs_not_in_pathways

			dic2['n_degs_up_ensembl_in_pathways'] = self.n_degs_up_ensembl_in_pathways
			dic2['n_degs_dw_ensembl_in_pathways'] = self.n_degs_dw_ensembl_in_pathways

			dic2['n_degs_up_ensembl_not_in_pathways'] = self.n_degs_up_ensembl_not_in_pathways
			dic2['n_degs_dw_ensembl_not_in_pathways'] = self.n_degs_dw_ensembl_not_in_pathways

			'''------------- Pathways ------------------'''
			dic2['pathway_list']	 = self.pathway_list
			dic2['pathway_id_list']  = self.pathway_id_list
			dic2['pathway_fdr_list'] = self.pathway_fdr_list

			'''--------------- DEGS ------------------'''
			dic2['degs'] = self.degs
			dic2['degs_up'] = self.degs_up
			dic2['degs_dw'] = self.degs_dw
			dic2['degs_up_ensembl'] = self.degs_up_ensembl
			dic2['degs_dw_ensembl'] = self.degs_dw_ensembl

			dic2['degs_in_pathways'] = self.degs_in_pathways
			dic2['degs_ensembl_in_pathways'] = self.degs_ensembl_in_pathways
			dic2['degs_not_in_pathways'] = self.degs_not_in_pathways
			dic2['degs_ensembl_not_in_pathways'] = self.degs_ensembl_not_in_pathways

			dic2['degs_up_ensembl_in_pathways'] = self.degs_up_ensembl_in_pathways
			dic2['degs_up_ensembl_not_in_pathways'] = self.degs_up_ensembl_not_in_pathways
			dic2['degs_dw_ensembl_in_pathways'] = self.degs_dw_ensembl_in_pathways
			dic2['degs_dw_ensembl_not_in_pathways'] = self.degs_dw_ensembl_not_in_pathways

		self.df_summ = pd.DataFrame(dic_summ).T

		ret2 = pdwritecsv(self.df_summ, self.fname_degs_and_pathways_summary, self.root_ressum, verbose=verbose)

		return ret

	def open_enriched_pathways_summary(self, verbose:bool=False) -> bool:
		self.df_summ = pdreadcsv(self.fname_degs_and_pathways_summary, self.root_ressum, verbose=verbose)
		return self.df_summ is not None and not self.df_summ.empty

	def list_of_degs_set_params(self, abs_lfc_cutoff:float, fdr_lfc_cutoff:float, force:bool=False,
								save_file:bool=False, prompt_verbose:bool=False, verbose:bool=False) -> (List, List, pd.DataFrame):

		self.reset_degs_and_df_enr()

		self.abs_lfc_cutoff = abs_lfc_cutoff
		self.fdr_lfc_cutoff = fdr_lfc_cutoff

		# degs, degs_ensembl, dflfc
		return self.list_of_degs(force=force, save_file=save_file, prompt_verbose=prompt_verbose, verbose=verbose)


	def reset_degs_and_df_enr(self):
		self.dflfc, self.dfdegs_up, self.dfdegs_dw = None, None, None
		self.dflfc_ensembl, self.dfdegs_up_ensembl, self.dfdegs_dw_ensembl = None, None, None

		self.degs, self.n_degs  = [], 0
		self.degs_up, self.n_degs_up  = [], 0
		self.degs_dw, self.n_degs_dw  = [], 0

		self.degs_ensembl, self.n_degs_ensembl  = [], 0
		self.degs_up_ensembl, self.n_degs_up_ensembl  = [], 0
		self.degs_dw_ensembl, self.n_degs_dw_ensembl  = [], 0

		self.biotype_list = []
		''' group by biotype '''
		self.dfg_dflc_ori , self.dfg_dflc = None, None

		self.valid_genes, self.annotated_genes = [], []
		self.n_total_genes, self.n_annotated_genes = -1, -1

		self.n_pathways_best, self.n_degs_in_pathways_best = 0, 0
		self.all_genes_annotatted_in_pathway = []

		self.df_enr, self.df_enr_reactome = None, None
		self.degs_ensembl_in_pathways = []
		self.degs_in_pathways = []
		self.degs_not_in_pathways, self.degs_ensembl_not_in_pathways = [], []

		self.pathway_list, self.pathway_id_list, self.pathway_fdr_list  = [], [], []

		self.degs_up_ensembl_in_pathways, self.degs_up_ensembl_not_in_pathways = [], []
		self.degs_dw_ensembl_in_pathways, self.degs_dw_ensembl_not_in_pathways = [], []

		self.n_pathways, self.n_degs_in_pathways, self.n_degs_not_in_pathways = 0, 0, 0

		self.n_degs_up_ensembl_in_pathways, self.n_degs_dw_ensembl_in_pathways = 0, 0
		self.n_degs_up_ensembl_not_in_pathways, self.n_degs_dw_ensembl_not_in_pathways = 0, 0
		self.n_degs_ensembl_not_in_pathways = 0


	def list_of_degs(self, force:bool=False, save_file:bool=False,
					 prompt_verbose:bool=False, verbose:bool=False) -> (List, List, pd.DataFrame):
		''' calculates:
				self.degs, self.dflfc, self.n_degs,
				self.dfdegs_up, self.dfdegs_dw
				self.degs_up, self.degs_dw

				self.degs_ensembl, self.n_degs_ensembl
				self.degs_up_ensembl, self.n_degs_up_ensembl
				self.degs_dw_ensembl, self.n_degs_dw_ensembl
		'''
		stri = '_DAP_' if self.gene_protein == 'protein' else '_DEG_'

		if self.dflfc_ori is None or self.dflfc_ori.empty:
			print(f"No dflfc table was calculated for this case {self.case}")
			return [], [], None

		dflfc = self.dflfc_ori
		dflfc = dflfc[ (dflfc.abs_lfc >= self.abs_lfc_cutoff) & (dflfc.fdr < self.fdr_lfc_cutoff)].copy()

		if dflfc.empty:
			self.dflfc = dflfc
			if verbose: print(f"There are no {self.s_deg_dap}s for dflfc fdr < {self.fdr_lfc_cutoff:.3f} and lfc >= {self.abs_lfc_cutoff:.3f}")
			return [], [], None

		dflfc = dflfc.sort_values('fdr', ascending=True)
		dflfc.index = np.arange(0, len(dflfc))

		if save_file:
			fname = f'bca_{self.s_deg_dap}s_for_{self.s_project}_case_{self.case}_best_cutoff_abs_lfc_{self.abs_lfc_cutoff:.3f}_fdr_{self.fdr_lfc_cutoff:.3f}.tsv'
			ret = pdwritecsv(dflfc, fname, self.root_result, verbose=verbose)

			fname = f'bca_{self.s_deg_dap}s_for_{self.s_project}_case_{self.case}_best_cutoff_abs_lfc_{self.abs_lfc_cutoff:.3f}_fdr_{self.fdr_lfc_cutoff:.3f}.txt'
			text = "\n".join(dflfc.symbol)
			write_txt(text, fname, self.root_result, verbose=verbose)

		# if verbose: print(f"Found %d {self.s_deg_dap}"%(len(dflfc)))
		''' official HUGO names '''
		degs = list(dflfc.symbol)
		self.dflfc = dflfc

		self.degs = degs
		self.n_degs = len(degs)

		'''
			TEC - transcritps to be validate by experiment
			link: https://www.ensembl.org/info/genome/genebuild/biotypes.html
		'''
		dfdegs_ensembl = dflfc[ ~pd.isnull(dflfc.ensembl_id) ].copy()
		dfdegs_ensembl.index = np.arange(0, len(dfdegs_ensembl))
		degs_ensembl = list(dfdegs_ensembl.symbol)
		self.dfdegs_ensembl = dfdegs_ensembl
		self.degs_ensembl = degs_ensembl
		self.n_degs_ensembl = len(degs_ensembl)

		self.biotype_list = np.unique(dfdegs_ensembl.biotype)

		self.dfg_dflc_ori = self.get_dflfc_ori_biotypes()
		self.dfg_dflc	 = self.get_dflfc_biotypes()

		''' ---------- Up ------------------------'''
		self.dfdegs_up = dflfc[dflfc.lfc > 0].copy()
		if self.dfdegs_up.empty:
			self.degs_up = []
			self.n_degs_up = 0
		else:
			self.dfdegs_up.index = np.arange(0, len(self.dfdegs_up))
			self.degs_up = list(self.dfdegs_up.symbol)
			self.degs_up.sort()
			self.n_degs_up = len(self.degs_up)

		self.dfdegs_up_ensembl = dfdegs_ensembl[dfdegs_ensembl.lfc > 0].copy()
		if self.dfdegs_up_ensembl.empty:
			self.degs_up_ensembl = []
			self.n_degs_up_ensembl = 0
		else:
			self.dfdegs_up_ensembl.index = np.arange(0, len(self.dfdegs_up_ensembl))
			self.degs_up_ensembl = list(self.dfdegs_up_ensembl.symbol)
			self.degs_up_ensembl.sort()
			self.n_degs_up_ensembl = len(self.degs_up_ensembl)

		''' ---------- Down ------------------------'''
		self.dfdegs_dw = dflfc[dflfc.lfc < 0].copy()
		if self.dfdegs_dw.empty:
			self.degs_dw = []
			self.n_degs_dw = 0
		else:
			self.dfdegs_dw.index = np.arange(0, len(self.dfdegs_dw))
			self.degs_dw = list(self.dfdegs_dw.symbol)
			self.degs_dw.sort()
			self.n_degs_dw = len(self.degs_dw)

		self.dfdegs_dw_ensembl = dfdegs_ensembl[dfdegs_ensembl.lfc < 0].copy()
		if self.dfdegs_dw_ensembl.empty:
			self.degs_dw_ensembl = []
			self.n_degs_dw_ensembl = 0
		else:
			self.dfdegs_dw_ensembl.index = np.arange(0, len(self.dfdegs_dw_ensembl))
			self.degs_dw_ensembl = list(self.dfdegs_dw_ensembl.symbol)
			self.degs_dw_ensembl.sort()
			self.n_degs_dw_ensembl = len(self.degs_dw_ensembl)

		fname_final_ori, fname_ori, title_dummy = self.set_lfc_names()
		fname0 = fname_final_ori

		biotype_list = list(self.dfg_dflc.biotype)
		biotype_list.sort()

		self.dfg_up_biotype, self.dfg_dw_biotype = None, None

		if self.n_degs_up > 0:
			for biotype in biotype_list:
				dfa = self.dfdegs_up[self.dfdegs_up.biotype == biotype]

				if save_file and not dfa.empty:
					fname_up = fname0.replace('_ALL_corrected_', f"{stri}UP_{biotype}_").replace('.tsv', '.txt')
					fullname = os.path.join(self.root_result, fname_up)

					if not os.path.exists(fullname) or force:
						text = "\n".join(dfa.symbol)
						write_txt(text, fname_up, self.root_result, verbose=verbose)

		if self.n_degs_dw > 0:
			for biotype in biotype_list:
				dfa = self.dfdegs_dw[self.dfdegs_dw.biotype == biotype]

				if save_file and not dfa.empty:
					fname_dw = fname0.replace('_ALL_corrected_', f"{stri}DW_{biotype}_").replace('.tsv', '.txt')
					fullname = os.path.join(self.root_result, fname_dw)

					if not os.path.exists(fullname) or force:
						text = "\n".join(dfa.symbol)
						write_txt(text, fname_dw, self.root_result, verbose=verbose)


		if prompt_verbose:
			print(f"\t{self.s_deg_dap}s {len(self.degs)}")
			print(f"\t\tUp (#{len(self.degs_up)})")
			print(f"\t\tDw (#{len(self.degs_dw)})")

			print("\nUp-regulated per biotype")
			if self.n_degs_up > 0:
				dfg = self.get_df_biotypes(self.dfdegs_up)
				self.dfg_up_biotype = dfg
				print(dfg)

			print("\nDown-regulated per biotype")
			if self.n_degs_dw > 0:
				dfg = self.get_df_biotypes(self.dfdegs_dw)
				self.dfg_dw_biotype = dfg
				print(dfg)


		return degs, degs_ensembl, dflfc

	def open_simulation_table(self, force:bool=False, verbose:bool=False):

		if self.dfsim is not None and not force:
			return self.dfsim

		fullname = os.path.join(self.root_config, self.cfg.fname_lfc_cutoff)

		if os.path.exists(fullname):
			dfsim = pdreadcsv(self.cfg.fname_lfc_cutoff, self.root_config, verbose=verbose)
		else:
			print(f"Could not found simulation table: {fullname}")
			dfsim = None

		self.dfsim = dfsim

		if dfsim is not None and not dfsim.empty:
			self.fdr_list = np.unique(dfsim.fdr_lfc_cutoff)
			self.lfc_list = np.unique(dfsim.abs_lfc_cutoff)

		return dfsim


	'''
	cutoff_list = [(1, 0.05), ...]
	'''
	def calc_degs_cutoff_simulation(self, cutoff_list:List=None, force:bool=False, save_file:bool=False, 
									n_echo:int=-1, verbose:bool=False) -> pd.DataFrame:

		fullname = os.path.join(self.root_config, self.cfg.fname_lfc_cutoff)

		if os.path.exists(fullname) and not force:
			dfsim = pdreadcsv(self.cfg.fname_lfc_cutoff, self.root_config)
			self.dfsim = dfsim
			return dfsim

		if cutoff_list is None:
			cutoff_list = np.round([(x, y) for x in self.lfc_list for y in self.fdr_list],3)

		icount=-1
		dic= {}
		for case in self.case_list:

			print(">>>", case)
			if not self.open_case_simple(case):
				continue

			for abs_lfc_cutoff, fdr_lfc_cutoff in cutoff_list:
				# print("\t###", abs_lfc_cutoff, fdr_lfc_cutoff)
				_, _, _ = self.list_of_degs_set_params(abs_lfc_cutoff, fdr_lfc_cutoff, force=force, save_file=save_file, verbose=verbose)

				icount += 1
				dic[icount] = {}
				dic2 = dic[icount]

				dic2['case'] = case
				dic2['normalization'] = self.normalization

				dic2['cutoff'] = f"{abs_lfc_cutoff:.3f} - {fdr_lfc_cutoff:.3f}"

				dic2['abs_lfc_cutoff'] = abs_lfc_cutoff
				dic2['fdr_lfc_cutoff'] = fdr_lfc_cutoff

				dic2['degs'] = self.degs
				dic2['n_degs'] = self.n_degs
				dic2['degs_ensembl'] = self.degs_ensembl
				dic2['n_degs_ensembl'] = self.n_degs_ensembl

				dic2['degs_up'] = self.degs_up
				dic2['n_degs_up'] = self.n_degs_up
				dic2['degs_up_ensembl'] = self.degs_up_ensembl
				dic2['n_degs_up_ensembl'] = self.n_degs_up_ensembl

				dic2['degs_dw'] = self.degs_dw
				dic2['n_degs_dw'] =self.n_degs_dw
				dic2['degs_dw_ensembl'] = self.degs_dw_ensembl
				dic2['n_degs_dw_ensembl'] = self.n_degs_dw_ensembl

				if n_echo > 9 and icount%n_echo==0:

					print(f"{self.s_deg_dap} cutoff: lfc={self.abs_lfc_cutoff:.3f}; lfc_fdr={self.fdr_lfc_cutoff:.3f}")
					print(f"\tthere are {len(self.degs)} {self.s_deg_dap}s")
					print(f"\t\t{self.n_degs_up} Up: ({','.join(self.degs_up)})")
					print(f"\t\t{self.n_degs_dw} Dw: ({','.join(self.degs_dw)})")
					print("")

		dfsim = pd.DataFrame(dic).T
		self.dfsim = dfsim
		self.cfg.save_best_lfc_cutoff(dfsim, verbose=True)

		return dfsim


	def prepare_df_two_conditions(self, case, dfn=None):

		self.split_case(case)
		ret_lfc = self.open_dflfc_ori()

		df_lfc = self.dflfc_ori

		cols = ['symbol', 'lfc', 'pval', 'fdr']
		if dfn is None:
			dfn = pd.merge(self.dfsun, df_lfc[cols], how='outer', on='symbol')
		else:
			dfn = pd.merge(dfn, df_lfc[cols], how='outer', on='symbol')

		dfn = dfn[~pd.isnull(dfn.biopax_num)]
		dfn.biopax_num = dfn.biopax_num.astype(int)
		dfn.num_parent = dfn.num_parent.astype(int)

		cols = list(dfn.columns)
		dfn.lfc = [0 if pd.isnull(x) else x for x in dfn.lfc]
		cols[-1] = 'fdr_%s'%(case)
		cols[-2] = 'pval_%s'%(case)
		cols[-3] = 'lfc_%s'%(case)
		dfn.columns = cols

		return dfn


	def compare_two_conditions(self, verbose=False):
		if verbose: print("Comparing 2 conditions ...")
		case1, case2 = self.group1, self.group2

		df_comp = self.prepare_df_two_conditions(case1, None)
		df_comp = self.prepare_df_two_conditions(case2, df_comp)
		df_comp.index = np.arange(0, len(df_comp))

		exc_list=[]; diff_list=[]
		for i in range(len(df_comp)):
			lfc1 = df_comp.iloc[i]['lfc_%s'%(case1)]
			lfc2 = df_comp.iloc[i]['lfc_%s'%(case2)]
			diff = lfc2 - lfc1

			exc = True if lfc1*lfc2 < 0 else False

			exc_list.append(exc)
			diff_list.append(diff)

		df_comp['exchange_sign'] = exc_list
		df_comp['lfc_diff'] = diff_list
		df_comp['abs_lfc_diff'] = np.abs(diff_list)

		df_comp = df_comp.sort_values('abs_lfc_diff', ascending=False)
		df_comp.index = np.arange(0, len(df_comp))

		self.df_comp = df_comp
		return df_comp


	def calc_up_donw_reg(self, val2, val1, cutoff=0.4):

		diff = val2 - val1

		if val2 > cutoff:
			stri2 = 'up'
		elif val2 < -cutoff:
			stri2 = 'dw'
		else:
			stri2 = '--'

		if val1 > cutoff:
			stri1 = 'up'
		elif val1 < -cutoff:
			stri1 = 'dw'
		else:
			stri1 = '--'

		upup_dwdw = stri2 + '/' + stri1

		if stri2 == 'up' and stri1 == 'dw' or stri1 == 'up' and stri2 == 'dw':
			signal = ' and inverted'
		else:
			signal = '			 '

		if diff >= cutoff:
			modulated = 'upmodulated	'
		elif diff <= -cutoff:
			modulated = 'downmodulated  '
		else:
			modulated = '~ not modulated'

		return upup_dwdw, modulated+signal

	def find_genelist_in_one_case(self, case:str, genelist:list, verbose=False):
		ret, _, _, _ = self.open_case(case, verbose=verbose)
		if not ret:
			return None

		dflfc = self.dflfc.copy()
		dflfc = dflfc[dflfc.symbol.isin(genelist)].copy()

		return dflfc

	def find_gene_in_case(self, case, gene, verbose=False):
		ret, _, _, _ = self.open_case(case, verbose=verbose)
		if not ret:
			return None

		return self.dflfc[self.dflfc.symbol == gene].copy()

	def find_many_genes_in_degs_all_cases(self, gene_list:List, verbose:bool=False) -> (pd.DataFrame, List, dict):
		df_list = []
		dic_case = {}
		for case in self.case_list:
			dfg = self.find_genelist_in_one_case(case, gene_list, verbose=verbose)
			if dfg is None or dfg.empty: continue

			dfg['case'] = case
			cols = list(dfg.columns)
			cols = [cols[-1]] + cols[:-1]
			dfg = dfg[cols]
			df_list.append(dfg)

			dic_case[case] = dfg.symbol.to_list()

		if df_list != []:
			dfall = pd.concat(df_list)
			dfall.index = np.arange(0, len(dfall))

			unique_genes = np.unique(dfall.symbol)
		else:
			dfall = None
			unique_genes = []

		return dfall, unique_genes, dic_case


	def calc_venn_df_summ_between_two_cases(self, case1:str, case2:str, col:str, first150:bool=False,
												  use_ensembl_degs:bool=True, in_pathway:bool=False,
												  print_inverted:bool=True, figsize:tuple=(8,6),
												  title_font_size=15, verbose:bool=True):

		if self.df_summ is None or self.df_summ.empty:
			print("Impossible to continue: self.df_summ is None or empty")
			return None, None, None, None, None, None, None, None, None
		# 1145, 269, 980
		# 1285, 762, 1188
		if in_pathway:
			if col.startswith('degs_up'):
				col = 'degs_up_ensembl_in_pathways'
			else:
				col = 'degs_dw_ensembl_in_pathways'
		else:
			if use_ensembl_degs:
				if col.startswith('degs_up'):
					col = 'degs_up_ensembl'
				else:
					col = 'degs_dw_ensembl'

		# print(f">>>> {col} in_pathway {in_pathway} use_ensembl_degs {use_ensembl_degs}")
		print(f">>>> Selected column: {col}")

		if '_dw' in col:
			col_inv = col.replace('_dw', '_up')
		else:
			col_inv = col.replace('_up', '_dw')

		vals1 = self.df_summ.loc[self.df_summ.case == case1].iloc[0][col]
		if isinstance(vals1, str): vals1 = eval(vals1)
		if first150 and len(vals1) > 150:
			vals1 = vals1[:150]
		set1 = set(vals1)

		vals1_inv = self.df_summ.loc[self.df_summ.case == case1].iloc[0][col_inv]
		if isinstance(vals1_inv, str): vals1_inv = eval(vals1_inv)
		set1_inv = set(vals1_inv)

		# all_vals1 = vals1 + vals1_inv

		vals2 = self.df_summ.loc[self.df_summ.case == case2].iloc[0][col]
		if isinstance(vals2, str): vals2 = eval(vals2)
		if first150 and len(vals2) > 150:
			vals2 = vals2[:150]
		set2 = set(vals2)

		vals2_inv = self.df_summ.loc[self.df_summ.case == case2].iloc[0][col_inv]
		if isinstance(vals2_inv, str): vals2_inv = eval(vals2_inv)
		set2_inv = set(vals2_inv)

		# all_vals2 = vals2 + vals2_inv

		vals1_inverted = [x for x in vals1 if x in vals2_inv]
		vals2_inverted = [x for x in vals2 if x in vals1_inv]

		inverted_degs = np.unique(vals1_inverted + vals2_inverted)

		''' --- without contraries -------------'''
		all_vals = list(np.unique(vals1 + vals2))

		fig = plt.figure(figsize=figsize)

		venn2([set1, set2], (case1, case2))

		col2 = "_".join(col.split('_')[:2])
		if in_pathway:
			title = f"{case1} x {case2} for {col2} in pathways - has ensembl\ntotal #{len(all_vals)}"
		else:
			if use_ensembl_degs:
				title = f"{case1} x {case2} for {col2} not in pathways - has ensembl\ntotal #{len(all_vals)}"
			else:
				title = f"{case1} x {case2} for {col2} not in pathways - all\ntotal #{len(all_vals)}"

		if first150:
			title += ' (first 150 genes)'
		plt.title(title, size=title_font_size);

		common = list(set1.intersection(set2))
		common.sort()

		only1 = [x for x in vals1 if x not in common]
		only1.sort()

		only2 = [x for x in vals2 if x not in common]
		only2.sort()

		s_all	 = f"all {self.s_gene_protein}s (#{len(all_vals)}): {', '.join(all_vals)}"
		s_commons = f"common (#{len(common)}): {', '.join(common)}"
		s_case1   = f"only {case1} (#{len(only1)}): {', '.join(only1)}"
		s_case2   = f"only {case2} (#{len(only2)}): {', '.join(only2)}"


		if len(inverted_degs) == 0:
			s_inverted = 'No inversions'
		else:
			s_inverted = f"inverted (#{len(inverted_degs)}): {', '.join(inverted_degs)}"


		if verbose:
			print(f'\n{s_commons}\n')
			print(s_case1, '\n')
			print(s_case2, '\n')

			if print_inverted: print(s_inverted,'\n')
		else:
			if print_inverted: print(s_inverted)

		return fig, s_commons, s_case1, s_case2, s_inverted, common, only1, only2, inverted_degs

	def calc_venn_dfenrich_between_two_cases(self, case1:str, case2:str, figsize:tuple=(8,6), 
		                                     verbose:bool=True):

		if self.df_summ is None:
			self.open_enriched_pathways_summary()

		dfqq = self.df_summ.loc[self.df_summ.case == case1]
		if dfqq.empty:
			print("No summary for {case1}")
			return None, None, None, None, None, None, None
		vals1 = eval(dfqq.iloc[0].pathway_list)
		set1 = set(vals1)

		dfqq = self.df_summ.loc[self.df_summ.case == case2]
		if dfqq.empty:
			print("No summary for {case2}")
			return None, None, None, None, None, None, None
		vals2 = eval(dfqq.iloc[0].pathway_list)
		set2 = set(vals2)

		all_vals = list(np.unique(vals1 + vals2))

		fig = plt.figure(figsize=figsize)

		venn2([set1, set2], (case1, case2))

		title = f"{case1} x {case2}: total enriched pathways = #{len(all_vals)}"
		plt.title(title, size=20);

		common = list(set1.intersection(set2))
		common.sort()

		only1 = [x for x in vals1 if x not in common]
		only1.sort()

		only2 = [x for x in vals2 if x not in common]
		only2.sort()

		s_all	 = f"all pathways (#{len(all_vals)}): {', '.join(all_vals)}"
		s_commons = f"common (#{len(common)}): {', '.join(common)}"
		s_case1   = f"only {case1} (#{len(only1)}): {', '.join(only1)}"
		s_case2   = f"only {case2} (#{len(only2)}): {', '.join(only2)}"

		if verbose:
			print(s_commons, '\n')
			print(s_case1, '\n')
			print(s_case2)

		return fig, s_commons, s_case1, s_case2, common, only1, only2


	def summary_degs_up_down(self, per_biotype:bool=False, ensembl:bool=False,
							 save_file:bool=False, verbose:bool=False) -> pd.DataFrame:

		dic = {}; i=-1
		for case in self.case_list:
			ret, _, _, _ = self.open_case(case, verbose=False)
			if not ret:
				continue

			i+=1
			dic[i] = {}
			dic2 = dic[i]
			dic2['case'] = case
			dic2['tot_measured'] = len(self.dflfc_ori)

			if per_biotype:
				''' upregulated all or ensembl '''
				if ensembl:
					df = self.dfdegs_up_ensembl
					sufix = 'up_ens'
				else:
					df = self.dfdegs_up
					sufix = 'up'

				if df is not None and not df.empty:
					dfg = self.get_df_biotypes(df)

					for j in range(len(dfg)):
						row = dfg.iloc[j]
						dic2[f"{row.biotype}_{sufix}"] = row['n']

				''' dwregulated all or ensembl '''
				if ensembl:
					df = self.dfdegs_dw_ensembl
					sufix = 'dw_ens'
				else:
					df = self.dfdegs_dw
					sufix = 'dw'

				if df is not None and not df.empty:
					dfg = self.get_df_biotypes(df)

					for j in range(len(dfg)):
						row = dfg.iloc[j]
						dic2[f"{row.biotype}_{sufix}"] = row['n']

			else:
				dic2['n_degs']	= self.n_degs
				dic2['n_degs_up'] = self.n_degs_up
				dic2['n_degs_up_ensembl'] = self.n_degs_up_ensembl
				dic2['n_degs_dw'] = self.n_degs_dw
				dic2['n_degs_dw_ensembl'] = self.n_degs_dw_ensembl


		dfa = pd.DataFrame(dic).T
		dfa = dfa.infer_objects(copy=False).fillna(0)

		cols = list(dfa.columns)[1:]
		dfa[cols] = dfa[cols].astype(int)

		dfa = dfa.set_index('case')
		dfa = dfa.T


		if save_file:
			fname = f'resume_degs_per_biotype_{per_biotype}_ensembl_{ensembl}.tsv'
			pdwritecsv(dfa, fname, self.root_ressum, verbose=verbose)

		return dfa

	def barplot_up_down_genes_per_case(self, per_biotype:bool=False, ensembl:bool=False,
									   before_best_cutoff:bool=False, title:str=None,
									   log10:bool=False, show_val:bool=False,
									   yaxis_title:str=None, width:int=800, height:int=600,
									   plot_bgcolor:str='lightgray', nround:float=2, verbose:bool=False):

		dfa = self.summary_degs_up_down(per_biotype=per_biotype, ensembl=ensembl, verbose=verbose)
		dfa = dfa.reset_index()
		cols = list(dfa.columns)
		cols[0] = 'deg_col'
		dfa.columns = cols
		dfa.index.name = 'index'

		dfa = dfa[dfa.deg_col != 'tot_measured']
		dfa.index = np.arange(0, len(dfa))

		before = 'before best cutoff' if before_best_cutoff else 'with best cutoff'

		if title is None:
			if not per_biotype:
				title = f'Up and Down {self.s_deg_dap}s {before}'
			else:
				if not ensembl:
					title = f'Up and Down {self.s_deg_dap}s per biotype {before}'
				else:
					title = f'Up and Down {self.s_deg_dap}s per biotype having ensembl_id {before}'


		if yaxis_title is None:
			yaxis_title = f'# {self.s_deg_dap}s'


		if log10:
			yaxis_title = 'log10 ' + yaxis_title

		colors = plotly_my_colors

		fig = go.Figure()

		cols = list(dfa.columns)
		
		pos_plot = -2
		for case in self.list_order:
			pos_plot += 1

			general_max = dfa[case].max()
			if log10:
				general_max = np.log10(general_max)

			for i in range(len(dfa)):
				pos_plot += 1
				row = dfa.iloc[i]

				color = colors[i]
				col   = row.deg_col
				val0  = row[case]

				if log10:
					val = np.log10(val0) if val0 > 0 and pd.notnull(val0) else 0
				else:
					val = val0

				fig.add_trace(go.Bar(x=[pos_plot], y=[val], marker_color=color, name=col))

				if show_val:

					if pd.notnull(val) and val != 0:
						# in graphic_lib
						delta_y = define_delta_y(val, general_max)

						fig.add_annotation(text=str(np.round(val0,nround)), x=pos_plot, y=val+delta_y, showarrow=False)


		fig.update_layout(
					autosize=True,
					title=title,
					width=width,
					height=height,
					plot_bgcolor=plot_bgcolor,
					xaxis_title="cases",
					xaxis_showticklabels=False,
					yaxis_title=yaxis_title,
					showlegend=True,
					font=dict(
						family="Arial",
						size=14,
						color="Black"
					)
		)


		figname = title_replace(title)
		figname = os.path.join(self.root_figure, figname+'.html')

		fig.write_html(figname)
		if verbose: print(">>> HTML and png saved:", figname)
		fig.write_image(figname.replace('.html', '.png'))

		return fig, dfa

	def count_sampling(self, geneset_num_list:List=[0, 1, 2, 4, 5, 7], prompt_verbose:bool=False):

		dic={}; i=-1
		for geneset_num in geneset_num_list:
			self.set_db(geneset_num, verbose=True)

			s_start = f"enricher_{self.geneset_lib}"

			for case in self.case_list:
				files = [x for x in os.listdir(self.root_enrichment) if x.startswith(s_start) and case in x]
				if prompt_verbose: print("\tcase", case, len(files))

				i+=1
				dic[i] = {}
				dic2 = dic[i]

				dic2['geneset_num'] = self.geneset_num
				dic2['geneset_lib'] = self.geneset_lib
				dic2['case'] = case
				dic2['n'] = len(files)

			if prompt_verbose: print('')

		dfa = pd.DataFrame(dic).T
		return dfa

	def plot_degs_in_pathways_vs_toi_per_case(self, selected_toi_col:str='toi4_median', title:str=None,
								 width:int=1100, height:int=600, plot_all_dfi:bool=True, sel_colors:List=None,
								 plot_bgcolor:str='lightgray', verbose:bool=False):
		if title is None:
			title=f"The {selected_toi_col} space"

		if sel_colors is None:
			sel_colors = sel_degs_pathways_colors

		yaxis_title1 = f"# {self.s_deg_dap}s in pathways"
		yaxis_title2 = "# of pathways"

		dfcut = self.build_all_cutoffs_table(selected_toi_col, force=False, verbose=verbose)

		dfcut = dfcut.sort_values(['case', selected_toi_col], ascending=[True, True])
		dfcut.index = np.arange(0, len(dfcut))

		subplot_titles=[f"# {self.s_deg_dap}s in pathways", "# pathways", f"# pathways per {self.s_deg_dap}s in pathways"]
		fig = make_subplots(rows=3, cols=1, subplot_titles=subplot_titles)

		df_all_fdr = self.open_all_fdr_lfc_correlation()

		'''------------ calc max -------------'''
		maxi_x = 0
		for icase in range(len(self.case_list)):
			case = self.case_list[icase]

			df_fdr = df_all_fdr[df_all_fdr.case == case]
			if df_fdr.empty:
				print(f"Error: no correlation was calculated for case '{case}'")
				raise Exception('stop: plot_degs_in_pathways_vs_toi_per_case()')

			for fdr in df_fdr.fdr:
				df2 = dfcut[ (dfcut.case == case) & (dfcut.fdr_lfc_cutoff == fdr)  & (dfcut.med_max_ptw == 'median')]
				if df2.empty:
					continue

				maxi_x2 = np.round(df2[selected_toi_col].max(), 3) + 0.005
				if maxi_x2 > maxi_x:
					maxi_x = maxi_x2


		'''------------ plot loop ------------'''
		dic_visible = {}
		for icase in range(len(self.case_list)):
			case = self.case_list[icase]
			if plot_all_dfi: 
				dfi = self.calc_enrichment_cutoff_params_and_ndxs_per_case_and_geneset_lib(case)

			df_fdr = df_all_fdr[df_all_fdr.case == case]
			if df_fdr.empty:
				print(f"Error: no correlation was calculated for case '{case}'")
				raise Exception('stop: plot_degs_in_pathways_vs_toi_per_case()')

			dic_visible[case] = 0

			is_visible = True if icase == 0 else False
			i = -1;
			for fdr in df_fdr.fdr:
				i += 1

				df2 = dfcut[ (dfcut.case == case) & (dfcut.fdr_lfc_cutoff == fdr)  & 
							 (dfcut.med_max_ptw == 'median') &
							 (dfcut.n_pathways  >  0) ]
				if df2.empty:
					continue

				if plot_all_dfi:
					dfi2 = dfi[ (dfi.case == case) & (dfi.fdr_lfc_cutoff == fdr) ].copy()
					dfi2['pathways_per_degs'] = dfi2.n_pathways / dfi2.n_degs_in_pathways
				else:
					df2['pathways_per_degs'] = df2.n_pathways / df2.n_degs_in_pathways


				text_ini = f'case {case}<br>FDR_LFC cutoff={fdr:.3f}'

				hovertext_list = []
				for j in range(len(df2)):
					row = df2.iloc[j]
					text =  f'LFC_cutoff={row.abs_lfc_cutoff:.3f}<br>FDR_pathway={row.pathway_fdr_cutoff:.3f}<br>toi4={row.toi4_median:.3f}'
					text += f'<br>---------<br># {self.s_deg_dap}s in ptws={row.n_degs_in_pathways}<br># pathways={row.n_pathways}'
					hovertext_list.append(text_ini + '<br>' + text)

				'''--- mask of trues and falses '''
				dic_visible[case] += 3

				name1 = f"{case} fdr={fdr:.3f} for {self.s_deg_dap}s"
				name2 = f"{case} fdr={fdr:.3f} for pathways"
				name3 = f"{case} fdr={fdr:.3f} for pathways/{self.s_deg_dap}s"

				if plot_all_dfi: 
					dfi2 = dfi2.sort_values(selected_toi_col, ascending=True)

				color = sel_colors[i]

				if plot_all_dfi:
					fig.add_trace(go.Scatter(x=dfi2[selected_toi_col], y=dfi2.n_degs_in_pathways, line=dict(dash='dash'), marker_color=color, name=name1), row=1, col=1 )
					fig.add_trace(go.Scatter(x=dfi2[selected_toi_col], y=dfi2.n_pathways, line=dict(dash='dash'), marker_color=color, name=name2), row=2, col=1 )
					fig.add_trace(go.Scatter(x=dfi2[selected_toi_col], y=dfi2.pathways_per_degs, line=dict(dash='dash'), marker_color=color, name=name3), row=3, col=1 )
				else:
					fig.add_trace(go.Scatter(x=df2[selected_toi_col], y=df2.n_degs_in_pathways,  hovertext=hovertext_list, hoverinfo="text",
											 marker_color=color, visible=is_visible, name=name1), row=1, col=1 )
					fig.add_trace(go.Scatter(x=df2[selected_toi_col], y=df2.n_pathways, hovertext=hovertext_list, hoverinfo="text",
											 marker_color=color, visible=is_visible, name=name2), row=2, col=1 )
					fig.add_trace(go.Scatter(x=df2[selected_toi_col], y=df2.pathways_per_degs, hovertext=hovertext_list, hoverinfo="text",
											 marker_color=color, visible=is_visible, name=name3), row=3, col=1 )

				fig.update_layout(title=title,
								  width=width,
								  height=height,
								  xaxis_title=selected_toi_col,
								  yaxis_title=yaxis_title1,
								  yaxis2_title=yaxis_title2,
								  xaxis2_title=selected_toi_col,
								  xaxis=dict(range=[0, maxi_x]),
								  plot_bgcolor=plot_bgcolor,
								  legend_title="cases",
								  showlegend=True)

		# add dropdown menus to the figure
		buttons=[]
		for case in self.case_list:
			buttons.append(dict(method='update',
								label=case,
								visible=True,
								args=[ {'visible': list(sum( [tuple([True]  * dic_visible[case2]) if case == case2 else \
															  tuple([False] * dic_visible[case2]) for case2 in self.case_list], () ))} ]
								)
						  )

		# some adjustments to the updatemenus
		updatemenu = []
		your_menu = dict()
		updatemenu.append(your_menu)

		updatemenu[0]['buttons'] = buttons
		updatemenu[0]['direction'] = 'down'
		updatemenu[0]['showactive'] = True
		updatemenu[0]['showactive'] = True
		updatemenu[0]['x'] = 1
		updatemenu[0]['y'] = 1.2

		fig.update_layout(title=title,
						  width=width,
						  height=height,
						  xaxis_title=selected_toi_col,
						  yaxis_title=yaxis_title1,
						  yaxis2_title=yaxis_title2,
						  xaxis2_title=selected_toi_col,
						  xaxis=dict( range=[0, maxi_x]),
						  xaxis2=dict( range=[0, maxi_x]),
						  legend_title="cases",
						  showlegend=True,
						  plot_bgcolor=plot_bgcolor,
						  updatemenus=updatemenu)

		figname = title_replace(title)
		figname = os.path.join(self.root_figure, figname+'.html')

		fig.write_html(figname)
		if verbose: print(">>> HTML and png saved:", figname)
		fig.write_image(figname.replace('.html', '.png'))

		return fig

	def plot_degs_vs_lfc_per_fdr_per_case(self, selected_toi_col:str='toi4_median', title:str=None,
								 width:int=1100, height:int=600, plot_all_dfi:bool=True, sel_colors:List=None,
								 plot_bgcolor:str='lightgray', verbose:bool=False):
		if title is None:
			title = f'scatter plot - {self.s_deg_dap}s versus abs_LFC per FDR'

		if sel_colors is None:
			sel_colors = sel_degs_pathways_colors

		xaxis_title = f"abs LFC"
		yaxis_title = f"# {self.s_deg_dap}s"

		dfsim = self.open_simulation_table()

		dfsim = dfsim.sort_values(['case', 'fdr_lfc_cutoff', 'abs_lfc_cutoff'], ascending=[True, False, False])

		fig = go.Figure()

		dic_visible = {}
		for icase in range(len(self.case_list)):
			case = self.case_list[icase]

			dic_visible[case] = 0
			is_visible = True if icase == 0 else False
			i = -1;
			for i in range(len(self.fdr_list)):
				fdr_lfc_cutoff = self.fdr_list[i]
				color = sel_colors[i]
				name = f"{fdr_lfc_cutoff:.3f}"

				dfsim2 = dfsim[ (dfsim.case == case) & (dfsim.fdr_lfc_cutoff == fdr_lfc_cutoff)]
				if dfsim2.empty:
					continue

				dic_visible[case] += 1

				text_ini = f'case {case}<br>FDR_LFC cutoff={fdr_lfc_cutoff:.3f}'

				hovertext_list = []
				for j in range(len(dfsim2)):
					row = dfsim2.iloc[j]
					text =  f'LFC_cutoff={row.abs_lfc_cutoff:.3f}'
					text += f'# {self.s_deg_dap}s {row.n_degs}<br># Up={row.n_degs_up} Down={row.n_degs_dw}'
					hovertext_list.append(text_ini + '<br>' + text)

				fig.add_trace(go.Scatter(x=dfsim2.abs_lfc_cutoff, y=dfsim2.n_degs, hovertext=hovertext_list, hoverinfo="text",
										 mode='markers', marker={'color':color}, visible=is_visible, name=name))

			fig.update_layout(
						autosize=True,
						title=title,
						width=width,
						height=height,
						xaxis_title=xaxis_title,
						yaxis_title=yaxis_title,
						showlegend=True,
						legend_title='FDR_LFC cutoff',
						plot_bgcolor=plot_bgcolor,
						font=dict(
							family="Arial",
							size=14,
							color="Black"
						)
			)

		# add dropdown menus to the figure
		buttons=[]
		for case in self.case_list:
			buttons.append(dict(method='update',
								label=case,
								visible=True,
								args=[ {'visible': list(sum( [tuple([True]  * dic_visible[case2]) if case == case2 else \
															  tuple([False] * dic_visible[case2]) for case2 in self.case_list], () ))} ]
								)
						  )

		# some adjustments to the updatemenus
		updatemenu = []
		your_menu = dict()
		updatemenu.append(your_menu)

		updatemenu[0]['buttons'] = buttons
		updatemenu[0]['direction'] = 'down'
		updatemenu[0]['showactive'] = True
		updatemenu[0]['showactive'] = True
		updatemenu[0]['x'] = 1
		updatemenu[0]['y'] = 1.2

		fig.update_layout(
			autosize=True,
			title=title,
			width=width,
			height=height,
			xaxis_title=xaxis_title,
			yaxis_title=yaxis_title,
			showlegend=True,
			legend_title='FDR_LFC cutoff',
			font=dict(
				family="Arial",
				size=14,
				color="Black"
			),
			plot_bgcolor=plot_bgcolor,
			updatemenus=updatemenu
		)

		figname = title_replace(title)
		figname = os.path.join(self.root_figure, figname+'.html')

		fig.write_html(figname)
		if verbose: print(">>> HTML and png saved:", figname)
		fig.write_image(figname.replace('.html', '.png'))

		return fig


	def barplot_sampling_cutoffs(self, title:str='Sampling cutoffs per geneset', yaxis_title:str='number of samples',
								 width:int=1100, height:int=600, geneset_num_list:List=[0, 1, 2, 4, 5, 7],
								 colors:List=['green', 'red', 'blue', 'brown', 'yellow', 'cyan', 'lightgreen', 'pink', 'gray', 'lightblue'],
								 plot_bgcolor:str='lightgray', prompt_verbose:bool=False, verbose:bool=False):

		dfa = self.count_sampling(geneset_num_list=geneset_num_list, prompt_verbose=prompt_verbose)

		geneset_lista = dfa.geneset_lib.unique()
		colors_geneset = colors[:len(geneset_lista)]

		_n_rows = int(np.ceil(len(self.case_list)/2))
		fig = make_subplots(rows=_n_rows, cols=2, subplot_titles=self.case_list)

		nrow=1; ncol=0
		for case in self.case_list:

			dfa2 = dfa[dfa.case == case].copy()
			dfa2 = dfa2.sort_values('geneset_lib')
			dfa2.index = np.arange(0, len(dfa2))

			ncol += 1
			if ncol == 3:
				ncol = 1
				nrow += 1

			fig.add_trace(go.Bar(x=dfa2.geneset_lib, y=dfa2.n, marker_color=colors_geneset, name=case), row=nrow, col=ncol)

		fig.update_layout(
					autosize=True,
					title=title,
					width=width,
					height=height*_n_rows,
					plot_bgcolor=plot_bgcolor,
					xaxis_title="",
					xaxis2_title="",
					yaxis_title=yaxis_title,
					showlegend=False,
					font=dict(
						family="Arial",
						size=14,
						color="Black"
					)
		)

		figname = title_replace(title)
		figname = os.path.join(self.root_figure, figname+'.html')

		fig.write_html(figname)
		if verbose: print(">>> HTML and png saved:", figname)
		fig.write_image(figname.replace('.html', '.png'))

		return fig, dfa


	def barplot_degs_summary(self, title:str=None, yaxis_title:str=None,
							 colors:List=None, width:int=900, height:int=600,
							 verbose:bool=False):

		if title is None:
			title = f"Number of {self.s_deg_dap}s and pathways per case"
		if yaxis_title is None:
			yaxis_title = f"# {self.s_deg_dap}s and pathways"

		if colors is None:
			colors = ['darkgreen', 'darkred', 'navy', 'olivedrab', 'red', 'blue', 'darkcyan']
			'''
			'orange', 'brown', 'darksalmon', 'marron',
			'magenta', 'darkturquoise', 'orange', 'darkred', 'indigo', 'magenta',  'black',
			'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgrey', 'olivedrab', 'navy']
			'''

		if self.df_summ is None:
			self.calc_degs_and_pathways_summary()

		if self.df_summ is None or self.df_summ.empty:
			print("Nothing found")
			return None

		fig = go.Figure()

		cols = ['n_degs', 'n_degs_up', 'n_degs_dw', 'n_degs_in_pathways',
				'n_degs_up_ensembl_in_pathways', 'n_degs_dw_ensembl_in_pathways', 'n_pathways']
		for i in range(len(cols)):
			col = cols[i]
			color = colors[i]

			fig.add_trace(go.Bar(x=self.df_summ.case, y=self.df_summ[col], name=col, marker_color=color))

		fig.update_layout(
					autosize=True,
					title=title,
					width=width,
					height=height,
					xaxis_title="cases",
					yaxis_title=yaxis_title,
					showlegend=True,
					font=dict(
						family="Arial",
						size=14,
						color="Black"
					)
		)

		figname = title_replace(title)
		figname = os.path.join(self.root_figure, figname+'.html')

		fig.write_html(figname)
		if verbose: print(">>> HTML and png saved:", figname)
		fig.write_image(figname.replace('.html', '.png'))

		return fig

	def calc_pMPI_pivot_summary(self, force:bool=False,
								verbose:bool=False) -> (pd.DataFrame, pd.DataFrame, str, pd.DataFrame):
		'''
			pMPI - pseudo Modulation Pathway Index
		'''

		dff, text_all, df_sum_ptw = self.summary_pathway_modulation(force=force, verbose=verbose)

		if dff is None or dff.empty:
			self.dfpiv = pd.DataFrame()
			self.selected_pivot_pathway_list = []
			return self.dfpiv, dff

		dfpiv = pd.pivot_table(dff, values='mod_pathway_index', index='pathway', columns='case', fill_value=None)
		self.dfpiv = dfpiv

		selected_pivot_pathway_list = []

		for pathway in dfpiv.index:
			selected_pivot_pathway_list.append(pathway)

		self.selected_pivot_pathway_list = selected_pivot_pathway_list

		return dfpiv, dff, text_all, df_sum_ptw

	def calc_pivot_one_pathway_gene_modulations(self, pathway_and_id:List, force:bool=False, verbose:bool=False) -> (pd.DataFrame, pd.DataFrame):

		dff = self.calc_one_pathway_all_cases_gene_modulations(pathway_and_id, force=force, verbose=verbose)

		if dff is None or dff.empty:
			self.dfpiv_symbs_cases = pd.DataFrame()
			self.selected_pivot_symb_list = []
			return self.dfpiv_symbs_cases, dff

		dfpiv_symbs_cases = pd.pivot_table(dff, values='lfc', index='symbol', columns='case', fill_value=None)
		self.dfpiv_symbs_cases = dfpiv_symbs_cases

		selected_pivot_symb_list = []

		for symbol in dfpiv_symbs_cases.index:
			selected_pivot_symb_list.append(symbol)

		self.selected_pivot_symb_list = selected_pivot_symb_list

		return dfpiv_symbs_cases, dff


	def df_to_plotly(self, df):
		return {'z': df.values.tolist(),
				'x': df.columns.tolist(),
				'y': df.index.tolist()}

	def plot_pathway_heatmap_simple(self, pathway_list:List, s_name:str, all_cases:bool=True,
								   zlim:int=4, maxLen:int=50, width:int=1100, height:int=700,
								   plot_bgcolor:str='lightgray', vertical_spacing=0.10,
								   verbose:bool=False):

		if not isinstance(pathway_list, list) or pathway_list == []:
			print("Pathway list is empty or not a list.")
			return None, None

		dfpiv, dff, text_all, df_sum_ptw = self.calc_pMPI_pivot_summary(verbose=verbose)

		not_in = [x for x in pathway_list if x not in self.selected_pivot_pathway_list]

		if len(not_in) > 0:
			print("Some pathways look wrong, please fix the pathway list.")
			print(">>>", "; ".join(not_in))
			return None, None

		zmin = -zlim
		zmax = +zlim

		fig = go.Figure()

		if all_cases:
			dfpiv2 =  pd.DataFrame(dfpiv.loc[pathway_list, :].copy())
			title = f"pMPI heatmap for '{s_name}'"
		else:
			cols = self.case
			dfpiv2 =  pd.DataFrame(dfpiv.loc[pathway_list, cols].copy())
			title = f"Heatmap for '{s_name}' for {self.case}"

		dfpiv2.index = [break_line_per_length(x, sep='<br>', maxLen=maxLen) for x in dfpiv2.index]

		df_vectors = self.df_to_plotly(dfpiv2)

		fig.add_trace(go.Heatmap(df_vectors, zmin=zmin, zmax=zmax, colorscale='RdBu_r' ))


		fig.update_layout(
					autosize=True,
					title=title,
					width=width,
					height=height,
					plot_bgcolor=plot_bgcolor,
					xaxis_title="cases",
					yaxis_title='pathways',
					showlegend=False,
					font=dict(
						family="Arial",
						size=14,
						color="Black"
					)
		)

		figname = title_replace(title)
		figname = os.path.join(self.root_figure, figname+'.html')

		fig.write_html(figname)
		if verbose: print(">>> HTML and png saved:", figname)
		fig.write_image(figname.replace('.html', '.png'))

		return fig, dfpiv2

	def plot_pathway_heatmap_by_gender(self, pathway_list:List, s_name:str, zlim:int=None, maxLen:int=50,
									   width:int=1100, height:int=700, plot_bgcolor:str='lightgray',
									   vertical_spacing=0.10, verbose:bool=False):

		if not isinstance(pathway_list, list) or pathway_list == []:
			print("Pathway list is empty or not a list.")
			return None, None, None

		dfpiv, dff, text_all, df_sum_ptw = self.calc_pMPI_pivot_summary(verbose=verbose)

		not_in = [x for x in pathway_list if x not in self.selected_pivot_pathway_list]

		if len(not_in) > 0:
			print("Some pathways look wrong, please fix the pathway list.")
			print(">>>", "; ".join(not_in))
			return None, None, None


		gender_list = ['female', 'male', 'male-female']
		fig = make_subplots(rows=3, cols=1, subplot_titles=gender_list, vertical_spacing=vertical_spacing)


		dfpiv_female = dfpiv.loc[pathway_list, self.group_female_list].copy()
		dfpiv_female.index = [break_line_per_length(x, sep='<br>', maxLen=maxLen) for x in dfpiv_female.index]
		cols = list(dfpiv_female.columns)
		cols = [x.replace('_female','') for x in cols]
		dfpiv_female.columns = cols

		dfpiv_male = dfpiv.loc[pathway_list, self.group_male_list].copy()
		dfpiv_male.index = [break_line_per_length(x, sep='<br>', maxLen=maxLen) for x in dfpiv_male.index]
		dfpiv_male.columns = cols


		if zlim is not None:
			zlim = np.abs(zlim)
			zmin = -zlim
			zmax = +zlim
		else:
			df2 = pd.concat([dfpiv_female, dfpiv_male])
			zlim = int(np.max(np.abs(df2))+1)
			del df2
			zmin = -zlim
			zmax = +zlim


		for gender in gender_list:
			if gender == 'female':
				# cols = self.group_female_list
				dfpiv3 = dfpiv_female
				nrow = 1
			elif gender == 'male':
				# cols = self.group_male_list
				dfpiv3 = dfpiv_male
				nrow = 2
			else:
				dfpiv3 = dfpiv_male.infer_objects(copy=False).fillna(0) - dfpiv_female.infer_objects(copy=False).fillna(0)
				nrow = 3

			df_vectors = self.df_to_plotly(dfpiv3)

			fig.add_trace(go.Heatmap(df_vectors, zmin=zmin, zmax=zmax,
									 colorscale='RdBu_r', name=gender), row=nrow, col=1)

		title = f"Heatmap for '{s_name}'"

		fig.update_layout(
					autosize=True,
					title=title,
					width=width,
					height=height,
					plot_bgcolor=plot_bgcolor,
					xaxis3_title="cases",
					yaxis_title='pathways',
					showlegend=True,
					font=dict(
						family="Arial",
						size=14,
						color="Black"
					)
		)

		figname = title_replace(title)
		figname = os.path.join(self.root_figure, figname+'.html')

		fig.write_html(figname)
		if verbose: print(">>> HTML and png saved:", figname)
		fig.write_image(figname.replace('.html', '.png'))


		return fig, dfpiv_female, dfpiv_male


	def which_reacotme_id(self, pathway):
		if self.dfr is None:
			dfr =  self.reactome.open_reactome(verbose=False)
			if dfr is None or dfr.empty:
				return None
			self.dfr = dfr

		dfq = self.dfr[self.dfr.pathway == pathway]
		if dfq.empty:
			return None
		return dfq.iloc[0].pathway_id

	def rebuild_column(self, group_age, gender):
		mat = group_age.split('_')

		if len(mat) == 1:
			return group_age + '_' + gender

		return mat[0] + '_' + gender + '_' + mat[1]

	def plot_and_calc_pathway_modulation(self, pathway_and_id_list:List, zlim:int=None, maxLen:int=75,
									   type_modulation:str='all', at_least:int=2, diff_cutoff:int=2,
									   width:int=1200, row_height:int=60, header_height:int=200,
									   plot_bgcolor:str='lightgray', horizontal_spacing=0.20,
									   verbose:bool=False) -> (dict, pd.DataFrame):

		self.type_modulation = type_modulation
		self.at_least	= at_least
		self.diff_cutoff = diff_cutoff

		dic_fig = {}; icount=-1
		dfpiv_female_list, dfpiv_male_list, dfpiv_diff_list = [], [], []

		for pathway_and_id in pathway_and_id_list:
			pathway, pathway_id = pathway_and_id
			if verbose: print(">>>", pathway, pathway_id, '\n')

			fig, dff, dfpiv_female, dfpiv_male, dfpiv_diff = \
			self.plot_one_pathway_genes_heatmap(pathway_and_id, type_modulation=type_modulation, at_least=at_least,
												diff_cutoff=diff_cutoff, width=width, row_height=row_height,
												header_height=header_height, plot_bgcolor=plot_bgcolor,
												horizontal_spacing=horizontal_spacing, verbose=verbose)

			if fig is None: continue

			icount += 1
			dic_fig[icount] = {}
			dic2 = dic_fig[icount]

			dic2['pathway_id'] = pathway_id
			dic2['pathway'] = pathway
			dic2['fig'] = fig
			dic2['dff'] = dff
			dic2['dfpiv_female'] = dfpiv_female
			dic2['dfpiv_male']   = dfpiv_male
			dic2['dfpiv_diff']   = dfpiv_diff

			dfpiv_female_list.append(dfpiv_female)
			dfpiv_male_list.append(dfpiv_male)
			dfpiv_diff_list.append(dfpiv_diff)


		if dfpiv_female_list == []:
			print("Nothing found.")
			return None, None

		dfpiv_female = pd.concat(dfpiv_female_list)
		dfpiv_female = dfpiv_female.drop_duplicates()
		cols = dfpiv_female.columns
		cols = [ self.rebuild_column(x, 'female') for x in cols]
		dfpiv_female.columns = cols

		dfpiv_male2 = pd.concat(dfpiv_male_list)
		dfpiv_male2 = dfpiv_male2.drop_duplicates()
		cols = dfpiv_male2.columns
		cols = [ self.rebuild_column(x, 'male') for x in cols]
		dfpiv_male2.columns = cols

		dfpiv_diff2 = pd.concat(dfpiv_diff_list)
		dfpiv_diff2 = dfpiv_diff2.drop_duplicates()
		cols = dfpiv_diff2.columns
		cols = [x + '_diff' for x in cols]
		dfpiv_diff2.columns = cols

		df = pd.concat([dfpiv_female, dfpiv_male2, dfpiv_diff2], axis=1)
		df = df.reset_index()
		cols = list(df.columns)
		cols[0] = self.s_deg_dap
		df = df.sort_values(cols[-1])

		col_ini_male   = 'g2a_male'
		col_ini_female = 'g2a_female'

		col_end_male   = 'g3_male_elder'
		col_end_female = 'g3_female_elder'

		df['diff_ini_end_male']	   = df[col_end_male]   - df[col_ini_male]
		df['diff_ini_end_male_abs']   = np.abs(df['diff_ini_end_male'])

		df['diff_ini_end_female']	 = df[col_end_female] - df[col_ini_female]
		df['diff_ini_end_female_abs'] = np.abs(df['diff_ini_end_female'])

		fname = f'pathway_pseudo_modulation_table_filtered_{type_modulation}_at_least_{at_least}_diff_cutoff_{diff_cutoff}.tsv'
		fullname = os.path.join(self.root_ptw_modulation, fname)

		if os.path.exists(fullname) and not force:
			ret = pdwritecsv(df, fname, self.root_ptw_modulation, verbose=verbose)


		return df, dic_fig


	def plot_one_pathway_genes_heatmap_by_gender(self, pathway_and_id:List, zlim:int=None, maxLen:int=75,
									   type_modulation:str='all', at_least:int=2, diff_cutoff:int=2,
									   width:int=1100, row_height:int=70, header_height:int=200,
									   plot_bgcolor:str='lightgray', horizontal_spacing=0.20,
									   verbose:bool=False):
		'''
			type_modulation = all, similar, different
		'''

		if not isinstance(pathway_and_id, list) or pathway_and_id == []:
			print("Pathway and ID is an empty list.")
			return None, None, None, None, None

		pathway, pathway_id = pathway_and_id

		dfpiv, dff = self.calc_pivot_one_pathway_gene_modulations(pathway_and_id, verbose=verbose)

		dfpiv_female, dfpiv_male = None, None

		gender_list = ['female', 'male', 'male-female']
		fig = make_subplots(rows=1, cols=3, subplot_titles=gender_list, horizontal_spacing=horizontal_spacing)


		dfpiv_female = dfpiv.loc[:, self.group_female_list].copy()
		cols = list(dfpiv_female.columns)
		cols = [x.replace('_female','') for x in cols]
		dfpiv_female.columns = cols

		dfpiv_male = dfpiv.loc[:, self.group_male_list].copy()
		dfpiv_male.columns = cols

		dfpiv_diff = dfpiv_male.infer_objects(copy=False).fillna(0) - dfpiv_female.infer_objects(copy=False).fillna(0)
		dfpiv_diff = dfpiv_diff.copy()

		if type_modulation != 'all':
			goods = []
			for i in range(len(dfpiv_diff)):
				vals = np.abs(dfpiv_diff.iloc[i])

				vals = [val for val in vals if val >= diff_cutoff]
				goods.append(len(vals) >= at_least)


			self.goods = goods

			if type_modulation == 'similar':
				goods = [not x for x in goods]
				s_filtered = f'- similar: at_least={at_least} diff_cutoff={diff_cutoff}'
			else:
				s_filtered = f'- different: at_least={at_least} diff_cutoff={diff_cutoff}'


			dfpiv_female = dfpiv_female.loc[goods]
			dfpiv_male   = dfpiv_male.loc[goods]
			dfpiv_diff   = dfpiv_diff.loc[goods]

			if dfpiv_female.empty:
				print(f"No genes are modulated using the filter {type_modulation} for '{pathway}'")
				return None, None, None, None, None
		else:
			s_filtered = ''

		''' order by Spearman correlation '''
		xseq = np.arange(0, len(cols))

		stat_list = []
		for i in range(len(dfpiv_diff)):
			vals = np.array(dfpiv_diff.iloc[i])
			res = stats.spearmanr(vals, xseq)
			stat_list.append(res.statistic)

		df2 = dfpiv_diff.copy()
		df2['stat_spearman'] = stat_list

		df2 = df2.sort_values('stat_spearman', ascending=True)
		rows = df2.index

		dfpiv_female = dfpiv_female.loc[rows]
		dfpiv_male   = dfpiv_male.loc[rows]
		dfpiv_diff   = dfpiv_diff.loc[rows]

		if zlim is None:
			dfall = pd.concat([dfpiv_female, dfpiv_male, dfpiv_diff])
			zlim = int(np.max(np.abs(dfall))+1)

		zmin = -zlim
		zmax = +zlim
		# colorscale='RdBu_r',

		for gender in gender_list:

			if gender == 'female':
				ncol = 1
				dfpiv3 = dfpiv_female
			elif gender == 'male':
				ncol = 2
				dfpiv3 = dfpiv_male
			else:
				ncol = 3
				dfpiv3 = dfpiv_diff

			df_vectors = self.df_to_plotly(dfpiv3)

			fig.add_trace(go.Heatmap(df_vectors, zmin=zmin, zmax=zmax,
									 colorscale='RdBu_r', name=gender ), row=1, col=ncol)

		title = f"{self.s_deg_dap} heatmap for {pathway} {s_filtered}"
		title = title.strip()

		nrows = len(dfpiv3)
		height = row_height * nrows + header_height

		fig.update_layout(
					autosize=True,
					title=title,
					width=width,
					height=height,
					plot_bgcolor=plot_bgcolor,
					xaxis2_title="cases",
					yaxis_title=f"{self.s_deg_dap}s",
					yaxis2_title=f"{self.s_deg_dap}s",
					showlegend=True,
					font=dict(
						family="Arial",
						size=14,
						color="Black"
					)
		)

		figname = title_replace(title)
		figname = os.path.join(self.root_figure, figname+'.html')
		if verbose: print(">>> HTML and png saved:", figname)

		try:
			fig.write_html(figname)
			fig.write_image(figname.replace('.html', '.png'))
		except:
			print(f"Error writing {figname}")

		return fig, dff, dfpiv_female, dfpiv_male, dfpiv_diff


	def plot_one_pathway_genes_heatmap_by_cases(self, pathway_and_id:List, zlim:int=None,
									   type_modulation:str='all', similar_higher_than:int=2,
									   up_down_all:str='all', diff_cutoff:int=1,
									   width:int=1100, row_height:int=70, header_height:int=200,
									   plot_bgcolor:str='lightgray', horizontal_spacing=0.20,
									   verbose:bool=False):
		'''
			type_modulation = all, similar, different
		'''

		if not isinstance(pathway_and_id, list) or pathway_and_id == []:
			print("Pathway and ID is an empty list.")
			return None, None, None

		pathway, pathway_id = pathway_and_id

		dfpiv, dff = self.calc_pivot_one_pathway_gene_modulations(pathway_and_id, verbose=verbose)

		fig = go.Figure()

		cols = list(dfpiv.columns)

		dfpiv = dfpiv.fillna(0)
		dfpiv['diff'] = dfpiv[cols[1]] - dfpiv[cols[0]]
		dfpiv['abs_diff'] = np.abs(dfpiv[cols[1]] - dfpiv[cols[0]])
		dfpiv = dfpiv.sort_values('diff', ascending=True)

		if type_modulation == 'all':
			if up_down_all == 'up':
				dfpiv = dfpiv[ (dfpiv[cols[0]] >= similar_higher_than) | (dfpiv[cols[1]] >=  similar_higher_than) ]
				s_filtered = f'- upregulated: abs any LFC >= {similar_higher_than}'

			elif up_down_all == 'dw':
				dfpiv = dfpiv[ (dfpiv[cols[0]] <= -similar_higher_than) | (dfpiv[cols[1]] <= -similar_higher_than) ]
				s_filtered = f'- downregulated: abs any LFC <= {-similar_higher_than}'

			else:
				dfpiv = dfpiv[ (dfpiv[cols[0]] <= -similar_higher_than) | (dfpiv[cols[0]] >= similar_higher_than) |
				               (dfpiv[cols[1]] <= -similar_higher_than) | (dfpiv[cols[1]] >= similar_higher_than)]

				s_filtered = f'- all: abs any LFC >= {similar_higher_than}'

		elif type_modulation == 'similar':
			dfpiv = dfpiv[dfpiv.abs_diff < diff_cutoff]

			if up_down_all == 'up':
				dfpiv = dfpiv[ (dfpiv[cols[0]] >=  similar_higher_than) | (dfpiv[cols[1]] >=  similar_higher_than) ]
				s_filtered = f'- similar: abs diff < {diff_cutoff} and upregulated: abs any LFC >= {similar_higher_than}'

			elif up_down_all == 'dw':
				dfpiv = dfpiv[ (dfpiv[cols[0]] <= -similar_higher_than) | (dfpiv[cols[1]] <= -similar_higher_than) ]
				s_filtered = f'- similar: abs diff < {diff_cutoff} and downregulated: abs any LFC <= {-similar_higher_than}'
			else:
				dfpiv = dfpiv[ (dfpiv[cols[0]] <= -similar_higher_than) | (dfpiv[cols[0]] >= similar_higher_than) |
				               (dfpiv[cols[1]] <= -similar_higher_than) | (dfpiv[cols[1]] >= similar_higher_than)]

				s_filtered = f'- similar: abs diff < {diff_cutoff}'
		else:
			dfpiv = dfpiv[dfpiv.abs_diff >= diff_cutoff]

			if up_down_all == 'up':
				dfpiv = dfpiv[ (dfpiv[cols[0]] >= similar_higher_than) | (dfpiv[cols[1]] >=  similar_higher_than) ]
				s_filtered = f'- different: abs diff >= {diff_cutoff} and upregulated: abs any LFC >= {similar_higher_than}'

			elif up_down_all == 'dw':
				dfpiv = dfpiv[ (dfpiv[cols[0]] <= -similar_higher_than) | (dfpiv[cols[1]] <= -similar_higher_than) ]
				s_filtered = f'- different: abs diff >= {diff_cutoff} and downregulated: abs any LFC <= {-similar_higher_than}'

			else:
				dfpiv = dfpiv[ (dfpiv[cols[0]] <= -similar_higher_than) | (dfpiv[cols[0]] >= similar_higher_than) |
				               (dfpiv[cols[1]] <= -similar_higher_than) | (dfpiv[cols[1]] >= similar_higher_than)]

				s_filtered = f'- different: abs diff >= {diff_cutoff}'

		if dfpiv.empty:
			print(f"No genes are modulated using the filter {type_modulation} for '{pathway}'")
			return None, None, None

		if zlim is None:
			zlim = int(np.max(np.abs(dfpiv))+1)

		zmin = -zlim
		zmax = +zlim

		cols2 = cols + ['diff']
		df_vectors = self.df_to_plotly(dfpiv[cols2])

		'''self.df_vectors = df_vectors
		   self.dfpiv = dfpiv
		   self.zlim = zlim
		'''

		fig.add_trace(go.Heatmap(df_vectors, zmin=zmin, zmax=zmax,
								 colorscale='RdBu_r'))

		title = f"{self.s_deg_dap} heatmap for {pathway} {s_filtered}"
		title = title.strip()

		nrows = len(dfpiv)
		height = row_height * nrows + header_height

		fig.update_layout(
					autosize=True,
					title=title,
					width=width,
					height=height,
					plot_bgcolor=plot_bgcolor,
					xaxis_title="cases",
					yaxis_title=f"{self.s_deg_dap}s",
					showlegend=False,
					font=dict(
						family="Arial",
						size=14,
						color="Black"
					)
		)

		figname = title_replace(title)
		figname = os.path.join(self.root_figure, figname+'.html')
		if verbose: print(">>> HTML and png saved:", figname)

		try:
			fig.write_html(figname)
			fig.write_image(figname.replace('.html', '.png'))
		except:
			print(f"Error writing {figname}")

		return fig, dff, dfpiv


	def print_pathway_modulation_up_down(self, df3:pd.DataFrame, title:str, type_ptw_mod:str, case:str) -> (str, dict):
		dic = {}

		text = f"{title}\n"
		text += echo_print("-"*len(title))

		for i in range(len(df3)):
			row = df3.iloc[i]
			try:
				text += echo_print(f"{i+1}) {row.pathway} (FDR={row.fdr:.2e})")
			except:
				text += echo_print(f"{i+1}) {row.pathway} (no FDR)")

			text += echo_print(f"mod_pathway_index = {row.mod_pathway_index:.3f} ratio_up_dw = {row.ratio_up_dw:.3f}")
			text += echo_print(f"{self.s_deg_dap}s in pathways {row.n_mod_in_pathway} perc_degs_in_pathway = {row.perc_mod_in_pathway*100:.2f}%")
			text += echo_print("In pathway")

			if row.n_mod_up_in_pathway == 0:
				mod_up_in_pathway = []
				text += echo_print(f"\tUp ({row.n_mod_up_in_pathway}) = not found")
			else:
				try:
					if isinstance(row.mod_up_in_pathway, str):
						mod_up_in_pathway = eval(row.mod_up_in_pathway)
					else:
						mod_up_in_pathway = row.mod_up_in_pathway

					text += echo_print(f"\tUp ({row.n_mod_up_in_pathway}) = {','.join(mod_up_in_pathway)}")
				except:
					print("Warning/Error: parsing mod_up_in_pathway:", row.mod_up_in_pathway)
					pass

			if row.n_mod_dw_in_pathway == 0:
				mod_dw_in_pathway = []
				text += echo_print(f"\tDW ({row.n_mod_dw_in_pathway}) = not found")
			else:
				try:
					if isinstance(row.mod_dw_in_pathway, str):
						mod_dw_in_pathway = eval(row.mod_dw_in_pathway)
					else:
						mod_dw_in_pathway = row.mod_dw_in_pathway

					text += echo_print(f"\tDW ({row.n_mod_dw_in_pathway}) = {','.join(mod_dw_in_pathway)}")
				except:
					print("Warning/Error: parsing mod_dw_in_pathway:", row.mod_dw_in_pathway	 )
					pass

			dic[i] = {}
			dic2 = dic[i]

			dic2['case']		 = case
			dic2['type_ptw_mod'] = type_ptw_mod
			dic2['pathway']	  = row.pathway
			dic2['fdr']		  = row.fdr
			dic2['mod_pathway_index'] = row.mod_pathway_index
			dic2['ratio_up_dw']	   = row.ratio_up_dw

			dic2['n_mod_in_pathway']	= row.n_mod_in_pathway
			dic2['perc_mod_in_pathway'] = row.perc_mod_in_pathway
			dic2['n_mod_up_in_pathway'] = row.n_mod_up_in_pathway
			dic2['n_mod_dw_in_pathway'] = row.n_mod_dw_in_pathway

			dic2['mod_up_in_pathway'] = mod_dw_in_pathway
			dic2['mod_dw_in_pathway'] = mod_dw_in_pathway

			text += echo_print("\n")

		return text, dic

	def summary_pathway_modulation(self, force:bool=False, verbose:bool=False) -> (pd.DataFrame, str, pd.DataFrame):

		dff = self.calc_all_pathway_gene_indexes(force=force, verbose=verbose)
		self.dff = dff

		if dff is None or dff.empty:
			return None, ''

		text_all = ''
		df_list = []
		for case in self.case_list:
			text, dfa = self.to_text_pathway_modulation(dff, case=case)
			df_list.append(dfa)
			if text_all == '':
				text_all = text
			else:
				text_all += '\n\n' + text

		fname = f"summary_pathway_modulation_degs_{self.normalization}_geneset_{self.geneset_num}_{self.geneset_lib}.txt"
		write_txt(text_all, fname, self.root_ressum, verbose=verbose)

		df_sum_ptw = pd.concat(df_list)
		df_sum_ptw.index = np.arange(0, len(df_sum_ptw))

		return dff, text_all, df_sum_ptw

	def to_text_pathway_modulation(self, dff:pd.DataFrame, case:str) -> (str, pd.DataFrame):
		text = f"for case {case}\n"
		text += echo_print("-"*len(text))
		text += echo_print("")

		row = dff.iloc[0]
		text += echo_print(f"All {self.s_deg_dap}s calculated = {row.n_mod_in_pathway} where Up={row.n_mod_up_in_pathway} and Down={row.n_mod_dw_in_pathway}")
		text += echo_print("")

		title = "'Mean Up Modulated' pathways"
		type_ptw_mod = 'up'
		df2 = dff[dff.case == case]
		df3 = df2[df2.mod_pathway_index >= self.tolerance_pathway_index]
		df3 = df3.sort_values('mod_pathway_index', ascending=False)
		text2, dic_up = self.print_pathway_modulation_up_down(df3, title, type_ptw_mod, case)
		text += echo_print(text2)

		title = "'Not Modulated' pathways"
		type_ptw_mod = 'not'
		df3 = df2[ (df2.mod_pathway_index > -self.tolerance_pathway_index) & (df2.mod_pathway_index < self.tolerance_pathway_index) ]
		df3 = df3.sort_values('mod_pathway_index', ascending=False)
		text2, dic_not = self.print_pathway_modulation_up_down(df3, title, type_ptw_mod, case)
		text += echo_print(text2)

		title = "'Mean Down Modulated' pathways"
		type_ptw_mod = 'down'
		df3 = df2[df2.mod_pathway_index <= -self.tolerance_pathway_index]
		df3 = df3.sort_values('mod_pathway_index', ascending=True)
		text2, dic_dw = self.print_pathway_modulation_up_down(df3, title, type_ptw_mod, case)
		text += echo_print(text2)

		df_up  = pd.DataFrame(dic_up).T
		df_not = pd.DataFrame(dic_not).T
		df_dw  = pd.DataFrame(dic_dw).T

		dfa = pd.concat([df_up, df_not, df_dw])
		dfa.index = np.arange(0, len(dfa))

		return text, dfa


	def calc_all_pathway_gene_indexes(self, force:bool=False, verbose:bool=False) -> pd.DataFrame:

		if self.type_sat_ptw_index == 'discrete':
			s_saturation = 'none'
		elif self.type_sat_ptw_index == 'linear':
			s_saturation = 'none'
		elif self.type_sat_ptw_index == 'linear_sat':
			s_saturation = f'{self.saturation_lfc_index:.3f}'
		else:
			print("Please choose: discrete, linear, or linear_sat as type_sat_ptw_index")
			raise Exception("stop: calc_one_pathway_all_cases_gene_modulations()")


		fname = self.fname_pathway_index%(self.project, self.type_sat_ptw_index, s_saturation, self.normalization)
		fname = title_replace(fname)
		fullname = os.path.join(self.root_ressum, fname)

		if os.path.exists(fullname) and not force:
			dff = pdreadcsv(fname, self.root_ressum, verbose=verbose)
			return dff

		self.df_enr_all = self.list_all_df_erichments()

		df_list = []
		for case in self.case_list:
			dfa =  self.calc_pathway_gene_index_per_case(case, verbose=verbose)

			if dfa is None or dfa.empty:
				continue

			fname2 = self.fname_pathway_case_index%(self.project, case, self.normalization)
			ret = pdwritecsv(dfa, fname2, self.root_ressum, verbose=verbose)

			df_list.append(dfa)

		if df_list == []:
			print("Nothing was found.")
			return None

		dff = pd.concat(df_list)
		dff.index = np.arange(0, len(dff))

		ret = pdwritecsv(dff, fname, self.root_ressum, verbose=verbose)

		return dff

	def list_all_df_erichments(self, force:bool=False, verbose:bool=False) -> pd.DataFrame:

		fullname = os.path.join(self.root_result, self.fname_all_dfenr)

		if os.path.exists(fullname) and not force:
			dfall = pdreadcsv(self.fname_all_dfenr, self.root_result, verbose=verbose)
			return dfall

		df_list = []

		cols = ['pathway', 'pathway_id']
		for case in self.case_list:
			ret, _, _, _ = self.open_case(case, verbose=False)
			if not ret:
				continue

			try:
				dfa = self.df_enr[ cols]
			except:
				dfa = self.df_enr[ ['pathway'] ]

			df_list.append(dfa)

		dfall = pd.concat(df_list)
		dfall = dfall.drop_duplicates()
		dfall = dfall.sort_values('pathway')
		dfall.index = np.arange(0, len(dfall))

		pdwritecsv(dfall, self.fname_all_dfenr, self.root_result, verbose=verbose)
		return dfall


	def calc_one_pathway_all_cases_gene_modulations(self, pathway_and_id:list, force:bool=False, verbose:bool=False) -> pd.DataFrame:

		pathway, pathway_id = pathway_and_id

		fname = self.fname_one_pathway_gene_mods%(pathway, self.geneset_lib, self.normalization)
		fname = title_replace(fname)
		fullname = os.path.join(self.root_ressum, fname)

		if os.path.exists(fullname) and not force:
			dff = pdreadcsv(fname, self.root_ressum, verbose=verbose)
			return dff

		df_list = []
		for case in self.case_list:
			dfa =  self.calc_one_pathway_case_gene_modulations(pathway_and_id, case, verbose=verbose)
			if dfa is None or dfa.empty: continue
			df_list.append(dfa)

		if df_list == []:
			print(f"No gene modulations was found for {pathway}.")
			return None

		dff = pd.concat(df_list)
		dff = dff.sort_values(['case', 'symbol'])
		dff.index = np.arange(0, len(dff))

		ret = pdwritecsv(dff, fname, self.root_ressum, verbose=verbose)

		return dff


	def calc_pathway_gene_index_per_case(self, case:str, verbose:bool=False) -> pd.DataFrame:

		ret, _, _, dflfc = self.open_case(case, verbose=verbose)
		if not ret:
			return None

		icount = -1; dic = {}
		for i in range(len(self.df_enr_all)):
			row = self.df_enr_all.iloc[i]

			pathway_id = row.pathway_id
			pathway, genes_pathway = self.reactome_find_genes_in_pathway(pathway_id, _type='pathway_id')

			dfa = self.df_enr0[self.df_enr0.pathway_id == pathway_id]
			if dfa.empty:
				if verbose: print(f"No enrichment table for {case} - {pathway_id}: {pathway}")
				fdr  = None
				pval = None
			else:
				fdr  = self.df_enr0.iloc[0].fdr
				pval = self.df_enr0.iloc[0].pval


			if len(genes_pathway) == 0:
				print(f"No genes were found for '{pathway_id} - {pathway}'")
				continue

			if self.type_sat_ptw_index == 'discrete':
				''' only DEGs / DAPs '''
				dflfc_ptw = dflfc[dflfc.symbol.isin(genes_pathway)].copy()
			else:
				''' watch all genes/proteins '''
				dflfc_ptw = self.dflfc_ori[self.dflfc_ori.symbol.isin(genes_pathway)].copy()
				dflfc_ptw = dflfc_ptw[dflfc_ptw.abs_lfc >= self.min_lfc_modulation]

			if dflfc_ptw.empty:
				if verbose: print(f"No genes were found for '{pathway_id} - {pathway}' in LFC table for case {case}")
				continue

			dflfc_ptw.index = np.arange(0, len(dflfc_ptw))

			dflfc_ptw_up = dflfc_ptw[dflfc_ptw.lfc > 0]
			dflfc_ptw_dw = dflfc_ptw[dflfc_ptw.lfc < 0]

			mod_up_in_pathway = list(dflfc_ptw_up.symbol)
			mod_dw_in_pathway = list(dflfc_ptw_dw.symbol)

			if self.type_sat_ptw_index == 'discrete':
				mod_up = len(mod_up_in_pathway)
				mod_dw = len(mod_dw_in_pathway)

			elif self.type_sat_ptw_index == 'linear':
				mod_up = 0 if dflfc_ptw_up.empty else dflfc_ptw_up.lfc.sum()
				mod_dw = 0 if dflfc_ptw_dw.empty else dflfc_ptw_dw.lfc.sum()

			elif self.type_sat_ptw_index == 'linear_sat':
				if dflfc_ptw_up.empty:
					mod_up = 0
				else:
					mod_up = np.sum([x if x < self.saturation_lfc_index else self.saturation_lfc_index for x in dflfc_ptw_up.lfc])

				if dflfc_ptw_dw.empty:
					mod_dw = 0
				else:
					mod_dw = np.sum([-self.saturation_lfc_index if x < -self.saturation_lfc_index else x for x in dflfc_ptw_dw.lfc])

			mod_up += 1
			mod_dw = np.abs(mod_dw) + 1

			ratio = mod_up / mod_dw
			mod_pathway_gene_index = np.log2(ratio)

			'''
			if pathway == 'Platelet Aggregation (Plug Formation)':
													print(">>>", case, mod_up, mod_dw, ratio, mod_pathway_gene_index)
													print("up", dflfc_ptw_up.symbol, dflfc_ptw_up.lfc)
													print("dw", dflfc_ptw_dw.symbol, dflfc_ptw_dw.lfc, '\n\n')
			'''

			icount += 1
			dic[icount] = {}
			dic2 = dic[icount]

			dic2['case'] = case
			dic2['pathway_id'] = pathway_id
			dic2['pathway']	= pathway

			dic2['mod_pathway_index'] = mod_pathway_gene_index
			dic2['ratio_up_dw'] = ratio

			n_mod_in_pathway = len(mod_up_in_pathway) + len(mod_dw_in_pathway)

			N = len(genes_pathway)
			dic2['n_genes_annot_in_pathway'] = N

			dic2['n_mod_in_pathway']	= n_mod_in_pathway
			dic2['perc_mod_in_pathway'] = n_mod_in_pathway / N
			dic2['n_mod_up_in_pathway'] = len(mod_up_in_pathway)
			dic2['n_mod_dw_in_pathway'] = len(mod_dw_in_pathway)

			dic2['fdr'] = fdr
			dic2['pval'] = pval

			dic2['mod_up_in_pathway'] = mod_up_in_pathway
			dic2['mod_dw_in_pathway'] = mod_dw_in_pathway

			dic2['genes_annot_in_pathway'] = genes_pathway

		if len(dic) == 0:
			return None

		df = pd.DataFrame(dic).T
		df = df.sort_values("mod_pathway_index", ascending=False)
		df.index = np.arange(0, len(df))

		return df

	def calc_one_pathway_case_gene_modulations(self, pathway_and_id:List, case:str, verbose:bool=False) -> pd.DataFrame:

		if not self.open_case_simple(case, verbose=verbose):
			return None

		dflfc = self.dflfc_ori

		pathway2, pathway_id = pathway_and_id
		pathway, genes_pathway = self.reactome_find_genes_in_pathway(pathway_id, _type='pathway_id')

		if pathway.lower().strip() != pathway2.lower().strip():
			print(f"Diferent pathways '{pathway}' and '{pathway2}'")

		if len(genes_pathway) == 0:
			print(f"No genes were found for '{pathway_id} - {pathway}'")
			return None

		dflfc_ptw = dflfc[dflfc.symbol.isin(genes_pathway)].copy()
		if len(dflfc_ptw) == 0:
			print(f"No genes were found for '{pathway_id} - {pathway}' in LFC table")
			return None

		cols = ['symbol', 'lfc', 'abs_lfc', 'fdr']
		dflfc_ptw = dflfc_ptw[cols]
		dflfc_ptw.index = np.arange(0, len(dflfc_ptw))

		dflfc_ptw['pathway'] = pathway
		dflfc_ptw['case'] = case

		cols = ['case', 'pathway', 'symbol', 'lfc', 'abs_lfc', 'fdr']
		dflfc_ptw = dflfc_ptw[cols]

		return dflfc_ptw


	def degs_to_text_all_cases_summary(self, verbose:bool=False) -> str:
		now = datetime.now()

		s_date_time = now.strftime("%d/%m/%Y %H:%M:%S")

		text  = echo_print(f">>> Project {self.project}: {self.s_omics}")
		text += echo_print(f">>> Calculated in {s_date_time}")
		text += echo_print(f">>> geneset {self.geneset_num} - {self.geneset_lib}\n")

		icase = 0
		for case in self.case_list:
			icase += 1
			text += echo_print(f"{icase}) {case}\n")
			ret, _, _, _ = self.open_case(case, verbose=False)
			if not ret:
				continue

			text += echo_print(f"BCA's LFC {self.s_deg_dap} cutoffs: abs(lfc)={self.abs_lfc_cutoff:.3f}; fdr={self.fdr_lfc_cutoff:.3f}")
			text += echo_print("")

			text += echo_print(f"\tThere are a total of {self.n_degs} {self.s_deg_dap}s and {self.n_degs_ensembl} have ensembl_id")
			text += echo_print("")
			text += echo_print(f"\t\tAll {self.s_deg_dap}s Up ({self.n_degs_up}): {','.join(self.degs_up)}")
			text += echo_print("")
			text += echo_print(f"\t\tAll {self.s_deg_dap}s Dw ({self.n_degs_dw}): {','.join(self.degs_dw)}")

			text += echo_print("\n")
			text += echo_print(f"\tWith ensembl_id ({self.n_degs_ensembl})")
			text += echo_print("")
			text += echo_print(f"\t\t{self.s_deg_dap}s Up ({self.n_degs_up_ensembl}): {','.join(self.degs_up_ensembl)}")
			text += echo_print("")
			text += echo_print(f"\t\t{self.s_deg_dap}s Dw ({self.n_degs_dw_ensembl}): {','.join(self.degs_dw_ensembl)}")


			text += echo_print("\n")
			text += echo_print(f"\tWithout ensembl_id ({self.n_degs_not_ensembl})")
			text += echo_print("")
			text += echo_print(f"\t\t{self.s_deg_dap}s Up ({self.n_degs_up_not_ensembl}): {','.join(self.degs_up_not_ensembl)}")
			text += echo_print("")
			text += echo_print(f"\t\t{self.s_deg_dap}s Dw ({self.n_degs_dw_not_ensembl}): {','.join(self.degs_dw_not_ensembl)}")
			text += echo_print("\n")

			if self.df_enr is not None and not self.df_enr.empty:
				text += echo_print("")

				text += echo_print(f"\tBCA's pathway cutoffs: pval={self.pathway_pval_cutoff:.3f} fdr={self.pathway_fdr_cutoff:.3f} minimum number of genes={self.num_of_genes_cutoff}")
				text += echo_print(f"\tFound {self.n_pathways} (best={self.n_pathways_best}) pathways using '{self.geneset_lib}'")

				for iptw in range(len(self.pathway_list)):
					text += echo_print(f"\t\t{self.pathway_id_list[iptw]:18} FDR={self.pathway_fdr_list[iptw]:.2e} - {self.pathway_list[iptw]}")

				text += echo_print("")
				text += echo_print(f"\t{self.s_deg_dap}s found in enriched pathways:")
				text += echo_print(f"\t\t{self.n_degs_in_pathways} (best={self.n_degs_in_pathways_best}) in pathway {self.s_deg_dap}s out of {self.n_degs} {self.s_deg_dap}s")
				text += echo_print("")

				text += echo_print(f"\t\t{self.s_deg_dap}s in pathways - possibly only {self.s_deg_dap}s with ensembl_id are annotated in pathways")
				text += echo_print(f"\t\tUp and Dw {self.s_deg_dap}s in pathways:")
				text += echo_print("")
				text += echo_print(f"\t\t\tUp {self.n_degs_up_ensembl_in_pathways}: {','.join(self.degs_up_ensembl_in_pathways)}")
				text += echo_print("")
				text += echo_print(f"\t\t\tDw {self.n_degs_dw_ensembl_in_pathways}: {','.join(self.degs_dw_ensembl_in_pathways)}")
				text += echo_print("\n")

				text += echo_print(f"\t\tUp and Dw {self.s_deg_dap}s not in pathways:")
				text += echo_print(f"\t\t\tAll {self.s_deg_dap}s not in pathways {self.n_degs_not_in_pathways}")
				text += echo_print(f"\t\t\t{self.s_deg_dap}s with ensembl_id not in pathways {self.n_degs_ensembl_not_in_pathways}")
				text += echo_print("")
				text += echo_print(f"\t\t\tUp ensembl {self.n_degs_up_ensembl_not_in_pathways}: {','.join(self.degs_up_ensembl_not_in_pathways)}")
				text += echo_print("")
				text += echo_print(f"\t\t\tDw ensembl {self.n_degs_dw_ensembl_not_in_pathways}: {','.join(self.degs_dw_ensembl_not_in_pathways)}")
				text += echo_print("")

				if self.n_degs == self.n_degs_in_pathways + self.n_degs_not_in_pathways:
					# stri = "All right:"
					# text += echo_print(f"\t\t{stri} {self.n_degs} all {self.s_deg_dap}s == {self.n_degs_in_pathways} in + {self.n_degs_not_in_pathways:} not in pathways")
					pass
				else:
					stri = "There are problems:"
					text += echo_print(f"\t\t{stri} {self.n_degs} all {self.s_deg_dap}s != {self.n_degs_in_pathways} in + {self.n_degs_not_in_pathways:} not in pathways")

			else:
				text += echo_print("\nNo enrichment analysis was found.")

			text += echo_print("")

		fname = f"summary_all_cases_degs_geneset_{self.geneset_num}_{self.geneset_lib}.txt"
		write_txt(text, fname, self.root_ressum, verbose=verbose)

		return text


	def plot_cutoff_simulation_histograms(self, col:str='toi1_median', width:int=1100, height:int=700):

		lista=[]
		local_case_list = []
		for case in self.case_list:
			dfi = self.calc_enrichment_cutoff_params_and_ndxs_per_case_and_geneset_lib(case)
			if dfi is None or dfi.empty:
				continue
			median = np.median(dfi[col])
			mean = np.mean(dfi[col])
			std = np.std(dfi[col])

			stri = f'{median:.1f}/{mean:.1f} ({std:.1f})'
			lista.append(f"{case}<br>{stri}")
			local_case_list.append(case)

		_n_rows = int(np.ceil(len(local_case_list)/4))
		fig = make_subplots(rows=_n_rows, cols=4, subplot_titles=local_case_list)

		nrow=1
		ncol=0

		for case in local_case_list:
			dfi = self.calc_enrichment_cutoff_params_and_ndxs_per_case_and_geneset_lib(case)
			if dfi is None or dfi.empty:
				continue

			ncol += 1
			if ncol == 5:
				ncol = 1
				nrow += 1

			fig.add_trace(go.Histogram(x=dfi[col], name=case), row=nrow, col=ncol)

		fig.update_layout(title=f"{self.project} - frequency of {col} in pathways'",
						  width=width,
						  height=height*_n_rows,
						  xaxis_title=col,
						  yaxis_title="frequency",
						  showlegend=False)

		return fig


	def plot_all_degs_up_down_per_cutoffs(self, width:int=1100, height:int=400,
										  title:str=None, y_anchor:float=1.01, verbose:bool=False):

		dfsim = self.open_simulation_table()

		if title is None:
			title=f"{self.project} - Number of Up/Downregulated {self.s_deg_dap}s per LFC-FDR cutoffs"

		cases = self.case_list

		fig = make_subplots(rows=len(cases), cols=1, subplot_titles=cases)

		row = 0
		col = 1
		for case in cases:
			row+=1

			showlegend = row == 1

			df2 = dfsim[dfsim.case == case].copy()
			df2.index = np.arange(0, len(df2))

			fig.add_trace(go.Bar(x=df2.cutoff, y=df2.n_degs_up, marker={'color':'red'}, name='Up', showlegend=showlegend), row=row, col=col)
			fig.add_trace(go.Bar(x=df2.cutoff, y=df2.n_degs_dw, marker={'color':'blue'}, name='Down', showlegend=showlegend), row=row, col=col)


		yaxis_title = f"number of Up/Down {self.s_deg_dap}s"

		fig.update_layout(title=title,
						  width=width,
						  height=height*len(cases),
						  xaxis_title="",
						  xaxis2_title="LFC-FDR cutoffs",
						  yaxis_title=yaxis_title,
						  yaxis2_title=yaxis_title,
						  legend_title=f"{self.s_deg_dap}s regulation",
						  legend=dict( orientation="h",  yanchor="top", y=y_anchor, xanchor="right", x=1),
						  showlegend=True)

		figname = title_replace(title)
		figname = os.path.join(self.root_figure, figname+'.html')

		fig.write_html(figname)
		if verbose: print(">>> HTML and png saved:", figname)
		fig.write_image(figname.replace('.html', '.png'))

		return fig


	def calc_all_annoted_genes(self):
		if self.geneset_num != 0:
			print("Only defined for Reactome - geneset_num == 0")
			return False

		df_enr = self.merge_reactome(self.df_enr)

		annotated_genes = []
		for i in range(len(df_enr)):
			symbs = df_enr.iloc[i].genes_pathway
			symbs = eval(symbs) if isinstance(symbs, str) else list(symbs)
			annotated_genes += symbs

		self.annotated_genes = list(np.unique(annotated_genes))
		self.n_annotated_genes = len(self.annotated_genes)

		return self.n_annotated_genes > 0

	''' old: calc_best_cutoff_parameters_by_case_geneset '''
	def calc_enrichment_cutoff_params_and_ndxs_per_case_and_geneset_lib(self, case:str, force:bool=False,
																	   verbose:bool=False) -> pd.DataFrame:
		icount=-1

		fname = self.fname_enr_gene_stat%(case, self.geneset_lib, self.normalization)
		fullname = os.path.join(self.root_ressum, fname)

		if os.path.exists(fullname) and not force:
			dfi = pdreadcsv(fname, self.root_ressum)
			self.dfi = dfi
			return dfi

		# ret_lfc, degs, dfdegs = self.open_case(case, verbose=verbose)
		_, _, _, _ = self.open_case(case, verbose=verbose)
		if verbose: print(self.geneset_lib, case, self.normalization)

		files = [x for x in os.listdir(self.root_enrich) if '_pathway_pval_' in x \
				 and case in x and self.geneset_lib in x and self.normalization in x and not '~lock' in x]
		if files == []:
			if verbose: print("No files found in calc_enrichment_cutoff_params_and_ndxs_per_case_and_geneset_lib()")
			return None

		if verbose: print(f"For case {case} there are {len(files)} files")

		dic = {}
		icount = -1
		for fname_enr in files:
			df_enr = pdreadcsv(fname_enr, self.root_enrich)

			if df_enr is None:
				print(f"Could not read: '{fname_enr}'")
				continue

			if len(df_enr) < 3:
				if verbose: print(f"Warning: there is no sufficient data (#df_enr) in: '{fname_enr}'")
				continue

			'''
			name = 'enricher_Reactome_2022_medulloblastoma_microarray_for_WNT_x_ctrl_not_normalized_cutoff_lfc_0.950_fdr_0.200_pathway_pval_0.050_fdr_0.450_num_genes_3.tsv'
			np.array(name.split('_'))
			'''
			mat = fname_enr.split('_')
			try:
				abs_lfc_cutoff = float(mat[-11])
				fdr_lfc_cutoff = float(mat[-9])
				pathway_pval_cutoff = float(mat[-6])
				pathway_fdr_cutoff = float(mat[-4])
				num_of_genes_cutoff = int(mat[-1][:-4])
			except:
				print(f"Error: in {case} split={str(mat)}")
				return None

			self.df_enr_reactome = None
			df_enr = self.merge_reactome(df_enr)

			toi1_list, toi2_list, toi3_list, toi4_list = [], [], [], []
			all_genes, n_degs_in_pathways_lista, all_genes_annotatted_in_pathway = [], [], []
			for i in range(len(df_enr)):
				row		 = df_enr.iloc[i]
				fdr_pathway = row.fdr
				pathway	 = row.pathway
				pathway_id  = row.pathway_id

				genes = row.genes if isinstance(row.genes, list) else eval(row.genes)
				n_degs_in_pathways = len(genes)

				genes_annotatted_in_pathway = row.genes_pathway if isinstance(row.genes_pathway, list) else eval(row.genes_pathway)

				n_degs_in_pathways_lista.append(n_degs_in_pathways)
				all_genes += list(genes)
				all_genes_annotatted_in_pathway += list(genes_annotatted_in_pathway)

				n_genes_annotatted_in_pathway = row.ngenes_pathway
				ratio = n_degs_in_pathways / n_genes_annotatted_in_pathway

				toi1 = np.sqrt(-np.log10(fdr_pathway) * ratio)
				toi2 = np.sqrt(-np.log10(fdr_lfc_cutoff) * -np.log10(fdr_pathway))
				toi3 = (-np.log10(fdr_lfc_cutoff) * -np.log10(fdr_pathway) * ratio) ** 1/3.
				toi4 = (abs_lfc_cutoff * -np.log10(fdr_lfc_cutoff) * -np.log10(fdr_pathway) * ratio) ** 1/4.

				toi1_list.append(toi1)
				toi2_list.append(toi2)
				toi3_list.append(toi3)
				toi4_list.append(toi4)

			n_pathways = len(df_enr)
			toi1_mean = np.mean(toi1_list)
			toi1_median = np.median(toi1_list)
			toi1_std = np.std(toi1_list)

			toi2_mean = np.mean(toi2_list)
			toi2_median = np.median(toi2_list)
			toi2_std = np.std(toi2_list)

			toi3_mean = np.mean(toi3_list)
			toi3_median = np.median(toi3_list)
			toi3_std = np.std(toi3_list)

			toi4_mean = np.mean(toi4_list)
			toi4_median = np.median(toi4_list)
			toi4_std = np.std(toi4_list)

			icount += 1
			dic[icount] = {}
			dic2 = dic[icount]

			all_genes = list(np.unique(all_genes))
			n_degs_in_pathways = len(all_genes)
			all_genes_annotatted_in_pathway = list(np.unique(all_genes_annotatted_in_pathway))

			dic2['case'] = case
			dic2['cutoff'] = f'{abs_lfc_cutoff:.3f} - {fdr_lfc_cutoff:.3f}'
			dic2['abs_lfc_cutoff'] = abs_lfc_cutoff
			dic2['fdr_lfc_cutoff'] = fdr_lfc_cutoff
			dic2['pathway_pval_cutoff'] = pathway_pval_cutoff
			dic2['pathway_fdr_cutoff'] = pathway_fdr_cutoff
			dic2['num_of_genes_cutoff'] = num_of_genes_cutoff

			dic2['n_pathways'] = n_pathways
			dic2['all_genes_annotatted_in_pathway'] = len(all_genes_annotatted_in_pathway)
			dic2['n_degs_in_pathways'] = n_degs_in_pathways

			dic2['n_degs_in_pathways_mean']   = np.mean(n_degs_in_pathways_lista)
			dic2['n_degs_in_pathways_median'] = np.median(n_degs_in_pathways_lista)
			dic2['n_degs_in_pathways_std']	= np.std(n_degs_in_pathways_lista)

			dic2['toi1_mean'] = toi1_mean
			dic2['toi1_median'] = toi1_median
			dic2['toi1_std'] = toi1_std

			dic2['toi2_mean'] = toi2_mean
			dic2['toi2_median'] = toi2_median
			dic2['toi2_std'] = toi2_std

			dic2['toi3_mean'] = toi3_mean
			dic2['toi3_median'] = toi3_median
			dic2['toi3_std'] = toi3_std

			dic2['toi4_mean'] = toi4_mean
			dic2['toi4_median'] = toi4_median
			dic2['toi4_std'] = toi4_std

			dic2['all_genes'] = all_genes
			dic2['all_genes_annotatted_in_pathway'] = all_genes_annotatted_in_pathway

		dfi = pd.DataFrame(dic).T
		dfi = dfi.sort_values(['case', 'pathway_fdr_cutoff', 'fdr_lfc_cutoff', 'abs_lfc_cutoff',], ascending=[True, True, True, False])

		ret = pdwritecsv(dfi, fname, self.root_ressum, verbose=verbose)
		self.dfi = dfi

		return dfi


	def best_cutoff_quantiles(self, case:str, selected_col:str='toi1_median', med_max_ptw:str='median',
							  force:bool=False, verbose:bool=False) -> dict:
		''' best_cutoff_quantiles calculates the quantiles related to each toi = selected_toi_col '''

		self.med_max_ptw = med_max_ptw
		dfi = self.calc_enrichment_cutoff_params_and_ndxs_per_case_and_geneset_lib(case, force=False, verbose=verbose)

		''' impossible to calculate quantiles with less than 3 values '''
		if dfi is None or len(dfi) < 3:
			return None

		quantile_vals = np.quantile(dfi[selected_col], self.quantile_list)

		dic = {}
		for i in range(len(self.quantile_list)):
			quantile = self.quantile_list[i]

			if quantile < 0.95:
				lim_inf = quantile_vals[i]
				lim_sup = quantile_vals[i+1]

				df2 = dfi[ (dfi[selected_col] > lim_inf) & (dfi[selected_col] <= lim_sup) ].copy()
			else:
				lim_inf = quantile_vals[i]
				lim_sup = None
				df2 = dfi[ (dfi[selected_col] > lim_inf) ].copy()

			if df2.empty:
				if verbose: print(f"Warning: empty quantile: {quantile} and case {self.case} - {self.normalization}")

				mat = ['lim_inf', 'lim_sup', 'abs_lfc_cutoff', 'fdr_lfc_cutoff', 'pathway_pval_cutoff',
					   'pathway_fdr_cutoff', 'num_of_genes_cutoff', 'n_pathways', 'n_degs_in_pathways',
					   'n_degs_in_pathways_mean', 'n_degs_in_pathways_median ', 'n_degs_in_pathways_std',
					   'toi1_mean', 'toi1_median', 'toi1_std',
					   'toi2_mean', 'toi2_median', 'toi2_std',
					   'toi3_mean', 'toi3_median', 'toi3_std',
					   'toi4_mean', 'toi4_median', 'toi4_std'  ]

				dic[quantile] = [None] * len(mat)
			else:

				if med_max_ptw == 'pathway':
					df2 = df2.sort_values(['n_pathways', 'n_degs_in_pathways'], ascending=[False, False])
					df2.index = np.arange(0, len(df2))
					row = df2.iloc[0]
				else:
					df2 = df2.sort_values(selected_col, ascending=False)
					if med_max_ptw == 'median':
						med_max = df2[selected_col].median()
					else:
						''' maximum '''
						med_max = df2[selected_col].max()

					'''------ filter med_max, if there are more than 1 ro2 -----'''
					df3 = df2[ (df2[selected_col] == med_max)].copy()
					df3 = df3.sort_values(['n_pathways', 'n_degs_in_pathways'], ascending=[False, False])
					df3.index = np.arange(0, len(df3))
					row = df2.iloc[0]

				dic[quantile] = [lim_inf, lim_sup, row.abs_lfc_cutoff, row.fdr_lfc_cutoff, row.pathway_pval_cutoff,
								 row.pathway_fdr_cutoff, row.num_of_genes_cutoff, row.n_pathways, row.n_degs_in_pathways,
								 row.n_degs_in_pathways_mean, row.n_degs_in_pathways_median, row.n_degs_in_pathways_std,
								 row.toi1_mean, row.toi1_median, row.toi1_std,
								 row.toi2_mean, row.toi2_median, row.toi2_std,
								 row.toi3_mean, row.toi3_median, row.toi3_std,
								 row.toi4_mean, row.toi4_median, row.toi4_std  ]

				if verbose:
					print(f"For case {case} {row.n_degs_in_pathways_median:.1f} median genes for len={len(df2)}")
					print(f"\tthe best cutoff LFC = {row.abs_lfc_cutoff:.3f} and FDR = {row.fdr_lfc_cutoff:.3f} and Pathway FDR = {row.pathway_fdr_cutoff:.3f}")

		return dic

	def test_build_all_cutoffs_table(self, case:str='WNT',  selected_toi_col:str='toi4_median',
									 med_max_ptw:str='median', lista_fdr:List=[0.3, 0.35, 0.4],
									 force:bool=True, verbose:bool=False):

		dfcut = self.build_all_cutoffs_table(selected_toi_col)

		df2 = dfcut[(dfcut.case == case) & (dfcut.med_max_ptw == med_max_ptw) & (dfcut.fdr_lfc_cutoff.isin(lista_fdr))]
		df2 = df2.sort_values(selected_toi_col, ascending=False)
		return df2

	def build_all_cutoffs_table(self, selected_toi_col:str='toi4_median',
								force:bool=False, verbose:bool=False) -> pd.DataFrame:

		fname = self.fname_cutoff_table%(selected_toi_col, self.geneset_num, self.geneset_lib, self.normalization)
		fullname = os.path.join(self.root_ressum, fname)

		if os.path.exists(fullname) and not force:
			 return pdreadcsv(fname, self.root_ressum, verbose=verbose)

		dic={}; icount=-1
		for case in self.case_list:

			'''
			 dictionary with following values:

			 quantile, abs_lfc_cutoff, fdr_lfc_cutoff, pathway_pval_cutoff, pathway_fdr_cutoff,
			 num_of_genes_cutoff, n_pathways, n_degs_in_pathways, n_degs_in_pathways_mean, n_degs_in_pathways_median,
			 n_degs_in_pathways_std, toi1_mean, toi1_median, toi1_std, 2, 3, 4
			'''

			for med_max_ptw in ['median', 'maximum', 'pathway']:
				print(f">>> case {case} column {selected_toi_col} - {med_max_ptw}")
				self.med_max_ptw = med_max_ptw

				''' best_cutoff_quantiles calculates the quantiles related to each toi = selected_toi_col '''
				dic_quant = self.best_cutoff_quantiles(case, selected_toi_col, med_max_ptw,
													   force=force, verbose=verbose)

				if dic_quant is None or len(dic_quant) == 0:
					continue

				i=-1
				for quantile, (lim_inf, lim_sup, abs_lfc_cutoff, fdr_lfc_cutoff,
							   pathway_pval_cutoff, pathway_fdr_cutoff,
							   num_of_genes_cutoff, n_pathways, n_degs_in_pathways,
							   n_degs_in_pathways_mean, n_degs_in_pathways_median, n_degs_in_pathways_std,
							   toi1_mean, toi1_median, toi1_std,
							   toi2_mean, toi2_median, toi2_std,
							   toi3_mean, toi3_median, toi3_std,
							   toi4_mean, toi4_median, toi4_std) in dic_quant.items():

					icount += 1
					i += 1
					dic[icount] = {}
					dic2 = dic[icount]

					dic2['case'] = case
					dic2['geneset_num'] = self.geneset_num
					dic2['normalization'] = self.normalization
					dic2['med_max_ptw'] = med_max_ptw
					dic2['parameter'] = selected_toi_col
					dic2['quantile'] = quantile
					dic2['quantile_val_inf'] = lim_inf
					dic2['quantile_val_sup'] = lim_sup

					dic2['abs_lfc_cutoff'] = abs_lfc_cutoff
					dic2['fdr_lfc_cutoff'] = fdr_lfc_cutoff
					dic2['pathway_pval_cutoff'] = pathway_pval_cutoff
					dic2['pathway_fdr_cutoff']  = pathway_fdr_cutoff
					dic2['num_of_genes_cutoff'] = num_of_genes_cutoff
					dic2['n_pathways'] = n_pathways
					dic2['n_degs_in_pathways'] = n_degs_in_pathways
					dic2['n_degs_in_pathways_mean'] = n_degs_in_pathways_mean
					dic2['n_degs_in_pathways_median'] = n_degs_in_pathways_median
					dic2['n_degs_in_pathways_std'] = n_degs_in_pathways_std

					dic2['toi1_mean'] = toi1_mean
					dic2['toi1_median'] = toi1_median
					dic2['toi1_std'] = toi1_std

					dic2['toi2_mean'] = toi2_mean
					dic2['toi2_median'] = toi2_median
					dic2['toi2_std'] = toi2_std

					dic2['toi3_mean'] = toi3_mean
					dic2['toi3_median'] = toi3_median
					dic2['toi3_std'] = toi3_std

					dic2['toi4_mean'] = toi4_mean
					dic2['toi4_median'] = toi4_median
					dic2['toi4_std'] = toi4_std

		dfcut = pd.DataFrame(dic).T
		ret = pdwritecsv(dfcut, fname, self.root_ressum, verbose=verbose)

		return dfcut if ret else None

	def calc_best_cutoffs_params(self, selected_toi_col:str='toi4_median', n_best_sample:int=3,
								 save_config:bool=False, verbose:bool=False) -> pd.DataFrame:

		fname = self.fname_cutoff_table%(selected_toi_col, self.geneset_num, self.geneset_lib, self.normalization)
		fullname = os.path.join(self.root_ressum, fname)

		if not os.path.exists(fullname):
			print(f"Could not find: '{fullname}'")
			return False

		dfcut = pdreadcsv(fname, self.root_ressum, verbose=verbose)

		if dfcut is None or dfcut.empty:
			print(f"Error: could not read: '{fullname}'")
			return False

		df_all_fdr = self.open_all_fdr_lfc_correlation()

		df_list = []

		for case in self.case_list:
			df_fdr = df_all_fdr[df_all_fdr.case == case]
			if df_fdr.empty:
				print(f"Error: no correlation was calculated for case '{case}'")
				raise Exception('stop: plot_degs_in_pathways_vs_toi_per_case()')

			for med_max_ptw in ['median', 'maximum', 'pathway']:

				df2 = dfcut[(dfcut.case == case) & (dfcut.med_max_ptw == med_max_ptw) &
							(dfcut.fdr_lfc_cutoff.isin(df_fdr.fdr))].copy()

				if df2.empty: continue

				if med_max_ptw == 'pathway':
					df2 = df2.sort_values(['n_pathways', 'n_degs_in_pathways'], ascending=[False, False])
				else:
					df2 = df2.sort_values(selected_toi_col, ascending=False)

				df2.index = np.arange(0, len(df2))

				if len(df2) >= n_best_sample:
					dfa = pd.DataFrame(df2.iloc[n_best_sample-1]).T
				else:
					dfa = pd.DataFrame(df2.iloc[len(df2)-1]).T

				df_list.append(dfa)

		if df_list == []:
			dfconfig = None
		else:
			dfconfig = pd.concat(df_list)
			dfconfig.index = np.arange(0, len(dfconfig))

			if save_config:
				self.cfg.save_best_ptw_cutoff(dfconfig, verbose=True)
				print(f"Chosen n_best_sample: {n_best_sample}")

		return dfconfig

	def display_best_cutoff_params(self, npoints:int=10, selected_toi_col:str='toi4_median', 
								   med_max_ptw:str='median') -> pd.DataFrame:
		
		dic = {}; icount=-1
		for ipoint in range(1,npoints+1):
			dfconfig = self.calc_best_cutoffs_params(selected_toi_col=selected_toi_col, n_best_sample=ipoint, save_config=False, verbose=False)
			dfconfig = dfconfig[dfconfig.med_max_ptw == med_max_ptw]

			for j in range(len(dfconfig)):
				row = dfconfig.iloc[j]
				
				icount += 1
				dic[icount] = {}
				dic2 = dic[icount]
				dic2['case'] = row.case
				dic2['ipoint'] = ipoint
					
				dic2[f'n_ptws'] = row.n_pathways
				dic2[f'n_degs'] = row.n_degs_in_pathways

				dic2[f'fdr_lfc'] = row.fdr_lfc_cutoff
				dic2[f'lfc_cut'] = row.abs_lfc_cutoff
				dic2[f'fdr_ptw'] = row.pathway_fdr_cutoff

				dic2[f'toi4_median'] = row.toi4_median
				dic2[f'toi3_median'] = row.toi3_median
				dic2[f'toi2_median'] = row.toi2_median
				dic2[f'toi1_median'] = row.toi1_median

		dfpoints = pd.DataFrame(dic).T
		dfpoints = dfpoints.sort_values(['case', 'ipoint'])
		dfpoints.index = np.arange(0, len(dfpoints))

		return dfpoints


	def calc_multiple_best_cutoffs_params(self, selected_toi_col:str='toi4_median', n_best_sample_list:List=[],
								 save_config:bool=False, verbose:bool=False) -> pd.DataFrame:

		if len(n_best_sample_list) != len(self.case_list):
			print("Error: n_best_sample_list is a list of same dimension of case_list.")
			return pd.DataFrame()

		fname = self.fname_cutoff_table%(selected_toi_col, self.geneset_num, self.geneset_lib, self.normalization)
		fullname = os.path.join(self.root_ressum, fname)

		if not os.path.exists(fullname):
			print(f"Could not find: '{fullname}'")
			return False

		dfcut = pdreadcsv(fname, self.root_ressum, verbose=verbose)

		if dfcut is None or dfcut.empty:
			print(f"Error: could not read: '{fullname}'")
			return False

		df_all_fdr = self.open_all_fdr_lfc_correlation()

		df_list = []

		for i in range(len(self.case_list)):
			case = self.case_list[i]
			n_best_sample = n_best_sample_list[i]

			df_fdr = df_all_fdr[df_all_fdr.case == case]
			if df_fdr.empty:
				print(f"Error: no correlation was calculated for case '{case}'")
				raise Exception('stop: plot_degs_in_pathways_vs_toi_per_case()')

			for med_max_ptw in ['median', 'maximum', 'pathway']:

				df2 = dfcut[(dfcut.case == case) & (dfcut.med_max_ptw == med_max_ptw) &
							(dfcut.fdr_lfc_cutoff.isin(df_fdr.fdr))]

				if df2.empty: continue

				if med_max_ptw == 'pathway':
					df2 = df2.sort_values(['n_pathways', 'n_degs_in_pathways'], ascending=[False, False])
				else:
					df2 = df2.sort_values(selected_toi_col, ascending=False)
					''' get the first 3 (n_best_sample) rows
						there are cases that changing a little abs_lfc_cutoff or fdr_lfc_cutoff
						the number of n_pathways increases!
					'''
					df2 = df2.iloc[:n_best_sample]
					df2 = df2.sort_values(['n_pathways', 'n_degs_in_pathways'], ascending=[False, False])

				dfa = pd.DataFrame(df2.iloc[0]).T
				df_list.append(dfa)

		if df_list == []:
			dfconfig = None
		else:
			dfconfig = pd.concat(df_list)
			dfconfig.index = np.arange(0, len(dfconfig))

			if save_config:
				self.cfg.save_best_ptw_cutoff(dfconfig, verbose=True)
				print(f"Chosen n_best_sample: {n_best_sample}")

		return dfconfig


	def plot_genes_and_pathways_frequecies_per_cases(self, selected_toi_col:str='toi4_median',
													 med_max_ptw:str='median', width:int=800,
													 height:int=400, verbose:bool=False) -> List:

		dfcut = self.build_all_cutoffs_table(selected_toi_col, force=False, verbose=verbose)

		dfcut = dfcut.sort_values(['case','quantile'])
		dfcut.index = np.arange(0, len(dfcut))
		self.dfcut = dfcut

		colors = plotly_my_colors[0:len(self.quantile_list)]

		if dfcut is None or dfcut.empty:
			return []

		fig_list = []
		for _plot in ['genes', 'pathways']:

			fig = go.Figure()

			for i in range(len(self.quantile_list)):
				quantile = self.quantile_list[i]
				name=f'{quantile}'
				color = colors[i]

				df2 = dfcut[ (dfcut['quantile'] == quantile) & (dfcut.med_max_ptw == med_max_ptw)].copy()
				df2.index = np.arange(0, len(df2))

				if _plot == 'genes':
					fig.add_trace(go.Bar(x=df2.case, y=df2.n_degs_in_pathways, marker_color=color, name=name)) # marker_color=color,
					plot_name = f"'Best' number of {self.s_deg_dap}s in pathways"
					yaxis_title = f"# {self.s_deg_dap}s in pathways"
				else:
					fig.add_trace(go.Bar(x=df2.case, y=df2.n_pathways, marker_color=color, name=name))
					plot_name = f"'Best' number of enriched pathways"
					yaxis_title = "# of pathways"

			title=f"{plot_name} for {selected_toi_col}  {med_max_ptw} values in quantiles"
			fig.update_layout(title=title,
							  width=width,
							  height=height,
							  xaxis_title='cases x quantiles',
							  yaxis_title=yaxis_title,
							  legend_title="Quantiles",
							  showlegend=True)

			figname = title_replace(title)
			figname = os.path.join(self.root_figure, figname+'.html')

			fig.write_html(figname)
			if verbose: print(">>> HTML and png saved:", figname)
			fig.write_image(figname.replace('.html', '.png'))

			fig_list.append(fig)

		return fig_list

	def open_all_fdr_lfc_correlation(self, verbose:bool=False):
		fname = self.fname_all_fdr_lfc_correlation%(self.abs_lfc_cutoff_inf)
		fullname = os.path.join(self.root_config, fname)

		if os.path.exists(fullname):
			df_all_fdr = pdreadcsv(fname, self.root_config, verbose=verbose)
		else:
			print(f"File all_fdr_lfc_correlation not found: {fullname}")
			print("Please run: 'pubmed_taubate_new03_up_down_simulation'")
			df_all_fdr = None

		return df_all_fdr


	def open_fdr_lfc_correlation(self, case:str, abs_lfc_cutoff_inf:float=None, verbose:bool=False) -> pd.DataFrame:

		if abs_lfc_cutoff_inf is None:
			abs_lfc_cutoff_inf = self.abs_lfc_cutoff_inf

		fname = self.fname_dic_fdr_lfc_correlation%(case, abs_lfc_cutoff_inf)
		fullname = os.path.join(self.root_config, fname)

		if os.path.exists(fullname):
			dic = loaddic(fname, self.root_config, verbose=verbose)
		else:
			dic = None

		if dic is None or len(dic) == 0:
			print(f"Could not find {fullname}, thus df_fdr is empty")
			df_fdr = pd.DataFrame()
		else:
			df_fdr = dic['df_fdr']

		return df_fdr

	def calc_all_LFC_FDR_cutoffs(self, cols2:List=['n_degs', 'abs_lfc_cutoff'],
								 corr_cutoff:float=-0.90, nregs_fdr:int=5, method:str='spearman',
								 force:bool=False, verbose:bool=False):

		self.df_all_fdr = None

		fname = self.fname_all_fdr_lfc_correlation%(self.abs_lfc_cutoff_inf)
		fullname = os.path.join(self.root_config, fname)

		if os.path.exists(fullname) and not force:
			self.df_all_fdr = pdreadcsv(fname, self.root_config, verbose=verbose)
			return self.df_all_fdr


		df_list = []
		for case in self.case_list:

			ret, dic_return = self.calc_nDEG_curve_per_LFC_FDR(case=case, cols2=cols2,
															   corr_cutoff=corr_cutoff, nregs_fdr=nregs_fdr,
															   method=method, force=force, verbose=verbose)

			if not ret:
				continue

			df_fdr = dic_return['df_fdr']
			df_list.append(df_fdr)


		if df_list == []:
			df_all_fdr = None
		else:
			df_all_fdr = pd.concat(df_list)
			self.df_all_fdr = df_all_fdr
			df_all_fdr.index = np.arange(0, len(df_all_fdr))
			ret = pdwritecsv(df_all_fdr, fname, self.root_config, verbose=verbose)

		return df_all_fdr


	def plot_all_LFC_FDR_cutoffs(self, width:int=1100, height:int=700, title:str=None,
								 cols2:List=['n_degs', 'abs_lfc_cutoff'],
								 corr_cutoff:float=-0.90, nregs_fdr:int=5, method:str='spearman',
								 verbose:bool=False) -> dict:

		dic_fig = {}
		for case in self.case_list:

			ret, dic_fig_return, _ = self.plot_nDEG_curve_per_LFC_FDR(case,
										  width=width, height=height, title=title, cols2=cols2,
										  corr_cutoff=corr_cutoff, nregs_fdr=nregs_fdr, method=method, verbose=verbose)

			if not ret:
				continue

			dic_fig[case] = {}
			dic2 = dic_fig[case]

			for key, fig in dic_fig_return.items():
				dic2[key] = fig

		return dic_fig


	'''
		calc_and_plot_nDEG_curve_per_LFC_FDR was splited in
			1) calc_nDEG_curve_per_LFC_FDR
				   return ret, dic

			2) plot_nDEG_curve_per_LFC_FDR
				   return ret, dic_fig, df_fdr
	'''

	def calc_nDEG_curve_per_LFC_FDR(self, case:str,
									cols2:List=['n_degs', 'abs_lfc_cutoff'],
									corr_cutoff:float=-0.90, nregs_fdr:int=5,
									method:str='spearman', force:bool=False, verbose:bool=False) -> (bool, dict):
		'''
			calc_nDEG_curve_per_LFC_FDR
				calculates fdr cutoff when Pearson's correlation (e.g.) <= -.90
				correlation must always decrease
				no more than ireg_fdr, exception repeated correlations

				the DAP x LFC x fdr lanscape looks very irregular
				at least 5 regs are necessary aftare found a corr <= corr_cutoff

				somentime the curves jumps! other, the look the same (superposition)
				this ocurrs in ou first studies os PBMC proteomics COVID-19 and microarray Medulloblastoma

				saves nregs_fdr values
				dic contains df_fdr, name_list, fdrs --> necessary do draw the plot

				return ret, dic

			calc per case, fdr, lfc, correlations
		'''
		dfsim = self.open_simulation_table()

		fname = self.fname_dic_fdr_lfc_correlation%(case, self.abs_lfc_cutoff_inf)
		fullname = os.path.join(self.root_config, fname)

		if os.path.exists(fullname) and not force:
			dic = loaddic(fname, self.root_config, verbose=verbose)
			return True, dic

		dfsim = dfsim[dfsim.case == case].copy()

		fdrs = dfsim.fdr_lfc_cutoff.unique()

		dfsim = dfsim.sort_values(['fdr_lfc_cutoff', 'abs_lfc_cutoff'], ascending=[True, False])

		name_list, fdr_list = [], []

		dic = {}; icount=-1; found = False; ireg_fdr=0; corr_previous=0
		for fdr in fdrs:
			dfsim2 = dfsim[ (dfsim.fdr_lfc_cutoff == fdr) & (dfsim.abs_lfc_cutoff >= self.abs_lfc_cutoff_inf) ]
			corr = dfsim2[cols2].corr(method=method).iloc[0,1]

			n_degs_min = dfsim2.n_degs.min()
			dfsim2_min = dfsim2[dfsim2.n_degs == n_degs_min]

			n_degs_max = dfsim2.n_degs.max()
			dfsim2_max = dfsim2[dfsim2.n_degs == n_degs_max]

			''' correlation must be negative '''
			if pd.isnull(corr) or corr > corr_cutoff:
				name = f"fdr={fdr:.2f} not found corr."
				name_list.append(name)

				icount += 1
				dic[icount] = {}
				dic2 = dic[icount]

				# print(">>> NOT", icount, case, fdr, n_degs_min, n_degs_max)


				dic2['case'] = case
				dic2['first'] = False
				dic2['chosen'] = False
				dic2['fdr'] = fdr
				dic2['corr'] = corr

				dic2['label'] = name
				dic2['method'] = method

				dic2['n_degs_min'] = n_degs_min
				dic2['n_degs_max'] = n_degs_max

				dic2['abs_lfc_cutoff_inf'] = self.abs_lfc_cutoff_inf

				dic2['degs_min'] = dfsim2_min.iloc[0].degs
				dic2['degs_max'] = dfsim2_max.iloc[0].degs

			else:
				''' correlation is negativa, usually cor <= -0.90 '''
				if not found and corr <= corr_cutoff:
					found = True

					ireg_fdr = 1
					icount += 1
					dic[icount] = {}
					dic2 = dic[icount]

					dic2['case'] = case
					dic2['first'] = True
					dic2['chosen'] = True
					dic2['fdr'] = fdr
					dic2['corr'] = corr
					corr_previous = corr

					name = f"fdr={fdr:.2f} corr={corr:.3f} ***"
					dic2['label'] = name
					dic2['method'] = method

					dic2['n_degs_min'] = n_degs_min
					dic2['n_degs_max'] = n_degs_max

					dic2['abs_lfc_cutoff_inf'] = self.abs_lfc_cutoff_inf

					dic2['degs_min'] = dfsim2_min.iloc[0].degs
					dic2['degs_max'] = dfsim2_max.iloc[0].degs

					if verbose: print(">>> ***", icount, case, fdr, n_degs_min, n_degs_max)
				else:
					icount += 1

					''' correlation must improve '''
					if corr < corr_previous or corr <= -0.99:
						corr_previous = corr
						ireg_fdr += 1

					dic[icount] = {}
					dic2 = dic[icount]

					dic2['case'] = case
					dic2['first'] = False
					dic2['chosen'] = True
					dic2['fdr'] = fdr
					dic2['corr'] = corr

					name = f"fdr={fdr:.2f} corr={corr:.3f}"
					dic2['label'] = name
					dic2['method'] = method

					dic2['n_degs_min'] = n_degs_min
					dic2['n_degs_max'] = n_degs_max

					dic2['abs_lfc_cutoff_inf'] = self.abs_lfc_cutoff_inf

					dic2['degs_min'] = dfsim2_min.iloc[0].degs
					dic2['degs_max'] = dfsim2_max.iloc[0].degs

					if verbose: print(">>> NEXT", icount, case, fdr, n_degs_min, n_degs_max)

			name_list.append(name)
			fdr_list.append(fdr)

			''' no more than ireg_fdr, exception repeated correlations'''
			if ireg_fdr >= nregs_fdr:
				break

		if len(dic) == 0:
			print("Nothing found in calc_nDEG_curve_per_LFC_FDR()")
			return False, None

		df_fdr = pd.DataFrame(dic).T

		dic_return = {}
		dic_return['df_fdr']	= df_fdr
		dic_return['name_list'] = name_list
		dic_return['fdrs']	  = fdr_list

		ret = dumpdic(dic_return, fname, self.root_config, verbose=verbose)

		return ret, dic_return

	def plot_all_dfsim(self, dfsim:pd.DataFrame, case:str, fdr_list:List,
					   width:int=1100, height:int=700, title:str=None, verbose:bool=False):

		dfsim = dfsim[dfsim.case == case]

		if title is None:
			title = f"{self.s_deg_dap}s curve per fdr x lfc for {case} - all FDR's"

		title2 = title

		yaxis_title = f"num of {self.s_deg_dap}s"
		yaxis_title2 = yaxis_title
		xaxis_title = "abs_LFC cutoff"
		legend_title = 'FDR cutoff'

		dic_fig = {}
		for which in ['deg', 'up', 'down']:
			''' many figures: one case, deg, up and down '''
			fig = go.Figure()

			for fdr in fdr_list:

				dfsim2 = dfsim[dfsim.fdr_lfc_cutoff == fdr]
				name = f'{fdr:.2f}'

				if which == 'deg':
					col = 'n_degs'
					yaxis_title2 = yaxis_title
					title2 = title
				else:
					col = 'n_degs_up' if which == 'up' else 'n_degs_dw'
					stri = 'Up' if which == 'up' else 'Down'
					yaxis_title2 = yaxis_title + ' ' + stri
					title2 = title.replace('s curve', f's {stri} curve')

				legend_title = 'FDR cutoff'

				fig.add_trace(go.Scatter(x=dfsim2.abs_lfc_cutoff, y=dfsim2[col], name=name))


			fig.update_layout(
						autosize=True,
						title=title2,
						width=width,
						height=height,
						xaxis_title=xaxis_title,
						yaxis_title=yaxis_title2,
						legend_title=legend_title,
						showlegend=True,
						font=dict(
							family="Arial",
							size=14,
							color="Black"
						)
			)
			dic_fig[which] = fig

			figname = title_replace(title2)
			figname = os.path.join(self.root_figure, figname+'.html')

			fig.write_html(figname)
			if verbose: print(">>> HTML and png saved:", figname)
			fig.write_image(figname.replace('.html', '.png'))

		return dic_fig


	def plot_nDEG_curve_per_LFC_FDR(self, case:str,
									width:int=1100, height:int=700, title:str=None,
									cols2:List=['n_degs', 'abs_lfc_cutoff'],
									corr_cutoff:float=-0.90, nregs_fdr:int=5, method:str='spearman', verbose:bool=False):

		'''
			plot_nDEG_curve_per_LFC_FDR

			call calc_nDEG_curve_per_LFC_FDR
					   return ret, dic

			return ret, dic_fig, df_fdr
		'''

		ret, dic_return = self.calc_nDEG_curve_per_LFC_FDR(case=case, cols2=cols2,
														   corr_cutoff=corr_cutoff, nregs_fdr=nregs_fdr,
														   method=method, verbose=verbose)

		if not ret:
			print("Could not get data from calc_nDEG_curve_per_LFC_FDR()")
			return ret, None, None


		df_fdr	= dic_return['df_fdr']
		name_list = dic_return['name_list']
		# fdrs	  = df_fdr.fdr.to_list()

		if title is None:
			title = f'{self.s_deg_dap}s curve per fdr x lfc for {case}<br>for corr_cutoff={corr_cutoff:.3f} and correlation dependes on abs_lfc_cutoff_inf={self.abs_lfc_cutoff_inf:.3f}'

		title2 = title

		yaxis_title = f"num of {self.s_deg_dap}s"
		yaxis_title2 = yaxis_title
		xaxis_title = "abs_LFC cutoff"
		legend_title = 'FDR cutoff'

		dic_fig = {}
		for which in ['deg', 'up', 'down']:
			''' many figures: one case, deg, up and down '''
			fig = go.Figure()

			for i in range(len(df_fdr)):
				row  = df_fdr.iloc[i]
				fdr  = row['fdr']
				corr = row['corr']

				if which == 'deg':
					try:
						name = row.label
					except:
						name = f'{fdr:.2f} \t {corr:.3f}'
				else:
					name = f'{fdr:.2f}'


				dfsim = self.dfsim[self.dfsim.case == case]
				dfsim = dfsim[ (dfsim.fdr_lfc_cutoff == fdr) & (dfsim.abs_lfc_cutoff >= self.abs_lfc_cutoff_inf) ]

				if which == 'deg':
					col = 'n_degs'
					yaxis_title2 = yaxis_title
					title2 = title
					legend_title = 'FDR cutoff - Spearman corr.'
				else:
					col = 'n_degs_up' if which == 'up' else 'n_degs_dw'
					stri = 'Up' if which == 'up' else 'Down'
					yaxis_title2 = yaxis_title + ' ' + stri
					title2 = title.replace('s curve', f's {stri} curve')
					legend_title = 'FDR cutoff'

				fig.add_trace(go.Scatter(x=dfsim.abs_lfc_cutoff, y=dfsim[col], name=name))


			fig.update_layout(
						autosize=True,
						title=title2,
						width=width,
						height=height,
						xaxis_title=xaxis_title,
						yaxis_title=yaxis_title2,
						legend_title=legend_title,
						showlegend=True,
						font=dict(
							family="Arial",
							size=14,
							color="Black"
						)
			)
			dic_fig[which] = fig

			figname = title_replace(title2)
			figname = os.path.join(self.root_figure, figname+'.html')

			fig.write_html(figname)
			if verbose: print(">>> HTML and png saved:", figname)
			fig.write_image(figname.replace('.html', '.png'))

		return True, dic_fig, df_fdr


	def plot_toi_versus_genes_and_pathways(self, case:str, selected_toi_col:str, _plot:str='genes',
				width:int=1100, height:int=500, plot_all_dfi:bool=True,
				colors:List=['olivedrab', 'navy', 'red', 'darkcyan', 'darkgreen', 'orange', 'brown', 'darksalmon',
							 'magenta', 'darkturquoise', 'orange', 'darkred', 'indigo', 'magenta', 'maroon', 'black',
							 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen', 'navy'],
				verbose:bool=False) -> List:
		'''
			_plot = ['genes', 'pathways']
		'''

		# colors += plotly_colors_proteins

		dfcut = self.build_all_cutoffs_table(selected_toi_col, force=False, verbose=verbose)
		# fig = make_subplots(rows=2, cols=1, subplot_titles=['genes', 'pathways'])

		if _plot == 'genes':
			title=f"Best number of {self.s_deg_dap}s x {selected_toi_col} for {case}"
			yaxis_title = f"# {self.s_deg_dap}s in pathways"
		else:
			title = f"Best number of enriched pathways x {selected_toi_col} for {case}"
			yaxis_title = "# of pathways"

		fig = go.Figure()
		i = -1

		df2  = dfcut[ (dfcut.case == case) ]
		fdr_list = np.unique(df2.fdr_lfc_cutoff)

		if plot_all_dfi: dfi = self.calc_enrichment_cutoff_params_and_ndxs_per_case_and_geneset_lib(case)

		for fdr in fdr_list:
			i += 1

			df2  = dfcut[ (dfcut.case == case) & (dfcut.fdr_lfc_cutoff == fdr) ]
			if plot_all_dfi: dfi2 = dfi[   (dfi.case == case)   & (dfi.fdr_lfc_cutoff == fdr) ]

			if df2.empty:
				continue

			name1 = f"{case} fdr={fdr:.3f} for {_plot}"
			name2 = name1 + '_all'
			df2  = df2.sort_values( selected_toi_col, ascending=True)
			if plot_all_dfi: dfi2 = dfi2.sort_values(selected_toi_col, ascending=True)
			color = colors[i]

			if _plot == 'genes':
				fig.add_trace(go.Scatter(x=df2[selected_toi_col],   y=df2.n_degs_in_pathways,  marker_color=color, name=name1) ) # , row=1, col=1)
				if plot_all_dfi: fig.add_trace(go.Scatter(x=dfi2[selected_toi_col], y=dfi2.n_degs_in_pathways, line=dict(dash='dash'), marker_color=color, name=name2)) # , row=1, col=1)
			else:
				fig.add_trace(go.Scatter(x=df2[selected_toi_col],   y=df2.n_pathways, marker_color=color, name=name1)) #, row=2, col=1)
				if plot_all_dfi: fig.add_trace(go.Scatter(x=dfi2[selected_toi_col], y=dfi2.n_pathways, line=dict(dash='dash'), marker_color=color, name=name2)) #, row=2, col=1)

		fig.update_layout(title=title,
						  width=width,
						  height=height,
						  xaxis_title=selected_toi_col,
						  yaxis_title=yaxis_title,
						  # xaxis2_title=selected_toi_col,
						  legend_title="cases",
						  showlegend=True)

		figname = title_replace(title)
		figname = os.path.join(self.root_figure, figname+'.html')

		fig.write_html(figname)
		if verbose: print(">>> HTML and png saved:", figname)
		fig.write_image(figname.replace('.html', '.png'))

		#fig['layout']['yaxis']['title']=yaxis_title1
		# fig['layout']['yaxis2']['title']=yaxis_title2
		return fig


	def calc_min_max(self, vals):
		mini = np.min(vals)
		maxi = np.max(vals)

		if mini * maxi < 0:
			all_max0 = maxi if np.abs(maxi) >= np.abs(mini) else mini
			all_max0_abs = np.abs(all_max0)
			all_max = int(np.round(all_max0_abs, 0))

			if all_max < all_max0_abs:
				all_max += 1

			ret = [-all_max, +all_max]
		else:
			mini_abs = np.abs(mini)
			maxi_abs = np.abs(maxi)

			all_max0 = maxi_abs if maxi_abs >= mini_abs else mini_abs
			all_max = int(np.round(all_max0, 0))

			if all_max < all_max0:
				all_max += 1

			if mini < 0:
				ret = [-all_max, 0]
			else:
				ret = [0, all_max]

		return ret


	def plot_all_lfc_path(self, pathway_and_id:List, width:int=500, height:int=300, verbose:bool=False) -> dict:

		# pathway, pathway_id = pathway_and_id
		dfpiv, _  = self.calc_pivot_one_pathway_gene_modulations(pathway_and_id, force=False, verbose=verbose)

		dic = {}
		for i in range(len(dfpiv)):
			row  = dfpiv.iloc[i]
			symbol = row.name
			vals_female = np.array(row[self.group_female_list])
			vals_male   = np.array(row[self.group_male_list])
			fig = self.plot_lfc_path(symbol, vals_female, vals_male, width=width, height=height, verbose=verbose)

			dic[symbol] = fig

		return dic


	def plot_genes_lfc_path(self, subtitle:str, df:pd.DataFrame, symbol_list:List, width:int=500, height:int=300, verbose:bool=False):


		dic = {}
		for symbol in symbol_list:
			dfq  = df[df.symbol == symbol]
			if dfq.empty: continue

			row = dfq.iloc[0]

			vals_female = np.array(row[self.group_female_list])
			vals_male   = np.array(row[self.group_male_list])
			fig = self.plot_lfc_path(symbol, vals_female, vals_male, subtitle, width=width, height=height, verbose=verbose)

			dic[symbol] = fig

		return dic

	def plot_lfc_path(self, symbol, vals_female, vals_male, subtitle:str='', width:int=500, height:int=300, verbose:bool=False):
		groups = self.group_list
		colors = self.group_colors

		line_traces = []
		for i in range(len(groups)-1):
			name  = f"{groups[i]}-{groups[i+1]}"
			line_traces.append(go.Scatter(x=vals_female[i:(i+1)], y=vals_male[i:(i+1)],
										  marker_color=colors[i+1], name=name, mode="lines",  ))

		arrow_traces = []
		for i in range(len(vals_female) - 1):
			arrow_trace = go.layout.Annotation(
				dict( x=vals_female[i+1], y=vals_male[i+1],
					  xref="x", yref="y", text="",
					  showarrow=True, axref="x", ayref="y",
					  ax=vals_female[i],  ay=vals_male[i],
					  arrowhead=3,  arrowwidth=1, arrowcolor=colors[i+1],
				)
			)
			arrow_traces.append(arrow_trace)

		fig = go.Figure(data=line_traces)

		x_range = self.calc_min_max(vals_female)
		y_range = self.calc_min_max(vals_male)

		title = f"LFC path for {symbol}"
		if subtitle is not None and subtitle != '':
			title += '<br>' + subtitle

		fig.update_layout(title=title,
					  width=width,
					  height=height,
					  annotations=arrow_traces,
					  xaxis_title="female",
					  yaxis_title="male",
					  showlegend=True,
					  xaxis_range = x_range,
					  yaxis_range = y_range)

		figname = title_replace(title)
		figname = os.path.join(self.root_figure, figname+'.html')

		fig.write_html(figname)
		if verbose: print(">>> HTML and png saved:", figname)
		fig.write_image(figname.replace('.html', '.png'))

		return fig


	def open_affymetrix_human_table(self, fname_affy:str='Human_Agilent_WholeGenome_4x44k_v2_MSigDB_v71.tsv',
									verbose:bool=False) -> pd.DataFrame:
		self.fname_affy = fname_affy
		df_affy = pdreadcsv(fname_affy, self.root_affymetrix, verbose=verbose)

		try:
			df_affy.columns = ['probe', 'symbol', 'description']
		except:
			print("Review df_affy columns, please.")

		self.df_affy = df_affy

		return df_affy

	def test_affymetrix_probe(self, probe:str) -> bool:
		return not self.df_gpl[self.df_gpl.probe == probe].empty

	def open_new_ALL_LFC_table(self, verbose:bool=False):
		fname_final_ori = self.fname_final_lfc_table0%(self.case, self.normalization)

		dflfc_new = pdreadcsv(fname_final_ori, self.root_result, verbose=verbose)
		self.dflfc_new = dflfc_new

		return dflfc_new


	def calc_unique_LFC_replace_synonyms(self, df):

		df = df.sort_values(['symbol', 'abs_lfc'], ascending=[True, False])
		df.index = np.arange(0, len(df))

		previous = ''; goods = []
		for i in range(len(df)):

			if not isinstance(df.iloc[i].symbol, str) or  df.iloc[i].symbol == '':
				goods.append(True)
			elif df.iloc[i].symbol != previous:
				previous = df.iloc[i].symbol
				goods.append(True)
			else:
				goods.append(False)

		df = df[goods]
		df['symb_or_syn'] = df.symbol
		df.loc[:, 'symbol'] = [self.gene.replace_synonym_to_symbol(x) for x in df.symbol]

		return df


	def review_LFC_table_with_affymetrix_annotation(self, case, force:bool=False, calc_interm_tables:bool=False, verbose:bool=False) -> bool:
		'''
			set case
			call review_LFC_table_with_affy_annot_wo_case
		'''

		ret, _, _, _ = self.open_case(case, verbose=verbose)
		if not ret:
			return False

		ret = self.review_LFC_table_with_affy_annot_wo_case(force=force, calc_interm_tables=calc_interm_tables, verbose=verbose)

	def review_LFC_table_with_affy_annot_wo_case(self, force:bool=False, calc_interm_tables:bool=False, verbose:bool=False) -> bool:
		'''
		There are 3 final tables:
			self.df_good, self.df_new_symbol, self.df_empty_gpl_lnc_new, self.df_empty

			self.df_good: all annotated RNAs
			self.df_empty_gpl_lnc_new:  new not annotated LNC - df_empty_gpl_lnc_new['_type'] = 'LNC'
			self.df_empty: all empty symbols and not LNC

			all data is stored in dflfc_new

		'''
		case = self.case
		print(">>>", case)

		self.dflfc_all, self.df_good, self.df_empty = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
		self.df_good_gpl, self.df_empty_gpl = pd.DataFrame(), pd.DataFrame()
		self.df_empty_gpl_lnc_new, self.df_empty_gpl_lnc, self.df_empty_gpl_not_lnc = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

		fname_final_ori = self.fname_final_lfc_table0%(case, self.normalization)
		fullname = os.path.join(self.root_result, fname_final_ori)

		if not calc_interm_tables and os.path.exists(fullname) and not force:
			dflfc_new = pdreadcsv(fname_final_ori, self.root_result, verbose=verbose)
			self.dflfc_new = dflfc_new
			return True

		fname = self.fname_lfc_table0%(self.case, self.normalization)
		dflfc_all = pdreadcsv(fname, self.root_result)
		if verbose: print(">>> Before clean duplicates: len(dflfc_all)", len(dflfc_all))
		dflfc_all = self.calc_unique_LFC_replace_synonyms(dflfc_all)
		if verbose: print(">>> After clean duplicates: len(dflfc_all)", len(dflfc_all))
		self.dflfc_all = dflfc_all

		if dflfc_all is None or dflfc_all.empty:
			return False

		df_good = dflfc_all[ ~pd.isnull(dflfc_all.symbol) ].copy()
		df_good.index = np.arange(0, len(df_good))
		df_good['_type'] = 'hasSymbol'
		self.df_good = df_good
		cols_good_rename = ['probe', 'symbol_prev', 'lfc', 'abs_lfc', 'pval', 'fdr', 'description_prev', 'symbol_pipe',
							'mean_exp', 't', 'B', 'symb_or_syn', '_type']
		df_good.columns = cols_good_rename

		df_empty = dflfc_all[ pd.isnull(dflfc_all.symbol) ].copy()
		self.df_empty = df_empty
		df_empty['_type'] = 'UNK'
		df_empty.index = np.arange(0, len(df_empty))
		df_empty.columns = cols_good_rename

		if len(dflfc_all) != len(df_good)+len(df_empty):
			print(f"Problems with lens: dflfc_all:{len(dflfc_all)}, df_good={len(df_good)}, and df_empty={len(df_empty)}")
			raise Exception("Stop: review_LFC_table_with_affymetrix_annotation()")

		df_good_gpl = pd.merge(df_good, self.df_gpl, how='inner', on='probe')
		df_good_gpl.index = np.arange(0, len(df_good_gpl))
		self.df_good_gpl = df_good_gpl
		if verbose: print(">>>> LFC table symbols valid and annotated: df_good_gpl", len(df_good), 'x', len(df_good_gpl))

		cols_good_gpl  = ['probe', 'symbol', 'symbol_prev', 'symb_or_syn', '_type', 'lfc', 'abs_lfc', 'pval', 'fdr', 'mean_exp', 't', 'B',
						  'accession', 'ensembl_id',  'geneid', 'cytoband', 'description', 'description_prev',
						  'symbol_pipe', 'seqname', 'start', 'end', 'go_id', 'seq' ]
		df_good_gpl = df_good_gpl[cols_good_gpl]

		if len(df_good_gpl) != len(df_good):
			print("df_good files have diferent sizes: df_gpl did not map correctly")
			return False


		df_empty_gpl = pd.merge(df_empty, self.df_gpl, how='inner', on='probe')
		self.df_empty_gpl = df_empty_gpl
		if verbose: print(">>>> LFC table symbols empty and annotated: df_good_gpl", len(df_empty), 'x', len(df_empty_gpl))
		df_empty_gpl = df_empty_gpl[cols_good_gpl]

		if df_empty_gpl is None or df_empty_gpl.empty:
			print("No data for df_empty_gpl")
			return False

		if len(df_empty_gpl) != len(df_empty):
			print("df_empty files have diferent sizes: df_gpl did not map correctly")
			return False

		''' after merging, some df_empty_gpl are annotated, some not '''
		df_empty_gpl_symb = df_empty_gpl[ ~pd.isnull(df_empty_gpl.symbol)].copy()
		df_empty_gpl_symb.index = np.arange(0, len(df_empty_gpl_symb))
		self.df_empty_gpl_symb = df_empty_gpl_symb

		df_empty_gpl_nosymb = df_empty_gpl[ pd.isnull(df_empty_gpl.symbol)].copy()
		df_empty_gpl_nosymb.index = np.arange(0, len(df_empty_gpl_nosymb))
		self.df_empty_gpl_nosymb = df_empty_gpl_nosymb

		good_lncs = [ True if ( isinstance( df_empty_gpl_nosymb.iloc[i].description, str) and \
							  ('lincRNA' in df_empty_gpl_nosymb.iloc[i].description or \
							   'lncRNA'  in df_empty_gpl_nosymb.iloc[i].description) ) else False \
					   for i in range(len(df_empty_gpl_nosymb))]

		df_empty_gpl_lnc = df_empty_gpl_nosymb[good_lncs].copy()
		df_empty_gpl_lnc.index = np.arange(0, len(df_empty_gpl_lnc))
		df_empty_gpl_lnc.loc[:,'symbol'] = [f'lncRNA{i+1}' for i in range(len(df_empty_gpl_lnc))]
		df_empty_gpl_lnc['_type'] = 'LNC'
		self.df_empty_gpl_lnc = df_empty_gpl_lnc

		df_empty_gpl_not_lnc = df_empty_gpl_nosymb[np.invert(good_lncs)].copy()
		df_empty_gpl_not_lnc.index = np.arange(0, len(df_empty_gpl_not_lnc))
		self.df_empty_gpl_not_lnc = df_empty_gpl_not_lnc

		'''
		df_empty_gpl = df_empty_gpl_symb + df_empty_gpl_nosymb
		df_empty_gpl_nosymb = df_empty_gpl_lnc + df_empty_gpl_not_lnc

		df_empty_gpl_lnc_new  = df_empty_gpl_symb + df_empty_gpl_lnc + df_empty_gpl_not_lnc
		'''
		df_empty_gpl_lnc_new  = pd.concat([df_empty_gpl_symb, df_empty_gpl_lnc, df_empty_gpl_not_lnc])
		df_empty_gpl_lnc_new.index = np.arange(0, len(df_empty_gpl_lnc_new))
		self.df_empty_gpl_lnc_new = df_empty_gpl_lnc_new

		''' ----------- GFF3 annoation -----------------'''
		dfgff = self.gene.prepare_final_gff(force=False, verbose=False)
		# & ( ~pd.isnull(dfgff.ensembl_id) )
		dfgff_symb = dfgff[ ~pd.isnull(dfgff.symbol) ].copy()
		dfgff_symb.index = np.arange(0, len(dfgff_symb))

		cols_gff = ['symbol', 'ensembl_id', 'biotype', 'description', ]
		dfgff_symb = dfgff_symb[cols_gff]

		dfgff_symb = dfgff_symb.sort_values(['symbol', 'biotype'])
		if verbose: print(">>> GFF3 annotation: before removing = ", len(dfgff_symb))

		'''
		There are repeated gff annotations for one gene with different ensembl_id

			 	symbol 	ensembl_id 			biotype						   description
		1556 	ZPLD2P 	ENSG00000236155 	transcribed_unitary_pseudogene   zona pellucida like dom...
		1557 	ZPLD2P 	ENSG00000293494 	lncRNA						   zona pellucida like domain ...
		'''
		previous = ''; goods = []
		for i in range(len(dfgff_symb)):

			if dfgff_symb.iloc[i].symbol != previous:
				previous = dfgff_symb.iloc[i].symbol
				goods.append(True)
			else:
				goods.append(False)

		dfgff_symb = dfgff_symb[goods]

		if len(dfgff_symb) == len(dfgff_symb.symbol.unique()):
			if verbose: print(">>> GFF3 annotation: after remove duplicates", len(dfgff_symb), 'Ok.')
		else:
			print("Error: GFF3 annotation: after remove duplicates {len(dfgff_symb)} != {len(dfgff_symb.symbol.unique())}")
		self.dfgff_symb = dfgff_symb

		'''------ reviewing tables ---------------------------------------------'''
		dflfc_new = pd.concat([df_good_gpl, df_empty_gpl_lnc_new])
		dflfc_new = dflfc_new.sort_values('probe')
		dflfc_new.index = np.arange(0, len(dflfc_new))
		self.dflfc_new = dflfc_new.copy()

		''' some symbols may be fulfilled '''
		df_new_empty = dflfc_new[ pd.isnull(dflfc_new.symbol) ].copy()
		df_new_empty.index = np.arange(0, len(df_new_empty))
		self.df_new_empty = df_new_empty

		df_new_symbol = dflfc_new[ (~pd.isnull(dflfc_new.symbol)) & (dflfc_new._type != 'LNC') ].copy()
		df_new_symbol.index = np.arange(0, len(df_new_symbol))
		self.df_new_symbol = df_new_symbol

		df_lnc_new = dflfc_new[ (~pd.isnull(dflfc_new.symbol)) & (dflfc_new._type == 'LNC') ].copy()
		df_lnc_new.index = np.arange(0, len(df_lnc_new))
		self.df_lnc_new2 = df_lnc_new

		df_new_symbol_gff = pd.merge(df_new_symbol, dfgff_symb, how='inner', on=['symbol'])
		''' ensembl_id --> ensembl_transc_id is a transcript '''
		cols_rename = ['probe', 'symbol', 'symbol_prev', 'symb_or_syn', '_type', 'lfc', 'abs_lfc', 'pval', 'fdr', 'mean_exp', 't',
					   'B', 'accession', 'ensembl_transc_id', 'geneid', 'cytoband', 'description', 'description_prev',
					   'symbol_pipe', 'seqname', 'start', 'end', 'go_id', 'seq', 'ensembl_id', 'biotype', 'desc_gff']
		df_new_symbol_gff.columns = cols_rename

		''' final columns to be saved and used '''
		cols_final = ['probe', 'symbol', 'symbol_prev', 'symb_or_syn', 'biotype', '_type', 'lfc', 'abs_lfc',
					  'pval', 'fdr', 'mean_exp', 't', 'B', 'description', 'desc_gff',
					  'description_prev', 'accession', 'ensembl_id', 'ensembl_transc_id',
					  'geneid', 'cytoband', 'symbol_pipe', 'seqname', 'start', 'end', 'go_id', 'seq' ]
		df_new_symbol_gff = df_new_symbol_gff[cols_final]
		self.df_new_symbol_gff = df_new_symbol_gff

		df_new_symbol_not_gff = df_new_symbol[~df_new_symbol.probe.isin(df_new_symbol_gff.probe)].copy()
		''' did not merge, remove last 3 columns '''
		df_new_symbol_not_gff.columns = cols_rename[:-3]
		df_new_symbol_not_gff['ensembl_id'] = None
		df_new_symbol_not_gff['biotype'] = df_new_symbol_not_gff['_type']
		df_new_symbol_not_gff['desc_gff'] = None
		df_new_symbol_not_gff = df_new_symbol_not_gff[cols_final]
		df_new_symbol_not_gff.index = np.arange(0, len(df_new_symbol_not_gff))
		self.df_new_symbol_not_gff = df_new_symbol_not_gff

		'''--------------------- fix mistakes for LOC --------------------------'''

		''' LNC -> there are descriptions like
			"BROAD lincRNAs version v2 (http://www.broadinstitute.org/genome_bio/human_lincrnas/)"
		'''
		df_new_unk = df_new_symbol_not_gff[ df_new_symbol_not_gff._type == 'UNK' ]
		if not df_new_unk.empty:

			probe_list = [df_new_unk.iloc[i].probe for i in range(len(df_new_unk)) \
						  if  isinstance(df_new_unk.iloc[i].description, str) and \
						  ('lincRNA' in df_new_unk.iloc[i].description or 'lncRNA' in df_new_unk.iloc[i].description)]
			if len(probe_list) > 0:
				df_new_symbol_not_gff.loc[df_new_symbol_not_gff.probe.isin(probe_list), ('biotype', '_type')] = [('LNC', 'LNC')]*len(probe_list)

			''' LOCxxxx - uncharacterized RNA - ncRNA '''
			probe_list = [df_new_unk.iloc[i].probe for i in range(len(df_new_unk)) if df_new_unk.iloc[i].symbol.startswith('LOC')]
			if len(probe_list) > 0:
				'''												non-coding, uncharacterized '''
				df_new_symbol_not_gff.loc[df_new_symbol_not_gff.probe.isin(probe_list), ('biotype', '_type')] = [('ncRNA', 'UNC')]*len(probe_list)

		'''----------------------- end fix mistakes ----------------------------'''


		''' did not merge, remove last 3 columns '''
		df_lnc_new.columns = cols_rename[:-3]
		df_lnc_new['ensembl_id'] = None
		df_lnc_new['biotype'] = df_lnc_new['_type']
		df_lnc_new['desc_gff'] = None
		df_lnc_new = df_lnc_new[cols_final]
		self.df_lnc_new = df_lnc_new

		df_empty_new = dflfc_new[ pd.isnull(dflfc_new.symbol) ].copy()
		df_empty_new.index = np.arange(0, len(df_empty_new))
		''' did not merge, remove last 3 columns '''
		df_empty_new.columns = cols_rename[:-3]
		df_empty_new['ensembl_id'] = None
		df_empty_new['biotype'] = df_empty_new['_type']
		df_empty_new['desc_gff'] = None
		df_empty_new = df_empty_new[cols_final]
		self.df_empty_new = df_empty_new

		''' concat: good_gpl, lnc, not_lnc '''
		dflfc_new_final = pd.concat([df_new_symbol_gff, df_new_symbol_not_gff, df_lnc_new, df_empty_new])
		dflfc_new_final = dflfc_new_final.sort_values('symbol')
		dflfc_new_final.index = np.arange(0, len(dflfc_new_final))

		previous = ''; goods = []
		for i in range(len(dflfc_new_final)):

			if not isinstance(dflfc_new_final.iloc[i].symbol, str):
				goods.append(False)
			elif dflfc_new_final.iloc[i].symbol != previous:
				previous = dflfc_new_final.iloc[i].symbol
				goods.append(True)
			else:
				goods.append(False)

		dflfc_new_final = dflfc_new_final[goods]
		dflfc_new_final = dflfc_new_final.sort_values('probe')
		dflfc_new_final.index = np.arange(0, len(dflfc_new_final))
		self.dflfc_new_final = dflfc_new_final

		'''------ again ---  calc_interm_tables --------------------------------'''
		if not os.path.exists(fullname) or force:
			pdwritecsv(dflfc_new_final, fname_final_ori, self.root_result, verbose=verbose)

		print(f"There are {len(self.dflfc_all)} unique RNAs/probes in the microarray expression table")
		print(f"And the processed data len(dflfc_new_final) = {len(self.dflfc_new_final)}")

		if len(self.dflfc_all) == len(self.dflfc_new_final):
			print(f"It is ok")
		else:
			print(f"Error: there are problems, {len(self.dflfc_all)} != {len(self.dflfc_new_final)}")

		dflfc_lncRNA = self.dflfc_new_final[self.dflfc_new_final.biotype == 'lncRNA']
		len(dflfc_lncRNA)

		print(f"Related to the processed RNAs ({len(self.dflfc_new_final)})")
		print(f"\tSymbols	 annotated in GFF = {len(self.df_new_symbol_gff)}")
		print(f"\tSymbols not annotated in GFF = {len(self.df_new_symbol_not_gff)}")
		print(f"\tEmpty symbols = {len(self.df_empty_new)}")
		print(f"\tNot annotated LNC = {len(self.df_lnc_new)}")
		print(f"\tAnnotated	 LNC = {len(dflfc_lncRNA)}")
		print("")

		self.dflfc_ori = dflfc_new_final
		print(f"\tdflfc_ori: contains {len(dflfc_new_final)}")

		self.valid_genes = [x for x in dflfc_new_final.symbol if isinstance(x, str)]
		self.n_total_genes = len(self.valid_genes)

		print(f"\tdflfc_ori: has {self.n_total_genes} valid symbols.")

		return True

	def get_dflfc_biotypes(self):
		dfg = self.dflfc.groupby('biotype').count().reset_index().iloc[:, :2]
		dfg.columns = ['biotype', 'n']
		return dfg

	def get_dflfc_ori_biotypes(self):
		dfg = self.dflfc_ori.groupby('biotype').count().reset_index().iloc[:, :2]
		dfg.columns = ['biotype', 'n']
		return dfg

	def get_df_biotypes(self, df:pd.DataFrame):
		dfg = df.groupby('biotype').count().reset_index().iloc[:, :2]
		dfg.columns = ['biotype', 'n']
		return dfg

	def split_chr_location(self, x:str):
		if x == 'unmapped':
			return None, None, None
		if pd.isnull(x) or x == 'nan':
			return None, None, None

		try:
			mat =  x[3:].split(":")
			seqname = mat[0]
			start, end = mat[1].split('-')
		except:
			print(x)

		start = int(start) if isint(start) else start
		end   = int(end)   if isint(end)   else end

		return seqname, start, end

	def open_affymetrix_final_table(self, fname_affy:str="GPL14550-9757.tsv", verbose:bool=False) -> pd.DataFrame:

		fname_geo = fname_affy.replace('.tsv', '_location.tsv')
		self.fname_geo = fname_geo
		fullname = os.path.join(self.root_affy, fname_geo)

		if not os.path.exists(fullname):
			print(f"Could not find '{fullname}'")
			self.df_gpl = pd.DataFrame()
			return self.df_gpl

		df_gpl = pdreadcsv(fname_geo, self.root_affy, verbose=verbose)
		self.df_gpl = df_gpl

		return df_gpl

	def open_affymetrix_table(self, fname_geo:str, verbose:bool=False) -> pd.DataFrame:
		self.fname_geo = fname_geo
		df_gpl = pdreadcsv(fname_geo, self.root_affy, verbose=verbose)

		cols = ['probe', 'probe_id', 'CONTROL_TYPE', 'refseq', 'accession', 'geneid', 'symbol', 'name',
	   'unigene_id', 'ensembl_id', 'TIGR_ID', 'ACCESSION_STRING', 'chr_location', 'cytoband', 'description', 'go_id', 'seq']
		df_gpl.columns = cols
		cols2  = ['probe', 'refseq', 'accession', 'geneid', 'symbol', 'name',
				  'unigene_id', 'ensembl_id', 'chr_location', 'cytoband', 'description', 'go_id', 'seq']

		df_gpl = df_gpl[cols2]
		self.df_gpl = df_gpl

		return df_gpl

	def prepare_gpl(self, fname_affy:str="GPL14550-9757.tsv", force:bool=False, verbose:bool=False) -> bool:

		fname = fname_affy.replace('.tsv', '_location.tsv')
		fullname = os.path.join(self.root_affy, fname)

		if os.path.exists(fullname) and not force:
			df_gpl = pdreadcsv(fname, self.root_affy, verbose=verbose)
			self.df_gpl = df_gpl
			return True

		df_gpl = self.open_affymetrix_table(fname_geo=fname_affy, verbose=verbose)

		df_gpl.loc[:, ['seqname', 'start', 'end']] = [self.split_chr_location(x) for x in df_gpl.chr_location]

		cols = ['probe', 'refseq', 'accession', 'geneid', 'symbol', 'name', 'description',
				'unigene_id', 'ensembl_id', 'cytoband', 'seqname', 'start', 'end', 'go_id', 'seq']

		df_gpl = df_gpl[cols]
		self.df_gpl = df_gpl

		ret = pdwritecsv(df_gpl, fname, self.root_affy, verbose=verbose)
		return ret


	def calc_degs_for_bayesian(self, dflfc:pd.DataFrame, FDR_cutoff:float, abs_lfc_cutoff:float):
		return len(dflfc[ (dflfc.fdr < FDR_cutoff) & (dflfc.abs_lfc >= abs_lfc_cutoff) ])

	def calc_norm_cdf_for_bayesian(self, lfc:float):
		p = norm.cdf(lfc, loc=1, scale=1.0)
		if p > 0.5: p = 1 - p
		return p



	def calc_bayesian_cutoffs(self, case:str, ndraws:int=1000, xaxis_title:str='lfc', yaxis_title:str='p',
					 fdr_lista:List=np.round(np.arange(0.05, 0.80, 0.05), 2), perc_delta:float=0.01,
					 width:int=1100, height:int=600, plot_bgcolor:str='lightgray', verbose:bool=False) -> dict:


		ret, _, _, _ = self.open_case(case, verbose=False)
		dflfc = self.dflfc_ori
		N = len(dflfc)

		lfc_samples = np.random.normal(loc=1, scale=1.0, size=ndraws)
		lfc_samples = [x for x in lfc_samples if x >= 0]

		subtitles = ['LFC', 'p(DEG|LFC)', 'nDEG', 'p(LFC)', 'p(LFC|DEG)']

		fig_list = {}

		dic = {}; icount = -1
		for FDR in fdr_lista:
			title = f'Bayesian simulation for {case} with {ndraws} draws and FDR = {FDR}'

			pis   = [ self.calc_degs_for_bayesian(dflfc, FDR, lfc_sample)/N for lfc_sample in lfc_samples]
			ndegs = [ self.calc_degs_for_bayesian(dflfc, FDR, lfc_sample)   for lfc_sample in lfc_samples]

			df = pd.DataFrame( {'lfc': lfc_samples, 'p': pis, 'ndeg': ndegs} )
			df = df.sort_values('lfc')
			df.index = np.arange(0, len(df))

			nrows = len(df)

			df['p_lfc'] = [ self.calc_norm_cdf_for_bayesian(lfc) for lfc in df.lfc]
			# normalizing
			total = df['p_lfc'].sum()
			df['p_lfc'] = df['p_lfc']/total
			
			df['p_posterior'] = [df.iloc[i].p * df.iloc[i].p_lfc for i in range(len(df))]
			total = df['p_posterior'].sum()
			df['p_posterior'] = df['p_posterior']/total
			
			mu = np.sum( df['p_posterior'] * df['lfc'] )
			delta = perc_delta * mu
			dfm = df[ (df.lfc >= mu-delta) & (df.lfc <= mu+delta) ]
			i_mu = int(np.median(dfm.index))
			
			maxi = df.p_posterior.max()
			delta = 0.001 * maxi
			dfm = df[ (df.p_posterior >= maxi-delta) & (df.p_posterior <= maxi+delta) ]
			lfc_max = dfm.lfc.mean()

			icount += 1
			dic[icount] = {}
			dic2 = dic[icount]
			dic2['case'] = case
			dic2['fdr'] = FDR
			dic2['lfc_max'] = lfc_max
			dic2['lfc_mean'] = mu

			stri = f'For FDR={FDR}, the lfc_max = {lfc_max:.3f}'

			for alpha in [0.9, 0.8, 0.7]:
				delta = int(np.round(nrows * alpha/ 2, 0))

				lim_inf = i_mu - delta
				if lim_inf < 0: lim_inf = 0
				lim_sup = i_mu + delta
				if lim_sup > nrows: lim_sup = nrows-1

				ci = np.round([df.iloc[lim_inf].lfc, df.iloc[lim_sup].lfc], 3)
				stri = f"\tfor alpha={alpha} the CI = {ci}"

				dic2[f'CI_{alpha}'] = ci
			
			fig = make_subplots(rows=2, cols=3, subplot_titles=subtitles)
			
			for i in range(5):
				if i == 0:
					nrow=1
					ncol=1
					vals = df.lfc
				elif i == 1:
					nrow=1
					ncol=2
					vals = df.p
				elif i == 2:
					nrow=1
					ncol=3
					vals = df.ndeg
				elif i == 3:
					nrow=2
					ncol=1
					vals = df.p_lfc
				else:
					nrow=2
					ncol=2
					vals = df.p_posterior
			
				name = subtitles[i]
			
				if i == 0:
					fig.add_trace(go.Histogram(x=vals, name=name) , row=nrow, col=ncol)
				else:
					fig.add_trace(go.Scatter(x=df.lfc, y=vals, name=name) , row=nrow, col=ncol)
			
			
			fig.update_layout(
						autosize=True,
						title=title,
						width=width,
						height=height,
						xaxis_title=xaxis_title,
						yaxis_title=yaxis_title,
						showlegend=True,
						legend_title='probs',
						plot_bgcolor=plot_bgcolor,
						font=dict(
							family="Arial",
							size=14,
							color="Black"
						) )

			dic2['fig'] = fig

			figname = title_replace(title)
			figname = os.path.join(self.root_figure, figname+'.html')

			fig.write_html(figname)
			if verbose: print(">>> HTML and png saved:", figname)
			fig.write_image(figname.replace('.html', '.png'))

		return dic


	def plot_bayesian_cutoff_series(self, case:str, dfc:pd.DataFrame, xaxis_title:str='FDR',
					                width:int=1000, height:int=600, plot_bgcolor:str='lightgray', verbose:bool=False):

		subtitles = ['lfc_max', 'lfc_mean']

		title = f'max/mean x FDR for {case}'

		for i in range(2):

			fig = make_subplots(rows=2, cols=1, subplot_titles=subtitles)
			
			for i in range(2):
				if i == 0:
					nrow=1
					ncol=1
					vals = dfc.lfc_max
				elif i == 1:
					nrow=2
					ncol=1
					vals = dfc.lfc_mean
			
				name = subtitles[i]
			
				fig.add_trace(go.Scatter(x=dfc.fdr, y=vals, name=name) , row=nrow, col=ncol)
			
			
		fig.update_layout(
					autosize=True,
					title=title,
					width=width,
					height=height,
					xaxis2_title=xaxis_title,
					yaxis_title ='max LFC',
					yaxis2_title ='mean LFC',
					showlegend=False,
					legend_title='probs',
					plot_bgcolor=plot_bgcolor,
					font=dict(
						family="Arial",
						size=14,
						color="Black"
					) )

		return fig