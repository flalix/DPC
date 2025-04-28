#!/usr/bin/python
#!python
# -*- coding: utf-8 -*-

import json, requests
import os, sys, io
import pandas as pd
# import enforce
import numpy as np
from numpy import pi, sin, cos
from typing import Optional, Iterable, Set, Tuple, Any, List

import Bio.KEGG as kegg
from Bio.KEGG import REST
from Bio.KEGG.REST import *
from Bio.KEGG.KGML import KGML_parser

import plotly.graph_objects as go
import networkx as nx

import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.stats import f_oneway

from Basic import *
from venn_lib import *
from biopax_lib import *
from stat_lib import *

class enricheR(Biopax):
	def __init__(self, gene_protein:str, s_omics:str, project:str, s_project:str, root0:str,
				 case_list:List, has_age:bool=True, has_gender:bool=True, clone_objects:bool=False,
				 exp_normalization:bool=None, geneset_num:int=0, num_min_degs_for_ptw_enr:int=3,
				 tolerance_pathway_index:float=.15, s_pathw_enrichm_method:str='enricher',
				 root_data_aux:str='../../../data_aux/',
				 abs_lfc_cutoff_inf:float=0.40, fdr_ptw_cutoff_list:List=[],
				 num_of_genes_list:List=[3], lfc_list:List=[], fdr_list:List=[],
				 min_lfc_modulation:float=0.20, type_sat_ptw_index:str='linear_sat', saturation_lfc_index:float=5):

		super().__init__(gene_protein, s_omics, project, s_project, root0,
						 case_list, has_age, has_gender, clone_objects,
						 exp_normalization, geneset_num, num_min_degs_for_ptw_enr,
						 tolerance_pathway_index, s_pathw_enrichm_method,
						 root_data_aux, abs_lfc_cutoff_inf,
						 fdr_ptw_cutoff_list, num_of_genes_list, lfc_list, fdr_list, 
						 min_lfc_modulation, type_sat_ptw_index, saturation_lfc_index)


		self.pathid = None
		self.kgml, self.pathway_kgml = None, None
		self.dfgc = None

		self.old_pathway_cols = ['pathway', 'overlap', 'pathway_pval_cutoff', 'pathway_fdr_cutoff', 'Old P-value', 'Old Adjusted P-value', 'odds_ratio', 'combined_score', 'genes'],
		self.sel_pathway_cols = ['pathway', 'overlap', 'pathway_pval_cutoff', 'pathway_fdr_cutoff','odds_ratio', 'combined_score', 'genes']

		self.parse_gc_fields = ['ENTRY', 'SYMBOL', 'NAME', 'SEQUENCE', 'ORTHOLOGY', 'ORGANISM',
								'TYPE', 'REMARK', 'COMMENT', 'PATHWAY', 'NETWORK', 'BRITE', 'DBLINKS',
								'ELEMENT', 'DISEASE', 'POSITION', 'MOTIF', 'STRUCTURE', 'AASEQ', 'NTSEQ']

		self.dfk = None
		# if not self.get_kegg_pathays():
		# 	print("Problems getting KEGG pathway")



	def set_df_enr_ipath(self, ipath: int):
		try:
			pathway = self.df_enr.iloc[ipath].pathway
			genes   = self.df_enr.iloc[ipath].genes.split(';')
		except:
			pathway, genes = None, []

		self.pathway = pathway
		self.genes = genes

		return pathway, genes

	def rest_kegg_pathays(self):
		try:
			result = REST.kegg_list("pathway").read()
			dfk = to_df(result)
			dfk.columns = ['path_id', 'pathway']

			if dfk.empty:
				ret = False
			else:
				ret = pdwritecsv(dfk, fname, self.root_kegg)
		except:
			print("Error: accessing/reading REST.kegg_list('pathway')")
			dfk = None
			ret = False

		return ret, dfk

	def get_kegg_pathays(self):
		if self.dfk is not None: return True

		filefull = os.path.join(self.root_kegg, self.kegg_fname)

		if os.path.exists(filefull):
			dfk = pdreadcsv(self.kegg_fname, self.root_kegg)

			if dfk is None or dfk.empty:
				ret, dfk = self.rest_kegg_pathays()
			else:
				ret = True
		else:
			ret, dfk = self.rest_kegg_pathays()

		self.dfk = dfk
		return ret

	def set_kegg_ipath(self, ipath: int):
		try:
			path_id = self.dfk.iloc[ipath].path_id
			pathway = self.dfk.iloc[ipath].pathway
		except:
			pathway, path_id = None, []

		self.pathway = pathway
		self.path_id = path_id

		return path_id, pathway

	def set_kegg_path_id(self, path_id: int):
		try:
			pathway = self.dfk[self.dfk.path_id == path_id].iloc[0].pathway
		except:
			pathway = None

		self.path_id = path_id
		self.pathway = pathway

		return pathway


	def find_kegg_pathway_by_name(self, pathway, verbose=False):
		try:
			path_id = self.dfk[self.dfk.pathway == pathway].iloc[0].path_id
		except:
			path_id = None

		if path_id is None:
			print("Could not find pathway: '%s'"%(pathway))

		pathid = path_id.replace('path:', '')

		self.pathid = pathid
		self.path_id = path_id
		self.pathway = pathway

		return pathid


	def open_table(self, fname, case):
		self.case = case

		filefull = os.path.join(self.root_enrichment, fname)

		if not os.path.exists(filefull):
			print("File does not exists '%s'"%(filefull))
			return None

		df_enr = pdreadcsv(fname, self.root_enrichment)
		return df_enr

	'''--- get_id_list ----------'''
	def open_session_upload_symbols(self, deg_list, description=None):
		self.deg_list = deg_list
		self.shortId, self.userListId = None, None

		if not isinstance(deg_list, list):
			print("deg_list must be a list.")
			return None, None

		if deg_list == []:
			print("deg_list is empty!")
			return None, None

		if description is None:
			description = self.s_project + ' ' + self.case

		ENRICHR_URL = 'https://maayanlab.cloud/Enrichr/addList'
		genes_str = '\n'.join(deg_list)

		payload = {
			'list': (None, genes_str),
			'description': (None, description)
		}

		try:
			response = requests.post(ENRICHR_URL, files=payload)
			if not response.ok:
				print('Error analyzing gene list')
				return None, None

			data = json.loads(response.text)
			shortId = data['shortId']
			userListId = data['userListId']
		except:
			print("Problems in request.")
			return None, None

		self.shortId, self.userListId = shortId, userListId

		return shortId, userListId

	def is_ok_symbols(self, deg_list):
		ENRICHR_URL = 'https://maayanlab.cloud/Enrichr/view?userListId=%s'

		response = requests.get(ENRICHR_URL % self.userListId)
		if not response.ok:
			print('Error getting gene list')
			return False, []

		try:
			data = json.loads(response.text)
			genes = data['genes']
		except:
			print("Problems in request.")
			return False, []

		good_genes = [x for x in genes if x not in deg_list]
		return True, good_genes

	def open_reactome_enrichment_analysis(self, case, force=False, species='Homo sapiens', verbose=False):
		fname = 'taubate_%s_reactome_enrichment_results.csv'%(case)
		fname_sig = fname.replace('.csv', '') + '_sig.tsv'

		filefull = os.path.join(self.root_enrichment, fname_sig)
		if os.path.exists(filefull) and not force:
			df_enr = pdreadcsv(fname_sig, self.root_enrichment)
			self.df_enr = df_enr

			dicgenes = {df_enr.iloc[i].pathway_id: df_enr.iloc[i].submitted_entities_found.split(';') for i in range(len(df_enr)) }
			self.dicgenes = dicgenes
			return df_enr

		filefull = os.path.join(self.root_enrichment, fname)
		if not os.path.exists(filefull):
			print("Could not find: %s"%(filefull))
			self.df_enr = None
			self.dicgenes = None
			return None

		df_enr = pdreadcsv(fname, self.root_enrichment, sep=',')

		cols = ['pathway_id', 'pathway', 'found_entities', 'total_entities', 'interactors_found', 'interactors_total',
			   'entities_ratio', 'pathway_pval_cutoff', 'pathway_fdr_cutoff', 'reactions_found', 'ractions_total', 'reactions_ratio', 'species_identifier',
			   'species_name', 'submitted_entities_found', 'mapped_entities', 'submitted_entities_hit_interactor', 'interacts_with',
			   'found_reaction_identifiers']

		df_enr.columns = cols

		df_enr = df_enr[df_enr.species_name == species]
		df_enr = df_enr[ (df_enr.pathway_fdr_cutoff < self.pathway_fdr_cutoff) & (df_enr.num_of_genes >= self.num_of_genes_cutoff)]
		df_enr = df_enr.sort_values('pathway_fdr_cutoff', ascending=True)
		df_enr.index = np.arange(0, len(df_enr))

		cols.remove("species_identifier")
		cols.remove("species_name")
		df_enr = df_enr[cols]

		ret = pdwritecsv(df_enr, fname_sig, self.root_enrichment)
		self.df_enr = df_enr

		dicgenes = {df_enr.iloc[i].pathway_id: df_enr.iloc[i].submitted_entities_found.split(';') for i in range(len(df_enr)) }
		self.dicgenes = dicgenes
		return df_enr

	def calc_default_enrichment_analysis(self, geneset_num_list:List=[0, 1, 2, 4, 5, 7], force:bool=False, verbose:bool=False):

		abs_lfc_cutoff = 1.
		fdr_lfc_cutoff = 0.05
		pathway_fdr_cutoff = 0.05

		for case in self.case_list:
			icount=-1

			for geneset_num in geneset_num_list:
				self.set_db(geneset_num)

				if verbose: print(f">>> {case}, geneset = {self.geneset_lib}, lfc cutoff = {self.abs_lfc_cutoff} and fdr cutoff = {self.fdr_lfc_cutoff}")

				ret, degs, degs_ensembl, dflfc = \
							self.open_case_params(case, abs_lfc_cutoff=abs_lfc_cutoff, fdr_lfc_cutoff=fdr_lfc_cutoff, \
												  pathway_fdr_cutoff=pathway_fdr_cutoff, verbose=verbose)

				self.calc_EA_dataset_symbol(degs_ensembl, default=True, force=force, verbose=verbose)


	def calc_all_enrichment_analysis(self, geneset_num_list:List=[0, 1, 2, 4, 5, 7], method:str='spearman', force:bool=False, verbose:bool=False):

		dfsim = self.open_simulation_table()
		if dfsim.empty:
			print(f"Error: no previous simulation.")
			raise Exception("Stop: calc_all_enrichment_analysis()")

		for case in self.case_list:
			icount=-1

			'''
				open_case_simple:
					set case
					open dflfc_ori
			'''
			if not self.open_case_simple(case):
				print(f"Problems for case='{case}'")
				continue

			''' get filtered fdr's written in json files '''
			df_fdr = self.open_fdr_lfc_correlation(case, self.abs_lfc_cutoff_inf)
			if df_fdr is None or df_fdr.empty:
				print(f"Problems with df_fdr json for case='{case}'")
				continue

			''' filter simulation df for case, df_fdr.fdr, abs_lfc_cutoff_inf, and num_min_degs_for_ptw_enr '''
			dfsim2 = dfsim[ (dfsim.normalization == self.normalization) & (dfsim.case == case) &
							(dfsim.fdr_lfc_cutoff.isin(df_fdr.fdr) ) &
							(dfsim.abs_lfc_cutoff >= self.abs_lfc_cutoff_inf) &
							(dfsim.n_degs >= self.num_min_degs_for_ptw_enr)]

			if dfsim2.empty:
				print(f"No previous simulation for case='{case} with num of {self.s_deg_dap}s >= {self.num_min_degs_for_ptw_enr}'")
				continue

			for geneset_num in geneset_num_list:
				self.set_db(geneset_num)

				for i in range(len(dfsim2)):
					icount += 1

					row = dfsim2.iloc[i]
					abs_lfc_cutoff = row.abs_lfc_cutoff
					fdr_lfc_cutoff = row.fdr_lfc_cutoff

					'''
						if df_fdr has no correlation, than only calculate enrichment analysis for abs_lfc_cutoff == 1
					'''
					df_fdr2 = df_fdr[ (df_fdr.case == case) & 
									  (df_fdr.fdr  == fdr_lfc_cutoff) &
									  (df_fdr.method == method) ]

					if df_fdr2.empty:
						print(f"Could not find df_fdr for {case} fdr {fdr:.2f} and method={method}")
						continue

					if pd.isnull(df_fdr2.iloc[0]['corr']):
						if abs_lfc_cutoff != 1:
							continue

					if verbose: print(f">>> {case}, geneset = {self.geneset_lib}, lfc cutoff = {abs_lfc_cutoff} and fdr cutoff = {fdr_lfc_cutoff}")

					if isinstance(row.degs_ensembl, list):
						degs_ensembl2 = row.degs_ensembl
					else:
						degs_ensembl2 = eval(row.degs_ensembl)

					degs2, degs_ensembl, dflfc = self.list_of_degs_set_params(abs_lfc_cutoff, fdr_lfc_cutoff, verbose=False)


					if len(degs_ensembl) != len(degs_ensembl2):
						print(f"Error: mismatch len for {self.s_deg_dap}s for {case}, lfc cutoff = {abs_lfc_cutoff} and fdr cutoff = {fdr_lfc_cutoff} -> {len(degs_ensembl)} pre calc != {len(degs_ensembl2)} calc now")
						continue

					# if i > 10:break
					self.calc_EA_dataset_symbol(degs_ensembl, default=False, force=force, verbose=verbose)
					if icount%50==0:
						print(case, len(degs_ensembl), self.abs_lfc_cutoff, self.fdr_lfc_cutoff)
						self.echo_degs()
						print("")
						self.echo_enriched_pathways()
						print("\n")


	def calc_random_EA_dataset_symbol(self, i:int, deg_list:List, verbose:bool=False):
		'''
		Calc Enerichment Analysis given a dataset and a DEG list.
		Saven in enrichment_random dir
		allway cut and save the default cutoffs
		save as many send i - to perform statistics
		'''
		self.df_enr0 = None
		self.df_enr = None

		if len(deg_list) < self.num_min_degs_for_ptw_enr:
			print(f"Warning: Random {i}, please send a deg_list with >= {num_min_degs_for_ptw_enr} {self.s_deg_dap}s")
			return

		self.deg_list = deg_list
		df_enr0 = self.calc_get_enriched_pathway(deg_list, return_value=True, verbose=verbose, i=i)
		self.df_enr0 = df_enr0

		if df_enr0 is None or df_enr0.empty:
			return

		df_enr = self.df_enr0[(self.df_enr0.fdr  < self.pathway_fdr_cutoff) &
							  (self.df_enr0.pval < self.pathway_pval_cutoff) &
							  (self.df_enr0.num_of_genes >= self.num_of_genes_cutoff)].copy()

		if verbose: print("<<<", pathway_fdr_cutoff, pathway_pval_cutoff, num_of_genes_cutoff, len(df_enr))

		if df_enr.empty:
			return

		df_enr = df_enr.sort_values(['fdr', 'num_of_genes'], ascending=[True, False])
		df_enr.index = np.arange(0, len(df_enr))
		self.df_enr = df_enr

		fname, fname_cutoff = self.set_enrichment_name()
		fname_cutoff = fname_cutoff.replace('.tsv', f'_random_{i}.tsv')
		ret = pdwritecsv(df_enr, fname_cutoff, self.root_enrich_random, verbose=verbose)

		return

	def calc_EA_dataset_symbol(self, deg_list:List, calc_many_sig:bool=True, default:bool=False, force:bool=False, verbose:bool=False):
		'''
		Calc Enerichment Analysis given a dataset and a DEG list.
		if calc_many_sig: calc the fdr cutoffs = [0.05, 0.75]
		'''

		if len(deg_list) < self.num_min_degs_for_ptw_enr:
			if verbose: print(f"Please send a deg_list with >= {num_min_degs_for_ptw_enr} {self.s_deg_dap}s")
			return

		self.deg_list = deg_list
		df_enr0 = self.calc_get_enriched_pathway(deg_list, return_value=True, force=force, verbose=verbose)
		self.df_enr0 = df_enr0

		if (calc_many_sig or default) and df_enr0 is not None and not df_enr0.empty:
			self.calc_many_sig_enrich_pathways(default=default, force=force, verbose=verbose)

		return

	def calc_get_enriched_pathway(self, deg_list:List, return_value:bool=True,
								  force:bool=False, verbose:bool=False, i:int=None) -> pd.DataFrame:
		'''
		Calc enrichment analysis -> send call to Enrichr Web Service
		if i is None:
			force --> if alrealdy exists read table if return_value else return None
		else:
			force has no meaning
			always send a call to the Web Service
		'''

		self.df_enr = None
		self.all_enr_degs, self.enr_found_degs, self.enr_not_found_degs = [], [], []

		fname, fname_cutoff = self.set_enrichment_name()

		if i is None:
			root_enrichment = self.root_enrichment
			filefull = os.path.join(root_enrichment, fname)

			if os.path.exists(filefull) and not force:  #  and not calc_many_sig
				if not return_value:
					return None

				return pdreadcsv(fname, self.root_enrichment)
		else:
			root_enrichment = self.root_enrich_random
			fname = fname.replace(".tsv", f"_random_{i}.tsv")
			filefull = os.path.join(root_enrichment, fname)

		shortId, userListId = self.open_session_upload_symbols(deg_list)

		if shortId is None or userListId is None:
			print(f"Problems in open_session_upload_symbols().")
			return None

		# print(f"Enrichr web service: for '{filefull}'")
		ENRICHR_URL = 'https://maayanlab.cloud/Enrichr/enrich'
		query_string = '?userListId=%s&backgroundType=%s'

		try:
			response = requests.get(
				ENRICHR_URL + query_string%(userListId, self.geneset_lib)
			 )
		except:
			print(f"Problems in request geneset {self.geneset_lib} and userListId {str(userListId)}.")
			return None

		if not response.ok:
			print('Error fetching enrichment results: %s in %s'%(userListId, self.geneset_lib))
			return None

		try:
			data = json.loads(response.text)
		except:
			print('Error getting data')
			return None

		dic = {}

		try:
			mats = data[self.geneset_lib]

			if len(mats) == 0:
				print("No results")
				return None

			for mat in mats:
				# print(mat)
				count = mat[0]
				desc  = mat[1]
				pval2  = mat[2]
				odds_ratio = mat[3]
				combined_score = mat[4]
				genes  = mat[5]
				fdr2  = mat[6]
				unk0  = mat[7]
				unk1  = mat[8]

				dic[count] ={}
				dic2 = dic[count]

				if "Reactome" in self.geneset_lib:
					# Neuronal System R-HSA-112316
					mat = desc.split(' ')
					pathway = " ".join(mat[:-1])
					_id  = mat[-1]
					dic2['pathway'] = pathway
					dic2['pathway_id'] = _id
				else:
					dic2['pathway'] = desc

				dic2['pval'] = pval2
				dic2['fdr'] = fdr2
				dic2['odds_ratio'] = odds_ratio
				dic2['combined_score'] = combined_score
				dic2['genes'] = genes
				dic2['num_of_genes'] = len(genes)

		except:
			print('Error Enrichr response')
			return None

		if len(dic) == 0:
			return None

		df_enr = pd.DataFrame(dic).T

		df_enr = df_enr.sort_values(['fdr', 'num_of_genes'], ascending=[True, False])
		df_enr.index = np.arange(0, len(df_enr))
		ret = pdwritecsv(df_enr, fname, root_enrichment, verbose=verbose)

		return df_enr


	def calc_many_sig_enrich_pathways(self, pathway_pval_cutoff:float=0.05, default:bool=False, force:bool=False, verbose:bool=False):

		if default:
			fdr_ptw_cutoff_list = [0.05]
		else:
			fdr_ptw_cutoff_list = self.fdr_ptw_cutoff_list

		for pathway_fdr_cutoff in fdr_ptw_cutoff_list:
			for num_of_genes_cutoff in self.num_of_genes_list:

				''' to get the correct fname_cutoff '''
				self.set_pathway_cutoff_params(pathway_fdr_cutoff, pathway_pval_cutoff, num_of_genes_cutoff, verbose=verbose)

				fname, fname_cutoff = self.set_enrichment_name()
				filefull = os.path.join(self.root_enrichment, fname_cutoff)

				if os.path.exists(filefull) and not force:
					continue

				df_enr = self.df_enr0[(self.df_enr0.fdr < pathway_fdr_cutoff) &
									  (self.df_enr0.pval < pathway_pval_cutoff) &
									  (self.df_enr0.num_of_genes >= num_of_genes_cutoff)].copy()

				if verbose: print("<<<", pathway_fdr_cutoff, pathway_pval_cutoff, num_of_genes_cutoff, len(df_enr))

				if df_enr.empty:
					continue

				df_enr.index = np.arange(0, len(df_enr))
				ret = pdwritecsv(df_enr, fname_cutoff, self.root_enrichment, verbose=verbose)

	''' old: set_enriched_pathway_line '''
	def get_enriched_pathway_line(self, i_line):
		'''
			assign:
				self.genes_in_pathway, self.pathway, self.pathway_id
		'''
		if not isinstance(self.df_enr, pd.DataFrame) or self.df_enr.empty:
			print("No self.df_enr was found or empty")
			return None

		try:
			row = self.df_enr.iloc[i_line]
		except:
			print("df_enr has not this line.")
			return None

		genes = row.genes
		if isinstance(genes, str):
			genes = eval(genes)

		self.genes_in_pathway = genes
		self.pathway = row.pathway
		try:
			self.pathway_id = row.pathway_id
		except:
			self.pathway_id = None

		return row

	def open_db_pathway(self, force=False):
		if self.geneset_lib == 'BioPlanet_2019':
			return self.open_bioplanet_pathway(force=force)
		if self.geneset_lib == 'KEGG_2021_Human':
			return self.open_kegg_pathway(force=force)
		if self.geneset_lib == 'Reactome_2022':
			return self.open_reactome_pathway(force=force)

		print("DB %s is not developed")
		return None

	def open_reactome_pathway(self, species='Homo sapiens', force=False):
		fname = 'ReactomePathways.tsv'
		filefull = os.path.join(self.root_bioplanet, fname)

		if not os.path.exists(filefull) and not force:
			print("Could not found: %s"%(filefull))
			return None

		df_enr = pdreadcsv(fname, self.root_bioplanet)
		df_enr = df_enr[df_enr.species == species]
		df_enr.index = np.arange(0, len(df_enr))

		self.df_enr = df_enr
		return df_enr

	def open_bioplanet_pathway(self, force=False):
		fname = 'pathway.tsv'
		filefull = os.path.join(self.root_bioplanet, fname)

		if os.path.exists(filefull) and not force:
			df_enr = pdreadcsv(fname, self.root_bioplanet)
			df_enr.columns = ['pathway_id', 'pathway', 'gene_id', 'symbol']
			self.df_enr = df_enr
			return df_enr

		df_enr = pdreadcsv('pathway.csv', self.root_bioplanet, sep=',')
		fields = ['pathway_id', 'pathway', 'gene_id', 'symbol']
		df_enr.columns = [fields]
		df_enr.index = np.arange(0, len(df_enr))

		df_enr = df_enr.drop_duplicates()
		df_enr.index = np.arange(0, len(df_enr))
		df_enr.columns = ['pathway_id', 'pathway', 'gene_id', 'symbol']

		ret = pdwritecsv(df_enr, fname, self.root_bioplanet)
		self.df_enr = df_enr

		return df_enr

	def open_bioplanet_category(self, force=False):
		cols = ['pathway_id', 'pathway', 'category']

		fname = 'pathway-category.tsv'
		filefull = os.path.join(self.root_bioplanet, fname)

		if os.path.exists(filefull) and not force:
			dfbiop_cat = pdreadcsv(fname, self.root_bioplanet)
			dfbiop_cat.columns = cols
			self.dfbiop_cat = dfbiop_cat
			return dfbiop_cat

		dfbiop_cat = pdreadcsv('pathway-category.csv', self.root_bioplanet, sep=',')

		dfbiop_cat.columns = cols
		dfbiop_cat.index = np.arange(0, len(dfbiop_cat))

		dfbiop_cat = dfbiop_cat.drop_duplicates()
		dfbiop_cat.index = np.arange(0, len(dfbiop_cat))
		ret = pdwritecsv(dfbiop_cat, fname, self.root_bioplanet)

		self.dfbiop_cat = dfbiop_cat

		return dfbiop_cat

	def open_bioplanet_disease(self, force=False):

		fname = 'pathway-disease-mapping.tsv'
		filefull = os.path.join(self.root_bioplanet, fname)

		if os.path.exists(filefull) and not force:
			dfdisease = pdreadcsv(fname, self.root_bioplanet)
			self.dfdisease = dfdisease
			return dfdisease

		dfdisease = pdreadcsv(fname, self.root_bioplanet)
		cols = ['pathway_id', 'pathway', 'geneid', 'symbol', 'mim_id', 'disease', 'symbols', 'chr_location', 'phenotype']
		dfdisease.columns = cols
		dfdisease.index = np.arange(0, len(dfdisease))

		dfdisease = dfdisease.drop_duplicates()
		dfdisease.index = np.arange(0, len(dfdisease))
		ret = pdwritecsv(dfdisease, fname, self.root_bioplanet)

		self.dfdisease = dfdisease

		return dfdisease

	# A bit of code that will help us display the PDF output
	def PDF(self, filename):
		return HTML('<iframe src=%s width=700 height=350></iframe>' % filename)

	# Some code to return a Pandas dataframe, given tabular text
	def to_df(self, result):
		return pd.read_table(io.StringIO(result), header=None)

	def open_kegg_pathway(self, force=False):

		filefull = os.path.join(self.root_kegg, self.fname_kegg_pathways)

		if os.path.exists(filefull) and not force:
			df_enr = pdreadcsv(self.fname_kegg_pathways, self.root_kegg)
			self.df_enr = df_enr
			return df_enr

		result = REST.kegg_list("pathway").read()
		df_enr = self.to_df(result)
		df_enr.columns = ['path_id', 'pathway']
		ret = pdwritecsv(df_enr, self.fname_kegg_pathways, self.root_kegg)

		self.df_enr = df_enr
		return df_enr

	def get_kegg_pathway_image(self):
		try:
			return REST.kegg_get(self.pathid, "image").read()
		except:
			print("Error: problems with internet?")
			return None

	def get_kegg_kgml(self, verbose=False, force=False):

		pathid_hsa = self.pathid.replace('map', 'hsa')
		self.pathid_hsa = pathid_hsa

		if self.pathid is None or self.pathid == '':
			print("pathid is not defined '%s'"%(str(self.pathid_hsa)))
			self.kgml, self.pathway_kgml = None, None
			return False

		fname	 = "kgml_%s.xml"%(title_replace(self.pathway))
		filefull  = os.path.join(self.root_enrichment, fname)

		if os.path.exists(filefull) and not force:
			'''
			try:
				kgml = kegg_get(pathid_hsa, "kgml").read()
			except:
				print("Error: problems with internet?")
				self.kgml, self.pathway_kgml = None, None
				return False
			'''

			self.kgml = read_txt(fname, self.root_enrichment)
			self.kgml = "\n".join(self.kgml)
			self.pathway_kgml = KGML_parser.read(self.kgml)

			return True

		try:
			self.kgml = kegg_get(pathid_hsa, "kgml").read()
			self.pathway_kgml = KGML_parser.read(self.kgml)
			ret = True
		except:
			print("Could not find KGML for '%s'"%(pathid_hsa))
			self.kgml, self.pathway_kgml = None, None
			ret = False


		ret = write_txt(self.kgml, fname, self.root_enrichment, verbose=True)

		return ret

	def get_kegg_human_gene_annotation(self, gene_id, force=False, verbose=False):
		''' like "hsa:5624" '''
		kegg_id = gene_id
		self.kegg_id = kegg_id
		self.dfgc = None
		filefull = os.path.join(self.root_kegg, self.fname_kegg_gene_comp)

		if os.path.exists(filefull) and not force:
			df = pdreadcsv(self.fname_kegg_gene_comp, self.root_kegg)
			dfgc = df[df.kegg_id == kegg_id].copy()

			if not dfgc.empty:
				dfgc.index = np.arange(0, len(dfgc))
				self.dfgc = dfgc
				return True

		if 'hsa' not in gene_id:
			self.gene_id = None
			print("This is not a human gene_id (without 'hsa'): '%s'"%(gene_id))
			return None

		try:
			result = REST.kegg_get(gene_id).read()
			self.kegg_id = gene_id
			self.result = result
			ret = self.parse_kegg_gene_compound_annotation(gene_id, result, force=force, verbose=verbose)
		except:
			print("Could not get REST.kegg_get(gene = '%s')"%(gene_id))
			self.kegg_id = None
			self.result = None
			ret = False

		return ret

	def get_kegg_compound_annotation(self, compound_id, force=False, verbose=False):

		kegg_id = compound_id
		self.kegg_id = kegg_id
		self.dfgc = None
		filefull = os.path.join(self.root_kegg, self.fname_kegg_gene_comp)

		if os.path.exists(filefull) and not force:
			df = pdreadcsv(self.fname_kegg_gene_comp, self.root_kegg)
			dfgc = df[df.kegg_id == kegg_id].copy()

			if not dfgc.empty:
				dfgc.index = np.arange(0, len(dfgc))
				self.dfgc = dfgc
				return True

		try:
			result = REST.kegg_get(compound_id).read()
			self.kegg_id = compound_id
			self.result = result
			ret = self.parse_kegg_gene_compound_annotation(compound_id, result, force=force, verbose=verbose)

		except:
			print("Could not get REST.kegg_get(compound = '%s')"%(compound_id))
			self.kegg_id = None
			self.result = None
			ret = False

		return ret

	def parse_kegg_gene_compound_annotation(self, kegg_id, result, force=False, verbose=False):

		if not isinstance(result, str):
			print("Parse result must be a string")
			return None

		if len(result) <= 10:
			print("Parse result must exists: '%s'"%(result))
			self.dfgc = None
			return False

		filefull = os.path.join(self.root_kegg, self.fname_kegg_gene_comp)

		if os.path.exists(filefull):
			df = pdreadcsv(self.fname_kegg_gene_comp, self.root_kegg)
			dfgc = df[df.kegg_id != kegg_id].copy()
		else:
			dfgc = None

		comp_list = result.split('\n')

		i = -1; dic={}; stop = False; current=''
		for comp in comp_list:
			comp = comp.strip()

			terms = comp.split(' ')
			j = -1
			for term in terms:
				if term == '///':
					stop = True
					break

				j += 1
				if term == '': continue
				if j == 0:
					if term in self.parse_gc_fields:
						i += 1
						dic[i] = {}
						dic2 = dic[i]
						current = term
						dic2['term'] = term
						dic2['val'] = ''
						continue

				#print("**", i, ')', dic2['term'])
				mat = []
				for term2 in terms[j:]:
					if term2 != '': mat.append(term2)

				stri = " ".join(mat)

				if dic2['val'] == '':
					dic2['val'] = stri
					# print("### start", i, dic2['term'], dic2['val'])
				else:
					dic2['val'] += ' \n' + stri
					# print("### mergin", i, dic2['term'], dic2['val'])

				# print("$$ %d) '%s'"%(i, stri))

				break


			if stop: break

		df = pd.DataFrame(dic).T
		df['kegg_id'] = kegg_id
		fields = ['kegg_id', 'term', 'val']
		df = df[fields]

		if dfgc is None:
			dfgc = df
		else:
			dfgc = pd.concat([dfgc, df])

		dfgc = dfgc.sort_values(['kegg_id', 'term'])
		dfgc.index = np.arange(0, len(dfgc))
		ret = pdwritecsv(dfgc, self.fname_kegg_gene_comp, self.root_kegg)

		dfgc = df[df.kegg_id == kegg_id].copy()
		dfgc.index = np.arange(0, len(dfgc))
		self.dfgc = dfgc

		return True

	def find_kegg_gene_name_symbol(self, gene_id, force=False, verbose=False):

		ret = self.get_kegg_human_gene_annotation(gene_id, force=force, verbose=verbose)

		if ret:
			dfq = self.dfgc[ self.dfgc.kegg_id == gene_id ]
			ret = not dfq.empty

		if ret:
			gene_name	 = "; ".join( dfq[dfq.term == 'NAME'].val)
			gene_synonyms = "; ".join( dfq[dfq.term == 'SYMBOL'].val)
			gene_symbol   = gene_synonyms.split('; ')[0]
		else:
			gene_name, gene_symbol, gene_synonyms = None, None, None

		return gene_name, gene_symbol, gene_synonyms

	def find_kegg_compound_name_type(self, compound_id, force=False, verbose=False):

		ret = self.get_kegg_compound_annotation(compound_id, force=force, verbose=verbose)

		if ret:
			dfq = self.dfgc[ self.dfgc.kegg_id == compound_id ]
			ret = not dfq.empty

		if ret:
			compound_name = "; ".join( dfq[dfq.term == 'NAME'].val)
			compound_type = "; ".join( dfq[dfq.term == 'TYPE'].val)

			compound_name = compound_name.replace("\n", ", ")
			compound_type = compound_type.replace("\n", ", ")
		else:
			compound_name, compound_type = None, None

		return compound_name, compound_type


	def find_kegg_gene_diseases(self, gene_id_list, force=False, verbose=False):

		if not isinstance(gene_id_list, list):
			print("Error: gene_id_list must be a list")
			return None

		diseases = []
		for gene_id in gene_id_list:
			ret = self.get_kegg_human_gene_annotation(gene_id, force=force, verbose=verbose)
			if not ret: continue

			dfd = self.dfgc[ (self.dfgc.kegg_id == gene_id) & (self.dfgc.term == 'DISEASE')]
			if not dfd.empty:
				diseases.append(dfd.val)

		return diseases

		''' Color gradient: blue --> white -> red
			fade (linear interpolate) from color c1, c2, c3 using mix = [0,1]
		'''
	def three_colorFader(self, c1='blue',c2='white',c3='red',mix=0):
		c1=np.array(mpl.colors.to_rgb(c1))
		c2=np.array(mpl.colors.to_rgb(c2))
		c3=np.array(mpl.colors.to_rgb(c3))

		if mix <= .5:
			color = mpl.colors.to_hex((1-2*mix)*c1 + 2*mix*c2)
		else:
			mix -= 0.5
			color = mpl.colors.to_hex((1-2*mix)*c2 + 2*mix*c3)

		return color


	'''
	https://community.plotly.com/t/how-to-draw-ellipse-on-top-of-scatter-plot/36576
	@mutia'''
	def ellipse(self, x_center=0, y_center=0, ax1 = [1, 0],  ax2 = [0,1], a=1, b =1,  N=100):
		'''
		x_center, y_center the coordinates of ellipse center
		ax1 ax2 two orthonormal vectors representing the ellipse axis directions
		a, b the ellipse parameters
		'''
		if np.linalg.norm(ax1) != 1 or np.linalg.norm(ax2) != 1:
			raise ValueError('ax1, ax2 must be unit vectors')
		if  np.abs(np.dot(ax1, ax2)) > 1e-06:
			raise ValueError('ax1, ax2 must be orthogonal vectors')
		t = np.linspace(0, 2*pi, N)
		#ellipse parameterization with respect to a system of axes of directions a1, a2
		xs = a * cos(t)
		ys = b * sin(t)
		#rotation matrix
		R = np.array([ax1, ax2]).T
		# coordinate of the  ellipse points with respect to the system of axes [1, 0], [0,1] with origin (0,0)
		xp, yp = np.dot(R, [xs, ys])
		x = xp + x_center
		y = yp + y_center
		return x, y


	def return_gene_compound(self, node_names, node_type):

		s_gene_name, s_gene_symbol, s_gene_synonyms, s_compound_name, s_compound_type = '', '', '', '', ''

		# print(node_type, type(node_names), node_names)
		if node_type != 'gene' and node_type != 'compound':
			return s_gene_name, s_gene_symbol, s_gene_synonyms, s_compound_name, s_compound_type

		for kegg_id in node_names:
			gene_name, gene_symbol, gene_synonyms, compound_name, compound_type = None, None, None, None, None

			if node_type == 'compound':
				kegg_id = kegg_id[4:]  # without 'cpd:'
				compound_name, compound_type = self.find_kegg_compound_name_type(kegg_id)

				if s_compound_name == '':
					s_compound_name = compound_name
					s_compound_type = compound_type
				else:
					s_compound_name += '||' + compound_name
					s_compound_type += '||' + compound_type

			else:
				gene_name, gene_symbol, gene_synonyms = self.find_kegg_gene_name_symbol(kegg_id)

				if s_gene_name == '':
					s_gene_name = gene_name
					s_gene_symbol = gene_symbol
					s_gene_synonyms = gene_synonyms
				else:
					s_gene_name += '||' + gene_name
					s_gene_symbol += '||' + gene_symbol
					s_gene_synonyms += '||' + gene_synonyms


		return s_gene_name, s_gene_symbol, s_gene_synonyms, s_compound_name, s_compound_type


	# @enforce.runtime_validation
	def define_KEGG_pathway(self, ipath: int, verbose:bool = False) -> bool:
		pathway, genes = self.set_df_enr_ipath(ipath)

		if pathway is None:
			self.pathway = None
			self.pathway_genes = None
			self.pathid = None
			return False

		if verbose:
			print("pathway:", pathway)
			print("genes:",   ", ".join(genes))

		self.pathway = pathway
		self.pathway_genes = genes

		pathid = self.find_kegg_pathway_by_name(pathway)
		self.pathid = pathid

		if verbose:
			print(self.pathid, pathway)

		return True


	def do_ellipse(self, size_x=1, size_y=1, x_center=0, y_center=0,color='red', fillcolor='pink', angle=0):

		# obj_type = dfcolor.iloc[i].obj_type
		# print(">> label", i, obj_type, label, color, fillcolor, text)
		ax1 = [cos(angle), sin(angle)]
		ax2 = [-sin(angle),cos(angle)]

		x, y = self.ellipse(x_center=x_center, y_center=y_center,
					   ax1=ax1, ax2=ax2, a=size_x, b=size_y)

		return go.Scatter(x=x, y=y, line_color=color, fill='tozeroy',fillcolor=fillcolor,
						  line_width=2, mode='lines', hoverinfo='none',
						 )

	def do_rectangle(self, x, y, width, height, line_color, fillcolor):

		del_x = width/2
		del_y = height/2

		x0 = x-del_x
		y0 = y-del_y

		x1 = x0+width
		y1 = y0+height

		x_list = [x0, x1, x1, x0, x0]
		y_list = [y0, y0, y1, y1, y0]

		return go.Scatter(x=x_list, y=y_list, showlegend=False,
							 fill='toself', fillcolor=fillcolor, line_color=line_color,
							 mode='lines', hoverinfo='none', )


	def do_nodes(self, font_size=10, font_color='navy', font_family='sans serif'):

		label_list, text_list, x_list, y_list, bkg_color_list, opacity_list = \
		self.label_list, self.text_list, self.x_list, self.y_list, self.bkg_color_list, self.opacity_list

		good_is = [i for i in range(len(label_list)) if opacity_list[i] == 1.0]
		good_is = [i for i in good_is if not label_list[i].startswith('TITLE:')]

		labels = [label_list[i] for i in good_is]
		texts  = [text_list[i]  for i in good_is]
		colors = [bkg_color_list[i] for i in good_is]

		xs = [x_list[i] for i in good_is]
		ys = [y_list[i] for i in good_is]

		return go.Scatter(x=xs, y=ys, showlegend=False,
						  marker=dict(color=colors,
									  size=10,
									  line=dict(
											color= colors,
											width=0.1
									  )
									),
						  mode='markers+text',
						  opacity=1,
						  hoverinfo='text',
						  text = labels,
						  hovertemplate = texts,
						  textposition='middle center',
						  textfont=dict(family=font_family,
										size=font_size,
										color=font_color ),
						 )



	def is_close_x(self, x0, x1):
		return np.abs(x1-x0) <= self.w_close

	def is_close_y(self, y0, y1):
		return np.abs(y1-y0) <= self.h_close


	def calc_col_name(self, i):
		group = self.dfpa.iloc[i].group.strip()
		num   = int(self.dfpa.iloc[i].group_num)

		if num < 10:
			num = '0'+str(num)
		else:
			num = str(num)

		return group + '_' + num

	def open_proteomics_table_log2(self, fname = 'pacientes_x_amostra_proteomica.tsv', lim_elder=55, lim_obese=30):
		fnamefull = os.path.join(self.root_data, fname)
		if not os.path.exists(fnamefull):
			print("File not found: '%s'"%(fnamefull))
			return None

		dfpa = pdreadcsv(fname, self.root_data)
		dfpa.columns = ['group', 'group_num', '_id', 'sex', 'age', 'spec_id', 'bmi']
		dfpa['pac_id'] = [x.split('_')[1] for x in dfpa._id]
		dfpa['gender'] = [0 if x == 'F' else 1 for x in dfpa.sex]
		dfpa['elder']  = [0 if x < lim_elder else 1 for x in dfpa.age]
		dfpa['obese']  = [0 if x < lim_obese else 1 for x in dfpa.bmi]

		self.dfpa = dfpa
		dfpa['group_num'] = [ self.calc_col_name(i) for i in range(len(dfpa))]
		self.dfpa = dfpa

		return dfpa


	def open_log2_table_limma(self, want_ctrl=False):

		fname = 'Log2_limma_Ctrl_x_%s.tsv'%(self.case)
		fnamefull = os.path.join(self.root_data, fname)
		if not os.path.exists(fnamefull):
			print("File not found: '%s'"%(fnamefull))
			return None

		dfraw = pdreadcsv(fname, self.root_data)

		if want_ctrl:
			cols = [x.replace('C_','ctrl_') if x.startswith('C_') else x for x in dfraw.columns]
			dfraw.columns = cols
			case2 = 'ctrl'
		else:
			case2 = self.case

		cols = [x for x in list(dfraw.columns) if case2 in x]

		dfraw = dfraw[ ['Majority.protein.IDs'] + cols ]
		dfraw.columns =  ['uniprot_id'] + cols
		self.dfraw = dfraw
		return dfraw

	def select_patients(self, _all=True, only_obese=False, obese=1, only_elder=False, elder=1, want_ctrl=False):
		if want_ctrl:
			case2 = 'ctrl'
		else:
			case2 = self.case

		if _all:
			dfpa_sel = self.dfpa[(self.dfpa.group == case2) ]
			s_title = 'all samples'
		else:
			if only_obese:
				s_title = 'obese' if obese == 1 else 'not obese'
				dfpa_sel = self.dfpa[(self.dfpa.group == case2) & (self.dfpa.obese == obese)].copy()
			elif only_elder:
				s_title = 'elder' if elder == 1 else 'not elder'
				dfpa_sel = self.dfpa[(self.dfpa.group == case2) & (self.dfpa.elder == elder)].copy()
			else:
				s_title = 'obese' if obese == 1 else 'not obese'
				s_title += ' and ' + 'elder' if elder == 1 else 'not elder'
				dfpa_sel = self.dfpa[(self.dfpa.group == case2) &(self.dfpa.elder == elder) & (self.dfpa.obese==obese)].copy()

		self.dfpa_sel = dfpa_sel
		return dfpa_sel, s_title


	def clean_select_patients(self):
		cols = list(self.dfpa_sel.group_num)

		cols = [x for x in cols if x in self.dfraw.columns]

		dfraw_sel = self.dfraw[  ['uniprot_id'] + cols ].copy()
		dfraw_sel = dfraw_sel.dropna(axis=0)
		self.dfraw_sel = dfraw_sel

		return dfraw_sel


	def calc_anova_patients(self):
		cols = list(self.dfraw_sel.columns)
		cols = cols[1:]

		df2 = self.dfraw_sel[cols]
		ncols = len(cols)

		if ncols == 3:
			stat, pathway_pval_cutoff = f_oneway(df2.iloc[:,0], df2.iloc[:,1], df2.iloc[:,2])
		elif ncols == 4:
			stat, pathway_pval_cutoff = f_oneway(df2.iloc[:,0], df2.iloc[:,1], df2.iloc[:,2], df2.iloc[:,3])
		elif ncols == 5:
			stat, pathway_pval_cutoff = f_oneway(df2.iloc[:,0], df2.iloc[:,1], df2.iloc[:,2], df2.iloc[:,3], df2.iloc[:,4])
		elif ncols == 6:
			stat, pathway_pval_cutoff = f_oneway(df2.iloc[:,0], df2.iloc[:,1], df2.iloc[:,2], df2.iloc[:,3], df2.iloc[:,4], df2.iloc[:,5])
		elif ncols == 7:
			stat, pathway_pval_cutoff = f_oneway(df2.iloc[:,0], df2.iloc[:,1], df2.iloc[:,2], df2.iloc[:,3], df2.iloc[:,4], df2.iloc[:,5], df2.iloc[:,6])
		elif ncols == 8:
			stat, pathway_pval_cutoff = f_oneway(df2.iloc[:,0], df2.iloc[:,1], df2.iloc[:,2], df2.iloc[:,3], df2.iloc[:,4], df2.iloc[:,5], df2.iloc[:,6], df2.iloc[:,7])
		elif ncols == 9:
			stat, pathway_pval_cutoff = f_oneway(df2.iloc[:,0], df2.iloc[:,1], df2.iloc[:,2], df2.iloc[:,3], df2.iloc[:,4], df2.iloc[:,5], df2.iloc[:,6], df2.iloc[:,7], df2.iloc[:,8])
		elif ncols == 10:
			stat, pathway_pval_cutoff = f_oneway(df2.iloc[:,0], df2.iloc[:,1], df2.iloc[:,2], df2.iloc[:,3], df2.iloc[:,4], df2.iloc[:,5], df2.iloc[:,6], df2.iloc[:,7], df2.iloc[:,8], df2.iloc[:,9])
		elif ncols == 11:
			stat, pathway_pval_cutoff = f_oneway(df2.iloc[:,0], df2.iloc[:,1], df2.iloc[:,2], df2.iloc[:,3], df2.iloc[:,4], df2.iloc[:,5], df2.iloc[:,6], df2.iloc[:,7], df2.iloc[:,8], df2.iloc[:,9], df2.iloc[:,10])
		elif ncols == 12:
			stat, pathway_pval_cutoff = f_oneway(df2.iloc[:,0], df2.iloc[:,1], df2.iloc[:,2], df2.iloc[:,3], df2.iloc[:,4], df2.iloc[:,5], df2.iloc[:,6], df2.iloc[:,7], df2.iloc[:,8], df2.iloc[:,9], df2.iloc[:,10], df2.iloc[:,11])
		else:
			stat, pathway_pval_cutoff = ('error', 'wrong number of cols = %d'%(ncols))

		return stat, pathway_pval_cutoff


	def boxplot_patients(self, s_title, pathway_pval_cutoff):

		cols = list(self.dfraw_sel.columns)
		cols = cols[1:]

		df2 = self.dfraw_sel[cols]

		fig = plt.figure(figsize=(12,8))
		ax = plt.axes()
		plt.boxplot(df2, showmeans=True, meanline=True)

		title = 'Boxplot for %s\n%s - ANOVA = %.2e'%(self.case, s_title, pathway_pval_cutoff)
		plt.title(title)
		plt.xlabel('samples')
		plt.xticks(np.arange(1, len(cols)+1))
		ax.set_xticklabels(cols)
		plt.ylabel('log2(protein)')

		return fig

	def define_case(self, case, _all=True, only_obese=False, obese=1, only_elder=False, elder=1):
		self.case = case

		dfpa = self.open_proteomics_table_log2(fname = 'pacientes_x_amostra_proteomica.tsv', lim_elder=55, lim_obese=30)

		print("Opened, groups:", dfpa.group.unique())

		self.dfraw_ctrl, self.dfpa_sel_ctrl, self.dfraw_sel_ctrl = None, None, None
		self.dfraw_case, self.dfpa_sel_case, self.dfraw_sel_case = None, None, None

		want_ctrl = True
		dfraw = self.open_log2_table_limma(want_ctrl=want_ctrl)
		if dfraw is None: return False
		dfpa_sel, s_title = self.select_patients(_all=_all, only_obese=only_obese, obese=obese, only_elder=only_elder, elder=elder, want_ctrl=want_ctrl)
		if dfpa_sel is None: return False
		dfraw_sel = self.clean_select_patients()
		if dfraw_sel is None or dfraw_sel.empty: return False

		self.dfraw_ctrl = dfraw
		self.dfpa_sel_ctrl = dfpa_sel
		self.dfraw_sel_ctrl = dfraw_sel

		want_ctrl = False
		dfraw = self.open_log2_table_limma(want_ctrl=want_ctrl)
		if dfraw is None: return False
		dfpa_sel, s_title = self.select_patients(_all=_all, only_obese=only_obese, obese=obese, only_elder=only_elder, elder=elder, want_ctrl=want_ctrl)
		if dfpa_sel is None: return False
		dfraw_sel = self.clean_select_patients()
		if dfraw_sel is None or dfraw_sel.empty: return False

		self.dfraw_case = dfraw
		self.dfpa_sel_case = dfpa_sel
		self.dfraw_sel_case = dfraw_sel

		return True

	def calc_elder_data(self, case, elder=1):
		self.case = case
		self.elder = elder

		s_elder = 'elder' if elder == 1 else 'not_elder'
		self.s_complement = s_elder

		fname = 'new_taubate_LFC_%s_x_%s_%s.tsv'%(case, 'ctrl', s_elder)
		filefull = os.path.join(self.root_data, fname)

		if not os.path.exists(filefull):
			print("File does not exists '%s'"%(filefull))
			return None

		dfp = pdreadcsv(fname, self.root_data)

		dfn = pd.merge(dfp, self.df_uniprot, how='inner', on='uniID')
		print(len(dfp), len(dfn))
		dfn = dfn[(dfn.fdr < 0.05) & (np.abs(dfn.lfc) >= 1)]
		print(len(dfn))
		dfn = dfn.sort_values(['fdr', 'lfc'], ascending=[True, False])

		symbols = list(dfn.symbol)
		symbols.sort()

		return dfn, symbols

	def plot_venn2(self, lista1, lista2, names = ['elder', 'adult'],
				   title0 = "Comparing Proteins elder x adult - case %s in dataset %s",
				   verbose=True):

		set1 = set(lista1)
		set2 = set(lista2)

		inter_lista = list(set1.intersection(set2))
		inter_lista.sort()
		set1_only_lista = [x for x in set1 if x not in inter_lista]
		set1_only_lista.sort()
		set2_only_lista = [x for x in set2 if x not in inter_lista]
		set2_only_lista.sort()


		mat = get_venn_sections( (set1, set2) )

		vals = []; geneList =[]; index=[]
		dic = {}
		for i in range(len(mat)):
			genes = mat[i][1]
			index.append(mat[i][0])
			vals.append(len(mat[i][1]))
			geneList.append(mat[i][1])
			dic[mat[i][0]] = len(mat[i][1])

		if verbose:
			print(index)
			print(vals)
			print("")
			for key in dic.keys():
				print(key, dic[key])


		fig = plt.figure(figsize=(14,10))

		from matplotlib_venn import venn2

		v2 = venn2([set1, set2], set_labels = None, alpha=0.3)

		for i in range(len(mat)):
			# v2.get_patch_by_id(index[i]).set_color(i)
			v2.get_patch_by_id(index[i]).set_edgecolor('none')
			v2.get_label_by_id(index[i]).set_text('%s\n%d'%( defineClass(index[i], names), vals[i]))

			label = v2.get_label_by_id(index[i])
			label.set_fontsize(18)
			# label.set_family('arial')
			# label.set_x(label.get_position()[0] + 0.1)

		title = title0%(self.case, self.geneset_lib)
		plt.title(title, fontsize=20)

		fnamefig = title_replace(title)
		filefull = os.path.join(self.root_figure, fnamefig + '.png')
		plt.savefig(filefull, dpi=300, format='png', facecolor='white')


		stri = "inter_lista (%d): %s\n"%(len(inter_lista), "; ".join(inter_lista))
		#--- elder
		stri += "\nonly elder (%d): %s\n"%(len(set1_only_lista), "; ".join(set1_only_lista))
		#--- adult
		stri += "\nonly adults (%d): %s\n"%(len(set2_only_lista), "; ".join(set2_only_lista))

		write_txt(stri, fnamefig+'.txt', self.root_figure, verbose=verbose)

		return fig, stri


	def calc_enriched_pathways_random_genes(self, i:int, case:str, abs_lfc_cutoff_default:float=1,
											fdr_lfc_cutoff_default:float=0.05, pathway_fdr_cutoff_default:float=0.05,
											prompt_verbose:bool=False, verbose:bool=False) -> pd.DataFrame:
		'''
		calc DEGs with the default cutoff
		usually less than the best cutoff
		calc diff = n_degs_bca - n_degs_default
		fulfill the list of default degs with random diff not used genes
		calc the enriched analysis with the random fulfill gene list
		filter the enrichment analyis table with the same best cutoff
		calc in and not_in genes in pathway to perform the chi-square table
		'''
		ret, degs_default, degs_ensembl_default, dflfc_default = self.open_case_params(case, abs_lfc_cutoff=abs_lfc_cutoff_default,
																					   fdr_lfc_cutoff=fdr_lfc_cutoff_default)
		self.degs_in_pathways_default = self.degs_in_pathways
		self.degs_not_in_pathways_default = self.degs_not_in_pathways
		self.degs_default = self.degs

		''' get best cutoff parameters'''

		ret, degs_bca, degs_ensembl_bca, dflfc_bca = self.open_case(case)

		degs_in_pathways_bca	 = self.degs_in_pathways
		degs_not_in_pathways_bca = self.degs_not_in_pathways

		assert(degs_in_pathways_bca + degs_not_in_pathways_bca, degs_bca)

		self.degs_in_pathways_bca	 = degs_in_pathways_bca
		self.degs_not_in_pathways_bca = degs_not_in_pathways_bca
		self.degs_bca = degs_bca

		self.n_degs_in_pathways_bca	 = len(degs_in_pathways_bca)
		self.n_degs_not_in_pathways_bca = len(degs_not_in_pathways_bca)
		self.n_degs_bca = len(degs_bca)

		n_genes = self.n_degs
		n_diff = n_genes - len(self.degs_default)
		if prompt_verbose: print(f"Randomizing {n_diff} = {n_genes} - {len(self.degs_default)} default DEGs")

		if n_diff == 0:
			flag_ok = True
			print("There is the same number of DEGs BCA and default. No improvement")
		else:
			flag_ok = False

			if n_diff > 0:
				bca_has_more = True
			else:
				bca_has_more = False
				print("DEGs BCA is less than DEGs default! No improvement")

		if flag_ok or not bca_has_more:
			self.degs_in_pathways_random, self.degs_not_in_pathways_random = [], []
			self.n_degs_in_pathways_random, self.n_degs_not_in_pathways_random = 0, 0
			return None

		""" remove the default and get only symbols with known ensembl_id """
		dfa = self.dflfc_ori.copy()
		dfa = dfa[ (~pd.isnull(dfa.ensembl_id)) & (~pd.isnull(dfa.symbol)) & \
				   (~dfa.symbol.isin(self.degs_in_pathways_default)) &
				   (~dfa.symbol.isin(self.degs_default)) ]
		dfa.index = np.arange(0, len(dfa))

		rows = np.random.randint(0, len(dfa), n_diff)
		random_genes = list(dfa.iloc[rows].symbol)
		random_genes.sort()

		random_fulfill_genes = self.degs_default + random_genes
		if prompt_verbose: print(f"Calculating {len(random_fulfill_genes)} total default + random genes.")

		self.calc_random_EA_dataset_symbol(i, random_fulfill_genes, verbose=False)

		if self.df_enr is None or self.df_enr.empty:
			if prompt_verbose: print("No pathway was found.")
			df_enr = None
			self.degs_in_pathways_random, self.degs_not_in_pathways_random = [], random_fulfill_genes
			self.n_degs_in_pathways_random, self.n_degs_not_in_pathways_random  = 0, len(random_fulfill_genes)
		else:
			df_enr = self.df_enr
			if prompt_verbose: print(f"There are {len(df_enr)} enriched pathways.")

			all_enr_degs = []
			for i in range(len(df_enr)):
				genes = df_enr.iloc[i].genes
				if isinstance(genes, str):
					genes = eval(genes)
				all_enr_degs += genes

			all_enr_degs = list(np.unique(all_enr_degs))

			degs_in_pathways_random	    = [x for x in random_fulfill_genes if x	 in all_enr_degs]
			degs_not_in_pathways_random = [x for x in random_fulfill_genes if x not in all_enr_degs]

			self.degs_in_pathways_random = degs_in_pathways_random
			self.degs_not_in_pathways_random = degs_not_in_pathways_random

			self.n_degs_in_pathways_random = len(degs_in_pathways_random)
			self.n_degs_not_in_pathways_random = len(degs_not_in_pathways_random)

		return df_enr


	def build_matrix_calc_chi_square(self, n_degs_in_pathways_bca:int, n_degs_not_in_pathways_bca:int,
									 n_degs_in_pathways_random:int, n_degs_not_in_pathways_random:int):

		mat = [ [n_degs_in_pathways_bca, n_degs_not_in_pathways_bca],
				[n_degs_in_pathways_random, n_degs_not_in_pathways_random],
			  ]
		dfmat = pd.DataFrame(mat)
		dfmat.index = ['BCA', 'random']
		dfmat.columns = ['degs_in', 'degs_out']

		try:
			ret_chi, dof, stat, pvalue, stri_stat = chisquare_2by2(dfmat)
		except:
			print("Something wrong with the entry table: perhaps a line is Zero.")
			ret_chi, dof, stat, pvalue, stri_stat = "Error", -1, -1, "Error"

		return dfmat, ret_chi, dof, stat, pvalue, stri_stat

	def run_n_simulations(self, n_sim:int, case:str, abs_lfc_cutoff_default:float=1,
						  fdr_lfc_cutoff_default:float=0.05, pathway_fdr_cutoff_default:float=0.05,
						  force:bool=False, verbose:bool=False) -> pd.DataFrame:

		fname = self.fname_enr_simulation%(case, self.normalization)
		filefull = os.path.join(self.root_ressum, fname)

		if os.path.exists(filefull) and not force:
			return pdreadcsv(fname, self.root_ressum)

		dic = {}
		for i in range(n_sim):
			print(i,end=' ')
			df_enr = self.calc_enriched_pathways_random_genes(i, case, abs_lfc_cutoff_default, fdr_lfc_cutoff_default,
															  pathway_fdr_cutoff_default, verbose=verbose)

			try:
				dfmat, ret_chi, dof, stat, pvalue, stri_stat = self.build_matrix_calc_chi_square(\
					self.n_degs_in_pathways_bca, self.n_degs_not_in_pathways_bca, \
					self.n_degs_in_pathways_random, self.n_degs_not_in_pathways_random)
			except:
				print(f"\n {i}) Error", self.n_degs_in_pathways_bca, self.n_degs_not_in_pathways_bca,
										self.n_degs_in_pathways_random, self.n_degs_not_in_pathways_random)
				continue

			dic[i] = {}
			dic2 = dic[i]

			dic2['stat_sig'] = ret_chi
			dic2['dof'] = dof
			dic2['stat'] = stat
			dic2['pvalue'] = f'{pvalue:.3e}'
			dic2['stri_stat'] = stri_stat
			dic2['dfmat'] = dfmat

		dff = pd.DataFrame(dic).T
		_ = pdwritecsv(dff, fname, self.root_ressum)

		return dff
