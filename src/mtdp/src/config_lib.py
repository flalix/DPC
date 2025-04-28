#!/usr/bin/python
#!python
# -*- coding: utf-8 -*-
# Created on 2024/01/29
# @author: Flavio Lichtenstein
# @institution: Butantan Institute, Molecular Biology, Bioinformatics, CENTD

import numpy as np
import gzip, os, sys
import pandas as pd
from typing import Optional, Iterable, Set, Tuple, Any, List

from Basic import *

class Config(object):
    def __init__(self, project:str, s_project:str, case_list:List, root0:str):

        self.root0 = root0
        self.root_config = create_dir(root0, 'config')
        self.project = project
        self.s_project = s_project
        self.case_list = case_list

        self.normalization = 'not_normalized'

        self.dfbest_lfc_cutoff = None
        self.dfbest_cutoffs = None

        self.abs_lfc_cutoff = 1
        self.fdr_lfc_cutoff = 0.05

        self.n_degs = -1
        self.n_degs_up = -1
        self.n_degs_dw = -1

        self.quantile = -1
        self.geneset_num = 0
        self.pathway_fdr_cutoff = 0.05
        self.num_of_genes_cutoff = 3

        self.n_genes_annot_ptw = -1
        self.n_degs = -1
        self.n_degs_in_ptw = -1
        self.n_degs_not_in_ptw = -1
        self.degs_in_all_ratio = -1

        self.toi1 = -1
        self.toi2 = -1

        self.param_lfc_defaults = 1, 0.05, -1, -1, -1

        '''
        return row.abs_lfc_cutoff, row.fdr_lfc_cutoff, row.pathway_fdr_cutoff,  \
               row.n_genes_annot_ptw, row.n_degs, row.n_degs_in_ptw, row.n_degs_not_in_ptw, row.degs_in_all_ratio, row.toi1, row.toi2
        '''
        self.param_ptw_defaults = 0.9, 1, 0.05, 0.05, 0.05, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1

        fname_lfc_cutoff = f'all_lfc_cutoffs_{s_project}.tsv'
        self.fname_lfc_cutoff = title_replace(fname_lfc_cutoff)

        fname_ptw_cutoff = f'best_ptw_cutoffs_{project}.tsv'
        self.fname_ptw_cutoff = title_replace(fname_ptw_cutoff)

    def set_default_best_lfc_cutoff(self, normalization:str, abs_lfc_cutoff:float=1, fdr_lfc_cutoff:float=.05):

        self.quantile = -1
        self.normalization = normalization

        self.abs_lfc_cutoff = abs_lfc_cutoff
        self.fdr_lfc_cutoff = fdr_lfc_cutoff
        self.cutoff = f"{abs_lfc_cutoff:.3f} - {fdr_lfc_cutoff:.3f}"
        self.abs_lfc_cutoff = abs_lfc_cutoff
        self.fdr_lfc_cutoff = fdr_lfc_cutoff

        self.n_degs = -1
        self.n_degs_up = -1
        self.n_degs_dw = -1


    def open_all_lfc_cutoff(self, verbose=True) -> pd.DataFrame:
        filefull = os.path.join(self.root_config, self.fname_lfc_cutoff)
        if not os.path.exists(filefull):
            if verbose: print(f"Best parameter file for LFC does not exist {filefull}")
            self.dfbest_lfc_cutoff = None
            return None

        dfbest_lfc_cutoff = pdreadcsv(self.fname_lfc_cutoff, self.root_config)
        self.dfbest_lfc_cutoff = dfbest_lfc_cutoff

        return dfbest_lfc_cutoff

    def save_best_lfc_cutoff(self, dfi, verbose:bool=False) -> bool:
        dfbest_lfc_cutoff = self.open_all_lfc_cutoff(verbose=False)

        if dfbest_lfc_cutoff is None or dfbest_lfc_cutoff.empty:
            dfbest_lfc_cutoff = dfi
        else:
            dfbest_lfc_cutoff = dfbest_lfc_cutoff[(dfbest_lfc_cutoff.normalization != self.normalization)]

            if dfbest_lfc_cutoff.empty:
                dfbest_lfc_cutoff = dfi
            else:
                dfbest_lfc_cutoff = pd.concat([dfbest_lfc_cutoff, dfi])

        dfbest_lfc_cutoff.index = np.arange(0, len(dfbest_lfc_cutoff))

        ret = pdwritecsv(dfbest_lfc_cutoff, self.fname_lfc_cutoff, self.root_config, verbose=verbose)
        return ret

    def get_best_lfc_cutoff(self, case:str, normalization:str, verbose:bool=False) -> (float, float, int, int, int):
        if self.dfbest_lfc_cutoff is None:
            _ = self.open_all_lfc_cutoff()
            if self.dfbest_lfc_cutoff is None:
                if verbose:
                    print("Houston we have a problem: no CONFIGURATION file was found.")
                    print(">>> run: new03_up_down_simulation")
                return self.param_lfc_defaults

        dfa = self.dfbest_lfc_cutoff[(self.dfbest_lfc_cutoff.case == case) & (self.dfbest_lfc_cutoff.normalization == normalization) ]
        if dfa.empty:
            return self.param_lfc_defaults

        row = dfa.iloc[0]

        return row.abs_lfc_cutoff, row.fdr_lfc_cutoff, row.n_degs,  row.n_degs_up,  row.n_degs_dw


    def set_default_best_ptw_cutoff(self, normalization:str, geneset_num:int=0, quantile:int=0.5,
                                    abs_lfc_cutoff:float=1, fdr_lfc_cutoff:float=.05, pathway_fdr_cutoff:float=0.05,
                                    n_genes_annot_ptw:int=0, n_degs:int=0, n_degs_in_ptw:int=0,  n_degs_not_in_ptw:int=0, degs_in_all_ratio:int=0):

        self.normalization = normalization
        self.geneset_num = geneset_num
        self.quantile = quantile

        self.cutoff = f"{abs_lfc_cutoff:.3f} - {fdr_lfc_cutoff:.3f}"
        self.abs_lfc_cutoff = abs_lfc_cutoff
        self.fdr_lfc_cutoff = fdr_lfc_cutoff
        self.pathway_fdr_cutoff = pathway_fdr_cutoff

        self.n_genes_annot_ptw = n_genes_annot_ptw

        self.n_degs = n_degs

        self.n_degs_in_ptw = n_degs_in_ptw
        self.n_degs_not_in_ptw = n_degs_not_in_ptw
        self.degs_in_all_ratio = degs_in_all_ratio

        self.toi1 = -1
        self.toi2 = -1


    def open_best_ptw_cutoff(self, verbose=False) -> pd.DataFrame:
        filefull = os.path.join(self.root_config, self.fname_ptw_cutoff)
        if not os.path.exists(filefull):
            if verbose: print(f"Best parameter file for Pathways does not exist {filefull}")
            self.dfbest_cutoffs = None
            return None

        dfbest_cutoffs = pdreadcsv(self.fname_ptw_cutoff, self.root_config, verbose=verbose)
        self.dfbest_cutoffs = dfbest_cutoffs

        return dfbest_cutoffs

    def save_best_ptw_cutoff(self, dfi, verbose:bool=False) -> bool:
        '''
        config table is build in
        pubmed_taubate_new06_summary_degs_and_pathways_save_config
        '''
        dfbest_cutoffs = self.open_best_ptw_cutoff(verbose=False)

        if dfbest_cutoffs is None or dfbest_cutoffs.empty:
            dfbest_cutoffs = dfi
        else:
            dfbest_cutoffs = dfbest_cutoffs[(dfbest_cutoffs.normalization != self.normalization) & (dfbest_cutoffs.geneset_num != self.geneset_num)]

            if dfbest_cutoffs.empty:
                dfbest_cutoffs = dfi
            else:
                dfbest_cutoffs = pd.concat([dfbest_cutoffs, dfi])

        dfbest_cutoffs = dfbest_cutoffs.drop_duplicates(['case', 'normalization', 'geneset_num', 'quantile', 'med_max_ptw'])
        dfbest_cutoffs.index = np.arange(0, len(dfbest_cutoffs))

        self.dfbest_cutoffs = dfbest_cutoffs

        ret = pdwritecsv(dfbest_cutoffs, self.fname_ptw_cutoff, self.root_config, verbose=verbose)
        return ret


    def get_cfg_best_ptw_cutoff(self, case:str, normalization:str, geneset_num:int,
                                med_max_ptw:str='median', verbose:bool=False) -> (float, float, int, int, int, float):
        if self.dfbest_cutoffs is None:
            _ = self.open_best_ptw_cutoff(verbose=verbose)
            if self.dfbest_cutoffs is None:
                if verbose:
                    print(f"Could not find best params for {case} {normalization} {geneset_num}")
                    print("Houston we have a problem: No best parameter file for Pathways was found.")
                    print(">>> run: pubmed_taubate_new05_best_cutoffs_sim_and_save_config.ipynb")
                return self.param_ptw_defaults

        ''' remove: (self.dfbest_cutoffs.geneset_num == geneset_num) '''
        dfa = self.dfbest_cutoffs[(self.dfbest_cutoffs.case == case) &
                                  (self.dfbest_cutoffs.normalization == normalization) ]
        if dfa.empty:
            print(f"Could not find best params for {case} {normalization}") #{geneset_num}
            return self.param_ptw_defaults

        dfa = dfa[dfa.med_max_ptw == med_max_ptw]
        if dfa.empty:
            return self.param_lfc_defaults

        '''
         return values:

         ['case', 'geneset_num', 'normalization', 'parameter', 'quantile',
          'quantile_val', 'abs_lfc_cutoff', 'fdr_lfc_cutoff',
         'pathway_pval_cutoff', 'pathway_fdr_cutoff', 'num_of_genes_cutoff',
         'n_pathways', 'n_degs_in_pathways', 'n_degs_in_pathways_mean', 'n_degs_in_pathways_median',
         'n_degs_in_pathways_std', 'toi1_median', 'toi2_median', 'toi3_median', 'toi4_median'],
        '''
        row = dfa.iloc[0]

        if verbose:
            print( 'quantile, abs_lfc_cutoff, fdr_lfc_cutoff, ' +
                   'pathway_pval_cutoff, pathway_fdr_cutoff, num_of_genes_cutoff,  ' +
                   'n_pathways, n_degs_in_pathways, ' +
                   'n_degs_in_pathways_mean, n_degs_in_pathways_median, n_degs_in_pathways_std, ' +
                   'toi1_median, toi2_median, toi3_median, toi4_median')

            print( row['quantile'], row.abs_lfc_cutoff, row.fdr_lfc_cutoff, \
                   row.pathway_pval_cutoff, row.pathway_fdr_cutoff, row.num_of_genes_cutoff, \
                   row.n_pathways, row.n_degs_in_pathways, \
                   row.n_degs_in_pathways_mean, row.n_degs_in_pathways_median, row.n_degs_in_pathways_std, \
                   row.toi1_median, row.toi2_median, row.toi3_median, row.toi4_median)

        return row['quantile'], row.abs_lfc_cutoff, row.fdr_lfc_cutoff, \
               row.pathway_pval_cutoff, row.pathway_fdr_cutoff, row.num_of_genes_cutoff, \
               row.n_pathways, row.n_degs_in_pathways, \
               row.n_degs_in_pathways_mean, row.n_degs_in_pathways_median, row.n_degs_in_pathways_std, \
               row.toi1_median, row.toi2_median, row.toi3_median, row.toi4_median


    def get_any_ptw_cutoff_REVIEW_TODO(self, case:str, normalization:str, geneset_num:int,
                           quantile:float, verbose:bool=False) -> (float, float, int, int, int, float):
        if self.dfbest_cutoffs is None:
            _ = self.open_best_ptw_cutoff()
            if self.dfbest_cutoffs is None:
                if verbose:
                    print("Houston we have a problem: No best parameter file for Pathways was found.")
                    print(">>> run: new06_enricher_statistics_and_save_config_table.ipynb")
                return self.param_ptw_defaults

        dfa = self.dfbest_cutoffs[(self.dfbest_cutoffs.case == case) & (self.dfbest_cutoffs.normalization == normalization) &
                                     (self.dfbest_cutoffs.geneset_num == geneset_num) & (self.dfbest_cutoffs['quantile'] == quantile) ]
        if dfa.empty:
            return self.param_ptw_defaults


        '''
         return values:

         parameter (toi1), quantile_val, abs_lfc_cutoff, fdr_lfc_cutoff,
         pathway_pval_cutoff, pathway_fdr_cutoff, num_of_genes_cutoff, n_pathways,
         n_genes_in_pahtways, n_degs_in_pathways_mean, n_degs_in_pathways_median, n_degs_in_pathways_std,
         toi1_median, toi2_median, toi3_median, toi4_median
        '''
        row = dfa.iloc[0]

        return row.abs_lfc_cutoff, row.fdr_lfc_cutoff, \
               row.pathway_pval_cutoff, row.pathway_fdr_cutoff, row.num_of_genes_cutoff, \
               row.n_pathways, row.n_genes_in_pahtways, \
               row.n_degs_in_pathways_mean, row.n_degs_in_pathways_median, row.n_degs_in_pathways_std, \
               row.toi1_median, row.toi2_median, row.toi3_median, row.toi4_median
