#!/usr/bin/python
#!python
# -*- coding: utf-8 -*-

from Bio import Entrez
from Bio import Medline
import numpy as np
import pandas as pd
import os, sys, time, shutil
from typing import Optional, Iterable, Set, Tuple, Any, List

import spacy, re
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

import nltk
from nltk.stem.porter import *

from transformers import AutoTokenizer, pipeline
import torch, warnings

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

import multiprocessing
from multiprocessing import Pool, freeze_support
import subprocess

# sys.path.insert(1, '../src/')
from Basic import *
from pubmed_lib import *
from biopax_lib import *


class NLP(Pubmed):

    def __init__(self, email:str, prefix:str, root0:str,
                 remove_synonym_list:List=[], sleep_entrez:List=[30, 90, 300],
                 retmax:int=100000, try_all_text:bool=True, text_quote:str='',
                 root_colab:str='../../colaboracoes/', dec_ncpus:int=2,
                 sleep_TIKA:List=[10, 20, 30], min_words_text:int=100):


        super().__init__(email=email, prefix=prefix, root0=root0,
                         remove_synonym_list=remove_synonym_list, sleep_entrez=sleep_entrez,
                         retmax=retmax, try_all_text=try_all_text, text_quote=text_quote,
                         root_colab=root_colab, dec_ncpus=dec_ncpus,
                         sleep_TIKA=sleep_TIKA, min_words_text=min_words_text)


        self.s_project, self.task =  None, None
        self.device, self.classifier, self.llm_name = None, None, None

        self.cluster_model, self.tfidf_alg, self.n_components = None, None, None
        self.tfidf_alg, self.tfidf, self.cv, self.dtm = None, None, None, None

        self.df_lemma_scv, self.previous_df_lemma_scv = None, None
        self.previous_tfidf_alg   = None
        self.previous_perc_min, self.previous_perc_max = -1, -1

        self.perc_min, self.perc_max = 0.20, 0.80

        self.fname_cluster_counts    = 'pubmed_search_pmids_cluster_classif_counts_model_%s_alg_%s_n_components_%s_nwords_%d.tsv'
        self.fname_cluster_counts_ed = 'pubmed_search_pmids_cluster_classif_counts_model_%s_alg_%s_n_components_%s_nwords_%d_edited.tsv'
        self.fname_cluster           = 'pubmed_search_pmids_cluster_classif_model_%s_alg_%s_n_components_%s_nwords_%d.tsv'

        self.fname_subcluster_counts = 'pubmed_search_pmids_filtered_subcluster_%s_classif_counts_model_%s_alg_%s_n_components_%s_nwords_%d.tsv'
        self.fname_subcluster        = 'pubmed_search_pmids_filtered_subcluster_%s_classif_model_%s_alg_%s_n_components_%s_nwords_%d.tsv'

        self.dfclu, self.dfcts  = None, None
        self.dfsubclu, self.dfsubcts = None, None
        self.subcluster_name = ''

        self.selected_cluster_file, self.selected_cluster_dir = None, None
        self.selected_cluster_components, self.selected_cluster_nwords  = -1, -1

    def calc_vectorizing(self, verbose:bool=False):

        if self.previous_df_lemma_scv is not None:
            if all_equal_list(self.previous_df_lemma_scv.pmid, self.df_lemma_scv.pmid):
                if self.previous_tfidf_alg is not None:
                    if self.previous_perc_min == self.perc_min and self.previous_perc_max == self.perc_max:

                        if self.previous_tfidf_alg == 'tfidf':
                            if self.tfidf is not None and self.dtm is not None:
                                return

                        elif self.previous_tfidf_alg == 'countvect':
                            if self.cv is not None and self.dtm is not None:
                                return

        self.previous_df_lemma_scv = self.df_lemma_scv
        self.previous_tfidf_alg   = self.tfidf_alg
        self.previous_perc_min = self.perc_min
        self.previous_perc_max = self.perc_max

        if verbose:
            if self.subcluster_name == '':
                print(f"Vectorizing with original lemma and {self.tfidf_alg} ....")
            else:
                print(f"Vectorizing with filtered '{self.subcluster_name}' lemma and {self.tfidf_alg} ....")

        if self.tfidf_alg == 'tfidf':
            tfidf = TfidfVectorizer(max_df=self.perc_max, min_df=self.perc_min, stop_words='english')
            dtm = tfidf.fit_transform(self.df_lemma_scv['lemma'])

            self.tfidf = tfidf
            self.cv = None
            self.dtm = dtm

        elif self.tfidf_alg == 'countvect':
            cv = CountVectorizer(max_df=self.perc_max, min_df=self.perc_min, stop_words='english')
            dtm = cv.fit_transform(self.df_lemma_scv['lemma'])

            self.tfidf = None
            self.cv = cv
            self.dtm = dtm
        else:
            self.tfidf = None
            self.cv = None
            self.dtm = None

            print(f"Bad Vectorizing algorithm: {self.tfidf_alg}")
            raise Exception('Stop: Vectorizing algorithm.')


    def set_params(self, cluster_model:str, tfidf_alg:str, perc_max:float, perc_min:float,
                    n_components:int, n_words:int, random_state:int, subcluster_name:str, terms:List, verbose:bool=False):

        self.n_components = n_components
        self.n_words = n_words

        self.subcluster_name = subcluster_name
        self.terms = terms

        self.dfsubclu, self.dfsubcts = None, None

        self.cluster_model = cluster_model
        self.random_state = random_state

        if perc_min < 0 or perc_min > .5:
            print('Error: perc_min = [0, .5]')
            raise Exception('Stop: perc_min')

        if perc_max < 0.2 or perc_max > 1:
            print('Error: perc_max = [0.2, 1]')
            raise Exception('Stop: perc_max')

        if perc_max <= perc_min:
            print('Error: perc_max must be > perc_min')
            raise Exception('Stop: perc_max > perc_min')

        self.tfidf_alg = tfidf_alg
        self.perc_min = perc_min
        self.perc_max = perc_max

        if self.cluster_model == 'LDA':
            self.root_cluster0 = self.root_clu_lda
        elif self.cluster_model == 'NMF':
            self.root_cluster0 = self.root_clu_nmf
        elif self.cluster_model == 'tSNE':
            self.root_cluster0 = self.root_clu_tsne
        elif self.cluster_model == 'RF':
            self.root_cluster0 = self.root_clu_rfor
        else:
            print(f"Which model you want? {self.cluster_model }, please define a root.")
            raise Exception('Stop root-model')

    def set_perc_min_max_dir(self, perc_min:float, perc_max:float) -> bool:
        self.s_perc_min = str(int(perc_min*100))
        self.s_perc_max = str(int(perc_max*100))

        self.root_sim = os.path.join(self.root_cluster0, f'cluster_max_{self.s_perc_max}_min_{self.s_perc_min}')

        ret = True

        try:
            if not os.path.exists(self.root_sim):
                os.mkdir(self.root_sim)
        except:
            ret = False

        return ret

    def calc_rows_percentages(self, row, perc_cutoff:float=0.10) -> (int, int):
        nrows = row.nrows
        percs = row.percs

        if isinstance(percs, str):
            percs = eval(percs)

        percs_filtered = [x for x in percs if x >= perc_cutoff]
        nrows_rep = len(percs_filtered)

        return nrows, nrows_rep

    def select_best_cluster_result(self, nrows0_min:int=3, nrows0_rep_min:int=3,
                                   perc_cutoff:float=0.10, verbose:bool=False) -> (pd.DataFrame, pd.DataFrame):

        self.selected_cluster_file, self.selected_cluster_dir = None, None
        self.selected_cluster_components, self.selected_cluster_nwords  = -1, -1

        dirs = [x for x in os.listdir(self.root_cluster0) if x.startswith('cluster_max')]

        dic = {}; icount=-1
        for _dir in dirs:
            final_dir = os.path.join(self.root_cluster0, _dir)

            fnames = [x for x in os.listdir(final_dir) if x.startswith('cluster_max')]
            if fnames == []:
                print(f"Problems finding cluster_max_min file, see '{final_dir}'")
                continue

            fname = fnames[0]
            dfh = pdreadcsv(fname, final_dir)

            h0 = dfh.iloc[0].h
            h1 = dfh.iloc[1].h
            h2 = dfh.iloc[2].h

            nrows0, nrows_rep0 = calc_rows_percentages(dfh.iloc[0])
            nrows1, nrows_rep1 = calc_rows_percentages(dfh.iloc[1])
            nrows2, nrows_rep2 = calc_rows_percentages(dfh.iloc[2])

            icount += 1
            dic[icount] = {}
            dic2 = dic[icount]

            dic2['fname'] = fname
            dic2['h0'] = h0
            dic2['h1'] = h1
            dic2['h2'] = h2

            dic2['nrows0'] = nrows0
            dic2['nrows1'] = nrows1
            dic2['nrows2'] = nrows2

            dic2['nrows_rep0'] = nrows_rep0
            dic2['nrows_rep1'] = nrows_rep1
            dic2['nrows_rep2'] = nrows_rep2

        dfh = pd.DataFrame(dic).T

        dfh = dfh[ (dfh.nrows0 >= nrows0_min) & (dfh.nrows_rep0 >= nrows0_rep_min) ] # & (dfh.nrows0 == dfh.nrows_rep0)
        dfh = dfh.sort_values(['nrows0', 'h0'], ascending=[True,False])

        fname = 'cluster_entropy.tsv'
        ret = pdwritecsv(dfh, fnme, self.root_cluster0, verbose=verbose)


        '''------ find best results -----------------------'''
        fname = dfh.iloc[0].fname
        self.selected_cluster_file = fname

        _dir = fname[:-4]
        self.selected_cluster_dir = _dir
        final_dir = os.path.join(self.root_cluster0, _dir)

        dfh_sel = pdreadcsv(fname, final_dir, verbose=False)

        sel_components = dfh_sel.iloc[0].components
        sel_nwords = dfh_sel.iloc[0].nwords

        self.selected_cluster_components = sel_components
        self.selected_cluster_nwords     = nwords

        print(f"Selected: {_dir}.{fname}, components={sel_components} and nwords={nwords}")

        return dfh, dfh_sel


    def loop_many_cluster_model_all_loops(self, all_pdf_html:str='all', filter:str=None,
                                          cluster_model:str='LDA',
                                          n_word_list:List=[10, 15, 20, 25, 30, 35, 40],
                                          perc_min_list:List=[0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25],
                                          perc_max_list:List=[0.99, 0.95, 0.90, 0.85, 0.75, 0.60, 0.50],
                                          tfidf_alg:str='tfidf', random_state:int=14,
                                          force:bool=False, verbose:bool=False) -> bool:
        '''
            loop many perc_min x perc_max
            try to filter a special concept, otherwise all papers' text (corpus) are included


            filter = "((df_lemma.lemma.str.contains('g4')) | (df_lemma.lemma.str.contains('wnt'))) & " + \
                     "((df_lemma.lemma.str.contains('transcript')) | (df_lemma.lemma.str.contains('proteom')) )"
        '''
        tuples_subclusters = []
        subcluster_name = ''
        terms = []
        tuples_subclusters.append((subcluster_name, terms))

        for perc_min in perc_min_list:
            for perc_max in perc_max_list:

                if not set_perc_min_max_dir(self, perc_min, perc_max):
                    return False

                print(f">>>> perc max {perc_max} x perc min {perc_min}")
                self.cluster_model_all_loops(all_pdf_html=all_pdf_html, filter=filter,
                                            tuples_subclusters=tuples_subclusters, cluster_model=cluster_model,
                                            n_word_list=n_word_list, tfidf_alg=tfidf_alg,
                                            perc_min=perc_min, perc_max=perc_max,
                                            random_state=random_state, force=force, verbose=verbose)

                files = [x for x in os.listdir(self.root_sim) if 'counts_model' in x and not '~lock' in x]

                i = -1; dic={}
                for fname in files:
                    df = pdreadcsv(fname, self.root_sim)
                    i += 1

                    total = df.n.sum()
                    percs = [x/total for x in df.n]
                    ''' --- informational Shannon Entropy = h ------------------'''
                    h = np.sum([-p*np.log(p) for p in percs])

                    dic[i] = {}
                    dic2 = dic[i]
                    dic2['fname'] = fname
                    mat = fname.split('_')

                    components = int(mat[-3])
                    nwords     = int(mat[-1][:-4])

                    dic2['nrows'] = len(percs)
                    dic2['percs'] = percs
                    dic2['components'] = components
                    dic2['nwords'] = nwords

                    dic2['h'] = h

                dfh = pd.DataFrame(dic).T
                dfh = dfh.sort_values(['h', 'nwords', 'components'], ascending=[False,True,True])

                fname_final = f'cluster_max_{self.s_perc_max}_min_{self.s_perc_min}.tsv'
                ''' the first 3 lines are sufficient, here we defined the first 10 lines '''
                ret = pdwritecsv(dfh.head(10), fname_final, self.root_sim, verbose=verbose)
                if not ret:
                    return False

        return True


    '''
        tuples_subclusters = []

        subcluster_name = 'G4'
        terms = ['g4', 'group 4']
        tuples_subclusters.append((subcluster_name, terms))

        subcluster_name = 'G3'
        terms = ['g3', 'group 3']
        tuples_subclusters.append((subcluster_name, terms))

        subcluster_name = 'SHH'
        terms = ['shh', 'hedgehog']
        tuples_subclusters.append((subcluster_name, terms))

        subcluster_name = 'wnt'
        terms = ['wnt']
        tuples_subclusters.append((subcluster_name, terms))

    '''
    def cluster_model_all_loops(self, all_pdf_html:str='all', filter:str=None,
                                tuples_subclusters:List=[], cluster_model:str='LDA',
                                n_word_list:List=[5, 7, 10, 15],
                                tfidf_alg:str='tfidf', perc_min:float=0.10, perc_max:float=0.8,
                                random_state:int=14, force:bool=False, verbose:bool=False):

        ''' repeated, if one call this method directly '''
        if not set_perc_min_max_dir(self, perc_min, perc_max):
            return False

        for subcluster_name, terms in tuples_subclusters:
            for n_words in n_word_list:
                if n_words <= 5:
                    comp_list = [7, 10, 12]
                elif n_words <= 7:
                    comp_list = [7, 10, 12, 15]
                elif n_words <= 10:
                    comp_list = [7, 10, 12, 15, 20]
                else:
                    comp_list = [7, 10, 12, 15, 20, 25]

                for n_components in comp_list:
                    self.set_params(cluster_model, tfidf_alg, perc_max, perc_min,
                                    n_components, n_words, random_state, subcluster_name,
                                    terms, verbose=verbose)

                    if verbose: print(f"n_words={self.n_words} n_components={self.n_components}")
                    ret = self.cluster_model_loop(all_pdf_html=all_pdf_html, filter=filter, force=force, verbose=verbose)


    def cluster_model_loop(self, all_pdf_html:str='all', filter:str=None, force:bool=False, verbose:bool=False):
        '''
        this functions does:
           a) cluster
           b) subcluster

        todo subcluster one may passe a name a the terms
        in subcluster it may create terms that does not have relation with the filters
        '''


        if self.subcluster_name == '':
            fname1 = self.fname_cluster%(self.cluster_model, self.tfidf_alg, self.n_components, self.n_words)
            fname2 = self.fname_cluster_counts%(self.cluster_model, self.tfidf_alg, self.n_components, self.n_words)

            filefull1 = os.path.join(self.s_perc_min, fname1)
            filefull2 = os.path.join(self.s_perc_min, fname2)

            if not force and os.path.exists(filefull1) and os.path.exists(filefull2):
                return True

            if filter is not None and isinstance(filter, str) and len(filter) > 10:
                self.open_lemma(all_pdf_html=all_pdf_html, open_all=False, verbose=False)

                df_lemma = self.df_lemma.copy()
                df_lemma = df_lemma[ eval(filter)]
                df_lemma.index = np.arange(0, len(df_lemma))
                self.df_lemma = df_lemma

                if verbose: print(f"There are {len(self.df_lemma)} papers with filter.")
            else:
                if self.df_lemma is None:
                    self.open_lemma(all_pdf_html=all_pdf_html, open_all=False, verbose=False)

                if verbose: print(f"There are {len(self.df_lemma)} papers without filter.")

            self.df_lemma_scv = self.df_lemma

        else:
            fname1 = self.fname_subcluster%(self.subcluster_name, self.cluster_model, self.tfidf_alg, self.n_components, self.n_words)
            fname2 = self.fname_subcluster_counts%(self.subcluster_name, self.cluster_model, self.tfidf_alg, self.n_components, self.n_words)

            filefull1 = os.path.join(self.s_perc_min, fname1)
            filefull2 = os.path.join(self.s_perc_min, fname2)

            if  not force and os.path.exists(filefull1) and os.path.exists(filefull2):
                return True

            _, _ = self.open_cluster(verbose=False)

            if self.dfclu is None or self.dfclu.empty or \
               self.dfcts is None or self.dfcts.empty:
               print("Could not find dfclu/dfcts, impossible to continue")
               return False

            dfsubcts, dfsubclu = self.filter_cluster_terms(verbose=verbose)

            if dfsubclu is None or dfsubclu.empty:
                # print(f"No subcluster: {self.subcluster_name}")
                return False

            self.df_lemma_scv = dfsubclu

        if len(self.df_lemma_scv) < 3:
            print("There are no sufficient data (pmids) to perform clusterization.")
            return False

        ''' must be sorted, to compare any change '''
        self.df_lemma_scv = self.df_lemma_scv.sort_values('pmid')
        self.calc_vectorizing()

        if self.cluster_model == 'LDA':
            if verbose: print(f"Fitting {self.cluster_model} with n_words={self.n_words} and n_components={self.n_components}, wait ...")
            LDA = LatentDirichletAllocation(n_components=self.n_components, random_state=self.random_state)
            LDA.fit(self.dtm)
            ''' fns = feature names '''
            if self.tfidf_alg == 'tfidf':
                fns = self.tfidf.get_feature_names_out()
            else:
                fns = self.cv.get_feature_names_out()

            # grap topics
            cmps = LDA.components_
            self.cmps = cmps

        elif self.cluster_model =='NMF':
            if verbose: print(f"Fitting {self.cluster_model}, with n_components={self.n_components} and n_words={self.n_words}, wait ...")
            NMF = NMF(n_components=self.n_components, random_state=self.random_state)
            NMF.fit(self.dtm)
            # feature names
            if self.tfidf_alg == 'tfidf':
                fns = self.tfidf.get_feature_names_out()
            else:
                fns = self.cv.get_feature_names_out()

            # grap topics
            cmps = NMF.components_
            self.cmps = cmps

        else:
            self.cmps = None
            print(f"Model not found: {self.cluster_model}")
            raise Exception('stop cluster model.')


        dic, mytopic_dic={}, {}
        if verbose:
            print(f'The top words for {self.cluster_model} and n_components={self.n_components} and n_words={self.n_words} and random_state={self.random_state}')

        for index, topic in enumerate(self.cmps):
            single_topic = self.cmps[index]
            # argsort: index positions from least to greatest
            ndx = single_topic.argsort()
            highest_ndx = ndx[-self.n_words:]

            group_n_features = [fns[i] for i in highest_ndx]
            dic[index] = group_n_features

            words = ", ".join(group_n_features)
            mytopic_dic[index] = words

            if verbose:
                stri = f"{index}: '{words}',"
                print(stri)

        dfcomp = pd.DataFrame(dic).T

        if self.cluster_model == 'LDA':
            topic_results = LDA.transform(self.dtm)
        elif self.cluster_model =='NMF':
            topic_results = NMF.transform(self.dtm)
        else:
            print(f"Bad model ??? {self.cluster_model} for transform")
            raise Exception('stop: model.transform()')

        cols = ['pmid', 'lemma']
        dfclu = self.df_lemma_scv[cols].copy()
        dfclu['topic_id'] = topic_results.argmax(axis=1)
        dfclu['topic']    = dfclu['topic_id'].map(mytopic_dic)
        dfclu = dfclu.sort_values(['topic_id', 'pmid'])

        dfcts = dfclu.groupby(['topic_id', 'topic']).count().reset_index().iloc[:,:3]
        dfcts.columns = ['topic_id', 'topic', 'n']
        dfcts = dfcts.sort_values('n', ascending=False)
        dfcts.index = np.arange(0, len(dfcts))
        dfcts['rank'] = np.arange(1, len(dfcts)+1)

        if self.subcluster_name != '':
            ''' in subcluster it may create terms that does not have relation with the filters '''
            self.dfsubclu = dfclu
            self.dfsubcts = dfcts

            ret = self.save_subcluster_files(verbose=verbose)
        else:
            self.dfclu = dfclu
            self.dfcts = dfcts

            ret = self.save_cluster_files(verbose=verbose)

        return ret


    def save_cluster_files(self, verbose=False):
        fname1 = self.fname_cluster%(self.cluster_model, self.tfidf_alg, self.n_components, self.n_words)
        fname2 = self.fname_cluster_counts%(self.cluster_model, self.tfidf_alg, self.n_components, self.n_words)

        ret1 = pdwritecsv(self.dfclu, fname1, self.root_sim, verbose=verbose)
        ret2 = pdwritecsv(self.dfcts, fname2, self.root_sim, verbose=verbose)

        return ret1*ret2==1

    def save_subcluster_files(self, verbose=False):
        fname1 = self.fname_subcluster%(self.subcluster_name, self.cluster_model, self.tfidf_alg, self.n_components, self.n_words)
        fname2 = self.fname_subcluster_counts%(self.subcluster_name, self.cluster_model, self.tfidf_alg, self.n_components, self.n_words)

        ret1 = pdwritecsv(self.dfsubclu, fname1, self.root_sim, verbose=verbose)
        ret2 = pdwritecsv(self.dfsubcts, fname2, self.root_sim, verbose=verbose)

        return ret1*ret2==1


    def open_cluster(self, verbose:bool=False) -> (pd.DataFrame, pd.DataFrame):

        self.dfclu, self.dfcts, self.df_lemma_scv = None, None, None

        self.s_perc_min = str(int(perc_min*100))
        self.s_perc_max = str(int(perc_max*100))

        self.root_sim = os.path.join(self.root_cluster0, f'cluster_max_{self.s_perc_max}_min_{self.s_perc_min}')

        fname1 = self.fname_cluster%(self.cluster_model, self.tfidf_alg, self.n_components, self.n_words)
        filefull1 = os.path.join(self.root_sim, fname1)

        if not os.path.exists(filefull1):
            print(f"File does not exist: '{filefull1}'")
            return None, None

        fname2 = self.fname_cluster_counts%(self.cluster_model, self.tfidf_alg, self.n_components, self.n_words)
        filefull2 = os.path.join(self.root_sim, fname2)

        if not os.path.exists(filefull2):
            print(f"File does not exist: '{filefull2}'")
            return None, None

        dfclu = pdreadcsv(fname1, self.root_sim, verbose=verbose)
        dfcts = pdreadcsv(fname2, self.root_sim, verbose=verbose)

        self.dfclu = dfclu
        self.dfcts = dfcts

        self.df_lemma_scv = dfclu

        return dfcts, dfclu


    def open_subcluster(self, verbose=False):

        self.dfsubclu, self.dfsubcts = None, None

        fname1 = self.fname_subcluster%(self.subcluster_name, self.cluster_model, self.tfidf_alg, self.n_components, self.n_words)
        filefull1 = os.path.join(self.root_sim, fname1)

        if not os.path.exists(filefull1):
            if verbose: print(f"File does not exist: '{filefull1}'")
            return None, None

        fname2 = self.fname_subcluster_counts%(self.subcluster_name, self.cluster_model, self.tfidf_alg, self.n_components, self.n_words)
        filefull2 = os.path.join(self.root_sim, fname2)

        if not os.path.exists(filefull2):
            if verbose: print(f"File does not exist: '{filefull2}'")
            return None, None

        dfsubclu = pdreadcsv(fname1, self.root_sim, verbose=verbose)
        dfsubcts = pdreadcsv(fname2, self.root_sim, verbose=verbose)

        self.dfsubclu = dfsubclu
        self.dfsubcts = dfsubcts

        return dfsubcts, dfsubclu


    # terms = ['endothel', 'neutrophil']
    def filter_cluster_terms(self, verbose:bool=False) -> (pd.DataFrame, pd.DataFrame):

        df_list = []
        for term in self.terms:
            dfa = self.dfcts[self.dfcts.topic.str.contains(term)]

            if not dfa.empty:
                df_list.append(dfa)

        if df_list == []:
            print(f"Nothing found with these terms {','.join(self.terms)}")
            return None, None

        df = pd.concat(df_list)
        df = df.drop_duplicates('topic_id')
        topic_ids = df.topic_id.to_list()

        print(f"There are {len(topic_ids)} groups with the terms={','.join(self.terms)}")
        if topic_ids == []:
            return None, None

        dfsubcts = self.dfcts[self.dfcts.topic_id.isin(topic_ids)].copy()
        dfsubcts.index = np.arange(0, len(dfsubcts))

        cols = ['pmid', 'lemma',]

        dfsubclu = self.dfclu[self.dfclu.topic_id.isin(topic_ids)][cols].copy()
        dfsubclu.index = np.arange(0, len(dfsubclu))
        print(f"There are {len(dfsubclu)} pmids with the terms={self.terms}")

        return dfsubcts, dfsubclu

    def is_cuda_available(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        return device


    def open_pipeline(self, checkpoint:str="emilyalsentzer/Bio_ClinicalBERT",
                      task:str="zero-shot-classification") -> bool:

        self.task = task
        self.classifier, self.llm_name = None, None

        device = self.is_cuda_available()

        mat = checkpoint.split("/")
        if len(mat) != 2:
            print("Error: checkpoint must be in the form: <author>/<bert_model>")
            return False

        self.llm_name = mat[1]

        try:
            self.classifier = pipeline(task=task, model=checkpoint, device=device)
            ret = True
        except:
            print(f"Error: could not create the classifier with checkpoint={checkpoint} and task={task}")
            ret = False

        if device != 'cuda':
            print("Warning: to run the classifier one must configure the NVIDIA driver and cuda.")
            ret = False

        return ret

    def calc_zero_shot_pmid_versus_labels(self, dfpub:pd.DataFrame, candidate_labels:List,
                                          project:str, disease_prefix:str, suffix:str, len_max:int=512,
                                          set_warnings:bool=False, force:bool=False, verbose:bool=False) -> pd.DataFrame:

        if not set_warnings:
            warnings.filterwarnings("ignore")

        if self.device is None or self.device != 'cuda':
            print("Warning: to run the classifier one must configure the NVIDIA driver and cuda.")
            print("Please run open_pipeline()")
            return None

        self.dfpub = dfpub
        self.candidate_labels = candidate_labels
        self.project = project

        self.disease_prefix = disease_prefix

        fname_llm = f'llm_{project}_{disease_prefix}_all_text_using_LLM_{self.llm_name}_{suffix}.tsv'
        self.fname_llm = fname_llm
        fullname = os.path.join(self.root_llm, fname_llm)

        if os.path.exists(fullname) and not force:
            df = pdreadcsv(fname_llm, self.root_llm, verbose=verbose)
            return df

        # print("Error????", fullname)
        # raise Exception('stop: testing')

        print(f">>> total={len(dfpub)}:", end=' ')
        dic = {}
        ### todo - overlap of phrases
        for i in range(len(dfpub)):
            print(i, end=' ')
            dic[i] = {}
            dic2 = dic[i]

            pmid = dfpub.iloc[i].pmid
            dic2['pmid'] = pmid

            fname = f'{pmid}.txt'
            fullname = os.path.join(self.root_pdf_txt, fname)

            if not os.path.exists(fullname):
                print(f"Could not find '{fullname}'")
                continue

            read_text = False
            lista = read_txt(fname, self.root_pdf_txt, verbose=False)
            if lista != []:
                text = " ".join(lista)
                read_text = True if len(text.split(' ')) >= self.min_words_text else False

            if not read_text:
                print(f"Could not read '{fullname}'")
                continue

            '''---
                spliting the paper in many phrases with <= 512 characters
                text = set of many phrases
            ----'''
            phrases = text.replace("\n","").split(".")
            N_phrases = len(phrases)

            ''' merging phrases untile len(phrase) > 512 '''
            itext = -1; dic_phrase = {}; iphrase=-1; phrase = ''
            lista_phrase = []
            while(True):
                itext += 1

                if itext < N_phrases:
                    if phrase == '':
                        phrase = phrases[itext]
                        # first time, len < len_max include the phrase
                        if len(phrase) < len_max:
                            continue
                        else:
                            # else: cut the first frase and classifier_clin
                            phrase = phrase[:len_max]
                    else:
                        if itext < len(phrases):
                            if len(phrase + phrases[itext]) < len_max:
                                # if less than len_max, add new phrase and continue
                                phrase += phrases[itext]
                                continue
                            else:
                                # remove the last prase, explodes len_max
                                # classifier_clin the phrase without adding next phrase
                                itext -= 1
                        else:
                            # end of phrases
                            pass

                # print(itext, phrase, '\n')
                if phrase != '':
                    lista_phrase.append(phrase)
                    phrase = ''

                '''---- end of the while(True) loop -----'''
                if itext >= N_phrases: break

            if len(lista_phrase) > 0:
                print(f"#{len(lista_phrase)}", end=' ')
                result_list = self.classifier(lista_phrase, candidate_labels)

                dic_phrase = {}
                for iphrase in range(len(result_list)):
                    dic_phrase[iphrase] = {}
                    dic2_phrase = dic_phrase[iphrase]

                    dica = result_list[iphrase]
                    for j in range(len(dica['labels'])):
                        label = dica['labels'][j]
                        score = dica['scores'][j]
                        dic2_phrase[label] = score

                df_phrase = pd.DataFrame(dic_phrase).T
                df_phrase = df_phrase[candidate_labels]

                '''--- negative logic = 1-p
                       get the 1 - min value of each column ~ pathway'''
                df_phrase = 1-pd.DataFrame(df_phrase.min(axis=0)).T

                '''--- dic having the pmid + probabilistic pathway columns '''
                for col2 in candidate_labels:
                    dic2[col2] = df_phrase.iloc[0][col2]

        print("\n------------ end -----------")

        df = pd.DataFrame(dic).T
        pdwritecsv(df, fname_llm, self.root_llm, verbose=verbose)

        return df
