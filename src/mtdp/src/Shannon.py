#!/usr/bin/python
#!python
# -*- coding: utf-8 -*-

'''
Updated: 2023/01/02, 2022/12/22, 2022/09/28, 2022/07/25; 2022/04/25
Created: 2020/06/10
@author: Flavio Lichtenstein
@email:  flalix@gmail.com, flavio.lichtenstein@butantan.gov.br
@local:  Instituto Butantan / CENTD / Molecular Biology / Bioinformatics & Systems Biology
'''

# http://blog.notdot.net/2010/07/Getting-unicode-right-in-Python
# from __future__ import unicode_literals

import copy, os, re, random, sys, time
# from math import pow
from collections import OrderedDict
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Iterable, Set, Tuple, Any, List

from scipy.stats import t
import warnings

from markdown2 import Markdown
from IPython.core.display import HTML

import matplotlib
matplotlib.use('Agg') # Use backend agg to access the figure canvas as an RGB string and then convert it to an array and pass it to Pillow for rendering.
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import BioPythonClass
import Sequence as ms
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

from Basic import *
from util_general import *


from graphic_lib import plotly_colors, plotly_colors_proteins

b = BioPythonClass.Basic()

colors2 = ['navy', 'red', 'darkcyan', 'gold', 'mediumvioletred',
           'green', 'darkgreen', 'orange', 'olivedrab', 'bisque', 'gray', 'pink',
           'aquamarine', 'darkgreen', 'darksalmon', 'yellowgreen', 'magenta',
           'lime', 'yellow']
"""
Shannon class:
    define variables
    metadata
    consensus
    calculates
        Shannon Entropy
        Mutual Information
"""
class Shannon:
    ''' for sarscov2 - fileReffasta  = 'MT789690.1.fasta', '''
    def __init__(self, prjName:str, version:str, isGisaid:bool, isProtein:bool, root0:str,
                 countryRef:str, cityRef:str, yearRef:int, monthRef:int, descriptionRef:str,
                 protList:List, dicNucAnnot:dict, virus:str, previous_version:str=None,
                 nCount:int=500, year_max:int=None, month_max:int=None, one_month:int=None,
                 gt_then_pi_cutoff:float=0.05, cutoff_freq_pseudvar:int=3, cutoff_freq_country:int=20,
                 badAA_cutoff_perc:float=0.10, min_gaps:float=0.20, cutoff_good:float=0.25,
                 fileReffasta:str=None, file_big_fasta=None, date_mask:str='%Y-%m-%d',
                 file_template:str='table_template.html', seaviewPath:str = '../../tools/seaview'):

        '''
            Shannon class:\n
                - main functions:
                    - init_vars(self, country, protein=None, protein_name=None)
                    - init_vars_year_month(self, country, protein, protein_name, year, month)
                    - get and calculate
                        - consensus
                        - metadata (human, animal + environment)
                        - entropy & polymorphism
                        - mutations
                    - Data:
                        - open_DNA
                        - open_CDS
                        - open_protein
                        - open_nCount_sample_DNA
                        - open_nCount_sample_CDS
                        - open_nCount_sample_protein
                    - pseudo-variants get and calc
                    - variantes get and calc
                    - suggested peptides

        '''
        yearmonth = version[:6]
        """General varibles:"""
        self.prjName   = prjName
        self.yearmonth = yearmonth
        self.version   = version
        self.previous_version = previous_version

        self.isGisaid  = isGisaid
        self.virus     = virus

        self.pid = None

        # Flavio 2024/07/16
        self.protList    = protList
        self.proteinList = [x[0] for x in protList]
        self.dicNucAnnot = dicNucAnnot

        self.seaviewPath = seaviewPath

        self.nCount    = nCount
        self.gt_then_pi_cutoff    = gt_then_pi_cutoff
        self.cutoff_freq_pseudvar = cutoff_freq_pseudvar
        self.cutoff_freq_country  = cutoff_freq_country

        self.badAA_cutoff_perc = badAA_cutoff_perc
        self.min_gaps = min_gaps
        self.cutoff_good = cutoff_good

        now = datetime.now()

        if year_max is None:
            year_max  = now.year

        if month_max is None:
            month_max  = now.month

        self.year_max = year_max
        self.month_max = month_max

        self.one_month = one_month

        self.isProtein = isProtein
        self.define_isProtein(isProtein)

        self.country = None
        self.protein = None; self.protein_name = None
        self.year    = None; self.month = None

        self.root0 = root0

        self.set_filelog('pipeline.log')

        if not os.path.exists(root0):
            stri = "There is no dir '%s'"%(root0)
            log_save(stri, filename=self.filelog, verbose=True)
            raise Exception(stri)

        #-- Define Gisaid filenames -----
        base_filename = 'msa_%s_%s.fasta'

        self.base_filename = base_filename

        # --- root_gisaid ---
        self.root_gisaid          = create_dir(root0, "gisaid")
        self.root_gisaid_results  = create_dir(self.root_gisaid, 'results')
        self.root_gisaid_data     = create_dir(self.root_gisaid, 'data')

        if not os.path.exists(self.root_gisaid_results):
            stri = "There is no dir '%s'"%(self.root_gisaid_results)
            log_save(stri, filename=self.filelog, verbose=True)
            raise Exception(stri)

        self.root_templates = create_dir(self.root_gisaid, 'templates' )
        self.root_ensembl   = create_dir(self.root_gisaid, 'ensembl' )
        self.root_gisaid_ncbi      = create_dir(self.root_gisaid, 'ncbi' )
        self.fileReffasta   = fileReffasta

        stri_ref = self.fileReffasta.replace(".fasta", "").replace('.', '_')
        self.file_cut_position      = 'cut_position_%s_reference_%s.tsv'%(self.version, stri_ref)
        if self.previous_version is None:
            self.file_cut_position_previous_version = None
            self.file_metadata_previous_version     = None
        else:
            self.file_cut_position_previous_version = 'cut_position_%s_reference_%s.tsv'%(self.previous_version, stri_ref)
            self.file_metadata_previous_version     = 'metadata_%s.tsv'%(previous_version)

        if file_big_fasta is None:
            self.file_big_fasta = 'unmasked_%s.fasta'%(version)
        else:
            self.file_big_fasta = file_big_fasta

        self.file_metadata  = 'metadata_%s.tsv'%(version)

        # self.file_txt       = self.file_big_fasta.replace(".fasta", ".txt")

        self.file_meta_cym = "metadata_cym_%s_y%d_m%d.tsv"
        self.file_meta_new = "metadata_new_%s_batch_%d.tsv"
        self.file_meta_animal_env_cym = "metadata_%s_cym_%s_y%d_m%d.tsv"

        #  in self.root_gisaid_results
        self.file_meta_checked = 'metadata_checked_GISAID_%s.tsv'%(version)
        self.file_meta_animal_env_checked = 'metadata_checked_animal_environment_GISAID_%s.tsv'%(version)
        self.file_meta_summary = 'metadata_summary_GISAID_%s.tsv'%(version)
        self.dir_data          = create_dir(self.root_gisaid_results, 'data')
        self.root_samples      = create_dir(self.root_gisaid_results, 'wgs' )

        if previous_version is None:
            self.file_meta_checked_last = None
            self.file_meta_summary_last = None
        else:
            self.file_meta_checked_last = 'metadata_checked_GISAID_%s.tsv'%(previous_version)
            self.file_meta_summary_last = 'metadata_summary_GISAID_%s.tsv'%(previous_version)

        self.root_gisaid_DNA        = create_dir(self.root_gisaid_results, "dna/")
        self.root_gisaid_CDS        = create_dir(self.root_gisaid_results, "cds/")
        self.root_gisaid_protein    = create_dir(self.root_gisaid_results, "protein/")
        self.root_gisaid_animal_env = create_dir(self.root_gisaid_results, "animal/")

        self.root_gisaid_DNA_fasta   = create_dir(self.root_gisaid_DNA, 'fasta')
        self.root_gisaid_DNA_entropy = create_dir(self.root_gisaid_DNA, 'entropy')
        self.root_gisaid_DNA_sampled = create_dir(self.root_gisaid_DNA, 'sampled')
        self.root_gisaid_DNA_result  = create_dir(self.root_gisaid_DNA, 'results')
        self.root_gisaid_DNA_html    = create_dir(self.root_gisaid_DNA, 'html')

        self.fname_entropy_dna_summary = 'entropy_dna_for_%s_y%d.tsv'

        # self.root_fasta_  human    = create_dir(self.root_gisaid_CDS, 'fasta_human')
        # self.root_fasta_nothuman = create_dir(self.root_gisaid_CDS, 'fasta_not_human')

        # self.root_metadata_human    = create_dir(self.root_gisaid_CDS, 'metadata_human')
        # self.root_metadata_nothuman = create_dir(self.root_gisaid_CDS, 'metadata_not_human')

        self.root_gisaid_CDS_fasta    = create_dir(self.root_gisaid_CDS, "fasta/")
        self.root_gisaid_CDS_sampled  = create_dir(self.root_gisaid_CDS, "sampled/")
        self.root_gisaid_CDS_metadata = create_dir(self.root_gisaid_CDS, "metadata")
        self.root_gisaid_CDS_entropy  = create_dir(self.root_gisaid_CDS, "entropy/")
        self.root_gisaid_CDS_figure   = create_dir(self.root_gisaid_CDS, 'figure/')
        self.root_gisaid_CDS_html     = create_dir(self.root_gisaid_CDS, "html/")

        self.root_gisaid_animal_env_cds      = create_dir(self.root_gisaid_animal_env, "cds/")
        self.root_gisaid_animal_env_dna      = create_dir(self.root_gisaid_animal_env, "dna/")
        self.root_gisaid_animal_env_metadata = create_dir(self.root_gisaid_animal_env, "metadata")
        self.root_gisaid_animal_env_entropy  = create_dir(self.root_gisaid_animal_env, "entropy/")
        self.root_gisaid_animal_env_figure   = create_dir(self.root_gisaid_animal_env, 'figure/')
        self.root_gisaid_animal_env_html     = create_dir(self.root_gisaid_animal_env, "html/")

        self.root_gisaid_prot_fasta   = create_dir(self.root_gisaid_protein, 'fasta/')
        self.root_gisaid_prot_sampled = create_dir(self.root_gisaid_protein, "sampled/")
        self.root_gisaid_prot_entropy = create_dir(self.root_gisaid_protein, 'entropy/')
        self.root_gisaid_prot_figure  = create_dir(self.root_gisaid_protein, 'figure/')
        self.root_gisaid_prot_html    = create_dir(self.root_gisaid_protein, "html/")

        self.root_iedb_manual    = create_dir(self.root_gisaid, 'data_manual' )
        # --- iedb ---
        self.root_iedb           = create_dir(self.root_gisaid, 'result_IEDB_mutations' )
        self.root_iedb_summary   = create_dir(self.root_iedb,   'summary')
        self.root_iedb_pseudovar = create_dir(self.root_iedb,   'pseudo_variants')
        self.root_iedb_html      = create_dir(self.root_iedb,   'html')
        self.root_iedb_figure    = create_dir(self.root_iedb,   'figure' )

        self.root_params = create_dir(self.root_iedb, 'params')
        self.root_params_cg  = create_dir(self.root_params, 'cg')
        self.root_params_cai = create_dir(self.root_params, 'cai')
        self.root_params_mkt = create_dir(self.root_params, 'mkt')

        self.file_template = file_template
        self.date_mask     = date_mask

        self.dfmeta, self.dfanimeta, self.dfmeta_previous= None, None, None

        self.dfcut  = None
        self.seq_nuc_consensus = None

        self.dfmeta_cy   = None
        self.mseq_DNA_cy = None
        self.mseq_CDS_cym = None


        self.fname_mut_freq_variant = "mutated_posis_for_%s_y%d_m%d_variant_%s.xlsx"
        self.fname_pseudo_variant_main_mutated = "pseudo_variant_main_mutated_posis_for_%s_%s_y%d_m%d_variant_%s.tsv"
        self.fname_summ_mut_cy_variant = "mutations_summary_for_%s_y%d_variant_%s.xlsx"
        self.fname_possible_pseudo = "possible_pseudo_variants_for_%s_%s_y%d.tsv"
        # self.fname_summ_mut_reference   = "mutations_summary_for_%s_year%d_reference_%s.xlsx"
        self.file_analytics_cy           = "analytics_mutations_for_%s_year%d.tsv"
        self.fname_variants_of_the_year  = "variants_of_year_%s_%d.tsv"

        ''' %(protein_name, year, month, s_countries)
            in os.path.join(self.root_iedb_summary, fname)
        '''
        self.fname_most_voted_anal = "most_voted_peptides_analytics_protein_%s_y%d_m%d_%s.tsv"
        self.fname_most_voted_summ = "most_voted_peptides_summary_protein_%s_y%d_m%d_%s.tsv"

        ''' ---- for reference like Wuhan  ----'''
        self.countryRef = countryRef
        self.cityRef    = cityRef
        self.yearRef    = yearRef
        self.monthRef   = monthRef
        self.descriptionRef = descriptionRef
        self._idRef     = None
        self.variantRef = 'unknown'

        self.refID = None

        #---- Define fasta header constants
        self.HEADER_ID = 0
        self.HEADER_COUNT = 1
        self.HEADER_COUTRY = 2
        self.HEADER_YEAR = 3
        self.HEADER_MONTH = 4
        self.HEADER_COLL_DATE = 5
        self.HEADER_HOST = 6
        self.HEADER_CONCERN = 7
        self.HEADER_PANGO = 8
        self.HEADER_PANGO_COMPLETE = 9
        self.HEADER_CLADE = 10
        self.HEADER_VARIANT = 11
        self.HEADER_SUBVARIANT = 12
        self.HEADER_SUBVARIANT1 = 13
        self.HEADER_SUBVARIANT2 = 14
        self.HEADER_REGION = 15

        self.dfvar = None

        self.open_template()

    def set_path(self, root, _dir=None):
        """set / create path (dir)"""


    def define_isProtein(self, isProtein):
        """Define if is protein or dna"""
        self.isProtein = isProtein
        self.dna_protein = 'Protein' if isProtein else 'DNA'

    def set_filelog(self, fname:str, root:str='./logs'):

        filefull = os.path.join(root, fname)

        if not os.path.exists(root):
            os.mkdir(root)

        self.filelog = filefull

    def init_vars(self, country, protein=None, protein_name=None):
        """Init - country & protein/protein_name"""
        self.seq_nuc_consensus = None
        self.mseq_cym = None

        country = country.replace(" ","_")
        self.set_filelog(f'pipeline_shannon_{country}.log')

        ''' if only protein changed must not load again mseq_CDS_cym '''
        if self.country is None:
            self.dfmeta_cy   = None
            self.mseq_DNA_cy = None
            self.mseq_CDS_cym = None
        else:
            if self.country != country:
                self.dfmeta_cy   = None
                self.mseq_DNA_cy = None
                self.mseq_CDS_cym = None

        self.country   = country
        self.protein   = protein
        if protein_name is None and protein is not None:
            protein_name = protein
        self.protein_name = protein_name

        self.file_DNA_country  = self.base_filename%(self.virus, self.country)
        # self.file_DNA_cy       = self.file_DNA_country.replace('.fasta', '_y%d.fasta')

        self.file_DNA_cym     = self.file_DNA_country.replace('.fasta', '_y%d_m%d.fasta')
        self.file_entropy_cym = "entropy_" + self.file_DNA_cym.replace(".fasta", ".pickle")

        self.file_CDS_cym0            = "cds_%s_%s_%s_y%d_m%d.fasta"
        self.file_CDS_animal_env_cym0 = "cds_%s_%s_%s_%s_y%d_m%d.fasta"

        if protein is not None:
            base_filename = self.base_filename.replace("msa_", "cds_")
            self.file_CDS_cym  = base_filename.replace('.fasta', '_%s.fasta')%(self.virus, self.country, protein)
            self.file_CDS_cym  = self.file_CDS_cym.replace('.fasta', '_y%d_m%d.fasta')

            self.file_CDS_ref  = self.file_CDS_cym%(self.yearRef, self.monthRef)

            base_filename = self.base_filename.replace("msa_", "prot_")
            self.file_prot_cym = base_filename.replace('.fasta', '_%s.fasta')%(self.virus, self.country, protein)
            self.file_prot_cym           = self.file_prot_cym.replace('.fasta', '_y%d_m%d.fasta')
            self.file_prot_cym_reference = self.file_prot_cym.replace('.fasta', '_reference.fasta')
            self.file_prot_cym_entropy   = "entropy_" + self.file_prot_cym.replace(".fasta", ".pickle")
        else:
            self.file_CDS_cym  = None
            self.file_CDS_ref  = None

            self.file_prot_cym           = None
            self.file_prot_cym_reference = None
            self.file_prot_cym_entropy   = None

        # self.dir_fasta_country     = create_dir(self.root_gisaid_CDS_fasta, self.country)
        # self.dir_fasta_DNA_country = create_dir(self.dir_fasta_country, "fasta")

        self.dicPiList          = []
        self.dicNList           = []
        self.HShannonList       = []
        self.SeHShannonList     = []
        # self.HShannonCorrList   = []
        # self.SeHShannonCorrList = []
        self.df_ids_descriptions = None
        self.dfvar = None

    def init_vars_year_month(self, country, protein, protein_name, year, month):
        if self.one_month is not None and isinstance(self.one_month, int):
            month = self.one_month

        self.init_vars(country, protein, protein_name)
        self.year = year
        self.month = month
        self.dfvar = None


    def consensus_get_aa_from_seqs(self, seqs):
        if isinstance(seqs[0], str):
            seqs = np.array([list(x) for x in seqs])
        else:
            seqs = np.array(seqs)

        nrows, ncols = seqs.shape

        ref_consensus = ''
        for j in range(ncols):
            nuc = best_amino_acid(seqs[:,j], with_gaps=True)
            ref_consensus += '-' if nuc is None else nuc

        self.seq_aa_consensus = ref_consensus
        return ref_consensus

    ''' calc_Seq_consensus --> get_any_reference_consensus -->
        get_consensus_from_seqs --> consensus_get_nucleotide_from_seqs
        seqs = np.array(list)
    '''
    def consensus_get_nucleotide_from_seqs(self, seqs):
        if isinstance(seqs[0], str):
            seqs = np.array([list(x) for x in seqs])
        else:
            seqs = np.array(seqs)

        nrows, ncols = seqs.shape

        ref_consensus = ''
        for j in range(ncols):
            nuc = best_nucleotide(seqs[:,j], with_gaps=True)
            ref_consensus += '-' if nuc is None else nuc

        dif = len(ref_consensus)%3
        if dif != 0:
            ref_consensus = ref_consensus[:-dif]

        self.seq_nuc_consensus = ref_consensus

        return ref_consensus

    ''' consensus_get
            input: seqs = list of sequences (string)
            output: a string called consensus
    '''
    def consensus_get(self, seqs, fix=False):
        if isinstance(seqs[0], str):
            seqs = np.array([list(x) for x in seqs])

        seqs = np.array(seqs)
        nrows, ncols = seqs.shape

        if self.seq_nuc_consensus is None:
            stri = "Error: consensus fix_seq_row_nucleotides() - seq_nuc_consensus is None"
            log_save(stri, filename=self.filelog, verbose=True)
            return None

        if fix:
            seqs = np.array([  list(self.consensus_fix_seq_row_nucleotides(seqs[i], seqs) ) for i in range(nrows) ])
        else:
            seqs = np.array([  list(seqs[i]) for i in range(nrows) ])

        consensus = ''
        for j in range(ncols):
            nuc = best_nucleotide(seqs[:,j], with_gaps=True)
            consensus += '-' if nuc is None else nuc

        return consensus

    def consensus_fix_sequence(self, seqs):
        seqs = np.array(seqs)
        nrows, ncols = seqs.shape

        seqs = np.array([  list(self.consensus_fix_seq_row_nucleotides(seqs[i], seqs) ) for i in range(nrows) ])

        return seqs

    ''' bad nucs ["-", "N", "Y", "S", "R", "K", "W", "M", "B", "H", "D", "G", "V"]
    '''
    def replace_wrong_nuc(self, nuc):
        return nuc if nuc in ['A', 'C', 'G', 'T'] else '-'

    def set_consensus(self, seq_nuc_consensus):
        self.seq_nuc_consensus = seq_nuc_consensus

    def adjust_length_multi3(self, seq_records):
        if seq_records is None or len(seq_records)==0:
            return seq_records, None

        ncol = len(seq_records[0].seq)
        dif = ncol%3

        if dif > 0:
            for i in range(len(seq_records)):
                seq_records[i].seq = Seq(str(seq_records[i].seq)[:-dif])

        seqs = np.array([ list(seq_rec.seq) for seq_rec in seq_records])

        return seq_records, seqs


    def get_nucleotide_consensus_from_seqs(self, seqs, with_gaps=False, multi3=False):
        nrows, ncols = seqs.shape

        seq_nuc_consensus = ''
        for j in range(ncols):
            nuc = best_nucleotide(seqs[:,j], with_gaps=with_gaps)
            seq_nuc_consensus += '-' if nuc is None else nuc

        if multi3:
            dif = len(seq_nuc_consensus)%3
            if dif > 0:
                seq_nuc_consensus = seq_nuc_consensus[:-dif]

        self.seq_nuc_consensus = seq_nuc_consensus
        return seq_nuc_consensus

    def consensus_fix_seq_row_nucleotides(self, seq_row, seqs, with_gaps=True, multi3=False, verbose=False):
        ''' used in:
        a) consensus_get(self, seqs, fix=False) <br>
        b) consensus_fix_sequence(self, seqs)

        there is an line with: seqs[:,pos] --> therefore, seqs must be an anp.array
        '''

        if self.seq_nuc_consensus is None:
            stri = "Error: consensus fix_seq_row_nucleotides() - seq_nuc_consensus is None"
            log_save(stri, filename=self.filelog, verbose=True)
            return None

        seqs = np.array(seqs)

        seq_row = "".join([self.replace_wrong_nuc(nuc) for nuc in seq_row])

        for match in re.finditer('-', seq_row):
            posi = list(match.span())[0]
            if isinstance(posi, int):
                posi = [posi]
            else:
                posi = list(posi)

            for pos in posi:
                nuc = best_nucleotide(seqs[:,pos], with_gaps=with_gaps)
                if nuc is None: nuc = self.seq_nuc_consensus[pos]
                seq_row = seq_row[:pos] + nuc + seq_row[(pos+1):]

        ''' adjusting length '''
        if multi3:
            dif = len(seq_row)%3
            if dif > 0:
                seq_row = seq_row[:-dif]

        return seq_row

    ''' --- testing fix --------------------------------------------------------
        seq_nuc_consensus = 'ACACGTACGGTA'
        seq_row           = 'A-ACGT-CGGTA'
        seqs = np.array([list(seq_row), list(seq_row)])
        seqs

        for match in re.finditer('-', seq_row):
            posi = list(match.span())[0]
            # print("####", posi, type(posi))

            if isinstance(posi, int):
                posi = [posi]
            else:
                posi = list(posi)

            for pos in posi:
                nuc = best_nucleotide(seqs[:,pos], with_gaps=True)
                if nuc is None: nuc = seq_nuc_consensus[pos]
                seq_row = seq_row[:pos] + nuc + seq_row[(pos+1):]

        seq_row, seq_row==seq_nuc_consensus
    -------------------------------------------------------------------------'''

    def open_template(self):
        self.template = ''
        filefull = os.path.join(self.root_templates, self.file_template)

        if not os.path.exists(filefull):
            stri = "HTML Template file does not exists '%s'"%(filefull)
            log_save(stri, filename=self.filelog, verbose=True)
            return False

        try:
            with open(filefull, 'r') as fh:
                self.template = fh.read()
        except:
            stri = "Could not read HTML Template file '%s'"%(filefull)
            log_save(stri, filename=self.filelog, verbose=True)
            return False

        return True

    def open_big_metadata(self):
        root_metadata = os.path.join(self.root0, "gisaid/sarscov2_%s/metadata")%(self.yearmonth)
        filemetadata  = 'metadata_%s.tsv'%(self.version)
        self.dfbigmeta = pdreadcsv(filemetadata, root_metadata)
        if self.dfbigmeta is None or len(self.dfbigmeta) == 0:
            stri = "Error: dfbigmeta"
            log_save(stri, filename=self.filelog, verbose=True)
            return False

        stri = "ok - dfbigmeta"
        log_save(stri, filename=self.filelog, verbose=True)

        return True

    def open_metadata_checked(self, only_human:bool=True) -> bool:
        return self.open_gisaid_metadata_checked(only_human)

    def open_gisaid_metadata_checked(self, only_human:bool=True) -> bool:
        self.only_human = only_human

        if self.isGisaid:
            ''' if Gisaid metadata - not Zika (comes from NCBI) '''
            if only_human:
                return self.open_human_gisaid_metadata_checked()
            else:
                return self.open_animal_env_gisaid_metadata_checked()

        print("This procedure belongs only to Gisaid metadata.")
        return False


    ''' changed from root_gisaid_results to root_gisaid_data '''
    def open_human_gisaid_metadata_checked(self, verbose=False):

        if self.dfmeta is None:
            filefull = os.path.join(self.root_gisaid_data, self.file_meta_checked)

            if os.path.exists(filefull):
                stri = "reading checked metadata, be patient ..."
                log_save(stri, filename=self.filelog, verbose=True)

                self.dfmeta = pdreadcsv(self.file_meta_checked, self.root_gisaid_data)
                if self.dfmeta is None or self.dfmeta.empty:
                    self.dfmeta = None
                    stri = "Error: dfmeta is empty"
                    log_save(stri, filename=self.filelog, verbose=True)
                    return False

                stri = "Ok - dfmeta checked"
                log_save(stri, filename=self.filelog, verbose=True)
            else:
                stri = "Warning: dfmeta was not created: '%s'"%(filefull)
                stri += "\nRunning check_gisaid_metadata()"
                log_save(stri, filename=self.filelog, verbose=True)

                ''' building gisaid metadata '''
                self.check_gisaid_metadata(verbose=verbose)

                if os.path.exists(filefull):
                    stri = "reading checked metadata, be patient ..."
                    log_save(stri, filename=self.filelog, verbose=True)

                    self.dfmeta = pdreadcsv(self.file_meta_checked, self.root_gisaid_data)
                    if self.dfmeta is None or self.dfmeta.empty:
                        self.dfmeta = None
                        stri = "Error: dfmeta is empty"
                        log_save(stri, filename=self.filelog, verbose=True)

                        return False

                    stri = "Ok - dfmeta checked"
                    log_save(stri, filename=self.filelog, verbose=True)
                else:
                    stri = "Error: dfmeta was not created: '%s'"%(filefull)
                    stri += "\nRun check_gisaid_metadata()"
                    log_save(stri, filename=self.filelog, verbose=True)
                    self.dfmeta = None
                    return False

            stri = ">> dfmeta has %.3f million lines"%(len(self.dfmeta)/1000000)
            log_save(stri, filename=self.filelog, verbose=True)
        else:
            stri = "dfmeta already opened"
            log_save(stri, filename=self.filelog, verbose=verbose)

        return True

    def open_animal_env_gisaid_metadata_checked(self):

        if self.dfanimeta is None:
            filefull = os.path.join(self.root_gisaid_data, self.file_meta_animal_env_checked)

            if os.path.exists(filefull):
                stri = "reading checked metadata ..."
                log_save(stri, filename=self.filelog, verbose=True)

                self.dfanimeta = pdreadcsv(self.file_meta_animal_env_checked, self.root_gisaid_data)
                if self.dfanimeta is None or self.dfanimeta.empty:
                    self.dfanimeta = None
                    stri = "Error: dfanimeta is empty"
                    log_save(stri, filename=self.filelog, verbose=True)

                    return False

                stri = "Ok - dfanimeta checked"
                log_save(stri, filename=self.filelog, verbose=True)
            else:
                stri = "Error: dfanimeta was not created: '%s'"%(filefull)
                stri += "\nRun check_gisaid_metadata() for animal & environment"
                log_save(stri, filename=self.filelog, verbose=True)

                return False
        else:
            stri = "dfanimeta already opened"
            log_save(stri, filename=self.filelog, verbose=verbose)

        return True

    def open_metadata(self, only_human:bool=True, verbose:bool=True) -> bool:
        self.only_human = only_human

        if self.isGisaid:
            ''' if Gisaid metadata - not Zika (comes from NCBI) '''        
            if only_human:
                return self.open_human_gisaid_metadata(verbose=verbose)
            else:
                return self.open_animal_env_gisaid_metadata(verbose=verbose)

        ''' NCBI '''
        return self.open_ncbi_metadata()


    def open_ncbi_metadata(self, verbose:bool=False) -> bool:
        filefull = os.path.join(self.root_gisaid_data, self.file_metadata)

        if not os.path.exists(filefull):
            print(f"Could not find '{filefull}'. Please, run zika_02_check_NCBI_metadata.ipynb")
            return False
        
        self.dfmeta = pdreadcsv(self.file_metadata, self.root_gisaid_data, verbose=verbose)
        return self.dfmeta is not None


    def fix_human_ncbi_metadata(self, verbose:bool=True) -> bool:
        if not self.open_ncbi_metadata(verbose=verbose):
            return False

        cols = list(self.dfmeta.columns)

        if 'variant' in cols and 'subvariant1' in cols:
            print("Already done")
            return True

        cols_must_exist = ['country', 'year', 'month', 'coll_date', 'host', 'concern', 'clade_gisaid',\
                           'variant', 'subvariant', 'subvariant1', 'subvariant2', 'region']

        no_cols = [x for x in cols_must_exist if x not in cols]

        term_list = ['zika-', 'Haiti/', 'ZIKV/', 'Zika-', 'Zika', 'zika virus/', 'zika virus',
                     'homo sapiens/', 'homo sapiens', 'zikv/', 'h.sapiens-wt/', 'virus/', 
                     'H.sapiens-wt/BRA/2016/', 'BRA/', 'Homo sapiens/PAN/', 'Homo sapiens/']

        def calc_subvariant(variant):
            if not isinstance(variant, str):
                return variant

            for term in term_list:
                if term in variant: variant = variant.replace(term, '')

            variant_low = variant.lower()
            if 'asia' in variant_low:
                return 'Asian'
            if 'africa' in variant_low:
                return 'African'

            return variant

        def calc_subvariant1(variant):
            if not isinstance(variant, str):
                return variant

            for term in term_list:
                if term in variant: variant = variant.replace(term, '')

            return variant.strip()


        def replace_variant(x):
            x = str(x)
            x = x.replace('H.sapiens-tc/', '')
            x = x.replace('human/', '')
            x = x.replace('Aedes aegypti/', '')
            x = x.replace('Dominican Republic/', '')
            x = x.replace('Homo_sapiens/', '')
            x = x.replace('MR 766', 'African')
            x = x.replace('ZIKV_', '')
            x = x.replace('ZIKV', '')
            x = x.replace('Aedes sp./', '')
            x = x.replace('Brazil_', '')
            x = x.replace('Brazil-', '')
            x = x.replace('H.sapiens/Brazil/', '')
            x = x.replace('Homo Sapiens/', '')
            x = x.replace('MEX/', '')
            x = x.replace('MEX_', '')
            x = x.replace('mosquito/', '')
            x = x.replace('Natal RGN', 'RGN')
            x = x.replace('Yucatan/', '')
            x = x.replace('-Nicaragua/', '')
            x = x.replace('Nicaragua/', '')

         
            return x.strip()            
            
        lista_sub  = [calc_subvariant(variant)  for variant in self.dfmeta.variant]
        lista_sub1 = [calc_subvariant1(variant) for variant in self.dfmeta.variant]

        if 'subvariant'   in no_cols: self.dfmeta['subvariant']  = lista_sub
        if 'subvariant1'  in no_cols: self.dfmeta['subvariant1'] = lista_sub1
        if 'subvariant2'  in no_cols: self.dfmeta['subvariant2'] = lista_sub1
        if 'clade_gisaid' in no_cols: self.dfmeta['clade_gisaid'] = None

        cols = np.array(self.dfmeta.columns)
        cols = ['id', 'count_fasta', 'type', 'variant', 'subvariant', 'subvariant1', 'subvariant2', 
                'concern', 'clade', 'taxon', 'coll_date', 'year', 'month', 'month2', 'country',
               'state', 'city', 'genotype', 'mol_type', 'host', 'strain',
               'isolate', 'isolation-source', 'culture-collection', 'clone',
               'lat-lon', 'attributes', 'country_state', 'gbkey', 'genome',
               'phase', 'note', 'identified-by', 'collected-by', 'serotype',
               'serogroup', 'gb', 'region', 'start', 'end', 'score', 'strand', 'clade_gisaid']
        self.dfmeta = self.dfmeta[cols]

        self.dfmeta['variant'] = [replace_variant(x) for x in self.dfmeta.variant]

        ret = pdwritecsv(self.dfmeta, self.file_metadata, self.root_gisaid_data, verbose=verbose)
        return ret


    def open_human_gisaid_metadata(self, verbose:bool=True) -> bool:
        if self.dfmeta is None:
            filefull = os.path.join(self.root_gisaid_data, self.file_meta_checked)
            if os.path.exists(filefull):
                stri = "reading checked metadata ... "
                log_save(stri, filename=self.filelog, verbose=True, end = ' - ')
                self.dfmeta = pdreadcsv(self.file_meta_checked, self.root_gisaid_data)
                if self.dfmeta is None or len(self.dfmeta) == 0:
                    self.dfmeta = None
                    stri = "Error: dfmeta checked is empty"
                    log_save(stri, filename=self.filelog, verbose=True)
                    return False
                else:
                    stri = "Ok - dfmeta not checked"
                    log_save(stri, filename=self.filelog, verbose=True)
            else:
                stri = "reading metadata ... "
                log_save(stri, filename=self.filelog, verbose=True, end = ' - ')
                self.dfmeta = pdreadcsv(self.file_metadata, self.root_gisaid_data)
                if self.dfmeta is None or len(self.dfmeta) == 0:
                    self.dfmeta = None
                    stri = "Error: dfmeta not checked is empty"
                    log_save(stri, filename=self.filelog, verbose=True)
                    return False
                else:
                    stri = "Ok - dfmeta not checked"
                    log_save(stri, filename=self.filelog, verbose=True)
        else:
            stri = "dfmeta already opened"
            log_save(stri, filename=self.filelog, verbose=verbose)

        return True

    def open_animal_env_gisaid_metadata(self, verbose:bool=True) -> bool:
        if self.dfanimeta is None:
            filefull = os.path.join(self.root_gisaid_data, self.file_meta_animal_env_checked)
            if os.path.exists(filefull):
                stri = "reading checked metadata ... "
                log_save(stri, filename=self.filelog, verbose=True, end = ' - ')
                self.dfanimeta = pdreadcsv(self.file_meta_animal_env_checked, self.root_gisaid_data)
                if self.dfanimeta is None or len(self.dfanimeta) == 0:
                    self.dfanimeta = None
                    stri = "Error: dfanimeta checked is empty"
                    log_save(stri, filename=self.filelog, verbose=True)
                    return False
                else:
                    stri  = "Ok - dfanimeta not checked"
                    log_save(stri, filename=self.filelog, verbose=True)
            else:
                stri = "reading metadata ... "
                log_save(stri, filename=self.filelog, verbose=True, end = ' - ')
                self.dfanimeta = pdreadcsv(self.file_metadata, self.root_gisaid_data)
                if self.dfanimeta is None or len(self.dfanimeta) == 0:
                    self.dfanimeta = None
                    stri = "Error: dfanimeta not checked is empty"
                    log_save(stri, filename=self.filelog, verbose=True)
                    return False
                else:
                    stri = "Ok - dfanimeta not checked"
                    log_save(stri, filename=self.filelog, verbose=True)
        else:
            stri = "dfanimeta already opened"
            log_save(stri, filename=self.filelog, verbose=verbose)

        return True


    def open_original_metadata(self):

        if self.dfmeta is None:
            stri = "reading metadata ... "
            log_save(stri, filename=self.filelog, verbose=True, end = ' - ')
            self.dfmeta = pdreadcsv(self.file_metadata, self.root_gisaid_data)
            if self.dfmeta is None or len(self.dfmeta) == 0:
                self.dfmeta = None
                stri = "Error: dfmeta not checked is empty: self.dfmeta"
                log_save(stri, filename=self.filelog, verbose=True)
                return False
            else:
                stri = "Ok - original dfmeta"
                log_save(stri, filename=self.filelog, verbose=True)
        else:
            stri = "original dfmeta already opened."
            log_save(stri, filename=self.filelog, verbose=verbose)

        return True

    def open_previous_version_metadata(self):

        if self.file_metadata_previous_version is None:
            self.dfmeta_previous = None
            stri = "Warning: dfmeta previous version was not defined."
            log_save(stri, filename=self.filelog, verbose=True)
            return False

        if self.dfmeta_previous is None:
            stri = "reading previous version metadata ... "
            log_save(stri, filename=self.filelog, verbose=True, end = ' - ')
            self.dfmeta_previous = pdreadcsv(self.file_metadata_previous_version, self.root_gisaid_data)
            if self.dfmeta_previous is None or self.dfmeta_previous.empty:
                self.dfmeta_previous = None
                stri = "Warning: dfmeta previous version is empty"
                log_save(stri, filename=self.filelog, verbose=True)
                return False
            else:
                stri = "Ok - previous version dfmeta"
                log_save(stri, filename=self.filelog, verbose=True)
        else:
            stri = "previous version dfmeta already opened."
            log_save(stri, filename=self.filelog, verbose=verbose)

        return True

    '''
        deprecated - get_metadata_country_year(self, year, verbose=False)
    '''
    def open_metadata_cym(self):

        file_meta_cym = self.file_meta_cym%(self.country, self.year, self.month)
        filefull_meta_cym = os.path.join(self.root_gisaid_CDS_metadata, file_meta_cym)

        if not os.path.exists(filefull_meta_cym):
            stri = "Error: metadata cym was not created: '%s'"%(filefull_meta_cym)
            log_save(stri, filename=self.filelog, verbose=True)

            self.dfmeta = None
            return False

        dfmeta = pdreadcsv(file_meta_cym, self.root_gisaid_CDS_metadata)
        self.dfmeta = dfmeta

        return True


    def open_dfcut(self, depth:int=15, large:int=20, start_offset:int=10, 
                   end_offset:int=100, cutoffNNN:float=.67, force:bool=False, verbose:bool=False) -> pd.DataFrame:

        if self.dfcut is None:
            stri = "Getting dfcut ...."
            log_save(stri, filename=self.filelog, verbose=verbose)

            self.dfcut = self.calc_dfcut_all_proteins_posi_based_on_reference_big_fasta(depth=depth,
                                large = large, start_offset=start_offset, end_offset=end_offset,
                                cutoffNNN=cutoffNNN, force=force, verbose=verbose)

        return self.dfcut


    def realign_seqProt_x_reference(self, filefull_prot, nrecs=5, seaviewPath=None, verbose=False):

        if seaviewPath is None:
            seaviewPath = self.seaviewPath

        mseqRef = self.open_protein_reference()
        if mseqRef.seq_records == []:
            return False

        ''' merge 5 records reference + mseqProt.seq_records '''
        seq_records = self.mseqProt.seq_records

        for i in range(nrecs):
            seq_records.append(mseqRef.seq_records[i])

        ''' save merged seq_records + reference '''
        ret = self.mseqProt.writeSequences(seq_records, filefull_prot, verbose=verbose)
        if not ret:
            stri = "realign_seqProt_x_reference(): could not save merged fastas."
            log_save(stri, filename=self.filelog, verbose=verbose)
            return False

        ret = self.mseqProt.align_seqs_muscle(filefull_prot, filefull_prot, seaviewPath, force=True, server=False, verbose=verbose)
        stri = "end ... %s\n"%(str(ret) )
        log_save(stri, filename=self.filelog, verbose=verbose)

        ''' delete the last 5 records '''
        ret = self.mseqProt.readFasta(filefull_prot, showmessage=False)
        if not ret:
            stri = "Could not reread aligned fasta."
            log_save(stri, filename=self.filelog, verbose=verbose)
            return False

        seq_records = self.mseqProt.seq_records[:-nrecs]

        ret = self.mseqProt.writeSequences(seq_records, filefull_prot, verbose=verbose)
        if not ret:
            stri = "Could not save aligned fasta."
            log_save(stri, filename=self.filelog, verbose=verbose)
            return False

        return True


        # reviewed: Flavio 02/06/2022 - file_entropy_cym, self.root_gisaid_CDS_entropy
    def entropy_calcHSannon_bias_correction(self, seqs, numOfLetters=2, force=False, verbose=False):

        file_entropy_cym = self.file_entropy_cym%(self.year, self.month)
        filefull = os.path.join(self.root_gisaid_CDS_entropy, file_entropy_cym)

        if os.path.exists(filefull) and not force:
            try:
                # read and return
                dic = pdloaddic(file_entropy_cym, path=self.root_gisaid_CDS_entropy, verbose=True)

                self.dicPiList          = dic['dicPiList']
                self.dicNList           = dic['dicNList']
                self.HShannonList       = dic['HShannonList']
                self.SeHShannonList     = dic['SeHShannonList']
                # self.HShannonCorrList   = dic['HShannonCorrList']
                # self.SeHShannonCorrList = dic['SeHShannonCorrList']
                return

            except:
                # if any error, calc again
                pass

        # if not fulfill the gaps, there may be bad chars
        if self.dna_protein == "DNA":
            valid = b.getDnaNucleotides()
        elif self.dna_protein == "Protein" or self.dna_protein.lower() == "amino acid":
            valid = b.getAA()
        else:
            raise Exception("Error: only DNA and Protein tables are accepted (type = 'DNA' or 'Protein')")

        ''' there can be gaps !!! '''
        valid.append('-')

        nrow = len(seqs)
        ncol = len(seqs[0])
        maxL = ncol-numOfLetters+1

        self.dicPiList = []; self.dicNList=[]
        self.HShannonList = []
        self.SeHShannonList = []
        # self.HShannonCorrList = []
        # self.SeHShannonCorrList = []

        if verbose:
            t1 = datetime.now()
            stri = "Start calc Shannon: for %s %s %d/%d"%(self.country, self.protein_list, self.year, self.month)
            log_save(stri, filename=self.filelog, verbose=verbose)

        for i in range(0,maxL,numOfLetters):
            dicPi = {}; dicN={}
            varPi = 0; hShan = 0

            n = 0 # new column counter
            for row in range(nrow):
                #-- bad chars?
                if np.sum([True for x in seqs[row,i:(i+numOfLetters)] if not x in valid]) > 0:
                    continue
                n += 1

                try:
                    dicN[ "".join(seqs[row,i:(i+numOfLetters)]) ] += 1
                except:
                    dicN[ "".join(seqs[row,i:(i+numOfLetters)]) ] = 1

            if n == 0:
                stri = "Error: bad sequece for column %d to %d for %s %s %d/%d"%(i, i+numOfLetters-1, self.country, self.protein_list, self.year, self.month)
                log_save(stri, filename=self.filelog, verbose=True)
                raise Exception(stri)

            for key in dicN.keys():
                pi = dicN[key] / n

                dicPi[key] = pi
                if (pi != 0.) and (pi != 1.):
                    hShan -= pi * np.log2(pi)
                    varPi += (1. + np.log2(pi))**2  * pi * (1.-pi)

            B = len(dicPi.keys())

            ''' Roulston Eq. 13 '''
            # hShanCorr = hShan + ((B-1)/ (2*n))

            VarCorr = 0
            ''' Roulston Eq. 40 '''
            for key in dicPi.keys():
                pi = dicPi[key]
                if (pi != 0.) and (pi != 1.):
                    VarCorr += (np.log2(pi)+hShan)**2 * pi * (1-pi)

            SeHShannon     = hShan * np.sqrt(varPi/n)
            # SeHShannonCorr = np.sqrt(VarCorr/n)

            self.dicPiList.append(dicPi)
            self.dicNList.append(dicN)
            self.HShannonList.append(hShan)
            self.SeHShannonList.append(SeHShannon)
            # self.HShannonCorrList.append(hShanCorr)
            # self.SeHShannonCorrList.append(SeHShannonCorr)

        if verbose:
            t2 = datetime.now()
            stri = "Calc Shannon: %.2f seconds"%((t2-t1).seconds)
            log_save(stri, filename=self.filelog, verbose=verbose)

        dic = {}
        dic['dicPiList']          = self.dicPiList
        dic['dicNList']           = self.dicNList
        dic['HShannonList']       = self.HShannonList
        dic['SeHShannonList']     = self.SeHShannonList
        # dic['HShannonCorrList']   = self.HShannonCorrList
        # dic['SeHShannonCorrList'] = self.SeHShannonCorrList

        ret = pddumpdic(dic, self.file_entropy_cym, path=self.root_gisaid_CDS_entropy, verbose=True)
        if ret and verbose:
            t3 = datetime.now()
            stri = "Saving '%s' and returning in %.2f seconds"%(file_entropy_cym, (t3-t2).seconds)
            log_save(stri, filename=self.filelog, verbose=verbose)


    ''' since it receives mseq --> who call has the responsability to chose the correta fasta data
        fast_entropy_calcHSannon_bias_correction_1pos --> fast calcHShannon_1pos '''
    def entropy_fast_calcHShannon_1pos(self, mseq, h_threshold=4, add_gaps=True, verbose=False, warning=False):
        # if not fulfill the gaps, there may be bad chars
        self.dicPiList    = []; self.dicNList = []
        self.HShannonList = []; self.SeHShannonList = []
        self.HShannonCorrList = []; self.SeHShannonCorrList = []
        self.df_ids_descriptions  = None

        if self.dna_protein == "DNA":
            valid = b.getDnaNucleotides()
        elif self.dna_protein == "Protein" or self.dna_protein == "Amino acid":
            valid = b.getAA()
        else:
            stri = "Error: only DNA and Protein tables are accepted (type = 'DNA' or 'Protein') for %s %s %d/%d - %s"%(self.country, self.protein_list, self.year, self.month, self.variant_subvar)
            log_save(stri, filename=self.filelog, verbose=True)
            return False

        ''' there can be gaps !!! '''
        if add_gaps: valid.append('-')
        '''--- add stop codons ---'''
        valid.append('*')

        #---------------- metadata for filtering - dic['metadata'] = dfn ----------------------
        descs = [seqrec.description  for seqrec in mseq.seq_records]
        ids   = [desc.split('||')[0] for desc   in descs]

        if len(ids) == 0:
            stri = "Error?? No records found in fast calcHShannon_1pos for %s %s %d/%d - %s"%(self.country, self.protein_list, self.year, self.month, self.variant_subvar)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return False

        dfa = pd.DataFrame(data={'id': ids, 'desc': descs})
        dfa['i'] = dfa.index
        dfa.index = dfa['id']
        self.df_ids_descriptions = dfa

        seqs = np.array(mseq.seqs)
        nrow, ncol = seqs.shape

        ''' remove stop codon if last char '''
        if seqs[0][-1] == '*':
            ncol -= 1

        t1 = datetime.now()
        if verbose: print("Start calc Shannon 1 letter")

        # --- looping each column ---------
        msa_error_msg = False
        for jj in range(ncol):
            dic = char_frequency(seqs[:,jj])

            badkeys = [c for c in dic.keys() if c not in valid]
            for badk in badkeys:
                del(dic[badk])

            ''' gaps are allowed
                '-' must be the major count
            '''
            if len(dic) > 1 and '-' in dic.keys():
                num_c = dic['-']
                total = np.sum(list(dic.values()))

                ''' if gap is less than 5% '''
                if num_c/total < 0.05:
                    del(dic['-'])

            # if self.protein == 'S': print(jj, dic)

            counts = np.array(list(dic.values()))
            dicPi  = OrderedDict(); dicN=OrderedDict()

            if len(counts) == 1:
                k = list(dic.keys())[0]
                dicPi[k] = 1
                dicN[k]  = counts[0]

                self.dicPiList.append(dicPi)
                self.dicNList.append(dicN)
                self.HShannonList.append(0)
                self.SeHShannonList.append(0)
                # print(">>> jj %d - len %d"%(jj, len(self.dicPiList) ))
                continue

            n      = np.sum(counts)
            percs  = counts / n
            hShan = 0; varPi = 0

            kcount = 0;
            for k in dic.keys():
                pi = percs[kcount]
                dicPi[k] = pi
                dicN[k]  = counts[kcount]
                if (pi != 0.) and (pi != 1.):
                    hShan -= pi * np.log2(pi)
                    varPi += (1. + np.log2(pi))**2  * pi * (1.-pi)

                kcount += 1

                ''' warnings '''
                if warning:
                    if hShan > h_threshold:
                        print('>>> jj', jj, hShan)
                        print("percs", percs, '\n')

            ''' Roulston Eq. 13 '''
            '''
            B = len(dic.keys())
            if hShan > 0:
                hShanCorr = hShan + ((B-1)/ (2*n))
            else:
                hShanCorr = 0
            if hShanCorr > 4:
                print('>>> jj1', jj, hShan)self.dic_meta_country.keys()
                print('>>> jj2', jj, "B", B, "n", n, "hShanCorr", hShanCorr)
                print("percs", percs, '\n')

            VarCorr = 0
            # Roulston Eq. 40
            for key in dicPi.keys():
                pi = dicPi[key]
                if (pi != 0.) and (pi != 1.):
                    VarCorr += (np.log2(pi)+hShan)**2 * pi * (1-pi)
            '''

            SeHShannon = 0 if hShan == 0 else hShan * np.sqrt(varPi/n)
            # SeHShannonCorr = np.sqrt(VarCorr/n)

            self.dicPiList.append(dicPi)
            self.dicNList.append(dicN)
            self.HShannonList.append(hShan)
            self.SeHShannonList.append(SeHShannon)
            # self.HShannonCorrList.append(hShanCorr)
            # self.SeHShannonCorrList.append(SeHShannonCorr)
            # print(">>> jj %d - len %d"%(jj, len(self.dicPiList) ))

        if verbose:
            t2 = datetime.now()
            print("Calc fast calcHSannon 1pos:", round( (t2-t1).microseconds / 1000), "ms")

        return True

    def entropy_plot_protein_variant_cym(self, country, protein, protein_name, year, month,
                                         factor=1000, filetype = "png", dpi=300, printplot=True,
                                         onlycalc=False, verbose=False):

        self.init_vars_year_month(country, protein, protein_name, year, month)
        self.define_isProtein(True)

        multiDic = self.entropy_get_protein_year_month_multidic(year, month, verbose=False)

        if multiDic is None or len(multiDic) == 0:
            return [], [], [], [], [], [], []

        variant_list, subvariant_list, pango_list, totalH_list, meanH_list, L_list, nrow_list = \
        [], [], [], [], [], [], []

        for variant_subvar in multiDic.keys():
            entDic = multiDic[variant_subvar]

            if entDic is None or len(entDic) == 0:
                continue

            variant, subvariant, pango = variant_subvar.split(" ")

            df   = entDic['df']
            nrow = entDic['nSamples']

            if df is None or len(df) == 0:
                continue

            hs = df['y']
            if len(hs) == 0:
                continue

            title = "Protein entropy plot for %s - %s %s %s\n%s %d/%d"%(country, variant, subvariant, pango, protein_name, year, month)

            filefig = os.path.join(self.root_gisaid_prot_figure, title_replace(title) + "." + filetype)
            totalH, meanH, L = self.entropy_plot(hs, nrow, factor, title, filename=filefig,
                                                filetype=filetype, dpi=dpi, printplot=printplot, onlycalc=onlycalc)

            pango_list.append(pango)
            variant_list.append(variant)
            subvariant_list.append(subvariant)
            totalH_list.append(totalH)
            meanH_list.append(meanH)
            L_list.append(L)
            nrow_list.append(nrow)

        return pango_list, variant_list, subvariant_list, totalH_list, meanH_list, L_list, nrow_list

    def consensus_get_by_variant_posi(self, variant_subvar, year, month, posis, force=False, verbose=False):

        multiDic = self.entropy_get_protein_year_month_multidic(year, month, verbose=verbose)

        self.pepdic, self.mutations, self.consensus = None, None, None
        try:
            entDic = multiDic[variant_subvar]
        except:
            stri = "consensus_get_by_variant_posi: multiDic could not find '%s' for %s %s %d/%d"%\
                    (variant_subvar, self.country, self.protein_name, self.year, self.month)
            log_save(stri, filename=self.filelog, verbose=True)
            return []

        if entDic is None or len(entDic) == 0:
            return []

        self.entDic = entDic

        self.pepdic, self.mutations, self.consensus = self.define_seq_peptides(entDic, verbose=verbose)

        try:
            aas = [self.consensus[pos-1] for pos in posis]
        except:
            print("---------------------------")
            print(len(self.consensus), self.protein_name, self.country, self.year, self.month)
            print("---------------------------")
            print(posis)
            print("---------------------------")
            print(self.consensus)
            print("---------------------------")

        return aas

    # https://www.uniprot.org/uniprot/P59594
    def define_seq_peptides(self, entDic, verbose=False):

        df = entDic['df']

        if df is None or len(df) == 0:
            stri = "define_seq_peptides(): entDic's df is empty. Please calc entropy for '%s' for %s %s %d/%d"%\
                    (variant_subvar, self.country, self.protein_name, self.year, self.month)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return None, None, None

        pepdic = OrderedDict()
        consensus = ''; consensus2 = ''; mutations = []

        for i in range(len(df)):
            mat = df.iloc[i].aas.split("; ")

            if len(mat) == 1:
                aa = mat[0].split(" ")[0]
                perc = 100
            else:
                perc = 0
                for m in mat:
                    mat2 = m.split(" ")
                    aa2 = mat2[0]
                    perc2 = float(mat2[1].replace("%",""))
                    if perc2 >= 1 and perc2 < 50:
                        if verbose: print("mutation pos: %d, %s %f"%(i, aa2,  perc2))
                        mutations.append(i)
                    if perc2 > perc:
                        perc = perc2
                        aa = aa2

            consensus += aa

        ini = 0; mut=0
        for mut in mutations:
            pepdic[ini] = {}
            pepdic[ini]['end'] = mut-1
            pepdic[ini]['peptide'] = consensus[ini:mut]
            ini = mut + 1

        pepdic[ini] = {}
        pepdic[ini]['end'] = mut-1
        pepdic[ini]['peptide'] = consensus

        if verbose:
            print("")
            for key, dic in pepdic.items():
                print(key, dic['end'], dic['peptide'])

        if verbose: print("\n\nconsensus %d: %s ..."%(len(consensus), consensus[:20]))

        return pepdic, mutations, consensus

    def entropy_get_protein_year_month_multidic(self, year, month, verbose=False):
        ''' data in: pickle entropy year month
            dic[variant_subvar] = {"df": df, "nSamples": nSamples, "maxVal": maxVal,
                                  "dicPiList": dicPiList, "dicNList": dicNList, "df_ids_descriptions": self.df_ids_descriptions}
            return loaddic()
        '''
        filee = self.file_prot_cym_entropy%(year, month)
        filefull = os.path.join(self.root_gisaid_prot_entropy, filee)

        if not os.path.exists(filefull):
            stri = "entropy get_protein_year_month(): could not find '%s'"%(filefull)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return None

        return loaddic(filee, path=self.root_gisaid_prot_entropy, verbose=verbose)

    ''' remove protList '''
    def entropy_many_calc_protein_cym(self, country:str, years_months:List,
                                     ini:int=0, length:int=-1, h_threshold:int=4, 
                                     warning:bool=False, force:bool=False, verbose:bool=False):

        if verbose: print(">>>", country, end=' ')
        
        prev_year = -1
        for year, month in years_months:
            
            if year != prev_year:
                if verbose: print(">>", year, end=' ')
                prev_year = year

            if verbose: print('>', month, end=' ')

            for protein, protein_name in self.protList:
                if verbose: print(protein, end=' ')
            
                _ = self.entropy_get_calc_protein_cym(country, protein, protein_name, year, month,
                                                      warning=warning, return_dic=False,
                                                      force=force, verbose=verbose)

        if verbose: print("")


    def entropy_get_calc_protein_cym(self, country:str, protein:str, protein_name:str, 
                                     year:int, month:int, ini:int=0, length:int=-1, 
                                     h_threshold:int=4, warning:bool=False,
                                     return_dic:bool=False, force:bool=False, verbose:bool=False) -> dict:

        try:
            logbackup = self.filelog
        except:
            logbackup = 'protein_log_pipe_04.tsv'

        self.set_filelog(logbackup)

        self.init_vars_year_month(country, protein, protein_name, year, month)

        t1 = datetime.now()
        stime = t1.strftime('%Y-%b-%d, %H h %M min %S sec')

        if self.pid == None:
            self.pid = 1

        stri = "%s\tstart\t%d\t%s\t%s\t%d/%d\tNone\tNone\tentropy_get_calc_protein_cym"%(stime, self.pid, country, protein_name, year, month)
        log_save(stri, filename=self.filelog, verbose=False)

        self.isProtein = True
        self.define_isProtein(self.isProtein)

        file_prot_cym_sampled = self.file_prot_cym%(self.year, self.month)
        file_prot_cym_sampled = file_prot_cym_sampled.replace(".fasta", "_sampled_%d_nCount.fasta")%(self.nCount)
        filefull_samp = os.path.join(self.root_gisaid_prot_sampled, file_prot_cym_sampled)

        ''' pickle entropy year month '''
        filee      = self.file_prot_cym_entropy%(year, month)
        filefull_h = os.path.join(self.root_gisaid_prot_entropy, filee)

        # print("\n<<<", os.path.exists(filefull_h), filefull_h, '\n', os.path.exists(filefull_samp), filefull_samp)
        if os.path.exists(filefull_h) and os.path.exists(filefull_samp) and not force:
            stri = "%s\tend-notfound\t%d\t%s\t%s\t%d/%d\tNone\tNone\tprotein and entropy files already calculated"%(stime, self.pid, country, protein_name, year, month)
            log_save(stri, filename=self.filelog, verbose=False)

            dic = loaddic(filee, path=self.root_gisaid_prot_entropy, verbose=verbose) if return_dic else None
            return dic

        stri = "%s\tend\t%d\t%s\t%s\t%d/%d\tNone\tNone\tfast_convert_CDS_to_protein start"%(stime, self.pid, country, protein_name, year, month)
        log_save(stri, filename=self.filelog, verbose=False)

        ''' calc self.mseqProt '''
        ret, mseqProt = super().fast_convert_CDS_to_protein(force=False, read_fasta=False, verbose=verbose)

        if not ret:
            stri = "%s\tend-notfound\t%d\t%s\t%s\t%d/%d\tNone\tNone\tfast_convert_CDS_to_protein PROBLEMS"%(stime, self.pid, country, protein_name, year, month)
            log_save(stri, filename=self.filelog, verbose=False)

            stri = "entropy get_calc_protein_cym(): could NOT convert CDS to protein for %s %s %d/%d"%(self.country, protein_name, year, month)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return None

        stri = "%s\tend\t%d\t%s\t%s\t%d/%d\tNone\tNone\tfast_convert_CDS_to_protein OK"%(stime, self.pid, country, protein_name, year, month)
        log_save(stri, filename=self.filelog, verbose=False)

        ''' sample nCount '''
        ret, mseqProt = self.open_nCount_sample_protein_cym_already_set(fulfilled=True, verbose=verbose, force=False)
        self.mseqProt = mseqProt

        if not ret or mseqProt is None or len(mseqProt.seq_records) == 0:
            stri = "%s\tstart\t%d\t%s\t%s\t%d/%d\tNone\tNone\topen_nCount_sample_protein_cym_already_set PROBLEM"%(stime, self.pid, country, protein_name, year, month)
            stri = "entropy get_calc_protein_cym(): could not sample protein for %s %s %d/%d"%(self.country, protein_name, year, month)
            log_save(stri, filename=self.filelog, verbose=True)
            return None

        stri = "%s\tend\t%d\t%s\t%s\t%d/%d\tNone\tNone\topen_nCount_sample_protein_cym_already_set OK"%(stime, self.pid, country, protein_name, year, month)
        log_save(stri, filename=self.filelog, verbose=False)
        dic = self.entropy_calc_protein_variant_cym(mseqProt, ini, length, warning, h_threshold, verbose=verbose)

        stri = "%s\tend\t%d\t%s\t%s\t%d/%d\tNone\tNone\tentropy_calc_protein_variant_cym"%(stime, self.pid, country, protein_name, year, month)
        log_save(stri, filename=self.filelog, verbose=False)

        self.filelog = logbackup

        return dic if return_dic else None

    def open_DNA_cym(self, country, year, month, verbose=False):

        self.init_vars(country, None, None)

        ret, mseq = self.open_DNA_year_month(year, month, verbose=verbose)
        if not ret:
            return False, mseq

        return ret, mseq

    def open_DNA_year_month(self, year, month, verbose=False):
        self.year = year
        self.month = month

        return self.open_DNA_cym_already_set(self, verbose=verbose)

    def open_DNA_cym_already_set(self, verbose=False):
        self.mseq = None
        self.L    = None

        file_DNA_cym = self.file_DNA_cym%(self.year, self.month)
        filefull = os.path.join(self.root_gisaid_DNA_fasta, file_DNA_cym)

        if not os.path.exists(filefull):
            if verbose: print("Could not find '%s'"%(filefull))
            return False, self.mseq

        mseq = ms.MySequence(self.prjName, root=self.root_gisaid_DNA_fasta)
        ret = mseq.readFasta(filefull, showmessage=verbose)

        self.mseq = mseq
        self.L    = len(mseq.seq_records[0].seq)

        return ret, mseq

    def open_CDS_cym(self, country, year, month, protein, protein_name, verbose=False):

        self.init_vars(country, protein, protein_name)

        ret, mseq_CDS_cym = self.open_CDS_year_month(year, month, verbose=verbose)

        return ret, mseq_CDS_cym

    '''  get_CDS_cym --> open_CDS_year_month '''
    def open_CDS_year_month(self, year, month, verbose=False):
        self.year = year
        self.month = month

        return open_CDS_cym_already_set(self, verbose=verbose)

    def open_CDS_cym_already_set(self, verbose=False):
        self.mseq_CDS_cym = None
        self.L    = None

        file_CDS_cym  = self.file_CDS_cym%(self.year, self.month)
        filefull = os.path.join(self.root_gisaid_CDS_fasta, file_CDS_cym )

        if not os.path.exists(filefull):
            print("Could not find '%s'"%(filefull))
            return False, mseq_CDS_cym

        mseq_CDS_cym = ms.MySequence(self.prjName, root=self.root0)
        ret = mseq_CDS_cym.readFasta(filefull, showmessage=verbose)
        if not ret:
            return False, mseq_CDS_cym

        L=len(mseq_CDS_cym.seq_records[0].seq)
        L=L-L%3

        self.mseq_CDS_cym = mseq_CDS_cym
        self.L = L

        return True, mseq_CDS_cym

    def open_protein_CYM(self, country, year, month, protein, protein_name, verbose=False):

        self.init_vars(country, protein, protein_name)

        return self.open_protein_CYM_year_month(year, month, verbose=verbose)

    '''  get_CDS_cym --> open_CDS_year_month '''
    def open_protein_CYM_year_month(self, year, month, verbose=False):
        self.year = year
        self.month = month

        return open_protein_CYM_already_set(self, verbose=verbose)

    def open_protein_CYM_already_set(self, verbose=False):
        self.mseq_prot = None
        self.L = None

        file_prot_cym  = self.file_prot_cym%(self.year, self.month)
        filefull = os.path.join(self.root_gisaid_prot_fasta, file_prot_cym )

        if not os.path.exists(filefull):
            if verbose: print("Could not find '%s'"%(filefull))
            return False, self.mseq_prot

        mseq_prot = ms.MySequence(self.prjName, root=self.root_gisaid_prot_fasta)
        ret = mseq_prot.readFasta(filefull, showmessage=verbose)

        self.mseq_prot = mseq_prot
        self.L = len(mseq_prot.seq_records[0].seq)

        return ret, mseq_prot


    ''' sample_big_fasta --> open_nCount_sample_DNA_cym_already_set '''
    def open_nCount_sample_DNA_cym(self, country, year, month, force=False, verbose=False):
        self.init_vars_year_month(country, None, None, year, month)
        return self.open_nCount_sample_DNA_cym_already_set(force=force, verbose=verbose)

    def open_nCount_sample_DNA_cym_year_month(self, year, month, force=False, verbose=False):
        self.year = year
        self.month = month

        return self.open_nCount_sample_DNA_cym_already_set(force=force, verbose=verbose)


    def open_nCount_sample_DNA_cym_already_set(self, force=False, verbose=False):
        file_DNA_cym = self.file_DNA_cym%(self.year, self.month)
        file_DNA_cym_sample = file_DNA_cym.replace(".fasta", "_sampled_%d_nCount.fasta")%(self.nCount)
        filefull_samp = os.path.join(self.root_gisaid_DNA_sampled, file_DNA_cym_sample)

        if not os.path.exists(filefull_samp) or force:
            ret, mseq = self.calc_fast_nCount_sample_DNA_cym(verbose=verbose, force=force)
        else:
            mseq = ms.MySequence(self.prjName, root=self.root_gisaid_DNA_sampled)
            ret = mseq.readFasta(filefull_samp, showmessage=verbose)

        if not ret or mseq is None or len(mseq.seq_records) == 0:
            return False, None

        mseq.seqs = np.array(mseq.seqs)
        self.mseq_sampled = mseq

        return ret, mseq

    def open_nCount_sample_CDS_cym(self, country, year, month, protein, protein_name, force=False, verbose=False):
        self.init_vars_year_month(country, protein, protein_name, year, month)
        return self.open_nCount_sample_CDS_cym_already_set(force=force, verbose=verbose)

    def open_nCount_sample_CDS_cym_already_set(self, verbose=False, force=False):

        file_CDS_cym_sampled = self.file_CDS_cym%(self.year, self.month)
        file_CDS_cym_sampled = file_CDS_cym_sampled.replace(".fasta", "_sampled_%d_nCount.fasta")%(self.nCount)

        filefull_samp = os.path.join(self.root_gisaid_CDS_sampled, file_CDS_cym_sampled)

        if not os.path.exists(filefull_samp) or force:
            ret, mseq = self.calc_fast_nCount_sample_CDS_cym(verbose=verbose, force=force)
        else:
            mseq = ms.MySequence(self.prjName, root=self.root_gisaid_CDS_sampled)
            ret = mseq.readFasta(filefull_samp, showmessage=verbose)

        if not ret or mseq is None or len(mseq.seq_records) == 0:
            return False, None

        mseq.seqs = np.array(mseq.seqs)
        self.mseq_sampled = mseq

        return ret, mseq

    def open_nCount_sample_protein_cym(self, country, year, month, protein, protein_name,
                                       fulfilled=False, force=False, verbose=False):

        self.init_vars_year_month(country, protein, protein_name, year, month)

        return self.open_nCount_sample_protein_cym_already_set(fulfilled=fulfilled, verbose=verbose)

    def open_nCount_sample_protein_cym_already_set(self, fulfilled=False, force=False, verbose=False):

        file_prot_cym_sampled = self.file_prot_cym%(self.year, self.month)
        file_prot_cym_sampled = file_prot_cym_sampled.replace(".fasta", "_sampled_%d_nCount.fasta")%(self.nCount)

        if fulfilled:
            file_prot_cym_sampled = file_prot_cym_sampled.replace(".fasta", "_fulfilled.fasta")

        filefull_samp = os.path.join(self.root_gisaid_prot_sampled, file_prot_cym_sampled)
        self.filefull_samp = filefull_samp

        if not os.path.exists(filefull_samp) or force:
            ret, mseq_prot = self.calc_fast_nCount_sample_protein_cym(verbose=verbose, force=force)
        else:
            mseq_prot = ms.MySequence(self.prjName, root=self.root_gisaid_prot_sampled)
            ret = mseq_prot.readFasta(filefull_samp, showmessage=verbose)

        if not ret or mseq_prot is None or len(mseq_prot.seq_records) == 0:
            stri = "Error: mseq_prot not found: %s %s for %d/%d"%(self.country, self.protein_name, self.year, self.month)
            log_save(stri, filename=self.filelog, verbose=True)
            return False, None

        try:
            mseq_prot.seqs = np.array(mseq_prot.seqs)
        except:
            mseq_prot.seqs = np.array([ list(seq_rec.seq) for seq_rec in mseq_prot.seq_records])

        self.mseq_prot_sampled = mseq_prot

        return ret, mseq_prot

    def variants_open_annotated_metadata(self, clear_unknown=False, verbose=False):

        file_meta_cym  = self.file_meta_cym%(self.country, self.year, self.month)
        filefull_meta_cym  = os.path.join(self.root_gisaid_CDS_metadata, file_meta_cym )

        if not os.path.exists(filefull_meta_cym):
            stri = "Could not find '%s'"%(filefull_meta_cym)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return None, []

        dfm = pdreadcsv(file_meta_cym, self.root_gisaid_CDS_metadata)

        if dfm is None or len(dfm) == 0:
            stri = "Could not read '%s'"%(filefull_meta_cym)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return None, []

        sub2variants = self.sub2variants_unique(dfm)
        sub2variants = self.variants_clear(sub2variants, clear_unknown=clear_unknown)

        return dfm, sub2variants

    def sub2variants_unique(self, df):

        if df is None or df.empty:
            return []

        sub2variants = [df.iloc[i].variant + ' ' + df.iloc[i].subvariant + ' ' + df.iloc[i].pango for i in range(len(df))]
        sub2variants = list(np.unique(sub2variants))
        sub2variants.sort()

        return sub2variants

    def variants_clear(self, sub2variants, clear_unknown=False, verbose=False):

        if clear_unknown:
            sub2variants = [x for x in sub2variants if isinstance(x, str) and x != 'unknown unknown']
        else:
            sub2variants = [x for x in sub2variants if isinstance(x, str)]

        if len(sub2variants) == 0:
            stri = "Error: in variants_clear() no sub2variants found."
            log_save(stri, filename=self.filelog, verbose=verbose)
            return []

        return sub2variants

    '''
    subvariant2_ok = variant_subvar_check_in_seq_rec
    '''
    def variant_subvar_check_in_seq_rec(self, seq_rec, variant_subvar):
        mat = seq_rec.id.split("||")
        pango      = mat[self.HEADER_PANGO]
        variant    = mat[self.HEADER_VARIANT]
        subvariant = mat[self.HEADER_SUBVARIANT]

        variant_seq = variant + ' ' + subvariant + ' ' + pango
        # print(">>>", variant_subvar, variant_seq)
        return variant_subvar == variant_seq

    def entropy_calc_protein_variant_cym(self, mseqProt, ini, length, warning, h_threshold, verbose=False):
        if length == -1:
            end0 = -1
        else:
            end0 = ini+length

        dfm, sub2variants = self.variants_open_annotated_metadata()

        if len(sub2variants) == 0:
            stri = "entropy calc_protein_variant_cym(): no variants found for %s %s %d/%d"%(self.country, self.protein_name, self.year, self.month)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return None

        dic = {}
        for variant_subvar in sub2variants:
            self.variant_subvar = variant_subvar
            variant, subvariant, pango = variant_subvar.split(' ')

            seq_records = [seq_rec for seq_rec in mseqProt.seq_records if self.variant_subvar_check_in_seq_rec(seq_rec, variant_subvar)]

            if len(seq_records) < 3:
                stri = "entropy calc_protein_variant_cym(): there are no sufficient seq_recs %d for variant %s for %s %s %d/%d"%(len(seq_records), variant_subvar, self.country, self.protein_name, self.year, self.month)
                log_save(stri, filename=self.filelog, verbose=verbose)
                continue

            seqs = [ list(seq_rec.seq) for seq_rec in seq_records]

            mseq = ms.MySequence(self.prjName, root=self.root_gisaid_prot_entropy)
            mseq.seq_records = seq_records
            mseq.seqs = np.array(seqs)

            ret = self.entropy_fast_calcHShannon_1pos(mseq, h_threshold=h_threshold, add_gaps=True, verbose=verbose, warning=warning)
            if not ret:
                stri = "entropy calc_protein_variant_cym(): error in calcHShannon_1pos() for variant %s for %s %s %d/%d"%(variant_subvar, self.country, self.protein_name, self.year, self.month)
                log_save(stri, filename=self.filelog, verbose=verbose)
                return None

            dicPiList = self.dicPiList
            dicNList  = self.dicNList
            hs        = self.HShannonList
            maxVal    = np.max(hs)

            maxi = len(hs)

            if end0 == -1:
                end = maxi
            elif end0 > maxi:
                end = maxi
            else:
                end = end0

            dicPiList = dicPiList[ini : end]
            dicNList  = dicNList[ ini : end]
            hs        = hs[ini : end]

            labelx = []; _seqx = []; _seqn = []; count = 1
            for jj in range(len(dicPiList)):
                #---- percent
                dic2 = dicPiList[jj]
                val = "; ".join(["%s %.2f%%"%(k, dic2[k]*100) for k in dic2.keys() if dic2[k] != 0])
                _seqx.append(val)

                valks = []
                for k in dic2.keys():
                    if dic2[k] == 0: continue

                    if dic2[k] == 1:
                        valks = [k]
                        break

                    valks.append("%s %.2f%%"%(k, dic2[k]*100) )

                labelx.append(str(count) + '-' + "; ".join(valks))

                #---- n or #
                dic3 = dicNList[jj]
                val = "; ".join(["%s %d"%(k, dic3[k]) for k in dic3.keys() if dic3[k] != 0])
                _seqn.append(val)

                count += 1

            seqx = np.arange(ini, end)
            n        = len(hs)
            nSamples = len(mseqProt.seqs)

            df = pd.DataFrame({'country': [self.country]*n, 'variant': [variant]*n, \
                               'subvariant': [subvariant]*n, 'pango': [pango]*n, \
                               'year': [self.year]*n, 'month': [self.month]*n, 'x': seqx, 'y': hs,
                               'aas': _seqx, 'nns': _seqn})

            dic[variant_subvar] = {"df": df, "nSamples": nSamples, "maxVal": maxVal,
                                   "dicPiList": dicPiList, "dicNList": dicNList, "df_ids_descriptions": self.df_ids_descriptions}

        if len(dic) < 3:
            stri = "There are no sufficient seqs to calcuate entropy for %s %s %d/%d"%(self.country, self.protein_name, self.year, self.month)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return None

        '''
        stri = "There is sufficient seqs to calcuate entropy for %s %s %d/%d"%(self.country, self.protein_name, self.year, self.month)
        print(stri)
        print("-------------------------")
        print(dic.keys())
        print("-------------------------")
        raise Exception('stop')
        '''
        filee = self.file_prot_cym_entropy%(self.year, self.month)
        ret = pddumpdic(dic, filee, path=self.root_gisaid_prot_entropy, verbose=verbose)
        return dic

    #--- return the sequences (i's) between dates ---------------
    def entropy_filter_hs_by_date(self, dfmeta, start_date, end_date, is_string=True, verbose=False):
        date_mask = self.date_mask

        if dfmeta.shape[0] == 0:
            print("dfmeta is empty")
            return []

        if is_string:
            try:
                start_date = datetime.strptime(start_date[:10], date_mask)
            except:
                start_date = None

            try:
                end_date = datetime.strptime(end_date[:10], date_mask)
            except:
                end_date = None

            if start_date is None and end_date is None:
                print("none")
                return []

        df2 = dfmeta.copy()
        df2 = df2.sort_values('date')

        df2 = df2[ [True if len(str(x)) == 10 else False for x in df2.date] ]
        if len(df2) == 0:
            print("Bad date format in metadata table")
            return []

        df2 = df2[ [test_date_Ymd(x) for x in df2.date] ]
        if len(df2) == 0:
            print("Bad date format in metadata table")
            return []

        df2.date = [datetime.strptime(x, '%Y-%m-%d') for x in df2.date]
        df2 = df2.sort_values('date')

        if not start_date is None:
            df2 = df2[ df2.date >= start_date ]
        if len(df2) == 0:
            return []

        if not end_date is None:
            df2 = df2[ df2.date < end_date ]
        if len(df2) == 0:
            return []

        if verbose:
            print('## dates:', start_date, end_date)
            print("### len(i)", len(list(df2.i)), '\n')
            # print(",".join([str(x) for x in df2.i]))

        return list(df2.i)


    def entropy_plot(self, hs, nrow, factor=1000, title='Entropy',
                    xlabel = "protein residues", ylabel = 'H (bits)', color='navy',
                    zoomRegion=None, moltype="Protein", filename=None,
                    filetype='png', dpi=300, printplot=True, onlycalc=False,
                    orientation='landscape', verbose=False):

        # zoomRegion = [min, max]
        if isinstance(zoomRegion, list) and len(zoomRegion) == 2:
            mini = zoomRegion[0]; maxi=zoomRegion[1]
        else:
            mini = 0; maxi = len(hs)

        seqx = np.arange(mini, maxi)
        L = len(hs)

        totalH = np.sum(hs)
        meanH  = totalH * factor / L

        if onlycalc:
            return totalH, meanH, L

        fig = plt.figure(num=None, figsize=(8, 6), dpi=dpi, facecolor='w', edgecolor='k')
        plt.plot(seqx, hs, color=color)

        if factor==1:
            unit = 'bits'
        elif factor==1000:
            unit = 'mbits'
        else:
            unit = "Factor is 1 or 1000 ???"
            # print(unit)

        title += "\nTotal H = %.2f bits, mean H = %.2f %s, %d sequences, length = %d"%(totalH, meanH, unit, nrow, L)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title);

        if printplot: plt.show()

        if filename:
            plt.savefig(filename, format=filetype, dpi=dpi, facecolor='w', edgecolor='w', orientation=orientation);
            if verbose: print("File saved: '%s'"%(filename))

        plt.close()

        return totalH, meanH, L

    def entropy_protein_variant_prepare_table(self, country, protein, protein_name,
                                              years_months, factor=1000, force=False, verbose=False):

        fname = 'protein_entropy_cym_%s_%s_from_%d_%d_to_%d_%d.tsv'% \
                 (country, protein_name, years_months[0][0], years_months[0][1], \
                  years_months[-1][0], years_months[-1][1])
        filefull = os.path.join(self.root_gisaid_prot_entropy, fname)

        if os.path.exists(filefull) and not force:
            df = pdreadcsv(fname, self.root_gisaid_prot_entropy)
            self.df = df
            return df

        dic = {}; count = -1
        prev_year = -1
        for year, month in years_months:
            if year != prev_year:
                print(">", year, end=' ')
                prev_year = year
            print(month, end=' ')

            pango_list, variant_list, subvariant_list, totalH_list, meanH_list, L_list, nrow_list = \
                self.entropy_plot_protein_variant_cym(country, protein, protein_name, year, month,
                                                      factor=factor,  filetype = "png", dpi=300,
                                                      printplot=False, onlycalc=True, verbose=verbose)

            for i in range(len(variant_list)):
                variant = variant_list[i]
                subvariant = subvariant_list[i]
                totalH = totalH_list[i]

                if totalH is None: continue

                meanH = meanH_list[i]
                L = L_list[i]
                nrow = nrow_list[i]

                count += 1
                dic[count] = {}
                dic2 = dic[count]

                dic2['country'] = self.country
                dic2['variant'] = variant
                dic2['subvariant'] = subvariant
                dic2['year']    = year
                dic2['month']   = month
                dic2['protein'] = protein_name

                if totalH is not None:
                    dic2['totalH'] = totalH
                    dic2['meanH'] = meanH
                    dic2['unit'] = 'bits' if factor == 1 else 'mbits'
                    dic2['len_protein'] = L
                    dic2['samples'] = nrow
                else:
                    dic2['totalH'] = None
                    dic2['meanH'] = None
                    dic2['unit'] = None
                    dic2['len_protein'] = None
                    dic2['samples'] = 0

        df = pd.DataFrame(dic).T
        self.df = df
        df['year-month'] = ["%s-%s"%(str(df.iloc[i].year), str(df.iloc[i].month) if df.iloc[i].month > 9 else '0'+str(df.iloc[i].month)) for i in range(len(df))]

        self.df = df
        if len(df) == 0:
            return None

        ret = pdwritecsv(df, fname, self.root_gisaid_prot_entropy)
        return df

    ''' entropy_plot_protein_variant_sub2_evolution --> entropy_plot_protein_variant_evolution '''
    def entropy_plot_protein_variant_evolution(self, countries, protein, protein_name,
                               years_months, cutoff_samples = 3, filter_unknown_var=True, factor=1000,
                               width=800, height=600, fontsize=12, fontcolor='navy',
                               force=False, verbose=False):

        if factor != 1 and factor != 1000:
            print('Factor must be 1 or 1000')
            return None, None

        title = 'Shannon entropy time-evolution' + \
                "<br>for %s"%(protein_name)
        minyear, minmonth = years_months[0]
        maxyear, maxmonth = years_months[-1]

        title += " from %d/%d to %d/%d"%(minyear, minmonth, maxyear, maxmonth)

        df_list = []
        for country in countries:
            if verbose or force: print("\n>>>", country, end=' ')

            df = self.entropy_protein_variant_prepare_table(country, protein, protein_name,
                                                            years_months, factor=factor,
                                                            force=force, verbose=verbose)
            if df is None or len(df) == 0:
                continue

            if cutoff_samples is not None:
                df = df[df.samples >= cutoff_samples]
            if filter_unknown_var:
                df = df[df.variant != 'unknown']

            df_list.append(df)

        dff = pd.concat(df_list, sort=False)

        if dff is None:
            print("No plot for protein: %s"%(protein_name))
            return None, None

        traces=[]

        for (country, variant, subvariant, pango), data in dff.groupby(['country', 'variant', 'subvariant', 'pango']):
            texts = ["%s<br>Variant: %s - %s %s<br>%d samples"%(country, variant, subvariant, pango,\
                     data.iloc[i].samples) for i in range(len(data))]
            traces.append(go.Scatter(x=data['year-month'], y=data.meanH,
                                     text=texts,
                                     hovertemplate = "%{x}<br><h> %{y:.2f} mbits</br>%{text}",
                                     name="%s: %s - %s"%(country, variant, subvariant), mode='lines+markers'))

        fig = go.Figure(data=traces)

        fig.update_layout(
                    autosize=True,
                    title=title,
                    width=width,
                    height=height,
                    # template=template,
                    # margin=margin,
                    font=dict(
                        family="Arial, bold, monospace",
                        size=fontsize,
                        color=fontcolor
                    ),
                    xaxis_title="years-months",
                    xaxis         = dict(tickangle=45, tickfont=dict(family='Arial', color='navy', size=12) ),
                    yaxis_title   = 'bits' if factor == 1 else 'mbits',
                    paper_bgcolor = "whitesmoke",
                    plot_bgcolor  = "whitesmoke", # lightgrey ivory gainsboro whitesmoke lightsteelblue 'lightcyan' 'azure', white, lightgrey, snow ivory beige powderblue
                    showlegend    = True
                )

        figfullname = os.path.join(self.root_gisaid_prot_html, prepare_figname(title, 'html'))
        if verbose: print(">>> HTML saved:", figfullname)
        fig.write_html(figfullname)

        return dff, fig


    ''' similar to Shannon.entropy_plot_protein_variant_evolution '''
    def entropy_plot_collected_sample_evolution(self, countries, protein, protein_name,
                                        years_months, whichPlot='both', factor=1000,
                                        title = 'samples per country-subvariants time-evolution',
                                        width=800, height=600, fontsize=12, fontcolor='navy',
                                        filetype = "png", dpi=300, printplot=False, onlycalc=True,
                                        verbose=False):
        dic = {}; count = -1
        for country in countries:
            print(">>>", country, end=' ')
            prev_year = -1
            for (year, month) in years_months:
                if year != prev_year:
                    prev_year = year
                    print(year, end='-')
                print(month, end=' ')

                pango_list, variant_list, subvariant_list, totalH_list, meanH_list, L_list, nrow_list = \
                    self.entropy_plot_protein_variant_cym(country, protein, protein_name, year, month,
                                                          factor=factor, filetype = "png", dpi=300,
                                                          printplot=False, onlycalc=True, verbose=verbose)
                for i in range(len(variant_list)):
                    variant = variant_list[i]
                    subvariant = subvariant_list[i]
                    totalH = totalH_list[i]
                    if totalH is None: continue

                    meanH = meanH_list[i]
                    L = L_list[i]
                    nrow = nrow_list[i]

                    count += 1
                    dic[count] = {}
                    dic2 = dic[count]

                    dic2['country'] = country
                    dic2['variant'] = variant
                    dic2['subvariant'] = subvariant
                    dic2['year'] = year
                    dic2['month'] = month
                    dic2['protein'] = protein_name

                    if totalH is not None:
                        dic2['totalH'] = totalH
                        dic2['meanH'] = meanH
                        dic2['len_protein'] = L
                        dic2['samples'] = nrow
                    else:
                        dic2['totalH'] = None
                        dic2['meanH'] = None
                        dic2['len_protein'] = None
                        dic2['samples'] = 0

        if len(dic) == 0:
            return None

        df = pd.DataFrame(dic).T
        df['year-month'] = ["%s-%s"%(df.iloc[i].year, df.iloc[i].month) for i in range(len(df))]
        totals = df.groupby(['year-month']).samples.sum()
        df['percentage'] = [np.round(100*df.iloc[i].samples / totals[df.iloc[i]['year-month']], 2) for i in range(len(df))]

        df = df.sort_values(['country', 'year', 'month', 'variant', 'subvariant', 'pango'])

        title1 = 'Percentual of ' + title.lower()

        title2  = "<br>for %s"%(protein_name)
        title2 += "<br>between %d/%d - %d/%d"%(years_months[0][0], years_months[0][1], years_months[-1][0], years_months[-1][1])

        if whichPlot == 'qtt':
            ok1 = True; ok2 = False
        elif whichPlot == 'perc':
            ok1 = False; ok2 = True
        else:
            ok1 = True; ok2 = True

        if ok1: self.entropy_plot_sample_stack(df, title + title2, width=width, height=height,
                                       fontsize=fontsize, fontcolor=fontcolor, verbose=False)
        if ok2: self.entropy_plot_perc_sample_stack(df, title1 + title2, width=width, height=height,
                                       fontsize=fontsize, fontcolor=fontcolor, verbose=False)

        return df

    def entropy_plot_sample_stack(self, df, title, width=800, height=600,
                          fontsize=12, fontcolor='navy', verbose=False):

        traces=[]

        for country, data in df.groupby('country'):
            texts = ["%s <br>Variant: %s - %s - %s"%(data.iloc[i].country, data.iloc[i].variant, data.iloc[i].subvariant, data.iloc[i].pango) for i in range(len(data))]
            traces.append(go.Bar(x=data['year-month'], y=data.samples,
                                     text=texts,
                                     hovertemplate = "%{x}<br>samples: %{y}</br>%{text}",
                                     name=country))

        fig = go.Figure(data=traces)

        fig.update_layout(
                    autosize=True,
                    barmode='stack',
                    title=title,
                    width=width,
                    height=height,
                    # template=template,
                    # margin=margin,
                    font=dict(
                        family="Arial, bold, monospace",
                        size=fontsize,
                        color=fontcolor
                    ),
                    xaxis_title="years-months",
                    xaxis = dict(tickangle=45, tickfont=dict(family='Arial', color=fontcolor, size=fontsize) ),
                    yaxis_title='number samples',
                    paper_bgcolor="whitesmoke",
                    plot_bgcolor= "whitesmoke",
                    showlegend  = True
                )

        fig.show()

        figfullname = os.path.join(self.root_gisaid_prot_html, prepare_figname(title, 'html'))
        if verbose: print(">>> HTML saved:", figfullname)
        fig.write_html(figfullname)

        return

    def entropy_plot_perc_sample_stack(self, df, title, width=800, height=600,
                               fontsize=12, fontcolor='navy', verbose=False):

        traces=[]
        df_ym = df.groupby(['year', 'month']).samples.sum().reset_index()
        df_ym.columns = ['year', 'month', 'tot_samples']

        for country, data in df.groupby('country'):
            data['perc'] = [data.iloc[j].samples / df_ym[ (df_ym.year == data.iloc[j].year) & (df_ym.month == data.iloc[j].month) ].iloc[0].tot_samples  for j in range(len(data))]
            texts = ["%s <br>Variant: %s - %s - %s<br>perc: %.1f%%"%(data.iloc[i].country, data.iloc[i].variant, data.iloc[i].subvariant, data.iloc[i].pango, data.iloc[i].perc*100) for i in range(len(data))]

            traces.append(go.Bar(x=data['year-month'], y=data.percentage,
                                 text=texts,
                                 hovertemplate = "%{x}<br>%{y}% samples</br>%{text}",
                                 name=country))

        fig = go.Figure(data=traces)

        fig.update_layout(
                    autosize=True,
                    barmode='stack',
                    title=title,
                    width=width,
                    height=height,
                    # template=template,
                    # margin=margin,
                    font=dict(
                        family="Arial, bold, monospace",
                        size=fontsize,
                        color=fontcolor
                    ),
                    xaxis_title="years-months",
                    xaxis = dict(tickangle=90, tickfont=dict(family='Arial', color=fontcolor, size=fontsize) ),
                    yaxis_title='percentual of samples',
                    paper_bgcolor="whitesmoke",
                    plot_bgcolor= "whitesmoke",
                    showlegend  = True
                )
        fig.show()

        figfullname = os.path.join(self.root_gisaid_prot_html, prepare_figname(title, 'html'))
        if verbose: print(">>> HTML saved:", figfullname)
        fig.write_html(figfullname)

        return


    '''
    country = 'Australia'
    countries = ['Brazil', "Argentina", "Chile", 'USA', "China"]  #'Italia',  , "Australia"
    key = 'ORF3a'
    key = 'S'

    start_date = '01/03/2020'
    end_date = '01/04/2020'
    '''
    #-- Flavio 07/08/2020
    def entropy_build_html_sarscov2_cym(self, countries, protein, start_date, end_date, verbose=False):

        scountry = ", ".join(countries)

        h1 = "Protein/NSP '%s'"%(protein)
        h2 = scountry
        h3 = "between dates: %s to %s"%(start_date, end_date)

        header, body, diccountry = self.entropy_calc_return_html(countries, protein, start_date, end_date, verbose=verbose)
        countries2 = diccountry.keys()

        self.h1 = h1
        self.h2 = h2
        self.h3 = h3
        self.header = header
        self.body = body

        if self.template == '':
            print("template vazio !!!!")
            self.open_template()

        try:
            html_text = self.template%(h1, h2, h3, header, body)
            HTML(html_text)
        except:
            print("Is template empty?? ")
            return False, diccountry, ''

        scountry = "_".join(countries2)
        file_html = "polymorphism_%s_for_protein_%s_dates_%s_to_%s.html"%(scountry, protein, start_date, end_date)
        file_html = title_replace(file_html)
        file_html = os.path.join(self.root_gisaid_CDS_html, file_html)

        try:
            with open(file_html, 'w') as fh:
                fh.write(html_text)
            print("HTML file saved at '%s'"%(file_html))
            ret = True

        except:
            print("Could not save '%s'"%(file_html))
            ret = False

        return ret, diccountry, html_text


    ''' rever todo  '''
    def entropy_calc_return_html(self, countries, protein, start_date, end_date, verbose=False):

        diccountry = OrderedDict(); bads = []

        for country in countries:
            self.init_vars(country, None, None)
            file_entropy_cym = self.file_entropy_cym%(year, month)
            filefull = os.path.join(self.root_gisaid_CDS_entropy, file_entropy_cym)

            self.init_vars(country=self.country, protein=protein)

            if not os.path.exists(filefull):
                print("Could not find entropy data for %s: '%s'"%(self.country, filefull))
                bads.append(self.country)
                continue

            if self.dic_meta_country is None:
                self.dic_meta_country = pdloaddic(file_entropy_cym, path=self.root_gisaid_CDS_entropy, verbose=verbose)

            if self.dic_meta_country is None:
                bads.append(self.country)
                continue

            dfmeta = self.dic_meta_country['metadata']

            #--- opening fasta to recalculate ...
            # rever !!!
            filefull = os.path.join(self.root_gisaid_prot_fasta, self.file_DNA_country_cut)

            #--- load protein object
            mseqProt = ms.MySequence(self.prjName, root=self.root_gisaid_prot_fasta)
            mseqProt.readFasta(filefull, showmessage=verbose)

            #-- filter de i's related to the filter - convid_seqs_lib.py
            ndxs = self.entropy_filter_hs_by_date(dfmeta, start_date, end_date)

            if len(ndxs) == 0:
                stri = "array is empty for for %s %s %d/%d"%(self.country, protein, self.year, self.month)
                log_save(stri, filename=self.filelog, verbose=True)

                bads.append(self.country)
                continue

            mseqProt.seq_records = [mseqProt.seq_records[i] for i in ndxs]
            mseqProt.seqs        = [mseqProt.seqs[i]        for i in ndxs]

            #--- must recalculate all entropy again !
            # old: ret = self.entropy_calcHSannon_bias_correction_1pos(mseqProt, verbose=verbose, force=True, savedump=False)
            ret = self.entropy_fast_calcHShannon_1pos(mseqProt, verbose=verbose, force=True)
            '''
            dicPiList = self.dicPiList
            dicNList  = self.dicNList
            hs       = self.HShannonList'''

            dicPiList = self.dicPiList
            # dicNList  = dic['dicNList']
            # hs       = dic['HShannonList']
            print(len(ndxs), len(dicPiList))  # , len(dicNList), len(hs))

            consensus = []; groups =[]; matpercs =[]; samples=[]
            snps = OrderedDict(); muts = OrderedDict()
            str_mask = '%s:%.2f%%'

            for j in range(len(dicPiList)):
                tups = [(k, dicPiList[j][k]) for k in dicPiList[j].keys() if dicPiList[j][k] != 0]
                maxi = 0; group = ''; strinuc = ''

                for tup in tups:
                    perc = tup[1]; nuc = tup[0]

                    if perc == 1:
                        group = '*'
                        strinuc = nuc
                    else:
                        if perc <= 0.01 and group != 'MUT':
                            group = 'SNP'
                        else:
                            if perc <= 0.99: group = 'MUT'

                        if strinuc == '':
                            strinuc = str_mask%(nuc, perc*100)
                        else:
                            strinuc += '\n' + str_mask%(nuc, perc*100)

                    # # print(j, perc, group, strinuc)

                    if perc > maxi:
                        maxi = perc
                        nuc_maxi  = nuc

                # print(j, nuc_maxi, maxi)
                consensus.append(nuc_maxi)
                groups.append(group)
                matpercs.append(strinuc)

            #-- end -- for j in range(len(dicPiList)):
            country = self.country
            diccountry[country] = {}
            diccountry[country]['consensus'] = consensus
            diccountry[country]['groups'] = groups
            diccountry[country]['matpercs'] = matpercs
            diccountry[country]['n'] = len(mseqProt.seqs)
        #--- end --- for country in countries:

        if len(diccountry.keys()) == 0:
            return "", "", diccountry

        L = 0
        for country in diccountry.keys():
            if L == 0:
                L = len(diccountry[country]['consensus'])
            else:
                L2 = len(diccountry[country]['consensus'])
                if L2 < L:
                    L = L2

        l_position  = '<th class="tg-0pky">residues</th>'
        l_groups    = '<th class="tg-0pky">type of polym.</th>'
        l_aa        = '<th class="tg-0pky">amino acids</th>'

        count = 0; body = ''
        for country in diccountry.keys():
            l_consensus = '<th class="tg-0pky">%s (#%d) %s</th>'%(country, diccountry[country]['n'], "(consensus)")
            consensus = diccountry[country]['consensus']
            groups    = diccountry[country]['groups']
            matpercs  = diccountry[country]['matpercs']

            body_consensus = " <tr> " + l_consensus + " ".join(['<td class="tg-0pky">%s</td>'%x for x in consensus[:L] ]) + "</tr>"
            body_groups    = " <tr> " + l_groups    + " ".join(['<td class="tg-0pky">%s</td>'%x for x in groups[:L] ]) + "</tr>"
            body_perc      = " <tr> " + l_aa        + " ".join(['<td class="tg-0pky">%s</td>'%x for x in matpercs[:L] ]) + "</tr>"

            if count == 0:
                header         = " <tr> " + l_position  + " ".join(['<th class="tg-0pky">%d</th>'%(num+1) for num in range(L) ]) + "</tr>"
                body = body_consensus + body_groups + body_perc
            else:
                body += body_consensus + body_groups + body_perc

            count += 1

        return header, body, diccountry

    def mutations_get_variants_year(self, verbose:bool=False) -> (pd.DataFrame, list):
        fname     = self.fname_variants_of_the_year%(self.country, self.year)
        filefull  = os.path.join(self.root_iedb_summary, fname)

        if not os.path.exists(filefull):
            self.dfvar = None
            return None, []

        dfvar = pdreadcsv(fname, self.root_iedb_summary)

        sub2variants = self.sub2variants_unique(dfvar)
        self.dfvar = dfvar

        return dfvar, sub2variants

    def mutations_save_variants_year(self, dfmut:pd.DataFrame, verbose:bool=False) -> (pd.DataFrame, list):
        df2 = dfmut[ ['country', 'year', 'month', 'variant', 'subvariant', 'pango']].copy()
        df2 = df2.drop_duplicates()

        fname     = self.fname_variants_of_the_year%(self.country, self.year)
        filefull  = os.path.join(self.root_iedb_summary, fname)

        if os.path.exists(filefull):
            dfvar = pdreadcsv(fname, self.root_iedb_summary)
            dfvar = pd.concat([dfvar, df2])
        else:
            dfvar = df2

        dfvar = dfvar.sort_values(['country', 'year', 'month', 'variant', 'subvariant', 'pango'])
        dfvar = dfvar.drop_duplicates()
        dfvar.index = np.arange(0, len(dfvar))

        ret = pdwritecsv(dfvar, fname, self.root_iedb_summary)

        sub2variants = self.sub2variants_unique(dfvar)
        self.dfvar = dfvar

        return dfvar, sub2variants

    ''' independ from proteins '''
    def mutations_find_dfvar_already_set(self, verbose:bool=False) -> (pd.DataFrame, list):
        dfvar = self.dfvar

        if dfvar is None:
            dfvar, sub2variants = self.mutations_get_variants_year()
            if len(sub2variants) == 0:
                fname = self.fname_variants_of_the_year%(self.country, self.year)

                stri = "Error???: there is no variants for %s year %d - '%s'"%(self.country, self.year, fname)
                log_save(stri, filename=self.filelog, verbose=verbose)
                return None, []

        df = dfvar[(dfvar.year == self.year) & (dfvar.month == self.month) ].copy()
        if df.empty:
            return None, []

        df.index = np.arange(0, len(df))
        sub2variants = self.sub2variants_unique(df)

        return df, sub2variants

    def variants_find_year_protein(self, year):

        fname_start = "mutations_summary_for_"
        fname_year  = "year%d_variant_"%year
        lista = [x for x in os.listdir(self.root_iedb_summary) if fname_start in x and fname_year in x]

        ''' self.fname_summ_mut_cy_variant  = "mutations_summary_for_%s_year%d_variant_%s.xlsx" '''
        sub2variants = [x.split("_variant_")[1].replace(".xlsx", "") for x in lista]
        sub2variants.sort()
        sub2variants = np.unique(sub2variants)

        return sub2variants

    def mutations_many_countries_years_summary(self, countries, years, cutoff_num_seqs=3,
                                            force=False, verbose=False, print_warning=False, nround=4):
        ret_all = True
        for country in countries:
            for year in years:
                print("\n> %s %d"%(country, year), end=' ')
                ''' reset when year changes '''
                self.dfvar = None
                ret = self.mutations_calc_country_year_summary(country, year, cutoff_num_seqs=cutoff_num_seqs, force=force,
                                                               verbose=verbose, print_warning=print_warning, nround=nround)
                ret_all *= ret

        return ret_all

    ''' calc_virus_protein_summary --> calc_protein_mutational_summary '''
    def mutations_calc_country_year_summary(self, country, year, cutoff_num_seqs=3,
                                            force = False, verbose=False, print_warning=False, nround=4):
        ''' test:
            /projects/CENTD/covid/gisaid/result_IEDB_mutations$ less -S summary/analytics_mutations_for_USA_year2022.tsv

            1) dfmut = self.mutations_country_year_analytics
                dfmut = self.mutations_calc_dfmut_already_set(nround=nround, verbose=verbose)
            2) dfvar, sub2variants = self.mutations_save_variants_year(dfmut)
            3) seve: self.fname_summ_mut_cy_variant

            self.file_analytics_cy           = "analytics_mutations_for_%s_year%d.tsv"
            self.fname_variants_of_the_year  = "variants_of_year_%s_%d.tsv"
            self.fname_summ_mut_cy_variant   = "mutations_summary_for_%s_y%d_variant_%s.xlsx"

            to restart:
                rm summary/mutations_summary_for_*
                rm summary/variants_of_year_
                rm summary/analytics_mutations_for_
        '''

        self.init_vars_year_month(country, None, None, year, None)
        self.dfvar = None
        self.isProtein = True
        self.define_isProtein(self.isProtein)

        dfmut = self.mutations_country_year_analytics(force=force, verbose=verbose, print_warning=print_warning, nround=nround)
        self.dfmut = dfmut

        stri = ">>> Ok analytics ... starting summary for %s %d"%(self.country, self.year)
        log_save(stri, filename=self.filelog, verbose=verbose)

        if dfmut is None or dfmut.empty:
            stri = "No dfmut, cannot calculate summary for %s %d"%(self.country, year)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return False

        dfvar, sub2variants = self.mutations_save_variants_year(dfmut)

        stri = "Starting mutations calc_country_year_summary for %s %d, cutoff_num_seqs=%d, variants=%d"%(self.country, year, cutoff_num_seqs, len(sub2variants))
        log_save(stri, filename=self.filelog, verbose=verbose)

        for variant_subvar in sub2variants:
            print(">>>", self.country, self.year, variant_subvar, end=' ')
            self.variant_subvar = variant_subvar
            variant, subvariant, pango = variant_subvar.split(' ')

            fileExcel = self.fname_summ_mut_cy_variant%(self.country, year, variant_subvar)
            fileExcel = replace_space(fileExcel)
            filefull  = os.path.join(self.root_iedb_summary, fileExcel)

            if os.path.exists(filefull) and not force:
                stri = "Variant already exists for %s - %s %d"%(variant_subvar, self.country, year)
                log_save(stri, filename=self.filelog, verbose=verbose)
                continue

            dicAll = OrderedDict()
            for protein, protein_name in self.protList:

                self.init_vars_year_month(self.country, protein, protein_name, year, None)
                aaSeqRef = self.reference_get_consensus_aa()

                if aaSeqRef == '':
                    stri = "Error: mutations calc_country_year_variant_summary(): could not find reference consensus for %s %s %d - %s"%(self.country, protein_name, year, variant_subvar)
                    log_save(stri, filename=self.filelog, verbose=True)
                    continue

                dicf = {}; count = -1
                for j in range(1,13):
                    if year == self.year_max and j > self.month_max:
                        stri = "mutations calc_country_year_summary: for year %d, month %d exceeded month max %d"%(year, j, self.month_max)
                        log_save(stri, filename=self.filelog, verbose=verbose)
                        continue

                    self.init_vars_year_month(self.country, protein, protein_name, year, j)

                    dfvar2 = dfvar[ (dfvar.variant == variant) & (dfvar.subvariant == subvariant) & (dfvar.pango == pango)]

                    if dfvar2.empty:
                        stri = "Error???: mutations calc_country_year_variant_summary(): could not find in df variant %s for %s %s %d/%d"%(variant_subvar, self.country, protein_name, year, j)
                        log_save(stri, filename=self.filelog, verbose=verbose)
                        continue

                    dfq = dfmut[(dfmut.protein == protein) & (dfmut.month == j) & \
                                (dfmut.variant == variant) & (dfmut.subvariant == subvariant) & \
                                (dfmut.pango == pango)].copy()
                    if dfq.empty:
                        stri = "Error???: mutations calc_country_year_variant_summary(): could not find rows in dfmut for %s %s %d/%d - %s"%(self.country, protein_name, year, j, variant_subvar)
                        log_save(stri, filename=self.filelog, verbose=verbose)
                        continue

                    dfq = dfq.sort_values('pos')
                    dfq.index = np.arange(0, len(dfq))

                    for r in range(len(dfq)):
                        count += 1
                        row = dfq.iloc[r]

                        dicf[count] = {}
                        dic2 = dicf[count]
                        dic2['month'] = j
                        dic2['row'] = r
                        dic2['pos'] = row.pos
                        dic2['aas'] = row.aas
                        dic2['num_seqs'] = row.totCt
                        dic2['mutation'] = row.MUT

                        polys = row.aas.split(";")
                        aaRef = aaSeqRef[row.pos-1]

                        if len(polys) == 1:
                            aa = row.aas.split(' ')[0]

                            if aa != aaRef:
                                dic2['fix'] = True
                                dic2['aa']  = aa
                                dic2['aa_ref'] = aaRef
                            else:
                                dic2['fix'] = False
                                dic2['aa']  = aa
                                dic2['aa_ref'] = aaRef
                        else:
                            dic2['fix'] = False
                            dic2['aa']  = None
                            dic2['aa_ref'] = aaRef

                '''--------------- for j in range(1,13) ------------'''
                if len(dicf) == 0:
                    stri = "Error???: mutations calc_country_year_variant_summary(): could not find data for all months for %s %s %d - %s"%(self.country, protein_name, year, variant_subvar)
                    log_save(stri, filename=self.filelog, verbose=verbose)
                    continue

                dffa = pd.DataFrame(dicf).T

                dffa = dffa[ (dffa.num_seqs >= cutoff_num_seqs) ]
                self.dffa = dffa
                if len(dffa) == 0:
                    stri = "Problems - mutations calc_country_year_variant_summary(): not sufficient sequences for dff for %s %s %d - %s - cutoff threshold %d"%(self.country, protein_name, year, variant_subvar, cutoff_num_seqs)
                    log_save(stri, filename=self.filelog, verbose=True)
                    continue

                ''' not conserved or fixed '''
                dffa = dffa[ np.array([False if 'CONS' in x else True for x in dffa.mutation]) + dffa.fix.to_list()]
                if len(dffa) == 0:
                    stri = "mutations calc_country_year_variant_summary(): there are only Conserved Residues for dff %s %s %s %d"%(self.country, variant_subvar, protein_name, year)
                    log_save(stri, filename=self.filelog, verbose=verbose)
                    continue

                #--- month with the same number os num_seqs (the minimum)
                maxSeqs = -1; prev_month = 0; vals=[]
                dff = dffa.copy()
                dff = dff.sort_values(["month", "pos"]); count_seqs = 0
                dff.index = np.arange(0, len(dff))

                for k in range(len(dff)):
                    row2 = dff.iloc[k]
                    num_seqs = row2.num_seqs
                    month    = row2.month

                    if prev_month != month:
                        if prev_month != 0:
                            vals += [maxSeqs]*count_seqs

                        prev_month   = month
                        count_seqs = 1
                        maxSeqs    = num_seqs
                    else:
                        count_seqs += 1
                        if num_seqs > maxSeqs: maxSeqs = num_seqs

                vals += [maxSeqs]*count_seqs
                dff['num_seqs'] = vals

                '''---------------- pivot_table -----------------------------'''
                stri = "pivot_table for %s %d %s - %s"%(self.country, year, protein_name, variant_subvar)
                log_save(stri, filename=self.filelog, verbose=verbose)

                self.dff = dff
                dfp = pd.pivot_table(dff, values='aas', index=['month', 'num_seqs'],
                                     columns=['pos'], aggfunc=np.sum, fill_value=None)

                '''
                if verbose2:
                    print("-----------------")
                    print(dfp)
                    print("-----------------")
                '''

                posis    = list(dfp.columns)
                i_months = np.arange(0, len(dfp.index))
                last_month = i_months[-1]

                self.dfp = dfp
                self.posis = posis

                aasRef = [aaSeqRef[pos-1] for pos in posis]

                ''' ------------- min month --------------------'''
                min_month = np.min([x[0] for x in dfp.index])
                aasConsFirst = self.consensus_get_by_variant_posi(variant_subvar, year, min_month, posis)

                if len(aasConsFirst) == 0:
                    stri = "mutations calc_country_year_variant_summary(): could not find aasConsFirst Consensus for %s %s %d/%d - %s"%\
                           (self.country, self.protein_name, year, min_month, variant_subvar)
                    log_save(stri, filename=self.filelog, verbose=True)
                    continue

                dicx = {}
                dicx[('consensus - %d'%(min_month), None)] = aasConsFirst
                ''' ------------- min month end --------------------'''

                ''' ------------- max month --------------------'''
                max_month = np.max([x[0] for x in dfp.index])
                aasConsLast = self.consensus_get_by_variant_posi(variant_subvar, year, max_month, posis)

                if len(aasConsLast) == 0:
                    stri = "mutations calc_country_year_variant_summary(): could not find aasConsLast Consensus for %s %s %d/%d - %s"%\
                           (self.country, self.protein_name, year, max_month, variant_subvar)
                    log_save(stri, filename=self.filelog, verbose=True)
                    continue

                dicx[('consensus - %d'%(max_month), None)] = aasConsLast
                ''' ------------- max month end --------------------'''

                ass_common = []
                for ii in range(len(posis)):
                    pos = posis[ii]
                    x = pos-1

                    if dfp.isnull().iloc[last_month,ii]:
                        if aasConsLast[ii] != aasRef[ii] :
                            ass_common.append("fix")
                        else:
                            ass_common.append(" = ")
                    else:
                        '''
                            cannot use split
                            stri = 'F 20.00%; L 80.00%F 20.00%; L 80.00%F 20.00%; ...'
                        '''
                        stri = dfp.iloc[last_month, ii]
                        stri = stri.replace("; ","")
                        stri = stri.replace("%", "")
                        # if verbose2: print("3b1", stri)

                        mat = []; imat=0
                        while(True):
                            posi = stri.find(".")+3
                            term = stri[0: posi]
                            mat.append(term)
                            stri = stri[posi:]
                            if stri=='': break

                        mat = np.unique(mat)
                        # if verbose2: print("3b2", mat)

                        maxi = 0; maxi_aa="?"
                        for stri in mat:
                            mat2 = stri.split(' ')
                            aa = mat2[0]
                            val = float(mat2[1])
                            if val > maxi:
                                maxi = val
                                maxi_aa = aa

                        # print(pos, ii, aasConsFirst[ii], maxi_aa, maxi)
                        ''' aa is varying '''
                        if len(mat) > 1:
                            if maxi_aa == aasRef[ii]:
                                ass_common.append("var")
                            else:
                                ass_common.append("chg")
                        else:
                            if maxi_aa == aasRef[ii]:
                                ass_common.append(" = ")
                            else:
                                ass_common.append("fix")


                dicx[('consensus - ref', None)] = aasRef
                dicx[('changes', None)] = ass_common

                ret, dicdom = self.sarscov2_build_domain_sequence(protein)
                dfp.columns = [ str(x) + ' - ' + self.do_dicdom(x) for x in posis]

                dfx = pd.DataFrame(dicx).T

                dfx.columns     = dfp.columns
                dfx.index.names = dfp.index.names
                dfp = pd.concat([dfp, dfx])

                dicAll[protein_name] = dfp

            ''' -------- for protein, protein_name in protList ---------------'''
            try:
                with pd.ExcelWriter(filefull) as writer:
                    for protein_name in dicAll.keys():
                        dfaux = dicAll[protein_name]
                        if dfaux is None or len(dfaux) == 0:
                            continue

                        dfaux.to_excel(writer, sheet_name=protein_name)

                ret = True
            except:
                ret = False

        stri = "Analytics & Summary ok for %s %d"%(self.country, year)
        log_save(stri, filename=self.filelog, verbose=verbose)

        return ret

    def mutations_country_year_analytics(self, force=False, verbose=False, print_warning=False, nround=4):

        '''   "analytics_mutations_for_%s_year%d.tsv" '''
        filemut_anal = self.file_analytics_cy%(self.country, self.year)
        filefull = os.path.join(self.root_iedb_summary, filemut_anal)

        if os.path.exists(filefull) and not force:
            stri = "file analytics already exists: for %s %d '%s'"%(self.country, self.year, filefull)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return pdreadcsv(filemut_anal, self.root_iedb_summary, verbose=verbose)

        df_list = []

        for protein, protein_name in self.protList:
            if verbose: print(">> analysis snps mutations for %s - %s"%(self.country, protein_name))

            for month in np.arange(1,13):
                if self.year == self.year_max and month > self.month_max:
                    continue

                self.init_vars_year_month(self.country, protein, protein_name, self.year, month)

                dfmut = self.mutations_calc_dfmut_already_set(nround=nround, verbose=verbose)
                if dfmut is None or dfmut.empty:
                    stri = "mutations calc_country_year_analytics(): no data found in dfmut for %s %s %d/%d"%(protein_name, self.country, self.year, self.month)
                    log_save(stri, filename=self.filelog, verbose=verbose)
                    continue

                df_list.append(dfmut)
                # print("##<", self.country, protein_name, self.year, month, len(df_list))

        if df_list != []:
            df = pd.concat(df_list)
            ret = pdwritecsv(df, filemut_anal, self.root_iedb_summary)
            if ret:
                sret = "mutations country_year_analytics:"
            else:
                sret = "Error writing: mutations country_year_analytics:"

            stri = "%s: returning %s %d len=%d"%(sret, self.country, self.year, len(df))
        else:
            stri = "mutations calc_country_year_analytics(): df_list is empty for %s %d/%d"%(self.country, self.year, self.month)
            log_save(stri, filename=self.filelog, verbose=True)
            df = None
            sret = "Error df is empty: mutations country_year_analytics:"
            stri = "%s: returning %s %d"%(sret, self.country, self.year)
            ret = False

        log_save(stri, filename=self.filelog, verbose=verbose)
        return df

    def entropy_get_multiDic(self, verbose=False):
        multiDic = self.entropy_get_protein_year_month_multidic(self.year, self.month, verbose=verbose)

        if multiDic is not None and len(multiDic) > 0: return multiDic

        years_months = [(self.year, self.month)]
        self.entropy_many_calc_protein_cym(self.country, self.protList, years_months,
                                           force=True, verbose=verbose)

        multiDic = self.entropy_get_protein_year_month_multidic(self.year, self.month, verbose=verbose)
        return multiDic

    def mutations_calc_dfmut_already_set(self, nround=4, verbose=False):
        multiDic = self.entropy_get_multiDic(verbose=verbose)

        if multiDic is None or len(multiDic) == 0:
            stri = "Entropy was not calculated for %s %s %d/%d"%(self.country, self.protein_name, self.year, self.month)
            log_save(stri, filename=self.filelog, verbose=True)
            return None

        self.multiDic = multiDic

        sub2variants = list(multiDic.keys())
        # print(">>>> mutations_calc_dfmut_already_set():", sub2variants)
        dicc = {}; count=-1;

        for variant_subvar in sub2variants:
            self.variant_subvar = variant_subvar
            variant, subvariant, pango = variant_subvar.split(' ')

            dfc = multiDic[variant_subvar]['df']

            ''' remove conserved residues '''
            xs = dfc.x.unique()
            xs.sort()
            dfc.index = np.arange(0, len(dfc))

            for i in range(len(dfc)):
                row   = dfc.iloc[i]
                x     = row.x
                h     = np.round(row.y, nround)
                ''' aas like: N 92.68%; - 7.32% '''
                aas   = row.aas
                if aas is None or aas == '': continue

                ''' nns like: N 38; - 3 '''
                polys = row.nns.split('; ')

                count += 1
                dicc[count] = {}
                dic2 = dicc[count]

                dic2['country'] = self.country
                dic2['variant'] = variant
                dic2['subvariant'] = subvariant
                dic2['pango']   = pango
                dic2['protein'] = self.protein
                dic2['year']    = self.year
                dic2['month']   = self.month

                dic2['pos']  = x+1
                dic2['h']    = h
                ''' aas: 'N 92.68%; - 7.32%'  nns: 'N 38; - 3' '''
                dic2['aas']  = aas
                dic2['poly'] = row.nns

                tot = 0; seqAA = []; seqCt = []
                for stri in polys:
                    mat = stri.split(' ')
                    aa = mat[0]
                    ct = int(mat[1])

                    tot += ct
                    seqAA.append(aa)
                    seqCt.append(ct)

                percs = [np.round(x/tot, nround) for x in seqCt]

                if len(polys) == 1:
                    dic2['MUT'] = 'DEL' if seqAA[0] == '-' else 'CONS'
                    isMUT = seqAA[0] == '-'
                else:
                    # if there is 0 or 1 > 0.01 is SNP
                    isMUT = np.sum([1 if x > 0.01 else 0 for x in percs]) > 1

                    stri = 'MUT' if isMUT else 'CONS+NOISE'
                    if '-' in seqAA and stri == 'MUT':
                        stri += '+DEL'
                    dic2['MUT'] = stri

                dic2['seqAA'] = seqAA
                dic2['totCt'] = tot
                dic2['seqCt'] = seqCt
                dic2['percs'] = percs

                if verbose:
                    print('\t', isMUT, '\t', polys, '\t', self.year,'\t', self.month, '\t', x,'file0\t', h, '\t', aas, '\t', polys,'\t', tot,'\t', seqAA,'\t', percs, '\t', seqCt)

        dfmut = pd.DataFrame(dicc).T

        return dfmut

    def do_dicdom(self, pos):
        try:
            return self.dicdom[pos]
        except:
            return '?'


    def sarscov2_build_domain_sequence(self, protein):
        ret = True

        isNormaldic = True
        if protein == 'S':
            dic = virus_seqs_lib.dicS
            exception_regions = ['CTD',  'RBM', 'IBM']
            # unknown_regions   = [(920, 930)]
        elif protein == 'M':
            dic = virus_seqs_lib.dicM
            exception_regions = []
        elif protein == 'E':
            dic = virus_seqs_lib.dicE
            exception_regions = []
        elif protein == 'N':
            dic = virus_seqs_lib.dicN
            exception_regions = ["SR region"]
        elif 'nsp' in protein:
            mataux = virus_seqs_lib.dicNSP[protein]
            dic = {}
            dic[0] = [protein, protein, 1, mataux[2]]
            exception_regions = []

        else:
            try:
                mat = self.dicNucAnnot[protein]
                isNormaldic = False
            except:
                raise ValueError("Protein '%s' no prepared for sarscov2_build_domain_sequence()"%(protein))


        dicdom = {}

        if isNormaldic:
            for i in dic.keys():
                mat = dic[i]
                domain = mat[0]
                region = mat[1]
                start  = mat[2]
                end    = mat[3]

                if region in exception_regions: continue

                # print(domain, region, start, end)
                for j in np.arange(start, end+1):
                    dicdom[j] = region
        else:
            domain = ''
            region = ''
            start  = mat[0]
            end    = mat[1]

            for j in np.arange(start, end+1):
                dicdom[j] = region

        self.dicdom = dicdom

        return ret, dicdom

    def variants_colors(self, sub2variants):
        dic_color={}

        for variant_subvar in sub2variants:
            self.variant_subvar = variant_subvar
            variant, subvariant, pango = variant_subvar.split(' ')

            if variant == 'Delta':
                dic_color[variant_subvar] = "deepskyblue"
            elif variant == 'Alpha':
                dic_color[variant_subvar] = "green"
            elif variant == 'Beta':
                dic_color[variant_subvar] = "turquoise"
            elif variant == 'Epsilon':
                dic_color[variant_subvar] = "rosybrown"
            elif variant == 'Gamma':
                dic_color[variant_subvar] = "pink"
            elif variant == 'Mu':
                dic_color[variant_subvar] = "tomato"
            elif variant == 'Omicron':
                if 'BA.1' in pango:
                    dic_color[variant_subvar] = "gold"
                elif 'BA.2' in pango:
                    dic_color[variant_subvar] = "yellow"
                elif 'BA.3' in pango:
                    dic_color[variant_subvar] = "yellowgreen"
                elif 'BA.4' in pango:
                    dic_color[variant_subvar] = "goldenrod"
                elif 'BA.5' in pango:
                    dic_color[variant_subvar] = "orange"
                else:
                    dic_color[variant_subvar] = "darkorange"
            elif variant == 'VUM':
                dic_color[variant_subvar] = "mediumpurple"
            elif variant == 'unknown':
                dic_color[variant_subvar] = "gray"
            else:
                dic_color[variant_subvar] = plotly_colors[i]

        return dic_color


    def variant_evolution_create_table(self, countries, years_months, protein_name):

        striRef = '%s%s'%(self.cityRef, str(self.monthRef) if self.monthRef > 9 else '0'+str(self.monthRef))
        fields = ["country_month", 'variant_subvar', 'frequency', 'description', 'id_list']
        fields2 = ["country", 'year', 'month', 'variant_subvar', 'frequency', 'description', 'id_list']

        dfall = None
        for country in countries:
            country = country.replace(' ','_')

            for year, month in years_months:
                fnames = [x for x in os.listdir(self.root_iedb_summary) if 'mutated_posis_for' in x and \
                country in x and '_y'+str(year) in x and '_m'+str(month) in x and 'xlsx' in x]

                for fname in fnames:
                    filemutfull = os.path.join(self.root_iedb_summary, fname)

                    try:
                        excel = pd.ExcelFile(filemutfull)
                    except:
                        stri = "Could not open protein %d - file %s"%(filemutfull, protein_name)
                        log_save(stri, filename=self.filelog, verbose=verbose)
                        continue

                    sheets = excel.sheet_names
                    if protein_name not in sheets: continue

                    df = pd.read_excel(filemutfull, sheet_name=protein_name)

                    df = df[fields]
                    df = df[df.country_month != striRef]
                    df['country']  = country
                    df['year']  = year
                    df['month'] = month
                    df = df[fields2]

                    if dfall is None:
                        dfall = df
                    else:
                        dfall = pd.concat([dfall, df])

        dfall = dfall.sort_values(["country", 'year', 'month', 'variant_subvar'])

        dfall['year_month'] = ["%dy-%sm"%(dfall.iloc[i].year, str(dfall.iloc[i].month) if dfall.iloc[i].month > 9 else '0'+str(dfall.iloc[i].month) ) for i in range(len(dfall))]
        dfall['country_year_month'] = ["%s-%d-%s"%(dfall.iloc[i].country, dfall.iloc[i].year, str(dfall.iloc[i].month) if dfall.iloc[i].month > 9 else '0'+str(dfall.iloc[i].month) ) for i in range(len(dfall))]

        return dfall


    def variant_evolution_plot(self, dfall, protein_name,
                               title=None, template='plotly_white',
                               width_layout = 1400, height_layout = 900,
                               fontsize = 12, fontcolor = 'navy', xaxis_tickangle=90,
                               xaxis_title = 'year-month', yaxis_title = '# samples',
                               savefig = True):

        if title is None:
            title = "sub2variants evolution for protein '%s'"%(protein_name)

        fig = go.Figure()

        xaxis_list = list(dfall.country_year_month.unique())
        xaxis_list.sort()

        dates = list(dfall.year_month.unique())
        dates.sort()
        dates= [x.replace('y','').replace('m','') for x in dates]

        countries = list(dfall.country.unique())
        countries.sort()

        sub2variants = list(dfall.variant_subvar.unique())
        sub2variants.sort()

        dic_color = self.variants_colors(sub2variants)

        for variant_subvar in sub2variants:
            self.variant_subvar = variant_subvar
            # variant, subvariant, pango = variant_subvar.split(' ')
            color = dic_color[variant_subvar]

            df1 = dfall[(dfall.variant_subvar == variant_subvar)]
            if len(df1) == 0: continue

            fig.add_trace(go.Bar(
                x = df1.country_year_month, y = df1.frequency, name=variant_subvar,  marker_color=color ))

        fig.update_xaxes(categoryorder='array', categoryarray= xaxis_list)

        fig.update_layout(
            autosize=False,
            barmode='stack',
            title=title,
            width=width_layout,
            height=height_layout,
            template=template,
            margin=dict(l=40, r=40, b=40, t=140, pad=4),
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            xaxis_tickangle=xaxis_tickangle,
            font=dict(family="Arial, bold, monospace", size=fontsize, color=fontcolor ),
            paper_bgcolor="whitesmoke",
            plot_bgcolor= "whitesmoke", # lightgrey ivory gainsboro whitesmoke lightsteelblue 'lightcyan' 'azure', white, lightgrey, snow ivory beige powderblue
            showlegend=True
        )

        if savefig:
            figname = title_replace(title) + ".png"
            figname = os.path.join(self.root_iedb_figure, figname)
            fig.write_image(figname)

            figname = title_replace(title) + ".html"
            figname = os.path.join(self.root_iedb_figure, figname)
            fig.write_html(figname)

        return fig

    def mutations_plot_venn_diagram(self, countries, protein, protein_name, year, month,
                       figsize=(8,8), titlefontsize=18, printplot=False, savefig=True,
                       font = {'family' : 'sans-serif', 'weight' : 'normal','size': 22} ):

        matplotlib.rc('font', **font)

        if len(countries) != 3:
            print("Please send 3 countries")
            return None, None, None, None, None, None, None

        if year < 2020:
            print("Please year >= 2020")
            return None, None, None, None, None, None, None

        if month < 1 or month > 12:
            print("Please month1 between 1 and 12")
            return None, None, None, None, None, None, None

        month1 = month
        month2 = month + 1
        if month2 == 13: month2 = 1

        dic = {}
        for country in countries:
            self.init_vars(country)
            self.year = year
            dic[self.country] = self.mutations_country_year_analytics(force=False, verbose=False, nround=4)

        dicval = {}

        for country in countries:
            self.init_vars(country)
            df = dic[self.country]
            dfmut = df[ (df.protein == protein) & (df.month == month) & (df.SNP != 'SNP')]

            if len(dfmut) == 0:
                stri = "mutations_plot_venn_diagram: no data found in dfmut for protein %s, %s %d/%d"%(protein_name, self.country, year, month)
                log_save(stri, filename=self.filelog, verbose=verbose)
                return None, None, None, None, None, None, None

            posis = len(dfmut.pos)
            # print(">>>", country, year, month, "len:", len(posis))

            #----------- extra mutations: deletion and fixations
            # error rever todo
            fileExcel = self.fname_summ_mut_cy_variant%(self.country, year, variant_subvar)
            fileExcel = replace_space(fileExcel)
            filefull  = os.path.join(self.root_iedb_summary, fileExcel)

            if os.path.exists(filefull):
                dfmeta_cym = pd.read_excel(filefull, sheet_name=protein_name)
                #--- is this residues present in the respective month
                #--- get the month
                dfmutmonth = dfmeta_cym[dfmeta_cym.month == month]
                if len(dfmutmonth) > 0:
                    cols = list(dfmeta_cym.columns)[2:]
                    vals = list(dfmeta_cym[dfmeta_cym.month == 'changes'].iloc[0,2:])
                    mat = [i for i in range(len(cols)) if vals[i].replace(" ","") == '*']

                    # print(">>> mat", self.country, month, mat)

                    if len(mat) > 0:
                        residues = [cols[i] for i in mat]
                        #--- for each residue that has '*' in change
                        #--- see if is null
                        oks = [dfmutmonth[col].notnull().iloc[0] for col in residues]
                        residues = [residues[i] for i in range(len(residues)) if oks[i]]

                        if len(residues) > 0:
                            #--- get the residue number
                            residues = [int(residue.split(' - ')[0]) for residue in residues]

                            print(">>> residues", self.country, month, residues)

                            if len(residues) > 0:
                                posis += residues
                                posis = list( set(posis) )
            else:
                print("Could not find summary table: '%s'"%(filefull))

            #---------------------------------------------------
            posis.sort()
            # print(">>> country", country, posis)
            dicval[self.country] = posis


        inter01 = np.intersect1d(dicval[countries[0]], dicval[countries[1]])
        inter02 = np.intersect1d(dicval[countries[0]], dicval[countries[2]])
        inter12 = np.intersect1d(dicval[countries[1]], dicval[countries[2]])

        inter111 = list(np.intersect1d(inter01, inter02))

        inter011 = [x for x in inter01 if x not in inter111]
        inter101 = [x for x in inter02 if x not in inter111]
        inter110 = [x for x in inter12 if x not in inter111]

        inter = inter111 + inter101 + inter011
        vals001  = [x for x in dicval[countries[0]] if x not in inter]

        inter = inter111 + inter110 + inter011
        vals010  = [x for x in dicval[countries[1]] if x not in inter]

        inter = inter111 + inter101 + inter110
        vals100  = [x for x in dicval[countries[2]] if x not in inter]

        fig = plt.figure(figsize=figsize);
        venn3(subsets = (len(vals001), len(vals010), len(inter011),
                         len(vals100), len(inter101), len(inter110), len(inter111)), set_labels = countries);

        title = "Protein %s - year=%d month=%d\nCountries: %s"%(protein_name, year, month, ",".join(countries))
        plt.title(title, fontsize=titlefontsize);

        if savefig:
            filefig = title_replace(title) + ".png"
            filefig = os.path.join(self.root_gisaid_prot_figure, filefig.lower())
            plt.savefig(filefig);

        if not printplot:
            plt.close(fig);

        return inter111, vals001, vals010, vals100, inter011, inter101, inter110

    def mutations_summary_join_contries(self, col, countries):
        countries2 = [c.replace(" ", "") for c in countries]

        mat = list(col)
        names = []
        for i in range(3):
            if mat[i] == '1':
                names.append(countries2[(2-i)])

        if len(names) == 0: return 'c_'+col
        return "_".join(names)

    # mutations_summary --> mutations_venn_diagram
    # todo - rever !
    def mutations_venn_diagram(self, matcountries, protein, protein_name, year,
                          figsize=(8,8), titlefontsize=18, printplot=False, savefig=True,
                          font = {'family' : 'sans-serif', 'weight' : 'normal','size': 22},
                          filemut_table = 'mutational_table_gisaid_prot_%s_version_%s.xlsx'):

        filemut_table = filemut_table%(protein_name, self.version)
        filemut_table = os.path.join(self.root_iedb_summary, filemut_table)
        dicAll = {}

        for countries in matcountries:
            sCountries = ",".join(countries)
            count = 0
            dicf = {}

            for month in np.arange(1,13):

                inter111, vals001, vals010, vals100, inter011, inter101, inter110 \
                = self.mutations_plot_venn_diagram(countries, protein, protein_name, \
                                      year, month, titlefontsize=titlefontsize, \
                                      font=font, printplot=printplot, savefig=savefig)

                if inter111 is None: continue

                dicf[count] = {}
                dic2 = dicf[count]
                dic2['countries'] = sCountries
                dic2['year'] = year
                dic2['month'] = month
                dic2['111'] = str(inter111).replace('[','').replace(']','')
                dic2['001'] = str(vals001).replace('[','').replace(']','')
                dic2['010'] = str(vals010).replace('[','').replace(']','')
                dic2['100'] = str(vals100).replace('[','').replace(']','')
                dic2['011'] = str(inter011).replace('[','').replace(']','')
                dic2['101'] = str(inter101).replace('[','').replace(']','')
                dic2['110'] = str(inter110).replace('[','').replace(']','')

                count += 1

            dff = pd.DataFrame(dicf).T
            cols = list(dff.columns)
            cols1 = cols[:3]
            cols2 = cols[3:]
            cols2 = [self.mutations_summary_join_contries(col, countries) for col in cols2]
            # cols2 = ['c_'+col for col in cols2]
            dff.columns = cols1+cols2

            sCountries2 = sCountries.replace(",","-").replace(" ","_")
            dicAll[sCountries2] = dff

        with pd.ExcelWriter(filemut_table) as writer:
            for sCountries2 in dicAll.keys():
                dff = dicAll[sCountries2]
                if dff is None or len(dff) == 0:
                    continue

                dff.to_excel(writer, sheet_name=sCountries2)

        if verbose: print("File saved at '%s'"%(filemut_table))


    def reference_get_consensus_aa(self, verbose=False):
        ret = self.reference_get_save_consensus(verbose=verbose)

        if not ret:
            return ''

        return str(self.mseqRef_consensus.seq_records[0].seq)

    def reference_get_save_consensus(self, force=False, verbose=False):
        ''' reference_Wuhan_get_save_consensus --> reference_get_save_consensus // open_reference_fasta
            reference data Country (China) and City (Wuhan) for sars-cov-2
            creates: self.mseqRef =  self.mseqRef_consensus
            returns: ret
        '''
        self.mseqRef = None; self.mseqRef_consensus = None

        prev_country = self.country
        prev_year    = self.year
        prev_month   = self.month

        self.init_vars_year_month(self.countryRef, self.protein, self.protein_name, self.yearRef, self.monthRef)

        file_prot_cym = self.file_prot_cym_reference%(self.yearRef, self.monthRef)
        filefull_prot = os.path.join(self.root_gisaid_prot_fasta, file_prot_cym)

        file_prot_cym_consensus = file_prot_cym.replace(".fasta", "_consensus.fasta")
        filefull_prot_consensus = os.path.join(self.root_gisaid_prot_fasta, file_prot_cym_consensus)

        if os.path.exists(filefull_prot_consensus) and os.path.exists(filefull_prot) and not force:
            mseqRef = ms.MySequence(self.prjName, root=self.root_gisaid_prot_fasta)
            ret = mseqRef.readFasta(filefull_prot, showmessage=verbose)

            mseqRef_consensus = ms.MySequence(self.prjName, root=self.root_gisaid_prot_fasta)
            ret = mseqRef_consensus.readFasta(filefull_prot_consensus, showmessage=verbose)

            self.init_vars_year_month(prev_country, self.protein, self.protein_name, prev_year, prev_month)
            self.mseqRef = mseqRef
            self.mseqRef_consensus = mseqRef_consensus

            return ret

        ret, mseqRef = self.open_nCount_sample_protein_cym_already_set(fulfilled=True, verbose=verbose, force=force)

        if not ret or mseqRef is None or len(mseqRef.seqs) == 0:
            stri = "Error: fasta not found for '%s'"%(self.filefull_samp)
            log_save(stri, filename=self.filelog, verbose=True)
            return False

        seq_records = [seq_rec for seq_rec in mseqRef.seq_records if (self.cityRef in seq_rec.description) or ('Wuha' in seq_rec.description)]

        if len(seq_records) == 0:
            stri = "Errror: fasta read, no records found having %s"%(self.cityRef)
            log_save(stri, filename=self.filelog, verbose=True)
            return False

        ret = mseqRef.writeSequences(seq_records, filefull_prot, verbose=verbose)

        ref_consensus = self.consensus_get_aa_from_seqs(mseqRef.seqs)

        ''' rebuild seq_records '''
        seq_record = SeqRecord(
                            seq=Seq(ref_consensus),
                            description=mseqRef.seq_records[0].description,
                            id=mseqRef.seq_records[0].id, name=''
                        )

        seq_records = [seq_record]

        mseqRef_consensus = ms.MySequence(self.prjName)
        ret = mseqRef_consensus.writeSequences(seq_records, filefull_prot_consensus, verbose=verbose)

        self.init_vars_year_month(prev_country, self.protein, self.protein_name, prev_year, prev_month)
        self.mseqRef = mseqRef
        self.mseqRef_consensus = mseqRef_consensus

        return ret

    def reference_calc_summary(self, cutoff_num_seqs=3, print_warning=False,
                               nround=4, force = False, verbose=False):

        ''' variant is fixed, otherwise, change here '''
        variant      = self.variantRef
        fileExcel    = self.fname_summ_mut_cy_variant%(self.countryRef, self.yearRef, variant)
        fileExcel = replace_space(fileExcel)
        filefullRef  = os.path.join(self.root_iedb_summary, fileExcel)

        if os.path.exists(filefullRef) and not force:
            return True

        ret = self.mutations_calc_country_year_summary(self.countryRef, self.yearRef, \
                   cutoff_num_seqs=cutoff_num_seqs, force=force, verbose=verbose, print_warning=print_warning, nround=nround)

        return ret

    ''' --- creates self.dfSumm_ref --- '''
    def reference_open_summary(self, force = False, verbose=False):
        self.dfSumm_ref  = None

        prev_country = self.country
        prev_year    = self.year
        prev_month   = self.month

        self.init_vars_year_month(self.countryRef, self.protein, self.protein_name, self.yearRef, self.monthRef)
        variant = self.variantRef

        ''' find reference summary: China - first month '''
        ret = self.reference_calc_summary(cutoff_num_seqs=3, print_warning=False,
                                          nround=4, force = False, verbose=verbose)

        if not ret:
            self.init_vars_year_month(prev_country, self.protein, self.protein_name, prev_year, prev_month)
            stri = "reference protein_open_summary(): reference file does not exist: for %s %d variant '%s'"%(self.countryRef, self.yearRef, self.variantRef)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return False

        fileExcelRef = self.fname_summ_mut_cy_variant%(self.countryRef, self.yearRef, variant_subvar)
        fileExcel = replace_space(fileExcel)
        filefullRef  = os.path.join(self.root_iedb_summary, fileExcelRef)

        try:
            dfSumm_ref = pd.read_excel(filefullRef, sheet_name=self.protein_name)
        except:
            self.init_vars_year_month(prev_country, self.protein, self.protein_name, prev_year, prev_month)
            stri = "reference protein_open_summary(): could not find tab for protein '%s' in file '%s'"%(self.protein_name, filefullRef)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return False

        if dfSumm_ref is None or len(dfSumm_ref) == 0:
            self.init_vars_year_month(prev_country, self.protein, self.protein_name, prev_year, prev_month)
            stri = "Warning: reference protein_open_summary(): could not find protein '%s' tab in file '%s'"%(self.protein_name, filefullRef)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return False

        self.dfSumm_ref = dfSumm_ref
        self.init_vars_year_month(prev_country, self.protein, self.protein_name, prev_year, prev_month)
        return True

    '''
        Calculate all possible 'pseudo-sub2variants and each frequency
        remove protList
    '''
    def mutations_calc_possible_pseudo_variants(self, countries, years, verbose=False, force=False):
        '''
            save in:
                self.fname_mut_freq_variant  = "mutated_posis_for_%s_y%d_m%d_variant_%s.xlsx"
                filemut_var     = self.fname_mut_freq_variant%(self.country, self.year, self.month, variant)
                filemutfull = os.path.join(self.root_iedb_summary, filemut_var)
        '''

        if not isinstance(years, list):
            stri = "Error: years must be a list of integers like [2021, 2022]"
            log_save(stri, filename=self.filelog, verbose=True)
            return

        for year in years:
            ''' reset when year changes '''
            self.dfvar = None
            print("\n\n>", year, end=' ')

            ''' looping all countries '''
            prev_country = ''
            for country in countries:
                ''' looping by all months of the year '''
                for month in np.arange(1,13):
                    self.init_vars_year_month(country, None, None, year, month)
                    if prev_country != country:
                        print("\n>", self.country, end=' ')
                        prev_country = country
                    print(month, end=' ')

                    dfvar, sub2variants = self.mutations_find_dfvar_already_set()

                    for variant_subvar in sub2variants:
                        self.variant_subvar = variant_subvar
                        ''' needs self.variant_subvar '''
                        ret = self.mutations_calc_pseudo_variants_cym(force=force, verbose=verbose)

    ''' used by parallel
        mutations_calc_possible_pseudo_variants_cym --> mutations_calc_all_pseudo_variants_cym
    '''
    def mutations_calc_all_pseudo_variants_cym(self, country, year, month, verbose=False, force=False):

        self.init_vars_year_month(country, None, None, year, month)

        ''' independ from proteins '''
        dfvar, sub2variants = self.mutations_find_dfvar_already_set()

        for variant_subvar in sub2variants:
            self.variant_subvar = variant_subvar
            ret = self.mutations_calc_pseudo_variants_cym(force=force, verbose=verbose)

    # remove protList
    def mutations_calc_pseudo_variants_cym(self, force=False, verbose=False):
        first_time = True

        filemut_var = self.fname_mut_freq_variant%(self.country, self.year, self.month, self.variant_subvar)
        filemut_var = replace_space(filemut_var)
        filemutfull = os.path.join(self.root_iedb_summary, filemut_var)

        if os.path.exists(filemutfull) and not force:
            stri = "mutations_calc_pseudo_variants_cym(): already calculated '%s'"%(filemutfull)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return True

        ''' read this Excel file below '''
        dicAll = {}
        for protein, protein_name in self.protList:
            self.init_vars_year_month(self.country, protein, protein_name, self.year, self.month)

            # # print("$$$ 3 mutations_calc_pseudo_variants_cym()", self.country, self.protein_name, self.year, self.month, self.variant_subvar, protein_name)
            # print("\t###", self.country, self.protein_name, self.year, self.month, self.variant_subvar)

            ''' --- creates self.dfSumm_ref ---
            if not self.reference_open_summary():
                stri = "Are there mutations? could not find reference protein summary for %s %s %d %d"%(self.country, protein_name, self.year, self.month)
                log_save(stri, filename=self.filelog, verbose=verbose)
                continue
            '''

            ret = self.reference_get_save_consensus(force=False, verbose=verbose)
            if not ret:
                stri = "Error: mutations calc_pseudo_variants_cym() could not find reference protein fasta"
                log_save(stri, filename=self.filelog, verbose=True)
                continue

            '''-------------------------------------------------------------------------
               --- now, look for the mutations in the current country, year, variant_subvar ---
            '''
            fileExcel = self.fname_summ_mut_cy_variant%(self.country, self.year, self.variant_subvar)
            fileExcel = replace_space(fileExcel)
            filefull  = os.path.join(self.root_iedb_summary, fileExcel)
            # # print("$$$ 4a mutations_calc_pseudo_variants_cym(): filefull", filefull)

            if not os.path.exists(filefull):
                stri = "mutations calc_pseudo_variants_cym(): summary mutation file does not exist: '%s'"%(filefull)
                log_save(stri, filename=self.filelog, verbose=True)
                continue

            try:
                self.dfSumm_cy = pd.read_excel(filefull, sheet_name=protein_name)
            except:
                stri = "mutations calc_pseudo_variants_cym(): protein '%s' tab in summary mutation file does not exist: '%s'"%(protein_name, filefull)
                log_save(stri, filename=self.filelog, verbose=True)
                continue

            if self.dfSumm_cy is None or self.dfSumm_cy.empty:
                stri = "mutations calc_pseudo_variants_cym(): no data found for protein %s in summary mutation file '%s'"%(protein_name, filefull)
                log_save(stri, filename=self.filelog, verbose=True)
                continue

            dicVar, dic_posi = self.mutations_calc_summ_reference_plus_months_already_set(force=force, verbose=verbose)
            self.dicVar   = dicVar
            self.dic_posi = dic_posi

            if dicVar == None or len(dicVar) == 0:
                stri = "mutations calc_pseudo_variants_cym(): no dicVar for %s %s %d/%d - %s "%(self.country, self.protein_name, self.year, self.month, self.variant_subvar)
                log_save(stri, filename=self.filelog, verbose=True)
                continue

            dicf = {}; count = -1

            for key in dicVar.keys():
                count += 1
                dicf[count] = {}
                dic3 = dicf[count]

                mat3 = key.split('\t')
                prefixRef = '%s01'%(self.cityRef)

                dic3['country_month'] = mat3[0]

                if mat3[0] == prefixRef:
                    dic3['variant_subvar'] = self.variantRef
                else:
                    dic3['variant_subvar'] = self.variant_subvar

                dic3['frequency']     = dicVar[key]['frequency']
                dic3['description']   = dicVar[key]['description']
                if dicVar[key]['id_list'] is None or len(dicVar[key]['id_list']) == 0:
                    dic3['id_list']   = None
                else:
                    dic3['id_list']   = ";".join(dicVar[key]['id_list'])
                month_dicVar          = dicVar[key]['month']

                aas = list(mat3[1])
                for i in range(len(aas)):
                    xpos = dic_posi[month_dicVar]['cols'][i]
                    dic3[xpos+1] = aas[i]

            if len(dicf) == 1:
                dicf[1] = dicf[0]
                dff = pd.DataFrame.from_dict(dicf).T.loc[[0]]
            else:
                dff = pd.DataFrame(dicf).T

            dicAll[self.protein_name] = dff

        ''' for country in countries: -- end -----------------------------------
        ------------ saving in Excel file --------------------------------------'''
        if len(dicAll) > 0:
            try:
                with pd.ExcelWriter(filemutfull) as writer:
                    for protein_name in dicAll.keys():
                        dff = dicAll[protein_name]
                        if dff is None or len(dff) == 0:
                            continue

                        dff = dff.drop_duplicates()
                        dff.to_excel(writer, sheet_name=protein_name, index=False)

                stri = "File saved at '%s' - %s %s %d %d"%(filemutfull, self.country, self.protein_name, self.year, self.month)
                ret = True
            except:
                stri = "Could not save at '%s' - %s %s %d %d"%(filemutfull, self.country, self.protein_name, self.year, self.month)
                ret = False
        else:
            stri = "No data found to save excel - %s %s %d %d"%(self.country, self.protein_name, self.year, self.month)
            ret = False

        log_save(stri, filename=self.filelog, verbose=True)
        return ret

    def mutations_calc_summ_reference_plus_months_already_set(self, force=False, verbose=False):
        stri = ">>> calc variant tot_samples for %s %d/%d - %s"%(self.country, self.year, self.month, self.variant_subvar)
        log_save(stri, filename=self.filelog, verbose=verbose)

        dicVar = OrderedDict(); dic_posi = OrderedDict()

        '''
        if self.dfSumm_ref is None or len(self.dfSumm_ref) == 0:
            stri = "Error: there is no dfSumm_ref"
            log_save(stri, filename=self.filelog, verbose=True)
            return None, None
        '''
        if self.mseqRef_consensus is None or len(self.mseqRef_consensus.seq_records) == 0:
            stri = "Error: there is no mseqRef consensus"
            log_save(stri, filename=self.filelog, verbose=True)
            return None, None

        # # print("$$$ m1 mutations_calc_summ_reference_plus_months_already_set(): fasta_cym", self.country, self.year, self.month, self.protein_name)
        ret, mseqProt = self.open_nCount_sample_protein_cym_already_set(fulfilled=True, verbose=verbose, force=force)

        if not ret or mseqProt is None or len(mseqProt.seqs) == 0:
            stri = "Error: fasta not found or empty for '%s'"%(self.filefull_samp)
            log_save(stri, filename=self.filelog, verbose=True)
            return None, None

        seq_records = [seq_rec for seq_rec in mseqProt.seq_records if self.variant_subvar_check_in_seq_rec(seq_rec, self.variant_subvar)]

        if len(seq_records) == 0:
            return None, None

        mseqProt.seq_records = seq_records
        mseqProt.seqs = np.array([ list(seq_rec.seq) for seq_rec in seq_records])

        ''' here calculates all mutations '''
        # # print("$$$ m2 mutations_calc_summ_reference_plus_months_already_set(): mutations_summary_build_country_cols")
        cols, posi = self.mutations_summary_build_country_cols(verbose=verbose) # self.dfSumm_cy, country=self.country, month=self.month

        if cols is None or len(cols) == 0:
            return None, None

        ''' --- build Wuhan - reference ---'''
        prefixRef = '%s01\t'%(self.cityRef)

        dic_posi[0] = {}
        dic_posi[0]['cols'] = cols
        dic_posi[0]['posi'] = posi

        seq         = "".join(self.mseqRef_consensus.seqs[0, cols])
        description = self.mseqRef_consensus.seq_records[0].description.strip()
        _id         = description.split("||")[0].strip()

        dicVar[prefixRef+seq] = {}
        dic2 = dicVar[prefixRef+seq]
        dic2['prefix']      = prefixRef+seq
        dic2['month']       = 0
        dic2['_id']         = _id
        dic2['id_list']     = None
        dic2['description'] = description
        dic2['frequency']   = None

        ''' -------------------------------------------------------------------- '''

        ''' first the reference - here the country '''
        prefix = '%s%d\t'%(self.country, self.month)
        dic_posi[self.month] = {}
        dic_posi[self.month]['cols'] = cols
        dic_posi[self.month]['posi'] = posi

        for i in range(len(mseqProt.seqs)):
            seq         = "".join(mseqProt.seqs[i, cols])
            description = mseqProt.seq_records[i].description.strip()
            _id         = description.split("||")[0].strip()

            if prefix+seq in dicVar.keys():
                dic2 = dicVar[prefix+seq]
                dic2['frequency'] += 1
                dic2['id_list']   += [_id]
            else:
                dicVar[prefix+seq] = {}
                dic2 = dicVar[prefix+seq]
                dic2['prefix']      = prefix+seq
                dic2['month']       = self.month
                dic2['_id']         = _id
                dic2['id_list']     = [_id]
                dic2['description'] = description
                dic2['frequency']   = 1

        # # print("$$$ m3 mutations_calc_summ_reference_plus_months_already_set(): return")
        return dicVar, dic_posi

    ''' here calculates all mutations '''
    def mutations_summary_build_country_cols(self, verbose=False):

        dfSumm = self.dfSumm_cy
        cols = list(dfSumm.columns)[2:]

        dfa = dfSumm[dfSumm.month == self.month]
        if len(dfa) == 0:
            stri = "Summary ok: there is no mutations for month %d in %s %s"%(self.month, self.country, self.protein_name)
            log_save(stri, filename=self.filelog, verbose=False)
            return None, None

        vals = list(dfa.iloc[0,2:])
        mat  = [i for i in range(len(cols)) if pd.notnull(vals[i])]

        mat.sort()
        mat = np.unique(mat)

        posi = [int(cols[i].split(' - ')[0]) for i in mat]
        cols = np.array(posi)-1

        return cols, posi


    ### rever toto - test please
    def reference_protein_get_id(self, variant_subvar, verbose=False):

        self._idRef = None
        self.variant_subvar = variant_subvar

        dfm2 = self.mutations_variant_posi_get_table_already_set(verbose=verbose)

        if dfm2 is None or len(dfm2) == 0:
            return None

        good = [True if self.cityRef in dfm2.iloc[i].country_month else False for i in range(len(dfm2)) ]
        dfm2 = dfm2[ good ]

        if dfm2 is None or len(dfm2) == 0:
            stri = "No Reference City in data (dfm2)"
            log_save(stri, filename=self.filelog, verbose=verbose)
            return None

        _idRef = dfm2.iloc[0].description.split("||")[0]
        self._idRef = _idRef

        return _idRef


    '''------------------ sub2variants -----------------------------------------------
       self.init_vars(country, protein, protein_name)
       self.year = year
       self.month = month
    ----------------------------------------------------------------------------'''
    def pseudovar_read_variant_already_set(self, verbose=False):

        df = self.mutations_variant_posi_get_table_already_set(verbose=verbose)

        if df is None or len(df) == 0:
            stri = "Warning: pseudovar read_variant: table not opened for %s %s %d/%d - %s"%(self.country, self.protein_name, self.year, self.month, self.variant_subvar)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return None

        stri = ">>> pseudovar read_variant: Ok len %d for %s %s %d/%d - %s"%(len(df), self.country, self.protein_name, self.year, self.month, self.variant_subvar)
        log_save(stri, filename=self.filelog, verbose=verbose)

        return df

    def dfmut_set(self, country, year, month,  protein, protein_name, variant_subvar, verbose=False):

        self.init_vars_year_month(country, protein, protein_name, year, month)
        self.variant_subvar = variant_subvar

        df = self.pseudovar_read_variant_already_set(verbose=verbose)

        if df is None or len(df) == 0:
            return None, None, None

        dfmeta, ids = self.mutations_table_find_country_month_ids(df, verbose=verbose)

        if dfmeta is None or ids is None or len(df) == 0:
            return None, None, None

        return df, dfmeta, ids

    def rebuild_ids(self, stri):
        ''' in the sum off the groupby agg loses the comma '''
        mat = stri.split(';')
        lista = []
        for elem in mat:
            mat2 = elem.split('EPI_')
            for elem2 in mat2:
                if elem2 != '': lista.append(elem2)

        lista2 = ['EPI_'+x for x in lista]
        return ";".join(lista2)

    '''
        # get only refID
        if self.refID is not None:
            good = [True if self.refID in x else False for x in df2.id_list]
            df4 = df2.loc[good, ('frequency', 'description', 'id_list')]
            return df4

        if df2.frequency.sum() < nCount:
            monthm1 = month - 1
            if monthm1 > 0:
                # the same table - different years may have different columns (aas)
                df3 = df[ (df.country_month == '%s%d'%(self.country, monthm1)) & (df.frequency >= self.cutoff_freq_pseudvar)]
                if len(df3) > 0: df2 = df2.append(df3)

        calc_main_strains --> variants_calc_main_pseudo_simple_table
    '''
    def variants_calc_main_pseudo_simple_table(self, df, verbose=False):

        df2 = df.loc[(df.country_month == '%s%d'%(self.country, self.month)) & (df.frequency >= self.cutoff_freq_pseudvar)]
        if df2.empty:
            stri = ">>> low frequence for %s %s %d/%d - %s"%(self.country, self.protein_name, self.year, self.month, self.variant_subvar)
            log_save(stri, filename=self.filelog, verbose=verbose)
            df2 = df.loc[(df.country_month == '%s%d'%(self.country, self.month))]
            low_frequency = True
        else:
            low_frequency = False

        if df2.empty:
            stri = ">>> dfms is empty (no variation) for %s %s %d/%d - %s"%(self.country, self.protein_name, self.year, self.month, self.variant_subvar)
            log_save(stri, filename=self.filelog, verbose=True)
            return None

        if len(df2) == 1:
            stri = ">>> dfms has only 1 subvar for %s %s %d/%d - %s"%(self.country, self.protein_name, self.year, self.month, self.variant_subvar)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return df2

        if not low_frequency:
            stri = ">>> dfms is Ok for %s %s %d/%d - %s"%(self.country, self.protein_name, self.year, self.month, self.variant_subvar)
            log_save(stri, filename=self.filelog, verbose=verbose)

        cols = list(df2.columns)
        ncol = [j for j in range(len(cols)) if cols[j] == 'id_list'][0]
        ncol += 1
        cols = cols[ncol:]
        cols.sort()

        selcols = [col for col in cols if self.is_polymorphic(df2, col)]

        if len(selcols) == 0:
            freq = df2.frequency.sum()
            desc = df2.iloc[0].description
            s_list = ";".join(df2.id_list)
            df4 = pd.DataFrame([[freq, desc, s_list]],  columns=['frequency', 'description', 'id_list'])

        else:
            fields2 = ['country_month', 'variant_subvar', 'frequency', 'description', 'id_list']
            df3 = df2[fields2  + selcols ].copy()
            df3 = df3.fillna(value='')
            df3.index = np.arange(0, len(df3))

            warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
            df4 = df3.groupby(selcols).agg({'frequency': 'sum', 'description': 'first', 'id_list': "sum"}).reset_index()
            warnings.simplefilter(action='default', category=pd.errors.PerformanceWarning)

            df4['id_list'] = [self.rebuild_ids(x) for x in df4.id_list]
            df4 = df4.sort_values('frequency', ascending=False)

            try:
                df4.frequency = df4.frequency.astype(int)
            except:
                pass

            df4.index = np.arange(0, len(df4))

            cols = list(df4.columns)[:-3]
            ''' from cols to posi '''
            cols = [x+1 for x in cols]
            cols = cols + ['frequency', 'description', 'id_list']
            df4.columns = cols

        return df4

    def mutations_table_find_country_month_ids(self, df, verbose=False):
        cutoff = self.cutoff_freq_country
        dfmeta = df[(df.country_month == '%s%d'%(self.country, self.month)) & (df.frequency >= self.cutoff_freq_country)]

        if len(dfmeta) == 0:
            cutoff = int(cutoff/2)
            dfmeta = df[(df.country_month == '%s%d'%(self.country, self.month)) & (df.frequency >= cutoff)]

            if len(dfmeta) == 0:
                cutoff = 3
                dfmeta = df[(df.country_month == '%s%d'%(self.country, self.month)) & (df.frequency >= cutoff)]

                if len(dfmeta) == 0:
                    if verbose: print("No data for %s - %s - %d"%(self.country, self.protein_name, self.month))
                    return None, []

        if verbose: print(">>> cutoff:", cutoff)

        try:
            ids = dfmeta.description.to_list()
        except:
            print("No data for %s - %s - %d"%(self.country, self.protein_name, self.month))
            return None, []

        if len is None or len(ids) == 0:
            print("<<< Didn't find any id for %s - %s - %d"%(self.country, self.protein_name, self.month))
            return None, []

        ids = [ x.split('||')[0].strip() for x in ids]
        ids.sort()
        ids = np.unique(ids)

        return dfmeta, ids

    def is_polymorphic(self, df, col):
        dic = {}

        letters  = df[col].to_list()
        freq     = df['frequency'].to_list()

        for i in range(len(letters)):
            let = letters[i]

            if let in dic.keys():
                dic[let] += freq[i]
            else:
                dic[let] = freq[i]

        total = np.sum(list(dic.values()))
        # print("total", total)

        gt_then_pi_count = 0
        ''' polymorphic if at least two aa > self.gt_then_pi_cutoff '''
        for let in dic.keys():
            if dic[let]/total >= self.gt_then_pi_cutoff:
                gt_then_pi_count += 1

        return gt_then_pi_count > 1

    def mutations_variant_posi_get_table_already_set(self, verbose=False):

        ''' self.fname_mut_freq_variant  = "mutated_posis_for_%s_y%d_m%d_variant_%s.xlsx" '''
        filemut_posi = self.fname_mut_freq_variant%(self.country, self.year, self.month, self.variant_subvar)
        filemut_posi = replace_space(filemut_posi)
        filemutfull = os.path.join(self.root_iedb_summary, filemut_posi)

        if not os.path.exists(filemutfull):
            stri = ">>> Could not find mutation table for %s %d/%d for %s and %s: '%s'"%(self.country, self.year, self.month, self.variant_subvar, self.protein_name, filemutfull)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return None

        try:
            excel = pd.ExcelFile(filemutfull)
        except:
            stri = "Could not open mutation table for %s %d/%d for %s and %s: '%s'"%(self.country, self.year, self.month, self.variant_subvar, self.protein_name, filemutfull)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return None

        sheets = excel.sheet_names

        if self.protein_name not in sheets:
            stri = "Warning: could not find protein '%s' in mutation table for %s %d/%d for %s: '%s'"%(self.protein_name, self.country, self.year, self.month, self.variant_subvar, filemutfull)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return None

        try:
            df = pd.read_excel(filemutfull, sheet_name=self.protein_name)
        except:
            stri = "Could not read sheet '%s' in mutation table for %s %d/%d for %s: '%s'"%(self.protein_name, self.country, self.year, self.month, self.variant_subvar, filemutfull)
            log_save(stri, filename=self.filelog, verbose=True)
            return None

        if df is None or len(df) == 0:
            stri = "Mutation table is empty for %s %d/%d for %s and %s: '%s'"%(self.country, self.year, self.month, self.variant_subvar, self.protein_name, filemutfull)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return None

        stri = "Mutation table is Ok for %s %d/%d for %s and %s"%(self.country, self.year, self.month, self.variant_subvar, self.protein_name)
        log_save(stri, filename=self.filelog, verbose=verbose)

        return df


    def suggested_peptides_calc_ids(self, year, month, variant_subvar, print_prompt=True, verbose=False):
        self.year = year; self.month = month
        self.variant_subvar = variant_subvar

        df = self.mutations_variant_posi_get_table_already_set(verbose=verbose)

        if df is None or len(df) == 0:
            stri = "suggested_peptides_calc_ids(): table not opened"
            log_save(stri, filename=self.filelog, verbose=verbose)
            return None

        dfmeta = df[(df.country_month == '%s%d'%(self.country, self.month)) & (df.frequency >= self.cutoff_freq_country)]

        if len(dfmeta) == 0:
            dfmeta = df[(df.country_month == '%s%d'%(self.country, self.month)) & (df.frequency >= int(self.cutoff_freq_country/2))]

            if len(dfmeta) == 0:
                dfmeta = df[(df.country_month == '%s%d'%(self.country, self.month)) & (df.frequency >= 2)]

                if len(dfmeta) == 0:
                    print("No data for %s - %s - %d with frequcency >= 2"%(self.country, self.protein_name, self.month))
                    return None

        dfa = df[[self.cityRef for x in df.country_month]]

        if len(dfa) > 0:
            dfmeta = pd.concat([dfmeta, dfa])
        else:
            print(">>> nothing found in country_month with the term '%s'"%(self.cityRef))

        ids = dfmeta.description.to_list()

        dfm2 = dfmeta.copy()
        ids = [ x.split('||')[0].strip() for x in ids]
        dfm2['_id'] = ids

        if len(dfa) > 0:
            dfm2['country'] = [self.country]*(len(ids)-1) + [self.countryRef]
        else:
            dfm2['country'] = [self.country]*(len(ids)-1)

        return dfm2

    def entropy_calc_DNA(self, country, years_months, nround=2, verbose=False, force=False):
        _ = self.open_dfcut()

        prev_year = -1; please_save = False
        for year, month in years_months:
            if year != prev_year:
                print("\n>> year", year, end=' ')
                if please_save:
                    ret = self.entropy_save_calc_DNA(dic)

                dic = {}; count = 0; prev_year = year; please_save=True

            print(month, end=' ')

            self.init_vars_year_month(country, None, None, year, month)

            fname = self.fname_entropy_dna_summary%(self.country, self.year)
            filefull = os.path.join(self.root_gisaid_DNA_result, fname)

            if os.path.exists(filefull) and not force:
                please_save = False
                continue

            dfm, sub2variants = self.variants_open_annotated_metadata()

            if len(sub2variants) == 0:
                stri = "entropy calc_DNA(): no sub2variants found for %s %d/%d"%(self.country, self.year, self.month)
                log_save(stri, filename=self.filelog, verbose=verbose)
                continue

            ret, mseq = self.open_nCount_sample_DNA_cym_already_set(verbose=False)
            if not ret:
                stri = "entropy calc_DNA(): could find DNA for %s %d/%d"%(country, year, month)
                log_save(stri, filename=self.filelog, verbose=verbose)
                continue

            for variant_subvar in sub2variants:
                self.variant_subvar = variant_subvar
                variant, subvariant, pango = variant_subvar.split(' ')

                seq_records = [seq_rec for seq_rec in mseq.seq_records if self.variant_subvar_check_in_seq_rec(seq_rec, variant_subvar)]

                if len(seq_records) < 3:
                    stri = "entropy calc_DNA(): there are no sufficient seq_recs %d for variant_subvar %s for %s %d/%d"%(len(seq_records), variant_subvar, self.country, self.year, self.month)
                    log_save(stri, filename=self.filelog, verbose=verbose)
                    continue
                seqs = [ list(seq_rec.seq) for seq_rec in seq_records]

                mseq2 = ms.MySequence(self.prjName, root=self.root_gisaid_prot_entropy)
                mseq2.seq_records = seq_records
                mseq2.seqs = np.array(seqs)

                n = len(mseq2.seq_records)
                L = len(mseq2.seq_records[0])

                ret = self.entropy_fast_calcHShannon_1pos(mseq2, h_threshold=4, add_gaps=False, verbose=False, warning=False)
                if not ret:
                    print(">>> Could not calculate Shannon_1pos: %d - %d"%(year, month))
                    continue

                dic[count] = {}
                dic2 = dic[count]
                count += 1

                dic2['country'] = self.country
                dic2['year']    = year
                dic2['month']   = month
                dic2['variant'] = variant
                dic2['subvariant'] = subvariant
                dic2['yearmonth']   = str(year) + '-' + str(month)
                dic2['CDS'] = 'all'
                dic2['HShannonList'] = str(self.HShannonList)
                dic2['mean1000'] = 1000 * np.mean(self.HShannonList)
                dic2['std1000']  = 1000 * np.std(self.HShannonList)
                dic2['n']        = len(self.HShannonList)
                dic2['SEM']      = dic2['std1000'] / np.sqrt(dic2['n'])

                for icut in range(len(self.dfcut)):
                    row     = self.dfcut.iloc[icut]
                    protein = row.protein
                    start   = row.start - 1
                    end     = row.end

                    dic[count] = {}
                    dic2 = dic[count]
                    count += 1

                    dic2['country'] = self.country
                    dic2['year']    = year
                    dic2['month']   = month
                    dic2['variant'] = variant
                    dic2['subvariant'] = subvariant
                    dic2['yearmonth']  = str(year) + '-' + str(month)
                    dic2['CDS'] = protein

                    lista = self.HShannonList[start:end]

                    dic2['HShannonList'] = str(lista)
                    dic2['mean1000'] = 1000 * np.mean(lista)
                    dic2['std1000']  = 1000 * np.std(lista)
                    dic2['n']        = len(lista)
                    dic2['SEM']      = dic2['std1000'] / np.sqrt(dic2['n'])

        ret = self.entropy_save_calc_DNA(dic)

        return ret

    def entropy_save_calc_DNA(self, dic):
        if len(dic) == 0:
            return False

        df = pd.DataFrame(dic).T

        df.year  = df.year.astype(int)
        df.month = df.month.astype(int)

        fname = self.fname_entropy_dna_summary%(self.country, self.year)
        filefull = os.path.join(self.root_gisaid_DNA_result, fname)

        ret = pdwritecsv(df, fname, self.root_gisaid_DNA_result)

        return ret

    def entropy_get_calc_DNA(self, country, year, nround=2, verbose=False, force=False):
        self.init_vars_year_month(country, None, None, year, None)

        fname = self.fname_entropy_dna_summary%(self.country, year)
        filefull = os.path.join(self.root_gisaid_DNA_result, fname)

        if not os.path.exists(filefull):
            year_months = [(year, x) for x in np.arange(1,13)]
            _ = self.entropy_calc_DNA(country, year_months, nround=nround, verbose=verbose, force=force)

            if not os.path.exists(filefull):
                return None

        df = pdreadcsv(fname, self.root_gisaid_DNA_result)
        return df

    ''' plot_DNA_mean_entropy_evolution ---> plot_DNA_mean_entropy_evolution_all_proteins '''
    def plot_DNA_mean_entropy_evolution_all_proteins(self, country, years, variant_subvar, withORFs=False,
                                        height=900,  width =1700, maxcol=4,
                                        want_save=True, verbose=False):
        variant, subvariant, pango = variant_subvar.split(' ')
        country = country.replace(" ","_")

        dflist = []
        for year in years:
            df = self.entropy_get_calc_DNA(country, year)
            if df is None:
                print("Summary entropy no found for %s %d"%(self.country, self.year))
                continue

            df = df[(df.variant == variant) & (df.subvariant == subvariant) & (df.pango == pango)]
            if len(df) == 0:
                continue

            dflist.append(df)

        if len(dflist) == 0:
            print("There are no data for %s %s %s"%(country, str(years), variant_subvar))
            return None

        df = pd.concat(dflist)
        df['yearmonth'] = ["%s-%s"%(str(df.iloc[i].year), str(df.iloc[i].month) if df.iloc[i].month > 9 else '0'+str(df.iloc[i].month)) for i in range(len(df))]

        if withORFs:
            proteins      = [prot_and_name[0] for prot_and_name in self.protList]
            protein_names = [prot_and_name[1] for prot_and_name in self.protList]
        else:
            proteins      = [prot_and_name[0] for prot_and_name in self.protList if 'ORF' not in prot_and_name[0]]
            protein_names = [prot_and_name[1] for prot_and_name in self.protList if 'ORF' not in prot_and_name[0]]

        proteins = ['all'] + proteins
        protein_names = ['all'] + protein_names

        nprot = len(proteins)
        maxrow = int(np.ceil(nprot/maxcol))

        count = 0
        row = 1; col = 1
        fig = make_subplots(rows=maxrow, cols=maxcol, subplot_titles=proteins)

        for i in range(len(proteins)):
            protein      = proteins[i]
            protein_name = protein_names[i]

            dfa = df[ df.CDS == protein ]
            if len(dfa) == 0: continue

            HShannonList = eval(dfa.iloc[0].HShannonList)
            x = np.arange(0, len(HShannonList))

            fig.add_trace(go.Scatter(x=dfa.yearmonth, y=dfa.mean1000, mode = 'lines',
                                     error_y=dict( type='data',color = plotly_colors_proteins[count],
                                     array=dfa['SEM'],visible=True),
                                     name=protein_name, line=dict(color=plotly_colors_proteins[count], width=2)),
                                     row=row, col=col)

            count += 1
            col   += 1
            if col > maxcol:
                col = 1
                row += 1

        title="Mean Informational Entropy Evolution for %s years %s variant %s"%(country, str(years), variant_subvar)

        fig.update_layout(title=title,
                          height=height,
                          width =width,
                          xaxis_title=None,
                          yaxis_title='mbits')


        if want_save:
            figname = title_replace(title) + ".html"
            figfullname = os.path.join(self.root_gisaid_DNA_html, figname)
            fig.write_html(figfullname)

        return fig

    def entropy_plot_DNA_evolution(self, country, years, protein=None,
                                   colors2=colors2, height=900,  width =1700,
                                   maxrow=5, maxcol=4, want_save=True, verbose=False):
        dflist = []
        for year in years:
            df = self.entropy_get_calc_DNA(country, year)
            if df is None:
                print("Summary entropy no found for %s %d"%(self.country, self.year))
                continue

            dflist.append(df)

        if len(dflist) == 0:
                return None

        df = pd.concat(dflist)

        if protein is None:
            try:
                proteins = ['all'] + list(self.dfcut.protein)
            except:
                _ = self.open_dfcut()
                proteins = ['all'] + list(self.dfcut.protein)
            fig = make_subplots(rows=maxrow, cols=maxcol,
                                x_title='year-month',
                                y_title='mbits',
                                subplot_titles=proteins)
        else:
            if isinstance(protein, list):
                proteins = protein[:2]
            else:
                proteins = ['all', protein]

            fig = make_subplots(rows=2, cols=1,
                                x_title='year-month',
                                y_title='mbits',
                                subplot_titles=proteins)
            maxcol = 1

        sub2variants = self.sub2variants_unique(df)

        colors = {sub2variants[i]: colors2[i] for i in range(len(sub2variants))}

        count = 0; row = 1; col = 1

        for protein in proteins:
            for variant_subvar in sub2variants:
                self.variant_subvar = variant_subvar
                variant, subvariant, pango = variant_subvar.split(' ')

                dfa = df[(df.protein == protein) & (df.variant == variant) & \
                         (df.subvariant == subvariant) & (df.pango == pango) ]

                prot_vari = "%s - %s %s %s"%(protein, variant, subvariant, pango)

                HShannonList = dfa.iloc[0].HShannonList
                x = np.arange(0, len(HShannonList))

                fig.add_trace(go.Scatter(x=dfa.yearmonth, y=dfa.mean1000, mode = 'lines',
                                         error_y=dict( type='data',color=colors[variant], array=dfa['SEM'],visible=True),
                                         name=prot_vari, line=dict(color=colors[variant], width=2)), row=row, col=col)

            count += 1
            col   += 1
            if col > maxcol:
                col = 1
                row += 1

            if row > maxrow:
                print("problems:", row, col)
                break

        virus = 'SARS-CoV-2' if self.virus == 'sarscov2' else self.virus
        title="%s mean DNA Shannon Entropy for %s - %d"%(virus, self.country, year)

        if protein is not None:
            title += "<br>proteins %s and %s"%(proteins[0], proteins[1])

        fig.update_layout(title=title,
                          height=height,
                          width =width,)
                          #xaxis_title='year-month',
                          #yaxis_title='mbits')

        if want_save:
            figname = title_replace(title) + ".html"
            figfullname = os.path.join(self.root_gisaid_DNA_html, figname)
            fig.write_html(figfullname)

        return fig

    def entropy_plot_DNA(self, country, year, month,
                         height=900,  width =1700, want_save=True,
                         h = None, color = 'cyan', line_color = 'royalblue',
                         colors2 = colors2, verbose=False):
        df = self.entropy_get_calc_DNA(country, year)
        if df is None:
            print("Summary entropy no found for %s %d"%(self.country, self.year))
            return None

        sub2variants = self.sub2variants_unique(df)

        colors = {sub2variants[i]: colors2[i] for i in range(len(sub2variants))}

        fig = go.Figure()
        df2 = df[ (df.month == month) & (df.CDS == 'all') ]

        if len(df2) == 0:
            return None

        hmaxi = 0
        for variant_subvar in sub2variants:
            self.variant_subvar = variant_subvar
            variant, subvariant, pango = variant_subvar.split(' ')

            dfa = df2[(df2.CDS == protein) & (df2.variant == variant) & \
                      (df2.subvariant == subvariant) & (df2.pango == pango) ]
            if len(dfa) == 0: continue
            # stri = "%d - %d - %s"%(year, month, variant)

            HShannonList = eval(dfa.iloc[0].HShannonList)
            maxi = np.max(HShannonList)
            if maxi > hmaxi:
                hmaxi = maxi

            x = np.arange(0, len(HShannonList))

            fig.add_trace(go.Scatter(x=x, y=HShannonList, mode = 'lines',
                                     name=variant_subvar, line=dict(color=colors[variant], width=2)))

        count2 = 0

        y = hmaxi * 1.15
        if h is None: h = hmaxi / 20

        try:
            dfcut = self.dfcut.copy()
        except:
            _ = self.open_dfcut()
            dfcut = self.dfcut.copy()

        dfcut = dfcut.sort_values('start')
        dfcut.index = np.arange(0, len(dfcut))

        for icut in range(len(dfcut)):
            row     = dfcut.iloc[icut]
            protein = row.protein
            start   = row.start - 1
            end     = row.end   - 1
            seq_start = row.seq_start
            seq_end   = row.seq_end

            if protein == 'ORF1ab':
                y0 = y
                color = 'gray'
                line_color = 'black'
            else:
                y0 = y+h*(count2+1)
                count2 += 1

                if protein == 'S':
                    color = 'red'
                    line_color = 'darkred'
                elif protein == 'N':
                    color = 'orange'
                    line_color = 'tomato'
                elif protein == 'M':
                    color = 'yellow'
                    line_color = 'gold'
                elif protein == 'E':
                    color = 'green'
                    line_color = 'darkgreen'
                else:
                    color = 'cyan'
                    line_color = 'royalblue'

            if count2 == 3: count2 = 0

            text = "%s<br>start at %d<br>%s"%(protein, start+1, seq_start)

            fig.add_trace(go.Scatter(x=[start, end,  end, start, start],
                                     y=[y0,     y0, y0+h,  y0+h, y0], showlegend=False,
                                     fill='toself', fillcolor=color, line_color=line_color, mode='lines',
                                     hoveron = 'fills', text=text, hoverinfo='x+text'))

            fig.add_annotation(
                    x=(start+end)/2, y=y0+h/2,
                    xref="x", yref="y",
                    text=protein,
                    showarrow=False,
                    # font=dict( family="Courier New, monospace", size=16, color="#ffffff" ),
                    align="left", opacity=1,
                    # bordercolor="#c7c7c7",
                    # borderwidth=2,
                    # borderpad=4,
                    # bgcolor="#ff7f0e",
                    )

        virus = 'SARS-CoV-2' if self.virus == 'sarscov2' else self.virus
        title="%s DNA Shannon Entropy for %s - %d/%d"%(virus, self.country, year, month)

        fig.update_layout(title=title,
                          height=height,
                          width =width,
                          xaxis_title='nucleotide position',
                          yaxis_title='mbits')


        if want_save:
            figname = title_replace(title) + ".html"
            figfullname = os.path.join(self.root_gisaid_DNA_html, figname)
            fig.write_html(figfullname)

        return fig

    '''
    file_seq = 'China_2020_x_Vero.fasta'
    root_seq = '../../../colaboracoes/renato'
    Nexp = how many last lines correspond to the experimental sequences
    '''
    def consensus_and_experiment_dif_between(self, file_seq, Nexp, root_seq, verbose = False):

        filefull_seq = os.path.join(root_seq, file_seq)

        if not os.path.exists(filefull_seq):
            if verbose: print("No data found for '%s'"%(filefull_seq))
            return None, None, None, None

        mseq = ms.MySequence(self.prjName)
        ret = mseq.readFasta(filefull_seq, showmessage=verbose)
        seqs_china = mseq.seqs[:(len(mseq.seqs) - Nexp), :]

        aa = self.consensus_get_aa_from_seqs(seqs_china)

        ''' rebuild seq_records '''
        seq_record = SeqRecord(
                            seq=Seq(aa),
                            description='',
                            id='china_consensus', name=''
                        )

        seq_records = [seq_record]
        for i in range((len(mseq.seqs)-4), len(mseq.seqs)):
            seq_records.append(mseq.seq_records[i])

        if ".fasta" in file_seq:
            file_new = file_seq.replace(".fasta", "_consensus.fasta")
        elif ".fas" in file_seq:
            file_new = file_seq.replace(".fas", "_consensus.fas")
        else:
            print("Error: fasta file name.")
            return None, None, None, None

        filefull_new = os.path.join(root_seq, file_new)

        mseq1 = ms.MySequence(self.prjName)
        ret   = mseq1.writeSequences(seq_records, filefull_new, verbose=verbose)

        text_mut, msg, dicmut, mseq1 = self.consensus_and_experiment_calc_dif_between(filefull_new, root_seq, verbose=verbose)
        return aa, text_mut, msg, dicmut, mseq1

    def consensus_and_experiment_calc_dif_between(self, filefull_new, root_seq, verbose = False):
        mseq1 = ms.MySequence(self.prjName)
        ret   = mseq1.readFasta(filefull_new, showmessage=verbose)
        aa = str(mseq1.seq_records[0].seq)

        mseq2 = ms.MySequence(self.prjName, root=root_seq)
        mseq2.seqs = mseq1.seqs[1:,:]
        mseq2.seq_records = mseq1.seq_records[1:]

        ret = self.entropy_fast_calcHShannon_1pos(mseq2, h_threshold=4, add_gaps=True, verbose=False, warning=False)
        if not ret:
            return None, None, None, None

        count = -1; ok = 0; diff = 0; dicmut = {}
        text_mut = ''
        for dic in self.dicPiList:
            count += 1

            nucl = aa[count]
            if nucl == '-': continue

            ok += 1

            if len( list(dic.keys())) == 1:
                for key in dic.keys():
                    if key != nucl:
                        diff += 1
                        text_mut += "pos: %d, %s -> %s, %.1f %%\n"%(count+1, nucl, key, 100.)

                        stri = nucl + '->' + key
                        dicmut[stri] = dicmut.get(stri, 0.) + 1
            else:
                diff += 1; first = True
                for key in dic.keys():
                    # print(count+1,  nucl, '->', key, np.round(100*dic[key], 1), end= '; ')
                    if key != nucl:
                        if first:
                            text_mut += "pos: %d, %s -> %s = %.1f %%"%(count+1, nucl, key, 100*dic[key] )
                        else:
                            text_mut += "; %s = %.1f %%"%(key, 100*dic[key] )

                        stri = nucl + '->' + key
                        dicmut[stri] = dicmut.get(stri, 0.) + 1
                text_mut += '\n'

        if diff == 0:
            msg = 'There are no mutations'
        else:
            msg = 'There are %d nucleotides, %d mutations, %.3f %%, one mutation for each %d nucleotide'%(\
                   ok, diff, 100*(diff/ok), int(ok/diff) )

        return text_mut, msg, dicmut, mseq1

    ''' variants_calc_pseudo_subvariants_many --> variants_calc_pseudo_subvariants_loops
        pseudovar_calc_subvariants_loop
        remove protList
    '''
    def pseudovar_calc_subvariants_loop(self, countries, years_months, verbose=False):
        for country in countries:
            if verbose: print(">>>", country, end=' ')
            for protein, protein_name in self.protList:
                if verbose: print(protein_name, end=' ')
                prev_year = -1
                for year, month in years_months:
                    if prev_year != year:
                        prev_year = year
                        if verbose: print(year, end='-')

                    if verbose: print(month, end=' ')
                    self.pseudovar_calc_main_subvariants(country, protein, protein_name, year, month, verbose=verbose)

                if verbose: print("")
            if verbose: print("\n")

    def pseudovar_calc_main_subvariants(self, country, protein, protein_name, year, month, verbose=False):

        self.init_vars_year_month(country, protein, protein_name, year, month)
        dfvar, sub2variants = self.mutations_find_dfvar_already_set()

        for variant_subvar in sub2variants:
            # # print("$$$ 0", country, protein, protein_name, year, month, variant_subvar)
            self.variant_subvar = variant_subvar
            _, _, _ = self.pseudovar_get_dfm_cols_posi_already_set(verbose=verbose)


    ''' mutations_build_table --> variants_calc_pseudo_subvariants  --> pseudovar get_dfm_cols_posi'''
    def pseudovar_get_dfm_cols_posi_already_set(self, verbose=False):
        ''' returns --> dfms2, cols, pos_ini'''

        col_header = ['id', 'year', 'month', 'collection_date', 'description','id_list', 'variant', 'subvariant', 'pango', 'frequency']

        dfms, ids, sampleFreqs = \
            self.pseudovar_get_main_subvariants_already_set(verbose=verbose)

        if dfms is None or len(ids) == 0:
            return None, None, None

        cols = list(dfms.columns)
        cutcol = [i for i in range(len(cols)) if cols[i] == 'frequency'][0]

        stri = "dfms len %d for %s - %s %d/%d - %s"%(len(dfms), self.country, self.protein_name, self.year, self.month, self.variant_subvar)
        log_save(stri, filename=self.filelog, verbose=verbose)

        ''' if already done '''
        if 'subvariant' in cols:
            stri = "### subvariant Ok for %s, %s, %d/%d - %s, cutoff_freq_pseudvar %d x max(freq)=%d"%(self.country, self.protein_name, self.year, self.month, self.variant_subvar, self.cutoff_freq_pseudvar, dfms.frequency.max())
            log_save(stri, filename=self.filelog, verbose=verbose)

            dfms2 = dfms
            ''' ncols in the end ...'''
            ncols = cols[(cutcol+1):]
        else:
            stri = "### subvariant BUILDING for %s, %s, %d/%d - %s, cutoff_freq_pseudvar %d x max(freq)=%d"%(self.country, self.protein_name, self.year, self.month, self.variant_subvar, self.cutoff_freq_pseudvar, dfms.frequency.max())
            stri += "len(dfms) %d"%(len(dfms))
            log_save(stri, filename=self.filelog, verbose=verbose)

            ''' otherwise, calculate ...
                ncols in the begining '''
            ncols = cols[:cutcol]

            coll_date_list, variant_list, pango_list, subvariant_list, clade_list = \
            [], [], [], [], []

            for i in range(len(dfms)):
                desc = dfms.iloc[i].description
                _id, count_read, _, _, _, \
                coll_date, host, concern, pango, pango_complete, clade, \
                variant, subvariant, subvariant1, subvariant2, region = self.split_fasta_head(desc)

                coll_date_list.append(coll_date)
                pango_list.append(pango)
                variant_list.append(variant)
                subvariant_list.append(subvariant)
                clade_list.append(clade)

            dfms2 = copy.deepcopy(dfms)

            dfms2['collection_date'] = coll_date_list
            dfms2['pango']      = pango_list
            dfms2['variant']    = variant_list
            dfms2['subvariant'] = subvariant_list
            dfms2['clade']      = clade_list
            dfms2['year']       = self.year
            dfms2['month']      = self.month

            dfms2['id'] = [desc.split('||')[0] for desc in dfms2.description]

            fields = col_header + ncols
            dfms2 = dfms2[fields].copy()
            try:
                dfms2.frequency = dfms2.frequency.astype(int)
            except:
                pass

            filemut_var = self.fname_pseudo_variant_main_mutated%(self.country, self.protein_name, self.year, self.month, self.variant_subvar)
            filemut_var = replace_space(filemut_var)
            pdwritecsv(dfms2, filemut_var, self.root_iedb_pseudovar)

            filefull = os.path.join(self.root_iedb_pseudovar, filemut_var)
            stri = "File pseudo_variant_main_mudated saved for max(freq)=%d: '%s'"%(dfms2.frequency.max(), filefull)
            log_save(stri, filename=self.filelog, verbose=verbose)

        self.dfms2 = dfms2
        fields = col_header + ncols
        dfms2 = dfms2[fields].copy()

        try:
            dfms2.frequency = dfms2.frequency.astype(int)
        except:
            pass

        pos_ini = len(col_header)

        return dfms2, cols, pos_ini


    def pseudovar_get_multi_subvariants(self, country, protein, protein_name, year, month, verbose=False):
        self.init_vars_year_month(country, protein, protein_name, year, month)

        dfvar, sub2variants = self.mutations_find_dfvar_already_set()
        if len(sub2variants) == 0:
            self.variant_subvar = None
            stri = "pseudovar_get_multi_subvariants(): EMPTY for %s - %s - %d/%d"%(self.country, self.protein_name, self.year, self.month)
            log_save(stri, filename=self.filelog, verbose=True)
            return None, [], []

        fields = ['frequency', 'description','id_list']

        df_list = []
        for variant_subvar in sub2variants:
            self.variant_subvar = variant_subvar
            dfms, ids, sampleFreqs = self.pseudovar_get_main_subvariants_already_set(verbose=verbose)

            if dfms is None or ids == []:
                continue

            dfms = dfms[fields]
            dfms['variant_subvar'] = variant_subvar
            df_list.append(dfms)

        if df_list == []:
            stri = "pseudovar_get_multi_subvariants(): EMPTY(2) for %s - %s - %d/%d"%(self.country, self.protein_name, self.year, self.month)
            log_save(stri, filename=self.filelog, verbose=True)
            return None, [], []

        dfms = pd.concat(df_list)
        dfms = dfms.sort_values('frequency', ascending=False)
        dfms.index = np.arange(0, len(dfms))

        id_list = [desc.split('||')[0] for desc in dfms.description]
        sampleFreq_list = dfms.frequency.to_list()

        return dfms, id_list, sampleFreq_list

    def pseudovar_get_main_subvariants_set_year_month_variant(self, year, month, variant_subvar, verbose=False):

        self.init_vars_year_month(self.country, self.protein, self.protein_name, year, month)
        self.variant_subvar = variant_subvar

        return self.pseudovar_get_main_subvariants_already_set(verbose=verbose)

    def pseudovar_get_main_subvariants_already_set(self, force=False, verbose=False):

        dfms = self.pseudovar_calc_main_subvariants_already_set(force=force, verbose=verbose)
        ids = []; sampleFreqs=[]

        if dfms is None or dfms.empty:
            ''' could not find above '''
            if not force:
                stri = "Warning: try to recalc dfms for %s, %s, %d/%d - %s"%(self.country, self.protein_name, self.year, self.month, self.variant_subvar)
                log_save(stri, filename=self.filelog, verbose=verbose)
                dfms = self.pseudovar_calc_main_subvariants_already_set(force=True, verbose=verbose)

            if dfms is None or dfms.empty:
                stri = "Error?: could not find dfms for %s %s %d/%d - %s"%(self.country, self.protein_name, self.year, self.month, self.variant_subvar)
                log_save(stri, filename=self.filelog, verbose=verbose)
                return dfms, ids, sampleFreqs

        # stri = ">>> dfms found for %s %s %d/%d - %s, max(freq)=%d"%(self.country, self.protein_name, self.year, self.month, self.variant_subvar, dfms.frequency.max())
        # log_save(stri, filename=self.filelog, verbose=verbose)

        dfms = dfms.sort_values('frequency', ascending=False)

        maxi = dfms.frequency.max()
        if maxi >= self.cutoff_freq_pseudvar:
            dfms = dfms[ dfms.frequency >= self.cutoff_freq_pseudvar ]
        else:
            ''' low frequency '''
            dfms = dfms[ dfms.frequency > 2 ]

            stri = ">>> pseudovar get_main_subvariants_already_set(): low frequency: '%s' for %s %s %d/%d - max(freq)=%d, cutoff = %d"%\
                   (self.variant_subvar, self.country, self.protein_name, self.year, self.month, maxi, self.cutoff_freq_pseudvar)
            log_save(stri, filename=self.filelog, verbose=verbose)

        if not dfms.empty:
            dfms.index = np.arange(0, len(dfms))
            ids = [x.split('||')[0] for x in dfms.description]
            sampleFreqs = dfms.frequency.to_list()

            stri = ">>> --------- dfms is Ok: pseudovar get_main_subvariants_already_set ------ %s %s y%d/m%d - %s"%(self.country, self.protein_name, self.year, self.month, self.variant_subvar)
        else:
            stri = ">>> --------- dfms is Empty: pseudovar get_main_subvariants_already_set ------ %s %s y%d/m%d - %s"%(self.country, self.protein_name, self.year, self.month, self.variant_subvar)

        log_save(stri, filename=self.filelog, verbose=verbose)

        return dfms, ids, sampleFreqs


    ''' get_calc_main_strains --> sub2variants calc_main_pseudo_subvariants '''
    def pseudovar_calc_main_subvariants_already_set(self, force=False, verbose=False):

        '''  self.fname_pseudo_variant_main_mutated: "pseudo_variant_main_mutated_posis_for_%s_%s_y%d_m%d_variant_%s.tsv" '''
        filemut_var = self.fname_pseudo_variant_main_mutated%(self.country, self.protein_name, self.year, self.month, self.variant_subvar)
        filemut_var = replace_space(filemut_var)
        filemutfull = os.path.join(self.root_iedb_pseudovar, filemut_var)

        '''
        if 'Omicron' in self.variant_subvar and self.protein_name == 'Spike':
            print(">>> file", filemutfull)
        '''

        if os.path.exists(filemutfull) and not force:
            dfms = pdreadcsv(filemut_var, self.root_iedb_pseudovar)

            try:
                dfms.frequency = dfms.frequency.astype(int)
            except:
                pass
        else:
            df = self.pseudovar_read_variant_already_set(verbose=verbose)
            if df is None or len(df) == 0:
                dfms = None
            else:
                stri = ">>> variants calc_main_pseudo_simple_table for %s %s - %d/%d - %s"%(self.country, self.protein_name, self.year, self.month, self.variant_subvar)
                log_save(stri, filename=self.filelog, verbose=verbose)
                dfms_qq = self.variants_calc_main_pseudo_simple_table(df, verbose=verbose)
                dfms = dfms_qq.copy()

                if dfms is not None and len(dfms) > 0:
                    try:
                        dfms.frequency = dfms.frequency.astype(int)
                    except:
                        pass

                    dfms.index = np.arange(0, len(dfms))

                    ret = pdwritecsv(dfms, filemut_var, self.root_iedb_pseudovar)
                    if not ret:
                        stri = ">>> ERROR: %s"%(filemut_var)
                        log_save(stri, filename=self.filelog, verbose=True)
                        dfms = None
                    else:
                        stri = ">>> saving file %s - %s - %d/%d - %d possible sub2variants - '%s'"%(self.country, self.protein_name, self.year, self.month, len(dfms), filemut_var)
                        log_save(stri, filename=self.filelog, verbose=verbose)
                else:
                    stri = "Warning: empty file for %s %s %d/%d - %s: '%s'"%(self.country, self.protein_name, self.year, self.month, self.variant_subvar, filemut_var)
                    log_save(stri, filename=self.filelog, verbose=True)
                    dfms = None

        return dfms

    def which_perc(self, row, dfg):
        total = dfg[ (dfg.year == row.year) & (dfg.month == row.month)].iloc[0].total
        return row.frequency / total

    ''' mutations_merge_table --> pseudovar_merge_tables '''
    def pseudovar_merge_tables(self, country, protein, protein_name, years_months, variant_subvar, verbose=False):
        first = True
        for year, month in years_months:

            self.init_vars_year_month(country, protein, protein_name, year, month)
            self.variant_subvar = variant_subvar

            dfm, cols, pos_ini = self.pseudovar_get_dfm_cols_posi_already_set(verbose=verbose)
            if dfm is None: continue

            if first:
                first = False
                dfall = copy.deepcopy(dfm)
                continue

            ncols_all = list(dfall.columns)[pos_ini:]
            common = np.intersect1d(ncols_all, cols)

            diff_all  = [x for x in cols     if x not in common]
            diff_this = [x for x in ncols_all if x not in common]

            if len(diff_all) > 0:
                for col in diff_all:
                    dfall[col] = None

            if len(diff_this) > 0:
                for col in diff_this:
                    dfm[col] = None

            cols = list(dfall.columns)[pos_ini:]
            cols.sort()

            fields = col_header + cols
            dfall = dfall[fields]
            dfm   = dfm[fields]

            dfall = pd.concat([dfall, dfm])

        if first:
            return None

        if cols[-1] == 'description':
            cols = cols[:-1]
            dfall = dfall.iloc[:,:-1]

        dfall.columns = col_header + [int(x) for x in cols]

        ncol_list = [int(x) for x in cols]
        ncol_list.sort()
        fields = col_header + ncol_list
        dfall = dfall[fields]

        dfg = dfall.groupby(['year', 'month']).agg({'frequency': 'sum'}).reset_index()
        dfg.columns = ['year', 'month', 'total']

        if len(dfall) <= 1:
            perc_list=[1]
        else:
            perc_list = []
            for i in range(len(dfall)):
                row = dfall.iloc[i]
                perc = self.which_perc(row, dfg)
                perc_list.append(perc)

        dfall2 = dfall.copy()
        dfall2['perc'] = perc_list
        return dfall2

    def mutations_plot_evolution_in_time(self, country, protein_name, dfall, colors, is_barmode=True,
                                         title=None, template='plotly_white',
                                         width_layout = 1400, height_layout = 900,
                                         fontsize = 12, fontcolor = 'navy', xaxis_tickangle=90,
                                         xaxis_title = 'year-month', yaxis_title = 'percentage (%)',
                                         savefig = True):

        if title is None: title = "mutational evolution for %s, %s"%(country, protein_name)

        fig = go.Figure()
        cols = list(dfall.columns)[5:-2]

        fields = ['id', 'variant', 'subvariant', 'pango', 'year', 'month', 'perc']
        df2 = dfall[fields].copy()

        df2['year_month'] = [ str(df2.iloc[i].year) + '-' + (str(df2.iloc[i].month) if df2.iloc[i].month > 9 else '0' + \
                              str(df2.iloc[i].month) ) for i in range(len(df2))]

        sub2variants = self.sub2variants_unique(df2)

        count = -1
        for variant_subvar in sub2variants:
            self.variant_subvar = variant_subvar
            count += 1
            variant, subvariant, pango = variant_subvar.split(' ')

            df3 = df2[ (df2.variant == variant) & (df2.subvariant == subvariant) & (df2.pango == pango)]
            texts = ["perc: %.2f%%<br>year-month: %s<br>variant: %s %s %s<br>ID: %s"% \
                    (df3.iloc[i].perc *100,  df3.iloc[i].year_month, df3.iloc[i].variant, \
                     df3.iloc[i].subvariant, df3.iloc[i].pango,      df3.iloc[i]['id']) for i in range(len(df3))]
            if is_barmode:
                fig.add_trace(go.Bar(x = df3.year_month, y = df3.perc,
                              name=variant_subvar, text='', hoverlabel=None,
                              hovertemplate = texts, marker_color=colors[count]) )
            else:
                fig.add_trace(go.Scatter(x = df3.year_month, y = df3.perc, name=variant_subvar, mode='markers',
                              text=texts, hovertemplate = "%{text}",
                              marker=dict( color=colors[count], size=10,  line=dict( color='black', width=2) ) ) )

        fig.update_layout(
            autosize = False,
            barmode = 'stack' if is_barmode else None,
            title=title,
            width=width_layout,
            height=height_layout,
            template=template,
            margin=dict(l=40, r=40, b=40, t=140, pad=4),
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            xaxis_tickangle=xaxis_tickangle,
            font=dict(family="Arial, bold, monospace", size=fontsize, color=fontcolor ),
            paper_bgcolor="whitesmoke",
            plot_bgcolor= "whitesmoke", # lightgrey ivory gainsboro whitesmoke lightsteelblue 'lightcyan' 'azure', white, lightgrey, snow ivory beige powderblue
            showlegend=True
        )

        if savefig:
            figname = title_replace(title) + ".html"
            filehtml = os.path.join(self.root_iedb_html, figname)
            stri = ">>> File saved: %s"%(filehtml)
            log_save(stri, filename=self.filelog, verbose=verbose)
            fig.write_html(filehtml)

        return fig

    def mutual_information_calc_vertical_2nuc_seqs(self, seqs1, seqs2, numOfLetters=3, frameshift=0):
        self.entropy_calcHSannon_bias_correction(seqs1, numOfLetters=numOfLetters)
        # h1 = self.HShannonCorrList
        h1 = self.HShannonList

        self.entropy_calcHSannon_bias_correction(seqs2, numOfLetters=numOfLetters)
        # h2 = self.HShannonCorrList
        h2 = self.HShannonList

        n1 = seqs1.shape[0]
        n2 = seqs2.shape[0]

        c1 = seqs1.shape[1]
        c2 = seqs2.shape[1]

        cMax1 = c1 - numOfLetters + 1
        cMax2 = c2 - numOfLetters + 1

        if n1 != n2:
            stri = "Both nrows must be equal"
            log_save(stri, filename=self.filelog, verbose=True)

        MI = np.zeros((c1//numOfLetters,c2//numOfLetters))
        seqsTwo = np.empty( (n1, 2*numOfLetters), dtype='str')

        #---------- the diagonal is 0-----------------------------
        #--- calc half matrix .... -------------------------------
        hot = {}; ci=0
        for i in np.arange(frameshift, cMax1, numOfLetters):
            cj=0
            for j in np.arange(frameshift, cMax2, numOfLetters):
                seqsTwo[:,0           :   numOfLetters ] = seqs1[:,i:(i+numOfLetters)]
                seqsTwo[:,numOfLetters:(2*numOfLetters)] = seqs2[:,j:(j+numOfLetters)]

                self.entropy_calcHSannon_bias_correction(seqsTwo, numOfLetters=2*numOfLetters)

                # mi = h1[ci] + h2[cj] - self.HShannonCorrList[0]
                mi = h1[ci] + h2[cj] - self.HShannonList[0]

                if mi > 0.01:
                    MI[ci,cj] = mi

                    if mi > .05:
                        # print(ci, cj, "\t", h1[ci], "\t", h2[cj], "\t", self.HShannonCorrList[0], "\t", mi)
                        hot[(ci,cj)] = [h1[ci], h2[cj], mi]
                cj += 1
            ci += 1

        self.MI = MI; self.hot = hot
        return MI, hot



    def mutual_information_plot_heatmap_2nuc_seqs(self, seqs1, seqs2, title="", numOfLetters=3, frameshift = 2, maxMER=None,
                                  xlabel = "nucleotides", ylabel = "nucleotides"):
        MI, hot = self.mutual_information_calc_vertical_2nuc_seqs(seqs1, seqs2, numOfLetters=numOfLetters, frameshift=frameshift)


        if maxMER == None:
            maxi = MI.max()
        else:
            maxi = maxMER

        x, y = np.meshgrid(np.arange(0,MI.shape[0]+1), np.arange(0,MI.shape[1]+1))
        z_min, z_max = 0, maxi

        fig, ax = plt.subplots(figsize=(12,9))
        c = ax.pcolormesh(x, y, MI.T, cmap='PuRd', vmin=z_min, vmax=z_max)  # cmap='YlOrRd'

        title = "%s \n max MI = %.3f bits \n numOfLetters = %d, frameshift=%d"% \
             (title, MI.max(), numOfLetters, frameshift)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # set the limits of the plot to the limits of the data
        ax.axis([x.min(), x.max(), y.min(), y.max()])
        fig.colorbar(c, ax=ax)



    def mutual_information_calc_2_seqs(self, seqs1, seqs2, letters, frameshift):
        MIs = []

        for numOfLetters in letters:
            MI, _ = self.mutual_information_calc_vertical_2nuc_seqs(seqs1, seqs2, numOfLetters=numOfLetters, frameshift=frameshift)
            MIs.append(MI.max())

        return MIs

    def mutual_information_plot(self, dic, letters, normalize=False):
        fig = plt.figure(num=None, figsize=(8, 6), dpi=600, facecolor='w', edgecolor='k')
        ax = plt.gca()

        colors = ["blue", "red", "green"]

        if normalize:
            ax.set_ylabel('Normalized MI (bits)')
            plt.title("Normalized Mutual information per #nucleotides per frameshift")

            for frameshift in dic.keys():
                dic[frameshift] = [dic[frameshift][i]/letters[i] for i in range(len(letters))]

        else:
            ax.set_ylabel('MI (bits)')
            plt.title("Mutual information per #nucleotides per frameshift")

        for frameshift in dic.keys():
            ax.plot(letters, dic[frameshift], color=colors[frameshift], label="frameshift: %d"%frameshift)

        plt.xticks(letters)
        ax.grid()

        ax.legend(loc='upper right')

        ax.set_xlabel("num of nucleotides")


    '''  save_reference_fasta --> save_protein_reference_fasta
         deprecated: see reference_get_save_consensus(self, force=False, verbose=False)
    '''
    def save_protein_reference_fasta_deprecated(self, protein, protein_name, refID=None, verbose=False):

        self.init_vars(self.countryRef, protein, protein_name)
        self.year = self.yearRef
        self.month = self.monthRef

        file_prot_cym = self.file_prot_cym_reference%(self.yearRef, self.monthRef)
        filefull_prot = os.path.join(self.root_gisaid_prot_fasta, file_prot_cym)

        if os.path.exists(filefull_prot):
            return True

        ret, mseq_cym = self.open_all_or_sampled_protein_CYM(self.countryRef, self.yearRef, self.monthRef,
                                                             protein, protein_name, verbose=verbose)

        if not ret:
            return False

        seq_records = []

        if refID is None:
            for seq_record in mseq_cym.seq_records:
                ''' get first '''
                seq_records = [seq_record]
                break
        else:
            for seq_record in mseq_cym.seq_records:
                if refID in seq_record.description:
                    seq_records = [seq_record]
                    break

        if len(seq_records) == 0:
            stri = ">>> save reference_fasta: could not find refID %s"%(refID)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return False

        ret = mseq_cym.writeSequences(seq_records, filefull_prot, verbose=verbose)
        return ret
