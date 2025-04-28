#!/usr/bin/python
#!python
# -*- coding: utf-8 -*-

'''
# @Updated on 2023/01/26, 2022/12/06, 2022/09/28, 2022/06/20 - 2022/04/25
# @Created on 2020/06/10
# @author: Flavio Lichtenstein
# @email:  flalix@gmail.com, flavio.lichtenstein@butantan.gov.br
# @local:  Instituto Butantan / CENTD / Molecular Biology / Bioinformatics & Systems Biology
'''

import copy, os, re, random, sys
from filelock import FileLock  # Timeout
# from math import pow
from collections import OrderedDict
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Iterable, Set, Tuple, Any, List

import BioPythonClass
import Sequence as ms
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

# import virus_seqs_lib
from Basic import *
b = Basic()
nucs = b.getDnaNucleotides()

from util_general import *
from Shannon import Shannon

import plotly
import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots
# from seq_toofvlbox import *

b = BioPythonClass.Basic()

class Gisaid(Shannon):

    def __init__(self, prjName:str, version:str, isGisaid:bool, isProtein:bool, root0:str,
                 countryRef:str, cityRef:str, yearRef:int, monthRef:int, descriptionRef:str,
                 protList:List, dicNucAnnot:dict, virus:str, previous_version:str=None,
                 nCount:int=500, year_max:int=None, month_max:int=None, one_month:int=None,
                 gt_then_pi_cutoff:float=0.05, cutoff_freq_pseudvar:int=3, cutoff_freq_country:int=20,
                 badAA_cutoff_perc:float=0.10, min_gaps:float=0.20, cutoff_good:float=0.25,
                 fileReffasta:str=None, file_big_fasta=None, date_mask:str='%Y-%m-%d',
                 file_template:str='table_template.html', seaviewPath:str = '../../tools/seaview'):

        if year_max  is None: year_max  = int(version[:4])
        if month_max is None: month_max = int(version[4:6])

        
        super().__init__(prjName=prjName, version=version, isGisaid=isGisaid, isProtein=isProtein, root0=root0,
                         countryRef=countryRef, cityRef=cityRef, yearRef=yearRef, monthRef=monthRef, descriptionRef=descriptionRef,
                         protList=protList, dicNucAnnot=dicNucAnnot, virus=virus, previous_version=previous_version,
                         nCount=nCount, year_max=year_max, month_max=month_max, one_month=one_month,
                         gt_then_pi_cutoff=gt_then_pi_cutoff, cutoff_freq_pseudvar=cutoff_freq_pseudvar,  
                         cutoff_freq_country=cutoff_freq_country, 
                         badAA_cutoff_perc=badAA_cutoff_perc, min_gaps=min_gaps, cutoff_good=cutoff_good,
                         fileReffasta=fileReffasta, file_big_fasta=file_big_fasta, date_mask=date_mask, 
                         file_template=file_template, seaviewPath=seaviewPath)

        '''
        header metadata:
        [_id, prev_count, country, year, month, coll_date, host, concern, pango, pango_complete, clade, \
         variant, subvariant, subvariant1, subvariant2, region]]
        '''
        self.head_meta = 'id\tcount\tcountry\tyear\tmonth\tcoll_date\thost\tconcern\tpango\tpango_complete\tclade\tvariant\tsubvariant\tsubvariant1\tsubvariant2\tregion'

        self.fasta_fields = ['id', 'count_fasta', 'country', 'year', 'month',  \
                             'coll_date', 'host', 'concern', 'pango', 'pango_complete', 'clade', 'variant',
                             'subvariant', 'subvariant1', 'subvariant2', 'region']

        '''
        header data: <br>
        _id, count_read, country, year, month, coll_date, host, concern, pango, pango_complete, clade,
         variant, subvariant, subvariant1, subvariant2, region = head.split('||')
        '''

        self.dfcodons, self.dicTrans, self.stop_list = None, None, None

        self.cols_gisaid_metadata = \
            ['Virus name', 'Last vaccinated', 'Passage details/history',
             'Type', 'Accession ID', 'Collection date', 'Location',
             'Additional location information', 'Sequence length', 'Host',
             'Patient age', 'Gender', 'Clade', 'Pango lineage', 'Pango version',
             'Variant', 'AA Substitutions', 'Submission date', 'Is reference?',
             'Is complete?', 'Is high coverage?', 'Is low coverage?', 'N-Content',
             'GC-Content']

        self.cols_custom_gisaid_metadata = \
            ['virus', 'last_vaccinated', 'history', 'virus_type', 'id', 'coll_date',
             'location', 'add_location', 'seq_length', 'host',
             'pat_age', 'pat_gender', 'clade', 'pango_lineage', 'pango_version',
             'variant', 'aa_substitutions', 'submission_date', 'is_reference',
             'is_complete', 'is_high_coverage', 'is_low_coverage', 'n_content', 'gc_content']

        self.cols_checked_gisaid_metadata = \
            ['virus', 'last_vaccinated', 'history', 'virus_type', 'id', 'coll_date',
             'year', 'month', 'day', 'location', 'country', 'continent', 'region',
             'add_location', 'seq_length', 'host',
             'pat_age', 'pat_gender', 'clade', 'clade_gisaid', 'pango', 'pango_complete', 'pango_version',
             'variant_gisaid', 'concern', 'variant', 'subvariant', 'subvariant1', 'subvariant2',
             'aa_substitutions', 'submission_date', 'is_reference',
             'is_complete', 'is_high_coverage', 'is_low_coverage', 'n_content', 'gc_content']


    def init_vars_year_month(self, country, protein, protein_name, year, month):
        super().init_vars_year_month(country, protein, protein_name, year, month)

    def open_metadata_last(self):
        print("reading metadata (last version) ... ", end = '')
        self.dfmeta_last = pdreadcsv(self.file_metadata, self.root_meta_fasta_last)
        if self.dfmeta_last is None or len(self.dfmeta_last) == 0:
            print("Error: dfmeta_last")
            return False

        print("ok - dfmeta_last")
        return True



    ''' check_human_gisaid_metadata --> check_gisaid_metadata
        there is no 'force' - if necessary delete the metadata checked file '''
    def check_gisaid_metadata(self, verbose=False):
        '''
        self.file_metadata = 'metadata_%s.tsv'%(version)
        '''
        filefull = os.path.join(self.root_gisaid_data, self.file_metadata)
        if not os.path.exists(filefull):
            print("Error: metadata was not downloaded: '%s'"%(filefull))
            return False

        stri = "reading checked metadata, be patient ..."
        log_save(stri, filename=self.filelog, verbose=True)
        self.dfmeta = pdreadcsv(self.file_metadata, self.root_gisaid_data)

        if np.sum(self.dfmeta.columns == self.cols_gisaid_metadata) != 24:
            stri = "Error: dfmeta has unrecognizable columns (ncols = 24). Columns: '%s'"%(str(list(self.dfmeta.columns)))
            log_save(stri, filename=self.filelog, verbose=True)
            return False

        self.dfmeta.columns = self.cols_custom_gisaid_metadata
        stri = ">> original metadata has %.3f million lines"%(len(self.dfmeta)/1000000)
        log_save(stri, filename=self.filelog, verbose=True)

        dfm = self.dfmeta

        locs = dfm.location.to_list()
        locsmat = [x.split('/') for x in locs]

        continents  = [x[0].strip() for x in locsmat]
        countries   = [x[1].strip() for x in locsmat]
        region_list = [" - ".join(x[2:]).strip() if len(x) >= 2 else None for x in locsmat]

        year  = [int(x.split('-')[0]) for x in dfm.coll_date]
        month = [int(x.split('-')[1]) if len(x)>=7  else -1 for x in dfm.coll_date]
        day   = [int(x.split('-')[2]) if len(x)>=10 else 15 for x in dfm.coll_date]

        '''
            Virus name  Last vaccinated Passage details/history Type
            Accession ID    Collection date Location    Additional location information
            Sequence length Host    Patient age Gender  Clade   Pango lineage   Pango version
            Variant AA Substitutions    Submission date Is reference?
            Is complete?    Is high coverage?   Is low coverage?    N-Content   GC-Content


            self.cols_custom_gisaid_metadata = \
                ['virus', 'last_vaccinated', 'history', 'virus_type',
                 'id', 'coll_date', 'location',
                 'add_location', 'seq_length', 'host',
                 'pat_age', 'pat_gender', 'clade', 'pango', 'pango_complete', 'pango_version',
                 'variant', 'aa_substitutions', 'submission_date', 'is_reference',
                 'is_complete', 'is_high_coverage', 'is_low_coverage', 'n_content', 'gc_content']
        '''
        pango_list, clade_gisaid_list, subvariant_list, concern_list = [], [], [], []
        variant_list, variant_gisaid_list, subvariant1_list, subvariant2_list = [], [], [], []
        unknown = 'unknown'

        for i in range(len(dfm)):
            if i % 200000 == 0: print(i, end=' ')

            row = dfm.iloc[i]

            _clade        = row['clade']
            pango         = row['pango_lineage']
            variant       = row['variant']

            if isinstance(variant, str):
                mat = variant.split(' ')
                variant = variant.replace(" ", "_").replace("/", "_")
                variant_gisaid_list.append(variant)
            else:
                variant_gisaid_list.append(unknown)
                mat = [unknown]

            if isinstance(pango, str):
                pango = pango.replace(" ", "_").replace("/", "_")
            else:
                pango = unknown

            pango_list.append(pango)

            '''
            VOC Delta GK (B.1.617.2+AY.*)
            VOC Omicron GRA (B.1.1.529+BA.*)
            VOC Delta GK (B.1.617.2+AY.*)
            unknown unknown unknown unknown
            VOC XXXX  XXX P.1+P.1.

            example: "VOC Omicron GRA (B.1.1.529+BA.*)"
            '''
            if len(mat) > 3:
                mat[3] = mat[3].replace('(','').replace(')','')
                mat3   = mat[3].split('+')
                subvariant = mat3[0].replace("/", "_")
                matsubvar = subvariant.split('.')

                if len(matsubvar) > 1:
                    subvariant1 = ".".join(matsubvar[:2])
                    if len(matsubvar) > 2:
                        subvariant2 = ".".join(matsubvar[:3])
                    else:
                        subvariant2 = subvariant1
                else:
                    subvariant1 = matsubvar[0]
                    subvariant2 = matsubvar[0]

                if mat[0] == 'VOC' or mat[0] == 'VOI' or mat[0] == 'VUM':
                    concern_list.append(mat[0])
                    variant_list.append(mat[1].replace("/", "_"))
                    clade_gisaid_list.append(mat[2])

                    subvariant_list.append(subvariant)
                    subvariant1_list.append(subvariant1)
                    subvariant2_list.append(subvariant2)
                else:
                    print(">>>", i, ">>>", mat[0], mat[1], mat[2], '\t', mat3)
                    raise Exception("stop")
            else:
                concern_list.append(unknown)
                variant_list.append(unknown)
                clade_gisaid_list.append(_clade)
                subvariant_list.append(unknown)
                subvariant1_list.append(unknown)
                subvariant2_list.append(unknown)

        '''--------------- subvariants end --------------------- '''
        clade_gisaid_list = [x.replace(" ", "_") if isinstance(x, str) else x for x in clade_gisaid_list]
        region_list = [x.strip().replace(" ", "_") if isinstance(x, str) else x for x in region_list]

        dfm['variant_gisaid'] = variant_gisaid_list
        dfm['variant']        = variant_list
        dfm['pango']          = [x if len(x.split('.')) <= 2 else ".".join( x.split('.')[:2] ) for x in pango_list]
        dfm['pango_complete'] = pango_list
        dfm['concern']        = concern_list
        dfm['subvariant']     = subvariant_list
        dfm['subvariant1']    = subvariant1_list
        dfm['subvariant2']    = subvariant2_list
        dfm['clade_gisaid']   = clade_gisaid_list

        dfm['continent'] = continents
        dfm['country'] = countries
        dfm['region'] = region_list

        dfm['year'] = year
        dfm['month'] = month
        dfm['day'] = day

        dfm = dfm[self.cols_checked_gisaid_metadata]

        dfm.loc[:, 'year']  = dfm['year'].astype(int)
        dfm.loc[:, 'month'] = dfm['month'].astype(int)
        dfm.loc[:, 'day']   = dfm['day'].astype(int)

        self.dfmeta = dfm[ (dfm.host == 'Human') & (dfm.country != 'NA') & (dfm.month != -1)].copy()

        stri = ">> filtered dfmeta - %.3f million lines"%(len(self.dfmeta)/1000000)
        log_save(stri, filename=self.filelog, verbose=True)
        ret = pdwritecsv(self.dfmeta, self.file_meta_checked, self.root_gisaid_results, verbose=True)

        self.dfanimeta = dfm[ (dfm.host != 'Human') & (dfm.country != 'NA') & (dfm.month != -1)].copy()

        stri = ">> filtered animal-env dfanimeta - %d lines"%(len(self.dfanimeta))
        log_save(stri, filename=self.filelog, verbose=True)

        ret = pdwritecsv(self.dfanimeta, self.file_meta_animal_env_checked, self.root_gisaid_results, verbose=True)

        return ret

    ''' check_gisaid_fasta_metadata_msa_uncheked deprecated
        check_animal_env_gisaid_metadata --> removed, integrated above'''
    def check_animal_env_gisaid_metadata(self, force=False, verbose=False):
        print("deprecated! use check_gisaid_metadata")
        return False


    def split_fasta_head(self, head):
        '''
         header data: defined in gisaid_lib<br>
        _id, count_read, country, year, month, coll_date, host, concern, pango, pango_complete, clade, \
        variant, subvariant, subvariant1, subvariant2, region = head.split('||')
        '''
        return head.split('||')

    ''' split_fasta_header --> split_fasta_head_fields '''
    def split_fasta_head_fields(self, stri):
        '''
        return: all fields +
        '''
        if stri[0] == '>': stri = stri[1:]
        mat = stri.split('||')
        return mat, self.fasta_fields[:len(mat)]

    def split_gisaid_big_fasta_by_cym(self, only_human=True, depth=15, large=20, start_offset=10,
                                      end_offset=100, cutoffNNN=.67, force=False, verbose=False):

        self.only_human = only_human

        if only_human:
            filelog = 'split_gisaid_big_fasta_by_cym.log'
        else:
            filelog = 'split_gisaid_big_fasta_by_cym_animal_env.log'

        self.set_filelog(filelog)

        '''--- start and end of each protein
        dfref = self.get_annotation_ref_start_end_positions(force=force, verbose=verbose)
        ---'''

        ''' --- define cuting residues
            file: 'cut_position_reference_%s_%s_y%d.tsv'%(stri_ref, self.country, self.year)
            path: root_gisaid_results

            get mseq_depth  (self.read_fast_depth_seqs(self.file_big_fasta, self.root_gisaid_results, depth=depth))
        ---'''
        _ = self.open_dfcut(depth=depth,  large=large, start_offset=start_offset,
                            end_offset=end_offset, cutoffNNN=cutoffNNN, force=force, verbose=verbose)

        if self.dfcut is None or len(self.dfcut) == 0:
            stri = "Problems with dfcut: cut_all_protein_posi_based_on_reference()"
            log_save(stri, filename=self.filelog, verbose=True)
            raise Exception("Error: stop")

        self.dfcut_ant = self.get_posi_based_on_reference_big_fasta_previous_version()
        '''--------------------- comparing structures - must be equal ---------------------'''
        if self.dfcut_ant is not None:
            ok1 = np.sum(self.dfcut.pos_cut_start - self.dfcut_ant.pos_cut_start) == 0
            if not ok1:
                stri = "Problems: change in structures between versions? 'pos_cut_start'"
                log_save(stri, filename=self.filelog, verbose=True)
                return False

            ok2 = np.sum(self.dfcut.pos_cut_end - self.dfcut_ant.pos_cut_end) == 0
            if not ok2:
                stri = "Problems: change in structures between versions? 'pos_cut_end'"
                log_save(stri, filename=self.filelog, verbose=True)
                return False

            ok3 = np.sum(self.dfcut.start - self.dfcut_ant.start) == 0
            if not ok3:
                stri = "Problems: change in structures between versions? 'start'"
                log_save(stri, filename=self.filelog, verbose=True)
                return False

            ok4 = np.sum(self.dfcut.end - self.dfcut_ant.end) == 0
            if not ok4:
                stri = "Problems: change in structures between versions? 'end'"
                log_save(stri, filename=self.filelog, verbose=True)
                return False


        if self.isGisaid:
            if not self.open_metadata_checked(only_human=only_human):
                stri = ">>Error: checked metadata was not prepared."
                log_save(stri, filename=self.filelog, verbose=True)
                return False

            if not only_human:
                self.dfmeta = self.dfanimeta
        else:
            if not self.open_ncbi_metadata():
                stri = ">>Error: human_ncbi metadata was not prepared."
                log_save(stri, filename=self.filelog, verbose=True)
                return False

        self.dfmeta.set_index('id', inplace=True)

        ret_previous = self.open_previous_version_metadata()
        if ret_previous:
            self.dfmeta_previous.set_index('id', inplace=True)

        '''
            in fast_read_save_gisaid_big_fasta_worker
            self.file_big_fasta is given by fname
        '''
        filefullfasta = os.path.join(self.root_gisaid_data, self.file_big_fasta)

        if not os.path.exists(filefullfasta):
            stri = "Could not find fasta '%s'"%(filefullfasta)
            log_save(stri, filename=self.filelog, verbose=True)
            return False

        print("\nreading lines, be patient ...")

        try:
            fh = open(filefullfasta, mode="r")
        except:
            stri = "Could not open '%s'"%(filefullfasta)
            log_save(stri, filename=self.filelog, verbose=True)
            return False

        previous_metadata_exists = True if self.dfmeta_previous else False

        self.head=''; self.seq=''; count=-1; prev_count=None
        N=50000; ret=True; doit=False; total_regs=0

        t1 = datetime.now()
        stri = t1.strftime('Start: %Y-%b-%d, %H h %M min %S sec')
        log_save(stri, filename=self.filelog, verbose=True)

        ''' split gisaid big fasta by cym '''
        while True:
            line = fh.readline()
            if not line: break

            if line[0] == '>':
                count += 1

                if count % N == 0:
                    t2 = datetime.now()
                    diff = t2-t1
                    stri = "> %d last=%.1f "%(int(count/N), diff.total_seconds())
                    log_save(stri, filename=self.filelog, verbose=True, end=' ')
                    t1 = t2

                if self.seq != '':
                    self.count = prev_count
                    total_regs += 1
                    ret = self.save_fasta_metadata_cds_protein_by_cym()


                # line = >gi_2718995740_gb_PP626409.1_Zika_virus_isolate_ZikaNIH24_complete_genome
                if line.startswith('>gi_'):
                    _id = line.split('_')[3]
                else:
                    _id = line[1:-1]
                self._id = _id

                try:
                    if previous_metadata_exists:
                        row = self.dfmeta_previous.loc[_id]
                        already_done = True if len(row) > 0 else False
                    else:
                        already_done = False
                except:
                    already_done = False

                if already_done:
                    # print("did not find '%s' - len(dfmeta)=%d\n"%(_id, len(self.dfmeta)))
                    self._id=''; self.head=''; self.seq = ''; prev_count=-1
                    continue

                try:
                    row = self.dfmeta.loc[_id]
                    doit = True if len(row) > 0 else False
                except:
                    doit = False

                if not doit:
                    stri = ">>Error: id %s not found in metadata"%(_id)
                    log_save(stri, filename=self.filelog, verbose=True)
                    self._id=''; self.head=''; self.seq = ''; prev_count=-1
                    continue

                prev_count = count

                if self.isGisaid:
                    country, year, month, coll_date, host, concern, pango, pango_complete, \
                    clade, variant, subvariant, subvariant1, subvariant2, region =  \
                    row[['country', 'year', 'month', 'coll_date', 'host', 'concern', 'pango', 'pango_complete', \
                         'clade', 'variant', 'subvariant', 'subvariant1', 'subvariant2', 'region'] ]
                else:
                    self.row = row

                    country, year, month, coll_date, host, concern,  \
                    clade, variant, subvariant, subvariant1, subvariant2, region =  \
                    row[['country', 'year', 'month', 'coll_date', 'host', 'concern', \
                         'clade', 'variant', 'subvariant', 'subvariant1', 'subvariant2', 'region'] ]

                    pango, pango_complete = None, None


                self.country = country.replace(" ", "_")
                self.year    = year
                self.month   = month

                '''--- human does not use this variable ---'''
                if only_human:
                    # dummy
                    self.animal_env = 'human'
                else:
                    self.animal_env = 'env' if host == 'Environment' else 'animal'

                '''
                _id, count_read, country, year, month, coll_date, host, concern, pango, pango_complete,\
                 clade, variant, subvariant, subvariant1, subvariant2, \
                 region = head.split('||')

                self.fasta_fields = ['id', 'count_fasta', 'country', 'year', 'month',  \
                                     'coll_date', 'host', 'concern', 'pango', 'pango_complete', 'clade', \
                                     'variant', 'subvariant', 'subvariant1', 'subvariant2', 'region']

                fasta header AND metadata tsv header:  head_fasta_metadata
                '''
                self.head_fasta_metadata = [str(x) for x in \
                        [_id, prev_count, self.country, year, month, \
                         coll_date, host, concern, pango, pango_complete, clade, \
                         variant, subvariant, subvariant1, subvariant2, region]]

                self.seq = ''

            elif doit:
                line = line.upper()
                self.seq +=  "".join([x if x in nucs else '-' for x in line[:-1]]) # big fasta DNA

        fh.close()
        if doit and ret and self.seq != '':
            self.count = prev_count
            total_regs += 1
            ret = self.save_fasta_metadata_cds_protein_by_cym()

        stri = "\n### finished, saving (%d regs)...."%(total_regs)
        log_save(stri, filename=self.filelog, verbose=True)

        ''' at the end clean lock files  -----------
        for root in [self.root_gisaid_CDS_metadata, self.root_gisaid_DNA_fasta, self.root_gisaid_CDS_fasta]:
            files = [x for x in root if x.endswith(".lock")]
            for fname in files:
                try:
                    os.unlink(os.path.join(root, fname))
                except:
                    pass
        '''
        return

    def save_fasta_metadata_cds_protein_by_cym(self, verbose=False):
        '''
            save fast:<br>
            &emsp; a) metadata (file_meta_cym)<br>
            &emsp; b) fasta (file_DNA_cym)<br>
            &emsp; c) all CDS (cut_write_CDS_cym: file_CDS_cym0)
        '''
        if self.only_human:
            file_meta = self.file_meta_cym%(self.country, self.year, self.month)
            filefull_meta = os.path.join(self.root_gisaid_CDS_metadata, file_meta)
        else:
            file_meta = self.file_meta_animal_env_cym%(self.animal_env, self.country, self.year, self.month)
            filefull_meta = os.path.join(self.root_gisaid_animal_env_metadata, file_meta)


        '''
        head_fasta_metadata =
        [_id, prev_count, self.country, year, month, \
         coll_date, host, concern, pango, pango_complete, clade, \
         variant, subvariant, subvariant1, subvariant2, region]
        '''
        if os.path.exists(filefull_meta):
            open_type = 'a+'
            stri_head_seq = "\t".join(self.head_fasta_metadata) + '\n'
        else:
            open_type = 'w'
            stri_head_seq = self.head_meta + '\n'
            stri_head_seq += "\t".join(self.head_fasta_metadata) + '\n'

        filefull_meta_lock = filefull_meta + ".lock"

        ret = True
        try:
            lock = FileLock(filefull_meta_lock)
            with lock:
                fh = open(filefull_meta, open_type)
                fh.write(stri_head_seq)
                fh.close()
        except:
            stri = "Error writing metadata and lock, line %d - id %s: '%s'"%(self.count, self._id, filefull_meta)
            log_save(stri, filename=self.filelog, verbose=True)
            ret = False
        finally:
            try:
                os.unlink(filefull_meta_lock)
            except:
                pass
                '''
                stri = "Could not remove %s"%(filefull_meta_lock)
                log_save(stri, filename=self.filelog, verbose=False)
                '''

        if not ret:
            return False

        ''' ------------------ fasta file --------------------------------------'''
        if self.only_human:
            file_DNA_cym = self.base_filename.replace('.fasta', '_y%d_m%d.fasta')%\
                           (self.virus, self.country, self.year, self.month)
            filefull_cym = os.path.join(self.root_gisaid_DNA_fasta, file_DNA_cym)
        else:
            file_DNA_cym = self.base_filename.replace('.fasta', '_%s_y%d_m%d.fasta')%\
                           (self.virus, self.country, self.animal_env, self.year, self.month)
            filefull_cym = os.path.join(self.root_gisaid_animal_env_dna, file_DNA_cym)


        self.head = ">" + "||".join(self.head_fasta_metadata)
        stri_head_seq = self.head + '\n' + self.seq + '\n'

        open_type = 'a+' if os.path.exists(filefull_cym) else 'w'
        filefull_lock = filefull_cym + ".lock"

        ret = True
        try:
            lock = FileLock(filefull_lock)
            with lock:
                fh = open(filefull_cym, open_type)
                fh.write(stri_head_seq)
                fh.close()
        except:
            stri = "Error writing fasta, line %d - id %s: '%s'"%(self.count, self._id, filefull_cym)
            log_save(stri, filename=self.filelog, verbose=True)
            ret = False
        finally:
            try:
                os.unlink(filefull_lock)
            except:
                pass
                '''
                stri = "Could not remove %s"%(filefull_lock)
                log_save(stri, filename=self.filelog, verbose=False)
                '''

        if not ret:
            return False

        ''' split in many proteins, nsps, orfs '''
        return self.cut_write_CDS_cym()

    def cut_write_CDS_cym(self):
        '''
            a) given a header and sequence (seq)<br>
            b) loop of protein in proteinList:<br>
            &emsp; b1) filtering dfcut for the protein<br>
            &emsp; b2) cut the sequence<br>
            &emsp; b3) save CDS (cym)
        '''

        for protein in self.proteinList:
            dfcut = self.dfcut[self.dfcut.protein == protein]

            if len(dfcut) != 1:
                if len(dfcut) == 0:
                    stri = "Error line %d - id %s - could not find protein %s in dfcut??? '%s'"%(self.count, self._id, protein)
                else:
                    stri = "Error line %d - id %s - dfcut is repeated for protein '%s'"%(self.count, self._id, protein)
                log_save(stri, filename=self.filelog, verbose=True)
                return False

            try:
                start  = int(dfcut.iloc[0].pos_cut_start) - 1
                end    = int(dfcut.iloc[0].pos_cut_end)

                seq_cut = self.seq[start : end]
            except:
                stri = "Error cuting sequence: line %d - id %s - cutting sequence: start %s and end % in protein '%s'"%(self.count, self._id, str(start), str(end), protein)
                log_save(stri, filename=self.filelog, verbose=True)
                return False

            '''--------------------------- CDS ---------------------------------------'''
            # print(">>> self.file_CDS_cym", self.country, protein, self.year, self.month)
            if self.only_human:
                # print("Only human", self.virus, self.country, protein, self.year, self.month)
                file_CDS_cym = self.file_CDS_cym0%(self.virus, self.country, protein, self.year, self.month)
                filefull_cym = os.path.join(self.root_gisaid_CDS_fasta, file_CDS_cym)
            else:
                # print("NOT human", self.virus, self.country, protein, self.year, self.month)
                file_CDS_cym = self.file_CDS_animal_env_cym0%(self.animal_env, self.virus, self.country, protein, self.year, self.month)
                filefull_cym = os.path.join(self.root_gisaid_animal_env_cds, file_CDS_cym)

            stri_head_seq = self.head + '\n' + seq_cut + '\n'

            open_type = 'a+' if os.path.exists(filefull_cym) else 'w'
            filefull_lock = filefull_cym + ".lock"

            ret = True
            try:
                lock = FileLock(filefull_lock)
                with lock:
                    fh = open(filefull_cym, open_type)
                    fh.write(stri_head_seq)
                    fh.close()
                
                # stri = f"Saving line {self.count} - id {self._id} - protein '{protein}': '{filefull_cym}'"
                # print(5)
                # log_save(stri, filename=self.filelog, verbose=verbose)
            except:
                stri = f"Error writing fasta, line {self.count} - id {self._id} - protein '{protein}': '{filefull_cym}'"
                log_save(stri, filename=self.filelog, verbose=True)
                ret = False
            finally:
                try:
                    os.unlink(filefull_lock)
                except:
                    stri = "Could not remove %s"%(filefull_lock)
                    log_save(stri, filename=self.filelog, verbose=False)

        return ret


    def open_prepare_codon_table(self, fname="codon_table.tsv", root="../src/"):
        dfcodons = pdreadcsv(fname, root)

        dicTrans = {}
        stop_list = []

        for i in range(len(dfcodons)):
            row = dfcodons.iloc[i]

            stri = row.codon1 + row.codon2 + row.codon3
            dicTrans[stri] = row.aa_symb2

            if row.aa_symb == "STP" :
                stop_list.append(stri)

        self.dfcodons = dfcodons
        self.dicTrans = dicTrans
        self.stop_list = stop_list
        return

    def fast_convert_CDS_to_protein(self, do_realign=False, read_fasta=False, force=False, verbose=False):
        year, month = self.year, self.month

        file_prot_cym = self.file_prot_cym%(year, month)
        filefull_prot = os.path.join(self.root_gisaid_prot_fasta, file_prot_cym)

        # print(">>> fast convert_CDS_to_protein", filefull_prot, os.path.exists(filefull_prot), read_fasta)
        if os.path.exists(filefull_prot) and not force:
            if read_fasta:
                self.mseqProt = ms.MySequence(self.prjName, root=self.root_gisaid_prot_fasta)
                ret = self.mseqProt.readFasta(filefull_prot, showmessage=verbose)
            else:
                ret = True
                self.mseqProt = None

            return ret, self.mseqProt

        file_CDS = self.file_CDS_cym%(year, month)
        filefullfasta = os.path.join(self.root_gisaid_CDS_fasta, file_CDS)

        if not os.path.exists(filefullfasta):
            # if self.country == 'Singapore': verbose = True
            stri = "fast convert_CDS_to_protein(): could not find CDS for %s %s %d/%d"%(self.country, self.protein_name, self.year, self.month)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return False, None

        '''
        if self.country == 'Singapore':
            stri = "fast convert_CDS_to_protein(): converting CDS for %s %s %d/%d"%(self.country, self.protein_name, self.year, self.month)
            log_save(stri, filename=self.filelog, verbose=True)
        '''

        if self.dicTrans is None: self.open_prepare_codon_table()

        h = open(filefullfasta, mode="r")

        if os.path.exists(filefull_prot):
            os.unlink(filefull_prot)

        ''' create an empty file '''
        f = open(filefull_prot,'w')
        f.write('')
        f.close()

        head = ''; seq = ''; count=0; ret=True
        ''' Loop fast convert_CDS_to_protein '''
        while True:
            line = h.readline()
            if not line : break

            if line[0] == '>':
                count += 1
                if count > 1:
                    ret = self.convert_CDS_to_Protein_save_fasta(head, seq, count, filefull_prot)

                head = line # with > and CR
                seq  = ''
                if count % 20000 == 0:
                    print(count, end=' ')

            else:
                seq += "".join([x if x in nucs else '-' for x in line[:-1]])

        if ret:
            ret = self.convert_CDS_to_Protein_save_fasta(head, seq, count, filefull_prot)

        self.mseqProt = ms.MySequence(self.prjName, root=self.root_gisaid_prot_fasta)

        if do_realign:
            print("------ realigning ------------")
            ret = self.mseqProt.readFasta(filefull_prot, ignore_msa=True, showmessage=False)
            ret = self.realign_seqProt_x_reference(filefull_prot, nrecs=5)
            print("---------- end ---------------")

        if read_fasta:
            ret = self.mseqProt.readFasta(filefull_prot, showmessage=verbose)
        else:
            ret = True
            self.mseqProt = None

        return ret, self.mseqProt

    def convert_codon(self, nuc3):

        try:
            aa = self.dicTrans[nuc3]
        except:
            aa = '-'
        return aa

    def translate_CDS(self, seq):
        if self.dicTrans is None: self.open_prepare_codon_table()

        ncols = len(seq)
        diff = ncols%3
        if diff != 0:
            ncols -= diff
            seq = seq[:ncols]

        aaSeq = ''

        for j in range(0, ncols, 3):
            aa = self.convert_codon(seq[j:(j+3)])
            if aa == '*':
                stri = "Found a STOP codon for protein '%s' at seq %d residue %d"%(self.protein, count, int(j/3)+1)
                log_save(stri, filename=self.filelog, verbose=False)
            aaSeq += aa

        return aaSeq


    ''' bad name: save_line_CDS_fasta --> convert_CDS_to_Protein_save_fasta '''
    def convert_CDS_to_Protein_save_fasta(self, head, seq, count, filefull_prot):
        if self.protein == 'nsp12':
            ''' ribosome slipping '''
            seq = seq[:27] + 'C' + seq[27:]

        ncols = len(seq)
        diff = ncols%3
        if diff != 0:
            ncols -= diff
            seq = seq[:ncols]

        aaSeq = ''

        for j in range(0, ncols, 3):
            aa = self.convert_codon(seq[j:(j+3)])
            if aa == '*':
                stri = "Found a STOP codon for protein '%s' at seq %d residue %d"%(self.protein, count, int(j/3)+1)
                log_save(stri, filename=self.filelog, verbose=False)
            aaSeq += aa

        stri = head + aaSeq + '\n'

        ret = True
        try:
            f = open(filefull_prot,'a+')
            f.write(stri)
        except:
            print("Error in f.write: '%s'"%(filefullfasta))
            ret =  False
        finally:
            f.close()

        return ret


    ''' numlines for testing '''
    def check_gisaid_fasta_metadata_full_fasta(self, iniLine=None, endLine=None, force=False):

        filefullmain = os.path.join(self.root_gisaid_results, self.file_meta_checked)
        filefullsumm = os.path.join(self.root_gisaid_results, self.file_meta_summary)

        if os.path.exists(filefullmain) and os.path.exists(filefullsumm) and not force:
            print("Reading dfm and dftot .... wait ....")
            dfm   = pdreadcsv(self.file_meta_checked, self.root_gisaid_results)
            dftot = pdreadcsv(self.file_meta_summary, self.root_gisaid_results)

            print("Sorting countries")
            countries = list(dftot[dftot.n >= 10].country)
            countries.sort()

            return dfm, dftot, countries

        filefullfasta = os.path.join(self.root_meta_fasta, self.file_big_fasta)
        filefulltxt   = os.path.join(self.root_meta_fasta, self.file_headers)

        if not os.path.exists(filefullfasta):
            print("Could not find fasta '%s'"%(filefullfasta))
            return None, None, None

        if not os.path.exists(filefulltxt) or force:
            print("reading lines, be patient ...")
            text = ''; count=0

            try:
                fh = open(filefullfasta, mode="r")
                ''' check_gisaid_fasta_metadata_full_fasta '''
                while True:
                    line = fh.readline()
                    if not line : break

                    if line[0] == '>':
                        text += line[1:]

                        count += 1
                        if count % 100000 == 0:
                            print(int(count/1000))

                fh.close()
                write_txt(text, self.file_headers, self.root_meta_fasta)
            except:
                print("Error: while reading '%s'"%(filefullfasta))
                return None, None, None

        headers = read_txt(self.file_headers, self.root_meta_fasta, iniLine=iniLine, endLine=endLine)
        if len(headers) == 0:
            print("Could not read '%s'"%(filefullfasta))
            return None, None, None

        dic=OrderedDict(); count = 0 # ; countRelative = 0 if iniLine is None else iniLine
        print("### Total headers = %d - df metadata = %d"%(len(headers), len(self.dfmeta)))
        for line in headers:
            # print(line)
            if count % 100000 == 0:
                print("### ",count)

            '''  example line: >hCoV-19/Australia/NT12/2020|2020|2020-04-17 '''
            mat = line[1:].split('/')

            mat[1] = mat[1].strip()
            if mat[1] in ['pangolin', 'env', 'mouse', 'mink', 'tiger', 'cat', 'dog', \
                          'lion', 'leopard', 'gorilla', 'bat', 'hamster']:
                organism = mat[1]
                if len(mat) <= 4:
                    mat[1] = 'Unfefined (species?)'
                else:
                    mat.remove(mat[1])
            else:
                organism = 'hsa'

            ''' metadata - id is here, now'''
            meta = self.dfmeta.iloc[count]

            try:
                mat3 = mat[3].split('|')
            except:
                print("mat3 without PIPE", line)
                mat3 = [mat[3]]

            dic[count] = OrderedDict()
            dic2 = dic[count]

            dic2['id'] = meta['Accession ID'].strip()
            dic2['virus']   = mat[0].strip()

            dic2['virus_check'] = meta['Virus name']
            dic2['type_virus']  = meta['Type']
            dic2['organism']    = organism

            ''' example: Oceania / Australia / Northern Territory '''
            location = meta['Location']
            locs = location.split("/")

            try:
                dic2['continent'] = locs[0].strip()
            except:
                dic2['continent'] = ''

            try:
                dic2['country'] = locs[1].strip()
            except:
                dic2['country'] = 'unknown'

            try:
                dic2['region'] = locs[2].strip()
            except:
                dic2['region'] = ''

            dic2['country2'] = mat[1].strip()

            if dic2['country'].lower() != dic2['country2'].lower() and \
               dic2['country'] not in ['United Kingdom', 'China', 'Democratic Republic of the Congo', \
                                       'Republic of the Congo', 'Brasil', 'Unfefined (species?)', \
                                       'French Polynesia', 'Burkina Faso', 'Guam', 'Crimea', 'Cameroon', \
                                       'USA', 'Hong Kong', 'South Africa', 'Czech Republic']:
                print("Country error line %d - %s - %s"%(count, dic2['country'], line) )

            try:
                dic2['state'] = mat[2].strip()
            except:
                dic2['state'] = ''

            dic2['collection_date'] = meta['Collection date']
            try:
                dic2['year'] = mat3[0]
            except:
                dic2['year'] = dic2['collection_date']

            try:
                _date = mat3[2]
                dic2['date'] = _date
                dates = _date.split('-')
                dic2['month'] = int(dates[1])
                dic2['day']   = int(dates[2])
            except:
                dic2['date']  = 'date'+str(count)
                dic2['month'] = -1
                dic2['day']   = -1

            dic2['pangolin_version'] = meta['Pangolin version']

            count += 1
            # countRelative += 1

        print("\n### finished, saving (%d regs)...."%(count))

        df = pd.DataFrame.from_dict(dic).T

        iis = df.index[df.country.isin(['ITALY'])]
        df.iloc[iis, 1] = 'Italy'

        iis = df.index[df.country.isin(['DRC'])]
        df.iloc[iis, 1] = 'Congo'

        iis = df.index[df.country.isin(['NanChang'])]
        df.iloc[iis, 1] = 'Nanchang'

        dfm = df.copy()

        iis = dfm.index[dfm.country.isin(chinas)]
        dfm.iloc[iis, 2] = df.iloc[iis].country + ' | ' + dfm.iloc[iis].state
        dfm.iloc[iis, 1] = 'China'
        dfm.iloc[iis, 8] = 'Asia'

        iis = dfm.index[dfm.country.isin(uks)]
        dfm.iloc[iis, 2] = df.iloc[iis].country + ' | ' + dfm.iloc[iis].state
        dfm.iloc[iis, 1] = 'UK'
        dfm.iloc[iis, 8] = 'Europe'

        iis = dfm.index[dfm.country.isin(brasis)]
        dfm.iloc[iis, 2] = df.iloc[iis].country + ' | ' + dfm.iloc[iis].state
        dfm.iloc[iis, 1] = 'Brazil'
        dfm.iloc[iis, 8] = 'South America'


        iis = dfm.index[dfm.country.isin(['Andalusia'])]
        dfm.iloc[iis, 2] = df.iloc[iis].country + ' | ' + dfm.iloc[iis].state
        dfm.iloc[iis, 1] = 'Spain'
        dfm.iloc[iis, 8] = 'Europe'

        dfm.state = [x.replace(' ', ' | ') for x in dfm.state]
        dfm.state = [x.replace('-', ' | ') for x in dfm.state]
        dfm.state = [x.replace('_', ' | ') for x in dfm.state]
        dfm.state = [x.replace('| | |', '|') for x in dfm.state]
        dfm.state = [x.replace('| |', '|') for x in dfm.state]
        dfm.state = [x.replace(' | ', '|') for x in dfm.state]
        dfm.state = [x.replace('VIC', 'VIC | ') for x in dfm.state]

        if iniLine is None and endLine is None:
            ret = pdwritecsv(dfm, self.file_meta_checked, self.root_gisaid_results)

        dftot = dfm.groupby('country').count().iloc[:,:2].reset_index().iloc[:,:2]
        dftot.columns = ['country', 'n']
        dftot = dftot.sort_values('n', ascending=False)

        if iniLine is None and endLine is None:
            ret = pdwritecsv(dftot, self.file_meta_summary, self.root_gisaid_results)

        countries = list(dftot[dftot.n >= 10].country)
        # print(",".join(countries[:15]))
        countries.sort()

        return dfm, dftot, countries

    def calc_diff3(self, ini, end):
        diff = end - ini + 1
        if diff % 3 == 0: return diff

        diff += diff % 3

        return diff+1


    def parse_gtf(self, filegtf, force=False):
        # filegtf = 'Sars_cov_2.ASM985889v3.100.gtf'
        # filefull = os.path.join(root_ensembl, filegtf)

        filetsv = filegtf.replace('.gtf', '.tsv')
        if not '.tsv' in filetsv:
            print("Problems in converting .gtf to .tsv (lowercase)")
            return None, None, None, None, None

        filefull = os.path.join(self.root_gisaid_results, filegtf)

        if os.path.exists(os.path.join(self.root_gisaid_results, filetsv)) and not force:
            return pdreadcsv(filetsv, self.root_gisaid_results), None, None, None, None

        try:
            lines = read_txt(filegtf, self.root_gisaid_results)
        except:
            print("Could not read '%s'"%(filefull))
            return None, None, None, None, None

        iterlines = iter(lines)
        dic = OrderedDict(); count = 0
        _build, _version, _date, _accession = None, None, None, None

        while True:
            try:
                line = next(iterlines).strip()
            except:
                break

            if count > 3:
                dic[count-4] = OrderedDict()
                dic2 = dic[count-4]

                mat = line.split('\t')

                dic2['seqname']   = mat[0]
                dic2['source']    = mat[1]
                dic2['feature']   = mat[2]
                dic2['start']     = mat[3]
                dic2['end']       = mat[4]
                dic2['score']     = mat[5]
                dic2['strand']    = mat[6]
                dic2['frame']     = mat[7]
                attribute         = mat[8]

                mat2 = [x.strip() for x in attribute.split(';')]

                for mat in mat2:
                    mtwo = mat.split(' ')
                    if mtwo == ['']: break
                    # print(mtwo)
                    field1 = mtwo[0]
                    field2 = mtwo[1].replace('"','')
                    dic2[field1] = field2

            elif count == 0:
                mat = line.split(' ')
                _build = mat[1]
            elif count == 1:
                mat = line.split(' ')
                _version = mat[1]
            elif count == 2:
                mat = line.split(' ')
                _date = mat[1]
            elif count == 3:
                mat = line.split(' ')
                _accession = mat[1]

            count += 1

        df = pd.DataFrame.from_dict(dic).T
        df.end   = df.end.astype(int)
        df.start = df.start.astype(int)

        df['length'] = [self.calc_diff3(int(df.iloc[i].start), int(df.iloc[i].end)) for i in range(df.shape[0]) ]
        df = df[ ['seqname', 'source', 'feature', 'start', 'end', 'length', 'score', 'gene_id', 'strand', 'frame', 'transcript_name', 'exon_id'] ]

        ret = pdwritecsv(df, filetsv, self.root_gisaid_results)
        if not ret:
            return None, None, None, None, None

        return df, _build, _version, _date, _accession


    def calc_fast_nCount_sample_protein_cym(self, force=False, verbose=False):

        file_prot_cym_sampled = self.file_prot_cym%(self.year, self.month)
        file_prot_cym_sampled = file_prot_cym_sampled.replace(".fasta", "_sampled_%d_nCount.fasta")%(self.nCount)

        filefull_samp = os.path.join(self.root_gisaid_prot_sampled, file_prot_cym_sampled)

        if os.path.exists(filefull_samp) and not force:
            mseq = ms.MySequence(self.prjName, root=self.root_gisaid_prot_sampled)
            ret = mseq.readFasta(filefull_samp, showmessage=verbose)
            self.mseq_protein_sampled = mseq
            return True, mseq

        file_prot_cym = self.file_prot_cym%(self.year, self.month)
        filefull_prot = os.path.join(self.root_gisaid_prot_fasta, file_prot_cym)

        if not os.path.exists(filefull_prot):
            stri = "calc fast_nCount_sample_protein_cym(): file does not exist: '%s'"%(filefull_prot)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return False, None

        dfm, sub2variants = self.variants_open_annotated_metadata()
        self.dfm = dfm

        if dfm is None or dfm.empty:
            stri = "calc fast_nCount_sample_protein_cym() loop: error calc fast_nCount_sample_protein_cym(): no dfm and sub2variants"
            log_save(stri, filename=self.filelog, verbose=verbose)
            return False, None

        try:
            ''' open the file '''
            h = open(filefull_prot, mode="r")
        except:
            stri = "calc fast_nCount_sample_protein_cym(): Could not open '%s'"%(filefull_prot)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return False, None


        dic = {}; iloop=-1; dummy_n=10000
        for variant_subvar in sub2variants:
            self.variant_subvar = variant_subvar
            variant, subvariant, pango = variant_subvar.split(' ')

            if verbose:
                if iloop == -1:
                    print("Variant:", variant_subvar, end=' ')
                else:
                    print(variant_subvar, end=' ')

            dfm2 = dfm[ [isinstance(x, str) for x in dfm.variant] ]
            dfm2 = dfm2[ (dfm2.variant == variant) & (dfm2.subvariant == subvariant) & (dfm2.pango == pango)]

            if len(dfm2) == 0:
                stri = "calc fast_nCount_sample_protein_cym() loop: there are no seqs for variant '%s'"%(variant_subvar)
                log_save(stri, filename=self.filelog, verbose=True)
                continue

            id_list = dfm2["id"].to_list()

            if len(id_list) > self.nCount:
                random.shuffle(id_list)

            # print("$$$ fast 5ab -", self.country, self.protein_name, self.year, self.month, variant_subvar, len(id_list))

            ''' ---------  loop protein ------------
                if to many '-' empty aa -- will not save in dic
                perhaps the mseqprot will not be saved and also its entropy
            '''
            previous_head = ''; seq  = ''; count=-1; iloop+=1; is_sufficient = False
            h.seek(0)

            while True:
                line = h.readline()
                if not line :  break

                if line[0] == '>':
                    line = line[:-1] # get rid carriage return
                    if previous_head != '':
                        _id = previous_head.split('||')[0].strip()

                        if _id in id_list:
                            count += 1

                            if seq != '':
                                perc = seq.count('-')/len(seq)

                                if perc >= self.badAA_cutoff_perc:
                                    count -= 1
                                    previous_head = ''
                                else:
                                    dic[count + iloop*dummy_n] = {}
                                    dic2 = dic[count + iloop*dummy_n]

                                    dic2['id'] = _id
                                    dic2['pango'] = pango
                                    dic2['variant'] = variant
                                    dic2['subvariant'] = subvariant
                                    dic2['head'] = previous_head
                                    dic2['seq'] = seq


                        ''' -------------- end ------------- '''
                        if count + 1 >= self.nCount:
                            is_sufficient = True
                            break

                    previous_head = line[1:]
                    seq  = ''
                else:
                    seq += line[:-1] # protein

            ''' ---- end - while True ----------------------'''

            if not is_sufficient and previous_head != '':
                _id = previous_head.split('||')[0].strip()

                if _id in id_list:
                    perc = seq.count('-')/len(seq)

                    if perc < self.badAA_cutoff_perc:
                        count += 1

                        dic[count + iloop*dummy_n] = {}
                        dic2 = dic[count + iloop*dummy_n]

                        dic2['id'] = _id
                        dic2['pango'] = pango
                        dic2['variant'] = variant
                        dic2['subvariant'] = subvariant
                        dic2['head'] = previous_head
                        dic2['seq'] = seq
        ''' ----- end variant loop ------------------- '''


        if len(dic) == 0:
            stri = "calc fast_nCount_sample_protein_cym(): No sequences found for %s, %s, %d/%d"%(self.country, self.protein_name, self.year, self.month)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return False, None

        df = pd.DataFrame(dic).T


        ''' droping duplicates
        df2 = df.groupby('id').count().reset_index().iloc[:,:2]
        df2.columns = ['id', 'n']
        df2 = df2[df2.n > 1].copy()
        df2.index = np.arange(len(df2))

        lista = []
        for i in range(len(df2)):
            _id = df2.iloc[i]['id']
            mat = df[df['id'] == _id].variant.to_list()
            mat.sort()
            if mat not in lista:
                lista.append(mat)

        for var_list in lista:
            df.loc[df.variant == var_list[1], ('variant')] = var_list[0]

        df = df.drop_duplicates()
        df.index = np.arange(0, len(df))
        '''

        seq_records = []

        for i in range(len(df)):
            head = df.iloc[i]['head'].strip()
            seq  = df.iloc[i]['seq']

            record = SeqRecord(
                        seq=Seq(seq),
                        id=head,
                        description=head
                    )
            seq_records.append(record)

        ''' fulfilled=False '''
        ret, mseq = self.write_sample_protein_CYM(seq_records, read_fasta=False, fulfilled=False, force=force, verbose=verbose)
        self.mseq_protein_sampled = mseq
        mseq_good = copy.deepcopy(mseq)

        ''' fix gaps - write new fasta fulfilled=True '''
        mseq = ms.MySequence(self.prjName)

        all_seq_records = []
        sub2variants = self.sub2variants_unique(df)

        for variant_subvar in sub2variants:
            self.variant_subvar = variant_subvar
            variant, subvariant, pango = variant_subvar.split(' ')

            df2 = df[(df.variant == variant) & (df.subvariant == subvariant) & (df.pango == pango)].copy()
            if df2.empty: continue

            df2.index = np.arange(0, len(df2))

            seq_records = []; seqs = []
            for i in range(len(df2)):
                head = df2.iloc[i]['head']
                seq  = df2.iloc[i]['seq']

                record = SeqRecord(
                            seq=Seq(seq),
                            id=head,
                            description=head
                        )
                seq_records.append(record)
                seqs.append(list(seq))

            mseq = ms.MySequence(self.prjName, root=self.root_gisaid_prot_sampled)
            mseq.seq_records = seq_records
            mseq.seqs = seqs

            mseq = self.fulfill_split_good_bad_seqrec(mseq=mseq, variant_subvar=variant_subvar, isProtein=True, verbose=verbose)
            if mseq is not None:
                all_seq_records += mseq.seq_records

        ''' Flavio 2023/02/10 '''
        if len(all_seq_records) == 0:
            stri = "calc_fast_nCount_sample_protein_cym(): no seq_records"
            log_save(stri, filename=self.filelog, verbose=verbose)
            return False, None

        ''' fulfilled=True '''
        ret, _ = self.write_sample_protein_CYM(all_seq_records, read_fasta=False, fulfilled=True, force=force, verbose=verbose)

        return True, mseq_good

    def write_sample_protein_CYM(self, seq_records, read_fasta=True, fulfilled=False, force=False, verbose=False):
        self.mseq_prot_sampled = None

        file_prot_cym_sampled = self.file_prot_cym%(self.year, self.month)
        file_prot_cym_sampled = file_prot_cym_sampled.replace(".fasta", "_sampled_%d_nCount.fasta")%(self.nCount)

        if fulfilled:
            file_prot_cym_sampled = file_prot_cym_sampled.replace(".fasta", "_fulfilled.fasta")

        filefull_samp = os.path.join(self.root_gisaid_prot_sampled, file_prot_cym_sampled)
        mseq = ms.MySequence(self.prjName, root=self.root_gisaid_prot_sampled)

        if os.path.exists(filefull_samp) and not force:
            if read_fasta:
                ret = mseq.readFasta(filefull_samp, showmessage=verbose)
                if ret:
                    self.mseq_prot_sampled = mseq
                    return True, mseq
            else:
                return False, None

        ret  = mseq.writeSequences(seq_records, filefull_samp, verbose=verbose)
        mseq.seqs = np.array(mseq.seqs)
        self.mseq_prot_sampled = mseq

        return ret, mseq

    def calc_fast_nCount_sample_DNA_cym(self, force=True, verbose=False):

        file_DNA_cym_sampled = self.file_DNA_cym%(self.year, self.month)
        file_DNA_cym_sampled = file_DNA_cym_sampled.replace(".fasta", "_sampled_%d_nCount.fasta")%(self.nCount)

        filefull_samp = os.path.join(self.root_gisaid_DNA_sampled, file_DNA_cym_sampled)

        if os.path.exists(filefull_samp) and not force:
            mseq = ms.MySequence(self.prjName, root=self.root_gisaid_DNA_sampled)
            ret = mseq.readFasta(filefull_samp, ignore_msa=False, showmessage=verbose)
            self.mseq_DNA_sampled = mseq
            return ret, mseq

        file_DNA_cym = self.file_DNA_cym%(self.year, self.month)
        filefull_DNA = os.path.join(self.root_gisaid_DNA_fasta, file_DNA_cym)

        if not os.path.exists(filefull_DNA):
            stri = "File does not exist: '%s'"%(filefull_DNA)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return False, None

        dfm, sub2variants = self.variants_open_annotated_metadata()

        if dfm is None or len(dfm) == 0 or len(sub2variants) == 0:
            stri = "calc fast_nCount_sample_DNA_cym(): error no dfm and sub2variants"
            log_save(stri, filename=self.filelog, verbose=verbose)
            return False, None

        if len(sub2variants) == 0:
            stri = "calc fast_nCount_sample_DNA_cym(): error there are no sub2variants after clear_variants()"
            log_save(stri, filename=self.filelog, verbose=verbose)
            return False, None

        try:
            ''' open the file '''
            h = open(filefull_DNA, mode="r")
        except:
            stri = "calc fast_nCount_sample_DNA_cym(): Could not open '%s'"%(filefull_DNA)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return False, None


        dic = {}; iloop=-1
        for variant_subvar in sub2variants:
            self.variant_subvar = variant_subvar
            variant, subvariant, pango = variant_subvar.split(' ')

            if verbose:
                if iloop == -1:
                    print("Variant:", variant_subvar, end=' ')
                else:
                    print(variant_subvar, end=' ')

            dfm2 = dfm[ [isinstance(x, str) for x in dfm.variant] ]
            dfm2 = dfm2[ (dfm2.variant == variant) & (dfm2.subvariant == subvariant) & (dfm2.pango == pango)]

            if len(dfm2) == 0:
                stri = "calc fast_nCount_sample_DNA_cym(): there are no seqs for variant '%s'"%(variant_subvar)
                log_save(stri, filename=self.filelog, verbose=True)
                continue

            id_list = dfm2["id"].to_list()

            if len(id_list) > self.nCount:
                random.shuffle(id_list)

            ''' ---------  loop DNA ---------------- '''
            head = ''; seq  = ''; count=-1; iloop+=1
            h.seek(0)
            is_sufficient = False
            while True:
                line = h.readline()
                if not line : break
                line = line[:-1] # get rid carriage return

                if line[0] == '>':
                    ''' previous head '''
                    _id = head.split('||')[0].strip()

                    if _id in id_list:
                        count += 1

                        if seq != '':
                            perc = seq.count('-')/len(seq)

                            if perc >= badNuc_cutoff_perc:
                                count -= 1
                                # print("\n>> bad", _id, seq, end=' ')
                            else:
                                # print(_id,end=' ')
                                dic[count + iloop*self.nCount] = {}
                                dic2 = dic[count + iloop*self.nCount]

                                dic2['id'] = _id
                                dic2['pango'] = pango
                                dic2['variant'] = variant
                                dic2['subvariant'] = subvariant
                                dic2['head'] = head
                                dic2['seq'] = seq

                    ''' -------------- end ------------- '''
                    if count + 1 >= self.nCount:
                        is_sufficient = True
                        break

                    head = line[1:]
                    seq  = ''
                else:
                    seq += "".join([x if x in nucs else '-' for x in line]) # DNA

            if not is_sufficient and _id in id_list:
                perc = seq.count('-')/len(seq)

                if perc < badNuc_cutoff_perc:
                    count += 1

                    dic[count + iloop*self.nCount] = {}
                    dic2 = dic[count + iloop*self.nCount]

                    dic2['id'] = _id
                    dic2['pango'] = pango
                    dic2['variant'] = variant
                    dic2['subvariant'] = subvariant
                    dic2['head'] = head
                    dic2['seq'] = seq
        ''' ----- end variant loop ------------------- '''

        if len(dic) == 0:
            stri = "calc fast_nCount_sample_DNA_cym(): No sequences found for %s, %d/%d"%(country, year, month)
            log_save(stri, filename=self.filelog, verbose=verbose, end="  ")
            return False, None

        df = pd.DataFrame(dic).T

        ''' droping duplicates

        df2 = df.groupby('id').count().reset_index().iloc[:,:2]
        df2.columns = ['id', 'n']
        df2 = df2[df2.n > 1].copy()
        df2.index = np.arange(len(df2))

        lista = []
        for i in range(len(df2)):
            _id = df2.iloc[i]['id']
            mat = df[df['id'] == _id].variant.to_list()
            mat.sort()
            if mat not in lista:
                lista.append(mat)

        for var_list in lista:
            df.loc[df.variant == var_list[1], ('variant')] = var_list[0]

        df = df.drop_duplicates()
        df.index = np.arange(0, len(df))
        '''

        seq_records = []

        for i in range(len(df)):
            head = df.iloc[i]['head']
            seq  = df.iloc[i]['seq']

            record = SeqRecord(
                        seq=Seq(seq),
                        id=head,
                        description=head
                    )
            seq_records.append(record)

        ret, mseq = self.write_sample_DNA_cym(seq_records, read_fasta=False, fulfilled=False, force=force, verbose=verbose)
        self.mseq_DNA_sampled = mseq
        mseq_good = copy.deepcopy(mseq)

        ''' fix gaps - write new fasta fulfilled=True '''
        mseq = ms.MySequence(self.prjName)

        all_seq_records = []
        sub2variants = self.sub2variants_unique(df)

        ''' fulfill_polymorphic per variant '''
        for variant_subvar in sub2variants:
            self.variant_subvar = variant_subvar
            variant, subvariant, pango = variant_subvar.split(' ')

            df2 = df[(df.variant == variant) & (df.subvariant == subvariant) & (df.pango == pango)].copy()
            if len(df2) == 0:
                continue

            df2.index = np.arange(0, len(df2))

            seq_records = []; seqs = []
            for i in range(len(df2)):
                head = df2.iloc[i]['head']
                seq  = df2.iloc[i]['seq']

                record = SeqRecord(
                            seq=Seq(seq),
                            id=head,
                            description=head
                        )
                seq_records.append(record)
                seqs.append(list(seq))

            mseq = ms.MySequence(self.prjName, root=self.root_gisaid_DNA_sampled)
            mseq.seq_records = seq_records
            mseq.seqs = seqs

            mseq = self.fulfill_split_good_bad_seqrec(mseq, variant_subvar, isProtein=False, verbose=verbose)
            if mseq is not None:
                all_seq_records += mseq.seq_records

        ''' Flavio 2023/02/10 '''
        if len(all_seq_records) == 0:
            stri = "calc_fast_nCount_sample_DNA_cym(): no seq_records"
            log_save(stri, filename=self.filelog, verbose=verbose)
            return False, None

        ret, _ = self.write_sample_DNA_cym(all_seq_records, read_fasta=False, fulfilled=True, force=force, verbose=verbose)

        return True, mseq_good


    def write_sample_DNA_cym(self, seq_records, read_fasta=True, fulfilled=False, force=False, verbose=False):

        file_DNA_cym_sampled = self.file_DNA_cym%(self.year, self.month)
        file_DNA_cym_sampled = file_DNA_cym_sampled.replace(".fasta", "_sampled_%d_nCount.fasta")%(self.nCount)

        if fulfilled:
            file_DNA_cym_sampled = file_DNA_cym_sampled.replace(".fasta", "_fulfilled.fasta")

        filefull_samp = os.path.join(self.root_gisaid_DNA_sampled, file_DNA_cym_sampled)
        mseq = ms.MySequence(self.prjName, root=self.root_gisaid_DNA_sampled)

        if os.path.exists(filefull_samp) and not force:
            if read_fasta:
                ret = mseq.readFasta(filefull_samp, showmessage=verbose)
                return ret, mseq
            else:
                return False, None

        ret  = mseq.writeSequences(seq_records, filefull_samp, verbose=verbose)
        mseq.seqs = np.array(mseq.seqs)

        return ret, mseq

    def calc_fast_nCount_sample_CDS_cym(self, force=True, verbose=False):

        file_CDS_cym_sampled = self.file_CDS_cym%(self.year, self.month)
        file_CDS_cym_sampled = file_CDS_cym_sampled.replace(".fasta", "_sampled_%d_nCount.fasta")%(self.nCount)

        filefull_samp = os.path.join(self.root_gisaid_CDS_sampled, file_CDS_cym_sampled)

        if os.path.exists(filefull_samp) and not force:
            mseq = ms.MySequence(self.prjName, root=self.root_gisaid_CDS_sampled)
            ret = mseq.readFasta(filefull_samp, ignore_msa=False, showmessage=verbose)
            self.mseq_CDS_sampled = mseq
            return ret, mseq

        file_CDS_cym = self.file_CDS_cym%(self.year, self.month)
        filefull_CDS = os.path.join(self.root_gisaid_CDS_fasta, file_CDS_cym)

        if not os.path.exists(filefull_CDS):
            stri = "File does not exist: '%s'"%(filefull_CDS)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return False, None

        dfm, sub2variants = self.variants_open_annotated_metadata()
        self.dfm = dfm

        if dfm is None or len(dfm) == 0 or len(sub2variants) == 0:
            stri = "calc_fast_nCount_sample_CDS_cym error: no dfm and sub2variants"
            log_save(stri, filename=self.filelog, verbose=verbose)
            return False, None

        try:
            ''' open the file '''
            h = open(filefull_CDS, mode="r")
        except:
            stri = "Could not open '%s'"%(filefull_CDS)
            log_save(stri, filename=self.filelog, verbose=verbose)
            return False, None


        dic = {}; iloop=-1
        for variant_subvar in sub2variants:
            self.variant_subvar = variant_subvar
            variant, subvariant, pango = variant_subvar.split(' ')

            if verbose:
                if iloop == -1:
                    print("Variant:", variant_subvar, end=' ')
                else:
                    print(variant_subvar, end=' ')

            dfm2 = dfm[ [isinstance(x, str) for x in dfm.variant] ]
            dfm2 = dfm2[ (dfm2.variant == variant) & (dfm2.subvariant == subvariant) & (dfm2.pango == pango)]

            if len(dfm2) == 0:
                stri = "calc fast_nCount_sample_CDS_cym: there are no seqs for variant '%s %s %s'"%(variant, subvariant, pango)
                log_save(stri, filename=self.filelog, verbose=True)
                continue

            id_list = dfm2["id"].to_list()
            if len(id_list) > self.nCount:
                random.shuffle(id_list)

            ''' ---------  loop CDS ---------------- '''
            head = ''; seq  = ''; count=-1; iloop+=1
            h.seek(0)
            is_sufficient = False
            while True:
                line = h.readline()
                if not line : break
                line = line[:-1] # get rid carriage return

                if line[0] == '>':
                    ''' previous head '''
                    _id = head.split('||')[0].strip()

                    if _id in id_list:
                        count += 1

                        ''' previous seq '''
                        if seq != '':
                            perc = seq.count('-')/len(seq)

                            if perc >= badNuc_cutoff_perc:
                                count -= 1
                                # print("\n>> bad", _id, seq, end=' ')
                            else:
                                # print(_id,end=' ')
                                dic[count + iloop*self.nCount] = {}
                                dic2 = dic[count + iloop*self.nCount]

                                dic2['id'] = _id
                                dic2['pango'] = pango
                                dic2['variant'] = variant
                                dic2['subvariant'] = subvariant
                                dic2['head'] = head
                                dic2['seq'] = seq

                    ''' -------------- end ------------- '''
                    if count + 1 >= self.nCount:
                        is_sufficient = True
                        break

                    head = line[1:] # CDS
                    seq  = ''
                else:
                    seq += "".join([x if x in nucs else '-' for x in line])  # CDS

            if not is_sufficient and _id in id_list:
                perc = seq.count('-')/len(seq)

                if perc < badNuc_cutoff_perc:
                    count += 1

                    dic[count + iloop*self.nCount] = {}
                    dic2 = dic[count + iloop*self.nCount]

                    dic2['id'] = _id
                    dic2['pango'] = pango
                    dic2['variant'] = variant
                    dic2['subvariant'] = subvariant
                    dic2['head'] = head
                    dic2['seq'] = seq
        ''' ----- end variant loop ------------------- '''

        if len(dic) == 0:
            stri = "calc_fast_nCount_sample_CDS_cym(): No sequences found for %s, %s, %d/%d"%(self.country, self.protein_name, self.year, self.month)
            log_save(stri, filename=self.filelog, verbose=False, end="  ")
            return False, None

        df = pd.DataFrame(dic).T

        ''' droping duplicates

        df2 = df.groupby('id').count().reset_index().iloc[:,:2]
        df2.columns = ['id', 'n']
        df2 = df2[df2.n > 1].copy()
        df2.index = np.arange(len(df2))

        lista = []
        for i in range(len(df2)):
            _id = df2.iloc[i]['id']
            mat = df[df['id'] == _id].variant.to_list()
            mat.sort()
            if mat not in lista:
                lista.append(mat)

        for var_list in lista:
            df.loc[df.variant == var_list[1], ('variant')] = var_list[0]

        df = df.drop_duplicates()
        df.index = np.arange(0, len(df))
        '''

        seq_records = []

        for i in range(len(df)):
            head = df.iloc[i]['head']
            seq  = df.iloc[i]['seq']

            record = SeqRecord(
                        seq=Seq(seq),
                        id=head,
                        description=head
                    )
            seq_records.append(record)

        ret, mseq = self.write_sample_CDS_cym(seq_records, read_fasta=False, fulfilled=False, force=force, verbose=verbose)
        mseq_good = copy.deepcopy(mseq)
        self.mseq_CDS_sampled = mseq

        ''' fix gaps - write new fasta fulfilled=True '''
        mseq = ms.MySequence(self.prjName)

        all_seq_records = []
        sub2variants = self.sub2variants_unique(df)

        ''' fulfill_polymorphic per variant '''
        for variant_subvar in sub2variants:
            self.variant_subvar = variant_subvar
            variant, subvariant, pango = variant_subvar.split(' ')

            df2 = df[(df.variant == variant) & (df.subvariant == subvariant) & (df.pango == pango)].copy()
            if len(df2) == 0:
                continue

            df2.index = np.arange(0, len(df2))

            seq_records = []; seqs = []
            for i in range(len(df2)):
                head = df2.iloc[i]['head']
                seq  = df2.iloc[i]['seq']

                record = SeqRecord(
                            seq=Seq(seq),
                            id=head,
                            description=head
                        )
                seq_records.append(record)
                seqs.append(list(seq))

            mseq = ms.MySequence(self.prjName, root=self.root_gisaid_CDS_sampled)
            mseq.seq_records = seq_records
            mseq.seqs = seqs

            mseq = self.fulfill_split_good_bad_seqrec(mseq, variant_subvar, isProtein=False, verbose=verbose)
            if mseq is not None:
                all_seq_records += mseq.seq_records

        ''' Flavio 2023/02/10 '''
        if len(all_seq_records) == 0:
            stri = "calc_fast_nCount_sample_CDS_cym(): no seq_records"
            log_save(stri, filename=self.filelog, verbose=verbose)
            return False, None

        ret, _ = self.write_sample_CDS_cym(all_seq_records, read_fasta=False, fulfilled=True, force=force, verbose=verbose)

        return True, mseq_good

    def write_sample_CDS_cym(self, seq_records, read_fasta=True, fulfilled=False, force=False, verbose=False):
        file_CDS_cym_sampled = self.file_CDS_cym%(self.year, self.month)
        file_CDS_cym_sampled = file_CDS_cym_sampled.replace(".fasta", "_sampled_%d_nCount.fasta")%(self.nCount)

        if fulfilled:
            file_CDS_cym_sampled = file_CDS_cym_sampled.replace(".fasta", "_fulfilled.fasta")

        filefull_samp = os.path.join(self.root_gisaid_CDS_sampled, file_CDS_cym_sampled)
        mseq = ms.MySequence(self.prjName, root=self.root_gisaid_CDS_sampled)

        if os.path.exists(filefull_samp) and not force:
            if read_fasta:
                ret = mseq.readFasta(filefull_samp, showmessage=verbose)
                return ret, mseq
            else:
                return False, None

        ret  = mseq.writeSequences(seq_records, filefull_samp, verbose=verbose)
        mseq.seqs = np.array(mseq.seqs)

        return ret, mseq


    def make_multiple_alignments(self, protAligns, seaviewPath = "../../tools/seaview/", verbose=True):

        mseqProt = ms.MySequence(self.prjName, root=self.root_gisaid_prot_fasta)

        for protein in protAligns:
            self.init_vars(country, protein)

            file_merged_fasta  = self.base_filename.replace('.fasta', '_merged_y%d_%s.fasta')%(self.virus, self.country, year, protein)
            file_aligned_fasta = file_merged_fasta.replace('_merged_', '_aligned_')
            filefull_prot = os.path.join(elf.root_gisaid_prot_fasta, file_merged_fasta)
            filealigfull = os.path.join(elf.root_gisaid_prot_fasta, file_aligned_fasta)

            if verbose: print("Aligning:", filefull_prot)
            ret = mseqProt.align_seqs_clustalo(filefull_prot, filealigfull, seaviewPath, force=True, server=False)
            if verbose: print(">>> end", protein, ret, '\n')


    def get_DNA_cym(self, country, year, month, verbose=False):
        self.mseq_DNA = None

        self.init_vars(country)
        self.year = year
        self.month = month

        # alredy did init_vas()
        filefasta  = self.file_DNA_cym%(year, month)
        filefasta_full = os.path.join(self.root_gisaid_DNA_fasta, filefasta)

        if not os.path.exists(filefasta_full):
            print("Could not find '%s'"%(filefasta_full))
            return False

        self.mseq_DNA = ms.MySequence(self.prjName, root=self.root_gisaid_DNA_fasta)
        ret = self.mseq_DNA.readFasta(filefasta_full, showmessage=verbose)

        return ret

    def read_fast_depth_seqs(self, filename, root_fasta, depth):

        filefullfasta = os.path.join(root_fasta, filename)

        if not os.path.exists(filefullfasta):
            print("Could not find fasta '%s'"%(filefullfasta))
            return None

        h = open(filefullfasta, mode="r")
        text = ''; count=0; row=None; ret=True

        while True:
            line = h.readline()
            if not line : break

            if line[0] == '>':
                count += 1; seq = ''
                if count == depth: break

                text += line
                if count % 1000 == 0:
                    print("-->", count, end='')

            else:
                text += line.upper()

        filename2 = filename.replace(".fasta", "_depth_aux.fasta")
        ret = write_txt(text, filename2, root_fasta)
        if not ret:
            return None

        mseq = ms.MySequence(self.prjName, root=self.root_gisaid_ncbi)
        filefullfasta = os.path.join(root_fasta, filename2)
        mseq.readFasta(filefullfasta, ignore_msa=True)

        return mseq

    ''' get_gisaid_cut_positions --> calc msa_to_CDS
        fileReffasta = 'HA_3655151.fasta'

        printPlot --> calc_posi can plot a %graph
    '''
    def get_posi_based_on_reference_big_fasta_previous_version(self):

        if self.file_cut_position_previous_version is None:
            return None

        filefasta_full = os.path.join(self.root_gisaid_data, self.file_cut_position_previous_version)

        if not os.path.exists(filefasta_full):
            return None

        return pdreadcsv(self.file_cut_position_previous_version, self.root_gisaid_data, verbose=False)

    '''
        calc_cut_all_proteins_posi_based_on_reference_any_fasta -->
        calc_dfcut_all_proteins_posi_based_on_reference_big_fasta

        Flavio (2024/07/17) -> self.dicNucAnnot
    '''
    def calc_dfcut_all_proteins_posi_based_on_reference_big_fasta(self, depth:int=15, large:int=20,
                                                     start_offset:int=10, end_offset:int=100, cutoffNNN:float=.67,
                                                     force:bool=False, verbose:bool=False, printPlot:bool=False) -> pd.DataFrame:

        filefull = os.path.join(self.root_gisaid_data, self.file_cut_position)
        # if verbose: print(">>> calc_dfcut", filefull, os.path.exists(filefull))

        if os.path.exists(filefull) and not force:
            return pdreadcsv(self.file_cut_position, self.root_gisaid_data, verbose=verbose)

        dfref = self.get_annotation_ref_start_end_positions(force=False, verbose=verbose)

        mseq_depth = self.read_fast_depth_seqs(self.file_big_fasta, self.root_gisaid_data, depth=depth)
        if mseq_depth is None:
            return None

        dic = {}; count = 0
        for protein in self.proteinList:
            dfq = dfref[dfref.protein == protein]

            if len(dfq) == 0:
                print("###Error: %s is not in dfref !"%(protein))
                return None

            is_start = True
            row = dfq.iloc[0]
            seq_reference0 = row.seq_start
            start  = row.start-1
            start0 = start
            end    = start + len(seq_reference0)

            pos_begin = mseq_depth.calc_posi(protein, seq_reference0, is_start, start, end,
                                             large=large, start_offset=start_offset, end_offset=end_offset,
                                             printPlot=printPlot, verbose=verbose)

            pos_begin = int(pos_begin)

            is_start = False
            seq_reference1 = row.seq_end
            end   = row.end
            end0  = end
            start = end - len(seq_reference1)

            pos_end = mseq_depth.calc_posi(protein, seq_reference1, is_start, start, end,
                                           large=large, start_offset=start_offset, end_offset=end_offset,
                                           printPlot=printPlot, verbose=verbose)
            pos_end = int(pos_end)

            dic[count] = {}
            dic2 = dic[count]
            dic2['protein']       = protein
            dic2['pos_cut_start'] = pos_begin+1
            dic2['pos_cut_end']   = pos_end+1
            dic2['start']     = start0
            dic2['end']       = end0
            dic2['seq_start'] = seq_reference0
            dic2['seq_end']   = seq_reference1

            count += 1

        dfcut = pd.DataFrame(dic).T
        ret = pdwritecsv(dfcut, self.file_cut_position, self.root_gisaid_data, verbose=verbose)

        return dfcut

    ''' fileReffasta = 'MW400961.fasta'   - Wuhan
        fileReffasta = 'MT789690.1.fasta' - Turkey

        for Zika: 'NC_035889.1_nucleotide.fasta'
    '''
    def read_fasta_reference(self, verbose=False):

        mseqRef = ms.MySequence(self.prjName, root=self.root_gisaid_ncbi)
        filefull = os.path.join(self.root_gisaid_ncbi, self.fileReffasta)
        ret = mseqRef.readFasta(filefull, showmessage=True)

        return mseqRef


    def get_annotation_ref_start_end_positions(self, force:bool=False, verbose:bool=False) -> pd.DataFrame:
        self.dfref = None

        stri_ref = self.fileReffasta.replace(".fasta", "").replace('.', '_')
        fname    = 'reference_%s.tsv'%(stri_ref)
        filefull = os.path.join(self.root_gisaid_ncbi, fname)

        if os.path.exists(filefull) and not force:
            dfref = pdreadcsv(fname, self.root_gisaid_ncbi, verbose=verbose)
            self.dfref = dfref
            return dfref

        print("Start get_annotation_ref_start_end_positions ....")
        print("dicNucAnnot must be compatible with fileReffasta (%s)"%(filefull))

        mseqRef = self.read_fasta_reference(verbose=verbose)
        if len(mseqRef.seq_records) == 0:
            print("Could not read reference table '%s'"%(filefull))
            return None

        refseq = str(mseqRef.seq_records[0].seq)

        dic = {}; count = 0
        for prot, mat in self.dicNucAnnot.items():
            ini, end, seq = mat

            dna = refseq[(ini-1):end]
            if len(dna) > 60:
                seq_start = dna[:60]
                seq_end   = dna[-60:]
            else:
                seq_start = dna
                seq_end   = None

            if verbose:
                print(prot, '\t', mat[0], '\t', mat[1], '\tnuc:', mat[2])

                if seq_end is None:
                    print('\t', seq_start,'\t', str(seq_start) in mat[2], '\n')
                else:
                    print('\tseq_start:\t', seq_start,'\t', str(seq_start) in mat[2])
                    print('\tseq_end:\t', seq_end,'\t', '\n')

            dic[count] = {}
            dic2 = dic[count]
            dic2['protein'] = prot
            dic2['start'] = ini
            dic2['end'] = end
            dic2['nuc_length'] = end - ini + 1
            dic2['prot_length'] = dic2['nuc_length'] / 3
            dic2['seq_start'] = seq_start
            dic2['seq_end']   = seq_end

            count += 1

        dfref = pd.DataFrame(dic).T
        self.dfref = dfref
        ret = pdwritecsv(dfref, fname, self.root_gisaid_ncbi)

        return dfref

    def tree_subsample_to_create_tree_country_year(self, country, year, protein, protein_name, months=np.arange(1,13), verbose=False):
        seq_records = []

        for month in months:

            self.init_vars_year_month(country, protein, protein_name, year, month)

            file_meta_country = self.file_meta_cy0%(self.country)
            file_meta_country = file_meta_country.replace(".tsv", '_y%d_m%d.tsv')%(self.year, self.month)

            dfmeta = pdreadcsv(file_meta_country, self.root_gisaid_CDS_metadata)

            ret = self.subsample_to_create_tree(dfmeta, verbose=False)

            if ret:
                seq_records += list(self.init_vars_year_monthmseq_sampled.seq_records)
            else:
                print("Month %d not found"%(month))
                continue

        self.seq_records = seq_records

        if len(seq_records) == 0:
            print("No data found")
            return False

        if list(months) != list(np.arange(1,13)):
            print("Only a complete year will be saved.")
            return True

        file_all_CDS_cy_sampled = self.file_CDS_cym.replace("_m%d","_all_year")%(self.year)
        file_all_CDS_cy_sampled = file_all_CDS_cy_sampled.replace(".fasta", "_sampled_%d_nCount.fasta")%(self.nCount)

        filefull_sel = os.path.join(self.root_gisaid_CDS_sampled, file_all_CDS_cy_sampled)

        ret = self.mseq_sampled.writeSequences(seq_records, filefull_sel, verbose=True)
        return ret

    def fulfill_split_good_bad_seqrec(self, mseq, variant_subvar:str, isProtein:bool=True, min_seqs:int=3, verbose:bool=False):

        n = len(mseq.seqs)
        L = len(mseq.seqs[0])

        if n < min_seqs or L == 0:

            if variant_subvar is None:
                variant_subvar = 'Variant???'

            stri = "fulfill split_good_bad_seqrec(): mseq insuficient for %s %s variant %s %d %d - nrows=%d and ncols=%d"%\
                   (self.country, self.protein_name, variant_subvar, self.year, self.month, n, L)
            log_save(stri, filename=self.filelog, verbose=False)
            return mseq

        ''' sequences should not have more than 25% of gaps, ohterwiser, a bad sequence '''
        good_list = [True if list(mseq.seqs[i]).count('-') / L <= self.cutoff_good else False for i in range(len(mseq.seqs))]

        ''' if equal size or to little '''
        if np.sum(good_list) == n or np.sum(good_list) < min_seqs:
            seqs = mseq.fulfill_polymorphic_seqs_with_gaps(mseq.seqs, self.min_gaps, isProtein=True, min_seqs=min_seqs)
        else:

            seqs_good = [mseq.seqs[i] for i in range(len(good_list)) if     good_list[i]]
            seqs_bad  = [mseq.seqs[i] for i in range(len(good_list)) if not good_list[i]]

            seqs1 = mseq.fulfill_polymorphic_seqs_with_gaps(seqs_good, self.min_gaps, isProtein=True, min_seqs=min_seqs)
            # self.seqs1 = seqs1
            if isProtein:
                mseq2 = ms.MySequence(self.prjName, root=self.root_gisaid_prot_fasta)
                mseq2.seqs = seqs1
                # self.mseq2 = mseq2
                seq_consensus = mseq2.get_amino_acid_consensus()
            else:
                mseq2 = ms.MySequence(self.prjName, root=self.root_gisaid_DNA_fasta)
                mseq2.seqs = seqs1
                # self.mseq2 = mseq2
                seq_consensus = mseq2.get_nucleotide_consensus()

            self.seq_consensus = seq_consensus
            seqs2 = mseq.fulfill_seqs_with_consensus(seqs_bad, seq_consensus)
            # add good and bad records !
            seqs = list(seqs1) + list(seqs2)

        mseq.seqs = seqs
        for i in range(len(mseq.seq_records)):
            mseq.seq_records[i].seq = Seq("".join(seqs[i]))

        return mseq
