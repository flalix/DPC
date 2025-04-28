#!/usr/bin/python
#!python
# -*- coding: utf-8 -*-

''' http://blog.notdot.net/2010/07/Getting-unicode-right-in-Python
'''
from __future__ import unicode_literals
from typing import List

import requests, sys, os, xmltodict, re, pickle
from collections import OrderedDict

try:
    from urllib import urlencode
except ImportError:
    from urllib.parse import urlencode

import httplib2 as http
import json
try:
    from urlparse import urlparse
except:
    from urllib.parse import urlparse

from Bio import Entrez
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# write and read function + Basic (bioinformatics)
from Basic import *


class Entrez_conversions(object):
    def __init__(self, email, root_colab = "../../colaboracoes/", fileBadUniprotSymbs = "bad_uniprot_symbols.txt"):
        """Must Include Email"""
        self.params = {}
        self.email = email
        self.params['tool'] = 'PyEntrez'
        if re.match(r"[^@]+@[^@]+\.[^@]+", email):
            pass
        else:
            raise ValueError("Enter a valid Email Address")
        self.params["email"] = email
        self.entrez_email = urlencode(self.params, doseq=True)
        Entrez.email = email


        self.root_colab = root_colab
        self.root_refseq = create_dir(root_colab, 'refseq')
        self.root_hgnc   = create_dir(root_colab, 'hgnc')
        self.root_hgnc   = create_dir(root_colab, 'kegg')

        self.fname_conv = "conversion_symbol_uniprot_entrez.tsv"

        self.fileBadUniprotSymbs = fileBadUniprotSymbs

        filefull = os.path.join(self.root_refseq, fileBadUniprotSymbs)
        if os.path.exists(filefull):
            self.badunis = read_txt(fileBadUniprotSymbs, self.root_refseq)
        else:
            self.badunis = []

        self.fname_hgnc_set = 'hgnc_complete_set_202312.tsv'
        self.fname_ncbi_syn = 'refseq_synonyms_ncbi.tsv'


    def convert_ensembl_to_entrez(self, ensembl):
        """Convert Ensembl Id to Entrez Gene Id"""

        """
        if 'ENST' in ensembl:
            pass
        else:
            raise (IndexError)
        """
        # Submit resquest to NCBI eutils/Gene database
        server = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + self.entrez_email + "&db=gene&term={0}".format(
            ensembl)
        r = requests.get(server, headers={"Content-Type": "text/xml"})
        if not r.ok:
            r.raise_for_status()
            sys.exit()
        # Process Request
        response = r.text
        info = xmltodict.parse(response)
        try:
            geneId = info['eSearchResult']['IdList']['Id']
        except TypeError:
            raise (TypeError)
        return geneId


    def convert_hgnc_to_entrez(self, hgnc):
        """Convert HGNC Id to Entrez Gene Id"""
        entrezdict = {}
        server = "http://rest.genenames.org/fetch/hgnc_id/{0}".format(hgnc)
        r = requests.get(server, headers={"Content-Type": "application/json"})
        if not r.ok:
            r.raise_for_status()
            sys.exit()
        response = r.text
        info = xmltodict.parse(response)
        for data in info['response']['result']['doc']['str']:
            if data['@name'] == 'entrez_id':
                entrezdict[data['@name']] = data['#text']
            if data['@name'] == 'symbol':
                entrezdict[data['@name']] = data['#text']
        return entrezdict

    def convert_entrez_to_uniprot(self, entrez):
        """Convert Entrez Id to Uniprot Id"""
        server = f"http://www.uniprot.org/uniprot/?query=%22GENEID+{entrez}%22&format=xml"
        r = requests.get(server, headers={"Content-Type": "text/xml"})
        if not r.ok:
            r.raise_for_status()
            sys.exit()
        response = r.text
        info = xmltodict.parse(response)
        try:
            data = info['uniprot']['entry']['accession'][0]
            return data
        except TypeError:
            data = info['uniprot']['entry'][0]['accession'][0]
            return data

    def convert_uniprot_to_entrez(self, uniprot):
        """Convert Uniprot Id to Entrez Id"""
        # Submit request to NCBI eutils/Gene Database
        server = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + self.entrez_email + "&db=gene&term={0}".format(
            uniprot)
        r = requests.get(server, headers={"Content-Type": "text/xml"})
        if not r.ok:
            r.raise_for_status()
            sys.exit()
        # Process Request
        response = r.text
        try:
            info = xmltodict.parse(response)
            geneId = info['eSearchResult']['IdList']['Id']

        except:
            print("Input is uniprot accession number")
            return None

        try:
            if isinstance(geneId, str): return int(geneId)

            # check to see if more than one result is returned
            # if you have more than more result then check which Entrez Id returns the same uniprot Id entered.
            if isinstance(geneId, list):
                geneId = [int(geneId) for x in geneId]
        except:
            pass

        return geneId

    def convert_accession_to_taxid(self, accessionid):
        """Convert Accession Id to Tax Id """
        # Submit request to NCBI eutils/Taxonomy Database
        URL = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
        server = URL + self.entrez_email + "&db=nuccore&id={0}&retmode=xml".format(accessionid)
        r = requests.get(server, headers={"Content-Type": "text/xml"})
        if not r.ok:
            r.raise_for_status()
            sys.exit()
        # Process Request
        response = r.text
        records = xmltodict.parse(response)
        try:
            for i in records['GBSet']['GBSeq']['GBSeq_feature-table']['GBFeature']['GBFeature_quals']['GBQualifier']:
                for key, value in i.items():
                    if value == 'db_xref':
                        taxid = i['GBQualifier_value']
                        taxid = taxid.split(':')[1]
                        return taxid
        except:
            for i in records['GBSet']['GBSeq']['GBSeq_feature-table']['GBFeature'][0]['GBFeature_quals']['GBQualifier']:
                for key, value in i.items():
                    if value == 'db_xref':
                        taxid = i['GBQualifier_value']
                        taxid = taxid.split(':')[1]
                        return taxid
        return

    def convert_symbol_to_entrezid(self, symbol):
        """Convert Symbol to Entrez Gene Id"""
        entrezdict = {}
        server = "http://rest.genenames.org/fetch/symbol/{0}".format(symbol)
        r = requests.get(server, headers={"Content-Type": "application/json"})
        if not r.ok:
            r.raise_for_status()
            sys.exit()
        response = r.text
        info = xmltodict.parse(response)
        for data in info['response']['result']['doc']['str']:
            if data['@name'] == 'entrez_id':
                entrezdict[data['@name']] = data['#text']
            if data['@name'] == 'symbol':
                entrezdict[data['@name']] = data['#text']
        return entrezdict


    def get_ensembl_data(self, ensembl, verbose = False):
        geneid = self.rest_ensembl_to_entrez(ensembl)

        if geneid is None:
            return None

        annotations = self.retrieve_annotation([geneid], verbose = verbose)

        return self.get_annotation(ensembl, geneid, annotations,  to_print = verbose)


    # https://biopython.org/wiki/Annotate_Entrez_Gene_IDs
    # Retrieve and annotate Entrez Gene IDS with the Entrez module
    def retrieve_annotation(self, id_list, verbose = False):

        """Annotates Entrez Gene IDs using Bio.Entrez, in particular epost (to
        submit the data to NCBI) and esummary to retrieve the information.
        Returns a list of dictionaries with the annotations."""

        request = Entrez.epost("gene",id=",".join(id_list))
        try:
            result = Entrez.read(request)
        except RuntimeError as e:
            #FIXME: How generate NAs instead of causing an error with invalid IDs?
            print("An error occurred while retrieving the annotations.")
            print("The error returned was %s" % e)
            return None

        webEnv = result["WebEnv"]
        queryKey = result["QueryKey"]
        data = Entrez.esummary(db="gene", webenv=webEnv, query_key = queryKey)
        annotations = Entrez.read(data)

        if verbose: print("Retrieved %d annotations for %d genes" % (len(annotations), len(id_list)))

        return annotations


    def rest_ensembl_to_entrez(self, ensembl):
        # Always tell NCBI who you are
        try:
            handle = Entrez.esearch(db="gene", term=ensembl)
            record = Entrez.read(handle)
            handle.close()
        except:
            write_error("Could not find %s in Entrez.esearch.", fileLog, to_append=True)
            handle.close()
            return None

        if len(record['IdList']) == 0:
            return None

        return record['IdList'][0]

    def get_annotation(self, EnsemblId, geneId, annotations,  to_print = False):

        dic = annotations['DocumentSummarySet']['DocumentSummary'][0]

        dic['Ensembl'] = EnsemblId
        dic['Entrezid'] = geneId

        # print(EnsemblId, dic['NomenclatureSymbol'])

        keys = ['Ensembl','Entrezid', 'NomenclatureSymbol', 'NomenclatureName', "OtherAliases", "OtherDesignations", 'NomenclatureStatus', 'Mim', 'Summary']

        if to_print:
            for key in keys:
                print("\t %s: '%s'"%(key, dic[key]))

        return dic


    # https://www.ebi.ac.uk/training/online/sites/ebi.ac.uk.training.online/files/UniProt_programmatically_py3.pdf
    # https://biology.stackexchange.com/questions/48407/how-to-batch-convert-gene-names-to-protein-ids-in-uniprot
    def uniprots_to_entrezids(self, uniprots):

        if isinstance(uniprots, str):
            uniprots = uniprots.split(',')
        else:
            uniprots = list(uniprots)

        uniprots = [x for x in uniprots if isinstance(x, str)]

        uniprots = " ".join(uniprots)

        if not isinstance(uniprot, str):
            raise("Error, send a string between spaces or a list")

        url = 'https://www.uniprot.org/uploadlists/'

        params = {
            'from': 'ACC+ID',
            'to': 'ENSEMBL_ID',
            'format': 'tab',
            'query': uniprots
        }

        data = urllib.parse.urlencode(params)
        data = data.encode('utf-8')
        req = urllib.request.Request(url, data)

        with urllib.request.urlopen(req) as f:
            response = f.read()

        stri = response.decode('utf-8')
        dic = {x[0]: x[1] for x in [x.split('\t') for x in stri.strip().split("\n")[1:]]}

        return dic

    def create_summary_description(self, experiment, rootResult, fileExp, fileConv, fileDesc, verbose=False):
        dfexp = pd.read_csv(fileExp, sep="\t")
        print(dfexp.shape, "\n")

        symbols = dfexp.symbol.unique().tolist()
        dfc = self.find_complete_symbols(symbols, verbose=verbose)

        ids = dfc[dfc.symbol.isin(symbols)].ensembl_gene_id.tolist()

        dfe = self.ensembls_data_save(ids, fileDesc, verbose = verbose)

        df1    = dfc[ ['symbol', 'name', 'entrez_id', 'ensembl_gene_id', 'cd', 'locus_type', 'location']]
        df1.columns = ['symbol', 'name', 'entrez_id', 'ensembl',         'cd', 'locus_type', 'location']

        df2 = dfe[['symbol', 'description', 'ensembl', 'synonyms', 'other_designations',
                   'mim', 'genomic_info', 'gene_weight', 'summary', 'chr', 'start',
                   'organism', 'locationHist']]

        dfmerge = pd.merge(df1, df2, on=['symbol', 'ensembl'])

        dfmerge = dfmerge[['symbol', 'name', 'entrez_id', 'ensembl', 'mim', 'cd', 'summary',
                           'synonyms', 'locus_type', 'location',  'other_designations',
                           'genomic_info', 'gene_weight', 'chr', 'start', 'organism',
                           'locationHist']]

        fileMerge = rootResult + "gene_descriptions_for_%s.tsv"%(experiment)
        dfmerge = dfmerge.sort_values([ "symbol", "ensembl"])
        dfmerge.to_csv(fileMerge, sep="\t", index=False)

        print("File (%d, %d) saved at '%s'"%(dfmerge.shape[0], dfmerge.shape[1], fileMerge))
        return dfmerge

    def hugo_id_complete_data(self, hugoid):
        hugoid = str(13666)

        headers = {'Accept': 'application/json',}

        uri = 'http://rest.genenames.org'
        path = '/fetch/hgnc_id/%s'%(hugoid)

        target = urlparse(uri+path)
        method = 'GET'; body = ''

        h = http.Http()

        response, content = h.request(target.geturl(), method, body, headers)

        if response['status'] == '200':
            # assume that content is a json reply
            # parse content with the json module
            data = json.loads(content)
            print('Symbol:' + data['response']['docs'][0]['symbol'])
        else:
            print('Error detected: ' + response['status'])

        return data['response']['docs'][0]

    def open_conv_table_DEPRECATED(self) -> pd.DataFrame:
        filefull = os.path.join(self.root_refseq, self.fname_conv)

        if os.path.exists(filefull):
            dfc = pdreadcsv(self.fname_conv, self.root_refseq)
            print(f"There are {len(dfc)} symbols in {self.fname_conv}")
        else:
            dfc = None
            print(f"Convertion table does not exist: '{filefull}'")

        self.dfc = dfc
        return dfc

    def open_conv_table(self, verbose=True) -> pd.DataFrame:
        return self.open_hgnc(verbose=verbose)

    def open_hgnc(self, verbose=True) -> pd.DataFrame:
        filefull = os.path.join(self.root_hgnc, self.fname_hgnc_set)

        if os.path.exists(filefull):
            dfh = pdreadcsv(self.fname_hgnc_set, self.root_hgnc, verbose=verbose)
        else:
            dfh = None
            print(f"HGNC table does not exist: '{filefull}'")
            print("Download it in https://www.genenames.org/download/archive/")

        self.dfh = dfh
        return dfh

    def open_ncbi_synonym_table(self, verbose=True) -> pd.DataFrame:
        filefull = os.path.join(self.root_refseq, self.fname_ncbi_syn)

        if os.path.exists(filefull):
            dfn = pdreadcsv(self.fname_conv, self.root_refseq, verbose=verbose)
        else:
            dfn = None
            print(f"NCBI synonym table does not exist: '{filefull}'")

        self.dfn = dfn
        return dfn


    '''  symbol_to_complete_data ->  '''
    def find_complete_symbols(self, symbols: List, verbose=False) -> (List, int, int):

        dfc = self.open_hgnc(verbose=False)

        if symbols is None or len(symbols) == 0:
            return [], -1, -1

        n = len(symbols)
        true_list = np.isin(symbols, dfc.symbol, invert=True)
        newsymbs = np.array(symbols)[true_list]

        nnot = len(newsymbs)
        nyes = n - nnot

        if verbose:
            if nnot == 0:
                print(f"All symbols in HGNC table n={nyes}")
            else:
                print(f"There are {nnot} unknown HGNC symbols: {', '.join(newsymbs)}")
                print(f"Found {nyes} symbols.")

        return newsymbs, nyes, nnot


    def ensembls_data_save(self, ids, filename, verbose = False):
        if os.path.exists(filename):
            dfe = pd.read_csv(filename, sep='\t', index_col=False)

            newEnsembls = np.isin(ids, dfe.ensembl.tolist(), invert=True)

            # there are no new symbols
            if np.sum(newEnsembls) == 0:
                print("All ids (ensembls) already stored.")
                return dfe
            newEnsembls = np.array(ids)[newEnsembls]
        else:
            dfe = None
            newEnsembls = np.array(ids)

        print("Findindg new %s ensembls."%(len(newEnsembls)))

        mat = []; count=0
        for ensembl in newEnsembls:
            if verbose: print("id: %s - %d/%d" % (ensembl, count, len(newEnsembls)))
            dic = self.get_ensembl_data(ensembl, verbose = False)
            if dic: mat.append(dic)
            count += 1

        if len(mat) == 0:
            # write_error("There is no map in ensembls_data_save", fileLog, to_append=True)
            print("No new values")
            return dfe

        dfeNew = pd.DataFrame(mat)
        dfeNew = dfeNew[ ['Name', 'Description', 'Ensembl', 'Entrezid', 'Status', 'OtherAliases', 'OtherDesignations',
                          'NomenclatureSymbol', 'NomenclatureName', 'NomenclatureStatus', 'Mim', 'GenomicInfo', 'GeneWeight',
                          'Summary', 'ChrSort', 'ChrStart', 'Organism', 'LocationHist']]
        dfeNew.columns = ['symbol', 'description', 'ensembl', 'entrezid', 'status', 'synonyms', 'other_designations',
                          'nomenclature_symbol', 'nomenclature_name', 'nomenclature_status', 'mim', 'genomic_info',
                          'gene_weight', 'summary', 'chr', 'start', 'organism', 'locationHist']

        if dfe is None:
            dfe = dfeNew
        else:
            frames = [dfe, dfeNew]
            dfe = pd.concat(frames, sort=False)

        dfe = dfe.sort_values("symbol")
        dfe.to_csv(filename, sep="\t", index=False)
        print("File (%d, %d) saved at '%s'"%(dfe.shape[0], dfe.shape[1], filename))

        return dfe


    def from_uniprot_hugo_to_list_of_names(self, symbol, dfc):
        if not symbol in dfc.symbol.to_list():
            print("%s is not in unitprot table."%(symbol))
            return []

        rows = dfc[dfc.symbol == symbol][ ['symbol','name','prev_symbol','prev_name','alias_symbol','cd','alias_name'] ]

        alias = []
        for col in rows.columns:
            name = rows.iloc[0][col]
            if not isinstance(name, float):
                if '[' in name:
                    name = name.replace("]","").replace('[','').replace("'",'').split(', ')
                else:
                    name = [name]

                alias += name

        return alias
