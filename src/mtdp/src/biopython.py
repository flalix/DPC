# -*- coding: utf-8 -*-
# Created on 2010/11/03
# @author: Flavio Lichtenstein;

'''
Created on 18/06/2012
Updated on 25/06/2012
Updated on 08/05/2013  read_fasta
Updated on 12/02/2014  countManyLettersDic, calcShannonVerticalEntropy
Update  on 09/06/2015 - MI cannot be negative never
Update  on 14/09/2015 - read_fasta_samples
Update  on 27/09/2015 - full randomizing and shuffling: read_fasta_samples(..., sampledSeqs=None, method=None)

@author: Flavio Lichtenstein
@local: Unifesp Bioinformatica
'''
import os, sys
import numpy as np
import pandas as pd

from Bio import Entrez
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
# from Bio.Alphabet import IUPAC  # Alphabet was removed

class BioPython:
    def __init__(self, email):

        self.NucleotideAlphabet = ["A","C","G","T"]

        self.email = email

    def get_gene_id_given_symbol(self, symbol:str, db="gene"):
        """
        Retrieves transcript information (including accession numbers) for a given gene symbol from NCBI's Gene database.

        Args:
            gene_symbol: The gene symbol (e.g., "TP53").
            db: The database to query (default: "gene").
            email: Your email address (required by NCBI).

        Returns:
            A dictionary where keys are transcript accession numbers and values are dictionaries containing transcript information 
            (including the description). Returns None if no transcripts are found or if there's an error.
        """

        Entrez.email = self.email

        try:
            handle = Entrez.esearch(db=db, term=symbol + "[sym]")
            record = Entrez.read(handle)
            
        except Exception as err: 
            print(f"entrez efectch error: {err}")
            record = None
        finally:
            handle.close()

        if record is None:
            return None, None

        if not record["IdList"]:
            print(f"No gene found for symbol: {symbol}")
            return None

        lista = record["IdList"]
        return lista, record

    def get_transcripts_by_gene_symbol(self, symbol:str):
        """
        Retrieves transcript information (including accession numbers and sequences) for a given gene symbol 
        from NCBI's Gene database using a more robust method.

        Args:
            xxxxxxxxxxx

        Returns:
            xxxxxxxxxxxx
        """
        lista, record = self.get_gene_id_given_symbol(symbol)
        if lista == []:
            return None
        
        Entrez.email = email
        ids = " ".join(lista)
        try:
            #We don't parse the gene record anymore, only get accession numbers of transcripts.
            handle = Entrez.elink(dbfrom="gene", dbto="nuccore", id=lista[0])
            link_results = Entrez.read(handle)

        except Exception as err: 
            print(f"entrez elink error: {err}")
            link_results = None
        finally:
            handle.close()

        transcript_accessions = []
        if link_results and link_results[0]['LinkSetDb']:
            for link in link_results[0]['LinkSetDb'][0]['Link']:
                transcript_accessions.append(link['Id'])

        if not transcript_accessions:
            print(f"No transcripts found for gene ID: {gene_id}")
            return None

        transcripts = {}
        for i in range(len(transcript_accessions)):
            try:
                if i%10: print(i, end=' ')
                acc_num = transcript_accessions[i]
                handle = Entrez.efetch(db="nucleotide", id=acc_num, rettype="fasta", retmode="text")
                seq_record = SeqIO.read(handle, "fasta")
                transcripts[acc_num] = {'description': seq_record.description, 'sequence': str(seq_record.seq)}
            except Exception as e:
                print(f"Error fetching sequence for {acc_num}: {e}")
                transcripts[acc_num] = {'description': None, 'sequence': None}
            finally:
                if handle:
                    handle.close()

        print("\n\n----------------end---------------\n")      
        df = pd.DataFrame(transcripts).T
        print(len(df))

        return link_results


