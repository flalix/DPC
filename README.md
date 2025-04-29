# The Digital Pathway Curation (DPC) pipeline

## Molecular Target Discovery project (MTDp)

The Molecular Target Discovery project (MTDp) is composed by a set of 4 pipelines.

  1. Digital Pathway Curation (DPC)
  2. Best Cutoff Algorithm (BCA)
  3. pseudo-Pathway Modulation (pPMod)
  4. In-silico gene validation

Here, we are presenting the **Digital Pathway Curation** (DPC), a study case.

## Questions

The Digital Pathway Curation (DPC) pipeline was designed and tested to answer the following questions:  
	1. How can we demonstrate that a molecular biological pathway is related to a disease?  
	2. How can we demonstrate whether a set of molecular biological pathways, calculated beyond the default cutoffs using GSEA, is related to a disease case?  
	3. How can someone query Gemini in a natural language and retrieve reproducible answers without hallucinations?  
	4. How can we quantify AI answers, make inferences, and calculate statistics?  
	5. How can we demonstrate that Gemini delivers reproducible results using a set of systems biology questions?  

  
Regarding the relationship between a set of molecular biological pathways and a disease, we also considered the following questions:  
	6. How can we calculate the accuracy of the answers provided by Gemini, PubMed, and humans?  
	7. How do we uncover FP and FN in a given enriched table?  


DPC was written in Python 3.11 and integrated with the Gemini AI tool and PubMed through their web services. It stores an Ensemble of questions and answers to perform counts, comparisons and statistical analyses.

In this study, we defined and measured the concepts of consensus, reproducibility, crowdsourcing, and accuracy. Using a smaller dataset with two disease cases, we calculated crowdsource consensus (CSC) and assessed the accuracy of each of the "3 Sources": Gemini, PubMed, and human evaluations. Our findings indicate that Gemini's consensus accuracy outperformed the PubMed and human accuracies. Additionally, using Gemini, we could uncover False Positive (FP) and False Negative (FN) pathways related to each disease case and confirm the True Positive (TP) and True Negative (TN) pathways.  


## DPC aims to  

   1. calculate Gemini (LLM) reproducibility  
   2. calculate the consensus through the 4DSSQ  
   3. merge the "3 Sources" to calcualte the CSC  
   4. calculates each source accuracy.  


## Data

Data can be found [here](https://drive.google.com/drive/u/0/folders/1U6FBkKGE4SisHXUR9RhNiF6CyOQUa200)


## Caveat

The DPC was implemented using two studies (COVID-19 proteomics, and Medulloblastoma transcriptomis), and more tests are being performed with other OMIC experiments.

The DPC was necessary to support the soon-to-be-published Best Cutoff Algorithm (BCA).

After DPC and BCA are published, a Python library will be send to the official Python sites.

We intend to make a website for the Molecular Target Discovery pipeline (MTDp) available after publishing the 4 sub-pipelines.
