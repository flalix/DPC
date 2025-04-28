#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

library(DescTools)

#https://search.r-project.org/CRAN/refmans/DescTools/html/DunnettTest.html
# ‘DescTools’ version 0.99.56 is in the repositories but depends on R (>= 4.2.0)
# https://cran.r-project.org/web/packages/DescTools/readme/README.html

# needs gfortran

# conda install -y -c conda-forge r-base=4.4  (must be 4.4 or greater)
# install.packages("DescTools")

# library(DescTools)


if (length(args)==0) {
  file = './tmp/dunnett.tsv'
  conf.level = 0.95
} else {
  file = args[1]
  conf.level = as.numeric(args[2])
}

# file = './tmp/dunnett.tsv'
df = read.table(file, header=TRUE, sep="\t")

control = df[1,'group']

dn <- DunnettTest(df$exam, df$group, control=control, conf.level=conf.level)

# print(dn[control])
dn <- dn[control]

file = './tmp/dunnett_result.tsv'
write.table(dn, file, sep='\t')

# print("Ok dunnett-test")


