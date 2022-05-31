# Datasets

This directory contains the two preprocessed datasets used in our
experiments.
Each dataset consists of one CSV file, each line of which represents
one edge.
Please refer to the paper for a description of the preprocessing steps.

### VAST dataset

The original data can be found
[here](http://visualdata.wustl.edu/varepository/VAST%20Challenge%202013/challenges/MC3%20-%20Big%20Marketing/),
including the complete ground truth description of the hosts and
attacks.
We only use the NetFlow data in our experiments.
Top nodes are internal hosts, bottom nodes are external hosts, and edge
types are defined as (protocol, destination port, direction) triples.

### LANL dataset

The original data can be found
[here](https://csr.lanl.gov/data/cyber1/).
Our experiments rely on the authentication logs (`auth.txt.gz`), and
the red team events are listed in the file `redteam.txt.gz`.
Top nodes are users, bottom nodes are hosts, and edge types are
defined as (authentication package, logon type, direction) triples.
