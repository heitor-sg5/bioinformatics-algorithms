# Bioinformatics Algorithms

## Overview
This repository contains implementations of core algorithms from Bioinformatics Algorithms: An Active Learning Approach by Phillip Compeau & Pavel Pevzner, as well as some of my own bioinformatics projects. It provides Python scripts for a broad range of bioinformatics topics, serving as a reference for key computational techniques and algorithmic concepts.

---

## Table of Contents:
- [Overview](#bioinformatics-algorithms)
- [Covered Topics](#covered-topics)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Author](#author)
- [References](#references)

---

## Covered Topics
- **Origin Finding** – detecting replication origins in bacterial genomes  
- **Motif Search** – identifying common motifs across DNA sequences  
- **Genome Sequencing** – assembling genomes from reads or paired reads
- **Peptide Sequencing** – reconstructing cyclic peptides from mass spectrometry  
- **Sequence Alignment** – aligning DNA, RNA, and protein sequences  
- **Genome Rearrangement** – finding minimum 2-break operations between two signed genome permutations
- **Phylogenetic Analysis** – building evolutionary trees  
- **Clustering Analysis** – grouping biological data based on similarity  
- **Combinatorial Pattern Matching** – efficient pattern search algorithms  
- **HMM Classification Analysis** – sequence classification using profile hidden Markov model matrices
- **Peptide Vector Search** – sequences linear peptides from vector spectra and searches for database matches
- **ORF Analysis** - identifies and scores open reading frames using Markov models and motif search with position weight matrices.

---

## Project Structure

```
Bioinformatics-Algorithms/
├── 01_origin_finder/
│   ├── main.py
│   ├── origin_finder.py
│   ├── test.txt
│   └── results.txt
├── 02_motif_search/
│   ├── main.py
│   ├── motif_search.py
│   ├── test.txt
│   └── results.txt
├── 03_genome_sequencing/
│   ├── main.py
│   ├── genome_sequencing.py
│   ├── reads_and_pairs.txt
│   ├── test.txt
│   └── results.txt
├── 04_peptide_sequencing/
│   ├── main.py
│   ├── peptide_sequencing.py
│   ├── test.txt
│   └── results.txt
├── 05_sequence_alignment/
│   ├── main.py
│   ├── sequence_alignment.py
│   ├── local_test.txt
│   ├── global_test.txt
│   ├── fitting_test.txt
│   ├── msa_test.txt
│   └── results.txt
├── 06_genome_rearrangement/
│   ├── main.py
│   ├── genome_rearrangement.py
│   ├── 2b_test1.txt
│   ├── 2b_test2.txt
│   ├── bp_test1.txt
│   ├── bp_test2.txt
│   └── results.txt
├── 07_phylogenetic_analysis/
│   ├── main.py
│   ├── phylogenetic_analysis.py
│   ├── test.txt
│   └── results.txt
├── 08_clustering_analysis/
│   ├── main.py
│   ├── clustering_analysis.py
│   ├── test.txt
│   └── results.txt
├── 09_combinatorial_pattern_matching/
│   ├── main.py
│   ├── pattern_matching.py
│   ├── test1.txt
│   ├── test2.txt
│   ├── reads.txt
│   └── results.txt
├── 10_hmm_classification_analysis/
│   ├── main.py
│   ├── hmm_classification.py
│   ├── test1.txt
│   ├── test2.txt
│   ├── hmm_matrices.txt
│   └── results.txt
├── 11_peptide_vector_search/
│   ├── main.py
│   ├── peptide_search.py
│   ├── test1.txt
│   ├── test2.txt
│   └── results.txt
├── 12_orf_analysis/
│   ├── main.py
│   ├── orf_analysis.py
│   ├── test.txt
│   └── results.txt
└── README.md
```

---

## Installation

Prerequisites:

- Python 3.8 or higher
- Required Python packages: numpy, biopython, matplotlib
- MUSCLE for sequence alignment tasks

1. Clone the repository:

```bash
git clone https://github.com/heitor-sg5/Bioinformatics-Algorithms.git
cd Bioinformatics-Algorithms
```

2. Install Python dependencies:

```bash
pip install numpy biopython matplotlib
```

3. Install MUSCLE:

- Download MUSCLE from the official website: https://drive5.com/muscle/downloads_v3.htm
- Follow the instructions for your operating system to install it.
- Set the `muscle_path` variable (in `07_phylogenetic_analysis.py` and `10_hmm_classification_analysis.py`) to the location of the MUSCLE executable on your system.

---

## Usage

Each topic folder contains a `main.py` script that runs the corresponding algorithms.  

1. Navigate to the topic folder, for example:

```bash
cd 02_motif_search
```

2. Run the main script:

```bash
python main.py
```

3. The script will prompt you for the required inputs (files, parameters).

- If no files or parameters are selected, it will automatically use the provided test files and default variables.

---

## Author

Heitor Gelain do Nascimento | 
Email: heitorgelain@outlook.com | 
GitHub: @heitor-sg5

---

## References

Bioinformatics Algorithms: An Active Learning Approach (3rd Edition) by Phillip Compeau & Pavel Pevzner https://bioinformaticsalgorithms.com
