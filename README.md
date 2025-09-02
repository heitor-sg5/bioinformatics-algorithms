# Bioinformatics Algorithms

## Overview
This repository contains implementations of core algorithms from Bioinformatics Algorithms: An Active Learning Approach by Phillip Compeau & Pavel Pevzner, as well as some of my own bioinformatics projects. In addition, the repository includes a command-line interface with practical bioinformatics utilities such as GTF/GFF parsers, NCBI genome fetchers, and PubMed search tools. Together, these scripts offer both educational implementations of key computational techniques and practical tools for everyday bioinformatics workflows.

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
- **CpG Island Prediction** - finding CpG-rich regions using GC content, observed/expected CpG ratios, and first-order Markov model scoring
- **HMM Gene Prediction** - exon/intron gene prediction using hidden Markov models trained on known annotaed genomes

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
├── 13_cpg_island_prediction/
│   ├── main.py
│   ├── cpg_island_finder.py
│   ├── test.txt
│   └── results.txt
├── 14_hmm_gene_prediction/
│   ├── main.py
│   ├── gene_prediction.py
│   ├── test1.gff3
│   ├── test2.fasta
│   ├── test3.fasta
│   └── results.txt
├── bio_tools/
│   ├── __init__.py
│   ├── gff_parser.py
│   ├── ncbi_fetch.py
│   ├── pubmed_fetch.py
│   └── chart_builder.py
├── cli.py
├── requirements.txt
├── COMMANDS.md
└── README.md
```

---

## Installation

Prerequisites:

- Python 3.8 or higher
- Required Python packages: numpy, biopython, matplotlib, etc
- MUSCLE for sequence alignment tasks

1. Clone the repository:

```bash
git clone https://github.com/heitor-sg5/Bioinformatics-Algorithms.git
cd Bioinformatics-Algorithms
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Install MUSCLE:

- Download MUSCLE from the official website: https://drive5.com/muscle/downloads_v3.htm
- Follow the instructions for your operating system to install it.
- Set the `muscle_path` variable (in `07_phylogenetic_analysis.py` and `10_hmm_classification_analysis.py`) to the location of the MUSCLE executable on your system.

---

## Usage
### Algorithms

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

### Biotools

The `bio_tool` folder provides a unified CLI for working with data display, GFF files, NCBI records, and PubMed searches. The general command format is:

```bash
python cli.py <tool> <command> [options]
```

Where:

- `<tool>` is one of:
  - `gff` for parsing and summarizing GTF/GFF files
  - `ncbi` for fetching NCBI records
  - `pubmed` for searching and summarizing PubMed papers
  - `chart` for displaying data in charts

- `<command>` is the specific action (e.g., `summarize`, `fetch`, `search`)

- `[options]` are tool-specific flags (e.g., `--accession`, `--max`, `--out`)

All commands are listed in `COMMANDS.md`.

---

## Author

Heitor Gelain do Nascimento | 
Email: heitorgelain@outlook.com | 
GitHub: @heitor-sg5

---

## References

Bioinformatics Algorithms: An Active Learning Approach (3rd Edition) by Phillip Compeau & Pavel Pevzner https://bioinformaticsalgorithms.com
