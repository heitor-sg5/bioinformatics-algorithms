# NCBI Tools

---

## GTF/GFF3 File Parser:

1. Summarize a local GFF/GTF file
```bash
python cli.py gff summarize path/to/file.gff3
```

2. Summarize a GFF3 file by NCBI accession
```bash
python cli.py gff summarize --accession GCF_000002285.3
```

---

## NCBI Database Fetch:

1. Fetch a single GenBank record
```bash
python cli.py ncbi fetch NC_000913
```

2. Fetch multiple accession IDs
```bash
python cli.py ncbi fetch NC_000913,NC_005213
```

3. Specify a different NCBI database
```bash
python cli.py ncbi fetch NC_000913 --db nuccore
```

4. Specify output format GenBank
```bash
python cli.py ncbi fetch NC_000913 --format gb
```

5. Specify output format FASTA
```bash
python cli.py ncbi fetch NC_000913 --format fasta
```

6. Save fetched record to file
```bash
python cli.py ncbi fetch NC_000913 --out ecoli.gb
```

---

## PubMed Paper Fetch

1. Search PubMed by a single keyword
```bash
python cli.py pubmed search "CRISPR"
```

2. Search PubMed by multiple keywords
```bash
python cli.py pubmed search "CRISPR, genome editing"
```

3. Limit number of results
```bash
python cli.py pubmed search "CRISPR" --max 10
```

4. Filter results by publication year
```bash
python cli.py pubmed search "CRISPR" --from 2018 --to 2023
```

5. Minimum keywords in abstract
```bash
python cli.py pubmed search "CRISPR, genome editing" --min_count 2
```

6. Save results to a file
```bash
--out python cli.py pubmed search "CRISPR" --pubmed_results.txt