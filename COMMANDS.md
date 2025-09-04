# Biotools

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
python cli.py ncbi fetch --accession NC_000913
```

2. Fetch multiple accession IDs
```bash
python cli.py ncbi fetch --accession NC_000913,NC_005213
```

3. Fetch by organism name
```bash
python cli.py ncbi fetch "Escherichia coli"
```

---

## PubMed Paper Fetch

1. Search PubMed by a single keyword
```bash
python cli.py pubmed search "CRISPR"
```

2. Search PubMed by multiple keywords 

2a. (OR statement)
```bash
python cli.py pubmed search "CRISPR, genome editing"
```

2b. (AND statement)
```bash
python cli.py pubmed search "CRISPR AND genome editing"
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

6. Order results by

6a. (Most recent)
```bash
python cli.py pubmed search "CRISPR" --recent
```

6b. (Oldest)
```bash
python cli.py pubmed search "CRISPR" --oldest
```

6c. (Keyword count)
```bash
python cli.py pubmed search "CRISPR" --keyword
```

---

## Chart Builder

1. Build line chart
```bash
python cli.py chart build file/to/path.txt --line
```

2. Build bar chart
```bash
python cli.py chart build file/to/path.txt --bar
```

3. Build scatter chart
```bash
python cli.py chart build file/to/path.txt --scatter
```
