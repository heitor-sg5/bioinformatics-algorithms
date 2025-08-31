import typer
from Bio import Entrez, Medline
import pandas as pd
from io import StringIO
from pathlib import Path

app = typer.Typer(help="Search and summarize PubMed papers.")

Entrez.email = "test.biotools1@gmail.com"

@app.command("search")
def search_pubmed(
    keywords: str = typer.Argument(..., help="Search keywords, comma-separated, e.g., 'CRISPR,genome editing'"),
    max_results: int = typer.Option(5, "--max", "-m", help="Maximum number of papers to fetch"),
    year_from: int = typer.Option(None, "--from", help="Start year for filtering"),
    year_to: int = typer.Option(None, "--to", help="End year for filtering"),
    min_count: int = typer.Option(0, "--min_count", "-c", help="Minimum total keyword occurrences in abstract"),
    out_file: str = typer.Option("pubmed_summary.txt", "--out", "-o", help="File to save the results")
):
    query = " OR ".join([k.strip() for k in keywords.split(",")])
    if year_from and year_to:
        query += f" AND ({year_from}[PDAT] : {year_to}[PDAT])"

    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()

    pmids = record["IdList"]
    if not pmids:
        typer.echo("No papers found.")
        raise typer.Exit()

    typer.echo(f"Found {len(pmids)} paper(s) for query: '{keywords}'\n")

    handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="medline", retmode="text")
    papers = handle.read()
    handle.close()

    from Bio import Medline
    from io import StringIO
    records = Medline.parse(StringIO(papers))

    keyword_list = [k.strip() for k in keywords.split(",")]
    output_lines = []

    for rec in records:
        title = rec.get("TI", "")
        authors = ", ".join(rec.get("AU", []))
        journal = rec.get("JT", "")
        pub_year = rec.get("DP", "")
        abstract = rec.get("AB", "")

        total_count = sum(abstract.lower().count(k.lower()) for k in keyword_list)

        if total_count < min_count:
            continue 

        highlighted_abstract = abstract
        for k in keyword_list:
            highlighted_abstract = highlighted_abstract.replace(k, k.upper())
            highlighted_abstract = highlighted_abstract.replace(k.lower(), k.upper())

        output_lines.append("--------------------------------------------------")
        output_lines.append(f"Title: {title}")
        output_lines.append(f"Authors: {authors}")
        output_lines.append(f"Journal: {journal}")
        output_lines.append(f"Year: {pub_year}")
        output_lines.append(f"Total Keyword Count in Abstract: {total_count}")
        output_lines.append(f"Abstract: {highlighted_abstract}")
        output_lines.append("--------------------------------------------------\n")

    if not output_lines:
        typer.echo(f"No papers met the minimum keyword count ({min_count}).")
        raise typer.Exit()

    for line in output_lines:
        typer.echo(line)

    from pathlib import Path
    out_path = Path(out_file)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    typer.echo(f"\nResults saved to {out_path}")