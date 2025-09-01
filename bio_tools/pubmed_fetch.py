import typer
from Bio import Entrez, Medline
from pathlib import Path
from io import StringIO
import requests
import re

app = typer.Typer(help="Search and summarize PubMed papers.")

Entrez.email = "test.biotools1@gmail.com"

@app.command("search")
def search_pubmed(
    keywords: str = typer.Argument(..., help="Search keywords, comma-separated, e.g., 'CRISPR,genome editing'"),
    max_results: int = typer.Option(5, "--max", "-m", help="Maximum number of papers to fetch"),
    year_from: int = typer.Option(None, "--from", help="Start year for filtering"),
    year_to: int = typer.Option(None, "--to", help="End year for filtering"),
    min_count: int = typer.Option(0, "--min_count", "-c", help="Minimum total keyword occurrences in abstract"),
    order_by: str = typer.Option("recent", "--order", "-r", help="Order results: 'recent', 'oldest', 'keyword'")
):
    if " AND " in keywords.upper():
        query = " AND ".join([k.strip() for k in keywords.split("AND")])
    else:
        query = " OR ".join([k.strip() for k in keywords.split(",")])

    if year_from and year_to:
        query += f" AND ({year_from}[PDAT] : {year_to}[PDAT])"

    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="pub date" if order_by in ["recent", "oldest"] else None)
    record = Entrez.read(handle)
    handle.close()

    pmids = record["IdList"]
    if not pmids:
        typer.echo("No papers found.")
        raise typer.Exit()

    typer.echo(f"Found {len(pmids)} paper(s) for query: '{keywords}'\n")

    handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="medline", retmode="text")
    records = Medline.parse(StringIO(handle.read()))
    handle.close()

    keyword_list = [k.strip() for k in keywords.replace("AND", ",").replace("OR", ",").split(",")]

    papers_data = []
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

        papers_data.append({
            "title": title,
            "authors": authors,
            "journal": journal,
            "year": pub_year,
            "keyword_count": total_count,
            "abstract": highlighted_abstract,
            "pmid": rec.get("PMID", "")
        })

    if not papers_data:
        typer.echo(f"No papers met the minimum keyword count ({min_count}).")
        raise typer.Exit()

    if order_by == "keyword":
        papers_data.sort(key=lambda x: x["keyword_count"], reverse=True)
    elif order_by == "recent":
        papers_data.sort(key=lambda x: x["year"], reverse=True)
    elif order_by == "oldest":
        papers_data.sort(key=lambda x: x["year"])

    typer.echo(f"{len(papers_data)} papers matched filters.")
    save_results = typer.prompt("Do you want to save the results to a text file? (y/n)").lower() == "y"

    if save_results:
        out_file = typer.prompt("Enter a filename to save results (e.g., pubmed_summary.txt)").strip()
        lines = []
        for p in papers_data:
            lines.append("--------------------------------------------------")
            lines.append(f"Title: {p['title']}")
            lines.append(f"Authors: {p['authors']}")
            lines.append(f"Journal: {p['journal']}")
            lines.append(f"Year: {p['year']}")
            lines.append(f"Total Keyword Count in Abstract: {p['keyword_count']}")
            lines.append(f"PMID: {p['pmid']}")
            lines.append("Abstract:")
            lines.append(p['abstract'])
            lines.append("--------------------------------------------------\n")
        out_path = Path(out_file)
        out_path.write_text("\n".join(lines), encoding="utf-8")
        typer.echo(f"Results saved to {out_path}")

    download_pdf = typer.prompt("Do you want to download PDFs for any papers? (y/n)").lower() == "y"
    if download_pdf:
        choices = typer.prompt("Enter PMIDs separated by commas or 'all' for all papers").lower()
        pmid_list = [p["pmid"] for p in papers_data] if choices == "all" else [x.strip() for x in choices.split(",")]

        for pmid in pmid_list:
            try:
                handle = Entrez.elink(dbfrom="pubmed", id=pmid, db="pmc")
                records = Entrez.read(handle)
                handle.close()

                pmc_ids = []
                for linkset in records:
                    if "LinkSetDb" in linkset:
                        for ldb in linkset["LinkSetDb"]:
                            for link in ldb["Link"]:
                                pmc_ids.append(link["Id"])
                if not pmc_ids:
                    typer.echo(f"No free PDF available for PMID {pmid}.")
                    continue

                pmc_id = pmc_ids[0]
                pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/"
                response = requests.get(pdf_url)
                if response.status_code == 200:
                    title = next(p["title"] for p in papers_data if p["pmid"] == pmid)
                    clean_title = re.sub(r'[^\w\s-]', '', title)[:50].replace(' ', '_')
                    filename = f"{pmid}_{clean_title}.pdf"
                    with open(filename, "wb") as f:
                        f.write(response.content)
                    typer.echo(f"Downloaded PDF for PMID {pmid} as {filename}")
                else:
                    typer.echo(f"PDF not available for PMID {pmid}.")
            except Exception as e:
                typer.echo(f"Error fetching PDF for PMID {pmid}: {e}")