import typer
from ncbi_tool import gff_parser, ncbi_fetch, pubmed_fetch

app = typer.Typer(
    help="NCBI Tools: Parse GFF files, fetch NCBI data, and summarize PubMed papers."
)

app.add_typer(gff_parser.app, name="gff", help="Parse and summarize GFF/GTF files")
app.add_typer(ncbi_fetch.app, name="ncbi", help="Fetch and summarize NCBI data")
app.add_typer(pubmed_fetch.app, name="pubmed", help="Fetch and summarize PubMed papers")

if __name__ == "__main__":
    app()