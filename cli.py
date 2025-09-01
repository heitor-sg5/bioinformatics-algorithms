import typer
from bio_tools import gff_parser, ncbi_fetch, pubmed_fetch, chart_builder

app = typer.Typer(help="Parse GFF files, fetch NCBI data, summarize PubMed papers, and more!")

app.add_typer(gff_parser.app, name="gff", help="Parse and summarize GFF/GTF files")
app.add_typer(ncbi_fetch.app, name="ncbi", help="Fetch and summarize NCBI data")
app.add_typer(pubmed_fetch.app, name="pubmed", help="Fetch and summarize PubMed papers")
app.add_typer(chart_builder.app, name="chart", help="Display data in charts")

if __name__ == "__main__":
    app()