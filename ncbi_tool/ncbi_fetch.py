import typer
from Bio import Entrez, SeqIO

app = typer.Typer(help="Fetch data from NCBI databases.")

Entrez.email = "test.biotools1@gmail.com"

@app.command("fetch")
def fetch(
    accession: str = typer.Argument(..., help="NCBI accession IDs (comma-separated for multiple)"),
    db: str = typer.Option("nuccore", "--db", "-d", help="NCBI database: nuccore, protein, etc."),
    rettype: str = typer.Option("gb", "--format", "-f", help="Return type: gb (GenBank), fasta, etc."),
    out: str = typer.Option(None, "--out", "-o", help="Output file path (optional)")
):
    ids = [x.strip() for x in accession.split(",")]
    typer.echo(f"Fetching {len(ids)} record(s) from NCBI '{db}' database in '{rettype}' format...")

    try:
        handle = Entrez.efetch(db=db, id=",".join(ids), rettype=rettype, retmode="text")

        if out:
            with open(out, "w") as f:
                f.write(handle.read())
            typer.echo(f"Output saved to {out}")
        else:
            if rettype.lower() in ["gb", "genbank"]:
                for seq_record in SeqIO.parse(handle, "genbank"):
                    typer.echo("--------------------------------------------------")
                    typer.echo(f"Accession: {seq_record.id}")
                    typer.echo(f"Organism: {seq_record.annotations.get('organism', 'N/A')}")
                    typer.echo(f"Description: {seq_record.description}")
                    typer.echo(f"Sequence Length: {len(seq_record.seq)} bp")
                    typer.echo(f"Number of features: {len(seq_record.features)}")
                    typer.echo("--------------------------------------------------\n")
            elif rettype.lower() in ["fasta", "fa"]:
                for seq_record in SeqIO.parse(handle, "fasta"):
                    typer.echo(f">{seq_record.id} | Length: {len(seq_record.seq)}")
                    typer.echo(seq_record.seq[:100] + " ...")
            else:
                data = handle.read()
                typer.echo(data[:1000] + "\n...")
        handle.close()

    except Exception as e:
        typer.echo(f"Error fetching data from NCBI: {e}")
        raise typer.Exit(code=1)