import typer
import gffutils
import pandas as pd
import requests
import zipfile
import io
from pathlib import Path

app = typer.Typer(help="GFF/GTF parsing and summarization utilities.")

def download_gff_from_ncbi(accession):
    typer.echo(f"Downloading GFF3 for accession {accession} from NCBI...")

    url = f"https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/{accession}/download"
    response = requests.get(url)

    if response.status_code != 200:
        typer.echo(f"Failed to fetch data for {accession}. HTTP {response.status_code}")
        raise typer.Exit(code=1)

    typer.echo("Download complete. Extracting GFF3 file...")

    zip_bytes = io.BytesIO(response.content)
    with zipfile.ZipFile(zip_bytes) as z:
        gff_files = [f for f in z.namelist() if f.endswith(".gff") or f.endswith(".gff3")]
        if not gff_files:
            typer.echo("No GFF file found in the downloaded dataset.")
            raise typer.Exit(code=1)
        gff_filename = gff_files[0]
        gff_path = Path(f"{accession}.gff3")
        with open(gff_path, "wb") as f:
            f.write(z.read(gff_filename))

    typer.echo(f"GFF3 extracted to {gff_path}")
    return gff_path

def process_gff(file_path):
    typer.echo(f"Parsing {file_path} ... this might take a while.")

    db = gffutils.create_db(
        str(file_path),
        dbfn=":memory:",
        force=True,
        keep_order=True,
        merge_strategy="merge",
        sort_attribute_values=True,
    )
    typer.echo("GFF/GTF parsed successfully!")

    feature_counts = {ft: db.count_features_of_type(ft) for ft in db.featuretypes()}
    df_counts = pd.DataFrame(list(feature_counts.items()), columns=["Feature Type", "Count"])

    lengths = [(f.end - f.start + 1) for f in db.all_features()]
    df_stats = pd.DataFrame({
        "Total Features": [len(lengths)],
        "Min Length": [min(lengths)],
        "Max Length": [max(lengths)],
        "Mean Length": [sum(lengths) / len(lengths)],
    })

    chromosomes = set(f.seqid for f in db.all_features())
    chr_name = list(chromosomes)[0] if chromosomes else "Unknown"

    summary_lines = []
    summary_lines.append("--------------------------------------------------")
    summary_lines.append(f"File: {file_path.name}")
    summary_lines.append(f"Chromosome / Contig: {chr_name}\n")
    summary_lines.append("Feature Counts:")
    summary_lines.append(df_counts.to_string(index=False))
    summary_lines.append("\nFeature Length Stats:")
    summary_lines.append(df_stats.to_string(index=False))
    summary_lines.append("--------------------------------------------------\n")

    return "\n".join(summary_lines)

@app.command("summarize")
def summarize(
    file: str = typer.Argument(None, help="Path to GFF/GTF file (optional if --accession is used)"),
    accession: str = typer.Option(None, "--accession", "-a", help="NCBI accession ID to fetch GFF3 automatically")
):
    combined_summary = ""
    all_files = []

    while True:
        if accession:
            file_path = download_gff_from_ncbi(accession)
        elif file:
            file_path = Path(file)
            if not file_path.exists():
                typer.echo(f"Error: File '{file}' not found.")
                raise typer.Exit(code=1)
        else:
            typer.echo("You must provide either a GFF file path or an --accession ID.")
            raise typer.Exit(code=1)

        summary_text = process_gff(file_path)
        typer.echo(summary_text)

        combined_summary += summary_text + "\n"
        all_files.append(file_path.stem if accession is None else accession)

        another = typer.prompt("Do you want to analyze another GFF/GTF file? (y/n)").strip().lower()
        if another != "y":
            break

        choice = typer.prompt("Do you want to enter a file path or an accession? (file/accession)").strip().lower()
        if choice == "accession":
            accession = typer.prompt("Enter the NCBI accession ID")
            file = None
        else:
            file = typer.prompt("Enter the path to the GFF/GTF file")
            accession = None

    save_summary = typer.prompt("Do you want to save the summary to a text file? (y/n)").strip().lower()
    if save_summary == "y":
        filename = "gff_summary.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(combined_summary)
        typer.echo(f"Summary saved to: {filename}")