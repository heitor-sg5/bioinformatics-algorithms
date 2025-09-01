import typer
from Bio import Entrez, SeqIO
from pathlib import Path
import requests

app = typer.Typer(help="Fetch data from NCBI databases by accession or organism name.")

Entrez.email = "test.biotools1@gmail.com"

@app.command("fetch")
def fetch(
    query: str = typer.Argument(..., help="Scientific name of organism or accession ID"),
    accession: bool = typer.Option(False, "--accession", help="Specify if the query is an NCBI accession ID")
):
    try:
        if accession:
            ids = [query.strip()]
            db = "nuccore"

            typer.echo(f"Fetching {len(ids)} genome(s) from NCBI '{db}' database...")

            rettype = typer.prompt("Choose format (fasta/genbank/gb)").strip().lower()
            seq_format = "genbank" if rettype in ["gb", "genbank"] else "fasta"

            for genome_id in ids:
                handle = Entrez.efetch(db=db, id=genome_id, rettype=rettype, retmode="text")
                seq_records = list(SeqIO.parse(handle, seq_format))
                handle.close()

                if not seq_records:
                    typer.echo(f"No sequence data found for {genome_id}.")
                    continue

                for seq_record in seq_records:
                    typer.echo("--------------------------------------------------")
                    typer.echo(f"Accession: {seq_record.id}")
                    typer.echo(f"Organism: {seq_record.annotations.get('organism', 'N/A')}")
                    typer.echo(f"Description: {seq_record.description}")
                    typer.echo(f"Sequence Length: {len(seq_record.seq)} bp")
                    if rettype in ["gb", "genbank"]:
                        typer.echo(f"Number of features: {len(seq_record.features)}")
                    typer.echo("--------------------------------------------------\n")

                save_file = typer.prompt("Do you want to save this genome to a file? (y/n)").strip().lower()
                if save_file == "y":
                    handle = Entrez.efetch(db=db, id=genome_id, rettype=rettype, retmode="text")
                    content = handle.read()
                    handle.close()
                    organism_name = seq_records[0].annotations.get('organism', 'Unknown').replace(" ", "_")
                    filename = f"{organism_name}_{seq_records[0].id}.{rettype}"
                    Path(filename).write_text(content, encoding="utf-8")
                    typer.echo(f"Genome saved as: {filename}")
            return

        typer.echo(f"Searching NCBI for genomes of '{query}'...")
        handle = Entrez.esearch(db="assembly", term=f'"{query}"[Organism]', retmax=20)
        record = Entrez.read(handle)
        handle.close()

        if not record["IdList"]:
            typer.echo(f"No genomes found for '{query}'. Searching for similar organisms...")
            tax_handle = Entrez.esearch(db="taxonomy", term=query, retmax=5)
            tax_record = Entrez.read(tax_handle)
            tax_handle.close()

            if tax_record["IdList"]:
                suggestions = []
                for tid in tax_record["IdList"]:
                    summary_handle = Entrez.esummary(db="taxonomy", id=tid)
                    summary = Entrez.read(summary_handle)
                    summary_handle.close()
                    suggestions.append(summary[0]["ScientificName"])

                typer.echo("\nDid you mean one of these?")
                for s in suggestions:
                    typer.echo(f"  - {s}")
            else:
                typer.echo("No similar organisms found.")
            raise typer.Exit()

        typer.echo(f"Found {len(record['IdList'])} genome assemblies.")

        handle = Entrez.esummary(db="assembly", id=",".join(record["IdList"]), report="full")
        summaries = Entrez.read(handle)
        handle.close()

        available_assemblies = []
        for s in summaries['DocumentSummarySet']['DocumentSummary']:
            ftp_path = s['FtpPath_GenBank'] or s['FtpPath_RefSeq']
            if ftp_path:
                file_name = ftp_path.split("/")[-1] + "_genomic.fna.gz"
                file_url = f"{ftp_path}/{file_name}"
                available_assemblies.append({
                    "organism": s["Organism"],
                    "accession": s["AssemblyAccession"],
                    "status": s["AssemblyStatus"],
                    "url": file_url
                })

        if not available_assemblies:
            typer.echo("No downloadable genome files found.")
            raise typer.Exit()

        typer.echo("\nAvailable Assemblies:")
        for i, asm in enumerate(available_assemblies, start=1):
            typer.echo(f"  [{i}] {asm['organism']} | {asm['accession']} | {asm['status']}")

        typer.echo("\nEnter the number(s) of the assemblies to download (e.g. 1 or 1,3,5)")
        selection = typer.prompt("Your choice").strip()
        selected_indices = [int(x) - 1 for x in selection.split(",") if x.isdigit()]

        chosen_assemblies = [available_assemblies[i] for i in selected_indices if 0 <= i < len(available_assemblies)]

        typer.echo(f"\nDownloading {len(chosen_assemblies)} genome(s)...\n")

        for asm in chosen_assemblies:
            download_url = asm["url"].replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov")

            typer.echo(f"Downloading {asm['organism']} ({asm['accession']})...")
            response = requests.get(download_url)
            if response.status_code != 200:
                typer.echo(f"Failed to download {asm['organism']} from {download_url}")
                continue

            organism_name = asm["organism"].replace(" ", "_")
            filename = f"{organism_name}_{asm['accession']}_genomic.fna.gz"
            Path(filename).write_bytes(response.content)
            typer.echo(f"Saved: {filename}")

        typer.echo("\nDownload complete.")

    except Exception as e:
        typer.echo(f"Error fetching data from NCBI: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()