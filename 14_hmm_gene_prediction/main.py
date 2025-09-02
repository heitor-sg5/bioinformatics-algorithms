import tkinter as tk
from tkinter import filedialog
import time
from Bio import SeqIO
from Bio.Seq import Seq
import sys
from gene_prediction import ParseFiles, BuildHMM, FindViterbi, FindGenes, Charts

def get_user_inputs():
    root = tk.Tk()
    root.withdraw()
    print("Select Reference GFF3 file")
    gff3 = filedialog.askopenfilename(title="Select Reference GFF3 file")
    print("Select Reference FASTA file (same chromosome as GFF3)")
    ref_fa = filedialog.askopenfilename(title="Select Reference FASTA file")
    print("Select Query FASTA file")
    query_fa = filedialog.askopenfilename(title="Select Query FASTA file")
    return gff3, ref_fa, query_fa

def choose_frame():
    while True:
        print("Select reading frame to analyze (+1, +2, +3, -1, -2, -3):")
        frame = input().strip()
        if frame in ['+1', '+2', '+3', '-1', '-2', '-3']:
            return frame
        print("Invalid input. Please enter one of +1, +2, +3, -1, -2, -3.")

def adjust_sequence_for_frame(seq_str, frame):
    seq = Seq(seq_str.upper())
    if frame.startswith('-'):
        seq = seq.reverse_complement()
    shift = int(frame[1]) - 1
    if shift > 0:
        seq = seq[shift:]
    return str(seq)

def main():
    gff3, ref_fa, query_fa = get_user_inputs()
    ref_records = list(SeqIO.parse(ref_fa, 'fasta'))
    if len(ref_records) == 0:
        print('Reference FASTA appears empty.')
        sys.exit(1)
    ref_record = ref_records[0]
    parse = ParseFiles()
    hmm = BuildHMM()
    viterbi = FindViterbi()
    find = FindGenes()
    start_time = time.time()
    genes_exons, used_seqid = parse.parse_gff3(gff3, seqid_filter=ref_record.id)
    if not genes_exons:
        genes_exons, used_seqid = parse.parse_gff3(gff3, seqid_filter=None)
        if not genes_exons:
            print('No exon features parsed from GFF3.')
            sys.exit(1)
    print(f'Parsed {len(genes_exons)} genes from GFF3.')
    print('Building HMM...')
    model, labels = hmm.build_hmm_from_annotation(ref_record, genes_exons)
    query_records = list(SeqIO.parse(query_fa, 'fasta'))
    if len(query_records) == 0:
        print('Query FASTA appears empty.')
        sys.exit(1)
    query_record = query_records[0]
    frame = choose_frame()
    seq_str = adjust_sequence_for_frame(str(query_record.seq), frame)
    print(f"Analyzing sequence in frame {frame} (length {len(seq_str)})")
    logp, path, genes = viterbi.run(model, Seq(seq_str))
    print(f'Viterbi log-probability: {logp:.2f}')
    total_runtime = time.time() - start_time
    disp = Charts()
    disp.display(seq_str, genes)
    results = []
    results.extend(find.get_genes(seq_str, genes, path))
    results.append(f"Total Runtime: {total_runtime:.1f} seconds\n")
    with open('results.txt', 'w') as f:
        f.writelines(results)
    print("Results written to results.txt.")
    print(f"Total Runtime: {total_runtime:.1f} seconds")

if __name__ == "__main__": 
    main()