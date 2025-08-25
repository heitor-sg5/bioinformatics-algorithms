import tkinter as tk
from tkinter import filedialog
import time
import numpy as np
from sequence_alignment import GlobalAlignment, SmithWaterman, FittingAlignment, MSA

def get_user_inputs():
    align_scope = input("Align 2 sequences (p) or multiple (m)?: ").strip().lower()
    if align_scope not in ['p', 'm']:
        align_scope = 'p'
        print("Invalid input. Defaulting to pairwise alignment.")
    align_type = 'g'
    if align_scope == 'p':
        align_type = input("Alignment type: global (g), local (l), or fitting (f)?: ").strip().lower()
        if align_type not in ['g', 'l', 'f']:
            align_type = 'g'
            print("Invalid input. Defaulting to global alignment.")
    pgap_input = input("Enter gap penalty (default 1): ").strip()
    pmm_input = input("Enter mismatch penalty (default 1): ").strip()
    gap_open_input = input("Enter gap opening penalty (default 1): ").strip()
    gap_extend_input = input("Enter gap extension penalty (default 0.5): ").strip()
    pgap = float(pgap_input) if pgap_input else 1.0
    pmm = float(pmm_input) if pmm_input else 1.0
    gap_open = float(gap_open_input) if gap_open_input else 1.0
    gap_extend = float(gap_extend_input) if gap_extend_input else 0.5
    return align_scope, align_type, pgap, pmm, gap_open, gap_extend

def get_fasta_files():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a FASTA file",
        filetypes=[("FASTA files", "*.fasta *.fa *.fna *.txt"), ("All files", "*.*")]
    )
    if not file_path:
        print("No file selected. Using default 'test.txt'.")
        file_path = 'global_test.txt'
        try:
            sequences = load_fasta(file_path)
        except FileNotFoundError:
            print(f"{file_path} not found.")
            return
    else:
        sequences = load_fasta(file_path)
    return sequences

def load_fasta(file_path):
    sequences = []
    current_seq = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_seq:
                    sequences.append(''.join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line.replace(" ", "").upper())
        if current_seq:
            sequences.append(''.join(current_seq))
    return sequences

def main():
    sequences = get_fasta_files()
    if not sequences:
        return
    align_scope, align_type, pgap, pmm, gap_open, gap_extend = get_user_inputs()
    if align_scope == 'p' and len(sequences) != 2:
        print("Pairwise alignment requires exactly 2 sequences. Using first two.")
        sequences = sequences[:2]
    elif align_scope == 'm' and len(sequences) < 3:
        print("MSA requires at least 3 sequences. Switching to pairwise global alignment.")
        align_scope, align_type = 'p', 'g'
        sequences = sequences[:2]
    if align_scope == 'p' and align_type == 'f' and len(sequences[0]) > len(sequences[1]):
        print("Fitting alignment: swapping sequences so first is shorter.")
        sequences = [sequences[1], sequences[0]]
    print(f"Loaded {len(sequences)} sequences for {'pairwise' if align_scope == 'p' else 'MSA'} "
          f"{'global' if align_type == 'g' else 'local' if align_type == 'l' else 'fitting'} alignment.")
    if align_scope == 'p':
        if align_type == 'g':
            algorithms = [
                ('Needleman-Wunsch Global Alignment', GlobalAlignment('needleman_wunsch', pmm, pgap)),
                ('Global Alignment with Affine Gap Penalty', GlobalAlignment('affine_gap', pmm, pgap, gap_open, gap_extend)),
                ('Global Alignment with Hirschberg Linear Space', GlobalAlignment('hirschberg', pmm, pgap)),
                ('Global Alignment with PAM250 Scoring', GlobalAlignment('pam250', pmm, pgap))
            ]
        elif align_type == 'l':
            algorithms = [
                ('Smith-Waterman Local Alignment', SmithWaterman(pmm, pgap))
            ]
        else:
            algorithms = [
                ('Fitting Alignment', FittingAlignment(pmm, pgap))
            ]
    else:
        algorithms = [
            ('MSA Global Alignment (Pair-Sum Scoring)', MSA('pair_sum', pmm, pgap)),
            ('MSA Global Alignment (Entropy Scoring)', MSA('entropy', pmm, pgap))
        ]
    results = []
    overall_start = time.time()
    for algo_name, algo in algorithms:
        start_time = time.time()
        if align_scope == 'p':
            aligned_seqs, score = algo.run(sequences[0], sequences[1])
        else:
            aligned_seqs, score = algo.run(sequences)
        runtime = time.time() - start_time
        results.append(f"---{algo_name}---\n")
        results.append(f"Runtime: {runtime:.1f} seconds\n")
        for i, seq in enumerate(aligned_seqs, 1):
            results.append(f"Aligned Sequence {i}:\n{seq}\n")
        results.append(f"Score: {score}\n\n")
    total_runtime = time.time() - overall_start
    results.append(f"Total Runtime: {total_runtime:.1f} seconds\n")
    with open('results.txt', 'w') as f:
        f.writelines(results)
    print("Results written to results.txt.")
    print(f"Total Runtime: {total_runtime:.1f} seconds")

if __name__ == "__main__":
    main()