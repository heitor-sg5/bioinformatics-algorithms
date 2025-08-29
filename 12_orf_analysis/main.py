import tkinter as tk
from tkinter import filedialog
import time
from orf_analysis import TwoPassORF, Charts

def get_user_inputs():
    min_input = input("Enter min_size (default 50): ").strip()
    max_input = input("Enter max_overlap (default 3): ").strip()
    k_input = input("Enter k (default 6): ").strip()
    t_input = input("Set t by std or nth-percentile (0/1)? ").strip()
    t_input = int(t_input) if t_input else 0
    if int(t_input) == 1:
        t_input = input("Enter t nth-percentile (10-90, default 50): ")
        t = int(t_input) if t_input else 50
    else:
        t = 0
    L_input = input("Enter L (default 3000): ").strip()
    min_size = int(min_input) if min_input else 50
    max_overlap = int(max_input) if max_input else 3
    k = int(k_input) if k_input else 6
    L = int(L_input) if L_input else 3000
    return min_size, max_overlap, k, t, L

def get_fasta_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select FASTA file.",
        filetypes=[("FASTA files", "*.fasta *.fa *.txt *.fna"), ("All files", "*.*")]
    )
    if not file_path:
        print("No file selected. Using default 'test.txt'.")
        file_path = 'test.txt'
        try:
            text = load_fasta(file_path)
        except FileNotFoundError:
            print("test.txt not found.")
            return
    else:
        text = load_fasta(file_path)
    print("Genome loaded.")
    return text

def load_fasta(file_path):
    with open(file_path, 'r') as file:
        text = ''.join(line.strip() for line in file if not line.startswith(">"))
    return text

def main():
    text = get_fasta_file()
    min_size, max_overlap, k, t, L = get_user_inputs()
    tp = TwoPassORF()
    start_time = time.time()
    orfs = tp.two_pass(text, min_size, max_overlap, t, L, k)
    results = []
    if not orfs:
        results.append("No ORFs found.\n")
    else:
        disp = Charts()
        summary = disp.display(orfs, text)
        results.extend(summary)
        results.append(f"{len(orfs)} ORFs found.\n\n")
        for i, orf in enumerate(orfs, 1):
            results.append(f"ID: {i} | Length: {orf['len'] * 3} ({orf['len']}) | Pos: {orf['start']}-{orf['end']} | Frame: {orf['strand']}{orf['frame']} | Score: {orf['score']:.2f} | Upstream: {orf['upstream']} | Motif: {orf['motif']}\n")
            results.append(f"ORF: {orf['seq']}\n\n")
    total_runtime = time.time() - start_time
    results.append(f"Total Runtime: {total_runtime:.1f} seconds\n")
    with open('results.txt', 'w') as f:
        f.writelines(results)
    print("Results written to results.txt.")
    print(f"Total Runtime: {total_runtime:.1f} seconds")

if __name__ == "__main__":
    main()