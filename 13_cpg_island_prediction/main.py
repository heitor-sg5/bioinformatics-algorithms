import tkinter as tk
from tkinter import filedialog
import time
from cpg_island_finder import FindCpGIslands, Charts

def get_user_inputs():
    L_input = input("Enter window size (default 500): ").strip()
    w_input = input("Enter window extension (default 50): ").strip()
    s_input = input("Enter window step (default 100): ").strip()
    p_input = input("Enter tolerance (default 2): ").strip()
    minl_input = input("Enter min length (default 200): ").strip()
    maxl_input = input("Enter max length (default 5000): ").strip()
    mingc_input = input("Enter min GC% (default 0.5): ").strip()
    minoes_input = input("Enter min Obs/Exp (default 0.6): ").strip()
    t_input = input("Enter nth-percentile threshold (default 90): ").strip()
    mgap_input = input("Enter merge gap (default 100): ").strip()
    L = int(L_input) if L_input else 500
    w = int(w_input) if w_input else 50
    s = int(s_input) if s_input else 100
    p = int(p_input) if p_input else 2
    min_length = int(minl_input) if minl_input else 200
    max_length = int(maxl_input) if maxl_input else 5000
    min_gc = float(mingc_input) if mingc_input else 0.5
    min_obsexp = float(minoes_input) if minoes_input else 0.6
    t = float(t_input) if t_input else 90
    merge_gap = int(mgap_input) if mgap_input else 100
    return L, w, s, p, min_length, max_length, min_gc, min_obsexp, t, merge_gap

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
    text = text.upper()
    text = text.replace('N', '')
    L, w, s, p, min_length, max_length, min_gc, min_obsexp, t, merge_gap = get_user_inputs()
    cpg = FindCpGIslands()
    start_time = time.time()
    cpg_islands = cpg.run(text, L, s, w, p, min_gc, min_obsexp, min_length, max_length, t, merge_gap)
    results = []
    if not cpg_islands:
        results.append("No CpG islands found.\n")
    else:
        disp = Charts()
        summary = disp.display(cpg_islands, text)
        results.extend(summary)
        results.append(f"{len(cpg_islands)} CpG islands found.\n\n")
        for i, island in enumerate(cpg_islands, 1):
            results.append(f"ID: {i} | Length: {island['len']} | Position: {island['start']}-{island['end']} | GC%: {island['gc']:.2f} | Expected ratio: {island['obsexp']:.2f} | Score: {island['llr']:.2f}\n")
            results.append(f"CpG Island:\n{island['seq']}\n\n")
    total_runtime = time.time() - start_time
    results.append(f"Total Runtime: {total_runtime:.1f} seconds\n")
    with open('results.txt', 'w') as f:
        f.writelines(results)
    print("Results written to results.txt.")
    print(f"Total Runtime: {total_runtime:.1f} seconds")

if __name__ == "__main__":
    main()  