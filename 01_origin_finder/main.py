import tkinter as tk
from tkinter import filedialog
import time
from origin_finder import GCSkews, FrequentKmers

def get_user_inputs():
    k_input = input("Enter k (default 9): ").strip()
    d_input = input("Enter d (default 1): ").strip()
    L_input = input("Enter L (default 500): ").strip()
    k = int(k_input) if k_input else 9
    d = int(d_input) if d_input else 1
    L = int(L_input) if L_input else 500
    return k, d, L

def get_fasta_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a FASTA file",
        filetypes=[("FASTA files", "*.fasta *.fa *.fna *.txt"), ("All files", "*.*")]
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
    k, d, L = get_user_inputs()
    overall_start = time.time()
    skew = GCSkews()
    kmers = FrequentKmers()
    start_time = time.time()
    min_skew_pos = skew.run(text)
    skew_runtime = time.time() - start_time
    windows = [
        (max(0, min_skew_pos - L), min_skew_pos, f"-{L}"),
        (max(0, min_skew_pos - L//2), min_skew_pos + L//2, "0"),
        (min_skew_pos, min(min_skew_pos + L, len(text)), f"+{L}"),
    ]
    results = []
    results.append(f"Minimum Skew Position: {min_skew_pos}\nSkew Analysis Runtime: {skew_runtime:.1f} seconds\n\n")
    for start, end, window_name in windows:
        window_seq = text[start:end]
        window_start_time = time.time()
        frequent_kmers = kmers.run(window_seq, k, d)
        window_runtime = time.time() - window_start_time
        results.append(f"Window {window_name}: positions [{start}:{end}]\n")
        if not frequent_kmers:
            results.append("  No frequent patterns found.\n")
        else:
            for pattern, count in frequent_kmers:
                results.append(f"  Pattern: {pattern} | Count: {count}\n")
        results.append(f"  Window Runtime: {window_runtime:.1f} seconds\n\n")
    total_runtime = time.time() - overall_start
    results.append(f"Total Runtime: {total_runtime:.1f} seconds\n")
    with open('results.txt', 'w') as f:
        f.writelines(results)
    print("Results written to results.txt.")
    print(f"Total Runtime: {total_runtime:.1f} seconds")

if __name__ == "__main__":
    main()
