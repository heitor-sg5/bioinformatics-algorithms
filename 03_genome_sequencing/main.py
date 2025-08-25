import tkinter as tk
from tkinter import filedialog
import time
from genome_sequencing import Sequencing, DeBruijnEulerian, PairedDeBruijnEulerian, MaximalNonBranching
def get_user_inputs():
    k_input = input("Enter k (default 10): ").strip()
    d_input = input("Enter d (default 50): ").strip()
    k = int(k_input) if k_input else 10
    d = int(d_input) if d_input else 50
    return k, d

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
    genome = get_fasta_file()
    k, d = get_user_inputs()
    gs = Sequencing()
    kmers_str = gs.generate_kmers(genome, k)
    paired_reads_str = gs.generate_read_pairs(genome, k, d)
    with open('reads_and_pairs.txt', 'w') as rp_file:
        rp_file.write("Reads:\n")
        rp_file.write(kmers_str + "\n\n")
        rp_file.write("Paired reads:\n")
        rp_file.write(paired_reads_str + "\n")
    print("Reads and paired reads written to reads_and_pairs.txt.")
    algorithms = [
        ('De Bruijn Eulerian Path', DeBruijnEulerian()),
        ('Paired De Bruijn Eulerian Path', PairedDeBruijnEulerian(k, d)),
        ('Maximal Non-Branching Paths', MaximalNonBranching())
    ]
    results = []
    overall_start = time.time()
    for algo_name, algo in algorithms:
        start_time = time.time()
        if algo_name == 'Paired De Bruijn Eulerian Path':
            result = algo.run(paired_reads_str)
        else:
            result = algo.run(kmers_str)
        runtime = time.time() - start_time
        results.append(f"---{algo_name}---\n")
        results.append(f"Runtime: {runtime:.1f} seconds\n")
        if algo_name == 'Maximal Non-Branching Paths':
            for i in range(len(result)):
                results.append(f"Contig {i + 1}: {result[i]}\n")
        else:
            results.append(f"Reconstructed genome:\n{result}\n")
        results.append("\n")
    total_runtime = time.time() - overall_start
    results.append(f"Total Runtime: {total_runtime:.1f} seconds\n")
    with open('results.txt', 'w') as f:
        f.writelines(results)
    print("Results written to results.txt.")
    print(f"Total Runtime: {total_runtime:.1f} seconds")

if __name__ == "__main__":
    main()