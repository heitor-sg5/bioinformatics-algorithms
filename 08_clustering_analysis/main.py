import tkinter as tk
from tkinter import filedialog
import time
import ast
from clustering_analysis import KCentresClustering, LloydKMeansClustering, SoftKMeansClustering, CASTClustering, HierarchicalClustering

def get_user_inputs(): 
    k_input = input("Enter k (default 3): ").strip()
    beta_input = input("Enter beta (default 1.0): ").strip()
    theta_input = input("Enter theta (0.8): ").strip()
    linkage_input = input("Enter linkage type for Hierarchical Clustering (single, avg, complete; default avg): ").strip()
    k = int(k_input) if k_input and k_input.isdigit() else 3
    beta = float(beta_input) if beta_input and beta_input.replace('.', '', 1).isdigit() else 1.0
    theta = float(theta_input) if theta_input and theta_input.replace('.', '', 1).isdigit() else 0.8
    linkage = linkage_input if linkage_input in ["single", "avg", "complete"] else "avg"
    return k, beta, theta, linkage

def get_data_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select data matrix file",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    if not file_path:
        print("No file selected. Using default data matrix.")
        file_path = 'test.txt'
        try:
            data = load_data_matrix(file_path)
        except FileNotFoundError:
            print("test.txt not found.")
            return
    else:
        data = load_data_matrix(file_path)
    if not data:
        print("No valid data points found in the file.")
        return
    print("Data matrix loaded.")
    return data

def load_data_matrix(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        data = ast.literal_eval(content)
    return data

def main():
    data = get_data_file()
    k, beta, theta, linkage = get_user_inputs()
    results = []
    overall_start = time.time()
    matrix_str = "Data Matrix:\n"
    for row in data:
        matrix_str += f"{' '.join(f'{x:.1f}' for x in row)}\n"
    results.append(matrix_str + "\n")
    algorithms = [
        ("k-Centres Clustering", KCentresClustering()),
        ("Lloyd's k-Means Clustering", LloydKMeansClustering()),
        ("Soft k-Means Clustering", SoftKMeansClustering(beta)),
        ("CAST Clustering", CASTClustering(theta)),
        ("Hierarchical Clustering", HierarchicalClustering(linkage))
    ]
    for algo_name, algo in algorithms:
        start_time = time.time()
        result = algo.run(data, k)
        runtime = time.time() - start_time
        results.append(f"--- {algo_name} ---\n")
        results.append(f"Runtime: {runtime:.1f} seconds\n")
        results.append("Result:\n")
        result_str = algo.format_result(result, data, algo_name)
        results.append(result_str + "\n")
        results.append("\n")
    total_runtime = time.time() - overall_start
    results.append(f"Total Runtime: {total_runtime:.1f} seconds\n")
    with open('results.txt', 'w') as f:
        f.writelines(results)
    print("Results written to results.txt.")
    print(f"Total Runtime: {total_runtime:.1f} seconds")

if __name__ == "__main__":
    main()