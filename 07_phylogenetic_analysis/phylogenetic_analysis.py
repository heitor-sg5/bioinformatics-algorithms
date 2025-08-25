import subprocess
import tempfile
import os
import numpy as np
import random
from abc import ABC

class DistanceMatrixBase:
    def muscle_align(self, sequences, muscle_path=r"C:\Tools\muscle.exe"):
        if not sequences:
            raise ValueError("The sequence list is empty.")
        if not os.path.isfile(muscle_path):
            raise FileNotFoundError(f"MUSCLE executable not found at {muscle_path}")
        with tempfile.TemporaryDirectory() as temp_dir:
            input_fasta = os.path.join(temp_dir, "input.fasta")
            output_fasta = os.path.join(temp_dir, "output.fasta")
            with open(input_fasta, 'w') as f:
                for i, seq in enumerate(sequences):
                    f.write(f">seq{i+1}\n{seq}\n")
            result = subprocess.run(
                [muscle_path, "-align", input_fasta, "-output", output_fasta],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"MUSCLE failed:\n{result.stderr}")
            aligned_sequences = []
            with open(output_fasta, 'r') as f:
                seq = ""
                for line in f:
                    if line.startswith(">"):
                        if seq:
                            aligned_sequences.append(seq)
                            seq = ""
                    else:
                        seq += line.strip()
                if seq:
                    aligned_sequences.append(seq)
            return aligned_sequences
    
    def remove_indel_columns(self, sequences):
        aligned_seqs = self.muscle_align(sequences)
        if not aligned_seqs:
            return []
        transposed = list(zip(*aligned_seqs))
        columns_to_keep = [i for i, col in enumerate(transposed) if '-' not in col]
        return [''.join(seq[i] for i in columns_to_keep) for seq in aligned_seqs]
    
    def hamming_distance(self, v, w):
        return sum(1 for a, b in zip(v, w) if a != b)
    
    def kimura_distance(self, v, w):
        transitions = 0
        transversions = 0
        valid_positions = 0
        transitions_pairs = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}
        for a, b in zip(v, w):
            if a == '-' or b == '-':
                continue
            if a == b:
                valid_positions += 1
            else:
                valid_positions += 1
                if (a, b) in transitions_pairs:
                    transitions += 1
                else:
                    transversions += 1
        if valid_positions == 0:
            return 0.0
        P = transitions / valid_positions
        Q = transversions / valid_positions
        try:
            distance = -0.5 * np.log((1 - 2 * P - Q) * np.sqrt(1 - 2 * Q))
        except ValueError:
            distance = float('inf')
        return round(distance, 2)
    
    def build_distance_matrix(self, sequences, matrix_type):
        aligned_seqs = self.muscle_align(sequences)
        n = len(aligned_seqs)
        dist_matrix = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                dist = (self.hamming_distance if matrix_type == 0 else self.kimura_distance)(aligned_seqs[i], aligned_seqs[j])
                dist_matrix[i][j] = dist
                dist_matrix[j][i] = dist
        return dist_matrix
    
class PhyloBase(ABC):
    def format_tree(self, node):
        result_str = []
        def collect_tree(n, indent=0):
            result_str.append("  " * indent + str(n.name))
            for child, length in n.children:
                result_str.append("  " * (indent + 1) + f"|-- {child.name} (len={length:.4f})")
                collect_tree(child, indent + 2)
        collect_tree(node)
        return "\n".join(result_str)
    
    def format_newick(self, node):
        def to_newick(n):
            if n.is_leaf():
                return n.name
            children_str = ",".join(
                f"{to_newick(child)}:{length:.2f}" 
                for child, length in n.children
            )
            return f"({children_str}){n.name or ''}"
        return to_newick(node) + ";"

class Node:
    def __init__(self, name=None):
        self.name = name
        self.children = []
        self.age = 0.0
        self.label = None
        self.score = {}
        self.tagged = False
    
    def add_child(self, child, length=0.0):
        self.children.append((child, length))
    
    def is_leaf(self):
        return len(self.children) == 0

class UPGMA(PhyloBase):
    def run(self, D, labels):
        D = np.array(D, dtype=float)
        n = len(D)
        clusters = {i: [i] for i in range(n)}
        nodes = {i: Node(str(labels[i])) for i in range(n)}
        ages = {i: 0.0 for i in range(n)}
        current_id = n
        while len(clusters) > 1:
            min_dist = float('inf')
            to_merge = None
            cluster_keys = list(clusters.keys())
            for i in range(len(cluster_keys)):
                for j in range(i + 1, len(cluster_keys)):
                    ci, cj = cluster_keys[i], cluster_keys[j]
                    dist = sum(D[a][b] for a in clusters[ci] for b in clusters[cj]) / (len(clusters[ci]) * len(clusters[cj]))
                    if dist < min_dist:
                        min_dist = dist
                        to_merge = (ci, cj)
            ci, cj = to_merge
            new_cluster = clusters[ci] + clusters[cj]
            new_node = Node(str(current_id))
            new_node.add_child(nodes[ci], min_dist / 2 - ages[ci])
            new_node.add_child(nodes[cj], min_dist / 2 - ages[cj])
            new_node.age = min_dist / 2
            nodes[current_id] = new_node
            ages[current_id] = new_node.age
            new_dist_row = []
            for ck in cluster_keys:
                if ck != ci and ck != cj:
                    dist = sum(D[a][b] for a in new_cluster for b in clusters[ck]) / (len(new_cluster) * len(clusters[ck]))
                    new_dist_row.append((ck, dist))
            clusters.pop(ci)
            clusters.pop(cj)
            clusters[current_id] = new_cluster
            D = np.vstack([D, np.zeros((1, D.shape[1]))])
            D = np.hstack([D, np.zeros((D.shape[0], 1))])
            for ck, dist in new_dist_row:
                D[current_id][ck] = dist
                D[ck][current_id] = dist
            current_id += 1
        return nodes[list(clusters.keys())[0]]

class NeighborJoining(PhyloBase):
    def run(self, D, labels):
        D = np.array(D, dtype=float)
        n = len(D)
        if n == 2:
            node = Node(f"({labels[0]},{labels[1]})")
            node.add_child(Node(str(labels[0])), D[0,1] / 2)
            node.add_child(Node(str(labels[1])), D[0,1] / 2)
            return node
        total_dist = np.sum(D, axis=1)
        D_star = np.full((n,n), np.inf)
        for i in range(n):
            for j in range(n):
                if i != j:
                    D_star[i,j] = (n - 2) * D[i,j] - total_dist[i] - total_dist[j]
        i, j = np.unravel_index(np.argmin(D_star), D_star.shape)
        if j < i:
            i, j = j, i
        delta = (total_dist[i] - total_dist[j]) / (n - 2)
        limb_i = 0.5 * (D[i,j] + delta)
        limb_j = 0.5 * (D[i,j] - delta)
        new_label = f"({labels[i]},{labels[j]})"
        new_row = []
        for k in range(n):
            if k != i and k != j:
                dist = 0.5 * (D[i,k] + D[j,k] - D[i,j])
                new_row.append(dist)
        indices = [x for x in range(n) if x != i and x != j]
        new_D = np.zeros((n-1, n-1))
        for a, idx_a in enumerate(indices):
            for b, idx_b in enumerate(indices):
                new_D[a,b] = D[idx_a, idx_b]
        for a in range(n-1 - 1):
            new_D[a, -1] = new_D[-1, a] = new_row[a]
        new_D[-1, -1] = 0.0
        new_labels = [labels[x] for x in indices] + [new_label]
        subtree = self.run(new_D, new_labels)
        new_node = self.find_node(subtree, new_label)
        new_node.add_child(Node(str(labels[i])), limb_i)
        new_node.add_child(Node(str(labels[j])), limb_j)
        return subtree
    
    def find_node(self, node, target_label):
        if node.name == target_label:
            return node
        for child, _ in node.children:
            found = self.find_node(child, target_label)
            if found:
                return found
        return None

class SmallParsimonyAndNNI(PhyloBase):
    def run(self, sequences):
        alphabet = sorted(set("".join(sequences)))
        tree = self.generate_random_tree(sequences)
        current_score = self.small_parsimony(tree, alphabet)
        best_tree = tree
        improved = True
        while improved:
            improved = False
            variants = self.generate_nni_variants(tree)
            for variant in variants:
                score = self.small_parsimony(variant, alphabet)
                if score < current_score:
                    best_tree = variant
                    current_score = score
                    improved = True
                    break
        return best_tree
    
    def small_parsimony(self, root, alphabet):
        def postorder(node):
            if node.tagged:
                return
            for child, _ in node.children:
                postorder(child)
            if node.is_leaf():
                node.tagged = True
                for k in alphabet:
                    node.score[k] = 0 if node.label == k else float('inf')
            else:
                if len(node.children) != 2:
                    raise ValueError(f"Node '{node.name}' does not have exactly 2 children")
                left, right = [child for child, _ in node.children]
                node.tagged = True
                for k in alphabet:
                    min_left = min(left.score[i] + (0 if i == k else 1) for i in alphabet)
                    min_right = min(right.score[j] + (0 if j == k else 1) for j in alphabet)
                    node.score[k] = min_left + min_right
        
        def assign_labels(node, parent_label=None):
            if node.is_leaf():
                return
            if parent_label is None:
                node.label = min(node.score, key=node.score.get)
            else:
                options = sorted(alphabet, key=lambda k: (node.score[k] + (0 if k == parent_label else 1)))
                node.label = options[0]
            for child, _ in node.children:
                assign_labels(child, node.label)
        
        def reset_tags(node):
            node.tagged = False
            node.score = {}
            for child, _ in node.children:
                reset_tags(child)
        
        reset_tags(root)
        postorder(root)
        assign_labels(root)
        return min(root.score.values())
    
    def generate_random_tree(self, strings):
        leaves = [Node(name=str(i)) for i in range(len(strings))]
        for node, label in zip(leaves, strings):
            node.label = label
        nodes = leaves[:]
        idx = len(strings)
        while len(nodes) > 1:
            a = nodes.pop(random.randint(0, len(nodes) - 1))
            b = nodes.pop(random.randint(0, len(nodes) - 1))
            parent = Node(name=f"internal_{idx}")
            idx += 1
            parent.add_child(a, 0.0)
            parent.add_child(b, 0.0)
            nodes.append(parent)
        return nodes[0]
    
    def get_internal_edges(self, node, edges=None):
        if edges is None:
            edges = []
        for child, _ in node.children:
            if not node.is_leaf() and not child.is_leaf():
                edges.append((node.name, child.name))
            self.get_internal_edges(child, edges)
        return edges
    
    def deep_copy_tree(self, node):
        new_node = Node(name=node.name)
        new_node.label = node.label
        for child, length in node.children:
            new_node.add_child(self.deep_copy_tree(child), length)
        return new_node
    
    def find_node(self, node, target_name):
        if node.name == target_name:
            return node
        for child, _ in node.children:
            found = self.find_node(child, target_name)
            if found:
                return found
        return None
    
    def generate_nni_variants(self, tree):
        variants = []
        edges = self.get_internal_edges(tree)
        for u_name, v_name in edges:
            tree_copy = self.deep_copy_tree(tree)
            u = self.find_node(tree_copy, u_name)
            v = self.find_node(tree_copy, v_name)
            if not u or not v or len(v.children) != 2 or v not in [child for child, _ in u.children]:
                continue
            a, b = [child for child, _ in v.children]
            if len(u.children) != 2:
                continue
            other = [child for child, _ in u.children if child != v]
            if not other:
                continue
            for swap_child in (a, b):
                new_tree = self.deep_copy_tree(tree)
                u_new = self.find_node(new_tree, u_name)
                v_new = self.find_node(new_tree, v_name)
                if not u_new or not v_new:
                    continue
                c_new = [child for child, _ in u_new.children if child.name != v_new.name][0]
                a_new, b_new = [child for child, _ in v_new.children]
                if swap_child.name == a_new.name:
                    v_new.children = [(b_new, 0.0), (c_new, 0.0)]
                else:
                    v_new.children = [(a_new, 0.0), (c_new, 0.0)]
                u_new.children = [(child, l) for child, l in u_new.children if child.name != v_new.name]
                u_new.add_child(swap_child, 0.0)
                variants.append(new_tree)
        return variants