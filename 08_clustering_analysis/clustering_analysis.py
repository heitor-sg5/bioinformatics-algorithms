import random
import math
import numpy as np

class ClusteringBase:
    def euclidean(self, a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    
    def mean_point(self, points, weights=None):
        points = np.asarray(points, dtype=float)
        if weights is None:
            return tuple(points.mean(axis=0))
        weights = np.asarray(weights, dtype=float)
        return tuple(np.average(points, axis=0, weights=weights))
    
    def pearson_correlation(self, a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        am = a.mean()
        bm = b.mean()
        num = np.sum((a - am) * (b - bm))
        den = math.sqrt(np.sum((a - am) ** 2) * np.sum((b - bm) ** 2))
        return num / den if den != 0 else 0.0
    
    def pearson_distance(self, a, b):
        return 1.0 - self.pearson_correlation(a, b)
    
    def format_result(self, result, data, algo_name):
        if algo_name == "Hierarchical Clustering":
            root, edges = result
            result_str = []
            def collect_tree(node, indent=0):
                result_str.append("  " * indent + str(node.name))
                for child in node.children:
                    length = node.age - child.age
                    result_str.append("  " * (indent + 1) + f"|-- {child.name} (len={length:.4f})")
                    collect_tree(child, indent + 2)
            collect_tree(root)
            return "\n".join(result_str)
        else:
            centres, clusters = result
            result_str = []
            for i, centre in enumerate(centres):
                cluster_points = [data[idx] for idx in clusters.get(i, [])]
                centre_str = tuple(round(float(c), 1) for c in centre)
                points_str = [tuple(round(float(x), 1) for x in p) for p in cluster_points]
                result_str.append(f"Cluster {i+1}: {points_str} and centre = {centre_str}")
            return "\n".join(result_str)

class Node:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.age = 0.0
    
    def add_child(self, child):
        self.children.append(child)

class KCentresClustering(ClusteringBase):
    def run(self, data, k):
        centres = [random.choice(data)]
        while len(centres) < k:
            farthest_point = max(data, key=lambda p: min(self.euclidean(p, c) for c in centres))
            centres.append(farthest_point)
        clusters = {i: [] for i in range(k)}
        for idx, point in enumerate(data):
            nearest_idx = min(range(k), key=lambda i: self.euclidean(point, centres[i]))
            clusters[nearest_idx].append(idx)
        return centres, clusters

class LloydKMeansClustering(ClusteringBase):
    def run(self, data, k, max_iterations=100):
        centres = random.sample(data, k)
        for _ in range(max_iterations):
            clusters = {i: [] for i in range(k)}
            for idx, point in enumerate(data):
                nearest_idx = min(range(k), key=lambda i: self.euclidean(point, centres[i]))
                clusters[nearest_idx].append(idx)
            new_centres = []
            for i in range(k):
                if clusters[i]:
                    points = [data[idx] for idx in clusters[i]]
                    new_centres.append(self.mean_point(points))
                else:
                    new_centres.append(random.choice(data))
            if all(self.euclidean(centres[i], new_centres[i]) < 1e-6 for i in range(k)):
                break
            centres = new_centres
        return centres, clusters

class SoftKMeansClustering(ClusteringBase):
    def __init__(self, beta):
        self.beta = beta
    
    def run(self, data, k, max_iterations=100):
        centres = random.sample(data, k)
        for _ in range(max_iterations):
            responsibilities = []
            for point in data:
                weights = [math.exp(-self.beta * self.euclidean(point, c)) for c in centres]
                total_weight = sum(weights)
                responsibilities.append([w / total_weight for w in weights])
            new_centres = []
            for j in range(k):
                weights_j = [resp[j] for resp in responsibilities]
                new_centres.append(self.mean_point(data, weights_j))
            if max(self.euclidean(centres[i], new_centres[i]) for i in range(k)) < 1e-6:
                break
            centres = new_centres
        clusters = {i: [] for i in range(k)}
        for i, point in enumerate(data):
            assigned = max(range(k), key=lambda j: responsibilities[i][j])
            clusters[assigned].append(i)
        return centres, clusters

class CASTClustering(ClusteringBase):
    def __init__(self, theta):
        self.theta = theta
    
    def run(self, data, k):
        n = len(data)
        points = np.asarray(data, dtype=float)
        max_dist = max(self.euclidean(points[i], points[j]) for i in range(n) for j in range(i+1, n)) if n > 1 else 1.0
        R = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    R[i, j] = 1.0
                else:
                    dist = self.euclidean(points[i], points[j])
                    R[i, j] = R[j, i] = 1.0 - dist / max_dist if max_dist != 0 else 1.0
        unassigned = set(range(n))
        clusters = []
        while unassigned:
            degrees = {i: sum(1 for j in unassigned if j != i and R[i, j] >= self.theta) for i in unassigned}
            seed = max(degrees, key=degrees.get)
            C = {seed}
            changed = True
            while changed:
                changed = False
                best_add = None
                for i in unassigned - C:
                    aff = np.mean([R[i, j] for j in C]) if C else 0.0
                    if aff >= self.theta:
                        if best_add is None or aff > best_add[0]:
                            best_add = (aff, i)
                if best_add:
                    C.add(best_add[1])
                    changed = True
                worst_remove = None
                for i in list(C):
                    aff = np.mean([R[i, j] for j in C]) if len(C) > 1 else 1.0
                    if aff < self.theta:
                        if worst_remove is None or aff < worst_remove[0]:
                            worst_remove = (aff, i)
                if worst_remove:
                    C.remove(worst_remove[1])
                    changed = True
            clusters.append(sorted(C))
            unassigned -= C
        centres = [self.mean_point([data[i] for i in cl]) for cl in clusters]
        return centres, {i: cl for i, cl in enumerate(clusters)}

class HierarchicalClustering(ClusteringBase):
    def __init__(self, linkage):
        self.linkage = linkage
    
    def run(self, data, k):
        n = len(data)
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = self.pearson_distance(data[i], data[j])
        clusters = {i: [i] for i in range(n)}
        nodes = {i: Node(str(i)) for i in range(n)}
        ages = {i: 0.0 for i in range(n)}
        current_id = n
        while len(clusters) > 1:
            min_dist = float('inf')
            to_merge = None
            cluster_keys = list(clusters.keys())
            for i in range(len(cluster_keys)):
                for j in range(i + 1, len(cluster_keys)):
                    ci, cj = cluster_keys[i], cluster_keys[j]
                    vals = [D[a][b] for a in clusters[ci] for b in clusters[cj]]
                    dist = sum(vals) / (len(clusters[ci]) * len(clusters[cj])) if self.linkage == "avg" else max(vals) if self.linkage == "max" else min(vals)
                    if dist < min_dist:
                        min_dist = dist
                        to_merge = (ci, cj)
            ci, cj = to_merge
            new_cluster = clusters[ci] + clusters[cj]
            new_node = Node(str(current_id))
            new_node.add_child(nodes[ci])
            new_node.add_child(nodes[cj])
            new_node.age = min_dist / 2
            nodes[current_id] = new_node
            ages[current_id] = new_node.age
            new_dists = []
            for ck in cluster_keys:
                if ck != ci and ck != cj:
                    vals = [D[a][b] for a in new_cluster for b in clusters[ck]]
                    dist = sum(vals) / (len(new_cluster) * len(clusters[ck])) if self.linkage == "avg" else max(vals) if self.linkage == "max" else min(vals)
                    new_dists.append((ck, dist))
            clusters.pop(ci)
            clusters.pop(cj)
            clusters[current_id] = new_cluster
            D = np.pad(D, ((0, 1), (0, 1)), mode='constant', constant_values=0.0)
            for ck, dist in new_dists:
                D[current_id][ck] = dist
                D[ck][current_id] = dist
            current_id += 1
        root = nodes[list(clusters.keys())[0]]
        edges = []

        def collect_edges(node):
            for child in node.children:
                length = node.age - child.age
                edges.append((node.name, child.name, length))
                collect_edges(child)
        collect_edges(root)

        return root, edges