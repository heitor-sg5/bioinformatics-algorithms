from collections import defaultdict, deque
import re

class Sequencing:
    def reconstruct_genome_from_path(self, nodes, path):
        if not path:
            return "No path found"
        genome = nodes[0] if isinstance(nodes, list) else nodes
        for node in path[1:]:
            genome += node[-1] if isinstance(node, str) else node[0][-1]
        return genome

    def generate_kmers(self, genome, k):
        kmers = [genome[i:i+k] for i in range(len(genome) - k + 1)]
        kmers.sort()
        return ' '.join(kmers)

    def generate_read_pairs(self, genome, k, d):
        L = 2 * k + d
        pairs = []
        for i in range(len(genome) - L + 1):
            window = genome[i:i+L]
            first_kmer = window[:k]
            second_kmer = window[k + d:k + d + k]
            pairs.append(f"({first_kmer},{second_kmer})")
        pairs.sort()
        return ' '.join(pairs)

    def build_de_bruijn_graph(self, kmers_str):
        kmers = kmers_str.strip().split()
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        nodes = set()
        for kmer in kmers:
            prefix = kmer[:-1]
            suffix = kmer[1:]
            graph[prefix].append(suffix)
            out_degree[prefix] += 1
            in_degree[suffix] += 1
            nodes.update([prefix, suffix])
        return graph, nodes, in_degree, out_degree

class DeBruijnEulerian(Sequencing):
    def run(self, kmers_str):
        graph, nodes, in_degree, out_degree = self.build_de_bruijn_graph(kmers_str)
        start_node = None
        for node in nodes:
            outdeg = out_degree.get(node, 0)
            indeg = in_degree.get(node, 0)
            if outdeg - indeg == 1:
                start_node = node
                break
        if not start_node:
            for node in nodes:
                if out_degree.get(node, 0) > 0:
                    start_node = node
                    break
        if not start_node:
            return "No valid start node found"
        stack = [start_node]
        path = []
        local_graph = {u: deque(v) for u, v in graph.items()}
        while stack:
            u = stack[-1]
            if u in local_graph and local_graph[u]:
                v = local_graph[u].popleft()
                stack.append(v)
            else:
                path.append(stack.pop())
        path = path[::-1]
        return self.reconstruct_genome_from_path(path[0], path)

class PairedDeBruijnEulerian(Sequencing):
    def __init__(self, k, d):
        self.k = k
        self.d = d

    def parse_paired_reads(self, input_str):
        pairs = re.findall(r"\((\w+),(\w+)\)", input_str)
        return [(a, b) for a, b in pairs]

    def build_paired_de_bruijn_graph(self, pairs):
        graph = defaultdict(deque)
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        for prefix, suffix in pairs:
            from_node = (prefix[:-1], suffix[:-1])
            to_node = (prefix[1:], suffix[1:])
            graph[from_node].append(to_node)
            out_degree[from_node] += 1
            in_degree[to_node] += 1
        return graph, in_degree, out_degree

    def find_start_node(self, graph, in_degree, out_degree):
        for node in graph:
            if out_degree[node] - in_degree[node] == 1:
                return node
        return next(iter(graph), None)

    def run(self, paired_reads_str):
        pairs = self.parse_paired_reads(paired_reads_str)
        if not pairs:
            return "No valid paired reads found"
        graph, in_degree, out_degree = self.build_paired_de_bruijn_graph(pairs)
        start = self.find_start_node(graph, in_degree, out_degree)
        if not start:
            return "No valid start node found"
        graph_copy = {node: deque(neighbour) for node, neighbour in graph.items()}
        path = []
        stack = [start]
        while stack:
            u = stack[-1]
            if u in graph_copy and graph_copy[u]:
                v = graph_copy[u].popleft()
                stack.append(v)
            else:
                path.append(stack.pop())
        path.reverse()
        if not path:
            return "No Eulerian path found"
        prefix_string = path[0][0]
        suffix_string = path[0][1]
        for i in range(1, len(path)):
            prefix_string += path[i][0][-1]
            suffix_string += path[i][1][-1]
        for i in range(self.k + self.d, len(prefix_string)):
            if prefix_string[i] != suffix_string[i - self.k - self.d]:
                return "No valid genome can be reconstructed"
        return prefix_string + suffix_string[-(self.k + self.d):]

class MaximalNonBranching(Sequencing):
    def run(self, kmers_str):
        graph, nodes, in_degree, out_degree = self.build_de_bruijn_graph(kmers_str)

        def is_1_in_1_out(node):
            return in_degree[node] == 1 and out_degree[node] == 1
        
        paths = []
        visited = set()
        for node in nodes:
            if not is_1_in_1_out(node):
                if node in graph:
                    for neighbour in graph[node]:
                        path = [node, neighbour]
                        visited.add((node, neighbour))
                        current = neighbour
                        while is_1_in_1_out(current):
                            next_node = graph[current][0]
                            path.append(next_node)
                            visited.add((current, next_node))
                            current = next_node
                        paths.append(path)
        for node in nodes:
            if is_1_in_1_out(node):
                if (node, graph[node][0]) not in visited:
                    cycle = [node]
                    current = graph[node][0]
                    while current != node:
                        cycle.append(current)
                        current = graph[current][0]
                    cycle.append(node)
                    paths.append(cycle)
        return [self.reconstruct_genome_from_path(path[0], path) for path in paths]