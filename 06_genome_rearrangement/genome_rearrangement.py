from collections import defaultdict, deque
import matplotlib.pyplot as plt

class GenomeRearrangementBase:
    def pattern_to_number(self, pattern):
        num = 0
        for symbol in pattern:
            num = 4 * num + {'A': 0, 'C': 1, 'G': 2, 'T': 3}[symbol]
        return num
    
    def reverse_complement_number(self, kmer_num, k):
        rc = 0
        for _ in range(k):
            rc = (rc << 2) | (3 - (kmer_num & 3))
            kmer_num >>= 2
        return rc
    
    def build_synteny_graph(self, shared_kmers, max_distance):
        bin_size = max_distance
        bins = defaultdict(list)
        node_positions = []
        for idx, (i, j, _) in enumerate(shared_kmers):
            bin_x = i // bin_size
            bin_y = j // bin_size
            bins[(bin_x, bin_y)].append(idx)
            node_positions.append((i, j))
        adj = defaultdict(list)
        for idx, (i, j, _) in enumerate(shared_kmers):
            bin_x = i // bin_size
            bin_y = j // bin_size
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbor_bin = (bin_x + dx, bin_y + dy)
                    for neighbor_idx in bins.get(neighbor_bin, []):
                        if neighbor_idx == idx:
                            continue
                        i2, j2 = node_positions[neighbor_idx]
                        if abs(i - i2) <= max_distance and abs(j - j2) <= max_distance:
                            adj[idx].append(neighbor_idx)
        return adj
    
    def find_connected_components(self, adj):
        visited = set()
        components = []
        for node in adj:
            if node not in visited:
                comp = []
                queue = deque([node])
                visited.add(node)
                while queue:
                    current = queue.popleft()
                    comp.append(current)
                    for neighbor in adj[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                components.append(comp)
        return components

class SyntenyBlockConstruction(GenomeRearrangementBase):
    def __init__(self, k, max_distance, min_size):
        self.k = k
        self.max_distance = max_distance
        self.min_size = min_size
    
    def find_shared_kmers(self, seq1, seq2):
        index2 = defaultdict(list)
        for j in range(len(seq2) - self.k + 1):
            kmer = seq2[j:j+self.k]
            kmer_num = self.pattern_to_number(kmer)
            index2[kmer_num].append(j)
        shared = []
        for i in range(len(seq1) - self.k + 1):
            kmer = seq1[i:i+self.k]
            kmer_num = self.pattern_to_number(kmer)
            rc_kmer_num = self.reverse_complement_number(kmer_num, self.k)
            for j in index2.get(kmer_num, ()):
                shared.append((i, j, '+'))
            for j in index2.get(rc_kmer_num, ()):
                shared.append((i, j, '-'))
        return shared
    
    def find_shared_kmers_multichr(self, P, Q):
        P_offsets = [0]
        for chrom in P[:-1]:
            P_offsets.append(P_offsets[-1] + len(chrom))
        Q_offsets = [0]
        for chrom in Q[:-1]:
            Q_offsets.append(Q_offsets[-1] + len(chrom))
        shared_kmers = []
        for p_idx, chromP in enumerate(P):
            offset_i = P_offsets[p_idx]
            for q_idx, chromQ in enumerate(Q):
                offset_j = Q_offsets[q_idx]
                shared = self.find_shared_kmers_pair(chromP, chromQ)
                for i, j, orientation in shared:
                    shared_kmers.append((i + offset_i, j + offset_j, orientation))
        return shared_kmers
    
    def find_shared_kmers_pair(self, seq1, seq2):
        index2 = defaultdict(list)
        for j in range(len(seq2) - self.k + 1):
            kmer2 = seq2[j:j+self.k]
            index2[self.pattern_to_number(kmer2)].append(j)
        shared = []
        for i in range(len(seq1) - self.k + 1):
            kmer1 = seq1[i:i+self.k]
            kmer_num = self.pattern_to_number(kmer1)
            for j in index2.get(kmer_num, ()):
                shared.append((i, j, '+'))
            rc_kmer_num = self.reverse_complement_number(kmer_num, self.k)
            for j in index2.get(rc_kmer_num, ()):
                shared.append((i, j, '-'))
            rev_kmer_num = self.pattern_to_number(kmer1[::-1])
            for j in index2.get(rev_kmer_num, ()):
                shared.append((i, j, '-'))
        return shared
    
    def synteny_blocks(self, shared_kmers):
        adj = self.build_synteny_graph(shared_kmers, self.max_distance)
        comps = self.find_connected_components(adj)
        blocks = []
        for comp in comps:
            if len(comp) >= self.min_size:
                block = [shared_kmers[idx] for idx in comp]
                blocks.append(block)
        return blocks
    
    def signed_permutations(self, blocks):
        metas = []
        for block in blocks:
            is_ = [i for i, _, _ in block]
            js = [j for _, j, _ in block]
            orients = [o for *_, o in block]
            avg_i = sum(is_) / len(is_)
            avg_j = sum(js) / len(js)
            sign = '+' if orients.count('+') >= orients.count('-') else '-'
            metas.append({'avg_i': avg_i, 'avg_j': avg_j, 'sign': sign})
        metas_sorted_by_p = sorted(metas, key=lambda x: x['avg_i'])
        for idx, m in enumerate(metas_sorted_by_p, start=1):
            m['id'] = idx
        perm1 = [m['id'] for m in metas_sorted_by_p]
        metas_sorted_by_q = sorted(metas, key=lambda x: x['avg_j'])
        perm2 = [m['id'] if m['sign'] == '+' else -m['id'] for m in metas_sorted_by_q]
        return perm1, perm2
    
    def permutations_grouped_by_chromosomes(self, P, Q):
        m, n = len(P), len(Q)
        blocks_by_pair = {(i, j): [] for i in range(m) for j in range(n)}
        for i, chrP in enumerate(P):
            for j, chrQ in enumerate(Q):
                shared = self.find_shared_kmers_pair(chrP, chrQ)
                if shared:
                    blocks = self.synteny_blocks(shared)
                    if blocks:
                        blocks_by_pair[(i, j)] = blocks
        P_lists = [[] for _ in range(m)]
        Q_entries = defaultdict(list)
        next_id = 1
        for i in range(m):
            metas = []
            for j in range(n):
                for block in blocks_by_pair[(i, j)]:
                    is_ = [ii for ii, _, _ in block]
                    js = [jj for _, jj, _ in block]
                    orients = [o for *_, o in block]
                    avg_i = sum(is_) / len(is_)
                    avg_j = sum(js) / len(js)
                    sign = '+' if orients.count('+') >= orients.count('-') else '-'
                    metas.append({'avg_i': avg_i, 'q_idx': j, 'avg_j': avg_j, 'sign': sign})
            metas.sort(key=lambda x: x['avg_i'])
            for meta in metas:
                block_id = next_id
                next_id += 1
                P_lists[i].append(block_id)
                signed = block_id if meta['sign'] == '+' else -block_id
                Q_entries[meta['q_idx']].append((meta['avg_j'], signed))
        Q_lists = []
        for j in range(n):
            items = sorted(Q_entries.get(j, []), key=lambda t: t[0])
            Q_lists.append([signed_id for _, signed_id in items])
        return P_lists, Q_lists
    
    def plot_dotplot(self, shared_kmers, len1, len2, scale, scale_unit):
        x_f, y_f = [], []
        x_r, y_r = [], []
        for i, j, orientation in shared_kmers:
            if orientation == '+':
                x_f.append(i / scale)
                y_f.append(j / scale)
            else:
                x_r.append(i / scale)
                y_r.append(j / scale)
        plt.figure(figsize=(10, 8))
        plt.scatter(x_f, y_f, color='red', s=10, label='(+) Matches')
        plt.scatter(x_r, y_r, color='blue', s=10, label='(-) Matches')
        plt.xlabel(f'P ({scale_unit})')
        plt.ylabel(f'Q ({scale_unit})')
        plt.title(f'Shared {self.k}-mers Dot-Plot')
        plt.legend()
        plt.xlim([0, len1 / scale])
        plt.ylim([0, len2 / scale])
        plt.grid(True)
        plt.show(block=True)
    
    def plot_dotplot_multichr(self, shared_kmers, P, Q, scale, scale_unit):
        x_f, y_f = [], []
        x_r, y_r = [], []
        P_offsets = [0]
        for chrom in P[:-1]:
            P_offsets.append(P_offsets[-1] + len(chrom))
        Q_offsets = [0]
        for chrom in Q[:-1]:
            Q_offsets.append(Q_offsets[-1] + len(chrom))
        for i, j, orientation in shared_kmers:
            x_shift = 0
            for offset in P_offsets:
                if i >= offset:
                    x_shift = offset
            y_shift = 0
            for offset in Q_offsets:
                if j >= offset:
                    y_shift = offset
            if orientation == '+':
                x_f.append((i - x_shift + x_shift) / scale)
                y_f.append((j - y_shift + y_shift) / scale)
            else:
                x_r.append((i - x_shift + x_shift) / scale)
                y_r.append((j - y_shift + y_shift) / scale)
        plt.figure(figsize=(10, 8))
        plt.scatter(x_f, y_f, color='red', s=10, label='(+) Matches')
        plt.scatter(x_r, y_r, color='blue', s=10, label='(-) Matches')
        plt.xlabel(f'P ({scale_unit})')
        plt.ylabel(f'Q ({scale_unit})')
        plt.title(f'Shared {self.k}-mers Dot-Plot')
        plt.legend()
        plt.xlim([0, sum(len(c) for c in P) / scale])
        plt.ylim([0, sum(len(c) for c in Q) / scale])
        plt.grid(True)
        plt.show(block=True)

class BreakpointSort(GenomeRearrangementBase):
    def count_breakpoints(self, perm):
        return sum(1 for i in range(len(perm) - 1) if perm[i + 1] - perm[i] != 1)
    
    def apply_reversal(self, perm, i, j):
        return perm[:i] + [-x for x in perm[i:j + 1][::-1]] + perm[j + 1:]
    
    def format_perm(self, perm):
        return '[' + ' '.join(f"{'+' if x > 0 else ''}{x}" for x in perm) + ']'
    
    def run(self, perm):
        n = len(perm)
        Q = [0] + perm + [n + 1]
        steps = [perm[:]]
        reversals = 0
        while self.count_breakpoints(Q) > 0:
            best_q = None
            best_breaks = self.count_breakpoints(Q)
            for i in range(1, len(Q) - 1):
                for j in range(i, len(Q) - 1):
                    new_q = self.apply_reversal(Q, i, j)
                    new_breaks = self.count_breakpoints(new_q)
                    if new_breaks < best_breaks:
                        best_breaks = new_breaks
                        best_q = new_q
                        if best_breaks == self.count_breakpoints(Q) - 2:
                            break
                if best_q and best_breaks == self.count_breakpoints(Q) - 2:
                    break
            if best_q:
                Q = best_q
                reversals += 1
                steps.append(Q[1:-1])
            else:
                print("Stuck: no reversal reduces breakpoints.")
                return reversals, steps
        return reversals, steps

class TwoBreakSort(GenomeRearrangementBase):
    def cycle_to_chromosome(self, nodes):
        chromosome = []
        for j in range(len(nodes) // 2):
            if nodes[2 * j] < nodes[2 * j + 1]:
                chromosome.append(nodes[2 * j + 1] // 2)
            else:
                chromosome.append(-nodes[2 * j] // 2)
        return chromosome
    
    def chromosome_to_cycle(self, chromosome):
        nodes = [0] * (2 * len(chromosome))
        for j in range(len(chromosome)):
            i = chromosome[j]
            if i > 0:
                nodes[2 * j] = 2 * i - 1
                nodes[2 * j + 1] = 2 * i
            else:
                nodes[2 * j] = -2 * i
                nodes[2 * j + 1] = -2 * i - 1
        return nodes
    
    def colored_edges(self, genome):
        edges = set()
        for chromosome in genome:
            nodes = self.chromosome_to_cycle(chromosome)
            nodes.append(nodes[0])
            for j in range(len(chromosome)):
                edges.add((nodes[2 * j + 1], nodes[2 * j + 2]))
        return edges
    
    def two_break_on_genome_graph(self, edges, i0, i1, j0, j1):
        edges.discard((i0, i1))
        edges.discard((i1, i0))
        edges.discard((j0, j1))
        edges.discard((j1, j0))
        edges.add((i0, j0))
        edges.add((i1, j1))
        return edges
    
    def find_and_merge(self, elements):
        parent = {x: x for x in elements}
        rank = {x: 0 for x in elements}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def merge(x, y):
            x_root = find(x)
            y_root = find(y)
            if x_root == y_root:
                return
            if rank[x_root] > rank[y_root]:
                parent[y_root] = x_root
            else:
                parent[x_root] = y_root
                if rank[x_root] == rank[y_root]:
                    rank[y_root] += 1
        
        return find, merge
    
    def group_nodes(self, edges):
        elements = set()
        for a, b in edges:
            elements.update([a, b])
            elements.update([a + 1 if a % 2 else a - 1])
            elements.update([b + 1 if b % 2 else b - 1])
        find, merge = self.find_and_merge(elements)
        for a, b in edges:
            merge(a, b)
            merge(a, a + 1 if a % 2 else a - 1)
            merge(b, b + 1 if b % 2 else b - 1)
        nodes_id = {x: find(x) for x in elements}
        return nodes_id
    
    def build_edge_dict(self, edges, nodes_id):
        edge_dict = dict()
        for e in edges:
            id = nodes_id[e[0]]
            if id not in edge_dict:
                edge_dict[id] = dict()
            edge_dict[id][e[0]] = e[1]
            edge_dict[id][e[1]] = e[0]
        return edge_dict
    
    def two_break_on_genome(self, genome, i0, i1, j0, j1):
        edges = self.two_break_on_genome_graph(self.colored_edges(genome), i0, i1, j0, j1)
        nodes_id = self.group_nodes(edges)
        edge_dict = self.build_edge_dict(edges, nodes_id)
        nodes_dict = dict()
        for id, edge in edge_dict.items():
            nodes_dict[id] = []
            curr0 = list(edge)[0]
            while edge:
                nodes_dict[id].append(curr0)
                if curr0 % 2 == 1:
                    curr1 = curr0 + 1
                else:
                    curr1 = curr0 - 1
                nodes_dict[id].append(curr1)
                new_node = edge[curr1]
                del edge[curr0]
                del edge[curr1]
                curr0 = new_node
        new_genome = []
        for nodes in nodes_dict.values():
            new_genome.append(self.cycle_to_chromosome(nodes))
        new_genome.sort(key=lambda x: abs(x[0]))
        return new_genome
    
    def edge_from_non_trivial_cycle(self, edges, red_edges, blue_edges, blocks):
        elements = set()
        for a, b in edges:
            elements.update([a, b])
        find, merge = self.find_and_merge(elements)
        for a, b in edges:
            merge(a, b)
        nodes_id = {}
        nodes_sets = set()
        for a, b in edges:
            root = find(a)
            nodes_id[a] = root
            nodes_id[b] = root
            nodes_sets.add(root)
        cycles = len(nodes_sets)
        has_non_trivial_cycle = cycles != blocks
        removed = []
        if has_non_trivial_cycle:
            edge = None
            edge_dict = {}
            red_edge_dict = {}
            for a, b in edges:
                cid = nodes_id[a]
                edge_dict.setdefault(cid, {})[a] = b
                edge_dict[cid][b] = a
                if (a, b) in red_edges or (b, a) in red_edges:
                    red_edge_dict.setdefault(cid, {})[a] = b
                    red_edge_dict[cid][b] = a
                if edge is None and len(edge_dict[cid]) > 2 and (a, b) in blue_edges:
                    edge = (a, b)
                    edge_id = cid
            if edge:
                removed.append((edge[0], red_edge_dict[edge_id][edge[0]]))
                removed.append((edge[1], red_edge_dict[edge_id][edge[1]]))
        return has_non_trivial_cycle, removed
    
    def shortest_rearrangement_scenario(self, P, Q):
        blocks = sum(len(chrom) for chrom in P)
        result = [P]
        red_edges = self.colored_edges(P)
        blue_edges = self.colored_edges(Q)
        breakpoint_graph = red_edges.union(blue_edges)
        has_non_trivial_cycle, removed = self.edge_from_non_trivial_cycle(breakpoint_graph, red_edges, blue_edges, blocks)
        while has_non_trivial_cycle:
            red_edges = self.two_break_on_genome_graph(red_edges, removed[0][0], removed[0][1], removed[1][0], removed[1][1])
            breakpoint_graph = red_edges.union(blue_edges)
            P = self.two_break_on_genome(P, removed[0][0], removed[0][1], removed[1][0], removed[1][1])
            has_non_trivial_cycle, removed = self.edge_from_non_trivial_cycle(breakpoint_graph, red_edges, blue_edges, blocks)
            result.append(P)
        return result
    
    def run(self, P, Q):
        steps = self.shortest_rearrangement_scenario(P, Q)
        distance = len(steps) - 1
        return distance, steps
    
    def format_genome(self, genome):
        return ''.join(['[' + ' '.join(f"{'+' if x > 0 else ''}{x}" for x in chrom) + ']' for chrom in genome])