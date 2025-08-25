from collections import defaultdict
from abc import ABC

class PatternMatchingBase(ABC):
    def burrows_wheeler_transform(self, text):
        rotations = [text[i:] + text[:i] for i in range(len(text))]
        rotations_sorted = sorted(rotations)
        return ''.join(rotation[-1] for rotation in rotations_sorted)
    
    def build_suffix_array(self, text):
        return sorted(range(len(text)), key=lambda i: text[i:])
    
    def build_checkpoints(self, bwt, step=5):
        counts = defaultdict(list)
        total_counts = defaultdict(int)
        chars = set(bwt)
        for c in chars:
            counts[c] = []
        for i in range(len(bwt)):
            char = bwt[i]
            total_counts[char] += 1
            if i % step == 0:
                for c in chars:
                    counts[c].append((i, total_counts[c]))
        if (len(bwt) - 1) % step != 0:
            for c in chars:
                counts[c].append((len(bwt), total_counts[c]))
        return counts, step
    
    def count_symbol(self, checkpoints, bwt, symbol, pos, step):
        if pos == 0 or symbol not in checkpoints:
            return 0
        idx = pos // step
        idx_checkpoint = max(0, min(idx - 1, len(checkpoints[symbol]) - 1))
        checkpoint_pos, checkpoint_count = checkpoints[symbol][idx_checkpoint]
        count = checkpoint_count
        for i in range(checkpoint_pos, pos):
            if i >= len(bwt):
                break
            if bwt[i] == symbol:
                count += 1
        return count

class BWTMatching(PatternMatchingBase):
    def run(self, text, pattern):
        bwt = self.burrows_wheeler_transform(text)
        suffix_array = self.build_suffix_array(text)
        checkpoints, step = self.build_checkpoints(bwt)
        first_col = ''.join(sorted(bwt))
        first_occurrence = {}
        for i, char in enumerate(first_col):
            if char not in first_occurrence:
                first_occurrence[char] = i
        top = 0
        bottom = len(bwt) - 1
        while top <= bottom:
            if pattern:
                symbol = pattern[-1]
                pattern = pattern[:-1]
                if symbol not in first_occurrence:
                    return [], 0
                top = first_occurrence[symbol] + self.count_symbol(checkpoints, bwt, symbol, top, step)
                bottom = first_occurrence[symbol] + self.count_symbol(checkpoints, bwt, symbol, bottom + 1, step) - 1
                if top > bottom:
                    return [], 0
            else:
                positions = sorted(suffix_array[top:bottom + 1])
                return positions, len(positions)
        return [], 0

class SuffixArrayMatching(PatternMatchingBase):
    def run(self, text, pattern):
        sa = self.build_suffix_array(text)
        n = len(text)
        min_index = 0
        max_index = n - 1
        while min_index <= max_index:
            mid_index = (min_index + max_index) // 2
            suffix = text[sa[mid_index]:]
            if pattern > suffix:
                min_index = mid_index + 1
            else:
                max_index = mid_index - 1
        first = min_index
        if not text[sa[first]:].startswith(pattern):
            return [], 0
        min_index = first
        max_index = n - 1
        while min_index <= max_index:
            mid_index = (min_index + max_index) // 2
            suffix = text[sa[mid_index]:]
            if suffix.startswith(pattern):
                min_index = mid_index + 1
            else:
                max_index = mid_index - 1
        last = max_index
        positions = sorted([sa[i] + 1 for i in range(first, last + 1)])
        return positions, len(positions)

class PrefixTrieMatching(PatternMatchingBase):
    def run(self, text, patterns):
        trie = self.construct_trie(patterns)
        result = {pattern: ([], 0) for pattern in patterns}

        for i in range(len(text)):
            matches = self.prefix_trie_matching(text[i:], trie)
            for match in matches:
                result[match][0].append(i + 1)
                result[match] = (result[match][0], len(result[match][0]))
        for pattern in result:
            positions, count = result[pattern]
            result[pattern] = (sorted(positions), count)
        return result

    def construct_trie(self, patterns):
        trie = [{}]
        self.pattern_end_nodes = {}
        for pattern in patterns:
            current_node = 0
            for symbol in pattern:
                if symbol in trie[current_node]:
                    current_node = trie[current_node][symbol]
                else:
                    trie.append({})
                    new_node = len(trie) - 1
                    trie[current_node][symbol] = new_node
                    current_node = new_node
            self.pattern_end_nodes[current_node] = pattern
        return trie

    def prefix_trie_matching(self, text, trie):
        matches = []
        index = 0
        v = 0
        while True:
            if v in self.pattern_end_nodes:
                matches.append(self.pattern_end_nodes[v])
            if index >= len(text):
                break
            symbol = text[index]
            if symbol in trie[v]:
                v = trie[v][symbol]
                index += 1
            else:
                break
        return matches

class ApproximatePatternMatching(PatternMatchingBase):
    def __init__(self, d):
        self.d = d
    
    def run(self, text, patterns):
        bwt = self.burrows_wheeler_transform(text)
        suffix_array = self.build_suffix_array(text)
        first_col = ''.join(sorted(bwt))
        first_occurrence = {}
        for i, char in enumerate(first_col):
            if char not in first_occurrence:
                first_occurrence[char] = i
        checkpoints, step = self.build_checkpoints(bwt)
        
        result = {}
        for pattern in patterns:
            positions = self.approximate_bw_matching(bwt, pattern, suffix_array, first_occurrence, checkpoints, step)
            result[pattern] = (sorted(positions), len(positions))
        return result
    
    def approximate_bw_matching(self, bwt, pattern, suffix_array, first_occurrence, checkpoints, step):
        def recursive_match(pattern, top, bottom, mismatches_left):
            if top > bottom:
                return set()
            if not pattern:
                return set(suffix_array[top:bottom+1])
            symbol = pattern[-1]
            pattern_rest = pattern[:-1]
            matches = set()
            if symbol in first_occurrence:
                new_top = first_occurrence[symbol] + self.count_symbol(checkpoints, bwt, symbol, top, step)
                new_bottom = first_occurrence[symbol] + self.count_symbol(checkpoints, bwt, symbol, bottom+1, step) - 1
                if new_top <= new_bottom:
                    matches |= recursive_match(pattern_rest, new_top, new_bottom, mismatches_left)
            if mismatches_left > 0:
                for alt_symbol in first_occurrence.keys():
                    if alt_symbol == symbol:
                        continue
                    new_top = first_occurrence[alt_symbol] + self.count_symbol(checkpoints, bwt, alt_symbol, top, step)
                    new_bottom = first_occurrence[alt_symbol] + self.count_symbol(checkpoints, bwt, alt_symbol, bottom+1, step) - 1
                    if new_top <= new_bottom:
                        matches |= recursive_match(pattern_rest, new_top, new_bottom, mismatches_left - 1)
            return matches
        
        return list(recursive_match(pattern, 0, len(bwt)-1, self.d))

class SuffixTree(PatternMatchingBase):
    class Node:
        def __init__(self):
            self.children = {}
            self.indexes = []
    
    def run(self, text):
        root = self.suffix_tree_construction(text)
        compressed_root = self.suffix_tree(root)
        longest_substr, length = self.find_longest_repeated_substring(compressed_root)
        return longest_substr, length
    
    def suffix_tree_construction(self, text):
        root = self.Node()
        for i in range(len(text)):
            current_node = root
            for j in range(i, len(text)):
                symbol = text[j]
                if symbol in current_node.children:
                    current_node = current_node.children[symbol]
                else:
                    new_node = self.Node()
                    current_node.children[symbol] = new_node
                    current_node = new_node
            current_node.indexes.append(i)
        return root
    
    def suffix_tree(self, node):
        compressed_node = self.Node()
        for edge_label, child in node.children.items():
            current_label = edge_label
            current_child = child
            while len(current_child.children) == 1 and not current_child.indexes:
                (next_label, next_child), = current_child.children.items()
                current_label += next_label
                current_child = next_child
            compressed_child = self.suffix_tree(current_child)
            compressed_node.children[current_label] = compressed_child
            compressed_child.indexes = current_child.indexes
        return compressed_node
    
    def find_longest_repeated_substring(self, node, path=""):
        if len(node.children) == 0:
            return "", 0
        longest_substring = ""
        max_length = 0
        if len(node.children) > 1 or len(node.indexes) > 1:
            longest_substring = path
            max_length = len(path)
        for edge_label, child in node.children.items():
            sub_str, sub_len = self.find_longest_repeated_substring(child, path + edge_label)
            if sub_len > max_length:
                longest_substring = sub_str
                max_length = sub_len
        return longest_substring, max_length

class GeneralizedSuffixTree(PatternMatchingBase):
    class Node:
        def __init__(self):
            self.children = {}
            self.indexes = []
            self.color = None
            
    def run(self, text1, text2):
        root = self.build_generalized_suffix_tree(text1, text2)
        compressed_root = self.suffix_tree(root)
        self.color_tree(compressed_root)
        longest_shared, _ = self.find_longest_shared(compressed_root)
        shortest_non_shared, _ = self.find_shortest_non_shared(compressed_root)
        return longest_shared, shortest_non_shared
    
    def build_generalized_suffix_tree(self, text1, text2):
        root = self.Node()
        for i in range(len(text1)):
            self.insert_suffix(root, text1[i:], i, 1)
        for i in range(len(text2)):
            self.insert_suffix(root, text2[i:], i, 2)
        return root
    
    def insert_suffix(self, root, suffix, start_idx, source):
        node = root
        for char in suffix:
            if char not in node.children:
                node.children[char] = self.Node()
            node = node.children[char]
        node.indexes.append((start_idx, source))
    
    def suffix_tree(self, node):
        compressed = self.Node()
        compressed.indexes = node.indexes[:]
        for char, child in node.children.items():
            edge_label = char
            current_child = child
            while len(current_child.children) == 1 and not current_child.indexes:
                (next_char, next_node), = current_child.children.items()
                edge_label += next_char
                current_child = next_node
            compressed_child = self.suffix_tree(current_child)
            compressed.children[edge_label] = compressed_child
            compressed.indexes.extend(compressed_child.indexes)
        return compressed
    
    def color_tree(self, node):
        sources = set(src for _, src in node.indexes)
        for edge_label, child in node.children.items():
            child_sources = self.color_tree(child)
            sources.update(child_sources)
        if sources == {1}:
            node.color = 'blue'
        elif sources == {2}:
            node.color = 'red'
        else:
            node.color = 'purple'
        return sources
    
    def find_longest_shared(self, node, path="", best=("", 0)):
        if node.color == 'purple' and len(path) > best[1]:
            best = (path, len(path))
        for edge_label, child in node.children.items():
            best = self.find_longest_shared(child, path + edge_label, best)
        return best
    
    def find_shortest_non_shared(self, node, path="", best=None, terminals={"#", "$"}):
        if node.color in ('blue', 'red') and node.indexes and not any(t in path for t in terminals):
            if best is None or len(path) < best[1]:
                best = (path, len(path))
        for edge_label, child in node.children.items():
            best = self.find_shortest_non_shared(child, path + edge_label, best)
        return best