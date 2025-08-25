from abc import ABC
from collections import defaultdict

class OFBase(ABC):
    def reverse_complement(self, sequence):
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        return ''.join(complement[base] for base in reversed(sequence))

    def hamming_distance(self, s1, s2):
        return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

    def generate_neighbors(self, pattern, d):
        if d == 0:
            return {pattern}
        if len(pattern) == 0:
            return {''}
        neighborhood = set()
        suffix_neighbors = self.generate_neighbors(pattern[1:], d)
        for text in suffix_neighbors:
            if self.hamming_distance(pattern[1:], text) < d:
                for nucleotide in "ACGT":
                    neighborhood.add(nucleotide + text)
            else:
                neighborhood.add(pattern[0] + text)
        return neighborhood

    def count_kmer_occurrences(self, text, k):
        counts = defaultdict(int)
        for i in range(len(text) - k + 1):
            kmer = text[i:i+k]
            counts[kmer] += 1
        return counts

class GCSkews(OFBase):
    def run(self, text):
        skew = 0
        min_skew = (float('inf'), -1)
        for i, base in enumerate(text):
            if base == 'G':
                skew += 1
            elif base == 'C':
                skew -= 1
            if skew < min_skew[0]:
                min_skew = (skew, i)
        return min_skew[1]

class FrequentKmers(OFBase):
    def run(self, text, k, d):
        counts = self.count_kmer_occurrences(text, k)
        frequent_patterns = defaultdict(int)
        processed = set()
        for kmer in counts:
            if kmer in processed:
                continue
            neighborhood = self.generate_neighbors(kmer, d)
            total_count = 0
            for approx_kmer in neighborhood:
                rev_comp = self.reverse_complement(approx_kmer)
                total_count += counts.get(approx_kmer, 0)
                if rev_comp != approx_kmer:
                    total_count += counts.get(rev_comp, 0)
                processed.add(approx_kmer)
                processed.add(rev_comp)
            frequent_patterns[kmer] = total_count
        if not frequent_patterns:
            return []
        max_count = max(frequent_patterns.values())
        return [(pattern, count) for pattern, count in frequent_patterns.items() if count == max_count]