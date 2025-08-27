import numpy as np
import heapq
import math

class ORFBase:
    base_to_int_map = {'A':0,'C':1,'G':2,'T':3}

    codon_usage = {
        'TTT': {'aa': 'F', 'freq': 0.64}, 'TTC': {'aa': 'F', 'freq': 0.36},
        'TTA': {'aa': 'L', 'freq': 0.18}, 'TTG': {'aa': 'L', 'freq': 0.13},
        'CTT': {'aa': 'L', 'freq': 0.15}, 'CTC': {'aa': 'L', 'freq': 0.10},
        'CTA': {'aa': 'L', 'freq': 0.06}, 'CTG': {'aa': 'L', 'freq': 0.38},
        'ATT': {'aa': 'I', 'freq': 0.47}, 'ATC': {'aa': 'I', 'freq': 0.31},
        'ATA': {'aa': 'I', 'freq': 0.21}, 'ATG': {'aa': 'M', 'freq': 1.00},
        'GTT': {'aa': 'V', 'freq': 0.32}, 'GTC': {'aa': 'V', 'freq': 0.19},
        'GTA': {'aa': 'V', 'freq': 0.19}, 'GTG': {'aa': 'V', 'freq': 0.29},
        'TCT': {'aa': 'S', 'freq': 0.18}, 'TCC': {'aa': 'S', 'freq': 0.14},
        'TCA': {'aa': 'S', 'freq': 0.18}, 'TCG': {'aa': 'S', 'freq': 0.11},
        'CCT': {'aa': 'P', 'freq': 0.24}, 'CCC': {'aa': 'P', 'freq': 0.16},
        'CCA': {'aa': 'P', 'freq': 0.23}, 'CCG': {'aa': 'P', 'freq': 0.37},
        'ACT': {'aa': 'T', 'freq': 0.22}, 'ACC': {'aa': 'T', 'freq': 0.31},
        'ACA': {'aa': 'T', 'freq': 0.25}, 'ACG': {'aa': 'T', 'freq': 0.22},
        'GCT': {'aa': 'A', 'freq': 0.22}, 'GCC': {'aa': 'A', 'freq': 0.26},
        'GCA': {'aa': 'A', 'freq': 0.27}, 'GCG': {'aa': 'A', 'freq': 0.25},
        'TAT': {'aa': 'Y', 'freq': 0.65}, 'TAC': {'aa': 'Y', 'freq': 0.35},
        'TAA': {'aa': '*', 'freq': 0.58}, 'TAG': {'aa': '*', 'freq': 0.09},
        'CAT': {'aa': 'H', 'freq': 0.63}, 'CAC': {'aa': 'H', 'freq': 0.37},
        'CAA': {'aa': 'Q', 'freq': 0.35}, 'CAG': {'aa': 'Q', 'freq': 0.65},
        'AAT': {'aa': 'N', 'freq': 0.59}, 'AAC': {'aa': 'N', 'freq': 0.41},
        'AAA': {'aa': 'K', 'freq': 0.71}, 'AAG': {'aa': 'K', 'freq': 0.29},
        'GAT': {'aa': 'D', 'freq': 0.65}, 'GAC': {'aa': 'D', 'freq': 0.35},
        'GAA': {'aa': 'E', 'freq': 0.64}, 'GAG': {'aa': 'E', 'freq': 0.36},
        'TGT': {'aa': 'C', 'freq': 0.52}, 'TGC': {'aa': 'C', 'freq': 0.48},
        'TGA': {'aa': '*', 'freq': 0.33}, 'TGG': {'aa': 'W', 'freq': 1.00},
        'CGT': {'aa': 'R', 'freq': 0.30}, 'CGC': {'aa': 'R', 'freq': 0.26},
        'CGA': {'aa': 'R', 'freq': 0.09}, 'CGG': {'aa': 'R', 'freq': 0.15},
        'AGT': {'aa': 'S', 'freq': 0.18}, 'AGC': {'aa': 'S', 'freq': 0.20},
        'AGA': {'aa': 'R', 'freq': 0.13}, 'AGG': {'aa': 'R', 'freq': 0.07},
        'GGT': {'aa': 'G', 'freq': 0.34}, 'GGC': {'aa': 'G', 'freq': 0.29},
        'GGA': {'aa': 'G', 'freq': 0.19}, 'GGG': {'aa': 'G', 'freq': 0.18}
    }

    def reverse_complement(self, seq):
        complement = {'A':'T','T':'A','C':'G','G':'C'}
        return ''.join(complement[b] for b in reversed(seq))

class FirstPassORF(ORFBase):
    def genome_to_vectors(self, genome, rev_genome, codon_to_vector):
        frames = []
        for seq in [genome, rev_genome]:
            for offset in range(3):
                codons = np.array([codon_to_vector[seq[i:i+3]] for i in range(offset, len(seq)-2, 3)])
                frames.append(codons)
        return frames
    
    def find_start_stop_codons(self, frames, codon_to_vector):
        start_codons = {codon_to_vector[c] for c, b in self.codon_usage.items() if b["aa"] == "M" or c in ['GTG','TTG']}
        stop_codons = {codon_to_vector[c] for c, b in self.codon_usage.items() if b["aa"] == "*"}
        orfs_per_frame = [[] for _ in range(6)]
        for idx, frame in enumerate(frames):
            frame_len = len(frame)
            start_mask = np.isin(frame, list(start_codons))
            stop_mask = np.isin(frame, list(stop_codons))
            start_positions = np.flatnonzero(start_mask)
            stop_positions = np.flatnonzero(stop_mask)
            stop_positions_extended = np.concatenate([stop_positions, stop_positions + frame_len])
            stop_idx = 0
            for start in start_positions:
                while stop_idx < len(stop_positions_extended) and stop_positions_extended[stop_idx] <= start:
                    stop_idx += 1
                if stop_idx < len(stop_positions_extended):
                    stop = stop_positions_extended[stop_idx] % frame_len
                    orfs_per_frame[idx].append((start, stop))
        return orfs_per_frame
    
    def score_orf(self, orf_seq):
        n = len(orf_seq) // 3
        codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
        freqs = np.array([self.codon_usage.get(c, {}).get('freq', 0.0) for c in codons])
        aa_flags = np.array([self.codon_usage.get(c, {}).get('aa', '*') != '*' for c in codons])
        eps = 1e-8
        freqs = np.where(aa_flags, freqs + eps, eps)
        return np.log(freqs).sum() / n
    
    def build_leaderboard(self, orfs, genome, min_size, L):
        leaderboard = []
        for orf_list in orfs:
            for start_idx, stop_idx in orf_list:
                nt_start = start_idx * 3
                nt_stop = stop_idx * 3 + 3
                orf_seq = genome[nt_start:nt_stop]
                orf_len = len(orf_seq) // 3
                if orf_len < min_size:
                    continue
                score = self.score_orf(orf_seq)
                if len(leaderboard) < L:
                    heapq.heappush(leaderboard, (score, {"seq": orf_seq, "score": score}))
                else:
                    if score > leaderboard[0][0]:
                        heapq.heappushpop(leaderboard, (score, {"seq": orf_seq, "score": score}))
        return [item[1] for item in sorted(leaderboard, key=lambda x: x[0], reverse=True)]

class SecondPassORF(ORFBase):
    def build_2nd_markov(self, leaderboard, codon_to_vector):
        transition_counts = np.zeros((64, 64, 64))
        for entry in leaderboard:
            seq = entry["seq"]
            for i in range(0, len(seq) - 9, 3):
                c1 = codon_to_vector[seq[i:i+3]]
                c2 = codon_to_vector[seq[i+3:i+6]]
                c3 = codon_to_vector[seq[i+6:i+9]]
                transition_counts[c1, c2, c3] += 1
        transition_probs = transition_counts + 1e-8
        transition_probs /= transition_probs.sum(axis=2, keepdims=True)
        return transition_probs

    def markov_score(self, orf, codon_to_vector, transition_probs):
        log_prob = 0.0
        transitions = 0
        for i in range(0, len(orf) - 9, 3):
            c1 = codon_to_vector[orf[i:i+3]]
            c2 = codon_to_vector[orf[i+3:i+6]]
            c3 = codon_to_vector[orf[i+6:i+9]]
            log_prob += math.log(transition_probs[c1, c2, c3])
            transitions += 1
        return log_prob / max(transitions, 1)

    def find_orfs(self, orfs, genome, rev_genome, codon_to_vector, min_size, t, transition_probs):
        n = len(genome)
        new_orfs = []
        for frame_idx, orf_list in enumerate(orfs):
            if frame_idx < 3:
                strand = "+"
                frame_number = frame_idx + 1
                for start_idx, stop_idx in orf_list:
                    nt_start = start_idx * 3 + frame_idx
                    nt_stop = stop_idx * 3 + frame_idx + 3
                    orf_seq = genome[nt_start:nt_stop]
                    orf_len = len(orf_seq) // 3
                    if orf_len < min_size:
                        continue
                    score = self.markov_score(orf_seq, codon_to_vector, transition_probs)
                    if score < t:
                        continue
                    new_orfs.append({
                        "seq": orf_seq,
                        "len": orf_len,
                        "start": nt_start,
                        "end": nt_stop,
                        "strand": strand,
                        "frame": frame_number,
                        "score": score
                    })
            else:
                strand = "-"
                frame_number = frame_idx - 2
                for start_idx, stop_idx in orf_list:
                    nt_stop = n - start_idx * 3 - frame_number
                    nt_start = n - stop_idx * 3 - frame_number - 3
                    orf_seq = rev_genome[start_idx*3 + (frame_idx-3): stop_idx*3 + (frame_idx-3) + 3]
                    orf_len = len(orf_seq) // 3
                    if orf_len < min_size:
                        continue
                    score = self.markov_score(orf_seq, codon_to_vector, transition_probs)
                    if score < t:
                        continue
                    new_orfs.append({
                        "seq": orf_seq,
                        "len": orf_len,
                        "start": nt_start,
                        "end": nt_stop,
                        "strand": strand,
                        "frame": frame_number,
                        "score": score
                    })
        return sorted(new_orfs, key=lambda x: x["start"])

    def orf_overlap(self, orfs, max_overlap):
        if max_overlap is None:
            return orfs
        orfs_sorted = sorted(orfs, key=lambda x: x["score"], reverse=True)
        filtered_orfs = []
        used_regions = []
        for orf in orfs_sorted:
            overlaps = 0
            for region in used_regions:
                if not (orf["end"] <= region[0] or orf["start"] >= region[1]):
                    overlaps += 1
                    if overlaps >= max_overlap:
                        break
            if overlaps < max_overlap:
                filtered_orfs.append(orf)
                used_regions.append((orf["start"], orf["end"]))
        return sorted(filtered_orfs, key=lambda x: x["start"])
    
class TwoPassORF(FirstPassORF, SecondPassORF):
    def first_scan(self, genome, rev_genome, codon_to_vector, min_size, L):
        frames = self.genome_to_vectors(genome, rev_genome, codon_to_vector)
        orfs = self.find_start_stop_codons(frames, codon_to_vector)
        leaderboard = self.build_leaderboard(orfs, genome, min_size, L)
        return leaderboard, orfs
    
    def second_scan(self, genome, rev_genome, codon_to_vector, min_size, max_overlap, t, leaderboard, unfiltered):
        transitions = self.build_2nd_markov(leaderboard, codon_to_vector)
        orfs = self.find_orfs(unfiltered, genome, rev_genome, codon_to_vector, min_size, t, transitions)
        filtered = self.orf_overlap(orfs, max_overlap)
        return filtered
    
    def two_pass(self, genome, min_size, max_overlap, t, L):
        codon_to_vector = {c: self.base_to_int_map[c[0]]*16 + self.base_to_int_map[c[1]]*4 + self.base_to_int_map[c[2]] for c in self.codon_usage.keys()}
        rev_genome = self.reverse_complement(genome)
        leaderboard, unfiltered = self.first_scan(genome, rev_genome, codon_to_vector, min_size, L)
        orfs = self.second_scan(genome, rev_genome, codon_to_vector, min_size, max_overlap, t, leaderboard, unfiltered)
        return orfs