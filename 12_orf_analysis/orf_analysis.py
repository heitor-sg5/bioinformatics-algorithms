import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import random
import heapq
import math

class ORFBase:
    base_to_int_map = {'A':0,'C':1,'G':2,'T':3}

    codons = [
        'AAA','AAC','AAG','AAT','ACA','ACC','ACG','ACT',
        'AGA','AGC','AGG','AGT','ATA','ATC','ATG','ATT', 
        'CAA','CAC','CAG','CAT','CCA','CCC','CCG','CCT',
        'CGA','CGC','CGG','CGT','CTA','CTC','CTG','CTT',
        'GAA','GAC','GAG','GAT','GCA','GCC','GCG','GCT',
        'GGA','GGC','GGG','GGT','GTA','GTC','GTG','GTT',
        'TAA','TAC','TAG','TAT','TCA','TCC','TCG','TCT',
        'TGA','TGC','TGG','TGT','TTA','TTC','TTG','TTT'
    ]

    def reverse_complement(self, seq):
        complement = {'A':'T','T':'A','C':'G','G':'C'}
        return ''.join(complement[b] for b in reversed(seq))
    
    def build_2nd_markov(self, leaderboard, codon_to_vector):
        transition_counts = np.zeros((64, 64, 64))
        for entry in leaderboard:
            seq = entry["seq"]
            for i in range(0, len(seq) - 9, 3):
                c1 = codon_to_vector[seq[i:i+3]]
                c2 = codon_to_vector[seq[i+3:i+6]]
                c3 = codon_to_vector[seq[i+6:i+9]]
                transition_counts[c1, c2, c3] += 1
        alpha = 1.0  
        transition_counts += alpha
        transition_probs = transition_counts / transition_counts.sum(axis=2, keepdims=True)
        return transition_probs
    
    def build_1st_markov(self, intergenic_regions, codon_to_vector):
        transition_counts = np.zeros((64, 64))
        for entry in intergenic_regions:
            seq = entry["seq"]
            for i in range(0, len(seq) - 6, 3):
                c1 = codon_to_vector.get(seq[i:i+3])
                c2 = codon_to_vector.get(seq[i+3:i+6])
                transition_counts[c1, c2] += 1
        alpha = 1.0
        transition_counts += alpha
        transition_probs = transition_counts / transition_counts.sum(axis=1, keepdims=True)
        return transition_probs
    
    def build_base_markov(self, intergenic_regions):
        trans_counts = np.zeros((4, 4))
        for entry in intergenic_regions:
            seq = entry["seq"]
            for i in range(len(seq) - 1):
                b1, b2 = seq[i], seq[i+1]
                if b1 in self.base_to_int_map and b2 in self.base_to_int_map:
                    trans_counts[self.base_to_int_map[b1], self.base_to_int_map[b2]] += 1
        alpha = 1.0
        trans_counts += alpha
        trans_probs = trans_counts / trans_counts.sum(axis=1, keepdims=True)
        return trans_probs
        
    def build_pwm(self, orfs):
        upstreams = [orf["motif"] for orf in orfs]
        pwm = np.ones((4, 20))
        for seq in upstreams:
            for i, b in enumerate(seq):
                if b in self.base_to_int_map:
                    pwm[self.base_to_int_map[b], i] += 1
        pwm /= pwm.sum(axis=0, keepdims=True)
        return pwm

    def log_odds_score(self, orf, codon_to_vector, c_transitions, nc_transitions):
        log_odds = 0.0
        transitions = 0
        for i in range(0, len(orf) - 9, 3):
            c1 = codon_to_vector.get(orf[i:i+3])
            c2 = codon_to_vector.get(orf[i+3:i+6])
            c3 = codon_to_vector.get(orf[i+6:i+9])
            p_coding = c_transitions[c1, c2, c3]
            p_bg = nc_transitions[c2, c3]
            log_odds += math.log(p_coding / p_bg)
            transitions += 1
        return log_odds / max(transitions, 1)
    
    def pwm_score(self, upstream, pwm, transitions):
        score = 0.0
        if pwm is None:
            return score
        for i, base in enumerate(upstream):
            if i >= pwm.shape[1] or i >= transitions.shape[1]:
                break
            if base in self.base_to_int_map:
                base_idx = self.base_to_int_map[base]
                p = pwm[base_idx, i]
                q = transitions[base_idx, i]
            score += math.log((p + 1e-12) / (q + 1e-12))
        return score
    
class FirstPassORF(ORFBase):
    def genome_to_vectors(self, genome, rev_genome, codon_to_vector):
        frames = []
        for seq in [genome, rev_genome]:
            for offset in range(3):
                codons = np.array([codon_to_vector[seq[i:i+3]] for i in range(offset, len(seq)-2, 3)])
                frames.append(codons)
        return frames
    
    def find_start_stop_codons(self, frames, codon_to_vector):
        start_codons = {codon_to_vector[c] for c in ['GTG', 'TTG', 'ATG']}
        stop_codons = {codon_to_vector[c] for c in ['TGA', 'TAG', 'TAA']}
        orfs_per_frame = [[] for _ in range(6)]
        for idx, frame in enumerate(frames):
            start_mask = np.isin(frame, list(start_codons))
            stop_mask = np.isin(frame, list(stop_codons))
            start_positions = np.flatnonzero(start_mask)
            stop_positions = np.flatnonzero(stop_mask)
            stop_idx = 0
            for start in start_positions:
                while stop_idx < len(stop_positions) and stop_positions[stop_idx] <= start:
                    stop_idx += 1
                if stop_idx >= len(stop_positions):
                    continue
                stop = stop_positions[stop_idx]
                orfs_per_frame[idx].append((start, stop))
        return orfs_per_frame
    
    def extract_orfs_and_intergenic(self, orfs, genome, rev_genome, codon_to_vector, min_size):
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
                    if nt_start >= 20:
                        upstream = genome[nt_start - 20:nt_start]
                    else:
                        upstream = genome[0:nt_start]
                    new_orfs.append({
                        "seq": orf_seq,
                        "len": orf_len,
                        "start": nt_start,
                        "end": nt_stop,
                        "strand": strand,
                        "frame": frame_number,
                        "score": 0,
                        "upstream": upstream
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
                    if nt_start >= 20:
                        upstream = rev_genome[nt_start - 20:nt_start]
                    else:
                        upstream = rev_genome[0:nt_start]
                    new_orfs.append({
                        "seq": orf_seq,
                        "len": orf_len,
                        "start": nt_start,
                        "end": nt_stop,
                        "strand": strand,
                        "frame": frame_number,
                        "score": 0,
                        "upstream": upstream
                    })
        if not new_orfs:
            intergenic = [{"seq": genome, "score": 0}] if len(genome) >= 6 else []
            return new_orfs, intergenic
        new_orfs.sort(key=lambda x: x["start"])
        merged = []
        cur_s = new_orfs[0]["start"]
        cur_e = new_orfs[0]["end"]
        for e in new_orfs[1:]:
            if e["start"] <= cur_e:
                cur_e = max(cur_e, e["end"])
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = e["start"], e["end"]
        merged.append((cur_s, cur_e))
        intergenic = []
        prev = 0
        for s, e in merged:
            if s > prev:
                seq = genome[prev:s]
                if len(seq) >= 6:
                    intergenic.append({"seq": seq, "score": 0})
            prev = max(prev, e)
        if prev < n:
            seq = genome[prev:]
            if len(seq) >= 6:
                intergenic.append({"seq": seq, "score": 0})
        return new_orfs, intergenic

class SecondPassORF(ORFBase):
    def build_leaderboard(self, orfs, L, codon_to_vector, c_transitions, nc_transitions, pwm, base_transitions):
        leaderboard = []
        for orf in orfs:
            score = self.log_odds_score(orf['seq'], codon_to_vector, c_transitions, nc_transitions)
            score += self.pwm_score(orf['upstream'], pwm, base_transitions)
            orf['score'] = score
            if len(leaderboard) < L:
                heapq.heappush(leaderboard, (orf['score'], orf))
            else:
                if score > leaderboard[0][0]:
                  heapq.heappush(leaderboard, (orf['score'], orf))
        return [item[1] for item in sorted(leaderboard, key=lambda x: x[0], reverse=True)]

    def find_orfs(self, orfs, codon_to_vector, c_transitions, nc_transitions, pwm, base_transitions):
        for orf in orfs:
            score = self.log_odds_score(orf['seq'], codon_to_vector, c_transitions, nc_transitions)
            score += self.pwm_score(orf['upstream'], pwm, base_transitions)
            orf['score'] = score
        return orfs

    def orf_overlap(self, orfs, max_overlap, t_type):
        scores = np.array([orf['score'] for orf in orfs])
        if t_type == 0:
            mean = np.mean(scores)
            std = np.std(scores)
            t = mean + 1 * std
        else:
            t = np.percentile(scores, t_type)
        if max_overlap is None:
            return orfs
        orfs_sorted = sorted(orfs, key=lambda x: x["score"], reverse=True)
        filtered_orfs = []
        used_regions = []
        for orf in orfs_sorted:
            overlaps = 0
            if orf["score"] < t:
                continue
            for region in used_regions:
                if not (orf["end"] <= region[0] or orf["start"] >= region[1]):
                    overlaps += 1
                    if overlaps >= max_overlap:
                        break
            if overlaps < max_overlap:
                filtered_orfs.append(orf)
                used_regions.append((orf["start"], orf["end"]))
        return sorted(filtered_orfs, key=lambda x: x["start"])
    
class GibbsSamplerMotifSearch(ORFBase):
    def create_profile_matrix(self, motifs):
        k = len(motifs[0])
        profile = {nuc: [1] * k for nuc in "ACGT"}
        for motif in motifs:
            for i, nucleotide in enumerate(motif):
                profile[nucleotide][i] += 1
        for i in range(k):
            total = sum(profile[nuc][i] for nuc in "ACGT")
            for nuc in "ACGT":
                profile[nuc][i] /= total
        return profile

    def score(self, motifs):
        k = len(motifs[0])
        t = len(motifs)
        score = 0
        for i in range(k):
            counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
            for motif in motifs:
                counts[motif[i]] += 1
            max_count = max(counts.values())
            score += (t - max_count)
        return score

    def consensus(self, motifs):
        k = len(motifs[0])
        consensus = ""
        for i in range(k):
            counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
            for motif in motifs:
                counts[motif[i]] += 1
            consensus += max(counts, key=counts.get)
        return consensus

    def most_probable_kmer(self, text, k, profile):
        max_prob = -1
        best_kmer = text[0:k]
        for i in range(len(text) - k + 1):
            kmer = text[i:i + k]
            prob = 1
            for j, nucleotide in enumerate(kmer):
                prob *= profile[nucleotide][j]
            if prob > max_prob:
                max_prob = prob
                best_kmer = kmer
        return best_kmer
    
    def profile_random_kmer(self, text, k, profile):
        n = len(text)
        probabilities = []
        for i in range(n - k + 1):
            kmer = text[i:i+k]
            prob = 1
            for j, nucleotide in enumerate(kmer):
                prob *= profile[nucleotide][j]
            probabilities.append(prob)
        total_prob = sum(probabilities)
        if total_prob == 0:
            return text[random.randint(0, n - k): random.randint(0, n - k) + k]
        probabilities = [p / total_prob for p in probabilities]
        chosen_index = random.choices(range(n - k + 1), weights=probabilities)[0]
        return text[chosen_index:chosen_index + k]
    
    def random_kmer(self, dna, k):
        start = random.randint(0, len(dna) - k)
        return dna[start:start + k]

    def motif_search(self, orfs, k, t, n):
        upstreams = [orf["upstream"] for orf in orfs]
        motifs = [self.random_kmer(dna, k) for dna in upstreams]
        best_motifs = motifs[:]
        best_score = self.score(best_motifs)
        for _ in range(n):
            i = random.randint(0, t - 1)
            motifs_except_i = motifs[:i] + motifs[i+1:]
            profile = self.create_profile_matrix(motifs_except_i)
            motifs[i] = self.profile_random_kmer(upstreams[i], k, profile)
            if self.score(motifs) < best_score:
                best_score = self.score(motifs)
                best_motifs = motifs[:]
        for orf, motif in zip(orfs, best_motifs):
            orf["motif"] = motif
        return orfs

class TwoPassORF(FirstPassORF, SecondPassORF, GibbsSamplerMotifSearch):
    def first_scan(self, genome, rev_genome, codon_to_vector, min_size):
        frames = self.genome_to_vectors(genome, rev_genome, codon_to_vector)
        orfs = self.find_start_stop_codons(frames, codon_to_vector)
        filtered_orfs, intergenic_regions = self.extract_orfs_and_intergenic(orfs, genome, rev_genome, codon_to_vector, min_size)
        base_transitions = self.build_base_markov(intergenic_regions)
        c_transitions = self.build_2nd_markov(filtered_orfs, codon_to_vector)
        nc_transitions = self.build_1st_markov(intergenic_regions, codon_to_vector)
        return c_transitions, nc_transitions, filtered_orfs, base_transitions

    def second_scan(self, codon_to_vector, max_overlap, t, L, orfs, c_transitions, nc_transitions, base_transitions, k, max_iter=3):
        pwm = None
        leaderboard = self.build_leaderboard(orfs, L, codon_to_vector, c_transitions, nc_transitions, pwm, base_transitions)
        leaderboard_with_motif = self.motif_search(leaderboard, k, L, n=200)
        pwm = self.build_pwm(leaderboard_with_motif)
        prev_pass_set = None
        for _ in range(1, max_iter + 1):
            new_c_transitions = self.build_2nd_markov(leaderboard_with_motif, codon_to_vector)
            rescored_orfs = self.find_orfs(orfs, codon_to_vector, new_c_transitions, nc_transitions, pwm, base_transitions)
            leaderboard = self.build_leaderboard(rescored_orfs, L, codon_to_vector, new_c_transitions, nc_transitions, pwm, base_transitions)
            leaderboard_with_motif = self.motif_search(leaderboard, k, L, n=200)
            pwm = self.build_pwm(leaderboard_with_motif)
            overlap_orfs = self.orf_overlap(rescored_orfs, max_overlap, t)
            current_pass_set = set((o['start'], o['end'], o.get('strand', o.get('frame'))) for o in overlap_orfs)
            if prev_pass_set is not None and current_pass_set == prev_pass_set:
                break
            prev_pass_set = current_pass_set
            c_transitions = new_c_transitions
        final_filtered_orfs = self.motif_search(overlap_orfs, k, L, n=200)
        return final_filtered_orfs

    def two_pass(self, genome, min_size, max_overlap, t, L, k, max_iter=3):
        codon_to_vector = {c: self.base_to_int_map[c[0]]*16 + self.base_to_int_map[c[1]]*4 + self.base_to_int_map[c[2]] for c in self.codons}
        rev_genome = self.reverse_complement(genome)
        c_transitions, nc_transitions, orfs, base_transitions = self.first_scan(genome, rev_genome, codon_to_vector, min_size)
        new_orfs = self.second_scan(codon_to_vector, max_overlap, t, L, orfs, c_transitions, nc_transitions, base_transitions, k, max_iter=max_iter)
        return new_orfs

class Charts(ORFBase):
    def combined_plots(self, genome, orfs, scores):
        window_size = 10000
        genome_len = len(genome)
        density = [0] * (genome_len // window_size + 1)
        for orf in orfs:
            start_bin = orf['start'] // window_size
            end_bin = orf['end'] // window_size
            for b in range(start_bin, end_bin + 1):
                density[b] += 1
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(scores, bins=100, color="skyblue", edgecolor="black")
        axes[0].set_xlabel("Score")
        axes[0].set_ylabel("ORFs")
        axes[0].set_title("Score Distribution")
        axes[1].bar(range(len(density)), density, width=1.0, color="lightgreen", edgecolor="black")
        axes[1].set_xlabel(f"Position (per {window_size} bp)")
        axes[1].set_ylabel("ORFs")
        axes[1].set_title("ORF Density")
        plt.tight_layout()
        plt.show()

    def codon_freq(self, orfs):
        codon_counts = Counter()
        for orf in orfs:
            seq = orf['seq']
            for i in range(0, len(seq), 3):
                codon = seq[i:i+3]
                codon_counts[codon] += 1
        total_codons = sum(codon_counts.values())
        return {c: count/total_codons for c, count in codon_counts.items()}

    def summary_info(self, orfs, scores, lengths, frequencies, motifs):
        results = []
        results.append(f"---Summary---\n")
        results.append(f"Mean score: {np.mean(scores):.1f} | Min: {np.min(scores):.1f} | Max: {np.max(scores):.1f}\n")
        results.append(f"Mean length: {np.mean(lengths) * 3:.0f} ({np.mean(lengths):.0f}) | Min: {np.min(lengths) * 3:.0f} ({np.min(lengths)}) | Max: {np.max(lengths) * 3:.0f} ({np.max(lengths)})\n")
        pos = sum(1 for orf in orfs if orf['strand'] == '+')
        neg = sum(1 for orf in orfs if orf['strand'] == '-')
        f1 = sum(1 for orf in orfs if orf['frame'] == 1)
        f2 = sum(1 for orf in orfs if orf['frame'] == 2)
        f3 = sum(1 for orf in orfs if orf['frame'] == 3)
        results.append(f"Strand distribution (+/-): {pos}/{neg} | Frame ratio (1:2:3): {f1}:{f2}:{f3}\n")
        results.append(f"Codon frequencies:\n")
        line = []
        for i, codon in enumerate(self.codons, start=1):
            freq = frequencies.get(codon, 0)
            line.append(f"{codon}: {freq:.2f}")
            if i % 8 == 0:
                results.append(" | ".join(line)+ "\n")
                line = []
        if line:
            results.append(" | ".join(line)+ "\n")
        gibbs = GibbsSamplerMotifSearch()
        results.append(f"Consensus: {gibbs.consensus(motifs)}")
        results.append("\n")
        return results
    
    def display(self, orfs, genome):
        scores = [orf["score"] for orf in orfs]
        lengths = [orf["len"] for orf in orfs]
        motifs = [orf["motif"] for orf in orfs]
        self.combined_plots(genome, orfs, scores)
        freq = self.codon_freq(orfs)
        summary = self.summary_info(orfs, scores, lengths, freq, motifs)
        return summary