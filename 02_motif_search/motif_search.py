import random

class MotifSearch:
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
    
    def random_kmer(self, dna, k):
        start = random.randint(0, len(dna) - k)
        return dna[start:start + k]

class GreedyMotifSearch(MotifSearch):
    def run(self, sequences, k, t, n=None):
        best_motifs = [dna[:k] for dna in sequences]
        first_string = sequences[0]
        for i in range(len(first_string) - k + 1):
            motifs = [first_string[i:i + k]]
            for j in range(1, t):
                profile = self.create_profile_matrix(motifs)
                next_motif = self.most_probable_kmer(sequences[j], k, profile)
                motifs.append(next_motif)
            if self.score(motifs) < self.score(best_motifs):
                best_motifs = motifs
        return best_motifs

class RandomMotifSearch(MotifSearch):
    def run(self, sequences, k, t, n):
        best_motifs = [self.random_kmer(dna, k) for dna in sequences]
        best_score = self.score(best_motifs)
        for _ in range(n):
            motifs = [self.random_kmer(dna, k) for dna in sequences]
            while True:
                profile = self.create_profile_matrix(motifs)
                motifs = [self.most_probable_kmer(dna, k, profile) for dna in sequences]
                current_score = self.score(motifs)
                if current_score < best_score:
                    best_score = current_score
                    best_motifs = motifs[:]
                else:
                    break
        return best_motifs

class GibbsSamplerMotifSearch(MotifSearch):
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

    def run(self, sequences, k, t, n):
        motifs = [self.random_kmer(dna, k) for dna in sequences]
        best_motifs = motifs[:]
        best_score = self.score(best_motifs)
        for _ in range(n):
            i = random.randint(0, t - 1)
            motifs_except_i = motifs[:i] + motifs[i+1:]
            profile = self.create_profile_matrix(motifs_except_i)
            motifs[i] = self.profile_random_kmer(sequences[i], k, profile)
            if self.score(motifs) < best_score:
                best_score = self.score(motifs)
                best_motifs = motifs[:]
        return best_motifs