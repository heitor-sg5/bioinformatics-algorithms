from collections import Counter
import random

amino_acid_mass = {
    'G': 57.0, 'A': 71.0, 'S': 87.0, 'P': 97.0, 'V': 99.0,
    'T': 101.1, 'C': 103.0, 'I': 113.1, 'L': 113.1, 'N': 114.1,
    'D': 115.1, 'K': 128.1, 'Q': 128.1, 'E': 129.1, 'M': 131.2,
    'O': 132.2, 'H': 137.1, 'F': 147.2, 'R': 156.2, 'Y': 163.2,
    'U': 168.1, 'W': 186.2
}

class TheoreticalSpectra:
    def cyclic_spectrum_with_error(self, peptide, p):
        prefix_mass = [0.0]
        for aa in peptide:
            prefix_mass.append(round(prefix_mass[-1] + amino_acid_mass[aa], 1))
        peptide_mass = prefix_mass[-1]
        spectrum = [0.0]
        n = len(peptide)
        for i in range(n):
            for j in range(i + 1, n + 1):
                sub_mass = round(prefix_mass[j] - prefix_mass[i], 1)
                spectrum.append(sub_mass)
                if i > 0 and j < n:
                    spectrum.append(round(peptide_mass - sub_mass, 1))
        n_remove = round(len(spectrum) * p)
        if n_remove == 0:
            return sorted(spectrum)
        indices_to_remove = set(random.sample(range(len(spectrum)), n_remove))
        noisy_spectrum = [mass for i, mass in enumerate(spectrum) if i not in indices_to_remove]
        return sorted(noisy_spectrum)

class Sequencing:
    def mass(self, peptide):
        return round(sum(amino_acid_mass[aa] for aa in peptide), 1)

    def parent_mass(self, spectrum):
        return round(max(spectrum), 1)

    def expand(self, peptides):
        expanded = []
        for peptide in peptides:
            for aa in amino_acid_mass.keys():
                expanded.append(peptide + aa)
        return expanded

    def linear_spectrum(self, peptide):
        prefix_mass = [0.0]
        for aa in peptide:
            prefix_mass.append(round(prefix_mass[-1] + amino_acid_mass[aa], 1))
        spectrum = [0.0]
        n = len(peptide)
        for i in range(n):
            for j in range(i + 1, n + 1):
                spectrum.append(round(prefix_mass[j] - prefix_mass[i], 1))
        return sorted(spectrum)

    def score(self, peptide, spectrum, cyclic=True, t=0.5):
        ts = TheoreticalSpectra()
        peptide_spectrum = ts.cyclic_spectrum_with_error(peptide, 0.0) if cyclic else self.linear_spectrum(peptide)
        peptide_spectrum = sorted(peptide_spectrum)
        spectrum = sorted(spectrum)
        score, i, j = 0, 0, 0
        while i < len(peptide_spectrum) and j < len(spectrum):
            diff = peptide_spectrum[i] - spectrum[j]
            if abs(diff) <= t:
                score += 1
                i += 1
                j += 1
            elif peptide_spectrum[i] < spectrum[j]:
                i += 1
            else:
                j += 1
        return score

class BranchAndBoundCyclopeptide(Sequencing):
    def consistent(self, peptide, spectrum):
        lin_spec = self.linear_spectrum(peptide)
        spec_counts = Counter(round(m, 1) for m in spectrum)
        lin_spec_counts = Counter(lin_spec)
        for mass_, count_ in lin_spec_counts.items():
            if spec_counts[mass_] < count_:
                return False
        return True

    def run(self, spectrum_str):
        spectrum = list(map(float, spectrum_str.strip().split()))
        spectrum = sorted(round(m, 1) for m in spectrum)
        peptides = [""]
        final_peptides = []
        parent_mass_val = self.parent_mass(spectrum)
        while peptides:
            peptides = self.expand(peptides)
            peptides_copy = peptides.copy()
            for peptide in peptides_copy:
                peptide_mass_val = self.mass(peptide)
                if abs(peptide_mass_val - parent_mass_val) < 1e-6:
                    ts = TheoreticalSpectra()
                    if ts.cyclic_spectrum_with_error(peptide, 0.0) == spectrum:
                        final_peptides.append(peptide)
                    peptides.remove(peptide)
                elif not self.consistent(peptide, spectrum):
                    peptides.remove(peptide)
        return final_peptides[0] if final_peptides else "No peptide found"

class LeaderboardAndConvolutionCyclopeptide(Sequencing):
    def __init__(self, n, m, t, c):
        self.n = n
        self.m = m
        self.t = t
        self.c = c

    def spectral_convolution(self, spectrum):
        convolution = []
        for i in range(len(spectrum)):
            for j in range(i+1, len(spectrum)):
                diff = spectrum[j] - spectrum[i]
                if 57 <= diff <= 200:
                    convolution.append(round(diff, 1))
        counts = Counter(convolution)
        if not counts:
            return []
        most_common = counts.most_common()
        result = []
        last_count = None
        for mass, count in most_common:
            if len(result) < self.m:
                result.append(mass)
                last_count = count
            elif count == last_count:
                result.append(mass)
            else:
                break
        return sorted(result)

    def filter_amino_acids_by_mass(self, allowed_masses):
        filtered = {}
        for aa, mass in amino_acid_mass.items():
            if any(abs(mass - m) <= self.t for m in allowed_masses):
                filtered[aa] = mass
        return filtered if filtered else amino_acid_mass

    def trim(self, leaderboard, spectrum):
        scored = [(peptide, self.score(peptide, spectrum, cyclic=False)) for peptide in leaderboard]
        scored.sort(key=lambda x: x[1], reverse=True)
        if len(scored) <= self.n:
            return [p for p, s in scored]
        cutoff_score = scored[self.n - 1][1]
        return [p for p, s in scored if s >= cutoff_score]

    def run(self, spectrum_str):
        spectrum = list(map(float, spectrum_str.strip().split()))
        spectrum = sorted(s - self.c for s in spectrum)
        parent = self.parent_mass(spectrum)
        allowed_masses = self.spectral_convolution(spectrum)
        filtered_amino_acid_mass = self.filter_amino_acids_by_mass(allowed_masses)
        leaderboard = [""]
        leader_peptide = ""
        while leaderboard:
            leaderboard = self.expand(leaderboard)
            leaderboard_copy = leaderboard.copy()
            for peptide in leaderboard_copy:
                peptide_mass_val = self.mass(peptide)
                if abs(peptide_mass_val - parent) <= self.t:
                    if self.score(peptide, spectrum) > self.score(leader_peptide, spectrum):
                        leader_peptide = peptide
                elif peptide_mass_val > parent + self.t:
                    leaderboard.remove(peptide)
            leaderboard = self.trim(leaderboard, spectrum)
        return leader_peptide