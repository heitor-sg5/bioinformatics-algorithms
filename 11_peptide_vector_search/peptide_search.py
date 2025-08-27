import numpy as np

class PeptideBase:
    amino_acid_mass = {
        'G': 57,  'A': 71,  'S': 87,  'P': 97,  'V': 99,
        'T': 101, 'C': 103, 'I': 113, 'L': 113, 'N': 114,
        'D': 115, 'K': 128, 'Q': 128, 'E': 129, 'M': 131,
        'O': 132, 'H': 137, 'F': 147, 'R': 156, 'Y': 163,
        'U': 168, 'W': 186
    }

    def score(self, peptide_vector, spectrum_vector):
        return sum(min(peptide_vector[i], spectrum_vector[i]) 
                for i in range(min(len(peptide_vector), len(spectrum_vector))))

    def peptide_to_vector(self, peptide):
        prefix_mass = [0]
        for aa in peptide:
            prefix_mass.append(prefix_mass[-1] + self.amino_acid_mass[aa])
        vector_len = prefix_mass[-1] + 1
        vector = [0] * vector_len
        n = len(peptide)
        for i in range(n):
            for j in range(i + 1, n + 1):
                mass = prefix_mass[j] - prefix_mass[i]
                vector[mass] += 1
        return vector

    @staticmethod
    def spectrum_to_vector(spectrum):
        max_mass = max(spectrum)
        vector = [0] * (max_mass + 1)
        for mass in spectrum:
            vector[mass] += 1
        return vector

class PeptideSearch(PeptideBase):
    def __init__(self, k, peptides, spectral_vectors):
        self.k = k
        self.peptides = peptides
        self.spectral_vectors = spectral_vectors
    
    def build_peptide_vectors(self, peptides):
        return {peptide: self.peptide_to_vector(peptide) for peptide in peptides}

    def peptide_identification(self, peptide_vectors, vector):
        max_score = -1
        best_peptide = None
        for peptide, peptide_vector in peptide_vectors.items():
            score = self.score(peptide_vector, vector)
            if score > max_score:
                max_score = score
                best_peptide = peptide
        return best_peptide, max_score
    
    def pms_search(self, peptide_vectors):
        pms = set()
        for vector in self.spectral_vectors:
            peptide, score = self.peptide_identification(peptide_vectors, vector)
            if score > self.k:
                pms.add((peptide, tuple(vector), score))
            else:
                pms.add(("No match above k.", tuple(vector), 0))
        return pms

    def run(self):
        peptide_vectors = self.build_peptide_vectors(self.peptides)
        pms = self.pms_search(peptide_vectors)
        return pms

class StatisticalSignificance(PeptideBase):
    def __init__(self, k, spectrum_vector):
        self.k = k
        self.spectrum_vector = spectrum_vector
        self.max_mass = len(spectrum_vector) - 1

    def probability_peptides_above_threshold(self):
        n_aa = len(self.amino_acid_mass)
        aa_masses = list(self.amino_acid_mass.values())
        dp = [{} for _ in range(self.max_mass + 2)]
        dp[0][0] = 1.0
        for mass in range(self.max_mass + 1):
            for score, prob in dp[mass].items():
                for aa_mass in aa_masses:
                    next_mass = mass + aa_mass
                    if next_mass > self.max_mass:
                        continue
                    increment = min(1, self.spectrum_vector[next_mass])
                    next_score = score + increment
                    dp_next = dp[next_mass]
                    dp_next[next_score] = dp_next.get(next_score, 0.0) + prob / n_aa
        total_prob = 0.0
        for mass_dict in dp:
            for score, prob in mass_dict.items():
                if score >= self.k:
                    total_prob += prob
        return min(total_prob, 1.0)

    def run(self):
        prob = self.probability_peptides_above_threshold()
        return prob

class SpectralAlignment(PeptideBase):
    def __init__(self, d, peptide, spectrum_vector):
        self.d = d
        self.peptide = peptide
        self.spectrum_vector = spectrum_vector

    def spectral_alignment_graph(self):
        diff = {}
        m_val = 0
        acceptable_rows = []
        for aa in self.peptide:
            m_val += self.amino_acid_mass[aa]
            acceptable_rows.append(m_val)
            diff[m_val] = self.amino_acid_mass[aa]
        delta = len(self.spectrum_vector) - m_val
        m_val += 1
        score = np.full((m_val, m_val + delta, self.d + 1), float('-inf'))
        score_array = [[["" for _ in range(self.d + 1)] for _ in range(m_val + delta)] for _ in range(m_val)]
        score[0][0][0] = 0
        for z in range(0, self.d + 1):
            for y in range(m_val + delta):
                for x in range(m_val):
                    if x not in acceptable_rows:
                        continue
                    c = []
                    if z != 0:
                        for j in range(1, y + 1):
                            c.append(score[x - diff[x]][j - 1][z - 1])
                    c.append(score[x - diff[x]][y - diff[x]][z])
                    arg_max = np.argmax(c)
                    if arg_max == len(c) - 1:
                        mstr = score_array[x - diff[x]][y - diff[x]][z]
                        mstr += "/" + str(y - diff[x])
                        score_array[x][y][z] = mstr
                    else:
                        mstr = score_array[x - diff[x]][arg_max][z - 1]
                        mstr += "/" + str(arg_max)
                        score_array[x][y][z] = mstr
                    m_max = max(c)
                    score[x][y][z] = self.spectrum_vector[y - 1] + m_max if y > 0 else m_max
        return m_val, m_val + delta, score, score_array
    
    def run(self):
        x_dim, y_dim, score, path_score = self.spectral_alignment_graph()
        max_val = float('-inf')
        max_index = 0
        for i in range(0, self.d + 1):
            if score[x_dim - 1][y_dim - 1][i] > max_val:
                max_index = i
                max_val = score[x_dim - 1][y_dim - 1][i]
        path_indices = list(map(int, path_score[x_dim - 1][y_dim - 1][max_index].split("/")[2:]))
        path_indices.append(len(self.spectrum_vector))
        mod_peptide = ''
        prev = 0
        for i, path_index in enumerate(path_indices):
            diff_val = path_index - prev
            diff_val -= self.amino_acid_mass[self.peptide[i]]
            mod_peptide += self.peptide[i]
            if diff_val > 0:
                mod_peptide += '(' + "+" + str(diff_val) + ')'
            elif diff_val < 0:
                mod_peptide += '(' + str(diff_val) + ')'
            prev = path_index
        return mod_peptide