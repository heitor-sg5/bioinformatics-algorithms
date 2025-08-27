import numpy as np
from Bio.Align import substitution_matrices
from itertools import product
from collections import Counter
import math

class AlignmentBase:
    def build_score_matrix(self, pmm, p_gap):
        bases = ['A', 'C', 'G', 'T', '-']
        score_matrix = {}
        for b1 in bases:
            for b2 in bases:
                if b1 == '-' or b2 == '-':
                    score = -p_gap
                elif b1 == b2:
                    score = 1
                else:
                    score = -pmm
                score_matrix[(b1, b2)] = score
        return score_matrix

    def backtrack_pairwise(self, backtrack, v, w, i, j, stop_condition='nonzero'):
        aligned_v, aligned_w = [], []
        while (i > 0 or j > 0) if stop_condition == 'edge' else backtrack[i][j] != '0':
            if backtrack[i][j] == 'D':
                if i > 0 and j > 0:
                    aligned_v.append(v[i - 1])
                    aligned_w.append(w[j - 1])
                i -= 1
                j -= 1
            elif backtrack[i][j] == 'V':
                if i > 0:
                    aligned_v.append(v[i - 1])
                    aligned_w.append('-')
                i -= 1
            else:
                if j > 0:
                    aligned_v.append('-')
                    aligned_w.append(w[j - 1])
                j -= 1
            if i <= 0 and j <= 0:
                break
        return ''.join(reversed(aligned_v)), ''.join(reversed(aligned_w))

class GlobalAlignment(AlignmentBase):
    def __init__(self, mode, pmm, p_gap, gap_open=None, gap_extend=None):
        self.mode = mode
        self.pmm = pmm
        self.p_gap = p_gap
        self.gap_open = gap_open if gap_open is not None else p_gap
        self.gap_extend = gap_extend if gap_extend is not None else p_gap
        self.score_matrix = self.build_score_matrix(pmm, p_gap) if mode != 'pam250' else substitution_matrices.load("PAM250")

    def needleman_wunsch(self, v, w):
        n, m = len(v), len(w)
        s = np.zeros((n+1, m+1), dtype=int)
        backtrack = np.full((n+1, m+1), '', dtype=object)
        for i in range(1, n+1):
            s[i][0] = s[i-1][0] + (self.score_matrix[(v[i-1], '-')] if self.mode != 'pam250' else self.p_gap)
            backtrack[i][0] = 'V'
        for j in range(1, m+1):
            s[0][j] = s[0][j-1] + (self.score_matrix[('-', w[j-1])] if self.mode != 'pam250' else self.p_gap)
            backtrack[0][j] = 'H'
        for i in range(1, n+1):
            for j in range(1, m+1):
                match_score = self.score_matrix.get((v[i-1], w[j-1]), -float('inf')) if self.mode == 'pam250' else self.score_matrix[(v[i-1], w[j-1])]
                match = s[i-1][j-1] + match_score
                delete = s[i-1][j] + (self.score_matrix[(v[i-1], '-')] if self.mode != 'pam250' else self.p_gap)
                insert = s[i][j-1] + (self.score_matrix[('-', w[j-1])] if self.mode != 'pam250' else self.p_gap)
                s[i][j] = max(match, delete, insert)
                if s[i][j] == match:
                    backtrack[i][j] = 'D'
                elif s[i][j] == delete:
                    backtrack[i][j] = 'V'
                else:
                    backtrack[i][j] = 'H'
        aligned_v, aligned_w = self.backtrack_pairwise(backtrack, v, w, n, m, 'edge')
        return [aligned_v, aligned_w], s[n][m]

    def affine_gap(self, v, w):
        n, m = len(v), len(w)
        M = np.full((n+1, m+1), -np.inf)
        Ix = np.full((n+1, m+1), -np.inf)
        Iy = np.full((n+1, m+1), -np.inf)
        backtrack = np.full((n+1, m+1), '', dtype=object)
        M[0][0] = 0
        for i in range(1, n+1):
            Iy[i][0] = -self.gap_open - (i-1) * self.gap_extend
            M[i][0] = Iy[i][0]
            backtrack[i][0] = 'V'
        for j in range(1, m+1):
            Ix[0][j] = -self.gap_open - (j-1) * self.gap_extend
            M[0][j] = Ix[0][j]
            backtrack[0][j] = 'H'
        for i in range(1, n+1):
            for j in range(1, m+1):
                match_score = self.score_matrix.get((v[i-1], w[j-1]), -float('inf'))
                M[i][j] = max(M[i-1][j-1], Ix[i-1][j-1], Iy[i-1][j-1]) + match_score
                Ix[i][j] = max(M[i][j-1] - self.gap_open, Ix[i][j-1] - self.gap_extend)
                Iy[i][j] = max(M[i-1][j] - self.gap_open, Iy[i-1][j] - self.gap_extend)
                
                if M[i][j] >= Ix[i][j] and M[i][j] >= Iy[i][j]:
                    backtrack[i][j] = 'D'
                elif Iy[i][j] >= Ix[i][j]:
                    backtrack[i][j] = 'V'
                else:
                    backtrack[i][j] = 'H'
        aligned_v, aligned_w = self.backtrack_pairwise(backtrack, v, w, n, m, 'edge')
        return [aligned_v, aligned_w], int(M[n][m])

    def hirschberg(self, v, w):
        def compute_score(v, w):
            prev = np.zeros(len(w)+1, dtype=int)
            for j in range(1, len(w)+1):
                prev[j] = prev[j-1] + (self.score_matrix[('-', w[j-1])] if self.mode != 'pam250' else self.p_gap)
            for i in range(1, len(v)+1):
                curr = np.zeros(len(w)+1, dtype=int)
                curr[0] = prev[0] + (self.score_matrix[(v[i-1], '-')] if self.mode != 'pam250' else self.p_gap)
                for j in range(1, len(w)+1):
                    match_score = self.score_matrix.get((v[i-1], w[j-1]), -float('inf')) if self.mode == 'pam250' else self.score_matrix[(v[i-1], w[j-1])]
                    match = prev[j-1] + match_score
                    delete = prev[j] + (self.score_matrix[(v[i-1], '-')] if self.mode != 'pam250' else self.p_gap)
                    insert = curr[j-1] + (self.score_matrix[('-', w[j-1])] if self.mode != 'pam250' else self.p_gap)
                    curr[j] = max(match, delete, insert)
                prev = curr
            return prev
        if len(v) == 0:
            return ["-" * len(w), w], sum(self.score_matrix.get(('-', b), self.p_gap) for b in w)
        if len(w) == 0:
            return [v, "-" * len(v)], sum(self.score_matrix.get((a, '-'), self.p_gap) for a in v)
        if len(v) == 1 or len(w) == 1:
            return self.needleman_wunsch(v, w)
        n, mid = len(v), len(v) // 2
        scoreL = compute_score(v[:mid], w)
        scoreR = compute_score(v[mid:][::-1], w[::-1])
        m = len(w)
        max_score, split = None, 0
        for j in range(m + 1):
            score = scoreL[j] + scoreR[m - j]
            if max_score is None or score > max_score:
                max_score, split = score, j
        (left_v, left_w), _ = self.hirschberg(v[:mid], w[:split])
        (right_v, right_w), _ = self.hirschberg(v[mid:], w[split:])
        aligned_v = left_v + right_v
        aligned_w = left_w + right_w
        score = sum(
            self.score_matrix.get((a, b), -float('inf')) if self.mode == 'pam250'
            else self.score_matrix[(a, b)]
            for a, b in zip(aligned_v, aligned_w)
        )
        return [aligned_v, aligned_w], score

    def run(self, v, w):
        if self.mode == 'needleman_wunsch':
            return self.needleman_wunsch(v, w)
        elif self.mode == 'affine_gap':
            return self.affine_gap(v, w)
        elif self.mode == 'hirschberg':
            return self.hirschberg(v, w)
        elif self.mode == 'pam250':
            return self.needleman_wunsch(v, w)

class SmithWaterman(AlignmentBase):
    def __init__(self, pmm, p_gap):
        self.pmm = pmm
        self.p_gap = p_gap
        self.score_matrix = self.build_score_matrix(pmm, p_gap)

    def run(self, v, w):
        n, m = len(v), len(w)
        s = np.zeros((n+1, m+1), dtype=int)
        backtrack = np.full((n+1, m+1), '', dtype=object)
        max_score, max_pos = 0, (0, 0)
        for i in range(1, n+1):
            for j in range(1, m+1):
                match = s[i-1][j-1] + self.score_matrix[(v[i-1], w[j-1])]
                delete = s[i-1][j] + self.score_matrix[(v[i-1], '-')]
                insert = s[i][j-1] + self.score_matrix[('-', w[j-1])]
                s[i][j] = max(match, delete, insert, 0)
                if s[i][j] == 0:
                    backtrack[i][j] = '0'
                elif s[i][j] == match:
                    backtrack[i][j] = 'D'
                elif s[i][j] == delete:
                    backtrack[i][j] = 'V'
                else:
                    backtrack[i][j] = 'H'
                if s[i][j] > max_score:
                    max_score, max_pos = s[i][j], (i, j)
        aligned_v, aligned_w = self.backtrack_pairwise(backtrack, v, w, *max_pos, 'nonzero')
        return [aligned_v, aligned_w], max_score

class FittingAlignment(AlignmentBase):
    def __init__(self, pmm, p_gap):
        self.pmm = pmm
        self.p_gap = p_gap
        self.score_matrix = self.build_score_matrix(pmm, p_gap)

    def run(self, v, w):
        n, m = len(v), len(w)
        s = np.zeros((n+1, m+1), dtype=int)
        backtrack = np.full((n+1, m+1), '', dtype=object)
        for i in range(1, n+1):
            s[i][0] = 0
            backtrack[i][0] = 'V'
        for j in range(1, m+1):
            s[0][j] = s[0][j-1] + self.score_matrix[('-', w[j-1])]
            backtrack[0][j] = 'H'
        for i in range(1, n+1):
            for j in range(1, m+1):
                match = s[i-1][j-1] + self.score_matrix[(v[i-1], w[j-1])]
                delete = s[i-1][j] + self.score_matrix[(v[i-1], '-')]
                insert = s[i][j-1] + self.score_matrix[('-', w[j-1])]
                s[i][j] = max(match, delete, insert)
                
                if s[i][j] == match:
                    backtrack[i][j] = 'D'
                elif s[i][j] == delete:
                    backtrack[i][j] = 'V'
                else:
                    backtrack[i][j] = 'H'
        max_score, max_i = float('-inf'), 0
        for i in range(n+1):
            if s[i][m] > max_score:
                max_score, max_i = s[i][m], i
        aligned_v, aligned_w = self.backtrack_pairwise(backtrack, v, w, max_i, m, 'edge')
        return [aligned_v, aligned_w], max_score

class MSA(AlignmentBase):
    def __init__(self, scoring_mode, pmm, p_gap):
        self.scoring_mode = scoring_mode
        self.pmm = pmm
        self.p_gap = p_gap
        self.score_matrix = self.build_score_matrix(pmm, p_gap)

    def multi_score(self, chars):
        score = 0
        t = len(chars)
        for i in range(t):
            for j in range(i+1, t):
                score += self.score_matrix[(chars[i], chars[j])]
        return score

    def entropy_score(self, chars):
        counts = Counter(chars)
        total = len(chars)
        entropy = 0
        for char, count in counts.items():
            p = count / total
            if char != '-':
                entropy -= p * math.log2(p)
            else:
                entropy += self.p_gap * count
        return -entropy

    def run(self, sequences):
        t = len(sequences)
        lengths = [len(seq) for seq in sequences]
        s = np.full(tuple(l+1 for l in lengths), float('-inf'))
        backtrack = np.empty(tuple(l+1 for l in lengths), dtype=object)
        s[(0,)*t] = 0
        moves = [move for move in product([0,1], repeat=t) if any(move)]
        for idx in np.ndindex(*tuple(l+1 for l in lengths)):
            if idx == (0,)*t:
                continue
            max_score, best_move = float('-inf'), None
            for move in moves:
                prev_idx = tuple(idx[i] - move[i] for i in range(t))
                if any(x < 0 for x in prev_idx):
                    continue
                chars = [sequences[i][prev_idx[i]] if move[i] else '-' for i in range(t)]
                prev_score = s[prev_idx]
                if prev_score == float('-inf'):
                    continue
                curr_score = prev_score + (self.multi_score(chars) if self.scoring_mode == 'pair_sum' else self.entropy_score(chars))
                if curr_score > max_score:
                    max_score, best_move = curr_score, move
            s[idx] = max_score
            backtrack[idx] = best_move
        aligned = ['']*t
        idx = tuple(lengths)
        while idx != (0,)*t:
            move = backtrack[idx]
            for i in range(t):
                aligned[i] = (sequences[i][idx[i]-1] if move[i] else '-') + aligned[i]
            idx = tuple(idx[i] - move[i] for i in range(t))
        score = s[tuple(lengths)] if self.scoring_mode == 'pair_sum' else int(s[tuple(lengths)])
        return aligned, score