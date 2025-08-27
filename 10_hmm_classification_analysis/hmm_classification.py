import numpy as np
import io
import os
import subprocess
import tempfile

class HMMBase:
    def __init__(self, states, transition, emission, iterations):
        self.states = states
        self.transition = transition
        self.emission = emission
        self.iterations = iterations
    
    @staticmethod
    def get_alphabet(x):
        return sorted(set(x))
    
    @staticmethod
    def format_hmm(states, transition, emission):
        buffer = io.StringIO()
        buffer.write("States:\n")
        buffer.write(' '.join(states) + "\n\n")
        buffer.write("Transition Matrix:\n")
        for row in transition:
            buffer.write(' '.join(map(str, row.tolist())) + "\n")
        buffer.write("\nEmission Matrix:\n")
        for row in emission:
            buffer.write(' '.join(map(str, row.tolist())) + "\n")
        return buffer.getvalue()
    
    def sequence_likelihood(self, x, transition, emission, alphabet):
        n_states = transition.shape[0]
        x2idx = {c:i for i, c in enumerate(alphabet)}
        x_list = np.array([x2idx[c] for c in x])
        n = len(x_list)
        log_transition = np.log(transition + 1e-12)
        log_emission = np.log(emission + 1e-12)
        forward = np.zeros((n, n_states))
        forward[0, :] = log_emission[:, x_list[0]] - np.log(n_states)
        for i in range(1, n):
            for j in range(n_states):
                prev = forward[i-1, :] + log_transition[:, j]
                max_prev = np.max(prev)
                forward[i, j] = np.log(np.sum(np.exp(prev - max_prev))) + max_prev + log_emission[j, x_list[i]]
        max_final = np.max(forward[-1, :])
        log_prob = np.log(np.sum(np.exp(forward[-1, :] - max_final))) + max_final
        return float(f"{log_prob:.3g}")

class ProfileHMM:
    def muscle_align(self, sequences, muscle_path=r"C:\Tools\muscle.exe"):
        if not sequences:
            raise ValueError("The sequence list is empty.")
        if not os.path.isfile(muscle_path):
            raise FileNotFoundError(f"MUSCLE executable not found at {muscle_path}")
        with tempfile.TemporaryDirectory() as temp_dir:
            input_fasta = os.path.join(temp_dir, "input.fasta")
            output_fasta = os.path.join(temp_dir, "output.fasta")
            with open(input_fasta, 'w') as f:
                for i, seq in enumerate(sequences):
                    f.write(f">seq{i+1}\n{seq}\n")
            result = subprocess.run(
                [muscle_path, "-align", input_fasta, "-output", output_fasta],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"MUSCLE failed:\n{result.stderr}")
            aligned_sequences = []
            with open(output_fasta, 'r') as f:
                seq = ""
                for line in f:
                    if line.startswith(">"):
                        if seq:
                            aligned_sequences.append(seq)
                            seq = ""
                    else:
                        seq += line.strip()
                if seq:
                    aligned_sequences.append(seq)
            return aligned_sequences
        
    def profile(self, theta, pseudocount, alphabet, alignment):
        alphabet_dict = ({alphabet[i]: i for i in range(len(alphabet))},
                        {i: alphabet[i] for i in range(len(alphabet))})
        n, l = alignment.shape
        threshold = theta * n
        kept = [sum(alignment[:, i] == '-') < threshold for i in range(l)]
        levels = [0] * l
        for i in range(l):
            levels[i] = levels[i-1]
            if kept[i]:
                levels[i] += 1

        def get_index(level, state):
            if level == 0:
                return 1 if state == 0 else 0
            return (3 * level - 2 + state) if state != 0 else (3 * level + 1)
        
        transition = np.zeros((sum(kept) * 3 + 3, 3), dtype=float)
        emission = np.zeros((sum(kept) * 3 + 3, len(alphabet)), dtype=float)
        for i in range(n):
            last_level = 0
            last_state = -1
            last_idx = get_index(last_level, last_state)
            for j in range(l):
                curr_level = levels[j]
                if kept[j]:
                    curr_state = 2 if alignment[i, j] == '-' else 1
                    curr_idx = get_index(curr_level, curr_state)
                    transition[last_idx, curr_state] += 1
                    if curr_state == 1:
                        emission[curr_idx, alphabet_dict[0][alignment[i, j]]] += 1
                    last_idx = curr_idx
                else:
                    if alignment[i, j] != '-':
                        curr_state = 0
                        curr_idx = get_index(curr_level, curr_state)
                        transition[last_idx, curr_state] += 1
                        emission[curr_idx, alphabet_dict[0][alignment[i, j]]] += 1
                        last_idx = curr_idx
            transition[last_idx, 2] += 1
        for i in range(transition.shape[0]):
            if np.sum(transition[i, :]) != 0:
                transition[i, :] /= np.sum(transition[i, :])
            if np.sum(emission[i, :]) != 0:
                emission[i, :] /= np.sum(emission[i, :])
        for i in range(transition.shape[0] - 4):
            transition[i, :] += pseudocount
            transition[i, :] /= np.sum(transition[i, :])
        for i in range(-4, -1):
            transition[i, 0] += pseudocount
            transition[i, 2] += pseudocount
            transition[i, :] /= np.sum(transition[i, :])
        for i in range(1, emission.shape[0] - 1):
            if i % 3 != 0:
                emission[i, :] += pseudocount
                emission[i, :] /= np.sum(emission[i, :])
        return transition, emission

    def get_all_states(self, n):
        states = [''] * n
        states[0] = 'S'
        states[-1] = 'E'
        states[1] = 'I0'
        s = ('M', 'D', 'I')
        for i in range(2, n-1):
            states[i] = s[(i+1) % 3] + str((i+1) // 3)
        return states

    def get_full_transition(self, transition):
        full_transition = np.zeros((transition.shape[0], transition.shape[0]), dtype=float)
        full_transition[0, 1:4] = transition[0, :]
        for i in range(1, transition.shape[0] - 4):
            if i % 3 == 1:
                full_transition[i, i:i+3] = transition[i, :]
            elif i % 3 == 2:
                full_transition[i, i+2:i+5] = transition[i, :]
            else:
                full_transition[i, i+1:i+4] = transition[i, :]
        for i in range(-4, -1):
            full_transition[i, -2] = transition[i, 0]
            full_transition[i, -1] = transition[i, -1]
        return full_transition
    
    def build_profile(self, x, theta, pseudocount, sequences):
        alignment = np.array([list(seq) for seq in self.muscle_align(sequences, muscle_path=r"C:\Tools\muscle.exe")])
        alphabet = HMMBase.get_alphabet(x)
        transition, emission = self.profile(theta, pseudocount, alphabet, alignment)
        states = self.get_all_states(transition.shape[0])
        full_transition = self.get_full_transition(transition)
        full_transition = np.round(full_transition, 3)
        emission = np.round(emission, 3)
        hmm = HMMBase.format_hmm(states, full_transition, emission)
        return hmm, states, full_transition, emission

class FindPath(HMMBase):
    def __init__(self, states, transition, emission, iterations):
        super().__init__(states, transition, emission, iterations)

    def find_hidden_path(self, x, transition_log, emission_log, alphabet):
        alphabet_to_index = {alphabet[i]: i for i in range(len(alphabet))}
        l = len(x)
        n = transition_log.shape[0]
        score = [[-np.inf for _ in range(l + 1)] for __ in range(n)]
        backtrack = [[None for _ in range(l + 1)] for __ in range(n)]
        score[3][0] = transition_log[0, 2]
        backtrack[3][0] = (0, 0)
        for i in range(6, n, 3):
            score[i][0] = score[i - 3][0] + transition_log[i - 3, 2]
            backtrack[i][0] = (i - 3, 0)
        score[1][1] = transition_log[0, 0] + emission_log[1, alphabet_to_index[x[0]]]
        backtrack[1][1] = (0, 0)
        score[2][1] = transition_log[0, 1] + emission_log[2, alphabet_to_index[x[0]]]
        backtrack[2][1] = (0, 0)
        score[3][1] = score[1][1] + transition_log[1, 2]
        backtrack[3][1] = (1, 1)
        for j in range(2, l + 1):
            score[1][j] = score[1][j - 1] + transition_log[1, 0] + emission_log[1, alphabet_to_index[x[j - 1]]]
            backtrack[1][j] = (1, j - 1)
            score[2][j] = score[1][j - 1] + transition_log[1, 1] + emission_log[2, alphabet_to_index[x[j - 1]]]
            backtrack[2][j] = (1, j - 1)
            score[3][j] = score[1][j] + transition_log[1, 2]
            backtrack[3][j] = (1, j)
        for i in range(4, n - 1):
            if i % 3 == 1:
                score[i][1] = score[i - 1][0] + transition_log[i - 1, 0] + emission_log[i, alphabet_to_index[x[0]]]
                backtrack[i][1] = (i - 1, 0)
                for j in range(2, l + 1):
                    candidates = ((i, j - 1), (i - 2, j - 1), (i - 1, j - 1))
                    scores = (
                        score[i][j - 1] + transition_log[i, 0] + emission_log[i, alphabet_to_index[x[j - 1]]],
                        score[i - 2][j - 1] + transition_log[i - 2, 0] + emission_log[i, alphabet_to_index[x[j - 1]]],
                        score[i - 1][j - 1] + transition_log[i - 1, 0] + emission_log[i, alphabet_to_index[x[j - 1]]]
                    )
                    idx = np.argmax(scores)
                    score[i][j] = scores[idx]
                    backtrack[i][j] = candidates[idx]
            elif i % 3 == 2:
                score[i][1] = score[i - 2][0] + transition_log[i - 2, 1] + emission_log[i, alphabet_to_index[x[0]]]
                backtrack[i][1] = (i - 2, 0)
                for j in range(2, l + 1):
                    candidates = ((i - 1, j - 1), (i - 3, j - 1), (i - 2, j - 1))
                    scores = (
                        score[i - 1][j - 1] + transition_log[i - 1, 1] + emission_log[i, alphabet_to_index[x[j - 1]]],
                        score[i - 3][j - 1] + transition_log[i - 3, 1] + emission_log[i, alphabet_to_index[x[j - 1]]],
                        score[i - 2][j - 1] + transition_log[i - 2, 1] + emission_log[i, alphabet_to_index[x[j - 1]]]
                    )
                    idx = np.argmax(scores)
                    score[i][j] = scores[idx]
                    backtrack[i][j] = candidates[idx]
            else:
                for j in range(1, l + 1):
                    candidates = ((i - 2, j), (i - 4, j), (i - 3, j))
                    scores = (
                        score[i - 2][j] + transition_log[i - 2, 2],
                        score[i - 4][j] + transition_log[i - 4, 2],
                        score[i - 3][j] + transition_log[i - 3, 2]
                    )
                    idx = np.argmax(scores)
                    score[i][j] = scores[idx]
                    backtrack[i][j] = candidates[idx]
        candidates = ((n - 2, l), (n - 4, l), (n - 3, l))
        scores = (
            score[n - 2][l] + transition_log[n - 2, 2],
            score[n - 4][l] + transition_log[n - 4, 2],
            score[n - 3][l] + transition_log[n - 3, 2]
        )
        idx = np.argmax(scores)
        backtrack[n - 1][l] = candidates[idx]
        path = []
        curr_pos = backtrack[n - 1][l]
        while curr_pos[0] != 0:
            path.insert(0, curr_pos)
            curr_pos = backtrack[curr_pos[0]][curr_pos[1]]
        return path

    def run(self, x):
        alphabet = self.get_alphabet(x)
        transition_log = np.log(np.where(self.transition > 0, self.transition, 1e-300))
        emission_log = np.log(np.where(self.emission > 0, self.emission, 1e-300))
        path = self.find_hidden_path(x, transition_log, emission_log, alphabet)
        prob = self.sequence_likelihood(x, self.transition, self.emission, alphabet)
        return " ".join([self.states[pos[0]] for pos in path]), prob
    
class BaumWelch(HMMBase):
    def __init__(self, states, transition, emission, iterations):
        super().__init__(states, transition, emission, iterations)

    def logsumexp(self, a):
        a_max = np.max(a)
        return np.log(np.sum(np.exp(a - a_max))) + a_max

    def forward_backward(self, seq, log_transition, log_emission):
        n_states = log_transition.shape[0]
        n = len(seq)
        alpha = np.zeros((n, n_states))
        alpha[0] = log_emission[:, seq[0]] - np.log(n_states)
        for t in range(1, n):
            for j in range(n_states):
                alpha[t, j] = self.logsumexp(alpha[t-1] + log_transition[:, j]) + log_emission[j, seq[t]]
        beta = np.zeros((n, n_states))
        beta[-1] = 0
        for t in range(n-2, -1, -1):
            for i in range(n_states):
                beta[t, i] = self.logsumexp(log_transition[i, :] + log_emission[:, seq[t+1]] + beta[t+1])
        log_gamma = alpha + beta
        log_gamma -= self.logsumexp(log_gamma[-1])
        log_xi = np.zeros((n_states, n_states, n-1))
        for t in range(n-1):
            for i in range(n_states):
                for j in range(n_states):
                    log_xi[i, j, t] = (alpha[t, i] + log_transition[i, j] +
                                       log_emission[j, seq[t+1]] + beta[t+1, j])
            log_xi[:, :, t] -= self.logsumexp(log_xi[:, :, t].flatten())
        return np.exp(log_gamma), np.exp(log_xi)

    def baum_welch(self, x, alphabet):
        x2idx = {c:i for i,c in enumerate(alphabet)}
        seq = np.array([x2idx[c] for c in x])
        n_states = len(self.states)
        transition = self.transition.copy()
        emission = self.emission.copy()
        for _ in range(self.iterations):
            log_transition = np.log(transition + 1e-12)
            log_emission = np.log(emission + 1e-12)
            gamma, xi = self.forward_backward(seq, log_transition, log_emission)
            transition = xi.sum(axis=2)
            transition /= transition.sum(axis=1, keepdims=True)
            emission = np.zeros_like(emission)
            for s in range(n_states):
                for t, o in enumerate(seq):
                    emission[s, o] += gamma[t, s]
            emission /= emission.sum(axis=1, keepdims=True)
        return transition, emission

    def run(self, x):
        alphabet = self.get_alphabet(x)
        updated_transition, updated_emission = self.baum_welch(x, alphabet)
        hmm = self.format_hmm(self.states, updated_transition, updated_emission)
        prob = self.sequence_likelihood(x, updated_transition, updated_emission, alphabet)
        path, y = FindPath(self.states, updated_transition, updated_emission, self.iterations).run(x)
        return hmm, prob, path

class Viterbi(HMMBase):
    def __init__(self, states, transition, emission, iterations):
        super().__init__(states, transition, emission, iterations)

    def viterbi_decode(self, x, log_transition, log_emission):
        n_states = log_transition.shape[0]
        n = len(x)
        score = np.zeros((n, n_states))
        backtrack = np.zeros((n, n_states), dtype=int)
        score[0] = log_emission[:, x[0]] - np.log(n_states)
        for t in range(1, n):
            for j in range(n_states):
                prob = score[t-1] + log_transition[:, j]
                backtrack[t, j] = np.argmax(prob)
                score[t, j] = np.max(prob) + log_emission[j, x[t]]
        path = np.zeros(n, dtype=int)
        path[-1] = np.argmax(score[-1])
        for t in range(n-2, -1, -1):
            path[t] = backtrack[t+1, path[t+1]]
        return path, np.max(score[-1])

    def viterbi_training(self, x, alphabet):
        x2idx = {c:i for i, c in enumerate(alphabet)}
        seq = np.array([x2idx[c] for c in x])
        n_states = len(self.states)
        n_symbols = len(alphabet)
        transition = self.transition.copy()
        emission = self.emission.copy()
        for _ in range(self.iterations):
            log_transition = np.log(transition + 1e-12)
            log_emission = np.log(emission + 1e-12)
            path, _ = self.viterbi_decode(seq, log_transition, log_emission)
            trans_counts = np.zeros_like(transition)
            for t in range(len(path)-1):
                trans_counts[path[t], path[t+1]] += 1
            row_sums = trans_counts.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            transition = trans_counts / row_sums
            transition = np.where(np.isnan(transition), 1/n_states, transition)
            emiss_counts = np.zeros_like(emission)
            for t, s in enumerate(path):
                emiss_counts[s, seq[t]] += 1
            row_sums = emiss_counts.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            emission = emiss_counts / row_sums
            emission = np.where(np.isnan(emission), 1/n_symbols, emission)
        return transition, emission

    def run(self, x):
        alphabet = self.get_alphabet(x)
        updated_transition, updated_emission = self.viterbi_training(x, alphabet)
        hmm = self.format_hmm(self.states, updated_transition, updated_emission)
        prob = self.sequence_likelihood(x, updated_transition, updated_emission, alphabet)
        path, y = FindPath(self.states, updated_transition, updated_emission, self.iterations).run(x)
        return hmm, prob, path

class MostDivergent(HMMBase):
    def __init__(self, states, transition, emission, iterations):
        super().__init__(states, transition, emission, iterations)
    
    def run(self, sequences):
        probs = []
        i = 0
        for x in sequences:
            i += 1
            bw = BaumWelch(self.states, self.transition, self.emission, self.iterations)
            hmm, prob, path = bw.run(x)
            probs.append((prob, i))
        best_prob, best_index = min(probs, key=lambda p: p[0])
        return best_prob, best_index