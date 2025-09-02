from collections import defaultdict
import numpy as np
from hmmlearn.hmm import CategoricalHMM
import matplotlib.pyplot as plt

class GeneBase:
    nuc_to_idx = {
        'A': 0,
        'C': 1,
        'G': 2,
        'T': 3,
        'U': 3,
        'N': 4,
        'X': 4
    }
    idx_to_nuc = {v: k for k, v in nuc_to_idx.items()}
    n_symbols = 5

    state_names = ['E', 'I', 'N']
    state_index = {s: i for i, s in enumerate(state_names)}

class ParseFiles(GeneBase):
    def parse_gff3(self, gff3_path, seqid_filter=None):
        genes_exons = defaultdict(list)
        used_seqid = None
        with open(gff3_path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) < 9:
                    continue
                seqid, source, ftype, start, end, score, strand, phase, attributes = parts[:9]
                if seqid_filter and seqid != seqid_filter:
                    continue
                if used_seqid is None:
                    used_seqid = seqid
                if ftype.lower() in ('exon','cds'):
                    start_i = int(start)
                    end_i = int(end)
                    attr = {k:v for k,v in (item.split('=') for item in attributes.split(';') if '=' in item)}
                    gene_id = None
                    if 'Parent' in attr:
                        gene_id = attr['Parent']
                    elif 'gene_id' in attr:
                        gene_id = attr['gene_id']
                    elif 'gene' in attr:
                        gene_id = attr['gene']
                    elif 'ID' in attr:
                        gene_id = attr['ID']
                    else:
                        gene_id = f"unknown_{seqid}_{start}_{end}"
                    genes_exons[gene_id].append((start_i, end_i))
        for gid, exons in genes_exons.items():
            exons_sorted = sorted(exons, key=lambda x: x[0])
            merged = []
            for s,e in exons_sorted:
                if not merged or s > merged[-1][1] + 1:
                    merged.append([s,e])
                else:
                    merged[-1][1] = max(merged[-1][1], e)
            genes_exons[gid] = [(a,b) for a,b in merged]
        return genes_exons, used_seqid
    
class BuildHMM(GeneBase):
    def build_hmm_from_annotation(self, ref_seq_record, genes_exons):
        chrom_len = len(ref_seq_record.seq)
        labels = np.full(chrom_len, self.state_index['N'], dtype=int)
        for gid, exons in genes_exons.items():
            for s,e in exons:
                labels[s-1:e] = self.state_index['E']
        introns, _ = self.compute_introns(genes_exons)
        for s,e in introns:
            labels[s-1:e] = self.state_index['I']
        counts_start = np.zeros(len(self.state_names), dtype=float)
        counts_trans = np.zeros((len(self.state_names), len(self.state_names)), dtype=float)
        counts_emit = np.zeros((len(self.state_names), self.n_symbols), dtype=float)
        counts_start[labels[0]] += 1
        for i in range(len(labels)-1):
            counts_trans[labels[i], labels[i+1]] += 1
        seqstr = str(ref_seq_record.seq).upper()
        for i, ch in enumerate(seqstr):
            st = labels[i]
            idx = self.nuc_to_idx.get(ch, 4)
            counts_emit[st, idx] += 1
        counts_start += 1.0
        counts_trans += 1.0
        counts_emit += 1.0
        startprob = counts_start / counts_start.sum()
        transmat = counts_trans / counts_trans.sum(axis=1, keepdims=True)
        emissionprob = counts_emit / counts_emit.sum(axis=1, keepdims=True)
        model = CategoricalHMM(n_components=len(self.state_names), n_iter=10, init_params='')
        model.n_features = self.n_symbols
        model.startprob_ = startprob
        model.transmat_ = transmat
        model.emissionprob_ = emissionprob
        return model, labels
    
    def compute_introns(self, genes_exons):
        introns = []
        gene_spans = {}
        for gid, exons in genes_exons.items():
            gene_start = exons[0][0]
            gene_end = exons[-1][1]
            gene_spans[gid] = (gene_start, gene_end)
            if len(exons) >= 2:
                for i in range(len(exons)-1):
                    intron_start = exons[i][1] + 1
                    intron_end = exons[i+1][0] - 1
                    if intron_start <= intron_end:
                        introns.append((intron_start, intron_end))
        return introns, gene_spans

class FindViterbi(GeneBase):
    def viterbi_on_query(self, model, query_seq):
        seq = str(query_seq).upper()
        obs = np.array([[self.nuc_to_idx.get(ch, 4)] for ch in seq], dtype=int)
        logprob, state_path = model.decode(obs, algorithm='viterbi')
        return logprob, state_path

    def find_genes_from_state_path(self, state_path):
        inter_idx = self.state_index['N']
        genes = []
        in_gene = False
        for i, st in enumerate(state_path):
            if st != inter_idx and not in_gene:
                in_gene = True
                cur_start = i+1
            if st == inter_idx and in_gene:
                in_gene = False
                cur_end = i
                genes.append((cur_start, cur_end))
        if in_gene:
            genes.append((cur_start, len(state_path)))
        return genes
    
    def run(self, model, query_seq_record):
        logprob, state_path = self.viterbi_on_query(model, query_seq_record)
        genes = self.find_genes_from_state_path(state_path)
        return logprob, state_path, genes
    
class FindGenes(GeneBase):
    def get_genes(self, seq_str, genes, path):
        results = []
        gene_lengths = [end-start+1 for start, end in genes]
        results.append(f"Mean length: {np.mean(gene_lengths):.0f} | Min: {np.min(gene_lengths):.0f} | Max: {np.max(gene_lengths):.0f}\n")
        results.append(f'{len(genes)} genes found.\n\n')
        for gene_idx, (gene_start, gene_end) in enumerate(sorted(genes, key=lambda x: x[0]), start=1):
            results.append(f'Gene {gene_idx} | Position: {gene_start}-{gene_end} | Length: {gene_end-gene_start+1}\n')
            gene_seq = seq_str[gene_start-1:gene_end]
            gene_path = path[gene_start-1:gene_end]
            current_state = gene_path[0]
            region_start = gene_start
            region_count = 1
            for i in range(1, len(gene_path)):
                if gene_path[i] != current_state:
                    region_end = gene_start + i - 1
                    region_seq = seq_str[region_start-1:region_end]
                    if current_state == self.state_index['I']:
                        results.append(f'Intron {region_count} | Position: {region_start}-{region_end} | Length: {region_end-region_start+1}\n')
                        results.append(f'{region_seq}\n')
                    elif current_state == self.state_index['E']:
                        results.append(f'Exon {region_count} | Position: {region_start}-{region_end} | Length: {region_end-region_start+1}\n')
                        results.append(f'{region_seq}\n')
                    region_start = region_end + 1
                    current_state = gene_path[i]
                    region_count += 1
            region_end = gene_end
            region_seq = seq_str[region_start-1:region_end]
            if current_state == self.state_index['I']:
                results.append(f'Intron {region_count} | Position: {region_start}-{region_end} | Length: {region_end-region_start+1}\n')
                results.append(f'{region_seq}\n')
            elif current_state == self.state_index['E']:
                results.append(f'Exon {region_count} | Position: {region_start}-{region_end} | Length: {region_end-region_start+1}\n')
                results.append(f'{region_seq}\n')
            results.append('\n')
        return results

class Charts(GeneBase):
    def display(self, seq, genes):
        self.gene_density_distribution(seq, genes)
        return

    def gene_density_distribution(self, seq, genes, window_size=1_000_000):
        genome_len = len(seq)
        n_bins = genome_len // window_size + 1
        density = [0] * n_bins
        for start, end in genes:
            start_bin = start // window_size
            end_bin = end // window_size
            for b in range(start_bin, end_bin + 1):
                density[b] += 1
        plt.figure(figsize=(14, 5))
        plt.bar(range(len(density)), density, width=1.0, color="skyblue", edgecolor="black")
        plt.xlabel(f"Position (per {window_size:,} bp)")
        plt.ylabel("Number of Genes")
        plt.title("Gene Density Across Genome")
        plt.tight_layout()
        plt.show()