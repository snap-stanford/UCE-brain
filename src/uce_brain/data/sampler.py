"""
Cell sentence sampling for UCE model.

This module provides functions to convert single-cell expression data into
structured genomic sequences that preserve chromosome organization.
"""

import torch
import numpy as np


def sample_cell_sentences_mapping_gene(
    counts,
    batch_weights,
    gene_protein_ids,
    gene_chroms,
    gene_starts,
    pad_length: int,
    positive_sample_num: int,
    negative_sample_num: int,
    mask_prop: float,
    sample_size: int,
    cls_token_idx: int,
    chrom_token_offset: int,
    chrom_token_right_idx: int,
    pad_token_idx: int,
    seed: int = None,
):
    """
    Sample cell sentences with gene mapping and chromosome ordering.

    This function converts single-cell expression data into structured genomic sequences
    that preserve chromosome organization and expression patterns. The process involves:

    1. Expression-weighted sampling of genes based on their expression levels
    2. Chromosome-based ordering to preserve genomic spatial relationships
    3. Generation of positive (expressed) and negative (non-expressed) gene pairs
    4. Creation of padded sequences with special tokens for structure
    5. Masking of expressed genes for self-supervised learning

    The output sequences follow the format:
    [CLS] [CHROM_1] gene1 gene2 ... [CHROM_END] [CHROM_2] gene3 gene4 ... [PAD]

    Args:
        counts: Gene expression counts for the cell of shape [num_genes].
        batch_weights: Normalized expression weights of shape [num_genes].
        gene_protein_ids: Protein embedding IDs for each gene of shape [num_genes].
        gene_chroms: Chromosome IDs for each gene of shape [num_genes].
        gene_starts: Genomic start positions for each gene of shape [num_genes].
        pad_length: Maximum sequence length for padding.
        positive_sample_num: Number of expressed genes to sample for positive examples.
        negative_sample_num: Number of non-expressed genes to sample for negative examples.
        mask_prop: Proportion of expressed genes to mask during sequence generation.
        sample_size: Total number of genes to sample when creating the cell sequence.
        cls_token_idx: Index of the classification (CLS) token.
        chrom_token_offset: Offset added to chromosome IDs to create unique chromosome tokens.
        chrom_token_right_idx: Token index marking the end of each chromosome region.
        pad_token_idx: Token index used for padding sequences to uniform length.
        seed: Random seed for reproducible masking and sampling.

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing:
            - "batch_sentences": Gene sequences with chromosome structure [1, pad_length]
            - "mask": Padding mask indicating valid positions [1, pad_length]
            - "cell_outputs_X_pe": Protein embedding indices for sampled genes [1, P+N]
            - "cell_outputs_Y": Binary labels for expression prediction [1, P+N]
            - "seq_len": Actual sequence length used before padding
            - "chroms": Chromosome IDs for the sampled genes [1, sample_size]
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = None

    dataset_idxs = gene_protein_ids

    cell_sentences = torch.zeros((counts.shape[0], pad_length), dtype=torch.long)
    mask = torch.zeros((counts.shape[0], pad_length), dtype=bool)
    cell_outputs_X = torch.zeros((counts.shape[0], positive_sample_num + negative_sample_num), dtype=torch.long)
    cell_outputs_Y = torch.zeros((counts.shape[0], positive_sample_num + negative_sample_num), dtype=torch.float32)

    chroms = gene_chroms
    starts = gene_starts
    longest_seq_len = 0

    for c, cell in enumerate(counts):
        pos_genes = torch.where(counts[c] > 0)[0]
        neg_genes = torch.where(counts[c] < 1)[0]

        if len(pos_genes) == 0:
            pos_genes = neg_genes

        weights = batch_weights[c].numpy()

        if len(weights) == 0:
            raise ValueError(f"Cell {c} has no genes available for processing")

        if weights.ndim == 0:
            weights = np.array([float(weights)])

        if mask_prop > 0 and len(pos_genes) > 0:
            mask_size = max(1, round(len(pos_genes) * mask_prop))
            mask_size = min(mask_size, len(pos_genes))
            if rng is not None:
                mask_weights = rng.choice(pos_genes, size=mask_size, replace=False)
            else:
                mask_weights = np.random.choice(pos_genes, size=mask_size, replace=False)

            if weights.ndim == 0:
                weights = np.array([float(weights)])

            weights[mask_weights] = 0

            weight_sum = sum(weights)
            if weight_sum > 0:
                weights = weights / weight_sum
            else:
                weights = np.ones_like(weights) / len(weights)
        else:
            mask_weights = np.array([])

        if rng is not None:
            choice_idx = rng.choice(np.arange(len(weights)), size=sample_size, p=weights, replace=True)
        else:
            choice_idx = np.random.choice(np.arange(len(weights)), size=sample_size, p=weights, replace=True)

        choosen_chrom = chroms[choice_idx]

        chrom_sort = np.argsort(choosen_chrom)
        choice_idx = choice_idx[chrom_sort]
        new_chrom = chroms[choice_idx]
        choosen_starts = starts[choice_idx]

        ordered_choice_idx = np.full((pad_length), cls_token_idx)
        i = 1

        uq_chroms = np.unique(new_chrom)
        if rng is not None:
            rng.shuffle(uq_chroms)
        else:
            np.random.shuffle(uq_chroms)

        for chrom in uq_chroms:
            ordered_choice_idx[i] = int(chrom) + chrom_token_offset
            i += 1

            loc = np.where(new_chrom == chrom)[0]

            sort_by_start = np.argsort(choosen_starts[loc])
            to_add = choice_idx[loc[sort_by_start]]

            ordered_choice_idx[i:(i + len(to_add))] = dataset_idxs[to_add]
            i += len(to_add)

            ordered_choice_idx[i] = chrom_token_right_idx
            i += 1

        longest_seq_len = max(longest_seq_len, i)

        remainder_len = pad_length - i
        cell_mask = torch.concat((torch.zeros(i, dtype=bool), torch.ones(remainder_len, dtype=bool)))
        mask[c, :] = cell_mask

        ordered_choice_idx[i:] = pad_token_idx
        cell_sentences[c, :] = torch.from_numpy(ordered_choice_idx)

        choice_idx_output_p = mask_weights
        if len(choice_idx_output_p) > positive_sample_num:
            if rng is not None:
                choice_idx_output_p = rng.choice(choice_idx_output_p, replace=False, size=positive_sample_num)
            else:
                choice_idx_output_p = np.random.choice(choice_idx_output_p, replace=False, size=positive_sample_num)
        elif len(choice_idx_output_p) < positive_sample_num:
            remainder = positive_sample_num - len(choice_idx_output_p)
            if rng is not None:
                choice_idx_output_p = np.append(choice_idx_output_p, rng.choice(pos_genes, size=remainder, replace=True))
            else:
                choice_idx_output_p = np.append(choice_idx_output_p, np.random.choice(pos_genes, size=remainder, replace=True))

        if negative_sample_num <= len(neg_genes):
            if rng is not None:
                choice_idx_output_n = rng.choice(np.arange(len(neg_genes)), size=negative_sample_num, replace=False)
            else:
                choice_idx_output_n = np.random.choice(np.arange(len(neg_genes)), size=negative_sample_num, replace=False)
        else:
            if rng is not None:
                choice_idx_output_n = rng.choice(np.arange(len(neg_genes)), size=negative_sample_num, replace=True)
            else:
                choice_idx_output_n = np.random.choice(np.arange(len(neg_genes)), size=negative_sample_num, replace=True)

        choice_idx_output_n = neg_genes[choice_idx_output_n]

        cell_outputs_X[c] = torch.tensor(np.concatenate((choice_idx_output_p, choice_idx_output_n)), dtype=torch.long)

        cell_outputs_Y[c] = torch.cat((torch.ones(positive_sample_num, dtype=torch.float32), torch.zeros(negative_sample_num, dtype=torch.float32)))

    cell_sentences_pe = cell_sentences.long()

    cell_outputs_X_pe = dataset_idxs[cell_outputs_X.long()]
    cell_outputs_X_pe = torch.from_numpy(cell_outputs_X_pe).long()

    sampled_result = {
        "batch_sentences": cell_sentences_pe,
        "mask": mask,
        "cell_outputs_X_pe": cell_outputs_X_pe,
        "cell_outputs_Y": cell_outputs_Y,
        "seq_len": pad_length,
    }

    return sampled_result
