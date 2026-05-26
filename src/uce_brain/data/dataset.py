"""
H5AD Dataset for UCE model inference.

This module provides a PyTorch Dataset that loads cells directly from h5ad files
for embedding extraction.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import scanpy as sc
import torch
from torch.utils.data import Dataset

from .sampler import sample_cell_sentences_mapping_gene

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


class H5ADDataset(Dataset):
    """
    PyTorch Dataset that loads cells directly from an h5ad file.

    This dataset processes cells on-the-fly without requiring pre-caching
    to a HuggingFace dataset format.
    """

    def __init__(
        self,
        adata: sc.AnnData,
        gene_mapping: Dict,
        pad_length: int = 2048,
        positive_sample_num: int = 100,
        negative_sample_num: int = 100,
        mask_prop: float = 0.0,
        sample_size: int = 1024,
        cls_token_idx: int = 1,
        # chrom_token_offset: int = 143574,
        # chrom_token_right_idx: int = 2,
        chrom_token_offset: int = 1000,
        chrom_token_right_idx: int = 2000,
        pad_token_idx: int = 0,
        use_raw: bool = False,
        max_cells: Optional[int] = None,
        gene_symbol_column: Optional[str] = None,
        species: str = "human",
        case_insensitive: bool = False,
    ):
        """
        Initialize the H5AD dataset.

        Args:
            adata: The AnnData object to use for the dataset
            gene_mapping: Dictionary mapping gene names to genomic features
                          (protein_embedding_id, chromosome_id, location).
                          Only genes present in both the h5ad and this mapping are used.
            pad_length: Maximum sequence length
            positive_sample_num: Number of expressed genes to sample
            negative_sample_num: Number of non-expressed genes to sample
            mask_prop: Proportion of genes to mask (0 for embedding extraction)
            sample_size: Total number of genes to sample for sequence
            cls_token_idx: CLS token index
            chrom_token_offset: Offset for chromosome tokens
            chrom_token_right_idx: Token index for chromosome end
            pad_token_idx: Padding token index
            use_raw: Whether to use adata.raw for expression data
            max_cells: Optional limit on number of cells to process
            gene_symbol_column: Column in adata.var containing gene symbols (auto-detected if None)
            species: Species of the dataset (e.g., "human", "mouse")
            case_insensitive: If True, normalize both h5ad gene names and vocab
                keys to uppercase before matching. Useful for mouse data, where
                MGI sentence-case symbols (``Gnai3``) need to align with the
                vocab's uppercase keys (``GNAI3``). Off by default to preserve
                exact-match behaviour for existing notebooks.
        """
        self.case_insensitive = case_insensitive
        self.adata = adata
        # Accept either the full multi-species dict (top-level species keys)
        # or a pre-indexed species sub-dict (passed for backward compat).
        # Detect by checking whether ``species`` is a top-level key.
        if species in gene_mapping:
            self.gene_mapping = gene_mapping[species]
            self.species = species
        elif SPECIES_VOCAB_ALIASES.get(species) in gene_mapping:
            alias = SPECIES_VOCAB_ALIASES[species]
            log.info(f"Resolved species alias '{species}' -> '{alias}'")
            self.gene_mapping = gene_mapping[alias]
            self.species = alias
        else:
            # Heuristic: if values look like gene-info dicts (have
            # protein_embedding_id), assume gene_mapping is already
            # pre-indexed for a single species.
            first_value = next(iter(gene_mapping.values()))
            if isinstance(first_value, dict) and "protein_embedding_id" in first_value:
                self.gene_mapping = gene_mapping
                self.species = species
            else:
                raise KeyError(
                    f"Species '{species}' not in gene_mapping. "
                    f"Top-level keys: {sorted(gene_mapping.keys())}"
                )
        self.pad_length = pad_length
        self.positive_sample_num = positive_sample_num
        self.negative_sample_num = negative_sample_num
        self.mask_prop = mask_prop
        self.sample_size = sample_size
        self.cls_token_idx = cls_token_idx
        self.chrom_token_offset = chrom_token_offset
        self.chrom_token_right_idx = chrom_token_right_idx
        self.pad_token_idx = pad_token_idx

        log.info(f"Loading AnnData object with shape {self.adata.shape}")

        if use_raw and self.adata.raw is not None:
            log.info("Using raw expression data from adata.raw")
            self.adata = self.adata.raw.to_adata()

        self.n_cells = self.adata.n_obs
        if max_cells is not None and max_cells < self.n_cells:
            self.n_cells = max_cells
            log.info(f"Limiting to {max_cells} cells")

        log.info(f"Loaded {self.n_cells} cells with {self.adata.n_vars} genes")

        self.h5ad_gene_names = self._get_gene_names(gene_symbol_column)
        self._create_gene_alignment()
        self._prepare_gene_arrays()

    def _get_gene_names(self, gene_symbol_column: Optional[str] = None) -> List[str]:
        """
        Get gene names from the h5ad file.

        If var_names are Ensembl IDs (ENSG...), try to use gene symbols from var columns.
        """
        var_names = list(self.adata.var_names)

        # Any Ensembl-style prefix triggers the symbol lookup: ENSG (human),
        # ENSMUSG (mouse), ENSMMUG (macaque), ENSCJAG (marmoset), etc.
        is_ensembl = var_names[0].startswith('ENS') if var_names else False

        if not is_ensembl:
            log.info("Using var_names as gene names (already gene symbols)")
            return var_names

        log.info(f"var_names look like Ensembl IDs (e.g. {var_names[0]!r}); looking for gene symbols...")

        possible_columns = ['feature_name', 'gene_symbols', 'gene_symbol', 'gene_name', 'symbol', 'name']
        if gene_symbol_column:
            possible_columns = [gene_symbol_column] + possible_columns

        for col in possible_columns:
            if col in self.adata.var.columns:
                gene_symbols = list(self.adata.var[col])
                valid_symbols = [g for g in gene_symbols if g and str(g) != 'nan']
                if len(valid_symbols) > len(gene_symbols) * 0.5:
                    log.info(f"Using gene symbols from column '{col}'")
                    return [str(g) if g and str(g) != 'nan' else var_names[i]
                            for i, g in enumerate(gene_symbols)]

        log.warning("No gene symbol column found, using Ensembl IDs as gene names")
        return var_names

    def _create_gene_alignment(self):
        """Create mapping from h5ad genes to gene_mapping vocabulary.

        Builds parallel arrays:
          * ``valid_h5ad_indices``: positions in ``self.h5ad_gene_names`` whose
            symbol is in the vocab (so a row of ``adata.X`` can be sliced).
          * ``_resolved_gene_keys``: the vocab keys to look up genomic features
            for. Same length as ``valid_h5ad_indices``. Identical to the h5ad
            name in literal-match mode; uppercased in case-insensitive mode.
        """
        log.info("Aligning h5ad genes to gene mapping...")

        if self.case_insensitive:
            # Pre-index the vocab once by uppercase key. If two vocab entries
            # collapse to the same uppercase key, the last one wins — log a
            # warning so the user knows.
            vocab_upper: Dict[str, str] = {}
            for k in self.gene_mapping.keys():
                u = k.upper()
                if u in vocab_upper and vocab_upper[u] != k:
                    log.warning(
                        f"case-insensitive collision: vocab keys {vocab_upper[u]!r} "
                        f"and {k!r} both map to {u!r}; keeping {k!r}"
                    )
                vocab_upper[u] = k
            log.info(f"  Built uppercase lookup for {len(vocab_upper)} vocab entries")

            self.valid_h5ad_indices = []
            self._resolved_gene_keys: List[str] = []
            for h5ad_idx, gene in enumerate(self.h5ad_gene_names):
                resolved = vocab_upper.get(gene.upper())
                if resolved is not None:
                    self.valid_h5ad_indices.append(h5ad_idx)
                    self._resolved_gene_keys.append(resolved)
        else:
            self.valid_h5ad_indices = []
            self._resolved_gene_keys = []
            for h5ad_idx, gene in enumerate(self.h5ad_gene_names):
                if gene in self.gene_mapping:
                    self.valid_h5ad_indices.append(h5ad_idx)
                    self._resolved_gene_keys.append(gene)

        self.valid_h5ad_indices = np.array(self.valid_h5ad_indices, dtype=np.int64)

        log.info(f"  H5AD genes: {len(self.h5ad_gene_names)}")
        log.info(f"  Gene mapping vocabulary: {len(self.gene_mapping)}")
        log.info(f"  Genes in common: {len(self.valid_h5ad_indices)}"
                 f"{' (case-insensitive)' if self.case_insensitive else ''}")

        if len(self.valid_h5ad_indices) == 0:
            raise ValueError("No genes found in common between h5ad file and gene mapping!")

    def _prepare_gene_arrays(self):
        """Prepare gene mapping arrays for efficient cell processing."""
        self.gene_protein_ids = []
        self.gene_chroms = []
        self.gene_starts = []
        self.aligned_gene_names = []

        for i, h5ad_idx in enumerate(self.valid_h5ad_indices):
            # ``_resolved_gene_keys`` is the vocab-side key; the displayed
            # name stays as it appeared in the h5ad for downstream sanity.
            vocab_key = self._resolved_gene_keys[i]
            mapping = self.gene_mapping[vocab_key]
            self.gene_protein_ids.append(mapping['protein_embedding_id'])
            self.gene_chroms.append(mapping['chromosome_id'])
            self.gene_starts.append(mapping['location'])
            self.aligned_gene_names.append(self.h5ad_gene_names[h5ad_idx])

        self.gene_protein_ids = np.array(self.gene_protein_ids, dtype=np.int64)
        self.gene_chroms = np.array(self.gene_chroms, dtype=np.int64)
        self.gene_starts = np.array(self.gene_starts, dtype=np.int64)

    def __len__(self):
        return self.n_cells

    def __getitem__(self, idx):
        """Get a processed cell sample."""
        cell_expr = self.adata.X[idx]
        if hasattr(cell_expr, 'toarray'):
            cell_expr = cell_expr.toarray().flatten()
        else:
            cell_expr = np.array(cell_expr).flatten()

        valid_expr = cell_expr[self.valid_h5ad_indices]

        counts_batch = torch.from_numpy(valid_expr.astype(np.float32)).unsqueeze(0)

        log_expr = torch.log1p(counts_batch)
        expr_sum = log_expr.sum(dim=1, keepdim=True)
        expr_sum = torch.clamp(expr_sum, min=1e-8)
        weights_batch = log_expr / expr_sum

        result = sample_cell_sentences_mapping_gene(
            counts=counts_batch,
            batch_weights=weights_batch,
            gene_protein_ids=self.gene_protein_ids,
            gene_chroms=self.gene_chroms,
            gene_starts=self.gene_starts,
            pad_length=self.pad_length,
            positive_sample_num=self.positive_sample_num,
            negative_sample_num=self.negative_sample_num,
            mask_prop=self.mask_prop,
            sample_size=self.sample_size,
            cls_token_idx=self.cls_token_idx,
            chrom_token_offset=self.chrom_token_offset,
            chrom_token_right_idx=self.chrom_token_right_idx,
            pad_token_idx=self.pad_token_idx,
            seed=idx,
        )

        result['idx'] = idx
        result['dataset_num'] = 0

        return result


# Aliases bridging dataset-style NCBI binomial species names to the legacy
# informal keys used in older single-species vocab JSONs. The multi-species
# vocab JSON keeps the new species (callithrix_jacchus, pan_troglodytes)
# under their dataset names already, so no alias is needed for those.
SPECIES_VOCAB_ALIASES: Dict[str, str] = {
    "homo_sapiens": "human",
    "mus_musculus": "mouse",
    "danio_rerio": "zebrafish",
    "microcebus_murinus": "mouse_lemur",
    "sus_scrofa": "pig",
    "sus_scrofa_domesticus": "pig",
}


def load_gene_mapping(gene_mapping_path: str) -> Dict:
    """Load a gene-mapping JSON.

    Supports both the legacy single-species vocab
    (``gene_data/human_gene_dict.json``) and the multi-species vocab
    (``gene_data/all_species_gene_dict_multi_2025-11-08.json``). Returns the
    raw top-level dict — species selection happens inside
    :class:`H5ADDataset` (or via ``H5ADDataset(species=...)``).

    The multi-species JSON has 10 top-level species keys:
        human, mouse, pig, macaca_fascicularis, zebrafish, frog,
        mouse_lemur, macaca_mulatta, callithrix_jacchus, pan_troglodytes.
    """
    log.info(f"Loading gene mapping from {gene_mapping_path}")
    with open(gene_mapping_path, 'r') as f:
        gene_mapping_data = json.load(f)
    log.info(f"Top-level species keys: {sorted(gene_mapping_data.keys())}")
    return gene_mapping_data
