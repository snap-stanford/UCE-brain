"""
Inference helpers: load a checkpoint (local directory or Hugging Face Hub repo
id) together with the tokenisation parameters it was trained with, and embed
an AnnData object.

    from uce_brain.inference import resolve_checkpoint, load_model, load_cell_sentence_params, embed_adata
    from uce_brain.data import load_gene_mapping, read_h5ad_subsampled

    ckpt = resolve_checkpoint("KuanP/brain-uce-pilot-mix-v3")
    model = load_model(ckpt, device="cuda")
    params = load_cell_sentence_params(ckpt)
    gene_mapping = load_gene_mapping("gene_data/all_species_gene_dict_v2026-09.json")
    adata = read_h5ad_subsampled("cells.h5ad", n_cells=5000)
    adata.obsm["X_uce"] = embed_adata(model, adata, gene_mapping, params, species="macaca_mulatta")
"""

from .embed import build_dataset, compute_per_cell_loss, embed_adata, embed_dataset
from .loader import (
    CELL_SENTENCE_KEYS,
    CellSentenceParams,
    find_gene_mapping,
    find_run_config,
    load_cell_sentence_params,
    load_model,
    resolve_checkpoint,
)

__all__ = [
    "CELL_SENTENCE_KEYS",
    "CellSentenceParams",
    "build_dataset",
    "compute_per_cell_loss",
    "embed_adata",
    "embed_dataset",
    "find_gene_mapping",
    "find_run_config",
    "load_cell_sentence_params",
    "load_model",
    "resolve_checkpoint",
]
