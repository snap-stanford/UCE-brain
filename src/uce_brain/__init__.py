"""
UCE-Brain: Universal Cell Embedding model for brain single-cell RNA-seq data.

This package provides the UCE model, data processing utilities, and inference
pipeline for extracting cell embeddings from h5ad files.
"""

from .model import UCEConfig, UCEModel, UCEForExpressionPrediction
from .data import UCEDataCollator, H5ADDataset, load_gene_mapping, read_h5ad_subsampled, resolve_species_key
from .inference import (
    CellSentenceParams,
    embed_adata,
    load_cell_sentence_params,
    load_model,
    resolve_checkpoint,
)

__all__ = [
    "UCEConfig",
    "UCEModel",
    "UCEForExpressionPrediction",
    "UCEDataCollator",
    "H5ADDataset",
    "load_gene_mapping",
    "read_h5ad_subsampled",
    "resolve_species_key",
    "CellSentenceParams",
    "embed_adata",
    "load_cell_sentence_params",
    "load_model",
    "resolve_checkpoint",
]
