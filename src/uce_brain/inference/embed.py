"""
Batched embedding extraction (and per-cell loss) over an AnnData object.

Row ``i`` of the returned array is row ``i`` of ``adata``: the loader runs
unshuffled and the source row index rides along in every batch, so the result
is re-sorted by it before returning.
"""

import contextlib
import logging
from typing import Any, Dict, Optional, Union

import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..data import H5ADDataset, UCEDataCollator
from ..model import UCEForExpressionPrediction
from .loader import CellSentenceParams

log = logging.getLogger(__name__)


def build_dataset(
    adata: sc.AnnData,
    gene_mapping: Dict,
    params: CellSentenceParams,
    species: str,
    mask_prop: Optional[float] = None,
    case_insensitive: bool = True,
    gene_symbol_column: Optional[str] = None,
    use_raw: bool = False,
) -> H5ADDataset:
    """Tokenise ``adata`` with a checkpoint's cell-sentence parameters.

    Args:
        adata: Cells x genes with raw UMI counts in ``X`` (or ``raw`` with ``use_raw``).
        gene_mapping: Output of :func:`~uce_brain.data.load_gene_mapping`.
        params: From :func:`~uce_brain.inference.load_cell_sentence_params`.
        species: Any spelling accepted by :func:`~uce_brain.data.resolve_species_key`
            (``"human"``/``"homo_sapiens"``, ``"macaque"``/``"macaca_mulatta"``, ...).
        mask_prop: Fraction of expressed genes hidden from the sentence. ``None``
            keeps the training value; use 0.0 for embeddings.
        case_insensitive: Match gene symbols ignoring case. Every key of the
            v2026-09 vocabulary is upper-case, so this only adds matches for
            sentence-case symbols (mouse ``Sox17`` -> ``SOX17``).
        gene_symbol_column: ``adata.var`` column with symbols when ``var_names``
            are Ensembl ids (auto-detected when ``None``).
        use_raw: Take counts from ``adata.raw``.
    """
    kwargs: Dict[str, Any] = params.as_kwargs()
    if mask_prop is not None:
        kwargs["mask_prop"] = mask_prop
    dataset = H5ADDataset(
        adata,
        gene_mapping=gene_mapping,
        species=species,
        gene_symbol_column=gene_symbol_column,
        case_insensitive=case_insensitive,
        use_raw=use_raw,
        **kwargs,
    )
    log.info(
        f"{len(dataset.aligned_gene_names)} / {adata.n_vars} genes mapped to vocab[{dataset.species}] "
        f"(mask_prop={kwargs['mask_prop']}, chrom_token_offset={params.chrom_token_offset})"
    )
    return dataset


def _autocast(device: torch.device, bf16: bool):
    if bf16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _dataloader(dataset: H5ADDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=UCEDataCollator(),
        pin_memory=torch.cuda.is_available(),
    )


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.inference_mode()
def embed_dataset(
    model: UCEForExpressionPrediction,
    dataset: H5ADDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    bf16: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """Cell embeddings for every cell of ``dataset``, in dataset order.

    The model L2-normalises the CLS output, so rows have unit norm and dot
    products are cosine similarities. ``bf16`` enables bf16 autocast on CUDA
    (the precision used in training).

    Returns:
        ``float32 [n_cells, output_embedding_dim]``.
    """
    device = _model_device(model)
    embeddings, indices = [], []
    with _autocast(device, bf16):
        for batch in tqdm(_dataloader(dataset, batch_size, num_workers), desc="Extracting embeddings", disable=not show_progress):
            out = model.extract_cell_embeddings(
                input_ids=batch["input_ids"].to(device, non_blocking=True),
                attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                return_dict=True,
            )
            embeddings.append(out.cell_embedding.float().cpu().numpy())
            indices.append(batch["cell_indices"].numpy())

    emb = np.vstack(embeddings).astype(np.float32)
    order = np.argsort(np.concatenate(indices), kind="stable")
    emb = emb[order]
    norms = np.linalg.norm(emb, axis=1)
    log.info(f"Embedded {emb.shape[0]} cells -> {emb.shape}; L2 norm min={norms.min():.4f} max={norms.max():.4f}")
    return emb


@torch.inference_mode()
def compute_per_cell_loss(
    model: UCEForExpressionPrediction,
    dataset: H5ADDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    bf16: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """Per-cell expression-prediction loss (the training objective), in dataset order.

    BCE-with-logits over ``positive_sample_num`` expressed + ``negative_sample_num``
    non-expressed target genes. Chance is ``ln 2 = 0.693``; a trained model scores
    well below that, so this is the quickest check that weights and tokenisation
    line up. Build the dataset with the training ``mask_prop`` (``mask_prop=None``).
    """
    device = _model_device(model)
    losses, indices = [], []
    with _autocast(device, bf16):
        for batch in tqdm(_dataloader(dataset, batch_size, num_workers), desc="Scoring cells", disable=not show_progress):
            out = model(
                input_ids=batch["input_ids"].to(device, non_blocking=True),
                attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                target_gene_ids=batch["target_gene_ids"].to(device, non_blocking=True),
                target_expression=batch["target_expression"].to(device, non_blocking=True),
                return_dict=True,
            )
            losses.append(out.per_sample_losses.float().cpu().numpy())
            indices.append(batch["cell_indices"].numpy())

    loss = np.concatenate(losses).astype(np.float32)
    loss = loss[np.argsort(np.concatenate(indices), kind="stable")]
    log.info(f"Per-cell loss over {loss.shape[0]} cells: mean={loss.mean():.4f} (chance = ln 2 = 0.6931)")
    return loss


def embed_adata(
    model: UCEForExpressionPrediction,
    adata: sc.AnnData,
    gene_mapping: Dict,
    params: CellSentenceParams,
    species: str,
    batch_size: int = 32,
    num_workers: int = 4,
    bf16: bool = True,
    mask_prop: float = 0.0,
    case_insensitive: bool = True,
    gene_symbol_column: Optional[str] = None,
    use_raw: bool = False,
    show_progress: bool = True,
) -> np.ndarray:
    """:func:`build_dataset` + :func:`embed_dataset` in one call.

    Returns ``float32 [adata.n_obs, output_embedding_dim]`` with row ``i`` being
    ``adata`` row ``i``; attach it with ``adata.obsm["X_uce"] = ...``.
    """
    dataset = build_dataset(
        adata,
        gene_mapping,
        params,
        species,
        mask_prop=mask_prop,
        case_insensitive=case_insensitive,
        gene_symbol_column=gene_symbol_column,
        use_raw=use_raw,
    )
    return embed_dataset(model, dataset, batch_size=batch_size, num_workers=num_workers, bf16=bf16, show_progress=show_progress)
