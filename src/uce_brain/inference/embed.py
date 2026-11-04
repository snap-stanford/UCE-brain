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


def _dataloader(dataset: H5ADDataset, batch_size: int, num_workers: int, trim_to_multiple: int = 0) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=UCEDataCollator(trim_to_multiple=trim_to_multiple),
        pin_memory=torch.cuda.is_available(),
    )


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _check_inputs(model: UCEForExpressionPrediction, dataset: H5ADDataset, trim_to_multiple: int) -> int:
    """Warn when ``dataset`` was not tokenised for ``model``; return the padding trim to apply.

    Only logs; it never changes a result. Every released checkpoint was trained with
    ``pad_length == max_sequence_length``, so a mismatch almost always means the dataset
    was not built from the checkpoint's parameters (e.g. the :class:`H5ADDataset`
    defaults, which are the legacy 1,024-gene / 2,048-token tokenisation).
    """
    pad_length = getattr(dataset, "pad_length", None)
    context = getattr(model.config, "max_sequence_length", None)
    if pad_length is not None and context is not None and pad_length != context:
        log.warning(
            f"The dataset's pad_length ({pad_length}) differs from the model's context "
            f"(max_sequence_length={context}), so it was probably not tokenised as in training. "
            "Build it with build_dataset(adata, gene_mapping, load_cell_sentence_params(ckpt), species)."
        )
    reduction = getattr(model.config, "embedding_reduction", "cls")
    if trim_to_multiple and reduction != "cls":
        log.warning(f"trim_to_multiple is ignored for embedding_reduction={reduction!r}: "
                    "mean pooling averages over the padding, so trimming would change the result.")
        return 0
    return trim_to_multiple


@torch.inference_mode()
def embed_dataset(
    model: UCEForExpressionPrediction,
    dataset: H5ADDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    bf16: bool = True,
    show_progress: bool = True,
    trim_to_multiple: int = 0,
) -> np.ndarray:
    """Cell embeddings for every cell of ``dataset``, in dataset order.

    The model L2-normalises the CLS output, so rows have unit norm and dot
    products are cosine similarities. ``bf16`` enables bf16 autocast on CUDA
    (the precision used in training). ``trim_to_multiple`` (e.g. 256; default
    0 = off) drops the all-padding columns of each batch (see
    :class:`~uce_brain.data.UCEDataCollator`). It saves time when the sentences
    fill only part of ``pad_length`` (human or macaque data) and little on
    genomes with many chromosome ids (tree shrew). It applies to CLS-pooled
    models (all released ones) and is ignored for mean pooling. Results agree
    with the untrimmed ones up to floating-point rounding; they were bitwise
    identical in our A100 tests, but that is not guaranteed on every GPU or
    torch build.

    Returns:
        ``float32 [n_cells, output_embedding_dim]``.
    """
    trim_to_multiple = _check_inputs(model, dataset, trim_to_multiple)
    device = _model_device(model)
    embeddings, indices = [], []
    with _autocast(device, bf16):
        for batch in tqdm(_dataloader(dataset, batch_size, num_workers, trim_to_multiple), desc="Extracting embeddings", disable=not show_progress):
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
    trim_to_multiple: int = 0,
) -> np.ndarray:
    """Per-cell expression-prediction loss (the training objective), in dataset order.

    BCE-with-logits over ``positive_sample_num`` expressed + ``negative_sample_num``
    non-expressed target genes. Chance is ``ln 2 = 0.693``; a trained model scores
    well below that, so this is the quickest check that weights and tokenisation
    line up. Build the dataset with the training ``mask_prop`` (``mask_prop=None``).
    ``trim_to_multiple`` as in :func:`embed_dataset`.
    """
    trim_to_multiple = _check_inputs(model, dataset, trim_to_multiple)
    device = _model_device(model)
    losses, indices = [], []
    with _autocast(device, bf16):
        for batch in tqdm(_dataloader(dataset, batch_size, num_workers, trim_to_multiple), desc="Scoring cells", disable=not show_progress):
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
    trim_to_multiple: int = 0,
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
    return embed_dataset(model, dataset, batch_size=batch_size, num_workers=num_workers, bf16=bf16,
                         show_progress=show_progress, trim_to_multiple=trim_to_multiple)
