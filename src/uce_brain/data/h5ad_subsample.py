"""Fast subsampled h5ad reader.

The default ``sc.read_h5ad(path)`` reads the entire X matrix sequentially —
fine when you want the whole dataset, but extremely wasteful when you only
need a few thousand cells out of hundreds of thousands.

The marmoset BICAN h5ad on disk is ~30 GB and takes ~4–5 minutes to load
fully. ``backed='r'`` doesn't help much: it still reads ``obs`` eagerly and
each row slice triggers small HDF5 reads.

This helper reads ONLY the requested rows of a CSR-encoded ``X``, plus the
matching slice of ``obs``, plus all of ``var`` (always small). Skips raw,
layers, obsm — none needed for embedding inference.

Speedup on the marmoset file: ~3 min full read → ~10 s for 5,000 rows.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

log = logging.getLogger(__name__)


def _read_dataframe(group: h5py.Group) -> pd.DataFrame:
    """Reconstruct a pandas DataFrame from an anndata-encoded h5py group."""
    encoding = group.attrs.get("encoding-type", "")
    if encoding == "dataframe":
        col_order = list(group.attrs.get("column-order", []))
        # Index column.
        idx_key = group.attrs.get("_index", "_index")
        if isinstance(idx_key, bytes):
            idx_key = idx_key.decode()
        index = _read_array_or_categorical(group[idx_key])
        df = pd.DataFrame(index=pd.Index(index))
        for c in col_order:
            c_str = c.decode() if isinstance(c, bytes) else str(c)
            df[c_str] = _read_array_or_categorical(group[c_str])
        return df
    # Fall back to treating every leaf as a column.
    return pd.DataFrame({k: _read_array_or_categorical(v) for k, v in group.items()})


def _read_array_or_categorical(node):
    """Read either a plain array dataset or an anndata categorical group."""
    if isinstance(node, h5py.Group):
        if node.attrs.get("encoding-type", "") == "categorical":
            codes = node["codes"][:]
            cats = _read_array_or_categorical(node["categories"])
            ordered = bool(node.attrs.get("ordered", False))
            return pd.Categorical.from_codes(codes, categories=cats, ordered=ordered)
        # Generic group: try to read as a dataset under itself, else give up.
        raise ValueError(f"Unsupported group at {node.name!r}")
    arr = node[:]
    # h5py loads strings as bytes; decode for pandas friendliness.
    if arr.dtype.kind == "O" or arr.dtype.kind == "S":
        return np.array([x.decode() if isinstance(x, bytes) else x for x in arr])
    return arr


def read_h5ad_subsampled(
    path: str,
    n_cells: Optional[int] = None,
    indices: Optional[np.ndarray] = None,
    seed: int = 0,
    keep_obs_columns: Optional[List[str]] = None,
) -> ad.AnnData:
    """Load only ``n_cells`` (or ``indices``) rows from a CSR-encoded h5ad.

    Args:
        path: Path to the h5ad file.
        n_cells: Number of cells to randomly subsample. Mutually exclusive with
            ``indices``. ``None`` means load every cell.
        indices: Explicit row indices to load (sorted; will be sorted if not).
        seed: RNG seed for random subsampling.
        keep_obs_columns: Restrict the ``.obs`` table to these columns (small
            speedup; default loads every column).

    Returns:
        AnnData with ``X`` (CSR), ``obs`` (subsampled rows), ``var`` (full).
    """
    with h5py.File(path, "r") as f:
        n_obs_total = f["X"].attrs.get("shape")
        if n_obs_total is not None:
            n_obs_total = int(n_obs_total[0])
        else:
            n_obs_total = int(f["X/indptr"].shape[0]) - 1

        # Pick row indices.
        if indices is None:
            if n_cells is None or n_cells >= n_obs_total:
                idx = np.arange(n_obs_total, dtype=np.int64)
            else:
                rng = np.random.default_rng(seed)
                idx = np.sort(rng.choice(n_obs_total, size=n_cells, replace=False))
        else:
            idx = np.sort(np.asarray(indices, dtype=np.int64))

        log.info(
            "Reading %d / %d rows from %s",
            len(idx), n_obs_total, path,
        )

        # --- X: read only the rows we need ---
        X_group = f["X"]
        encoding = X_group.attrs.get("encoding-type", "")
        if encoding not in ("csr_matrix", "csc_matrix"):
            raise ValueError(
                f"read_h5ad_subsampled only supports CSR/CSC X; got {encoding!r}. "
                f"Use sc.read_h5ad for dense files."
            )

        # Read the full indptr first (small: 8 bytes × n_obs).
        indptr_full = X_group["indptr"][:]
        # Per-row (start, end) spans we need.
        starts = indptr_full[idx]
        ends = indptr_full[idx + 1]
        lengths = (ends - starts).astype(np.int64)
        new_indptr = np.empty(len(idx) + 1, dtype=indptr_full.dtype)
        new_indptr[0] = 0
        new_indptr[1:] = np.cumsum(lengths)
        nnz = int(new_indptr[-1])

        # Allocate output and copy each row's (indices, data) slice.
        indices_ds = X_group["indices"]
        data_ds = X_group["data"]
        new_indices = np.empty(nnz, dtype=indices_ds.dtype)
        new_data = np.empty(nnz, dtype=data_ds.dtype)
        pos = 0
        for s, e in zip(starts.tolist(), ends.tolist()):
            if e > s:
                n = e - s
                new_indices[pos:pos + n] = indices_ds[s:e]
                new_data[pos:pos + n] = data_ds[s:e]
                pos += n

        # Shape: anndata stores attrs.shape if present, otherwise infer.
        if "shape" in X_group.attrs:
            n_var = int(X_group.attrs["shape"][1])
        else:
            n_var = int(indices_ds[:].max()) + 1  # fallback, scans the full indices
        X_sub = csr_matrix(
            (new_data, new_indices, new_indptr), shape=(len(idx), n_var)
        )

        # --- obs: read minimally ---
        obs_full = _read_dataframe(f["obs"])
        if keep_obs_columns is not None:
            obs_full = obs_full[[c for c in keep_obs_columns if c in obs_full.columns]]
        obs_sub = obs_full.iloc[idx].copy()
        obs_sub.index = obs_sub.index.astype(str)

        # --- var: always small ---
        var = _read_dataframe(f["var"])
        var.index = var.index.astype(str)

    adata = ad.AnnData(X=X_sub, obs=obs_sub, var=var)
    log.info("Built AnnData: %s, X nnz=%d", adata.shape, X_sub.nnz)
    return adata
