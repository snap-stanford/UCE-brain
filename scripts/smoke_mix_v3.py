#!/usr/bin/env python
"""GPU smoke test for a UCE-brain model trained with the v2026-09 vocabulary.

Loads a checkpoint (Hub repo id or local directory) through
``uce_brain.inference``, embeds a random subsample of a real h5ad and checks
the result: finite values, unit L2 norm, expected width, per-cell training loss
below ln 2, and k-nearest-neighbour label accuracy well above chance. With
``--control-chrom-offset`` the same cells are also embedded with a *wrong*
chromosome-token offset, to show that the offset has to come from the
checkpoint. ``--notebook`` additionally executes the example notebook with the
same inputs (nbclient) and prints its text outputs.

Run it on a GPU node (see scripts/smoke_mix_v3.sbatch); the last stdout line is
a JSON summary.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

from uce_brain.data import load_gene_mapping, read_h5ad_subsampled
from uce_brain.inference import (
    build_dataset,
    compute_per_cell_loss,
    embed_dataset,
    load_cell_sentence_params,
    load_model,
    resolve_checkpoint,
)

log = logging.getLogger("smoke_mix_v3")


def knn_eval(emb: np.ndarray, labels: np.ndarray, k: int = 15, folds: int = 5, seed: int = 0) -> dict:
    """Cosine kNN label transfer with k-fold cross-validation on the given cells."""
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.neighbors import KNeighborsClassifier

    labels = np.asarray(labels).astype(str)
    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    pred = cross_val_predict(clf, emb, labels, cv=cv)
    counts = np.unique(labels, return_counts=True)[1]
    return {
        "n_classes": int(len(counts)),
        "acc": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro")),
        "majority_baseline": float(counts.max() / counts.sum()),
        "chance": float(1.0 / len(counts)),
    }


def run_notebook(path: Path, params: dict, timeout: int = 1800) -> None:
    """Execute the example notebook with its parameter cell replaced by ``params``."""
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(str(path), as_version=4)
    param_cell = next(c for c in nb.cells if c.cell_type == "code")
    param_cell.source = "\n".join(f"{k} = {v!r}" for k, v in params.items())
    log.info(f"Executing {path} with parameters:\n{param_cell.source}")
    client = NotebookClient(
        nb, timeout=timeout, kernel_name=nb.metadata.get("kernelspec", {}).get("name", "python3"),
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                text = out.get("text", "")
            elif out.get("output_type") in ("execute_result", "display_data"):
                text = out.get("data", {}).get("text/plain", "")
                if "image/png" in out.get("data", {}):
                    text = (text + " [image/png]").strip()
            else:
                text = f"[{out.get('output_type')}] {out.get('ename', '')}: {out.get('evalue', '')}"
            text = str(text).strip()
            if text:
                print(f"[notebook cell {i}] " + text[-1500:])
    log.info("Notebook executed without errors")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hub repo id or local checkpoint directory")
    ap.add_argument("--gene-mapping", default=None, help="Gene-mapping JSON (default: the one inside the checkpoint)")
    ap.add_argument("--h5ad", required=True, help="h5ad with raw counts (CSR X)")
    ap.add_argument("--species", required=True)
    ap.add_argument("--n-cells", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label-keys", nargs="+", default=["class", "subclass"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--loss-cells", type=int, default=512, help="Cells scored with the training objective")
    ap.add_argument("--control-chrom-offset", type=int, default=None, help="Re-embed with this (wrong) offset as a control")
    ap.add_argument("--notebook", default=None, help="Example notebook to execute with the same inputs")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        log.error("CUDA requested but not available; this smoke test is GPU-only")
        return 2
    summary: dict = {"model": args.model, "h5ad": args.h5ad, "species": args.species, "n_cells": args.n_cells}
    t0 = time.time()

    ckpt = resolve_checkpoint(args.model)
    params = load_cell_sentence_params(ckpt)
    summary["cell_sentence_params"] = params.as_kwargs()
    gene_mapping_path = args.gene_mapping
    if gene_mapping_path is None:
        from uce_brain.inference import find_gene_mapping

        gene_mapping_path = find_gene_mapping(ckpt)
        if gene_mapping_path is None:
            log.error("No gene mapping inside the checkpoint; pass --gene-mapping")
            return 2
    gene_mapping = load_gene_mapping(str(gene_mapping_path))

    adata = read_h5ad_subsampled(args.h5ad, n_cells=args.n_cells, seed=args.seed)
    model = load_model(ckpt, device=args.device)
    summary["vocab_size"] = int(model.config.vocab_size)
    summary["load_seconds"] = round(time.time() - t0, 1)

    # --- embeddings ---
    t1 = time.time()
    dataset = build_dataset(adata, gene_mapping, params, args.species, mask_prop=0.0)
    summary["genes_mapped"] = f"{len(dataset.aligned_gene_names)}/{adata.n_vars}"
    emb = embed_dataset(model, dataset, batch_size=args.batch_size, num_workers=args.num_workers, show_progress=False)
    summary["embed_seconds"] = round(time.time() - t1, 1)
    norms = np.linalg.norm(emb, axis=1)
    cos = emb[:256] @ emb[:256].T
    off_diag = cos[~np.eye(len(cos), dtype=bool)]
    summary["embeddings"] = {
        "shape": list(emb.shape),
        "finite": bool(np.isfinite(emb).all()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "pairwise_cos_mean": float(off_diag.mean()),
        "pairwise_cos_max": float(off_diag.max()),
    }
    checks = {
        "shape": emb.shape == (adata.n_obs, model.config.output_embedding_dim),
        "finite": summary["embeddings"]["finite"],
        "unit_norm": bool(np.abs(norms - 1.0).max() < 1e-3),
        "not_collapsed": bool(off_diag.mean() < 0.95),
    }

    # --- kNN label transfer ---
    summary["knn"] = {}
    for key in args.label_keys:
        if key not in adata.obs.columns:
            log.warning(f"label column {key!r} not in obs; skipping")
            continue
        res = knn_eval(emb, adata.obs[key].values)
        summary["knn"][key] = res
        checks[f"knn_{key}_above_chance"] = res["acc"] > max(2 * res["chance"], res["majority_baseline"] + 0.1)
        log.info(f"kNN[{key}]: acc={res['acc']:.3f} macroF1={res['macro_f1']:.3f} "
                 f"(majority {res['majority_baseline']:.3f}, chance {res['chance']:.3f}, {res['n_classes']} classes)")

    # --- training objective on a subset ---
    n_loss = min(args.loss_cells, adata.n_obs)
    loss_ds = build_dataset(adata[:n_loss].copy(), gene_mapping, params, args.species, mask_prop=None)
    loss = compute_per_cell_loss(model, loss_ds, batch_size=args.batch_size, num_workers=args.num_workers, show_progress=False)
    summary["per_cell_loss"] = {"n": int(n_loss), "mean": float(loss.mean()), "std": float(loss.std()), "chance": float(np.log(2))}
    checks["loss_below_chance"] = bool(loss.mean() < 0.6)

    # --- control: wrong chromosome-token offset ---
    if args.control_chrom_offset is not None:
        from dataclasses import replace

        wrong = replace(params, chrom_token_offset=args.control_chrom_offset, source="control")
        ctrl_ds = build_dataset(adata, gene_mapping, wrong, args.species, mask_prop=0.0)
        ctrl = embed_dataset(model, ctrl_ds, batch_size=args.batch_size, num_workers=args.num_workers, show_progress=False)
        summary["control_wrong_offset"] = {"chrom_token_offset": args.control_chrom_offset, "knn": {}}
        for key in summary["knn"]:
            res = knn_eval(ctrl, adata.obs[key].values)
            summary["control_wrong_offset"]["knn"][key] = {"acc": res["acc"], "macro_f1": res["macro_f1"]}
            log.info(f"control offset {args.control_chrom_offset} kNN[{key}]: acc={res['acc']:.3f} macroF1={res['macro_f1']:.3f}")
        summary["control_wrong_offset"]["cos_to_correct_mean"] = float(np.mean(np.sum(ctrl * emb, axis=1)))

    summary["checks"] = checks
    summary["all_checks_passed"] = all(checks.values())
    summary["total_seconds"] = round(time.time() - t0, 1)

    # --- notebook ---
    if args.notebook:
        del model
        torch.cuda.empty_cache()
        label_keys = [k for k in ("class", "neighborhood", "subclass", "cell_type") if k in adata.obs.columns][:2]
        try:
            run_notebook(Path(args.notebook), {
                "MODEL": str(ckpt),
                "GENE_MAPPING_PATH": str(gene_mapping_path),
                "H5AD_PATH": args.h5ad,
                "SPECIES": args.species,
                "LABEL_KEYS": label_keys,
                "MAX_CELLS": args.n_cells,
                "BATCH_SIZE": args.batch_size,
                "NUM_WORKERS": min(args.num_workers, 4),
                "SUBSAMPLE_SEED": args.seed,
            })
            summary["notebook"] = "ok"
        except Exception as e:  # report, but keep the smoke summary
            log.exception("Notebook execution failed")
            summary["notebook"] = f"failed: {type(e).__name__}: {str(e)[:300]}"
            summary["all_checks_passed"] = False

    print("SMOKE_SUMMARY " + json.dumps(summary))
    return 0 if summary["all_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
