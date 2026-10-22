# UCE-brain

Cell embeddings from UCE-style models trained on brain single-cell data. This repository loads the released checkpoints and embeds an AnnData file of raw counts.

## Installation

Requires [uv](https://docs.astral.sh/uv/). To build the environment and install the package, run:

```bash
git clone https://github.com/snap-stanford/UCE-brain.git && cd UCE-brain
uv sync
```

The 14-layer models need `torch>=2.4`; `uv sync` installs a suitable version.

## Quick start

```python
import scanpy as sc
from uce_brain.data import load_gene_mapping
from uce_brain.inference import resolve_checkpoint, load_cell_sentence_params, load_model, embed_adata

ckpt = resolve_checkpoint("KuanP/brain-uce-pilot-mix-v3-d768l14v2-bican")  # Hub repo id or a local directory
params = load_cell_sentence_params(ckpt)        # tokenisation settings stored with the checkpoint
model = load_model(ckpt, device="cuda")
gene_mapping = load_gene_mapping("gene_data/all_species_gene_dict_v2026-09.json")

adata = sc.read_h5ad("cells.h5ad")              # raw UMI counts
adata.obsm["X_uce"] = embed_adata(model, adata, gene_mapping, params, species="homo_sapiens")
# -> [n_cells, 768], L2-normalised
```

- **Input.** `embed_adata` expects raw UMI counts. If `adata.X` holds normalised values and the counts are in `adata.raw`, pass `use_raw=True`. Gene symbols are read from `var_names`, or from `var["feature_name"]` when `var_names` are Ensembl ids (or pass `gene_symbol_column`).
- **Species.** `species` takes a key of the gene dictionary or a common alias: `human` / `homo_sapiens`, `mouse` / `mus_musculus`, `macaque` / `macaca_mulatta`, `marmoset` / `callithrix_jacchus`, `chimp`, `rat`, `pig`, `zebrafish`, `mouse_lemur`, `opossum`, `owl_monkey`, `tree_shrew`.
- **Tokenisation.** Always take the tokenisation settings from the checkpoint with `load_cell_sentence_params`, as above. A wrong setting does not raise an error; it only gives slightly worse embeddings.
- **Gene dictionary.** Use the dictionary listed for the model in the table below. The Hub repos of the v2026-09 models also ship it under `vocab/`, and `uce_brain.inference.find_gene_mapping(ckpt)` locates it.

## Demo

`notebooks/visualize_embeddings_brain.ipynb` embeds a public dataset of VPA-treated dorsal forebrain organoids with `KuanP/brain-uce-pilot-mix-v3-d768l14v2-bican` and plots a UMAP of the embeddings. Download the h5ad from the [CellxGene collection](https://cellxgene.cziscience.com/collections/c2879de0-affc-496b-8e2b-f57ed9ec3c34), or directly:

```bash
curl -L -O https://datasets.cellxgene.cziscience.com/281bf7bb-c74a-4daf-9069-f668e2b52c54.h5ad
```

Then set `H5AD_PATH` in the notebook's first code cell to the downloaded file.

`notebooks/visualize_embeddings_mix_v3.ipynb` runs any of the v2026-09 models on your own h5ad file and ends with a quick kNN label-transfer check.

## Models

| Model | Layers / width | Embedding | Gene dictionary | Trained on |
|---|---|---|---|---|
| [`KuanP/brain-uce-pilot-mix-v3-d768l14v2-bican`](https://huggingface.co/KuanP/brain-uce-pilot-mix-v3-d768l14v2-bican) | 14 / 768 | 768-d | `all_species_gene_dict_v2026-09.json` | the model below, further trained on BICAN brain data |
| [`KuanP/brain-uce-pilot-mix-v3-d768l14v2`](https://huggingface.co/KuanP/brain-uce-pilot-mix-v3-d768l14v2) | 14 / 768 | 768-d | `all_species_gene_dict_v2026-09.json` | 13 species, brain-weighted mixture (data mix v3) |
| [`KuanP/brain-uce-pilot-mix-v3`](https://huggingface.co/KuanP/brain-uce-pilot-mix-v3) | 8 / 512 | 512-d | `all_species_gene_dict_v2026-09.json` | 13 species, brain-weighted mixture (data mix v3) |
| [`KuanP/brain-uce-pilot-mix-v2`](https://huggingface.co/KuanP/brain-uce-pilot-mix-v2) | 8 / 512 | 512-d | `all_species_gene_dict_v2026-09.json` | 13 species, uniform mixture (data mix v2) |
| [`KuanP/uce-multispecies-2025-11-08`](https://huggingface.co/KuanP/uce-multispecies-2025-11-08) | 8 / 512 | 512-d | `all_species_gene_dict_multi_2025-11-08.json` | 10 species |
| [`KuanP/uce-cxg-2025-baseline-8l-512d`](https://huggingface.co/KuanP/uce-cxg-2025-baseline-8l-512d) | 8 / 512 | 512-d | `human_gene_dict.json` | human CellxGene data (2025) |
| [`KuanP/uce-brain-pilot-8l-512d`](https://huggingface.co/KuanP/uce-brain-pilot-8l-512d) | 8 / 512 | 512-d | `human_gene_dict.json` | human brain organoid data |

All gene dictionaries are in `gene_data/`. The four v2026-09 models share one vocabulary and use full species names such as `homo_sapiens`. The three older models use their own vocabularies and short species keys such as `human`. The multi-species notebooks (`notebooks/visualize_embeddings_{marmoset,mouse}_multispecies.ipynb`) show `uce-multispecies-2025-11-08`.

## Checking a checkpoint

- `load_model` checks every tensor against the checkpoint file and raises an error if the checkpoint needs a newer version of UCE-brain. In that case, update the repository.
- `compute_per_cell_loss` scores cells with the training objective. A mean well below ln 2 = 0.693 confirms that the weights and tokenisation match:

  ```python
  from uce_brain.inference import build_dataset, compute_per_cell_loss
  loss = compute_per_cell_loss(model, build_dataset(adata[:500].copy(), gene_mapping, params, "homo_sapiens"))
  print(loss.mean())
  ```
- `scripts/smoke_mix_v3.sbatch` runs these checks with a kNN label-transfer test on one GPU. Set `SMOKE_MODEL`, `SMOKE_H5AD` and `SMOKE_SPECIES` to choose the model and data; the script header lists all options.
