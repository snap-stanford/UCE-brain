
## Installation

Requires [uv](https://docs.astral.sh/uv/). To build the envrionment and install the package, run:

```bash
git clone https://github.com/snap-stanford/UCE-brain.git && cd UCE-brain
uv sync
```


## Run notebook

We use a dataset on VPA-treated dorsal forebrain organoids dataset ([here](https://cellxgene.cziscience.com/collections/c2879de0-affc-496b-8e2b-f57ed9ec3c34)) for demo. To run the notebook, update the paths in the first cell to point to the downloaded H5AD file, model checkpoint, and gene mapping JSON file on your system. 

The dataset can be downloaded from the cellxgene portal linked above, and the model checkpoint can be obtained from the Hugging Face model hub at `KuanP/uce-brain-pilot-8l-512d` (which should be automatically downloaded when you run the notebook, or downloaded from [here](https://huggingface.co/KuanP/uce-brain-pilot-8l-512d)). The gene mapping JSON file is included in the repository under `gene_data/human_gene_dict.json`.

## Model weights
- Brain pilot models: `KuanP/uce-brain-pilot-8l-512d` (8 layers, 512 embedding dimension, trained on brain organoid data, [here](https://huggingface.co/KuanP/uce-brain-pilot-8l-512d))
- cxg2025 models: `KuanP/uce-cxg-2025-baseline-8l-512d` (8 layers, 512 embedding dimension, trained on all data from cxg2025, [here](https://huggingface.co/KuanP/uce-cxg-2025-baseline-8l-512d))
- Multi-species model: `KuanP/uce-multispecies-2025-11-08` (8 layers, 512 embedding dimension, 10 species; gene dict `gene_data/all_species_gene_dict_multi_2025-11-08.json`, also shipped in the repo as `gene_mapping.json`; see `notebooks/visualize_embeddings_{marmoset,mouse}_multispecies.ipynb`, [here](https://huggingface.co/KuanP/uce-multispecies-2025-11-08))
- Data mix v2: `KuanP/brain-uce-pilot-mix-v2` (8 layers, 512 embedding dimension; uniform mix of the v2026-09 corpus, 13 species / 26 caches / 147.7M cells, 262,144 steps at global batch 256; gene dict `gene_data/all_species_gene_dict_v2026-09.json`, also shipped in the repo under `vocab/`, [here](https://huggingface.co/KuanP/brain-uce-pilot-mix-v2))
- Data mix v3: `KuanP/brain-uce-pilot-mix-v3` (8 layers, 512 embedding dimension; 13-species designed mixture ("mixture C") of the v2026-09 corpus, 131,072 steps at global batch 512; gene dict `gene_data/all_species_gene_dict_v2026-09.json`, also shipped in the repo under `vocab/`, [here](https://huggingface.co/KuanP/brain-uce-pilot-mix-v3))
- Data mix v3, v2 architecture: `KuanP/brain-uce-pilot-mix-v3-d768l14v2` (14 layers, d_model 768, **768 embedding dimension**; the v2 architecture described below, ~114.0M trainable parameters plus the frozen gene-token table; the same mixture C as data mix v3, 262,144 steps at global batch 256; gene dict `gene_data/all_species_gene_dict_v2026-09.json`, also shipped in the repo under `vocab/`; needs this version of UCE-brain or later and `torch>=2.4`, see [below](#data-mix-v3-v2-architecture-d768l14v2), [here](https://huggingface.co/KuanP/brain-uce-pilot-mix-v3-d768l14v2))

The first three models share one tokenisation (legacy vocabulary, species keys such as `human` / `mouse`, chromosome-token offset 1000 = the `H5ADDataset` defaults). The v2026-09 models (data mix v2 / v3 / v3-d768l14v2) use a different vocabulary (266,403 tokens, species keys are NCBI binomials such as `homo_sapiens` / `macaca_mulatta`, chromosome-token offset 237411), so their tokenisation parameters must be read from the checkpoint rather than defaulted; see the next section.

## v2026-09 models (data mix v2 / v3 / v3-d768l14v2)

`notebooks/visualize_embeddings_mix_v3.ipynb` is the runnable example (edit the first cell). The pieces it uses:

```python
from uce_brain.data import load_gene_mapping, read_h5ad_subsampled
from uce_brain.inference import resolve_checkpoint, load_model, load_cell_sentence_params, embed_adata

ckpt = resolve_checkpoint("KuanP/brain-uce-pilot-mix-v3")     # Hub repo id, or a local checkpoint directory
params = load_cell_sentence_params(ckpt)                       # pad_length, chrom_token_offset, ... from config.yaml / config.json
model = load_model(ckpt, device="cuda")                        # config.json + model.safetensors (frozen gene table included)
gene_mapping = load_gene_mapping("gene_data/all_species_gene_dict_v2026-09.json")

adata = read_h5ad_subsampled("cells.h5ad", n_cells=5000)       # raw UMI counts; reads only the sampled rows of a CSR X
adata.obsm["X_uce"] = embed_adata(model, adata, gene_mapping, params, species="macaca_mulatta")
```

- `species` accepts any key of the gene dict or an alias: `human`/`homo_sapiens`, `mouse`/`mus_musculus`, `macaque`/`rhesus`/`macaca_mulatta`, `marmoset`/`callithrix_jacchus`, `chimp`/`pan_troglodytes`, `rat`, `pig`, `zebrafish`, `mouse_lemur`, `opossum`, `owl_monkey`, `tree_shrew`. Pig-tailed macaque (`macaca_nemestrina`) data uses the `macaca_mulatta` vocabulary, as in training. The same aliases work with the legacy gene dicts.
- Gene symbols are matched case-insensitively by default in `embed_adata` (every vocabulary key is upper-case), so mouse/rat sentence-case symbols need no preprocessing. Ensembl `var_names` are resolved through `var["feature_name"]` (or pass `gene_symbol_column`).
- `load_cell_sentence_params` resolves each parameter from explicit `overrides`, then the checkpoint's `config.json`, then the training run's `config.yaml` next to the weights (the Hub repos ship it); it raises if a parameter is available from none of them, because a wrong chromosome-token offset does not raise: the embeddings come out unit-norm and only slightly different, so nothing downstream flags the mistake.
- `load_model` re-verifies every tensor against `model.safetensors` after `from_pretrained` (some transformers versions re-initialise submodules after loading) and logs how many had to be restored. It is strict: a `config.json` key this version does not define, or an option value it does not implement, raises `uce_brain.inference.UnsupportedCheckpointError` (a `ValueError`, and also an `AssertionError`, which is what such configs raised before); a tensor in the file the model has no slot for, or a model tensor the file lacks, raises a `RuntimeError`; a checkpoint without safetensors weights (`model.safetensors`, or the shards of `model.safetensors.index.json`) raises a `FileNotFoundError`, because nothing could be verified. The messages name the likely cause (a checkpoint from a newer architecture than the installed UCE-brain implements); a partly loaded model would still emit unit-norm embeddings. `strict=False` downgrades all of this to warnings (debugging only). Every released checkpoint passes. The frozen gene-token table comes back with `requires_grad=False` (`config.embedding_requires_grad`), so the logged trainable count is the trained part of the network. `uce_brain.inference.compute_per_cell_loss` scores cells with the training objective; a mean well below ln 2 = 0.693 confirms weights and tokenisation agree.
- `scripts/smoke_mix_v3.sbatch` is a one-GPU Slurm job running these checks (embedding shape / finiteness / unit norm, per-cell loss, kNN label accuracy, a wrong-offset control, and the notebook) on a 2,000-cell subsample. `SMOKE_MODEL`, `SMOKE_H5AD`, `SMOKE_SPECIES`, `SMOKE_LABEL_KEYS`, `SMOKE_CONTROL_OFFSET` and `SMOKE_NOTEBOOK` override the data mix v3 / HMBA-macaque defaults (see the script header).

## Data mix v3, v2 architecture (d768l14v2)

`KuanP/brain-uce-pilot-mix-v3-d768l14v2` is trained on the same designed mixture as data mix v3 and tokenised exactly like data mix v2 / v3 (same gene dict, chromosome-token offset 237411, 2,048-token sentences of 1,024 sampled genes), so the code of the previous section runs unchanged. Only the repo id changes, and the embeddings are **768-dimensional**:

```python
ckpt = resolve_checkpoint("KuanP/brain-uce-pilot-mix-v3-d768l14v2")
params = load_cell_sentence_params(ckpt)
model = load_model(ckpt, device="cuda")
gene_mapping = load_gene_mapping("gene_data/all_species_gene_dict_v2026-09.json")
adata.obsm["X_uce"] = embed_adata(model, adata, gene_mapping, params, species="homo_sapiens")  # [n_cells, 768], unit norm
```

```bash
SMOKE_MODEL=KuanP/brain-uce-pilot-mix-v3-d768l14v2 \
    sbatch -A infolab -p il -q il-lo --gpus=a100:1 --cpus-per-task=8 --mem=64G --time=00:40:00 scripts/smoke_mix_v3.sbatch
```

**Version requirements.** The model needs this version of UCE-brain (the v2 options in `UCEConfig` / `uce_brain.model.modeling`) or later. Earlier versions stop while reading `config.json` (`AssertionError: input_projection_type must be 'linear' or 'mlp', got linear_gelu_ln`). It also needs `torch>=2.4` (`nn.RMSNorm`); it has been tested with torch 2.10 (the `uv.lock` pin, with transformers 5.2) and torch 2.13 (with transformers 5.15). The older models are unaffected. Every v2 key defaults to the original layout, and all six older models (`uce-brain-pilot-8l-512d`, `uce-cxg-2025-baseline-8l-512d`, `uce-multispecies-2025-11-08`, data mix v2, data mix v3 and the 100M d768/12 run) give bitwise-identical embeddings and losses before and after the change, both on fixed token batches in fp32 and through `embed_adata` / `compute_per_cell_loss` (bf16 autocast) on real AnnData.

**Architecture.** The `config.json` keys select it. The original layout is the default of every key.

| | original layout (8-layer models) | v2 (`d768l14v2`) |
|---|---|---|
| gene token (frozen ESM2, 5,120-d) | LayerNorm | L2-normalised, parameter-free (`token_norm: l2`) |
| input projector | Linear + LayerNorm + GELU | Linear + GELU + LayerNorm (`input_projection_type: linear_gelu_ln`) |
| positional encoding | sinusoidal | sinusoidal x 1/sqrt(768) (`pe_scale: 0.0360844`) |
| encoder | `nn.TransformerEncoder`: post-norm, GELU FFN, biases | `UCEEncoder` (`encoder_impl: uce`): 14 pre-norm layers, d_model 768, 6 heads, RMSNorm, QK-norm, SwiGLU FFN 2048, no biases, final RMSNorm |
| cell embedding | Linear + LayerNorm on every token, CLS row, L2 norm | CLS row, MLP head 768 -> 1024 -> 768 -> 768 (Linear + LayerNorm + GELU + Dropout each) -> Linear(768, 768), L2 norm (`cell_head_type: mlp`) |
| decoder target genes (loss only) | the encoder's input projector | own Linear + GELU + LayerNorm (`separate_gene_layer`); decoder 1536 -> 2048 -> 512 -> 128 -> 1, no input dropout |

The model has 114.0M trainable parameters plus the frozen 266,403 x 5,120 gene-token table. Training: 262,144 steps at global batch 256 on 8 A100s, lr 5e-5 cosine to 1.5e-5, dropout 0.05, mask 0.20, with negative targets drawn from genes expressed in each cell's source dataset.

**Notes.**
- `load_cell_sentence_params` returns `mask_prop=0.2` (the training value). It applies only to `compute_per_cell_loss`; `embed_adata` masks nothing. `compute_per_cell_loss` draws negatives from the genes of the input AnnData that map to the vocabulary and have zero counts in the cell, while training drew them only from genes detected in the cell's source dataset, so its value approximates rather than reproduces the training loss (about 0.37 on HMBA macaque cells, 0.36 on human striatum).
- The embeddings equal the training code's. On the same token batches, with the same torch build (torch 2.13), the training suite and UCE-brain agree bitwise, in fp32 and under bf16 autocast, and with the same per-cell sentence seed UCE-brain reproduces the benchmark's cached embeddings (per-cell cosine 1.0). On the `uv.lock` stack (torch 2.10, transformers 5.2) the agreement is numerical rather than bitwise: max abs difference 6e-8 in fp32, and per-cell cosine >= 0.99998 under bf16 autocast, to the suite and to the cached embeddings (the same size as bf16 against fp32 within one stack). torch 2.10 may warn `Mismatch dtype between input and weight ... Cannot dispatch to fused implementation` from the QK RMSNorm under autocast; it is harmless. UCE-brain seeds each cell's 1,024-gene draw with the cell's row position in the AnnData; the benchmark seeds it with the dataset row. So a cell embedded in a different AnnData, or at a different position, gets a different but equally valid draw: mean per-cell cosine 0.985-0.988 to the cached embedding, and kNN label-transfer accuracy within 0.006 of it.
- On the 2,000-cell smoke subsamples, cosine kNN accuracy is 0.980 (HMBA macaque class) and 0.884 (subclass), against 0.930 and 0.738 for data mix v3. On human striatum (cell type) it is 0.756, against 0.710.
- The wrong-offset control of the smoke script separates poorly for this model. With chromosome-token offset 1000, embeddings stay close (mean cosine 0.85 on macaque, 0.996 on human) and kNN accuracy is unchanged. On human data even the per-cell loss does not move (0.3551 at both offsets; macaque 0.385 against 0.373). A correct offset therefore cannot be inferred from kNN scores or the loss; take it from the checkpoint (`load_cell_sentence_params`) as above.
