
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

The three models above the line share one tokenisation (legacy vocabulary, species keys such as `human` / `mouse`, chromosome-token offset 1000 = the `H5ADDataset` defaults). The v2026-09 models (data mix v2 / v3) use a different vocabulary (266,403 tokens, species keys are NCBI binomials such as `homo_sapiens` / `macaca_mulatta`, chromosome-token offset 237411), so their tokenisation parameters must be read from the checkpoint rather than defaulted; see the next section.

## v2026-09 models (data mix v2 / v3)

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
- `load_model` re-verifies every tensor against `model.safetensors` after `from_pretrained` (some transformers versions re-initialise submodules after loading) and logs how many had to be restored. `uce_brain.inference.compute_per_cell_loss` scores cells with the training objective; a mean well below ln 2 = 0.693 confirms weights and tokenisation agree.
- `scripts/smoke_mix_v3.sbatch` is a one-GPU Slurm job running these checks (embedding shape / finiteness / unit norm, per-cell loss, kNN label accuracy, a wrong-offset control, and the notebook) on a 2,000-cell subsample.