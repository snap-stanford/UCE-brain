
## Installation

Requires [uv](https://docs.astral.sh/uv/). To build the envrionment and install the package, run:

```bash
git clone https://github.com/snap-stanford/UCE-brain.git && cd UCE-brain
uv sync
```


## Run notebook

We use a dataset on VPA-treated dorsal forebrain organoids dataset ([here](https://cellxgene.cziscience.com/collections/c2879de0-affc-496b-8e2b-f57ed9ec3c34)) for demo. To run the notebook, update the paths in the first cell to point to the downloaded H5AD file, model checkpoint, and gene mapping JSON file on your system. 

The dataset can be downloaded from the cellxgene portal linked above, and the model checkpoint can be obtained from the Hugging Face model hub at `KuanP/uce-brain-pilot-8l-512d` (which should be automatically downloaded when you run the notebook, or downloaded from [here](https://huggingface.co/KuanP/uce-brain-pilot-8l-512d)). The gene mapping JSON file is included in the repository under `gene_data/human_gene_dict.json`.

