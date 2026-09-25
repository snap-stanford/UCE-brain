#!/usr/bin/env bash
# GPU smoke test for the v2026-09-vocabulary models (data mix v2 / v3).
#
# Submit from the repository root:
#   sbatch -A infolab -p il -q il-lo --gpus=a100:1 --cpus-per-task=8 --mem=64G --time=00:40:00 \
#       --output=/dfs/ssd/project/uce-brain/logs/uce_brain_v3_smoke_%j.out scripts/smoke_mix_v3.sbatch
#
# Environment overrides: SMOKE_MODEL (Hub id or local dir), SMOKE_GENE_MAPPING,
# SMOKE_H5AD, SMOKE_SPECIES, SMOKE_N_CELLS, UCE_BRAIN_REPO.
#SBATCH --job-name=uce_brain_v3_smoke
#SBATCH --account=infolab
#SBATCH --partition=il
#SBATCH --qos=il-lo
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
set -euo pipefail

REPO="${UCE_BRAIN_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

# Model under test: a Hub repo id (downloaded to the HF cache) or a local checkpoint
# directory (config.json + model.safetensors + config.yaml), e.g. a local clone of the
# Hub repo or the training run directory. Override with SMOKE_MODEL.
MODEL="${SMOKE_MODEL:-KuanP/brain-uce-pilot-mix-v3}"
GENE_MAPPING="${SMOKE_GENE_MAPPING:-$REPO/gene_data/all_species_gene_dict_v2026-09.json}"
H5AD="${SMOKE_H5AD:-/dfs/ssd/project/uce-brain/data/eval_data/hmba_v05/macaque/hmba_wb_v0.5_macaca_mulatta_nonoverlap.h5ad}"
SPECIES="${SMOKE_SPECIES:-macaca_mulatta}"
N_CELLS="${SMOKE_N_CELLS:-2000}"

if [[ -x "$REPO/.venv/bin/python" ]]; then
    PYTHON="$REPO/.venv/bin/python"
else
    PYTHON=/dfs/ssd/user/kuanpang/ucsc-brain/uce-training-suite/.venv/bin/python
    export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
fi

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} repo=$REPO python=$PYTHON"
echo "git: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"

"$PYTHON" scripts/smoke_mix_v3.py \
    --model "$MODEL" \
    --gene-mapping "$GENE_MAPPING" \
    --h5ad "$H5AD" \
    --species "$SPECIES" \
    --n-cells "$N_CELLS" \
    --label-keys class subclass \
    --num-workers "${SLURM_CPUS_PER_TASK:-6}" \
    --control-chrom-offset 1000 \
    --notebook notebooks/visualize_embeddings_mix_v3.ipynb
