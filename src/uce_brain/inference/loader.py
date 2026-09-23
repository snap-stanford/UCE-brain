"""
Checkpoint loading for UCE-brain inference.

Three things have to agree before an embedding means anything: the weights,
the gene vocabulary the model was trained with, and the cell-sentence
parameters (``pad_length``, the chromosome-token layout, ...). The last two are
vocabulary-specific. A legacy model (145k/178k-token vocab, chrom_token_offset
1000) and a v2026-09 model (266,403 tokens, chrom_token_offset 237411) share
the architecture but not the tokenisation, and a wrong offset does not raise:
it silently yields well-formed, meaningless embeddings. Nothing here therefore
carries a hard-coded vocabulary default. The parameters are read from the
checkpoint (``config.json``, or the training run's ``config.yaml`` shipped next
to the weights) or passed explicitly by the caller.
"""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import yaml  # installed by transformers / huggingface_hub

from ..model import UCEConfig, UCEForExpressionPrediction

log = logging.getLogger(__name__)


# Cell-sentence parameters that must be identical between training and
# inference. Every one of them changes the token sequence, hence the embedding.
CELL_SENTENCE_KEYS = (
    "pad_length",
    "positive_sample_num",
    "negative_sample_num",
    "mask_prop",
    "sample_size",
    "cls_token_idx",
    "chrom_token_offset",
    "chrom_token_right_idx",
    "pad_token_idx",
)

# File names of the gene-mapping JSON inside a checkpoint / Hub repo, in the
# order they are looked for (v2026-09 layout first, then the legacy Hub layout).
GENE_MAPPING_PATTERNS = (
    "vocab/all_species_gene_dict*.json",
    "all_species_gene_dict*.json",
    "gene_mapping.json",
)


@dataclass
class CellSentenceParams:
    """Tokenisation parameters of a checkpoint (see :func:`load_cell_sentence_params`).

    ``source`` records where the values came from (a config path or
    ``"overrides"``); it is not a tokenisation parameter.
    """

    pad_length: int
    positive_sample_num: int
    negative_sample_num: int
    mask_prop: float
    sample_size: int
    cls_token_idx: int
    chrom_token_offset: int
    chrom_token_right_idx: int
    pad_token_idx: int
    source: str = "overrides"

    def as_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for :class:`~uce_brain.data.H5ADDataset`."""
        kwargs = asdict(self)
        kwargs.pop("source")
        return kwargs


def resolve_checkpoint(model: Union[str, Path], **download_kwargs) -> Path:
    """Return a local directory for ``model``.

    A path to an existing directory is returned as is. Anything else is
    treated as a Hugging Face Hub repo id (e.g. ``KuanP/brain-uce-pilot-mix-v3``)
    and fetched with ``huggingface_hub.snapshot_download`` (cached under
    ``HF_HOME``; ``download_kwargs`` such as ``revision`` or ``token`` are
    passed through).
    """
    path = Path(model)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    log.info(f"Fetching {model} from the Hugging Face Hub")
    return Path(snapshot_download(repo_id=str(model), **download_kwargs))


def find_gene_mapping(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
    """Locate the gene-mapping JSON shipped with a checkpoint, if any."""
    checkpoint_dir = Path(checkpoint_dir)
    for pattern in GENE_MAPPING_PATTERNS:
        matches = sorted(checkpoint_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _candidate_config_paths(checkpoint_dir: Path) -> List[Path]:
    """Where a training run's ``config.yaml`` can sit relative to the weights.

    Covers a Hub repo / staging directory (``config.yaml`` next to
    ``model.safetensors``, plus the ``training/`` copy), a run directory
    (``<run>/<timestamp>/config.yaml``) and a ``checkpoint-<step>`` below it.
    """
    return [
        checkpoint_dir / "config.yaml",
        checkpoint_dir / "training" / "config.yaml",
        checkpoint_dir / ".hydra" / "config.yaml",
        checkpoint_dir.parent / "config.yaml",
        checkpoint_dir.parent / ".hydra" / "config.yaml",
    ]


def find_run_config(checkpoint_dir: Union[str, Path]) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    """Load the training run's ``config.yaml`` next to a checkpoint.

    Returns ``(config_dict, path)``, or ``(None, None)`` when no parsable
    config is found.
    """
    for cfg_path in _candidate_config_paths(Path(checkpoint_dir)):
        try:
            if not cfg_path.is_file():
                continue
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
        except OSError:
            continue
        except Exception as e:  # malformed yaml: keep probing
            log.warning(f"Failed to parse {cfg_path}: {e}")
            continue
        if isinstance(cfg, dict):
            log.info(f"Found run config at {cfg_path}")
            return cfg, cfg_path
    return None, None


def load_cell_sentence_params(
    checkpoint_dir: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
) -> CellSentenceParams:
    """Recover the cell-sentence parameters a checkpoint was trained with.

    Resolution order per key: ``overrides`` > the checkpoint's ``config.json``
    (any extra key stored in the ``UCEConfig``) > ``dataset.*`` of the training
    run's ``config.yaml`` found by :func:`find_run_config`. A key available from
    none of them raises, because the library defaults belong to the legacy
    vocabulary and would be silently wrong for any other model.

    For the legacy Hub models, which ship neither file, pass the values
    explicitly, e.g. ``overrides=dict(pad_length=2048, sample_size=1024,
    positive_sample_num=100, negative_sample_num=100, mask_prop=0.0,
    cls_token_idx=1, pad_token_idx=0, chrom_token_right_idx=2000,
    chrom_token_offset=1000)`` (the :class:`~uce_brain.data.H5ADDataset`
    defaults used by the legacy notebooks).
    """
    checkpoint_dir = Path(checkpoint_dir)
    overrides = dict(overrides or {})

    from_json: Dict[str, Any] = {}
    if (checkpoint_dir / "config.json").is_file():
        config = UCEConfig.from_pretrained(str(checkpoint_dir))
        from_json = {k: getattr(config, k) for k in CELL_SENTENCE_KEYS if getattr(config, k, None) is not None}

    run_cfg, cfg_path = find_run_config(checkpoint_dir)
    from_yaml = (run_cfg or {}).get("dataset") or {}

    resolved: Dict[str, Any] = {}
    sources: List[str] = []
    missing: List[str] = []
    for key in CELL_SENTENCE_KEYS:
        if overrides.get(key) is not None:
            resolved[key] = overrides[key]
            sources.append("overrides")
        elif from_json.get(key) is not None:
            resolved[key] = from_json[key]
            sources.append("config.json")
        elif from_yaml.get(key) is not None:
            resolved[key] = from_yaml[key]
            sources.append(str(cfg_path))
        else:
            missing.append(key)

    if missing:
        searched = "\n  ".join(str(p) for p in _candidate_config_paths(checkpoint_dir))
        raise ValueError(
            f"Could not resolve the cell-sentence parameters {missing} for {checkpoint_dir}.\n"
            f"config.yaml found: {cfg_path if cfg_path else 'none'} (searched:\n  {searched})\n"
            "Inference must tokenise exactly as training did; a mismatch is silent and "
            "produces plausible-looking but meaningless embeddings. Ship the run's "
            "config.yaml next to the weights, store the keys in config.json, or pass "
            "them via `overrides`."
        )

    params = CellSentenceParams(
        pad_length=int(resolved["pad_length"]),
        positive_sample_num=int(resolved["positive_sample_num"]),
        negative_sample_num=int(resolved["negative_sample_num"]),
        mask_prop=float(resolved["mask_prop"]),
        sample_size=int(resolved["sample_size"]),
        cls_token_idx=int(resolved["cls_token_idx"]),
        chrom_token_offset=int(resolved["chrom_token_offset"]),
        chrom_token_right_idx=int(resolved["chrom_token_right_idx"]),
        pad_token_idx=int(resolved["pad_token_idx"]),
        source=", ".join(sorted(set(sources))),
    )
    log.info(
        f"Cell-sentence params from {params.source}: pad_length={params.pad_length}, "
        f"sample_size={params.sample_size}, chrom_token_offset={params.chrom_token_offset}, "
        f"chrom_token_right_idx={params.chrom_token_right_idx}, cls_token_idx={params.cls_token_idx}, "
        f"pad_token_idx={params.pad_token_idx}, mask_prop={params.mask_prop}"
    )
    return params


def _reload_from_safetensors(model: torch.nn.Module, path: Path) -> Tuple[int, int]:
    """Overwrite every model tensor that differs from ``model.safetensors``.

    ``from_pretrained`` re-initialises submodules after loading on some
    transformers versions (the guard against that has changed across 4.x/5.x),
    and the failure is silent: a model whose transformer weights were reset
    still produces finite, unit-norm embeddings. Streaming the file tensor by
    tensor keeps peak memory at one tensor (the frozen gene table) and yields a
    diagnostic: the number of tensors that had to be restored.

    Returns ``(n_checked, n_restored)``.
    """
    from safetensors import safe_open

    state = model.state_dict()
    n_checked = n_restored = 0
    unexpected: List[str] = []
    seen = set()
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            if key not in state:
                unexpected.append(key)
                continue
            seen.add(key)
            target = state[key]
            tensor = f.get_tensor(key).to(device=target.device, dtype=target.dtype)
            if tensor.shape != target.shape:
                raise ValueError(f"{path}: {key} has shape {tuple(tensor.shape)}, model expects {tuple(target.shape)}")
            n_checked += 1
            if not torch.equal(target, tensor):
                with torch.no_grad():
                    target.copy_(tensor)
                n_restored += 1
    missing = [k for k in state if k not in seen]
    if missing:
        log.warning(f"{len(missing)} model tensors are not in {path.name}: {missing[:5]}")
    if unexpected:
        log.warning(f"{len(unexpected)} tensors in {path.name} have no counterpart in the model: {unexpected[:5]}")
    return n_checked, n_restored


def load_model(
    checkpoint_dir: Union[str, Path],
    device: Union[str, torch.device] = "cuda",
    dtype: Optional[Union[str, torch.dtype]] = None,
) -> UCEForExpressionPrediction:
    """Load a :class:`UCEForExpressionPrediction` from a local checkpoint directory.

    The directory holds ``config.json`` and ``model.safetensors`` (the layout
    written by ``save_pretrained`` / the HF Trainer and used by the Hub repos;
    use :func:`resolve_checkpoint` first for a repo id). The frozen gene-token
    table is part of ``model.safetensors``, so nothing else is needed.

    Args:
        checkpoint_dir: Directory with ``config.json`` + ``model.safetensors``.
        device: Where to put the model.
        dtype: Optional parameter dtype (``torch.bfloat16`` or ``"bfloat16"``).
            Leave ``None`` to keep the stored fp32 weights and rely on autocast
            at embedding time, as training did.

    Returns:
        The model on ``device`` in ``eval()`` mode.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not (checkpoint_dir / "config.json").is_file():
        raise FileNotFoundError(f"No config.json in {checkpoint_dir}")

    config = UCEConfig.from_pretrained(str(checkpoint_dir))
    log.info(
        f"Loading {checkpoint_dir}: vocab_size={config.vocab_size}, d_model={config.d_model}, "
        f"layers={config.num_layers}, nhead={config.nhead}, output_embedding_dim={config.output_embedding_dim}"
    )
    model = UCEForExpressionPrediction.from_pretrained(str(checkpoint_dir))

    weights = checkpoint_dir / "model.safetensors"
    if weights.is_file():
        n_checked, n_restored = _reload_from_safetensors(model, weights)
        log.info(f"Verified {n_checked} tensors against {weights.name}; restored {n_restored}")
    else:
        log.warning(
            f"No single model.safetensors in {checkpoint_dir}; relying on from_pretrained alone. "
            "If the per-cell loss sits at ln 2 = 0.693, the trained weights did not survive loading."
        )

    if dtype is not None:
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        model = model.to(dtype)
    model = model.to(device).eval()
    log.info(f"Model ready on {device} ({sum(p.numel() for p in model.parameters()):,} parameters)")
    return model
