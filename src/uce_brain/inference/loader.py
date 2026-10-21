"""
Checkpoint loading for UCE-brain inference.

Three things have to agree before an embedding means anything: the weights,
the gene vocabulary the model was trained with, and the cell-sentence
parameters (``pad_length``, the chromosome-token layout, ...). The last two are
vocabulary-specific. A legacy model (145k/178k-token vocab, chrom_token_offset
1000) and a v2026-09 model (266,403 tokens, chrom_token_offset 237411) share
the architecture but not the tokenisation, and a wrong offset does not raise:
it silently yields well-formed but different embeddings. Nothing here therefore
carries a hard-coded vocabulary default. The parameters are read from the
checkpoint (``config.json``, or the training run's ``config.yaml`` shipped next
to the weights) or passed explicitly by the caller.

The weights have to match the code, too. :func:`load_model` is strict: a
``config.json`` key this version of ``UCEConfig`` does not define, a tensor in
``model.safetensors`` the model has no slot for, or a model tensor the file
does not provide raises (training-only auxiliary heads excepted), because a
partly loaded or wrongly assembled model still emits finite, unit-norm
embeddings. The usual cause is a checkpoint written by newer training code
(a newer model architecture) than this package implements.
"""

import inspect
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import yaml  # installed by transformers / huggingface_hub
from transformers import PretrainedConfig

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

# Training-only auxiliary heads that a checkpoint may carry next to the model
# (the Matryoshka runs' per-prefix decoders and their config keys). The
# embedding path never uses them; they are skipped instead of rejected.
AUX_TRAINING_HEAD_PREFIXES = ("matryoshka_decoders.",)
AUX_TRAINING_CONFIG_PREFIXES = ("matryoshka_",)

# config.json keys without architecture meaning that may appear in addition to
# the UCEConfig arguments and PretrainedConfig's own attributes.
_BOOKKEEPING_CONFIG_KEYS = frozenset({
    "architectures", "model_type", "transformers_version", "dtype", "torch_dtype",
    "use_cache", "_name_or_path", "_commit_hash", "auto_map",
})

# Generic PretrainedConfig attributes that a ``save_pretrained`` of any supported
# transformers version can write into config.json: the union of
# ``PretrainedConfig().to_dict()`` under transformers 4.57.6 (56 keys, including
# the generation / tokenizer defaults that 5.x dropped) and 5.15.0 (13 keys, a
# subset). Spelled out so the strict check accepts the same config.json under
# every transformers version; the installed version's own keys are added at run
# time on top (see unknown_config_keys).
_PRETRAINED_CONFIG_KEYS = frozenset({
    "_name_or_path", "add_cross_attention", "architectures", "bad_words_ids", "begin_suppress_tokens",
    "bos_token_id", "chunk_size_feed_forward", "cross_attention_hidden_size", "decoder_start_token_id",
    "diversity_penalty", "do_sample", "dtype", "early_stopping", "encoder_no_repeat_ngram_size",
    "eos_token_id", "exponential_decay_length_penalty", "finetuning_task", "forced_bos_token_id",
    "forced_eos_token_id", "id2label", "is_decoder", "is_encoder_decoder", "label2id", "length_penalty",
    "max_length", "min_length", "model_type", "no_repeat_ngram_size", "num_beam_groups", "num_beams",
    "num_return_sequences", "output_attentions", "output_hidden_states", "output_scores", "pad_token_id",
    "prefix", "problem_type", "pruned_heads", "remove_invalid_values", "repetition_penalty", "return_dict",
    "return_dict_in_generate", "sep_token_id", "suppress_tokens", "task_specific_params", "temperature",
    "tf_legacy_loss", "tie_encoder_decoder", "tie_word_embeddings", "tokenizer_class", "top_k", "top_p",
    "torchscript", "transformers_version", "typical_p", "use_bfloat16",
})


class UnsupportedCheckpointError(ValueError, AssertionError):
    """A checkpoint's ``config.json`` describes a model this uce_brain version cannot build.

    Raised by :func:`load_model` and :func:`load_cell_sentence_params` for
    ``config.json`` keys this version does not define and for option values
    ``UCEConfig`` rejects. It subclasses ``ValueError`` and also
    ``AssertionError``, which is what ``UCEConfig`` validation raised before this
    check existed, so existing ``except AssertionError`` handlers still catch it.
    """

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
        config = _load_config(checkpoint_dir)
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
            "produces plausible-looking but different embeddings. Ship the run's "
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


def unknown_config_keys(checkpoint_dir: Union[str, Path]) -> List[str]:
    """``config.json`` keys that this version of :class:`~uce_brain.model.UCEConfig` does not define.

    Known keys are the ``UCEConfig`` arguments, the generic ``PretrainedConfig``
    attributes (a fixed list covering transformers 4.57 and 5.x, plus whatever
    the installed version defines), the cell-sentence keys
    (:data:`CELL_SENTENCE_KEYS`) and a few bookkeeping entries; keys of
    training-only auxiliary heads (:data:`AUX_TRAINING_CONFIG_PREFIXES`) are
    ignored. Anything else is most likely an architecture option of newer
    training code, which this version would silently drop (``PretrainedConfig``
    keeps unknown keys as inert attributes).
    """
    with open(Path(checkpoint_dir) / "config.json") as f:
        raw = json.load(f)
    known = set(inspect.signature(UCEConfig.__init__).parameters) - {"self", "kwargs"}
    known |= _PRETRAINED_CONFIG_KEYS | set(PretrainedConfig().to_dict())
    known |= set(CELL_SENTENCE_KEYS) | _BOOKKEEPING_CONFIG_KEYS
    return sorted(k for k in raw if k not in known and not k.startswith(AUX_TRAINING_CONFIG_PREFIXES))


def _load_config(checkpoint_dir: Path, unknown_keys: Optional[List[str]] = None) -> UCEConfig:
    """``UCEConfig.from_pretrained`` with validation failures reported as :class:`UnsupportedCheckpointError`."""
    try:
        return UCEConfig.from_pretrained(str(checkpoint_dir))
    except json.JSONDecodeError:
        raise
    except (AssertionError, ValueError) as e:
        raise UnsupportedCheckpointError(
            f"{checkpoint_dir / 'config.json'} is not a configuration this uce_brain version supports ({e}). "
            + _newer_architecture_hint(unknown_keys or [])
        ) from e


def _weight_files(checkpoint_dir: Path) -> List[Path]:
    """The safetensors weight files of a checkpoint: ``model.safetensors``, or the shards of its index."""
    single = checkpoint_dir / "model.safetensors"
    if single.is_file():
        return [single]
    index = checkpoint_dir / "model.safetensors.index.json"
    if index.is_file():
        with open(index) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        return [checkpoint_dir / s for s in shards]
    return []


def _newer_architecture_hint(unknown_keys: List[str]) -> str:
    keys = f" config.json keys this version does not know: {unknown_keys}." if unknown_keys else ""
    return (
        f"The checkpoint was most likely written for a newer model architecture than this uce_brain "
        f"version implements.{keys} Update UCE-brain (git pull / reinstall) to a version that supports it; "
        "do not load it with strict=False, which would embed with a partly random or mis-assembled model."
    )


def _reload_from_safetensors(model: torch.nn.Module, paths: List[Path]) -> Tuple[int, int, List[str], List[str]]:
    """Overwrite every model tensor that differs from the safetensors weights.

    ``from_pretrained`` re-initialises submodules after loading on some
    transformers versions (the guard against that has changed across 4.x/5.x),
    and the failure is silent: a model whose transformer weights were reset
    still produces finite, unit-norm embeddings. Streaming the file(s) tensor by
    tensor keeps peak memory at one tensor (the frozen gene table) and yields a
    diagnostic: the number of tensors that had to be restored. ``paths`` is
    ``[model.safetensors]`` or the shards of ``model.safetensors.index.json``.

    Tensors under :data:`AUX_TRAINING_HEAD_PREFIXES` are skipped. A shape
    mismatch raises here; the caller decides what to do with the rest.

    Returns ``(n_checked, n_restored, missing, unexpected)``: ``missing`` are
    model tensors absent from the file(s), ``unexpected`` are file tensors
    (other than auxiliary heads) the model has no slot for.
    """
    from safetensors import safe_open

    state = model.state_dict()
    n_checked = n_restored = 0
    unexpected: List[str] = []
    aux: List[str] = []
    seen = set()
    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key not in state:
                    (aux if key.startswith(AUX_TRAINING_HEAD_PREFIXES) else unexpected).append(key)
                    continue
                seen.add(key)
                target = state[key]
                tensor = f.get_tensor(key).to(device=target.device, dtype=target.dtype)
                if tensor.shape != target.shape:
                    raise ValueError(
                        f"{path}: {key} has shape {tuple(tensor.shape)}, model expects {tuple(target.shape)}. "
                        "config.json and model.safetensors disagree (or the config describes a newer architecture)."
                    )
                n_checked += 1
                if not torch.equal(target, tensor):
                    with torch.no_grad():
                        target.copy_(tensor)
                    n_restored += 1
    missing = [k for k in state if k not in seen]
    if aux:
        log.info(f"Ignoring {len(aux)} auxiliary training-head tensors ({aux[0]} ...)")
    return n_checked, n_restored, missing, unexpected


def load_model(
    checkpoint_dir: Union[str, Path],
    device: Union[str, torch.device] = "cuda",
    dtype: Optional[Union[str, torch.dtype]] = None,
    strict: bool = True,
) -> UCEForExpressionPrediction:
    """Load a :class:`UCEForExpressionPrediction` from a local checkpoint directory.

    The directory holds ``config.json`` and ``model.safetensors`` (the layout
    written by ``save_pretrained`` / the HF Trainer and used by the Hub repos;
    use :func:`resolve_checkpoint` first for a repo id; a sharded
    ``model.safetensors.index.json`` works too). The frozen gene-token table is
    part of ``model.safetensors``, so nothing else is needed. Both architecture
    generations load here: the original layout and the v2 layout
    (``KuanP/brain-uce-pilot-mix-v3-d768l14v2``); ``config.json`` selects it.

    Args:
        checkpoint_dir: Directory with ``config.json`` + ``model.safetensors``.
        device: Where to put the model.
        dtype: Optional parameter dtype (``torch.bfloat16`` or ``"bfloat16"``).
            Leave ``None`` to keep the stored fp32 weights and rely on autocast
            at embedding time, as training did.
        strict: Raise (default) when ``config.json`` has keys this version does
            not define (:class:`UnsupportedCheckpointError`), when the
            safetensors weights and the model built from ``config.json``
            disagree on the tensor set (``RuntimeError``; training-only
            auxiliary heads excepted), or when there are no safetensors weights
            to check against (``FileNotFoundError``). Each one means the code
            cannot confirm that it reproduces the trained network, typically
            because the checkpoint comes from a newer architecture.
            ``strict=False`` only logs warnings; use it for debugging, not for
            embeddings.

    Returns:
        The model on ``device`` in ``eval()`` mode. The gene-token table's
        ``requires_grad`` follows ``config.embedding_requires_grad`` (False for
        every released checkpoint: the table is frozen).
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not (checkpoint_dir / "config.json").is_file():
        raise FileNotFoundError(f"No config.json in {checkpoint_dir}")

    unknown = unknown_config_keys(checkpoint_dir)
    if unknown:
        msg = f"{checkpoint_dir / 'config.json'} has keys this uce_brain version does not define: {unknown}. "
        if strict:
            raise UnsupportedCheckpointError(msg + _newer_architecture_hint([]))
        log.warning(msg + "They are ignored (strict=False).")

    config = _load_config(checkpoint_dir, unknown)
    log.info(
        f"Loading {checkpoint_dir}: vocab_size={config.vocab_size}, d_model={config.d_model}, "
        f"layers={config.num_layers}, nhead={config.nhead}, output_embedding_dim={config.output_embedding_dim}"
        + (f", encoder_impl={config.encoder_impl}, cell_head_type={config.cell_head_type}"
           if config.encoder_impl != "torch" or config.cell_head_type != "linear_ln" else "")
    )
    model = UCEForExpressionPrediction.from_pretrained(str(checkpoint_dir))

    weights = _weight_files(checkpoint_dir)
    if weights:
        source = weights[0].name if len(weights) == 1 else f"{len(weights)} safetensors shards"
        n_checked, n_restored, missing, unexpected = _reload_from_safetensors(model, weights)
        problems = []
        if unexpected:
            problems.append(f"{len(unexpected)} tensor(s) in {source} have no counterpart in the model "
                            f"built from config.json (first: {unexpected[:5]})")
        if missing:
            problems.append(f"{len(missing)} model tensor(s) are not in {source} and would keep their "
                            f"random initialisation (first: {missing[:5]})")
        if problems:
            msg = f"{checkpoint_dir}: " + "; ".join(problems) + ". "
            if strict:
                raise RuntimeError(msg + _newer_architecture_hint(unknown))
            log.warning(msg + "Continuing because strict=False.")
        log.info(f"Verified {n_checked} tensors against {source}; restored {n_restored}")
    else:
        msg = (f"No model.safetensors (or model.safetensors.index.json) in {checkpoint_dir}, so the loaded "
               "tensors cannot be checked against the stored weights. ")
        if strict:
            raise FileNotFoundError(msg + "Convert the weights to safetensors, or pass strict=False to rely on "
                                    "from_pretrained alone.")
        log.warning(msg + "Relying on from_pretrained alone (strict=False). If the per-cell loss sits at "
                    "ln 2 = 0.693, the trained weights did not survive loading.")

    # from_pretrained does not keep the constructor's requires_grad on the frozen gene-token table.
    model.uce.embedding_layer.embedding.weight.requires_grad_(bool(config.embedding_requires_grad))

    if dtype is not None:
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        model = model.to(dtype)
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model ready on {device} ({n_params:,} parameters, {n_trainable:,} trainable)")
    return model
