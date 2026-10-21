"""
UCE Model for Hugging Face Transformers.

This module implements the Universal Cell Embedding (UCE) model architecture
following the Hugging Face transformers library conventions.

Two architecture generations share these classes, selected by ``UCEConfig``:

* the original layout (every ``UCEConfig`` default): optional LayerNorm on the
  frozen gene token, Linear + LayerNorm + GELU input projector,
  ``nn.TransformerEncoder``, Linear + LayerNorm output projector on every token,
  CLS row, L2 normalisation; decoder targets share the input projector;
* the v2 layout (e.g. ``KuanP/brain-uce-pilot-mix-v3-d768l14v2``): L2-normalised
  gene token, Linear + GELU + LayerNorm projector, scaled positional encoding,
  the in-repo pre-norm encoder (``UCEEncoder``: RMSNorm, QK-norm, SwiGLU, no
  biases, final norm), an MLP cell head on the CLS row, and a separate
  target-gene layer for the decoder.

The v2 pieces are only built when their config keys ask for them, so an
original-layout checkpoint gets the same modules, state-dict keys and forward
pass as before they existed. The v2 modules need ``torch>=2.4`` (``nn.RMSNorm``).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Tuple, Union
from dataclasses import dataclass

from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .config import UCEConfig


@dataclass
class UCEModelOutput(ModelOutput):
    """
    Output type of UCE model.

    Args:
        cell_embedding (torch.FloatTensor of shape `(batch_size, output_embedding_dim)`):
            Cell-level embeddings aggregated from gene embeddings.
        gene_embeddings (torch.FloatTensor of shape `(batch_size, sequence_length, output_embedding_dim)`):
            Gene-level embeddings from the transformer encoder.
        hidden_states (tuple(torch.FloatTensor), optional):
            Tuple of torch.FloatTensor (one for each layer) of shape
            `(batch_size, sequence_length, d_model)`.
        attentions (tuple(torch.FloatTensor), optional):
            Tuple of torch.FloatTensor (one for each layer) of shape
            `(batch_size, num_heads, sequence_length, sequence_length)`.
    """
    cell_embedding: torch.FloatTensor = None
    gene_embeddings: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class UCEForExpressionPredictionOutput(ModelOutput):
    """
    Output type for UCE expression prediction model.

    Args:
        loss (torch.FloatTensor of shape `(1,)`, optional):
            Binary cross-entropy loss for expression prediction (mean over batch and genes).
        logits (torch.FloatTensor of shape `(batch_size, num_target_genes)`):
            Expression prediction logits.
        cell_embedding (torch.FloatTensor of shape `(batch_size, output_embedding_dim)`):
            Cell-level embeddings.
        gene_embeddings (torch.FloatTensor of shape `(batch_size, sequence_length, output_embedding_dim)`):
            Gene-level embeddings.
        per_sample_losses (torch.FloatTensor of shape `(batch_size,)`, optional):
            Per-sample BCE losses averaged over target genes.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    cell_embedding: torch.FloatTensor = None
    gene_embeddings: torch.FloatTensor = None
    per_sample_losses: Optional[torch.FloatTensor] = None


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding for Transformer.

    ``scale`` multiplies the table at forward time (``UCEConfig.pe_scale``, v2);
    the stored ``pe`` buffer is the unscaled table either way, and 1.0 skips the
    multiplication.
    """

    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.1, scale: float = 1.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = float(scale)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        pe_slice = self.pe[:x.size(1), 0, :].to(x.dtype)
        if self.scale != 1.0:
            pe_slice = pe_slice * self.scale
        x = x + pe_slice
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned Positional Encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)


class EmbeddingAggregator(nn.Module):
    """Embedding Aggregator for converting gene embeddings to cell embeddings."""

    def __init__(self, method: str = 'cls') -> None:
        super().__init__()
        self.method = method
        assert method in ['cls', 'mean'], "Method must be either 'cls' or 'mean'."

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == 'cls':
            return x[:, 0, :]
        elif self.method == 'mean':
            return torch.mean(x, dim=1)


class ProteinEmbeddingLayer(nn.Module):
    """Protein/Gene Embedding Layer: the frozen token table plus its normalisation.

    ``norm_type`` (``UCEConfig.token_norm``): "layernorm" = a LayerNorm over
    ``embedding_dim`` iff ``use_layer_norm`` (the original behaviour); "l2" =
    parameter-free L2 normalisation of each token (v2, public UCE); "none".
    """

    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        padding_idx: int = 0,
        init_path: Optional[str] = None,
        use_layer_norm: bool = True,
        requires_grad: bool = False,
        norm_type: str = "layernorm",
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.norm_type = norm_type

        # Create embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

        # Load pre-trained embeddings if provided
        if init_path:
            self._load_pretrained_embeddings(init_path)

        # Set gradient requirements
        self.embedding.weight.requires_grad = requires_grad

        # Optional layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim) if (use_layer_norm and norm_type == "layernorm") else None

    def _load_pretrained_embeddings(self, init_path: str):
        """Load pre-trained embeddings from file."""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        elif self.norm_type == "l2":
            x = F.normalize(x, dim=-1)
        return x


def _build_input_gene_embedding_projector(
    embedding_dim: int,
    d_model: int,
    projection_type: str = "linear",
    expansion_factor: int = 4,
) -> nn.Sequential:
    """Project frozen protein/gene embeddings into the transformer's input space.

    Modes (mirroring the training code):
      - "linear":         Linear(embedding_dim, d_model) + LayerNorm + GELU. The
                          original layout (input_gene_embedding_projector.0/1/2.*).
      - "linear_gelu_ln": Linear + GELU + LayerNorm, the public-UCE order (v2; also
                          the decoder's separate ``gene_embedding_layer``).
      - "mlp":            embedding_dim -> d_model -> d_model*expansion_factor -> d_model,
                          GELU between the Linears and one terminating LayerNorm.
    """
    if projection_type == "linear":
        return nn.Sequential(
            nn.Linear(embedding_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
    if projection_type == "linear_gelu_ln":
        return nn.Sequential(
            nn.Linear(embedding_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
    if projection_type == "mlp":
        hidden_dim = d_model * expansion_factor
        return nn.Sequential(
            nn.Linear(embedding_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )
    raise ValueError(f"Unknown input_projection_type: {projection_type!r}")


class BinaryExpressionDecoder(nn.Module):
    """Binary Expression Decoder for predicting gene expression.

    ``input_dropout=False`` (``UCEConfig.decoder_input_dropout``, v2) puts an
    nn.Identity where the input Dropout sits, so the Sequential indices (and the
    state-dict keys ``decoder.1.*`` ...) stay those of the original layout.
    """

    def __init__(
        self,
        layer_dimensions: list,
        dropout: float = 0.1,
        input_dropout: bool = True,
    ) -> None:
        super().__init__()

        assert len(layer_dimensions) >= 2, "At least two layer dimensions are required."
        assert layer_dimensions[-1] == 1, "Last dimension must be 1 for binary classification."

        layers = []
        layers.append(nn.Dropout(dropout) if input_dropout else nn.Identity())

        for i in range(len(layer_dimensions) - 1):
            layers.append(nn.Linear(layer_dimensions[i], layer_dimensions[i + 1]))
            if i < len(layer_dimensions) - 2:  # Not the last layer
                layers.append(nn.LayerNorm(layer_dimensions[i + 1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


# ---------------------------------------------------------------------------
# v2 modules. Built only when the config selects them (cell_head_type "mlp",
# encoder_impl "uce"); the original layout never instantiates them.
# ---------------------------------------------------------------------------


def _full_block(in_features: int, out_features: int, p_drop: float) -> nn.Sequential:
    """Public UCE ``full_block``: Linear -> LayerNorm -> GELU -> Dropout."""
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.LayerNorm(out_features),
        nn.GELU(),
        nn.Dropout(p=p_drop),
    )


class CellEmbeddingHead(nn.Module):
    """v2 cell head (``cell_head_type="mlp"``): a ``_full_block`` per hidden width, then a plain Linear.

    Applied to the CLS row only. The head is per-token, so this equals applying it
    to every token and reading row 0. Keys: ``cell_head.net.<i>.{0,1}.*`` for the
    blocks, ``cell_head.net.<len(hidden_dims)>.*`` for the last Linear.
    """

    def __init__(self, in_dim: int, hidden_dims: list, out_dim: int, dropout: float) -> None:
        super().__init__()
        blocks = []
        prev = in_dim
        for h in hidden_dims:
            blocks.append(_full_block(prev, int(h), dropout))
            prev = int(h)
        blocks.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _rms_norm(dim: int, eps: float) -> nn.Module:
    if not hasattr(nn, "RMSNorm"):
        raise ImportError(
            f"This checkpoint uses RMSNorm (v2 architecture), which needs torch>=2.4; installed torch is {torch.__version__}."
        )
    return nn.RMSNorm(dim, eps=eps)


def _make_norm(norm_type: str, dim: int, eps: float, bias: bool) -> nn.Module:
    if norm_type == "rmsnorm":
        return _rms_norm(dim, eps)
    return nn.LayerNorm(dim, eps=eps, bias=bias)


class UCESelfAttention(nn.Module):
    """Multi-head self-attention on ``F.scaled_dot_product_attention``.

    Fused ``qkv`` projection (rows: q, k, v) and ``out_proj``. Optional QK-norm:
    RMSNorm over head_dim on q and k (one learned scale of length head_dim each,
    shared by the heads).
    """

    def __init__(self, d_model: int, nhead: int, bias: bool, qk_norm: bool, attn_dropout: float, eps: float):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.attn_dropout = attn_dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.q_norm = _rms_norm(self.head_dim, eps) if qk_norm else None
        self.k_norm = _rms_norm(self.head_dim, eps) if qk_norm else None

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: True at padding -> SDPA boolean mask: True = attend.
            attn_mask = (~key_padding_mask)[:, None, None, :]
        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        return self.out_proj(o.transpose(1, 2).reshape(B, L, D))


class UCEFeedForward(nn.Module):
    """Feed-forward block: gelu/relu (Linear-act-Dropout-Linear) or SwiGLU (fused gate|up in ``w_in``)."""

    def __init__(self, d_model: int, hidden: int, ffn_type: str, bias: bool, dropout: float):
        super().__init__()
        self.ffn_type = ffn_type
        self.dropout = nn.Dropout(dropout)
        self.w_in = nn.Linear(d_model, 2 * hidden if ffn_type == "swiglu" else hidden, bias=bias)
        self.w_out = nn.Linear(hidden, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.w_in(x)
        if self.ffn_type == "swiglu":
            gate, up = h.chunk(2, dim=-1)
            h = F.silu(gate) * up
        elif self.ffn_type == "gelu":
            h = F.gelu(h)
        else:
            h = F.relu(h)
        return self.w_out(self.dropout(h))


class UCEEncoderLayer(nn.Module):
    """Pre-norm (``norm_first``) or post-norm encoder block with dropout on both residual branches."""

    def __init__(self, config: UCEConfig):
        super().__init__()
        d = config.d_model
        p = config.dropout
        attn_p = config.attn_dropout if config.attn_dropout is not None else p
        eps = config.layer_norm_eps
        self.norm_first = bool(config.norm_first)
        self.norm1 = _make_norm(config.norm_type, d, eps, config.encoder_bias)
        self.norm2 = _make_norm(config.norm_type, d, eps, config.encoder_bias)
        self.self_attn = UCESelfAttention(d, config.nhead, config.encoder_bias, config.qk_norm, attn_p, eps)
        self.ffn = UCEFeedForward(d, config.dim_feedforward, config.ffn_type, config.encoder_bias, p)
        self.dropout1 = nn.Dropout(p)
        self.dropout2 = nn.Dropout(p)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.norm_first:
            x = x + self.dropout1(self.self_attn(self.norm1(x), key_padding_mask))
            x = x + self.dropout2(self.ffn(self.norm2(x)))
        else:
            x = self.norm1(x + self.dropout1(self.self_attn(x, key_padding_mask)))
            x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x


class UCEEncoder(nn.Module):
    """v2 encoder (``encoder_impl="uce"``): a stack of ``UCEEncoderLayer`` plus an optional final norm.

    Same call signature as ``nn.TransformerEncoder`` (``src_key_padding_mask``:
    True at padding). Keys: ``layers.<i>.{norm1,norm2,self_attn.{qkv,out_proj,q_norm,k_norm},ffn.{w_in,w_out}}.*``
    and ``final_norm.*``.
    """

    def __init__(self, config: UCEConfig):
        super().__init__()
        self.layers = nn.ModuleList([UCEEncoderLayer(config) for _ in range(config.num_layers)])
        self.final_norm = (_make_norm(config.norm_type, config.d_model, config.layer_norm_eps, config.encoder_bias)
                           if config.final_norm else None)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x


class UCEPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading
    and loading pretrained models.
    """

    config_class = UCEConfig
    base_model_prefix = "uce"
    supports_gradient_checkpointing = True

    def _load_pretrained_embeddings(self,
                                    embedding_weights: torch.Tensor):
        """Load pre-trained embeddings into the embedding layer."""
        pass


class UCEModel(UCEPreTrainedModel):
    """
    Universal Cell Embedding (UCE) Model.

    This model implements a transformer-based architecture for learning universal
    cell embeddings from single-cell RNA sequencing data.

    The architecture consists of:
    - Protein embedding layer for encoding gene sequences
    - Transformer encoder for learning sequence representations
    - Positional encoding for sequence position information
    - Embedding aggregation for creating cell-level embeddings

    Submodules by layout (see the module docstring):

    ==============================  ===============================  ===================================
    piece                           original (UCEConfig defaults)    v2
    ==============================  ===============================  ===================================
    ``embedding_layer``             token LayerNorm (if enabled)     L2-normalised token, no parameters
    ``input_gene_embedding_proj.``  Linear + LayerNorm + GELU        Linear + GELU + LayerNorm
    ``pos_encoder``                 sinusoidal table                 table x ``pe_scale``
    ``transformer_encoder``         ``nn.TransformerEncoder``        ``UCEEncoder``
    cell head                       ``output_embedding_projector``   ``cell_head`` (CLS row only)
    ==============================  ===============================  ===================================
    """

    def __init__(self, config: UCEConfig):
        super().__init__(config)
        self.config = config

        # Embedding layer
        self.embedding_layer = ProteinEmbeddingLayer(
            embedding_dim=config.embedding_dim,
            vocab_size=config.vocab_size,
            padding_idx=config.padding_idx,
            init_path=config.embedding_init_path,
            use_layer_norm=config.use_embedding_layer_norm,
            requires_grad=config.embedding_requires_grad,
            norm_type=getattr(config, "token_norm", "layernorm"),
        )

        # Project embeddings to transformer dimension. getattr keeps configs
        # written before these keys existed loadable ("linear" is the original).
        self.input_gene_embedding_projector = _build_input_gene_embedding_projector(
            embedding_dim=config.embedding_dim,
            d_model=config.d_model,
            projection_type=getattr(config, "input_projection_type", "linear"),
            expansion_factor=getattr(config, "input_projection_expansion_factor", 4),
        )

        # v2: x sqrt(d_model) on the projected tokens before the positional encoding
        self.token_scale = math.sqrt(config.d_model) if getattr(config, "token_scaling_sqrt_d", False) else 1.0

        # Positional encoding
        if config.positional_encoding_type == "sinusoidal":
            self.pos_encoder = SinusoidalPositionalEncoding(
                d_model=config.d_model,
                max_len=config.max_sequence_length,
                dropout=config.dropout,
                scale=getattr(config, "pe_scale", 1.0),
            )
        elif config.positional_encoding_type == "learned":
            self.pos_encoder = LearnedPositionalEncoding(
                d_model=config.d_model,
                max_len=config.max_sequence_length,
                dropout=config.dropout
            )

        # Transformer encoder
        if getattr(config, "encoder_impl", "torch") == "uce":
            self.transformer_encoder = UCEEncoder(config)
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation=config.activation,
                batch_first=True,
                norm_first=getattr(config, "norm_first", False),
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=config.num_layers
            )

        # Output projection (original) or v2 MLP cell head; exactly one of the two exists.
        self.cell_head = None
        if getattr(config, "cell_head_type", "linear_ln") == "mlp":
            head_dropout = config.cell_head_dropout if config.cell_head_dropout is not None else config.dropout
            self.cell_head = CellEmbeddingHead(
                config.d_model, config.cell_head_dims, config.output_embedding_dim, head_dropout)
        else:
            self.output_embedding_projector = nn.Sequential(
                nn.Linear(config.d_model, config.output_embedding_dim),
                nn.LayerNorm(config.output_embedding_dim),
            )

        # Embedding aggregator
        self.embedding_reduction = EmbeddingAggregator(method=config.embedding_reduction)

        # Initialize weights
        self.post_init()

    def _project_gene_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Project gene embeddings to transformer input space."""
        return self.input_gene_embedding_projector(x)

    def _transformer_forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Perform forward pass through transformer encoder.

        Args:
            x: Input embeddings of shape (batch_size, seq_len, d_model)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            output_hidden_states: Whether to return hidden states
            output_attentions: Whether to return attention weights

        Returns:
            Dictionary containing cell_embedding and gene_embeddings. With the
            original cell head, gene_embeddings are the projected per-token
            outputs (output_embedding_dim); with the v2 ``cell_head`` (applied to
            the CLS row only) they are the encoder's final hidden states (d_model).
        """
        # Apply positional encoding
        x = self.pos_encoder(x)

        # Convert attention mask to padding mask
        # attention_mask: 1 for real tokens, 0 for padding
        # src_key_padding_mask: True for padding, False for real tokens
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)

        # Pass through transformer encoder
        x = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )

        if self.cell_head is not None:
            # v2: reduce first (CLS row), then the per-token MLP head on that row
            cell_embedding = self.cell_head(self.embedding_reduction(x))
        else:
            # Project to output embedding space
            x = self.output_embedding_projector(x)

            # Aggregate to cell-level embedding
            cell_embedding = self.embedding_reduction(x)

        # Normalize cell embedding
        cell_embedding = F.normalize(cell_embedding, dim=-1)

        return {
            "cell_embedding": cell_embedding,
            "gene_embeddings": x
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, UCEModelOutput]:
        """
        Forward pass through the UCE model.

        Args:
            input_ids: Gene sequence token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            output_hidden_states: Whether to return hidden states
            output_attentions: Whether to return attention weights
            return_dict: Whether to return a ModelOutput object

        Returns:
            UCEModelOutput or tuple containing cell embeddings and gene embeddings
        """
        # Ensure input_ids are long integers
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        # Embed input sequences
        input_embeddings = self.embedding_layer(input_ids)

        # Project to transformer input space
        input_embeddings = self._project_gene_embeddings(input_embeddings)
        if self.token_scale != 1.0:
            input_embeddings = input_embeddings * self.token_scale

        # Forward through transformer
        outputs = self._transformer_forward(
            input_embeddings,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )

        if not return_dict:
            return (outputs["cell_embedding"], outputs["gene_embeddings"])

        return UCEModelOutput(
            cell_embedding=outputs["cell_embedding"],
            gene_embeddings=None,
            hidden_states=None,
            attentions=None,
        )

    def extract_cell_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, UCEModelOutput]:
        """
        Extract cell embeddings without masking for downstream tasks.

        Args:
            input_ids: Gene sequence token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            return_dict: Whether to return a ModelOutput object

        Returns:
            If return_dict=True: UCEModelOutput with cell_embedding and gene_embeddings
            If return_dict=False: Tuple of (cell_embedding, gene_embeddings)
        """
        return self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=return_dict,
        )

    def _load_pretrained_embeddings(self,
                                    embedding_weights: torch.Tensor):
        """Load pre-trained embeddings into the embedding layer."""
        self.embedding_layer.embedding.weight.data.copy_(embedding_weights)
        print("Loaded pretrained embeddings into UCE model.")

class UCEForExpressionPrediction(UCEPreTrainedModel):
    """
    UCE Model with expression prediction head.

    This model extends UCE with a decoder for binary gene expression prediction.
    Target genes are embedded with the encoder's token path (``uce.embedding_layer``
    + ``uce.input_gene_embedding_projector``), or, with ``separate_gene_layer``
    (v2), with ``uce.embedding_layer`` + this model's own ``gene_embedding_layer``
    (Linear + GELU + LayerNorm). The decoder only affects the loss, never the
    cell embedding.
    """

    def __init__(self, config: UCEConfig):
        super().__init__(config)
        self.config = config

        # Base UCE model
        self.uce = UCEModel(config)

        # v2: separate target-gene layer used only by the decoder
        self.gene_embedding_layer = None
        if getattr(config, "separate_gene_layer", False):
            self.gene_embedding_layer = _build_input_gene_embedding_projector(
                embedding_dim=config.embedding_dim, d_model=config.d_model,
                projection_type="linear_gelu_ln")

        # Expression decoder
        self.expression_decoder = BinaryExpressionDecoder(
            layer_dimensions=config.decoder_layer_dims,
            dropout=config.decoder_dropout,
            input_dropout=getattr(config, "decoder_input_dropout", True),
        )

        # Initialize weights
        self.post_init()

    def _decoder_forward(
        self,
        cell_embedding: torch.Tensor,
        target_gene_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict expression levels using cell and gene embeddings.

        Args:
            cell_embedding: Cell embedding of shape (batch_size, output_embedding_dim)
            target_gene_embeddings: Target gene embeddings of shape (batch_size, num_genes, output_embedding_dim)

        Returns:
            Expression logits of shape (batch_size, num_genes)
        """
        # Expand cell embedding to match gene dimensions
        cell_embedding_expanded = cell_embedding.unsqueeze(1).repeat(1, target_gene_embeddings.shape[1], 1)

        # Concatenate cell and gene embeddings
        combined = torch.cat([cell_embedding_expanded, target_gene_embeddings], dim=-1)

        # Predict expression
        logits = self.expression_decoder(combined).squeeze(-1)

        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_gene_ids: Optional[torch.Tensor] = None,
        target_expression: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        **kwargs
    ) -> Union[Tuple, UCEForExpressionPredictionOutput]:
        """
        Forward pass with expression prediction.

        Args:
            input_ids: Gene sequence token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            target_gene_ids: Target gene IDs for prediction of shape (batch_size, num_target_genes)
            target_expression: Ground truth expression labels of shape (batch_size, num_target_genes)
            output_hidden_states: Whether to return hidden states
            output_attentions: Whether to return attention weights
            return_dict: Whether to return a ModelOutput object

        Returns:
            UCEForExpressionPredictionOutput or tuple
        """
        # Get cell embeddings from UCE model
        outputs = self.uce(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True
        )

        cell_embedding = outputs.cell_embedding
        gene_embeddings = outputs.gene_embeddings

        loss = None
        logits = None
        per_sample_losses = None

        if target_gene_ids is not None:
            # Ensure target_gene_ids are long integers
            if target_gene_ids.dtype != torch.long:
                target_gene_ids = target_gene_ids.long()

            # Embed target genes
            target_gene_embeddings = self.uce.embedding_layer(target_gene_ids)
            if self.gene_embedding_layer is not None:
                target_gene_embeddings = self.gene_embedding_layer(target_gene_embeddings)
            else:
                target_gene_embeddings = self.uce._project_gene_embeddings(target_gene_embeddings)

            # Predict expression
            logits = self._decoder_forward(cell_embedding, target_gene_embeddings)

            # Compute loss if labels provided
            if target_expression is not None:
                # Compute mean loss for training (reduction='mean')
                loss = F.binary_cross_entropy_with_logits(logits, target_expression)

                # Compute per-sample losses for evaluation/monitoring
                # Shape: [batch_size, num_target_genes] -> [batch_size]
                per_gene_losses = F.binary_cross_entropy_with_logits(
                    logits, target_expression, reduction='none'
                )
                per_sample_losses = per_gene_losses.mean(dim=1)

        if not return_dict:
            output = (logits, cell_embedding, gene_embeddings)
            return ((loss,) + output) if loss is not None else output

        return UCEForExpressionPredictionOutput(
            loss=loss,
            logits=logits,
            per_sample_losses=per_sample_losses,
        )

    def extract_cell_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, UCEModelOutput]:
        """
        Extract cell embeddings without masking for downstream tasks.

        Args:
            input_ids: Gene sequence token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            return_dict: Whether to return a ModelOutput object

        Returns:
            If return_dict=True: UCEModelOutput with cell_embedding and gene_embeddings
            If return_dict=False: Tuple of (cell_embedding, gene_embeddings)
        """
        return self.uce.extract_cell_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )

    def _load_pretrained_embeddings(self,
                                    embedding_weights: torch.Tensor):
        """Load pre-trained embeddings into the embedding layer."""
        self.uce.embedding_layer.embedding.weight.data.copy_(embedding_weights)
        print("Loaded pretrained embeddings into UCE model.")
