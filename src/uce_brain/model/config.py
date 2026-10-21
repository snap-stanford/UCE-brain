"""
UCE Model Configuration for Hugging Face Transformers.

This module defines the configuration class for the Universal Cell Embedding (UCE) model,
following the Hugging Face transformers library conventions.
"""

from transformers import PretrainedConfig
from typing import Optional


class UCEConfig(PretrainedConfig):
    """
    Configuration class for UCE (Universal Cell Embedding) model.

    This class stores the configuration of a UCE model, including all parameters
    needed to instantiate the model architecture. It is used to instantiate a UCE
    model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from PretrainedConfig and can be used to control
    the model outputs. Read the documentation from PretrainedConfig for more information.

    Args:
        vocab_size (int, optional): Size of the vocabulary (number of genes/proteins). Defaults to 60000.
        embedding_dim (int, optional): Dimension of protein/gene embeddings. Defaults to 512.
        d_model (int, optional): Dimension of the transformer model. Defaults to 1024.
        nhead (int, optional): Number of attention heads in transformer. Defaults to 16.
        num_layers (int, optional): Number of transformer encoder layers. Defaults to 4.
        dim_feedforward (int, optional): Dimension of feedforward network in transformer. Defaults to 4096.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        activation (str, optional): Activation function for transformer. Defaults to "gelu".
        expansion_factor (int, optional): Expansion factor for transformer FFN. Defaults to 4.
        max_sequence_length (int, optional): Maximum sequence length for positional encoding. Defaults to 10000.
        padding_idx (int, optional): Index used for padding in embeddings. Defaults to 0.
        embedding_init_path (str, optional): Path to pre-trained embedding weights. Defaults to None.
        embedding_requires_grad (bool, optional): Whether embeddings are trainable. Defaults to True.
        use_embedding_layer_norm (bool, optional): Whether to use layer norm on embeddings. Defaults to True.
        embedding_reduction (str, optional): Method for aggregating embeddings ('cls' or 'mean'). Defaults to "cls".
        output_embedding_dim (int, optional): Dimension of output embeddings. Defaults to 1280.
        decoder_layer_dims (list, optional): Layer dimensions for expression decoder. Defaults to [2560, 1280, 1].
        decoder_expansion_factors (list, optional): Expansion factors for decoder MLPs. Defaults to [2, 2].
        decoder_dropout (float, optional): Dropout probability for decoder. Defaults to 0.1.
        positional_encoding_type (str, optional): Type of positional encoding ('sinusoidal' or 'learned'). Defaults to "sinusoidal".
        norm_first (bool, optional): Pre-norm (True) or post-norm (False) transformer layers. Defaults to False.
        input_projection_type (str, optional): How frozen gene embeddings are projected to d_model:
            'linear' (Linear + LayerNorm + GELU, the original layout), 'linear_gelu_ln'
            (Linear + GELU + LayerNorm, the public-UCE order; v2) or 'mlp'. Defaults to "linear".
        input_projection_expansion_factor (int, optional): Hidden-width multiplier of the 'mlp' projector. Defaults to 4.

    v2 architecture options (``KuanP/brain-uce-pilot-mix-v3-d768l14v2`` and later). Every
    default reproduces the original architecture: the same module tree, state-dict keys and
    forward pass, so configs written before these keys existed load unchanged.

        token_norm (str, optional): Normalisation of the frozen gene token before the input
            projector (and before the decoder's target-gene layer). 'layernorm' = the original
            behaviour (a LayerNorm over ``embedding_dim`` iff ``use_embedding_layer_norm``),
            'l2' = parameter-free L2 normalisation (public UCE), 'none' = raw token.
            Defaults to "layernorm".
        token_scaling_sqrt_d (bool, optional): Multiply the projected tokens by sqrt(d_model)
            before the positional encoding. Defaults to False.
        pe_scale (float, optional): Multiplier of the sinusoidal positional-encoding table
            (applied at forward time; the stored ``pe`` buffer is unchanged). Defaults to 1.0.
        separate_gene_layer (bool, optional): Embed the decoder's target genes with their own
            Linear + GELU + LayerNorm (``gene_embedding_layer``) instead of the encoder's input
            projector. Affects the expression-prediction loss only, not cell embeddings.
            Defaults to False.
        cell_head_type (str, optional): 'linear_ln' = Linear(d_model, output_embedding_dim) +
            LayerNorm on every token, CLS taken after (``output_embedding_projector``, the
            original); 'mlp' = CLS taken first, then ``cell_head``: one Linear + LayerNorm +
            GELU + Dropout block per width in ``cell_head_dims``, then a plain
            Linear(last width, output_embedding_dim). Defaults to "linear_ln".
        cell_head_dims (list, optional): Hidden widths of the 'mlp' cell head. Defaults to None.
        cell_head_dropout (float, optional): Dropout of the 'mlp' cell head (None = ``dropout``).
        decoder_input_dropout (bool, optional): Dropout on the decoder input (False puts an
            nn.Identity in its place, keeping the decoder's Sequential indices). Defaults to True.
        encoder_impl (str, optional): 'torch' = nn.TransformerEncoder (the original) or 'uce' =
            the in-repo encoder (``UCEEncoder``), required by the options below. Defaults to "torch".
        final_norm (bool, optional): Norm after the last encoder layer ('uce' only). Defaults to False.
        norm_type (str, optional): 'layernorm' or 'rmsnorm' in the 'uce' encoder. Defaults to "layernorm".
        qk_norm (bool, optional): RMSNorm over head_dim on queries and keys ('uce' only). Defaults to False.
        ffn_type (str, optional): 'gelu', 'relu' or 'swiglu' feed-forward ('uce' only; the torch
            encoder uses ``activation``). Defaults to "gelu".
        encoder_bias (bool, optional): Biases in the 'uce' encoder's Linear and LayerNorm layers.
            Defaults to True.
        attn_dropout (float, optional): Attention-probability dropout of the 'uce' encoder
            (None = ``dropout``).
        scaled_residual_init (bool, optional): Training-time initialisation option (residual
            output projections at std / sqrt(2 * num_layers)); stored for completeness, no
            effect on inference. Defaults to False.
        layer_norm_eps (float, optional): Epsilon of the 'uce' encoder's norms. Defaults to 1e-5.

    Any extra key in a checkpoint's ``config.json`` is kept as an attribute (PretrainedConfig
    behaviour). The inference loader uses that to pick up the cell-sentence parameters
    (``chrom_token_offset``, ``pad_length``, ...) when a checkpoint stores them there, and
    flags keys it does not recognise (see ``uce_brain.inference.load_model``).

    Example:
        >>> from uce_brain.model import UCEConfig, UCEModel
        >>> # Initializing a UCE configuration
        >>> config = UCEConfig()
        >>> # Initializing a model from the configuration
        >>> model = UCEModel(config)
    """

    model_type = "uce"

    def __init__(
        self,
        vocab_size: int = 145469,
        embedding_dim: int = 5120,
        d_model: int = 512,
        nhead: int = 16,
        num_layers: int = 4,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        expansion_factor: int = 4,
        max_sequence_length: int = 10000,
        padding_idx: int = 0,
        embedding_init_path: Optional[str] = None,
        embedding_requires_grad: bool = True,
        use_embedding_layer_norm: bool = True,
        embedding_reduction: str = "cls",
        output_embedding_dim: int = 512,
        decoder_layer_dims: Optional[list] = None,
        decoder_dropout: float = 0.1,
        positional_encoding_type: str = "sinusoidal",
        norm_first: bool = False,
        input_projection_type: str = "linear",
        input_projection_expansion_factor: int = 4,
        # v2 options; every default reproduces the original architecture exactly.
        token_norm: str = "layernorm",
        token_scaling_sqrt_d: bool = False,
        pe_scale: float = 1.0,
        separate_gene_layer: bool = False,
        cell_head_type: str = "linear_ln",
        cell_head_dims: Optional[list] = None,
        cell_head_dropout: Optional[float] = None,
        decoder_input_dropout: bool = True,
        encoder_impl: str = "torch",
        final_norm: bool = False,
        norm_type: str = "layernorm",
        qk_norm: bool = False,
        ffn_type: str = "gelu",
        encoder_bias: bool = True,
        attn_dropout: Optional[float] = None,
        scaled_residual_init: bool = False,
        layer_norm_eps: float = 1e-5,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Embedding configuration
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.embedding_init_path = embedding_init_path
        self.embedding_requires_grad = embedding_requires_grad
        self.use_embedding_layer_norm = use_embedding_layer_norm

        # Transformer configuration
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward if dim_feedforward is not None else d_model * expansion_factor
        self.dropout = dropout
        self.activation = activation
        self.expansion_factor = expansion_factor
        self.max_sequence_length = max_sequence_length
        self.positional_encoding_type = positional_encoding_type
        self.norm_first = norm_first

        # Input (gene embedding -> transformer) projection. "linear" keeps the
        # original nn.Sequential state_dict keys (input_gene_embedding_projector.0/1/2.*).
        self.input_projection_type = input_projection_type
        self.input_projection_expansion_factor = input_projection_expansion_factor

        # Embedding aggregation configuration
        self.embedding_reduction = embedding_reduction
        self.output_embedding_dim = output_embedding_dim

        # Decoder configuration
        self.decoder_layer_dims = decoder_layer_dims if decoder_layer_dims is not None else [d_model + output_embedding_dim, 512, 512, 1]
        self.decoder_dropout = decoder_dropout
        self.decoder_input_dropout = bool(decoder_input_dropout)

        # v2 options (see the class docstring)
        self.token_norm = token_norm
        self.token_scaling_sqrt_d = bool(token_scaling_sqrt_d)
        self.pe_scale = float(pe_scale)
        self.separate_gene_layer = bool(separate_gene_layer)
        self.cell_head_type = cell_head_type
        self.cell_head_dims = [int(v) for v in cell_head_dims] if cell_head_dims is not None else None
        self.cell_head_dropout = cell_head_dropout
        self.encoder_impl = encoder_impl
        self.final_norm = bool(final_norm)
        self.norm_type = norm_type
        self.qk_norm = bool(qk_norm)
        self.ffn_type = ffn_type
        self.encoder_bias = bool(encoder_bias)
        self.attn_dropout = attn_dropout
        self.scaled_residual_init = bool(scaled_residual_init)
        self.layer_norm_eps = float(layer_norm_eps)

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        assert self.d_model % self.nhead == 0, f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
        assert self.embedding_reduction in ['cls', 'mean'], f"embedding_reduction must be 'cls' or 'mean', got {self.embedding_reduction}"
        assert self.positional_encoding_type in ['sinusoidal', 'learned'], f"positional_encoding_type must be 'sinusoidal' or 'learned', got {self.positional_encoding_type}"
        assert len(self.decoder_layer_dims) >= 2, "decoder_layer_dims must have at least 2 elements"
        assert self.decoder_layer_dims[-1] == 1, "Last decoder layer dimension must be 1 for binary classification"
        assert self.input_projection_type in ('linear', 'linear_gelu_ln', 'mlp'), \
            f"input_projection_type must be 'linear', 'linear_gelu_ln' or 'mlp', got {self.input_projection_type}"
        assert self.token_norm in ('layernorm', 'l2', 'none'), f"token_norm: {self.token_norm!r}"
        assert self.cell_head_type in ('linear_ln', 'mlp'), f"cell_head_type: {self.cell_head_type!r}"
        if self.cell_head_type == 'mlp':
            assert self.cell_head_dims, "cell_head_type 'mlp' needs cell_head_dims (hidden widths)"
        assert self.encoder_impl in ('torch', 'uce'), f"encoder_impl: {self.encoder_impl!r}"
        assert self.norm_type in ('layernorm', 'rmsnorm'), f"norm_type: {self.norm_type!r}"
        assert self.ffn_type in ('gelu', 'relu', 'swiglu'), f"ffn_type: {self.ffn_type!r}"
        if self.encoder_impl == 'torch':
            # nn.TransformerEncoder has none of these; refuse rather than ignore them.
            assert not (self.final_norm or self.qk_norm or self.ffn_type != 'gelu' or not self.encoder_bias
                        or self.norm_type != 'layernorm' or self.scaled_residual_init), \
                "final_norm / qk_norm / ffn_type / encoder_bias / norm_type / scaled_residual_init need encoder_impl 'uce'"
