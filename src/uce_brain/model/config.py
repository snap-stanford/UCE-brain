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

        # Embedding aggregation configuration
        self.embedding_reduction = embedding_reduction
        self.output_embedding_dim = output_embedding_dim

        # Decoder configuration
        self.decoder_layer_dims = decoder_layer_dims if decoder_layer_dims is not None else [d_model + output_embedding_dim, 512, 512, 1]
        self.decoder_dropout = decoder_dropout

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        assert self.d_model % self.nhead == 0, f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
        assert self.embedding_reduction in ['cls', 'mean'], f"embedding_reduction must be 'cls' or 'mean', got {self.embedding_reduction}"
        assert self.positional_encoding_type in ['sinusoidal', 'learned'], f"positional_encoding_type must be 'sinusoidal' or 'learned', got {self.positional_encoding_type}"
        assert len(self.decoder_layer_dims) >= 2, "decoder_layer_dims must have at least 2 elements"
        assert self.decoder_layer_dims[-1] == 1, "Last decoder layer dimension must be 1 for binary classification"
