"""
UCE Model for Hugging Face Transformers.

This module implements the Universal Cell Embedding (UCE) model architecture
following the Hugging Face transformers library conventions.
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
    """Sinusoidal Positional Encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        pe_slice = self.pe[:x.size(1), 0, :].to(x.dtype)
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
    """Protein/Gene Embedding Layer."""

    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        padding_idx: int = 0,
        init_path: Optional[str] = None,
        use_layer_norm: bool = True,
        requires_grad: bool = False
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx

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
        self.layer_norm = nn.LayerNorm(embedding_dim) if use_layer_norm else None

    def _load_pretrained_embeddings(self, init_path: str):
        """Load pre-trained embeddings from file."""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x


class BinaryExpressionDecoder(nn.Module):
    """Binary Expression Decoder for predicting gene expression."""

    def __init__(
        self,
        layer_dimensions: list,
        dropout: float = 0.1
    ) -> None:
        super().__init__()

        assert len(layer_dimensions) >= 2, "At least two layer dimensions are required."
        assert layer_dimensions[-1] == 1, "Last dimension must be 1 for binary classification."

        layers = []
        layers.append(nn.Dropout(dropout))

        for i in range(len(layer_dimensions) - 1):
            layers.append(nn.Linear(layer_dimensions[i], layer_dimensions[i + 1]))
            if i < len(layer_dimensions) - 2:  # Not the last layer
                layers.append(nn.LayerNorm(layer_dimensions[i + 1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class UCEPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading
    and loading pretrained models.
    """

    config_class = UCEConfig
    base_model_prefix = "uce"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range if hasattr(self.config, 'initializer_range') else 0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range if hasattr(self.config, 'initializer_range') else 0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

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
            requires_grad=config.embedding_requires_grad
        )

        # Project embeddings to transformer dimension
        self.input_gene_embedding_projector = nn.Sequential(
            nn.Linear(config.embedding_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )

        # Positional encoding
        if config.positional_encoding_type == "sinusoidal":
            self.pos_encoder = SinusoidalPositionalEncoding(
                d_model=config.d_model,
                max_len=config.max_sequence_length,
                dropout=config.dropout
            )
        elif config.positional_encoding_type == "learned":
            self.pos_encoder = LearnedPositionalEncoding(
                d_model=config.d_model,
                max_len=config.max_sequence_length,
                dropout=config.dropout
            )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers
        )

        # Output projection
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
            Dictionary containing cell_embedding and gene_embeddings
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
    """

    def __init__(self, config: UCEConfig):
        super().__init__(config)
        self.config = config

        # Base UCE model
        self.uce = UCEModel(config)

        # Expression decoder
        self.expression_decoder = BinaryExpressionDecoder(
            layer_dimensions=config.decoder_layer_dims,
            dropout=config.decoder_dropout
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
