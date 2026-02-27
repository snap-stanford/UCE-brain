"""
Data collator for UCE model.

This module provides collators that batch preprocessed cell samples.
"""

import torch
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class UCEDataCollator:
    """
    Data collator for UCE model that batches already-processed samples.

    The samples are already processed by H5ADDataset.__getitem__() which calls
    sample_cell_sentences_mapping_gene(). This collator just stacks them into batches.
    """

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of preprocessed cell samples.

        Args:
            batch: List of dictionaries from H5ADDataset.__getitem__(), each containing:
                - "batch_sentences": Gene sequences [1, pad_length]
                - "mask": Padding mask [1, pad_length]
                - "cell_outputs_X_pe": Protein embedding indices [1, P+N]
                - "cell_outputs_Y": Binary expression labels [1, P+N]
                - "seq_len": Sequence length
                - "idx": Cell index
                - "dataset_num": Dataset ID

        Returns:
            Dictionary with batched tensors:
                - "input_ids": Gene sequences [batch_size, pad_length]
                - "attention_mask": Inverted padding mask [batch_size, pad_length]
                - "target_expression": Expression labels [batch_size, P+N]
                - "target_gene_ids": Protein embedding indices [batch_size, P+N]
                - "cell_indices": Original cell indices [batch_size]
                - "dataset_nums": Dataset IDs [batch_size]
        """
        # Stack tensors, removing the batch dimension of 1 from each sample
        input_ids = torch.cat([sample["batch_sentences"] for sample in batch], dim=0)
        padding_mask = torch.cat([sample["mask"] for sample in batch], dim=0)
        gene_ids = torch.cat([sample["cell_outputs_X_pe"] for sample in batch], dim=0)
        labels = torch.cat([sample["cell_outputs_Y"] for sample in batch], dim=0)

        # Invert padding mask to create attention mask
        # padding_mask: True for padding, False for valid
        # attention_mask: 1 for valid, 0 for padding
        attention_mask = (~padding_mask).long()

        # Extract scalar values
        cell_indices = torch.tensor([sample["idx"] for sample in batch], dtype=torch.long)
        dataset_nums = torch.tensor([sample["dataset_num"] for sample in batch], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_expression": labels,
            "target_gene_ids": gene_ids,
            "cell_indices": cell_indices,
            "dataset_nums": dataset_nums,
        }
