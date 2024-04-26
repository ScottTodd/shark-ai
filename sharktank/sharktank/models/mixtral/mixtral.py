# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...layers import *
from ...types import *

__all__ = [
    "MixtralModelConfig",
    "PagedMixtralModelV1",
]

################################################################################
# Config
################################################################################


@dataclass
class MixtralModelConfig:
    # hp: configs.LlamaHParams

    # # Block sequence stride for a paged KV cache. This must divide evenly
    # # into the context length.
    # block_seq_stride: int = 16

    # Either "paged" or "direct".
    kv_cache_type: str = "paged"

    # # The device on which to place intermediate state.
    # device: Optional[torch.device] = None

    # # Dtype to use for general FP activations not otherwise configured.
    # activation_dtype: torch.dtype = torch.float32

    # # Dtype to use for attention.
    # attention_dtype: torch.dtype = torch.float32

    # def create_kv_cache(self) -> BaseKVCache:
    #     hp = self.hp
    #     if self.kv_cache_type == "direct":
    #         return DirectKVCache(
    #             block_seq_stride=self.block_seq_stride,
    #             transformer_block_count=hp.block_count,
    #             attn_head_count=hp.attention_head_count_kv,
    #             attn_head_dim=hp.attn_head_dim,
    #             seq_length=hp.context_length,
    #             device=self.device,
    #             dtype=self.attention_dtype,
    #         )
    #     elif self.kv_cache_type == "paged":
    #         return PagedKVCache(
    #             transformer_block_count=hp.block_count,
    #             attn_head_count=hp.attention_head_count_kv,
    #             attn_head_dim=hp.attn_head_dim,
    #             cache_partition_count=2,  # One for each of K/V.
    #             block_seq_stride=self.block_seq_stride,
    #             device=self.device,
    #             dtype=self.attention_dtype,
    #         )
    #     else:
    #         raise NotImplementedError(f"kv_cache_type = {self.kv_cache_type}")


################################################################################
# Models
################################################################################


class PagedMixtralModelV1(BaseCausalLMModel):
    """MixtralModel with ...."""

    def __init__(self, theta: Theta, config: MixtralModelConfig):
        hp = config.hp

    #     super().__init__(
    #         theta,
    #         context_length=config.hp.context_length,
    #         device=config.device,
    #         activation_dtype=config.activation_dtype,
    #         attention_dtype=config.attention_dtype,
    #     )
    #     self.config = config
    #     self.hp = hp
    #     self.cache = config.create_kv_cache()
    #     self.activation_dtype = config.activation_dtype
    #     self.add_module(
    #         "token_embedding",
    #         TokenEmbeddingLayer(theta("token_embd"), dtype=config.activation_dtype),
    #     )
    #     self.add_module(
    #         "attention_embedding",
    #         RotaryEmbeddingLayer(
    #             rope_dimension_count=hp.rope_dimension_count,
    #             max_seqlen=hp.context_length,
    #             device=self.device,
    #         ),
    #     )
    #     self.add_module(
    #         "output_norm",
    #         RMSNormLayer(
    #             theta("output_norm"), epsilon=self.hp.attention_layer_norm_rms_epsilon
    #         ),
    #     )
    #     self.add_module("output_lm_head", LinearLayer(theta("output")))

    #     self.attn_blocks = nn.ModuleList(
    #         [
    #             PagedLlamaAttentionBlock(
    #                 theta("blk", n),
    #                 block_index=n,
    #                 cache=self.cache,
    #                 head_count=hp.attention_head_count,
    #                 head_dim=hp.attn_head_dim,
    #                 head_count_kv=hp.attention_head_count_kv,
    #                 rms_epsilon=hp.attention_layer_norm_rms_epsilon,
    #             )
    #             for n in range(hp.block_count)
    #         ]
    #     )

    # def prefill(
    #     self,
    #     # [bs, batch_seq_len]
    #     tokens: torch.Tensor,
    #     *,
    #     # [1, 1, batch_seq_len, batch_seq_len]
    #     attention_mask: torch.Tensor,
    #     # [bs, batch_seq_len // block_seq_stride]
    #     seq_block_ids: torch.Tensor,
    #     cache_state: list[torch.Tensor],
    # ):
    #     self._assert_device(tokens)
    #     self._assert_device(attention_mask, dtype=self.activation_dtype)
    #     self._assert_device(seq_block_ids)
    #     self._assert_device(*cache_state, dtype=self.activation_dtype)
    #     h = self.token_embedding(tokens)
    #     self.trace_tensor("llama.token_embedding", h)

    #     # Iterate over attention blocks.
    #     for block_idx, block in enumerate(self.attn_blocks):
    #         if block_idx == 0:
    #             self.trace_tensor(f"llama.attn_block.{block_idx}.input", h)
    #         h = block(
    #             h,
    #             embedding=self.attention_embedding,
    #             start_index=0,
    #             attention_mask=attention_mask,
    #             cache_state=cache_state,
    #             seq_block_ids=seq_block_ids,
    #         )
    #         self.trace_tensor(f"llama.attn_block.{block_idx}.output", h)

    #     h = self.output_norm(h)
    #     logits = self.output_lm_head(h)
    #     return logits

    # def decode(
    #     self,
    #     # [bs, 1]
    #     tokens: torch.Tensor,
    #     *,
    #     # [bs, 1, 1, batch_seq_len]
    #     attention_mask: torch.Tensor,
    #     # [bs] of starting positions
    #     start_positions: torch.Tensor,
    #     # [bs, batch_seq_len // block_seq_stride]
    #     seq_block_ids: torch.Tensor,
    #     cache_state: list[torch.Tensor],
    # ):
    #     self._assert_device(tokens)
    #     self._assert_device(attention_mask, dtype=self.activation_dtype)
    #     self._assert_device(start_positions)
    #     self._assert_device(*cache_state, dtype=self.activation_dtype)
    #     bs, _ = tokens.shape
    #     # Precompute a position based mask for computing rope embeddings
    #     # as it is the same for all blocks.
    #     embedding_batch_mask = self.attention_embedding.compute_batch_mask(
    #         start_positions, batch_seq_len=1
    #     )
    #     self.trace_tensor("llama.embedding_batch_mask", embedding_batch_mask)

    #     # Allocate per-block temporary K/V tensors. These temporaries hold
    #     # one block's K/V state for the maximum context length.
    #     xk_temp = torch.empty(
    #         [
    #             bs,
    #             self.context_length,
    #             self.hp.attention_head_count_kv,
    #             self.hp.attn_head_dim,
    #         ],
    #         dtype=self.config.activation_dtype,
    #         device=self.device,
    #     )
    #     xv_temp = torch.empty(
    #         [
    #             bs,
    #             self.context_length,
    #             self.hp.attention_head_count_kv,
    #             self.hp.attn_head_dim,
    #         ],
    #         dtype=self.config.activation_dtype,
    #         device=self.device,
    #     )

    #     h = self.token_embedding(tokens)
    #     self.trace_tensor("llama.token_embedding", h)

    #     # Iterate over attention blocks.
    #     for block_idx, block in enumerate(self.attn_blocks):
    #         if block_idx == 0:
    #             self.trace_tensor(f"llama.attn_block.{block_idx}.input", h)
    #         h = block(
    #             h,
    #             start_positions=start_positions,
    #             embedding=self.attention_embedding,
    #             embedding_batch_mask=embedding_batch_mask,
    #             attention_mask=attention_mask,
    #             cache_state=cache_state,
    #             seq_block_ids=seq_block_ids,
    #             xk_temp=xk_temp,
    #             xv_temp=xv_temp,
    #         )
    #         self.trace_tensor(f"llama.attn_block.{block_idx}.output", h)

    #     h = self.output_norm(h)
    #     logits = self.output_lm_head(h)
    #     return logits


################################################################################
# Layers
################################################################################
