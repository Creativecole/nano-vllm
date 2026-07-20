from types import SimpleNamespace

import pytest
import torch

from nanovllm.utils.context import get_context


def tiny_qwen35_kwargs(layer_types=None, tie_word_embeddings=False):
    layer_types = layer_types or ["linear_attention", "full_attention"]
    return {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": len(layer_types),
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "hidden_act": "silu",
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "layer_types": list(layer_types),
        "pad_token_id": 0,
        "tie_word_embeddings": tie_word_embeddings,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_interleaved": True,
            "mrope_section": [1, 1, 0],
        },
    }


@pytest.fixture
def tiny_config():
    return SimpleNamespace(**tiny_qwen35_kwargs())


@pytest.fixture
def hf_tiny_config():
    pytest.importorskip("transformers.models.qwen3_5")
    from transformers import Qwen3_5TextConfig

    config = Qwen3_5TextConfig(**tiny_qwen35_kwargs())
    config._attn_implementation = "eager"
    return config


def error_metrics(actual, expected):
    difference = (actual.float() - expected.float()).abs()
    return difference.max().item(), difference.mean().item()


def use_transformers_recurrent_reference(modeling, linear_attention):
    def recurrent_chunk(
        query,
        key,
        value,
        g,
        beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        **_kwargs,
    ):
        return modeling.torch_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    linear_attention.chunk_gated_delta_rule = recurrent_chunk


class TorchPagedAttentionReference(torch.nn.Module):
    def __init__(self, num_blocks, block_size, num_q_heads, num_kv_heads, head_dim, scale):
        super().__init__()
        self.block_size = block_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim)
        self.v_cache = torch.zeros_like(self.k_cache)

    def _store(self, key, value, slots):
        slots = slots.long()
        valid = slots >= 0
        self.k_cache.view(-1, self.num_kv_heads, self.head_dim).index_copy_(
            0, slots[valid], key[valid]
        )
        self.v_cache.view(-1, self.num_kv_heads, self.head_dim).index_copy_(
            0, slots[valid], value[valid]
        )

    def _gather_cache(self, block_table, context_len):
        positions = torch.arange(context_len)
        blocks = block_table[positions // self.block_size].long()
        offsets = positions % self.block_size
        return self.k_cache[blocks, offsets], self.v_cache[blocks, offsets]

    def _attend(self, query, key, value, query_start):
        repeats = self.num_q_heads // self.num_kv_heads
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
        scores = torch.einsum("qhd,khd->hqk", query, key) * self.scale
        q_positions = torch.arange(query.shape[0]) + query_start
        k_positions = torch.arange(key.shape[0])
        scores = scores.masked_fill(
            k_positions.view(1, 1, -1) > q_positions.view(1, -1, 1),
            float("-inf"),
        )
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.einsum("hqk,khd->qhd", probabilities, value)

    def forward(self, query, key, value):
        context = get_context()
        self._store(key, value, context.slot_mapping)
        outputs = []
        if context.is_prefill:
            q_offsets = context.cu_seqlens_q.tolist()
            k_offsets = context.cu_seqlens_k.tolist()
            for row in range(len(q_offsets) - 1):
                q_start, q_end = q_offsets[row], q_offsets[row + 1]
                context_len = k_offsets[row + 1] - k_offsets[row]
                if context.block_tables is None:
                    key_row = key[q_start:q_end]
                    value_row = value[q_start:q_end]
                else:
                    key_row, value_row = self._gather_cache(
                        context.block_tables[row], context_len
                    )
                outputs.append(
                    self._attend(
                        query[q_start:q_end],
                        key_row,
                        value_row,
                        context_len - (q_end - q_start),
                    )
                )
            return torch.cat(outputs, dim=0)

        for row, context_len in enumerate(context.context_lens.tolist()):
            key_row, value_row = self._gather_cache(context.block_tables[row], context_len)
            outputs.append(
                self._attend(query[row : row + 1], key_row, value_row, context_len - 1)
            )
        return torch.stack(outputs, dim=0)
