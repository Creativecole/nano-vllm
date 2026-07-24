from types import SimpleNamespace

import torch

from nanovllm.attention import HybridAttentionMetadataBuilder
from nanovllm.attention.backend import (
    DeltaNetBackend,
    FullAttentionBackend,
    get_attention_backend_registration,
)
from nanovllm.models.qwen3_5 import Qwen3_5Model
from nanovllm.utils.context import Context


def _tiny_config(layer_types):
    return SimpleNamespace(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        attention_bias=False,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=list(layer_types),
        pad_token_id=0,
        rope_parameters={
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 0],
        },
    )


def test_backend_registry_maps_qwen35_layer_types():
    full = get_attention_backend_registration("full_attention")
    delta = get_attention_backend_registration("linear_attention")

    assert full.backend_cls is FullAttentionBackend
    assert full.module_attr == "self_attn"
    assert delta.backend_cls is DeltaNetBackend
    assert delta.module_attr == "linear_attn"


def test_decoder_layers_bind_backends_without_changing_weight_prefixes():
    model = Qwen3_5Model(
        _tiny_config(["full_attention", "linear_attention"])
    )

    assert isinstance(model.layers[0].attention_backend, FullAttentionBackend)
    assert isinstance(model.layers[1].attention_backend, DeltaNetBackend)
    state_keys = set(model.state_dict())
    assert "layers.0.self_attn.q_proj.weight" in state_keys
    assert "layers.1.linear_attn.in_proj_qkv.weight" in state_keys
    assert not any(".mixer." in key for key in state_keys)


def test_metadata_builder_creates_typed_backend_views():
    positions = torch.tensor([0, 1, 2, 3])
    slot_mapping = torch.arange(4, dtype=torch.int32)
    context = Context(
        is_prefill=False,
        is_mixed=True,
        slot_mapping=slot_mapping,
        num_decode_requests=1,
        decode_context_lens=torch.tensor([8], dtype=torch.int32),
        decode_block_tables=torch.tensor([[0]], dtype=torch.int32),
        prefill_seq_lens=(3,),
        sequence_query_lens=(1, 3),
    )

    metadata = HybridAttentionMetadataBuilder().build(
        context=context,
        request_ids=[10, 20],
        query_lens=[1, 3],
        is_prefilling=[False, True],
        positions=positions,
    )

    assert metadata.common.request_ids == (10, 20)
    assert metadata.common.num_tokens == 4
    assert metadata.full_attention.slot_mapping is slot_mapping
    assert metadata.deltanet.sequence_query_lens == (1, 3)
