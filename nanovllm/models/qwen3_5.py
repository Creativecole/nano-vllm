from collections.abc import Mapping

import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.engine.layer_state import (
    DeltaNetState,
    DeltaNetStateSpec,
    PagedKVState,
    PagedKVStateSpec,
)
from nanovllm.utils.context import get_context
from nanovllm.utils.profiler import profile_range


def _config_value(config, name, default=None):
    value = getattr(config, name, default)
    return default if value is None else value


class Qwen3_5RMSNorm(nn.Module):
    """Qwen3.5 RMSNorm uses a learned `(1 + weight)` scale."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float()
        output = output * torch.rsqrt(output.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.to(x.dtype)


class Qwen3_5RMSNormGated(nn.Module):
    """Per-value-head RMSNorm followed by the DeltaNet SiLU gate."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        normalized = normalized * torch.rsqrt(
            normalized.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        # Match Transformers: round the normalized/weighted value to the input
        # dtype before applying the FP32 gate activation.
        normalized = self.weight * normalized.to(input_dtype)
        normalized = normalized * F.silu(gate.float())
        return normalized.to(input_dtype)


class Qwen3_5RotaryEmbedding(nn.Module):
    """Text-only Qwen3.5 partial RoPE with interleaved mRoPE support."""

    def __init__(self, config):
        super().__init__()
        rope_parameters = _config_value(config, "rope_parameters", {})
        if not isinstance(rope_parameters, Mapping):
            rope_parameters = dict(rope_parameters)
        head_dim = _config_value(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        partial_rotary_factor = rope_parameters.get(
            "partial_rotary_factor",
            _config_value(config, "partial_rotary_factor", 0.25),
        )
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        if self.rotary_dim <= 0 or self.rotary_dim % 2:
            raise ValueError(f"rotary_dim must be a positive even number, got {self.rotary_dim}")
        rope_theta = rope_parameters.get(
            "rope_theta",
            _config_value(config, "rope_theta", 10000000.0),
        )
        inv_freq = 1.0 / (
            rope_theta
            ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = tuple(rope_parameters.get("mrope_section", (11, 11, 10)))

    def _apply_interleaved_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        # freqs: [3, batch, seq_len, rotary_dim / 2]
        output = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            output[..., slice(offset, length, 3)] = freqs[dim, ..., slice(offset, length, 3)]
        return output

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_positions = position_ids.ndim == 1
        if flat_positions:
            position_ids = position_ids.view(1, 1, -1).expand(3, 1, -1)
        elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]
        elif position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        if position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError(
                "Qwen3.5 position_ids must have shape [batch, seq], [3, batch, seq], "
                f"or [4, batch, seq], got {tuple(position_ids.shape)}"
            )
        if not flat_positions and (
            position_ids.shape[1] != x.shape[0] or position_ids.shape[2] != x.shape[1]
        ):
            raise ValueError(
                "position_ids batch/sequence dimensions must match the input: "
                f"positions={tuple(position_ids.shape)}, input={tuple(x.shape)}"
            )

        # [3, batch, seq_len, rotary_dim / 2]
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq.float().view(1, 1, 1, -1)
        freqs = self._apply_interleaved_mrope(freqs)
        embeddings = torch.cat((freqs, freqs), dim=-1)
        cos = embeddings.cos().to(x.dtype)
        sin = embeddings.sin().to(x.dtype)
        if flat_positions:
            cos, sin = cos.squeeze(0), sin.squeeze(0)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # query/key: [batch, heads, seq_len, head_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = query[..., :rotary_dim], query[..., rotary_dim:]
    k_rot, k_pass = key[..., :rotary_dim], key[..., rotary_dim:]
    q_embed = q_rot * cos + rotate_half(q_rot) * sin
    k_embed = k_rot * cos + rotate_half(k_rot) * sin
    return torch.cat((q_embed, q_pass), dim=-1), torch.cat((k_embed, k_pass), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    # [batch, kv_heads, seq_len, head_dim] -> [batch, q_heads, seq_len, head_dim]
    if repeats == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, repeats, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * repeats, seq_len, head_dim)


def _causal_attention_mask(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    mask = torch.zeros(seq_len, seq_len, dtype=torch.float32, device=device)
    mask.masked_fill_(
        torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1),
        float("-inf"),
    )
    mask = mask.view(1, 1, seq_len, seq_len).expand(batch_size, 1, -1, -1)
    if attention_mask is None:
        return mask
    if attention_mask.ndim == 4:
        return mask + attention_mask.float()
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be rank 2 or 4, got {attention_mask.ndim}")
    key_mask = attention_mask.to(torch.bool).view(batch_size, 1, 1, seq_len)
    return mask.masked_fill(~key_mask, float("-inf"))


class Qwen3_5Attention(nn.Module):

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.head_dim = _config_value(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        self.scaling = self.head_dim**-0.5
        bias = _config_value(config, "attention_bias", False)
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim * 2,
            bias=bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=bias,
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.paged_attention = None

    def enable_paged_attention(self) -> None:
        from nanovllm.layers.attention import Attention

        self.paged_attention = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def bind_paged_state(self, state: PagedKVState) -> None:
        if self.paged_attention is None:
            self.enable_paged_attention()
        self.paged_attention.k_cache = state.k_cache
        self.paged_attention.v_cache = state.v_cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim == 2:
            if self.paged_attention is None:
                raise RuntimeError("Packed Qwen3.5 attention requires the paged serving backend")
            num_tokens = hidden_states.shape[0]
            query_and_gate = self.q_proj(hidden_states).view(
                num_tokens, self.num_heads, self.head_dim * 2
            )
            query, gate = query_and_gate.chunk(2, dim=-1)
            gate = gate.reshape(num_tokens, self.num_heads * self.head_dim)
            query = self.q_norm(query)
            key = self.k_norm(
                self.k_proj(hidden_states).view(num_tokens, self.num_kv_heads, self.head_dim)
            )
            value = self.v_proj(hidden_states).view(
                num_tokens, self.num_kv_heads, self.head_dim
            )
            query, key = apply_partial_rotary_pos_emb(
                query, key, *position_embeddings
            )
            output = self.paged_attention(query, key, value)
            if output.ndim == 4:
                output = output.squeeze(1)
            output = output.reshape(num_tokens, self.num_heads * self.head_dim)
            output = output * torch.sigmoid(gate)
            return self.o_proj(output)

        batch_size, seq_len, _ = hidden_states.shape
        query_and_gate = self.q_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim * 2,
        )
        query, gate = query_and_gate.chunk(2, dim=-1)
        gate = gate.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(
            self.k_proj(hidden_states).view(
                batch_size, seq_len, self.num_kv_heads, self.head_dim
            )
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        query, key = apply_partial_rotary_pos_emb(query, key, *position_embeddings)
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)

        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scaling
        scores = scores + _causal_attention_mask(
            batch_size, seq_len, hidden_states.device, attention_mask
        )
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(probabilities, value).transpose(1, 2).contiguous()
        output = output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        output = output * torch.sigmoid(gate)
        return self.o_proj(output)


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def gated_delta_rule_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential FP32 Gated DeltaNet recurrence.

    Args:
        query: [batch, seq_len, value_heads, key_head_dim]
        key: [batch, seq_len, value_heads, key_head_dim]
        value: [batch, seq_len, value_heads, value_head_dim]
        decay: [batch, seq_len, value_heads]
        beta: [batch, seq_len, value_heads]
        initial_state: [batch, value_heads, key_head_dim, value_head_dim]
    """
    input_dtype = query.dtype
    query = l2norm(query.float()) * (query.shape[-1] ** -0.5)
    key = l2norm(key.float())
    value = value.float()
    decay = decay.float()
    beta = beta.float()

    batch_size, seq_len, num_heads, key_dim = key.shape
    value_dim = value.shape[-1]
    if initial_state is None:
        state = torch.zeros(
            batch_size,
            num_heads,
            key_dim,
            value_dim,
            dtype=torch.float32,
            device=query.device,
        )
    else:
        expected = (batch_size, num_heads, key_dim, value_dim)
        if tuple(initial_state.shape) != expected:
            raise ValueError(
                f"initial_state has shape {tuple(initial_state.shape)}, expected {expected}"
            )
        state = initial_state.float()

    outputs = []
    for token_idx in range(seq_len):
        q_t = query[:, token_idx]
        k_t = key[:, token_idx]
        v_t = value[:, token_idx]
        decay_t = decay[:, token_idx].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, token_idx].unsqueeze(-1)

        decayed_state = state * decay_t
        retrieved = (decayed_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - retrieved) * beta_t
        state = decayed_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((state * q_t.unsqueeze(-1)).sum(dim=-2))

    output = torch.stack(outputs, dim=1).to(input_dtype)
    return output, state


def chunked_gated_delta_rule_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 PyTorch chunked Gated DeltaNet reference with BF16/FP16 output."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    input_dtype = query.dtype
    query = l2norm(query.float()) * (query.shape[-1] ** -0.5)
    key = l2norm(key.float())
    value = value.float()
    decay = decay.float()
    beta = beta.float()

    query, key, value = (
        tensor.transpose(1, 2).contiguous() for tensor in (query, key, value)
    )
    decay = decay.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous()
    batch_size, num_heads, seq_len, key_dim = key.shape
    value_dim = value.shape[-1]
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    decay = F.pad(decay, (0, pad_size))
    padded_len = seq_len + pad_size

    value_beta = value * beta.unsqueeze(-1)
    key_beta = key * beta.unsqueeze(-1)
    query, key, key_beta, value_beta = (
        tensor.reshape(
            batch_size,
            num_heads,
            -1,
            chunk_size,
            tensor.shape[-1],
        )
        for tensor in (query, key, key_beta, value_beta)
    )
    decay = decay.reshape(batch_size, num_heads, -1, chunk_size).cumsum(dim=-1)

    causal_mask = torch.triu(
        torch.ones(
            chunk_size,
            chunk_size,
            dtype=torch.bool,
            device=query.device,
        ),
        diagonal=0,
    )
    decay_mask = (
        (decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()
    ).tril()
    transition = -(
        (key_beta @ key.transpose(-1, -2)) * decay_mask
    ).masked_fill(causal_mask, 0)

    # Build every chunk's lower-triangular inverse in parallel. This loop is fixed by
    # chunk_size; it does not scale with the full prompt length.
    for row_idx in range(1, chunk_size):
        row = transition[..., row_idx, :row_idx].clone()
        lower = transition[..., :row_idx, :row_idx].clone()
        transition[..., row_idx, :row_idx] = row + (
            row.unsqueeze(-1) * lower
        ).sum(-2)
    transition = transition + torch.eye(
        chunk_size, dtype=transition.dtype, device=transition.device
    )

    value_updates = transition @ value_beta
    decayed_keys = transition @ (key_beta * decay.exp().unsqueeze(-1))
    if initial_state is None:
        state = torch.zeros(
            batch_size,
            num_heads,
            key_dim,
            value_dim,
            dtype=torch.float32,
            device=query.device,
        )
    else:
        expected = (batch_size, num_heads, key_dim, value_dim)
        if tuple(initial_state.shape) != expected:
            raise ValueError(
                f"initial_state has shape {tuple(initial_state.shape)}, expected {expected}"
            )
        state = initial_state.float()

    output = torch.zeros_like(value_updates)
    num_chunks = padded_len // chunk_size
    for chunk_idx in range(num_chunks):
        query_chunk = query[:, :, chunk_idx]
        key_chunk = key[:, :, chunk_idx]
        value_chunk = value_updates[:, :, chunk_idx]
        chunk_decay = decay[:, :, chunk_idx]
        attention = (
            query_chunk @ key_chunk.transpose(-1, -2)
        ) * decay_mask[:, :, chunk_idx]
        state_correction = decayed_keys[:, :, chunk_idx] @ state
        corrected_value = value_chunk - state_correction
        state_output = (query_chunk * chunk_decay.exp().unsqueeze(-1)) @ state
        output[:, :, chunk_idx] = state_output + attention @ corrected_value
        final_decay = chunk_decay[:, :, -1]
        state = state * final_decay.exp().unsqueeze(-1).unsqueeze(-1) + (
            key_chunk
            * (final_decay.unsqueeze(-1) - chunk_decay).exp().unsqueeze(-1)
        ).transpose(-1, -2) @ corrected_value

    output = output.reshape(batch_size, num_heads, padded_len, value_dim)
    output = output[:, :, :seq_len].transpose(1, 2).contiguous().to(input_dtype)
    return output, state


class Qwen3_5GatedDeltaNet(nn.Module):

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        if self.num_v_heads % self.num_k_heads:
            raise ValueError("linear value heads must be divisible by linear key heads")
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.deltanet_backend = _config_value(
            config, "nanovllm_deltanet_backend", "sequential"
        )
        self.deltanet_chunk_size = int(
            _config_value(config, "nanovllm_deltanet_chunk_size", 64)
        )
        if self.deltanet_backend not in ("sequential", "chunked"):
            raise ValueError(f"Unsupported DeltaNet backend {self.deltanet_backend!r}")
        if self.deltanet_chunk_size <= 0:
            raise ValueError("DeltaNet chunk size must be positive")
        self._diagnostics_enabled = False
        self.reset_diagnostics()

        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            bias=False,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def set_diagnostics(self, enabled: bool, reset: bool = True) -> None:
        self._diagnostics_enabled = bool(enabled)
        if reset:
            self.reset_diagnostics()

    def reset_diagnostics(self) -> None:
        self._diagnostics = {
            "layer_idx": self.layer_idx,
            "recurrence_calls": 0,
            "equal_length_batched_prefill_calls": 0,
            "variable_length_fallback_calls": 0,
            "fallback_sequences": 0,
            "calls": [],
        }

    def get_diagnostics(self) -> dict[str, object]:
        return {
            key: [dict(item) for item in value]
            if key == "calls"
            else value
            for key, value in self._diagnostics.items()
        }

    def _run_recurrence(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        decay: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_chunked = self.deltanet_backend == "chunked" and query.shape[1] > 1
        backend = "chunked" if use_chunked else "sequential"
        if self._diagnostics_enabled:
            context = get_context()
            self._diagnostics["recurrence_calls"] += 1
            self._diagnostics["calls"].append(
                {
                    "backend": backend,
                    "is_prefill": context.is_prefill,
                    "query_shape": list(query.shape),
                    "key_shape": list(key.shape),
                    "value_shape": list(value.shape),
                    "state_shape": list(initial_state.shape)
                    if initial_state is not None
                    else None,
                    "query_dtype": str(query.dtype),
                    "state_dtype": str(initial_state.dtype)
                    if initial_state is not None
                    else None,
                }
            )
        with profile_range(f"qwen35_deltanet_recurrence_{backend}"):
            if use_chunked:
                return chunked_gated_delta_rule_reference(
                    query,
                    key,
                    value,
                    decay,
                    beta,
                    initial_state=initial_state,
                    chunk_size=self.deltanet_chunk_size,
                )
            return gated_delta_rule_reference(
                query,
                key,
                value,
                decay,
                beta,
                initial_state=initial_state,
            )

    def _forward_stateful_chunk(
        self,
        hidden_states: torch.Tensor,
        layer_state: DeltaNetState,
    ) -> torch.Tensor:
        # hidden_states: [batch, seq_len, hidden_size]
        batch_size, seq_len, _ = hidden_states.shape
        with profile_range("qwen35_deltanet_conv"):
            raw_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
            conv_input = torch.cat(
                (layer_state.conv_state[:, :, 1:], raw_qkv), dim=-1
            )
            mixed_qkv = F.silu(
                F.conv1d(
                    conv_input,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    groups=self.conv_dim,
                )
            ).transpose(1, 2)
            new_conv_state = torch.cat((layer_state.conv_state, raw_qkv), dim=-1)[
                :, :, -self.conv_kernel_size :
            ]
            layer_state.conv_state.copy_(new_conv_state)

        query, key, value = mixed_qkv.split(
            (self.key_dim, self.key_dim, self.value_dim), dim=-1
        )
        query = query.view(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.view(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.view(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        repeats = self.num_v_heads // self.num_k_heads
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)
        beta = torch.sigmoid(self.in_proj_b(hidden_states))
        decay = -self.A_log.float().exp() * F.softplus(
            self.in_proj_a(hidden_states).float() + self.dt_bias.float()
        )
        core_output, final_state = self._run_recurrence(
            query,
            key,
            value,
            decay,
            beta,
            initial_state=layer_state.recurrent_state,
        )
        layer_state.recurrent_state.copy_(final_state)
        with profile_range("qwen35_deltanet_output"):
            z = self.in_proj_z(hidden_states).view(
                batch_size, seq_len, self.num_v_heads, self.head_v_dim
            )
            output = self.norm(core_output, z).reshape(
                batch_size, seq_len, self.value_dim
            )
            return self.out_proj(output)

    def _forward_packed(
        self,
        hidden_states: torch.Tensor,
        layer_state: DeltaNetState,
    ) -> torch.Tensor:
        context = get_context()
        if context.is_prefill:
            if context.prefill_seq_lens is None:
                raise RuntimeError("Packed prefill is missing sequence lengths")
            seq_lens = context.prefill_seq_lens
            batch_size = len(seq_lens)
            equal_length = len(set(seq_lens)) == 1
            if self.deltanet_backend == "chunked" and equal_length:
                seq_len = seq_lens[0]
                if hidden_states.shape[0] != batch_size * seq_len:
                    raise ValueError(
                        "Packed prefill token count does not match equal-length batch: "
                        f"tokens={hidden_states.shape[0]}, batch={batch_size}, "
                        f"seq_len={seq_len}"
                    )
                if layer_state.conv_state.shape[0] != batch_size:
                    raise ValueError(
                        "DeltaNet state batch does not match packed prefill batch: "
                        f"state={layer_state.conv_state.shape[0]}, batch={batch_size}"
                    )
                if self._diagnostics_enabled:
                    self._diagnostics["equal_length_batched_prefill_calls"] += 1
                batched = hidden_states.reshape(
                    batch_size, seq_len, hidden_states.shape[-1]
                )
                output = self._forward_stateful_chunk(batched, layer_state)
                return output.reshape(-1, output.shape[-1])

            if self._diagnostics_enabled:
                self._diagnostics["variable_length_fallback_calls"] += 1
                self._diagnostics["fallback_sequences"] += batch_size
            chunks = hidden_states.split(seq_lens, dim=0)
            outputs = []
            for batch_idx, chunk in enumerate(chunks):
                request_state = DeltaNetState(
                    layer_idx=self.layer_idx,
                    conv_state=layer_state.conv_state[batch_idx : batch_idx + 1],
                    recurrent_state=layer_state.recurrent_state[batch_idx : batch_idx + 1],
                )
                output = self._forward_stateful_chunk(chunk.unsqueeze(0), request_state)
                outputs.append(output.squeeze(0))
            return torch.cat(outputs, dim=0)
        output = self._forward_stateful_chunk(hidden_states.unsqueeze(1), layer_state)
        return output.squeeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_state: bool = False,
        layer_state: DeltaNetState | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if layer_state is not None:
            if hidden_states.ndim != 2:
                raise ValueError("Stateful DeltaNet expects packed/decode rank-2 hidden states")
            return self._forward_packed(hidden_states, layer_state)
        # hidden_states: [batch, seq_len, hidden_size]
        if attention_mask is not None and attention_mask.ndim == 2:
            hidden_states = hidden_states * attention_mask.to(hidden_states.dtype).unsqueeze(-1)
        batch_size, seq_len, _ = hidden_states.shape

        # mixed_qkv: [batch, conv_dim, seq_len]
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[..., :seq_len]).transpose(1, 2)
        query, key, value = mixed_qkv.split(
            (self.key_dim, self.key_dim, self.value_dim), dim=-1
        )
        query = query.view(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.view(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.view(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        repeats = self.num_v_heads // self.num_k_heads
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)
        beta = torch.sigmoid(self.in_proj_b(hidden_states))
        decay = -self.A_log.float().exp() * F.softplus(
            self.in_proj_a(hidden_states).float() + self.dt_bias.float()
        )

        core_output, final_state = self._run_recurrence(
            query,
            key,
            value,
            decay,
            beta,
        )
        # core_output/z: [batch, seq_len, value_heads, value_head_dim]
        z = self.in_proj_z(hidden_states).view(
            batch_size, seq_len, self.num_v_heads, self.head_v_dim
        )
        output = self.norm(core_output, z).reshape(batch_size, seq_len, self.value_dim)
        output = self.out_proj(output)
        if return_state:
            return output, final_state
        return output


class Qwen3_5MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(f"Qwen3.5 reference supports silu, got {config.hidden_act!r}")
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3_5DecoderLayer(nn.Module):

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.block_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        else:
            raise ValueError(
                f"Unsupported Qwen3.5 layer type {self.block_type!r} at layer {layer_idx}"
            )
        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        layer_state: DeltaNetState | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.block_type == "linear_attention":
            with profile_range("qwen35_deltanet_mixer"):
                hidden_states = self.linear_attn(
                    hidden_states,
                    attention_mask=attention_mask,
                    layer_state=layer_state,
                )
        else:
            with profile_range("qwen35_full_attention_mixer"):
                hidden_states = self.self_attn(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        with profile_range("qwen35_mlp"):
            hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3_5Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        if len(config.layer_types) != config.num_hidden_layers:
            raise ValueError(
                "layer_types length must match num_hidden_layers: "
                f"{len(config.layer_types)} != {config.num_hidden_layers}"
            )
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=_config_value(config, "pad_token_id", None),
        )
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5RotaryEmbedding(config)

    def set_deltanet_diagnostics(self, enabled: bool, reset: bool = True) -> None:
        for layer in self.layers:
            if layer.block_type == "linear_attention":
                layer.linear_attn.set_diagnostics(enabled, reset=reset)

    def reset_deltanet_diagnostics(self) -> None:
        for layer in self.layers:
            if layer.block_type == "linear_attention":
                layer.linear_attn.reset_diagnostics()

    def get_deltanet_diagnostics(self) -> dict[int, dict[str, object]]:
        return {
            layer.layer_idx: layer.linear_attn.get_diagnostics()
            for layer in self.layers
            if layer.block_type == "linear_attention"
        }

    def _normalize_positions(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if positions is None:
            return torch.arange(seq_len, device=input_ids.device).view(1, -1).expand(batch_size, -1)
        if positions.ndim == 1:
            if positions.numel() != seq_len:
                raise ValueError(
                    f"positions length {positions.numel()} does not match seq_len {seq_len}"
                )
            return positions.view(1, -1).expand(batch_size, -1)
        if positions.ndim == 2 and tuple(positions.shape) != (batch_size, seq_len):
            raise ValueError(
                f"positions has shape {tuple(positions.shape)}, expected {(batch_size, seq_len)}"
            )
        if positions.ndim not in (2, 3):
            raise ValueError(f"positions must be rank 1, 2, or 3, got {positions.ndim}")
        return positions

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        layer_states: dict[int, DeltaNetState] | None = None,
    ) -> torch.Tensor:
        if layer_states is not None:
            if input_ids.ndim != 1 or positions is None or positions.ndim != 1:
                raise ValueError(
                    "Stateful Qwen3.5 serving expects rank-1 packed input_ids and positions"
                )
            hidden_states = self.embed_tokens(input_ids)
            position_embeddings = self.rotary_emb(hidden_states, positions)
            for layer in self.layers:
                hidden_states = layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    layer_state=layer_states.get(layer.layer_idx),
                )
            return self.norm(hidden_states)

        squeeze_batch = input_ids.ndim == 1
        if squeeze_batch:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be rank 1 or 2, got {input_ids.ndim}")
        positions = self._normalize_positions(input_ids, positions)

        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(hidden_states, positions)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states.squeeze(0) if squeeze_batch else hidden_states


class Qwen3_5ForCausalLM(nn.Module):
    supports_stateful_serving = True
    supports_prefix_cache = False
    supports_cuda_graph = False
    packed_modules_mapping = {}
    checkpoint_prefix_mapping = (
        ("model.language_model.", "model."),
        ("language_model.", "model."),
    )
    intentionally_skipped_weight_prefixes = (
        "model.visual.",
        "visual.",
        "model.multi_modal_projector.",
        "multi_modal_projector.",
        "mtp.",
        "model.mtp.",
        "draft_model.",
        "auxiliary_head.",
    )

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3_5Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_word_embeddings = bool(_config_value(config, "tie_word_embeddings", False))
        if self.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        layer_states: dict[int, DeltaNetState] | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, attention_mask, layer_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if context.is_prefill and hidden_states.ndim == 2:
            hidden_states = hidden_states[context.cu_seqlens_q[1:] - 1].contiguous()
        return self.lm_head(hidden_states)

    def set_deltanet_diagnostics(self, enabled: bool, reset: bool = True) -> None:
        self.model.set_deltanet_diagnostics(enabled, reset=reset)

    def reset_deltanet_diagnostics(self) -> None:
        self.model.reset_deltanet_diagnostics()

    def get_deltanet_diagnostics(self) -> dict[int, dict[str, object]]:
        return self.model.get_deltanet_diagnostics()

    def get_layer_state_specs(self) -> list[PagedKVStateSpec | DeltaNetStateSpec]:
        parameter_dtype = self.model.embed_tokens.weight.dtype
        specs = []
        for layer in self.model.layers:
            if layer.block_type == "full_attention":
                specs.append(
                    PagedKVStateSpec(
                        layer_idx=layer.layer_idx,
                        layer_type=layer.block_type,
                        num_kv_heads=layer.self_attn.num_kv_heads,
                        head_dim=layer.self_attn.head_dim,
                        dtype=parameter_dtype,
                    )
                )
            else:
                mixer = layer.linear_attn
                specs.append(
                    DeltaNetStateSpec(
                        layer_idx=layer.layer_idx,
                        layer_type=layer.block_type,
                        conv_dim=mixer.conv_dim,
                        conv_width=mixer.conv_kernel_size,
                        num_value_heads=mixer.num_v_heads,
                        key_head_dim=mixer.head_k_dim,
                        value_head_dim=mixer.head_v_dim,
                        conv_dtype=parameter_dtype,
                    )
                )
        return specs

    def enable_paged_attention(self) -> None:
        for layer in self.model.layers:
            if layer.block_type == "full_attention":
                layer.self_attn.enable_paged_attention()

    def bind_paged_kv_states(self, states: dict[int, PagedKVState]) -> None:
        for layer in self.model.layers:
            if layer.block_type == "full_attention":
                layer.self_attn.bind_paged_state(states[layer.layer_idx])

    def forward_logits(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.compute_logits(self.forward(input_ids, positions, attention_mask))
