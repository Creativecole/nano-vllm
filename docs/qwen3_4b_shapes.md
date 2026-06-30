# Qwen3-4B Attention Shapes

Shapes are loaded from the HuggingFace config by `nanovllm.utils.shapes`.

```bash
python - <<'PY'
from nanovllm.utils.shapes import load_qwen_attention_shapes
print(load_qwen_attention_shapes("Qwen/Qwen3-4B"))
PY
```

The extracted fields include:

- `hidden_size`
- `num_hidden_layers`
- `num_attention_heads`
- `num_key_value_heads`
- `head_dim`
- `intermediate_size`
- `max_position_embeddings`
- `dtype`
- `gqa_ratio`
- Q/K/V/O projection shapes
- decode attention tensor shapes

No Qwen3-4B shape should be hard-coded inside benchmark scripts.

