# Qwen3.5 Text Weight Mapping

## Checkpoint Namespace

Multimodal Qwen3.5 checkpoints wrap the text decoder under
`model.language_model`. The nano-vLLM text-only module stores the same decoder under
`model`, so loading uses these explicit prefix mappings:

| Checkpoint prefix | Text-only model prefix |
|---|---|
| `model.language_model.` | `model.` |
| `language_model.` | `model.` |

`lm_head.weight` remains top-level. No layer number, layer type, head count, or tensor
shape is inferred from a checkpoint name.

## Explicit Non-Text Skips

Only declared names or prefixes may be skipped. Qwen3.5 currently declares the
following non-text namespaces:

- `model.visual.` and `visual.`
- `model.multi_modal_projector.` and `multi_modal_projector.`
- `mtp.` and `model.mtp.`
- `draft_model.`
- `auxiliary_head.`

The loader does not use broad substring checks such as `"visual" in name`. Any unknown
weight outside an explicit skip namespace is reported as an unexpected text weight.

## Strict Load Report

`WeightLoadReport` records:

- `loaded`: mapped model parameter names loaded from the checkpoint.
- `missing`: required model parameters not covered by any checkpoint tensor.
- `duplicate`: a parameter or packed shard loaded more than once.
- `unexpected_text_weights`: checkpoint tensors that are neither model parameters nor
  explicit non-text skips.
- `intentionally_skipped_non_text`: audited non-text tensors.
- `tied_aliases`: additional checkpoint names that resolve to an already loaded tied
  parameter.

Strict mode fails when `missing`, `duplicate`, or `unexpected_text_weights` is non-empty.
This makes partial text-model loading visible instead of continuing with random weights.
Packed Qwen3 parameters are checked at `(parameter, shard)` granularity, so loading only
Q while omitting K or V cannot satisfy strict coverage accidentally.

## Tied Embeddings

When `tie_word_embeddings` is enabled, `lm_head.weight` and
`model.embed_tokens.weight` are the same `nn.Parameter`. Coverage is tracked by parameter
identity, so loading either canonical tensor satisfies both aliases. A second declared
alias is recorded separately and is not mistaken for an ordinary duplicate.

## Qwen3 Compatibility

The generic loader retains nano-vLLM's packed Qwen3 mapping for Q/K/V and Gate/Up
projections. Packed mapping replaces a complete module-name component rather than an
arbitrary substring. Qwen3.5 Phase 1 uses native checkpoint-visible projections and
does not pack its weights.

## Validation

The tiny-model tests verify successful prefix mapping, explicit non-text skips,
missing text weights, duplicate targets, unexpected text weights, and tied embedding
coverage. On a real checkpoint, `load_model` prints a one-line count summary after all
safetensor shards have been scanned.
