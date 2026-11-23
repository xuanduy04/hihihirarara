# Baseline model quirks on the pinned NeMo-RL stack

What each baseline model needs before it trains on this stack, and why.

## Summary

| Model | Status | What it needs |
|---|---|---|
| Qwen3 family | Works out-of-the-box | Nothing |
| Nemotron-3-Nano-4B | Works after fixes | one-off embedding repair |
| GLM-4.7-Flash | Works after fixes | `config.json` edit |
| Magistral-Small-2507 | Works after fixes | HF tokenizer files; one may instead use the `unsloth/Magistral-Small-2507` mirror |

Note: the config's `mtp_num_layers` must be `0` for the Nemotron model and is `null` for all others.

---

## Qwen3

No issues, no changes.

---

## Nemotron-3-Nano-4B

**Issue:** `found Inf in local grad norm for bucket #0" at iteration 0` due to a defect in the released checkpoint. 79 input-embedding rows have a norm at or below 1e-4 (most around 1e-36; a healthy row is about 0.8). They are 13 invalid-UTF-8 byte tokens (ids 1192, 1193, 1245–1255) plus rare Telugu and Basque word tokens. Upstream reports that a lower learning rate and clipping do not help.

**Fix:** Repair, run once on the checkpoint (same idea as the upstream fix):

```python
import torch
from safetensors.torch import load_file, save_file

p = "<4B model dir>/model.safetensors"
sd = load_file(p)
E, H = sd["backbone.embeddings.weight"], sd["lm_head.weight"]
n = E.float().norm(dim=1)
bad = (n <= 1e-4).nonzero().squeeze(1)
assert len(bad) == 79, len(bad)
target = n[n > 1e-4].pow(2).mean().sqrt()  # 0.7768
h = H[bad].float()
E[bad] = (h / h.norm(dim=1, keepdim=True) * target).to(E.dtype)
save_file(sd, p + ".repaired", metadata={"format": "pt"})
```

Then rename the original to `model.safetensors.orig` and the new file to `model.safetensors`.

Upstream references: Megatron-Bridge issue 4967, Automodel PR 3136.

---

## GLM-4.7-Flash

**Issue:** `transformers 4.57.1` has no `glm4_moe_lite`. The architecture loads correctly as DeepSeek-V3.

**Fix:** Edit the top level of `config.json`:

```json
"model_type": "deepseek_v3",
"architectures": ["DeepseekV3ForCausalLM"],
"scoring_func": "sigmoid"
```

---

## Magistral-Small-2507

**Issue:**

1. `mistralai/Magistral-Small-2507` ships only `tekken.json` as its tokenizer. The driver then builds a `MistralCommonTokenizer`, and the Megatron worker venv has no `mistral-common` (`NameError: ValidationMode`).
2. Installing `mistral-common` is not enough: NeMo RL calls
   `apply_chat_template(add_generation_prompt=…, add_special_tokens=False)`, which raises `ValueError` with that tokenizer.
3. The repo contains `consolidated.safetensors`, and any repo containing that file flips vLLM to Mistral-native config, weights and tokenizer.

**Fix:** The `unsloth/Magistral-Small-2507` mirror was used for its tokenizer files, model weights were identical.

---