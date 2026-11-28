import os
import sys

from transformers import AutoTokenizer


THINK_TOKENS_BY_MODEL_FAMILY: dict[str, tuple[str, str]] = {
    "qwen3": ("<think>", "</think>"),
    "glm": ("<think>", "</think>"),
    "nemotron": ("<think>", "</think>"),
    "magistral": ("[THINK]", "[/THINK]"),
}  # fmt: skip


def get_token_ids_config(model_path: str):
    model_name = os.path.basename(model_path.rstrip("/")).lower()
    for family, (think_open, think_close) in THINK_TOKENS_BY_MODEL_FAMILY.items():
        if family in model_name:
            break
    else:
        raise ValueError(
            f"Unknown model family for {model_path} ({model_name=})\n"
            f"Supported families are: {sorted(THINK_TOKENS_BY_MODEL_FAMILY.keys())}"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    vocab = tokenizer.get_vocab()
    return {
        "eos": tokenizer.eos_token_id,
        "think_open": vocab[think_open],
        "think_close": vocab[think_close],
    }


if __name__ == "__main__":
    config = get_token_ids_config(sys.argv[1])
    print("++token_ids={" + ",".join(f"{k}:{v}" for k, v in config.items()) + "}")
