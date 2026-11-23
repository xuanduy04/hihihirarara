from pathlib import Path


def get_system_prompt(is_mistral_model: bool = False):
    prompts_dir = Path(__file__).parent
    system_prompt_prefix = (
        (prompts_dir / "mistral_prefix.txt").read_text().strip() + " "
        if is_mistral_model
        else (prompts_dir / "common_prefix.txt").read_text().strip() + "\n"
    )
    return system_prompt_prefix + (prompts_dir / "common_postfix.txt").read_text().strip()
