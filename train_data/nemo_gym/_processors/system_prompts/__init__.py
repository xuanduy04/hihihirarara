from pathlib import Path


def get_system_prompt(is_mistral_model: bool = False, *, is_instruct_data: bool = False):
    prompts_dir = Path(__file__).parent
    system_prompt_prefix = (
        (prompts_dir / "mistral_prefix.txt").read_text().strip()
        if is_mistral_model
        else (prompts_dir / "common_prefix.txt").read_text().strip()
    )
    system_prompt_separator = " " if is_mistral_model else "\n"
    system_prompt_postfix = "" if is_instruct_data else (prompts_dir / "common_postfix.txt").read_text().strip()
    return (system_prompt_prefix + system_prompt_separator + system_prompt_postfix).strip()


if __name__ == "__main__":
    print("=" * 67)
    for mistral in [True, False]:
        for instruct in [False, True]:
            mistral_str = f"{'' if mistral else 'Non-'}Mistral models"
            instruct_str = f"{'' if instruct else 'Non-'}Instruct data"
            print(f"\tSystem prompt of {mistral_str} for {instruct_str}:")
            print('"""')
            print(get_system_prompt(is_mistral_model=mistral, is_instruct_data=instruct))
            print('"""')
            print("=" * 67)
