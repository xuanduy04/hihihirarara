from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

_COLOR = "\033[1;32m" if sys.stderr.isatty() else ""
_RESET = "\033[0m" if sys.stderr.isatty() else ""
logging.basicConfig(level=logging.INFO, format=f"{_COLOR}%(asctime)s [%(levelname)s] %(message)s{_RESET}")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
HF_PROCESSORS_DIR = SCRIPT_DIR.parents[1] / "hf" / "_processors"
HF_DATA_DIR = SCRIPT_DIR.parents[1] / "hf" / "data"
DATA_DIR = SCRIPT_DIR.parent / "data"

TULU_DATASET = "allenai/tulu-3-sft-personas-instruction-following"
VARIANTS = (("mistral", True), ("common", False))


def _preprocess_underscore_args(argv: list[str]) -> list[str]:
    """Replace underscores with hyphens in ``--arg_name`` / ``--arg_name=val`` prefixes.

    argparse normalises ``-`` and ``_`` internally, so ``--batch_size`` and ``--batch-size`` would normally map to
    different destinations.  This function rewrites ``_`` to ``-`` in the leading ``--key`` portion so that argparse
    treats both spellings identically.
    """
    out: list[str] = []
    for arg in argv:
        if arg.startswith("--") and "=" in arg:
            key, _, val = arg.partition("=")
            arg = key.replace("_", "-") + "=" + val
        elif arg.startswith("--"):
            arg = arg.replace("_", "-")
        out.append(arg)
    return out


def _variant_commands(
    script_path: Path,
    input_path: Path,
    env_dir: str,
    stem: str,
    extra_args: list[str],
    output_suffixes: tuple[str, ...] = ("",),
) -> list[tuple[str, list[str], list[Path]]]:
    """Build the (label, command, outputs) triple of one conversion script for both prompt variants.

    Args:
        script_path: Conversion script to run once per prompt variant.
        input_path: Positional input file forwarded to the script.
        env_dir: Environment folder under ``data/{mistral,common}/`` receiving the outputs.
        stem: Output filename stem; the script receives ``f"{stem}_{variant}.jsonl"``.
        extra_args: Fixed flags appended after the output path.
        output_suffixes: Suffixes of the files the script actually writes; the declared outputs are
            ``f"{stem}_{variant}{suffix}.jsonl"``.

    Returns:
        Two (label, command, outputs) triples, mistral first, then common.
    """
    steps: list[tuple[str, list[str], list[Path]]] = []
    for variant, is_mistral in VARIANTS:
        mistral_flag = ["--is-mistral-model"] if is_mistral else []
        outputs = [DATA_DIR / variant / env_dir / f"{stem}_{variant}{suffix}.jsonl" for suffix in output_suffixes]
        steps.append(
            (
                f"{script_path.name} ({variant})",
                [
                    sys.executable,
                    str(script_path),
                    str(input_path),
                    "--output",
                    str(DATA_DIR / variant / env_dir / f"{stem}_{variant}.jsonl"),
                    *extra_args,
                    *mistral_flag,
                ],
                outputs,
            )
        )
    return steps


def _conversion_commands() -> list[tuple[str, list[str], list[Path]]]:
    """Build every dataset-conversion command in run order: math, code, then instruction-following.

    The variant-independent merge/dedup steps (``merge_coder1_deepcoder.py`` and ``process_tulu_if.py``) each run once
    and feed both prompt variants of their downstream converter.

    Returns:
        (label, command, outputs) triples in run order.
    """
    deepscaler_input = HF_DATA_DIR / "agentica-org" / "DeepScaleR-Preview-Dataset" / "deepscaler.json"
    hendrycks_matches = sorted((HF_DATA_DIR / "qwedsacf" / "competition_math").rglob("train-00000-of-00001-*.parquet"))
    hendrycks_input = (
        hendrycks_matches[0]
        if hendrycks_matches
        else HF_DATA_DIR / "qwedsacf" / "competition_math" / "data" / "train-00000-of-00001.parquet"
    )
    code_merged = HF_DATA_DIR / "code_merged.jsonl"
    tulu_if_dedup = HF_DATA_DIR / "tulu_if_dedup.jsonl"
    genrm_extra_args = ["--prompt-field", "prompt", "--ground-truth-field", "reference", "--dataset", TULU_DATASET]

    commands: list[tuple[str, list[str], list[Path]]] = []

    # Math
    # DeepScaleR-Preview-Dataset from huggingface has 6 entries with empty answer fields, so we skip those.
    commands += _variant_commands(
        SCRIPT_DIR / "deepscaler.py", deepscaler_input, "math_with_judge", "deepscaler", ["--skip-on-error"]
    )
    # qwedsacf/competition_math from huggingface has 2 entries with empty last \boxed{} in its 'solution' field, so we skip those
    commands += _variant_commands(
        SCRIPT_DIR / "math_hendrycks.py", hendrycks_input, "math_with_judge", "math_hendrycks", ["--skip-on-error"]
    )

    # Code
    commands.append(
        (
            "merge_coder1_deepcoder.py",
            [
                sys.executable,
                str(HF_PROCESSORS_DIR / "merge_coder1_deepcoder.py"),
                "--data-dir",
                str(HF_DATA_DIR),
                "--out",
                str(code_merged),
            ],
            [code_merged],
        )
    )
    commands += _variant_commands(
        SCRIPT_DIR / "convert_to_code_gen_simple_agent.py",
        code_merged,
        "code_gen",
        "coder1_deepcoder",
        [],
        ("-multiple_tests", "-single_test"),
    )

    # IF
    commands.append(
        (
            "process_tulu_if.py",
            [
                sys.executable,
                str(HF_PROCESSORS_DIR / "process_tulu_if.py"),
                "--data-dir",
                str(HF_DATA_DIR),
                "--out",
                str(tulu_if_dedup),
                # no dataset was skipped via the below deduplication, for the interest of speed, we comment out the below commands, you may verify for yourself that no samples were additionally removed.
                # "--near",
                # "0.9",
            ],
            [tulu_if_dedup],
        )
    )
    commands += _variant_commands(
        SCRIPT_DIR / "convert_to_llm_judge_with_ground_truth.py",
        tulu_if_dedup,
        "llm_judge_with_ground_truth",
        "tulu_if_processed",
        genrm_extra_args,
    )
    return commands


def _merge_commands(batch_size: int) -> list[tuple[str, list[str], list[Path]]]:
    """Build the combined-dataset merge command for each prompt variant, mistral first.

    Args:
        batch_size: Rows per proportionally mixed batch, forwarded as ``--batch-size``.

    Returns:
        Two (label, command, outputs) triples, mistral first, then common.
    """
    return [
        (
            f"merge_multi_environment_datasets.py ({variant})",
            [
                sys.executable,
                str(SCRIPT_DIR / "merge_multi_environment_datasets.py"),
                str(DATA_DIR / variant),
                "--batch-size",
                str(batch_size),
                "--output",
                str(DATA_DIR / variant / f"combined_data_{batch_size}.jsonl"),
            ],
            [DATA_DIR / variant / f"combined_data_{batch_size}.jsonl"],
        )
        for variant, _ in VARIANTS
    ]


def _confirm_continue(step_label: str) -> bool:
    """Ask whether the run should continue after *step_label* exited non-zero."""
    try:
        response = (
            input(f"{step_label} failed (see its output above). Continue with the remaining steps? Type [y]es/[n]o: ")
            .strip()
            .lower()
        )
    except EOFError:
        response = ""
    return response in ("y", "yes")


def main() -> None:
    sys.argv = _preprocess_underscore_args(sys.argv)

    parser = argparse.ArgumentParser(
        description=(
            "Run every dataset-processing script in order: convert the datasets under train_data/hf/data into "
            "train_data/nemo_gym/data/{mistral,common}/<environment>/ files, then merge each variant's environment "
            "datasets into combined_data_<batch_size>.jsonl."
        )
    )
    parser.add_argument(
        "--batch-size",
        "--bs",
        "--batch",
        dest="batch_size",
        type=int,
        required=True,
        help="Rows per proportionally mixed batch, forwarded to merge_multi_environment_datasets.py.",
    )
    parser.add_argument(
        "--yes",
        "--regenerate",
        "-y",
        dest="yes",
        action="store_true",
        default=False,
        help="Preemptively confirm every overwrite prompt in every called script (default: false).",
    )
    parser.add_argument(
        "--skip-processed",
        "--skip",
        "-s",
        dest="skip_processed",
        action="store_true",
        default=False,
        help="Skip any step whose output files all already exist instead of running it (default: false).",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error(f"Batch size must be a positive integer, got: {args.batch_size}")

    steps: list[tuple[str, list[str], list[Path], bool]] = [
        (label, command, outputs, False) for label, command, outputs in _conversion_commands()
    ] + [(label, command, outputs, True) for label, command, outputs in _merge_commands(args.batch_size)]
    if args.yes:
        steps = [(label, command + ["--yes"], outputs, fail_soft) for label, command, outputs, fail_soft in steps]

    failed: list[str] = []
    for index, (label, command, outputs, fail_soft) in enumerate(steps, start=1):
        logger.info(f"[{index}/{len(steps)}] {label}")
        if args.skip_processed and all(path.exists() for path in outputs):
            logger.info("skipped: all output files already exist (--skip-processed)\n")
            continue
        logger.info(f"command: {' '.join(command)}")
        result = subprocess.run(command)
        if result.returncode == 0:
            continue
        failed.append(label)
        if fail_soft:
            logger.error(f"{label} failed with exit code {result.returncode}; the other merge is still attempted.\n")
            continue
        if args.yes:
            logger.warning(f"{label} failed with exit code {result.returncode}; continuing (--yes).\n")
            continue
        if not _confirm_continue(label):
            logger.info("Aborting; outputs already written are kept.\n")
            sys.exit(1)
        logger.info("Continuing with the remaining steps.\n")

    if failed:
        logger.error(f"Failed steps: {', '.join(failed)}")
        sys.exit(1)
    logger.info("All steps finished successfully.")


if __name__ == "__main__":
    main()
