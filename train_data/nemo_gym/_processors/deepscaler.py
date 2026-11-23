from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

from tqdm.auto import tqdm

from system_prompts import get_system_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AGENT_NAME = "math_with_judge_simple_agent"
DEFAULT_DATASET = "DeepScaleR-Preview-Dataset"


def _preprocess_underscore_args(argv: list[str]) -> list[str]:
    """Replace underscores with hyphens in ``--arg_name`` / ``--arg_name=val`` prefixes.

    argparse normalises ``-`` and ``_`` internally, so ``--skip-on-error`` and ``--skip_on_error`` would normally map to
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


def _validate_output_path(output_path: Path) -> None:
    """Ensure *output_path* ends with ``.jsonl``, parent dir exists, and file does not already exist."""
    if output_path.suffix.lower() != ".jsonl":
        raise ValueError(f"Output path must end with .jsonl, got: {output_path}")

    if output_path.exists():
        raise FileExistsError(f"Output file already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)


def _convert_entry(entry: dict, system_prompt: str) -> dict:
    """Convert one raw DeepScaleR row to the math_with_judge JSONL schema."""
    for field in ("problem", "answer"):
        if field not in entry:
            raise ValueError(f"Entry has missing '{field}' field. (entry's fields: {list(entry.keys())})")
        if not isinstance(entry[field], str):
            raise ValueError(f"Entry has invalid '{field}' field. ({entry[field]=})")
        if not entry[field].strip():
            raise ValueError(f"Entry has empty '{field}' field. ({entry[field]=})")

    question = entry["problem"].strip()

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    out: dict = {
        "agent_ref": {"name": AGENT_NAME},
        "responses_create_params": {"input": messages},
        "question": question,
        "expected_answer": entry["answer"].strip(),
        "dataset": DEFAULT_DATASET,
    }

    return out


def _iter_entries(input_path: Path) -> Iterator[dict]:
    """Yield raw DeepScaleR rows from the ``.json`` array as it ships on the Hub, a ``.jsonl``, or a ``.parquet`` file."""
    if input_path.suffix.lower() == ".json":
        with input_path.open("r", encoding="utf-8") as fin:
            yield from json.load(fin)
        return

    if input_path.suffix.lower() == ".parquet":
        import pandas as pd

        yield from pd.read_parquet(input_path).to_dict(orient="records")
        return

    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def main() -> None:
    sys.argv = _preprocess_underscore_args(sys.argv)

    parser = argparse.ArgumentParser(
        description="Convert DeepScaleR (agentica-org/DeepScaleR-Preview-Dataset) dataset to math_with_judge format."
    )
    parser.add_argument(
        "input", type=Path, help="Path to the input data file (the Hub's .json, a .jsonl, or a .parquet)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Path to the output JSONL file (must end with .jsonl). "
            "Defaults to '../data/math_with_judge/deepscaler{_mistral,_common}.jsonl' relative to this script's directory."
        ),
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        help="Preemptively confirm overwrite of the output file (default: false).",
    )
    parser.add_argument(
        "--skip-on-error",
        action="store_true",
        default=False,
        help="Skip invalid entries with a warning instead of aborting (default: false).",
    )
    parser.add_argument(
        "--is-mistral-model",
        action="store_true",
        default=False,
        help="Use the Mistral system prompt and a '_mistral' default output suffix instead of '_common' (default: false).",
    )

    args = parser.parse_args()

    input_path: Path = args.input
    output_path: Path = (
        args.output
        if args.output is not None
        else Path(__file__).resolve().parent.parent
        / "data"
        / "math_with_judge"
        / f"deepscaler{'_mistral' if args.is_mistral_model else '_common'}.jsonl"
    )
    skip_on_error: bool = args.skip_on_error
    system_prompt: str = get_system_prompt(args.is_mistral_model)

    input_path = input_path.resolve()
    output_path = output_path.resolve()

    if not input_path.is_file():
        parser.error(f"Input file does not exist: {input_path}")
    if input_path.suffix.lower() not in (".json", ".jsonl", ".parquet"):
        parser.error(f"Input path must end with .json, .jsonl or .parquet, got: {input_path}")
    if input_path == output_path:
        parser.error(f"Input and output paths must differ, got: {input_path}")

    try:
        _validate_output_path(output_path)
    except FileExistsError:
        if not args.yes:
            response = input(f"{output_path} already exists. Override? Type [y]es/[n]o: ").strip().lower()
            if response not in ("y", "yes"):
                logger.info("Existing output file will not be overridden; exiting.")
                sys.exit(0)

    logger.info(f"Converting {input_path} -> {output_path}")

    written = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for rowno, entry in enumerate(tqdm(_iter_entries(input_path), desc="Converting", unit="row"), start=1):
            try:
                converted_entry = _convert_entry(entry, system_prompt)
            except Exception as exc:
                if skip_on_error:
                    logger.warning(f"Skipping row {rowno}: {exc}")
                    skipped += 1
                    continue
                raise
            json.dump(converted_entry, fout, ensure_ascii=False)
            fout.write("\n")
            written += 1

    if written == 0:
        output_path.unlink(missing_ok=True)
        raise ValueError("No valid entries found - nothing to write.")

    if skipped:
        logger.info(f"Skipped {skipped} invalid entries.")

    logger.info(f"Done - wrote {written} entries to {output_path}")


if __name__ == "__main__":
    main()
