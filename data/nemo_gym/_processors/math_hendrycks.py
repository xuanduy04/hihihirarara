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
DEFAULT_DATASET = "MATH"

# The final answer of a MATH solution is the content of its last box; a handful of solutions use ``\fbox`` instead.
BOX_COMMANDS = ("\\boxed", "\\fbox")


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


def _extract_boxed_answer(solution: str) -> str:
    r"""Return the content of the last ``\\boxed{...}`` / ``\\fbox{...}`` in a MATH solution.

    Also accepts the brace-less ``\\boxed 5`` form, whose content runs up to the next ``$`` (or the end of the string).

    Args:
        solution: The worked solution text of one MATH row.

    Returns:
        The final answer, stripped of surrounding whitespace.

    Raises:
        ValueError: If the solution has no box, the box is unbalanced, or the box is empty.
    """
    idx, command = max((solution.rfind(command), command) for command in BOX_COMMANDS)
    if idx < 0:
        raise ValueError("Entry has no '\\boxed' or '\\fbox' in its 'solution' field")

    rest = solution[idx + len(command) :]
    if rest.startswith("{"):
        answer = None
        depth = 0
        for i, char in enumerate(rest):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    answer = rest[1:i]
                    break
        if answer is None:
            raise ValueError("Entry has unbalanced braces in the last box of its 'solution' field")
    elif rest.startswith(" "):
        answer = rest.split("$")[0]
    else:
        raise ValueError("Entry has a malformed last box in its 'solution' field")

    answer = answer.strip()
    if not answer:
        raise ValueError("Entry has an empty last box in its 'solution' field")
    return answer


def _convert_entry(entry: dict, system_prompt: str) -> dict:
    """Convert one raw MATH row to the math_with_judge JSONL schema."""
    for field in ("problem", "solution"):
        if field not in entry or not isinstance(entry[field], str) or not entry[field].strip():
            raise ValueError(f"Entry has empty or missing '{field}' field")

    question = entry["problem"].strip()

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    out: dict = {
        "agent_ref": {"name": AGENT_NAME},
        "responses_create_params": {"input": messages},
        "question": question,
        "expected_answer": _extract_boxed_answer(entry["solution"]),
        "dataset": DEFAULT_DATASET,
    }

    return out


def _iter_entries(input_path: Path) -> Iterator[dict]:
    """Yield raw MATH rows from a ``.jsonl`` file or from the ``.parquet`` file as it ships on the Hub."""
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
        description="Convert Hendrycks' MATH (qwedsacf/competition_math) dataset to math_with_judge format."
    )
    parser.add_argument("input", type=Path, help="Path to the input data file (a .jsonl or the Hub's .parquet).")
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Path to the output JSONL file (must end with .jsonl). "
            "Defaults to '../data/math_with_judge/math_hendrycks{_mistral,_common}.jsonl' relative to this script's directory."
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
        / f"math_hendrycks{'_mistral' if args.is_mistral_model else '_common'}.jsonl"
    )
    skip_on_error: bool = args.skip_on_error
    system_prompt: str = get_system_prompt(args.is_mistral_model)

    input_path = input_path.resolve()
    output_path = output_path.resolve()

    if not input_path.is_file():
        parser.error(f"Input file does not exist: {input_path}")
    if input_path.suffix.lower() not in (".jsonl", ".parquet"):
        parser.error(f"Input path must end with .jsonl or .parquet, got: {input_path}")
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
