from __future__ import annotations

import argparse
import ast
import collections
import json
import logging
import math
import re
import sys
from pathlib import Path

from tqdm.auto import tqdm

from system_prompts import get_system_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AGENT_NAME = "code_gen_simple_agent"

MULTIPLE_TESTS_SUFFIX = "-multiple_tests"
SINGLE_TEST_SUFFIX = "-single_test"

# Final line of a TACO call-based test: `assert _deep_eq(fn(*i), o[0])`.
TACO_ASSERT_PATTERN = re.compile(r"assert _deep_eq\((\w+)\(\*i\), o\[0\]\)\s*$")
# Final line of a LeetCode test: `check(Solution().fn)`.
LEETCODE_ENTRY_PATTERN = re.compile(r"check\(Solution\(\)\.(\w+)\)\s*$")
# Fenced python blocks of the user prompt; the last one is the starter code.
PYTHON_BLOCK_PATTERN = re.compile(r"```python\n(.*?)```", re.DOTALL)


class UnconvertibleEntryError(Exception):
    """Raised when an entry cannot be expressed in the code_gen unit test schema.

    Attributes:
        reason: Short, stable description used to aggregate skipped entries.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


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


def _validate_unit_tests(unit_tests: dict) -> None:
    """Validate a converted ``verifier_metadata.unit_tests`` dict."""
    for key in ("inputs", "outputs"):
        values = unit_tests[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"Converted 'unit_tests.{key}' is invalid: expected a non-empty list")
        if not all(isinstance(value, str) for value in values):
            raise ValueError(f"Converted 'unit_tests.{key}' is invalid: all values must be strings")

    if len(unit_tests["inputs"]) != len(unit_tests["outputs"]):
        raise ValueError("Converted 'unit_tests' is invalid: 'inputs' and 'outputs' differ in length")

    fn_name = unit_tests["fn_name"]
    if fn_name is not None and not isinstance(fn_name, str):
        raise ValueError("Converted 'unit_tests.fn_name' is invalid: expected a string or null")


def _strict_equal(left: object, right: object) -> bool:
    """Compare two values recursively, requiring identical types at every level."""
    if type(left) is not type(right):
        return False
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_strict_equal(x, y) for x, y in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_strict_equal(left[key], right[key]) for key in left)
    if isinstance(left, float) and isinstance(right, float) and math.isnan(left):
        return math.isnan(right)
    return left == right


def _to_json(value: object, what: str) -> str:
    """Serialize *value* to JSON, requiring that it round-trips with identical types.

    The code_gen call-based grader rebuilds arguments and expected outputs with ``json.loads``, so tuples, sets, and
    non-string dict keys would silently change meaning.

    Args:
        value: Python literal taken from the dataset's test code.
        what: Either ``"input"`` or ``"output"``, used in the skip reason.

    Returns:
        The single-line JSON encoding of *value*.

    Raises:
        UnconvertibleEntryError: If *value* does not survive a JSON round trip.
    """
    try:
        encoded = json.dumps(value)
        survived = _strict_equal(json.loads(encoded), value)
    except (TypeError, ValueError):
        survived = False
    if not survived:
        raise UnconvertibleEntryError(f"test {what} does not round-trip through JSON")
    return encoded


def _has_nested_float(value: object, depth: int = 0) -> bool:
    """Return whether *value* holds a float inside a container."""
    if isinstance(value, float):
        return depth > 0
    if isinstance(value, (list, tuple)):
        return any(_has_nested_float(item, depth + 1) for item in value)
    if isinstance(value, dict):
        return any(_has_nested_float(item, depth + 1) for item in value.values())
    return False


def _build_call_based_tests(calls: list[tuple[list, object]], fn_name: str) -> dict:
    """Encode ``(args, expected)`` pairs the way the code_gen call-based grader reads them.

    Args:
        calls: One ``(positional_args, expected_output)`` pair per test.
        fn_name: Name of the function or ``Solution`` method under test.

    Returns:
        A ``unit_tests`` dict with ``inputs``, ``outputs``, and ``fn_name``.

    Raises:
        UnconvertibleEntryError: If any test cannot be graded faithfully.
    """
    if not calls:
        raise UnconvertibleEntryError("no test cases")

    inputs: list[str] = []
    outputs: list[str] = []
    for args, expected in calls:
        if not args:
            # The grader runs json.loads on every input line, and "" is not valid JSON.
            raise UnconvertibleEntryError("zero-argument call")
        if _has_nested_float(expected):
            # The grader only applies a float tolerance to scalar outputs.
            raise UnconvertibleEntryError("nested float in expected output")
        inputs.append("\n".join(_to_json(arg, "input") for arg in args))
        outputs.append(_to_json(expected, "output"))

    return {"inputs": inputs, "outputs": outputs, "fn_name": fn_name}


def _convert_stdio(tests: dict) -> dict:
    """Convert stdin/stdout tests, which already match the code_gen schema."""
    if not all(isinstance(value, str) for value in tests["inputs"] + tests["outputs"]):
        raise UnconvertibleEntryError("stdin_stdout test is not a string")
    if not all(value.strip() for value in tests["outputs"]):
        raise UnconvertibleEntryError("empty expected output")
    return {"inputs": list(tests["inputs"]), "outputs": list(tests["outputs"]), "fn_name": None}


def _convert_function_call(tests: dict, fn_name: str) -> dict:
    """Convert DeepCoder call-based tests: each input is an argument list, each output a one-element list."""
    calls: list[tuple[list, object]] = []
    for args, expected in zip(tests["inputs"], tests["outputs"], strict=True):
        if not isinstance(args, list) or not isinstance(expected, list) or len(expected) != 1:
            raise UnconvertibleEntryError("unrecognized function_call tests")
        calls.append((args, expected[0]))
    return _build_call_based_tests(calls, fn_name)


def _literal_assignment(node: ast.stmt, name: str) -> list:
    """Evaluate the list literal assigned to *name* by the statement *node*."""
    if (
        not isinstance(node, ast.Assign)
        or len(node.targets) != 1
        or not isinstance(node.targets[0], ast.Name)
        or node.targets[0].id != name
    ):
        raise UnconvertibleEntryError("unrecognized TACO test code")
    try:
        value = ast.literal_eval(node.value)
    except (ValueError, SyntaxError) as exc:
        raise UnconvertibleEntryError("non-literal TACO test values") from exc
    if not isinstance(value, list):
        raise UnconvertibleEntryError("unrecognized TACO test code")
    return value


def _convert_taco_functional(code: str) -> dict:
    """Convert a TACO call-based test (``_inputs`` / ``_outputs`` / ``_deep_eq`` loop).

    Args:
        code: The ``functional`` test code of the entry. It is parsed, never executed.

    Returns:
        A ``unit_tests`` dict with ``inputs``, ``outputs``, and ``fn_name``.

    Raises:
        UnconvertibleEntryError: If the test code does not have the expected shape.
    """
    match = TACO_ASSERT_PATTERN.search(code)
    if match is None:
        raise UnconvertibleEntryError("unrecognized TACO test code")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise UnconvertibleEntryError("unparseable test code") from exc
    if len(tree.body) < 2:
        raise UnconvertibleEntryError("unrecognized TACO test code")

    all_args = _literal_assignment(tree.body[0], "_inputs")
    all_expected = _literal_assignment(tree.body[1], "_outputs")
    if len(all_args) != len(all_expected):
        raise UnconvertibleEntryError("unrecognized TACO test code")

    calls: list[tuple[list, object]] = []
    for args, expected in zip(all_args, all_expected):
        if not isinstance(args, list) or not isinstance(expected, list) or len(expected) != 1:
            raise UnconvertibleEntryError("unrecognized TACO test code")
        calls.append((args, expected[0]))

    return _build_call_based_tests(calls, match.group(1))


def _starter_code_params(user_prompt: str, fn_name: str) -> list[str]:
    """Return the parameter names (without ``self``) of ``Solution.fn_name`` in the starter code."""
    blocks = PYTHON_BLOCK_PATTERN.findall(user_prompt)
    if not blocks:
        raise UnconvertibleEntryError("LeetCode starter code not found")
    try:
        tree = ast.parse(blocks[-1].rstrip() + "\n        pass\n")
    except SyntaxError as exc:
        raise UnconvertibleEntryError("LeetCode starter code not parseable") from exc

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Solution":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == fn_name:
                    return [arg.arg for arg in item.args.args][1:]
    raise UnconvertibleEntryError("LeetCode starter code not found")


def _convert_leetcode_functional(code: str, user_prompt: str) -> dict:
    """Convert a LeetCode test (``def check(candidate)`` followed by ``check(Solution().fn)``).

    Only tests made solely of ``assert candidate(<literals>) == <literal>`` are representable; tests relying on helpers
    such as ``tree_node`` are not.

    Args:
        code: The ``functional`` test code of the entry. It is parsed, never executed.
        user_prompt: The user message, whose starter code gives the argument order.

    Returns:
        A ``unit_tests`` dict with ``inputs``, ``outputs``, and ``fn_name``.

    Raises:
        UnconvertibleEntryError: If the test code does not have the expected shape.
    """
    match = LEETCODE_ENTRY_PATTERN.search(code)
    if match is None:
        raise UnconvertibleEntryError("unrecognized LeetCode test code")
    fn_name = match.group(1)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise UnconvertibleEntryError("unparseable test code") from exc
    if len(tree.body) != 2 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != "check":
        raise UnconvertibleEntryError("unrecognized LeetCode test code")

    params: list[str] | None = None
    calls: list[tuple[list, object]] = []
    for statement in tree.body[0].body:
        test = statement.test if isinstance(statement, ast.Assert) else None
        if (
            not isinstance(test, ast.Compare)
            or len(test.ops) != 1
            or not isinstance(test.ops[0], ast.Eq)
            or not isinstance(test.left, ast.Call)
            or not isinstance(test.left.func, ast.Name)
            or test.left.func.id != "candidate"
        ):
            raise UnconvertibleEntryError("LeetCode test needs helper functions")

        call = test.left
        try:
            args = [ast.literal_eval(arg) for arg in call.args]
            kwargs = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
            expected = ast.literal_eval(test.comparators[0])
        except (ValueError, SyntaxError) as exc:
            raise UnconvertibleEntryError("LeetCode test needs helper functions") from exc

        if kwargs:
            # The grader calls the method positionally, so order keywords by the signature.
            if params is None:
                params = _starter_code_params(user_prompt, fn_name)
            remaining = params[len(args) :]
            if set(kwargs) != set(remaining):
                raise UnconvertibleEntryError("LeetCode test arguments do not match starter code")
            args.extend(kwargs[name] for name in remaining)

        calls.append((args, expected))

    return _build_call_based_tests(calls, fn_name)


def _convert_entry(entry: dict, agent_name: str, system_prompt: str) -> dict:
    """Convert one row of the merged Code-R1 / DeepCoder JSONL to the code_gen JSONL schema.

    Args:
        entry: One row of the merged file.
        agent_name: Value for ``agent_ref.name``.
        system_prompt: System message placed before the user turn; empty omits it.

    Returns:
        The converted row.

    Raises:
        UnconvertibleEntryError: If the row's tests cannot be expressed for code_gen.
    """
    user_prompt = entry["problem"]
    if not isinstance(user_prompt, str) or not user_prompt:
        raise ValueError("Entry has missing or invalid 'problem' field")
    turns = [{"role": "user", "content": user_prompt}]
    if system_prompt:
        turns.insert(0, {"role": "system", "content": system_prompt})

    tests = entry["tests"]
    test_type = entry["test_type"]
    if test_type == "stdin_stdout":
        unit_tests = _convert_stdio(tests)
    elif test_type == "function_call":
        unit_tests = _convert_function_call(tests, entry["fn_name"])
    elif test_type == "assert_code":
        code = tests["code"]
        if code.startswith("_inputs = "):
            unit_tests = _convert_taco_functional(code)
        else:
            unit_tests = _convert_leetcode_functional(code, user_prompt)
    else:
        raise UnconvertibleEntryError("unrecognized test type")

    _validate_unit_tests(unit_tests)

    return {
        "agent_ref": {"type": "responses_api_agents", "name": agent_name},
        "responses_create_params": {"input": turns},
        "verifier_metadata": {"unit_tests": unit_tests},
        "dataset": entry["source"],
    }


def main() -> None:
    sys.argv = _preprocess_underscore_args(sys.argv)

    parser = argparse.ArgumentParser(
        description=(
            "Convert the merged Code-R1 / DeepCoder JSONL (output of ../../hf_data/_processors/merge_coder1_deepcoder.py) "
            "to NeMo-Gym's code_gen format, split by number of unit tests."
        )
    )
    parser.add_argument("input", type=Path, help="Path to the input data file (must be a .jsonl).")
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Base path of the output JSONL files (must end with .jsonl). Two files are written next to it: "
            f"'<stem>{MULTIPLE_TESTS_SUFFIX}.jsonl' (entries with 2+ unit tests) and "
            f"'<stem>{SINGLE_TEST_SUFFIX}.jsonl' (entries with exactly 1 unit test). "
            "Defaults to '../data/code_gen/coder1_deepcoder{_mistral,_common}.jsonl' relative to this script's directory."
        ),
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        help="Preemptively confirm overwrite of the output files (default: false).",
    )
    parser.add_argument("--agent-name", default=AGENT_NAME, help="Value for 'agent_ref.name' (default: %(default)s).")
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
        / "code_gen"
        / f"coder1_deepcoder{'_mistral' if args.is_mistral_model else '_common'}.jsonl"
    )
    agent_name: str = args.agent_name.strip()
    skip_on_error: bool = args.skip_on_error
    system_prompt: str = get_system_prompt(args.is_mistral_model)

    if not agent_name:
        parser.error("--agent-name must not be empty")

    input_path = input_path.resolve()
    output_path = output_path.resolve()

    if not input_path.is_file():
        parser.error(f"Input file does not exist: {input_path}")
    if input_path.suffix.lower() != ".jsonl":
        parser.error(f"Input file must be a .jsonl, got: {input_path}")
    if output_path.suffix.lower() != ".jsonl":
        parser.error(f"Output path must end with .jsonl, got: {output_path}")

    multiple_tests_path = output_path.with_name(f"{output_path.stem}{MULTIPLE_TESTS_SUFFIX}.jsonl")
    single_test_path = output_path.with_name(f"{output_path.stem}{SINGLE_TEST_SUFFIX}.jsonl")

    for path in (multiple_tests_path, single_test_path):
        try:
            _validate_output_path(path)
        except FileExistsError:
            if not args.yes:
                response = input(f"{path} already exists. Override? Type [y]es/[n]o: ").strip().lower()
                if response not in ("y", "yes"):
                    logger.info("Existing output file will not be overridden; exiting.")
                    sys.exit(0)

    logger.info(f"Converting {input_path} -> {multiple_tests_path} and {single_test_path}")

    with input_path.open(encoding="utf-8") as fin:
        entries = [json.loads(line) for line in fin if line.strip()]

    written_multiple = 0
    written_single = 0
    skipped = 0
    unconvertible: collections.Counter[str] = collections.Counter()
    with (
        multiple_tests_path.open("w", encoding="utf-8") as fout_multiple,
        single_test_path.open("w", encoding="utf-8") as fout_single,
    ):
        for index, entry in enumerate(tqdm(entries, desc="Converting", unit="row")):
            try:
                converted_entry = _convert_entry(entry, agent_name, system_prompt)
            except UnconvertibleEntryError as exc:
                unconvertible[exc.reason] += 1
                continue
            except Exception as exc:
                if skip_on_error:
                    logger.warning(f"Skipping row {index}: {exc}")
                    skipped += 1
                    continue
                raise

            if len(converted_entry["verifier_metadata"]["unit_tests"]["inputs"]) > 1:
                fout = fout_multiple
                written_multiple += 1
            else:
                fout = fout_single
                written_single += 1
            json.dump(converted_entry, fout, ensure_ascii=False)
            fout.write("\n")

    for path, written in ((multiple_tests_path, written_multiple), (single_test_path, written_single)):
        if written == 0:
            path.unlink(missing_ok=True)
            logger.info(f"No entries for {path}; file removed.")

    if written_multiple + written_single == 0:
        raise ValueError("No valid entries found - nothing to write.")

    if unconvertible:
        logger.info(f"Dropped {sum(unconvertible.values())} entries that code_gen cannot grade:")
        for reason, count in unconvertible.most_common():
            logger.info(f"  {count:>6}  {reason}")

    if skipped:
        logger.info(f"Skipped {skipped} invalid entries.")

    logger.info(f"Done - wrote {written_multiple} entries to {multiple_tests_path}.")
    logger.info(f"Done - wrote {written_single} entries to {single_test_path}.")


if __name__ == "__main__":
    main()
