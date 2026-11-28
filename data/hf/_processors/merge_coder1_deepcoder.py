"""Merge and deduplicate Code-R1 and DeepCoder (`primeintellect` and `taco`) datasets.

Expected sources: Code-R1 (ganler/code-r1-12k), DeepCoder (agentica-org/DeepCoder-Preview-Dataset).

Deduplication compares the bare statement: every instruction prefix, trailing boilerplate and starter-code block is
stripped before hashing. The output keeps the ORIGINAL text. When the same problem is in both datasets, the DeepCoder
copy is kept (see PRIORITY).

Output row:
  problem       original prompt text, untouched (prefix, boilerplate and starter code included)
  statement     stripped statement that was used as the deduplication key
  starter_code  starter code that came with the task ("" if none)
  fn_name       function / method under test ("" for stdin-stdout tasks)
  test_type     "stdin_stdout" | "function_call" | "assert_code"
  tests         stdin_stdout / function_call: {"inputs": [...], "outputs": [...]}
                assert_code: {"code": "<python asserts>"}   (Code-R1 functional tests)
  source        "code-r1/taco" | "code-r1/leetcode" | "deepcoder/taco" | "deepcoder/primeintellect"
  source_index  row index in the source split

Usage:
  python merge_coder1_deepcoder.py --out code_merged.jsonl
  python merge_coder1_deepcoder.py --data-dir /path/to/data   # read local copies first
Inputs are first looked up under --data-dir (default: ../data next to this script, i.e. train_data/hf/data) as
<repo>/<repo-relative path>, e.g. train_data/hf/data/ganler/code-r1-12k/data/train-00000-of-00001.parquet; only files
missing there are downloaded from the HuggingFace Hub. Needs: pyarrow, pandas, huggingface_hub, tqdm. Streams 64 rows at a
time (peak RSS about 2 GB on the full data).
"""

import argparse, fnmatch, glob, hashlib, json, os, re, sys
from collections import Counter

import pyarrow.parquet as pq
from tqdm.auto import tqdm

CR1_REPO = "ganler/code-r1-12k"
DC_REPO = "agentica-org/DeepCoder-Preview-Dataset"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
# kept-first order when the same problem appears in several sources: DeepCoder wins over Code-R1
PRIORITY = ["deepcoder/taco", "deepcoder/primeintellect", "code-r1/leetcode", "code-r1/taco"]

# ---------------------------------------------------------------- wrappers
PREFIXES = [
    r"Solve the programming task below in a Python markdown code block\.",
    r"Please solve the programming task below using a self-contained code snippet in a markdown code block\.",
    r"Solve the following coding problem using the programming language \w+:",
]
SUFFIXES = [  # applied repeatedly at the end of the text
    r"Now solve the problem (?:and return the code|by providing the code)\.",
    r"The input will be stdin and you should print your solution to stdout\.?",
    r"The input will be given via stdin and the output should be printed to stdout by your code\.",
    r"Your solution should (?:be )?implemented in the function \"[^\"\n]*\"\. The inputs will be passed to it and it should return the correct solution\.",
]
STARTER = [  # "<lead-in>\n```python\n<code>```" at the end of the text
    r"Also feel free to reuse/extend the following starter code:",
    r"Please complete the following python code precisely:",
    r"Write your solution by modifying this code:",
]
PREFIX_RE = re.compile(r"^\s*(?:%s)\s*" % "|".join(PREFIXES))
SUFFIX_RE = re.compile(r"\s*(?:%s)\s*$" % "|".join(SUFFIXES))
STARTER_RE = re.compile(r"\s*(?:%s)\s*```[a-zA-Z0-9]*\n(.*?)\n?[ \t]*```\s*$" % "|".join(STARTER), re.S)
FN_HINT_RE = re.compile(r"implemented in the function \"([^\"\n]*)\"")
# statement with no problem description: only a sample ("Example\n\nInput\n...", optionally after a
# "Time Limit: ... / Memory Limit: ..." header), or LeetCode's placeholder for problems without English text
NO_DESCRIPTION_RE = re.compile(
    r"\s*(?:(?:Time|Memory) Limit:[^\n]*\n\s*)*Examples?\s*\n\s*Input\b"
    r"|\s*English description is not available for the problem\."
)


def strip_wrappers(text):
    """-> (statement, starter_code, fn_name_hint). Used for the dedup key only."""
    m = FN_HINT_RE.search(text[-400:])
    fn_hint = m.group(1) if m else ""
    starter = ""
    while True:
        new = PREFIX_RE.sub("", text, count=1)
        new = SUFFIX_RE.sub("", new, count=1)
        m = STARTER_RE.search(new)
        if m:
            starter, new = m.group(1).rstrip(), new[: m.start()]
        if new == text:
            break
        text = new
    return text.strip(), starter, fn_hint


def dedup_key(statement):
    norm = re.sub(r"[^a-z0-9]+", "", statement.lower())
    return hashlib.md5(norm.encode()).hexdigest()


def fn_from_starter(starter):
    names = re.findall(r"^\s*def\s+(\w+)\s*\(", starter, re.M)
    return names[-1] if names else ""


# ---------------------------------------------------------------- loaders
def match_files(root, pattern):
    files = sorted(glob.glob(os.path.join(root, "**", os.path.basename(pattern)), recursive=True))
    return [f for f in files if fnmatch.fnmatch(os.path.relpath(f, root), pattern)]


def parquet_files(repo, pattern, data_dir):
    root = os.path.join(data_dir, repo)
    files = match_files(root, pattern) if os.path.isdir(root) else []
    if not files:
        from huggingface_hub import snapshot_download

        root = snapshot_download(repo, repo_type="dataset", allow_patterns=[pattern])
        files = match_files(root, pattern)
    assert files, f"no parquet files for {repo}:{pattern}"
    return files


def stream(files, columns):
    """Row tuples, a small batch at a time (test payloads are large)."""
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=64, columns=columns):
            yield from batch.to_pandas().itertuples(index=False)


def total_rows(files):
    return sum(pq.ParquetFile(f).metadata.num_rows for f in files)


def rows_code_r1(files):
    for i, r in enumerate(stream(files, ["prompt", "reward_model", "extra_info"])):
        original = r.prompt[-1]["content"]
        statement, starter, _ = strip_wrappers(original)
        gt = json.loads(r.reward_model["ground_truth"])
        leet = r.extra_info["dataset"] == "LeetCodeDataset"
        if "functional" in gt:
            code = gt["functional"]
            m = re.search(r"check\(Solution\(\)\.(\w+)\)", code) if leet else None
            fn = m.group(1) if m else fn_from_starter(starter)
            ttype, tests = "assert_code", {"code": code}
        else:
            fn, ttype, tests = "", "stdin_stdout", {"inputs": list(gt["inputs"]), "outputs": list(gt["outputs"])}
        yield dict(
            problem=original,
            statement=statement,
            starter_code=starter,
            fn_name=fn,
            test_type=ttype,
            tests=tests,
            source="code-r1/leetcode" if leet else "code-r1/taco",
            source_index=i,
        )


def rows_deepcoder(files, source):
    for i, r in enumerate(stream(files, ["problem", "tests"])):
        original = r.problem
        statement, starter, fn_hint = strip_wrappers(original)
        t = json.loads(r.tests) if isinstance(r.tests, str) else r.tests
        if isinstance(t, dict):  # taco: {"inputs","outputs"[,"fn_name"]}
            fn, ins, outs = t.get("fn_name") or "", t["inputs"], t["outputs"]
        else:  # primeintellect: [{"type","input","output"[,"fn_name"]}]
            fn = next((x["fn_name"] for x in t if x.get("fn_name")), "")
            ins, outs = [x["input"] for x in t], [x["output"] for x in t]
        fn = fn or fn_hint
        yield dict(
            problem=original,
            statement=statement,
            starter_code=starter,
            fn_name=fn,
            test_type="function_call" if fn else "stdin_stdout",
            tests={"inputs": list(ins), "outputs": list(outs)},
            source=source,
            source_index=i,
        )


def jsonable(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(type(o))


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(DATA_DIR, "code_merged.jsonl"))
    parser.add_argument(
        "--data-dir", default=DATA_DIR, help="checked first for <repo>/<repo-relative path> before the Hub"
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        help="Preemptively confirm overwrite of the output file (default: false).",
    )
    a = parser.parse_args()

    if os.path.exists(a.out) and not a.yes:
        response = input(f"{a.out} already exists. Override? Type [y]es/[n]o: ").strip().lower()
        if response not in ("y", "yes"):
            print("Existing output file will not be overridden; exiting.")
            sys.exit(0)

    # sources are streamed in PRIORITY order, so the first occurrence of a problem is the one kept
    cr1 = parquet_files(CR1_REPO, "**/train-*.parquet", a.data_dir)
    dc_taco = parquet_files(DC_REPO, "taco/train-*.parquet", a.data_dir)
    dc_pi = parquet_files(DC_REPO, "primeintellect/train-*.parquet", a.data_dir)
    streams = {
        "deepcoder/taco": rows_deepcoder(dc_taco, "deepcoder/taco"),
        "deepcoder/primeintellect": rows_deepcoder(dc_pi, "deepcoder/primeintellect"),
        "code-r1/leetcode": rows_code_r1(cr1),  # Code-R1 is read twice, once per subset
        "code-r1/taco": rows_code_r1(cr1),
    }
    totals = {
        "deepcoder/taco": total_rows(dc_taco),
        "deepcoder/primeintellect": total_rows(dc_pi),
        "code-r1/leetcode": total_rows(cr1),
        "code-r1/taco": total_rows(cr1),
    }
    seen = {}
    n_in, n_out, self_dup, cross_dup, empty, types = (Counter() for _ in range(6))
    with open(a.out, "w", encoding="utf-8") as f:
        for want in PRIORITY:
            for r in tqdm(streams[want], desc=want, total=totals[want], unit="rows"):
                s = r["source"]
                if s != want:
                    continue
                n_in[s] += 1
                if not re.search(r"[a-zA-Z0-9]", r["statement"]) or NO_DESCRIPTION_RE.match(r["statement"]):
                    empty[s] += 1
                    continue
                k = dedup_key(r["statement"])  # key = stripped statement
                if k in seen:
                    (self_dup if seen[k] == s else cross_dup)[s] += 1
                    continue
                seen[k] = s
                n_out[s] += 1
                types[r["test_type"]] += 1
                f.write(json.dumps(r, ensure_ascii=False, default=jsonable) + "\n")  # `problem` = original text

    print(f"{'source':26}{'in':>8}{'self-dup':>10}{'cross-dup':>11}{'empty':>7}{'out':>8}")
    for s in PRIORITY:
        print(f"{s:26}{n_in[s]:>8}{self_dup[s]:>10}{cross_dup[s]:>11}{empty[s]:>7}{n_out[s]:>8}")
    print(
        f"{'total':26}{sum(n_in.values()):>8}{sum(self_dup.values()):>10}{sum(cross_dup.values()):>11}"
        f"{sum(empty.values()):>7}{sum(n_out.values()):>8}"
    )
    print("test types:", dict(types))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
