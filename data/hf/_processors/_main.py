"""Download the HuggingFace datasets used in the experiments into train_data/hf/data.

Each repo below is fetched into train_data/hf/data/<org>/<name>/, mirroring the Hub layout, e.g.
train_data/hf/data/qwedsacf/competition_math/; the other scripts in _processors look there first
(--data-dir) before downloading from the Hub themselves.

By default datasets are fetched with huggingface_hub.snapshot_download. Pass --hfd PATH to a
custom hfd.sh instead; PATH must end in .sh and is invoked per dataset as:

    bash PATH <repo_id> --dataset --local-dir <dest>

Usage:
    python _main.py
    python _main.py --hfd /path/to/hfd.sh
"""

import argparse
import os
import subprocess

DATASETS = (
    # Math    
    "qwedsacf/competition_math",
    "agentica-org/DeepScaleR-Preview-Dataset",
    # Code
    "ganler/code-r1-12k",
    "agentica-org/DeepCoder-Preview-Dataset",
    # IF
    "allenai/tulu-3-sft-personas-instruction-following",
    "google/IFEval",  # for deduplication
    "allenai/IFBench_test",  # for deduplication
)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def main():
    ap = argparse.ArgumentParser(description="Download the experiment datasets into train_data/hf/data.")
    ap.add_argument(
        "--hfd-path",
        "--custom-hfd-path",
        "--hfd",
        dest="hfd_path",
        default=None,
        help="custom hfd.sh script used to download instead of the normal way (must end in .sh)",
    )
    args = ap.parse_args()
    if args.hfd_path is not None and not args.hfd_path.endswith(".sh"):
        ap.error(f"--hfd must point to a .sh file, got: {args.hfd_path}")
    for repo in DATASETS:
        dest = os.path.join(DATA_DIR, repo)
        print(f"downloading {repo} -> {dest}")
        if args.hfd_path is not None:
            subprocess.run(["bash", args.hfd_path, repo, "--dataset", "--local-dir", dest], check=True)
        else:
            from huggingface_hub import snapshot_download

            snapshot_download(repo, repo_type="dataset", local_dir=dest)
        print(f"done: {repo}")
    
    print("\n\nAll datasets downloaded")


if __name__ == "__main__":
    main()
