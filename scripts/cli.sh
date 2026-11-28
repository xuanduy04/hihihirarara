#!/bin/bash

# Usage: source cli.sh "$@" after setting SCRIPT_DIR and entering CONFIG_PATH, EXP_NUM and ADV_ESTIMATOR

display_help() {
    cat << EOF
Usage:
  $(basename "$0") [CONFIG] [--config CONFIG] [-e EXP_NUM] -a ADV_ESTIMATOR
Description:
  Launches GRPO + NeMo-Gym training. Values given here override the ones entered in the script.
Arguments:
  CONFIG          (Optional) Config file path, relative to the current directory. Only accepted as the first argument.
Options:
  --config        (Optional) Same as CONFIG. Setting both will result in undefined behaviour.
  -a, --adv, --advantage, --advantage-estimator
                  (Optional) Formal advantage estimator name: rloo, grpo, dr.grpo, reinforce++, krpo, jspo,
                  reinforce_with_moving_average_baseline, tan, tmn.
  -e, -r, --exp, --experiment, --resume
                  (Optional) Experiment number to resume. Non-integers (e.g. 'auto') start a new experiment.
  -h, --help      Show this help.
Example:
  bash $(basename "$0") qwen3-4b.yaml -a dr.grpo
  bash $(basename "$0") --config qwen3-4b.yaml -a reinforce++ --resume 3
EOF
    exit 1
}

if [[ $# -gt 0 && ! "$1" =~ ^- ]]; then
    CONFIG_PATH="$(realpath "$1")"
    shift
fi

while [[ $# -gt 0 ]]; do
    [[ $# -ge 2 ]] || display_help
    case $1 in
        --config) CONFIG_PATH="$(realpath "$2")"; shift 2 ;;
        -a|--a|--adv|--advantage|--advantage-estimator) ADV_ESTIMATOR="$2"; shift 2 ;;
        -e|-r|--exp|--experiment|--resume) EXP_NUM="$2"; shift 2 ;;
        *) display_help ;;
    esac
done
