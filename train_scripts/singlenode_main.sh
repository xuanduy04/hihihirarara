#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../" && pwd)"

# ------------------------------------------------------------------ #
# ENTER CONFIG NAME HERE
CONFIG_PATH="${SCRIPT_DIR}/configs/"
# ENTER ADVANTAGE ESTIMATOR HERE
ADV_ESTIMATOR=""
# ENTER EXPERIMENT NUMBER HERE (for resuming experiments)
EXP_NUM="auto"  # any number that is not an integer will use auto detection methods instead


IS_MULTINODE_TRAINING=False
source "${SCRIPT_DIR}/cli.sh" "$@"
source "${SCRIPT_DIR}/core.sh"

# ============================== main ============================== #
RANK=0
make_exp_dirs
printf '=%.0s' {1..67}
printf '\n'
printf "Run log is at: '${RUN_LOG_DIR}/rank${RANK}.txt'\n"
printf '=%.0s' {1..67}
printf '\n'

START_TS=$(date +%s)

main_python_call 2>&1 | tee "${RUN_LOG_DIR}/rank${RANK}.txt"

printf "Training ended.\nElapsed time: $(time_since "$START_TS")\n"
printf "Exiting...\n"
