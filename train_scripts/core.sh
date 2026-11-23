#!/bin/bash

# Usage: source core.sh after
#       setting  SCRIPT_DIR, ROOT_DIR
#       entering CONFIG_PATH, EXP_NUM and IS_MULTINODE_TRAINING

# ============================ Custom OPT ============================ #
NEMO_RL_BASE_DIR="${ROOT_DIR}/train_codebase"
NEMO_RL_UTILS_DIR="${NEMO_RL_BASE_DIR}/custom_utils/utils_training"

source "${NEMO_RL_UTILS_DIR}/use_custom_opt.sh" "${NEMO_RL_BASE_DIR}"
bash "${NEMO_RL_UTILS_DIR}/copy_gym_venvs.sh" "${CONFIG_PATH}"

# ======================== Auto set constants ======================== #
# Read config general helper
read_config() {
    python "${NEMO_RL_UTILS_DIR}/read_yaml.py" "${CONFIG_PATH}" "$1"
}

# >>> 1. Check for invalid configurations >>> #
N_GEN="$(read_config "cluster.num_generation")"
N_TRAIN="$(read_config "cluster.num_training")"

# N_NODES and N_GPUS_PER_NODE
case "$IS_MULTINODE_TRAINING" in
    True)  N_NODES=$(( N_GEN + N_TRAIN )); N_GPUS_PER_NODE=8;                      GEN_NODES=$N_GEN; GEN_GPUS_PER_NODE=8 ;;
    False) N_NODES=1;                      N_GPUS_PER_NODE=$(( N_GEN + N_TRAIN )); GEN_NODES=1;      GEN_GPUS_PER_NODE=$N_GEN ;;
    *)     printf "[ERROR] IS_MULTINODE_TRAINING must be True or False, got '${IS_MULTINODE_TRAINING}'\n"; exit 0 ;;
esac
export N_NODES N_GPUS_PER_NODE

WORLD_SIZE="${WORLD_SIZE:-$(nvidia-smi -L | grep -c ^GPU)}"
if (( N_NODES * N_GPUS_PER_NODE > WORLD_SIZE )); then
# Validate that the requested GPUs do not exceed the available WORLD_SIZE (auto set by torchrun in multi-node case)
    printf "[ERROR] Invalid configuration: N_NODES ($N_NODES) * N_GPUS_PER_NODE ($N_GPUS_PER_NODE) exceeds WORLD_SIZE ($WORLD_SIZE).\n"
    exit 0
fi

# check if reasoning parser file exists
# HUMAN_TODO: check if any of our models need this feature
# model_reasoning_parser="$(python "${NEMO_RL_UTILS_DIR}/read_yaml.py" "${CONFIG_PATH}" "policy.generation.vllm_cfg.http_server_serving_chat_kwargs.reasoning_parser")"
# if [ "$model_reasoning_parser" = "super_v3" ]; then
#     if [ ! -f "${NEMO_RL_BASE_DIR}/nemo_rl/utils/super_v3_reasoning_parser.py" ]; then
#         printf "[ERROR] missing 'super_v3' reasoning parser plugin from NeMo-RL codebase. It should be at:\n"
#         printf "\t${NEMO_RL_BASE_DIR}/nemo_rl/utils/super_v3_reasoning_parser.py\n"
#         exit 0
#     fi
# fi


# >>> 2. Create and assign variables based on config >>> #

# OUTPUT directory & RUN directory

# *Helpers
get_last_experiment_dir() {
    find "$1" -mindepth 1 -maxdepth 1 -type d -name "${2}*" 2>/dev/null | sort -V | tail -n 1
}

auto_detect_exp_num() {
    # Print the next experiment number (zero-padded to 3 digits)
    # for directories in $1 whose names start with the prefix $2.
    local exp_parent_dir="$1"
    local pattern="$2"
    local last num

    # find the last experiment directory
    last=$(get_last_experiment_dir "$exp_parent_dir" "$pattern")

    last="${last##*/}"            # keep only the directory name
    num="${last#"$pattern"}"      # strip the prefix
    num="${num%%[!0-9]*}"         # keep only the leading digits

    # add 1 to get the current experiment number
    printf "%03d\n" $((10#${num:-0} + 1))
}

# *Main
OUTPUT_DIR="${ROOT_DIR}/train_outputs"
MODEL_NAME="$(basename "$(read_config "policy.model_name")")"

if [[ "$EXP_NUM" =~ ^[0-9]+$ ]]; then
    EXP_NUM=$(printf "%03d" $((10#$EXP_NUM)))
else
    EXP_NUM=$(auto_detect_exp_num "$OUTPUT_DIR" "${MODEL_NAME}-exp")
fi
EXP_DIR="${OUTPUT_DIR}/${MODEL_NAME}-exp${EXP_NUM}"

# LOG directories
LOG_DIR="${EXP_DIR}/logs"
RUN_LOG_DIR="${LOG_DIR}/run_$(auto_detect_exp_num "$LOG_DIR" "run_")"
# In the multi-node case, RUN_LOG_DIR will be created after the ray cluster has finished initialization

# cache
export NRL_MEGATRON_CHECKPOINT_DIR="${EXP_DIR}/cache/nemo_rl"
export VLLM_CACHE_ROOT="${EXP_DIR}/cache/vllm"
export HF_HOME="${EXP_DIR}/cache/huggingface"
export HF_DATASETS_CACHE="${EXP_DIR}/cache/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="${EXP_DIR}/cache/huggingface/models"

# checkpoint directory
CHECKPOINT_DIR="${EXP_DIR}/ckpt"

# Directories are made after all nodes are in-sync
make_exp_dirs() {
    mkdir -p "${LOG_DIR}/nemo_gym" "${RUN_LOG_DIR}" "${CHECKPOINT_DIR}" \
        "${NRL_MEGATRON_CHECKPOINT_DIR}" "${VLLM_CACHE_ROOT}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"
}

# other environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NRL_IGNORE_VERSION_MISMATCH=1
export NRL_VLLM_ASYNC_TIMEOUT_SECONDS=36000
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=36000

# =============================== Sub-script helpers =============================== #

wait_until_success() {
    local WAIT_TIMEOUT=1800
    local start_time=$SECONDS
    local exit_code

    while true; do
        "$@"
        exit_code=$?

        case $exit_code in
            0)  # return 0: done
                return 0
                ;;
            1)  # returns 1: as expected, wait until 30 mins
                if (( SECONDS - start_time >= WAIT_TIMEOUT )); then
                    return 1
                fi
                sleep 5
                ;;
            *)  # else: unexpected error
                return "$exit_code"
                ;;
        esac
    done
}

time_since() {
    local s=$(( $(date +%s) - $1 ))
    printf "%dh %02dm %02ds" $((s/3600)) $(((s%3600)/60)) $((s%60))
}

# =============================== main python call =============================== #

main_python_call() {
    local data_path="${ROOT_DIR}/train_data/nemo_gym/data"
    local prompt_batch_size="$(read_config "grpo.num_prompts_per_step")"
    case "${MODEL_NAME,,}" in
        *mistral*|*magistral*) data_path="${data_path}/mistral" ;;
        *)                     data_path="${data_path}/common" ;;
    esac
    data_path="${data_path}/combined_data_${prompt_batch_size}.jsonl"

    # Base commands
    training_cmd_args=(
        "${NEMO_RL_BASE_DIR}/examples/nemo_gym/run_grpo_nemo_gym.py"
        "--config" "${CONFIG_PATH}"
        "++checkpointing.checkpoint_dir=${CHECKPOINT_DIR}"
        "++env.nemo_gym.nemo_gym_log_dir=${LOG_DIR}/nemo_gym"
        "++logger.log_dir=${LOG_DIR}"
        # data
        "++data.train=[{data_path:${data_path}}]"
        # node assignment
        "++cluster.gpus_per_node=${N_GPUS_PER_NODE}"
        "++cluster.num_nodes=${N_NODES}"
        "++policy.generation.colocated.resources.gpus_per_node=${GEN_GPUS_PER_NODE}"
        "++policy.generation.colocated.resources.num_nodes=${GEN_NODES}"
    )

    # Conditional commands
    # Currently empty, template kept for future development
    # if [ "something" = "true" ]; then
    #     training_cmd_args+=("some command")
    # fi

    # exec
    python "${training_cmd_args[@]}"
}
