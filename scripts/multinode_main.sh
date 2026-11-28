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


IS_MULTINODE_TRAINING=True
source "${SCRIPT_DIR}/cli.sh" "$@"
source "${SCRIPT_DIR}/core.sh"

# =============================== RAY =============================== #

# Ray variables
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0  # this is to silence random warnings while waiting for all ray nodes to connect
export RAY_BACKTRACE_LOGGING_LEVEL=debug  # setting logging level to debug to avoid silent treatment by ray
# # forcing Ray to be more proactive with memory management
# export RAY_object_spilling_threshold=0.67  # default 0.8 (multiply by 200 to get GB size)
# export RAY_min_spilling_size=$((67 * 1024 * 1024))  # default 100 * 1024 * 1024
# export RAY_max_spilling_file_size_bytes=$((67 * 1024 * 1024))
# export RAY_max_io_workers=16  # default 4
# export RAY_max_fused_object_count=670  # default 2000 - don't wait for 2000 objects before spilling
# preventing premature kills by ray
export RAY_health_check_period_ms=1200000
export RAY_health_check_timeout_ms=600000
export RAY_health_check_failure_threshold=1000
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.967
# respecting the system's auto port assignment
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-6767}"
export RAY_ADDRESS="$MASTER_ADDR:$MASTER_PORT"

# Connect nodes
if (( RANK == 0 )); then
    ray stop --force || true
    ray start --head \
        --include-dashboard=True \
        --node-ip-address="${MASTER_ADDR}" \
        --port="${MASTER_PORT}" \
        --num-gpus="${N_GPUS_PER_NODE}" \
        --system-config='{
            "gcs_rpc_server_connect_timeout_s": 4,
            "py_gcs_connect_timeout_s": 4,
            "nums_py_gcs_reconnect_retry": 3
        }'
else
    if ! wait_until_success ray health-check --address="${RAY_ADDRESS}" >/dev/null 2>&1; then
        printf "Timeout: ray head did not connect within 30 minutes."
        exit 0
    fi

    ray start \
        --address="${RAY_ADDRESS}" \
        --num-gpus "${N_GPUS_PER_NODE}"
fi

ray_all_nodes_connected() {
    python - <<'PY'
import logging, os, pprint, sys, time, ray
try:
    expected_nodes = int(os.environ["N_NODES"])
    expected_gpus_per_node = int(os.environ["N_GPUS_PER_NODE"])
    ray.init(address=os.environ["RAY_ADDRESS"], log_to_driver=False, logging_level=logging.CRITICAL)

    nodes = ray.nodes()
    alive_nodes = [n for n in nodes if n.get("Alive", False)]
    n_nodes = len(alive_nodes)

    # Check GPU counts from node resources advertised in the node table
    unexpected_gpu_nodes = 0
    for node in alive_nodes:
        resources = node.get("Resources", {})
        gpu_count = resources.get("GPU", 0)
        if gpu_count != expected_gpus_per_node:
            unexpected_gpu_nodes += 1

    if unexpected_gpu_nodes > 0:
        print(f"[ERROR] Found {unexpected_gpu_nodes} node(s) with GPU count != expected ({expected_gpus_per_node}), please check:")
        pprint.pprint(alive_nodes)
        sys.exit(67)

    if n_nodes > expected_nodes:
        print("="*67, f"[IMPORTANT] THERE ARE MORE NODES THAN EXPECTED ({n_nodes} > {expected_nodes})", "="*67, sep="\n\n")

    if n_nodes >= expected_nodes:
        print(f"All nodes connected. (Found {n_nodes}/{expected_nodes} alive nodes in Ray cluster)")
        pprint.pprint([node.get("Resources", {}) for node in alive_nodes])
        sys.exit(0)
    else:
        print(f"Waiting for all nodes to connect... (Nodes connected: {n_nodes}/{expected_nodes}) [Current time: {time.strftime('%H:%M:%S', time.localtime())}]")
        sys.exit(1)

except Exception as e:
    print(f"Exception raised: {e}")
    sys.exit(1)
PY
}
if ! wait_until_success ray_all_nodes_connected; then
    printf "Timeout: nodes did not connect within 30 minutes."
    exit 0
fi

# ============================== main ============================== #
make_exp_dirs
printf '=%.0s' {1..67}
printf '\n'
printf "Run log is at: '${RUN_LOG_DIR}/rank${RANK}.txt'\n"
printf '=%.0s' {1..67}
printf '\n'

START_TS=$(date +%s)
{
    if (( RANK != 0 )); then
        # Non-head nodes keep track of usage statistics
        while ray health-check --address="${RAY_ADDRESS}"; do
            printf "Ray liveness check (elapsed: $(time_since "$START_TS"))\n"
            python "${NEMO_RL_UTILS_DIR}/statistic_of_node.py" || true
            sleep 30
        done
        printf "Ray cluster is gone (elapsed: $(time_since "$START_TS")). Exiting...\n"
    else
        # The head node runs the training script
        main_python_call
    fi
} 2>&1 | tee "${RUN_LOG_DIR}/rank${RANK}.txt"

printf "Training ended.\nElapsed time: $(time_since "$START_TS")\n"

printf "\nWaiting 100 seconds for other nodes to properly complete...\n"
sleep 100
printf "Finished waiting. Exiting...\n"
