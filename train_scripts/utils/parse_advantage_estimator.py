import sys
import re


FORMALLY_NAMED_ADVANTAGE_ESTIMATORS: dict[str, dict[str, str | bool]] = {
    # Prompt-level baseline
    "rloo": {
        "name": "reinforce++",
        "normalize_rewards": False,
        "minus_baseline": True,
        "use_leave_one_out_baseline": True,
        "per_environment": False
    },
    "grpo": {
        "name": "grpo",
        "minus_baseline": True,
        "normalize_rewards": True
    },
    "drgrpo": {
        "name": "grpo",
        "minus_baseline": True,
        "normalize_rewards": False
    },
    "reinforce++": {  # I do argue that there is little merit to adding `REINFORCE++ w/ baseline` compared to others, as the method's main "contribution" is batch-level norm, its baseline is exactly to GRPO's.
        "name": "reinforce++",
        "normalize_rewards": True,
        "minus_baseline": True,
        "per_environment": False
    },
    "krpo": {  # KRPO (https://arxiv.org/abs/2505.07527)
        "name": "krpo"
    },

    # Batch-level baselines
    "jspo": {  # JSPO (https://arxiv.org/abs/2511.03710)
        "name": "jspo",
        "per_environment": False
    },
    # History-using baselines
    "reinforcewithmovingaveragebaseline": {  # REINFORCE with moving average baseline (https://aclanthology.org/2024.acl-long.662)
        "name": "reinforce_with_moving_average_baseline"
    },
    # Multi-task baselines
    "tan": {  # Task Advantage Normalization (https://arxiv.org/abs/2510.04206)
        "name": "tan"
    },
    "tmn": {  # Task-level Mean Normalization (TMN) (https://arxiv.org/abs/2605.19577)
        "name": "tmn",
        "use_reweight": False
    },
}  # fmt: skip


ALTERNATE_FORMALLY_NAMED_ADVANTAGE_ESTIMATORS: dict[str, str] = {
    "kalmanfilter": "krpo",
    "stein": "jspo",
    "taskadvantagenormalization": "tan",
    "tasklevelmeannormalization": "tmn",
    "taskmeannormalization": "tmn",
}  # fmt: skip


def get_advantage_estimator_config(formal_advantage_estimator_name: str):
    normalized_name = re.sub(r"[\s._-]+", "", formal_advantage_estimator_name.lower()).replace("environment", "env")

    while normalized_name in ALTERNATE_FORMALLY_NAMED_ADVANTAGE_ESTIMATORS:
        normalized_name = ALTERNATE_FORMALLY_NAMED_ADVANTAGE_ESTIMATORS[normalized_name]

    if normalized_name in FORMALLY_NAMED_ADVANTAGE_ESTIMATORS:
        return normalized_name, FORMALLY_NAMED_ADVANTAGE_ESTIMATORS[normalized_name]
    else:
        raise ValueError(
            f"Invalid advantage_estimator name: {formal_advantage_estimator_name} ({normalized_name=})\n"
            f"Supported estimators are: {sorted(FORMALLY_NAMED_ADVANTAGE_ESTIMATORS.keys())}\n"
            f"    with alternate names: {sorted(ALTERNATE_FORMALLY_NAMED_ADVANTAGE_ESTIMATORS.keys())}"
        )


if __name__ == "__main__":
    name, config = get_advantage_estimator_config(sys.argv[1])
    print(name, "++grpo.advantage_estimator={" + ",".join(f"{k}:{v}" for k, v in config.items()) + "}")
