import sys
import re


FORMALLY_NAMED_ADVANTAGE_ESTIMATORS: dict[str, dict[str, str | bool]] = {
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
    "reinforce++": {
        "name": "reinforce++",
        "normalize_rewards": True,
        "minus_baseline": True,
        "per_environment": False
    },
    "reinforce++perenv": {
        "name": "reinforce++",
        "normalize_rewards": True,
        "minus_baseline": True,
        "per_environment": True
    },
    "jspo": {
        "name": "jspo",
        "per_environment": False
    }
}  # fmt: skip


def get_advantage_estimator_config(formal_advantage_estimator_name: str):
    normalized_name = re.sub(r"[\s._-]+", "", formal_advantage_estimator_name.lower()).replace("environment", "env")
    if normalized_name in FORMALLY_NAMED_ADVANTAGE_ESTIMATORS:
        return normalized_name, FORMALLY_NAMED_ADVANTAGE_ESTIMATORS[normalized_name]
    else:
        raise ValueError(
            f"Invalid advantage_estimator name: {formal_advantage_estimator_name} ({normalized_name=})\n"
            f"Supported estimators are: {sorted(FORMALLY_NAMED_ADVANTAGE_ESTIMATORS.keys())}"
        )


if __name__ == "__main__":
    name, config = get_advantage_estimator_config(sys.argv[1])
    print(name, "++grpo.advantage_estimator={" + ",".join(f"{k}:{v}" for k, v in config.items()) + "}")
