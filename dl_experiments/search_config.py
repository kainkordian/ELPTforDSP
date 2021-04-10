from ray import tune

all_search_configs: dict = {
    "GRU_5min": {
        "input_dim": tune.choice([48, 96, 192]),
        "hidden_dim": tune.choice([2, 4, 8, 16, 32, 64]),
        "output_dim": tune.choice([1]),
        "dropout": tune.choice([0.0, 0.5]),
        "num_layers": tune.choice([1, 2]),
        "bidirectional": tune.choice([False, True])
    },
    "GRU_15min": {
        "input_dim": tune.choice([24, 48, 96]),
        "hidden_dim": tune.choice([2, 4, 8, 16, 32, 64]),
        "output_dim": tune.choice([4]),
        "dropout": tune.choice([0.0, 0.5]),
        "num_layers": tune.choice([1, 2]),
        "bidirectional": tune.choice([False, True])
    },
    "GRU_1h": {
        "input_dim": tune.choice([12, 24, 48]),
        "hidden_dim": tune.choice([2, 4, 8, 16, 32, 64]),
        "output_dim": tune.choice([12]),
        "dropout": tune.choice([0.0, 0.5]),
        "num_layers": tune.choice([1, 2]),
        "bidirectional": tune.choice([False, True])
    }
}


def get_search_space_config(model_name: str, sampling_rate: str):
    base_config: dict = {
        "lr": tune.choice([0.1, 0.01, 0.001]),
        "weight_decay": tune.choice([0.01, 0.001, 0.0001]),
    }
    add_config = all_search_configs.get(f"{model_name}_{sampling_rate}", {})

    search_space_config: dict = {**base_config, **add_config}
    return search_space_config
