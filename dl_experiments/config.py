import os
from typing import Any
from dl_experiments.model import MyCNN, MyGRU, MyCNNGRU
from dl_experiments.losses import *


class TuneConfig(object):

    scheduler: dict = {
        "grace_period": 100,
        "reduction_factor": 2
    }
    tune_best_trial: dict = {
        "scope": "all",
        "filter_nan_and_inf": True
    }
    stopping_criterion: dict = {
        "relation": "lt",
        "key": "validation_loss",
        "threshold": 0
    }
    tune_run: dict = {
        "local_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "ray_results"),  # local result folder
        "mode": "min",
        "metric": "validation_loss",
        "checkpoint_score_attr": "min-validation_loss",
        "keep_checkpoints_num": 3,
        "verbose": 1,
        "num_samples": 2,
        # can also be omitted
        "resources_per_trial": {
            "cpu": 1,  # how many cpu cores per trial?
            "gpu": 0  # needs to be "0" on cpu-only devices. You can also specify fractions
        }
    }
    concurrency_limiter: dict = {
        "max_concurrent": 4
    }

    optuna_search: dict = {}


class GeneralConfig(object):
    batch_size: int = 32
    epochs: int = 200
    result_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    best_checkpoint_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_checkpoints")


class BaseModelConfig(object):
    model_class: Any
    model_args: dict
    optimizer_class: Any
    optimizer_args: dict
    loss_class: Any
    loss_args: dict
    reporter: dict
    search_space_config: dict


class MyCNNConfig(BaseModelConfig):
    model_class = MyCNN
    model_args = {
        "input_dim": 100,
        "hidden_dim": 10,
        "output_dim": 1,
        "dropout": 0.0,
        "num_conv_kernels": 16,
        "conv_kernel_size": 9,
        "pool_kernel_size": 5,
    }
    optimizer_class = torch.optim.Adam
    optimizer_args = {
        "lr": 0.01,
        "weight_decay": 0.0001
    }
    loss_class = MSELoss
    loss_args = {}
    reporter = {
        "parameter_columns": ["lr", "weight_decay", "input_dim"]
    }
    search_space_config = {
        "lr": [0.1, 0.01, 0.001],
        "weight_decay": [0.01, 0.001],
        "input_dim": [50, 100]
    }


class MyGRUConfig(BaseModelConfig):
    model_class = MyGRU
    model_args = {
        "input_dim": 100,
        "hidden_dim": 16,
        "output_dim": 1,
        "dropout": 0.0,
        "num_layers": 1,
        "bidirectional": False
    }
    optimizer_class = torch.optim.Adam
    optimizer_args = {
        "lr": 0.01,
        "weight_decay": 0.0001
    }
    loss_class = MSELoss
    loss_args = {}
    reporter = {
        "parameter_columns": ["lr", "weight_decay", "input_dim"]
    }
    search_space_config = {
        "lr": [0.1, 0.01, 0.001],
        "weight_decay": [0.01, 0.001],
        "input_dim": [50, 100]
    }


class MyCNNGRUConfig(BaseModelConfig):
    model_class = MyCNNGRU
    model_args = {
        "input_dim": 100,
        "hidden_dim": 0,
        "output_dim": 1,
        "dropout": 0.0,
        "num_layers": 0,
        "num_conv_kernels": 0,
        "conv_kernel_size": 0,
        "pool_kernel_size": 0,
        "bidirectional": False
    }
    optimizer_class = torch.optim.Adam
    optimizer_args = {
        "lr": 0.01,
        "weight_decay": 0.0001
    }
    loss_class = MSELoss
    loss_args = {}
    reporter = {
        "parameter_columns": ["lr", "weight_decay", "input_dim"]
    }
    search_space_config = {
        "lr": [0.1, 0.01, 0.001],
        "weight_decay": [0.01, 0.001],
        "input_dim": [50, 100]
    }
