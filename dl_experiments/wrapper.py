from typing import Union, List, Any, Mapping

import torch
from torch.utils.data import DataLoader, TensorDataset

from dl_experiments.common import update_flat_dicts
from dl_experiments.config import MyGRUConfig, MyCNNConfig, MyCNNGRUConfig, BaseModelConfig, GeneralConfig
from dl_experiments.model import MyCNN, MyGRU, MyCNNGRU
from dl_experiments.model import MyBaseModel as BaseModel


class BaseWrapper(object):
    def __init__(self, model_class: Union[MyCNN, MyGRU, MyCNNGRU],
                 model_config: Union[MyCNNConfig, MyGRUConfig, MyCNNGRUConfig],
                 checkpoint: Mapping,
                 device: str = "cpu"):
        self.model_class: BaseModel = model_class
        self.model_config: BaseModelConfig = model_config
        self.checkpoint: Mapping = checkpoint
        self.device: str = device

        self.model_args, _, _ = update_flat_dicts(checkpoint["best_trial_config"],
                                                  [model_config.model_args,
                                                   model_config.optimizer_args,
                                                   model_config.loss_args])

        self.model_args = {**self.model_args, "device": device}

        self.instance = self.__get_shallow_model_instance__()

    def __get_shallow_model_instance__(self):
        model_state_dict = self.checkpoint.get("model_state_dict", None)

        model = self.model_class(**self.model_args).to(self.device).to(torch.double)
        if model_state_dict is not None:
            model.load_state_dict(model_state_dict)
        return model

    def predict(self, data: TensorDataset):
        self.instance.eval()
        # predict
        target_pred = []
        target_true = []
        with torch.no_grad():
            for features, targets in DataLoader(data, batch_size=GeneralConfig.batch_size, shuffle=False):
                features = features.to(self.device).to(torch.double)
                target_pred += [self.instance(features)]

                targets = targets.to(self.device).to(torch.double)
                target_true += [targets]

        if not (len(target_pred) and len(target_true)):
            return torch.tensor([], dtype=torch.double).to(self.device), \
                   torch.tensor([], dtype=torch.double).to(self.device)
        else:
            return torch.cat(target_pred, 0), torch.cat(target_true, 0)
