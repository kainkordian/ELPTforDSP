import argparse
import os
import sys
from typing import Mapping
import torch
import pandas as pd

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")
sys.path.append(parent_dir)

from dl_experiments.common import update_flat_dicts, create_tensor_dataset, create_dataset, create_dirs
from dl_experiments.optimization import HyperOptimizer
from dl_experiments.wrapper import BaseWrapper
from dl_experiments.config import MyGRUConfig, MyCNNConfig, MyCNNGRUConfig, BaseModelConfig, GeneralConfig
from dl_experiments.model import MyGRU, MyCNN, MyCNNGRU

parser = argparse.ArgumentParser()

parser.add_argument("-dp", "--dataset-path", type=str,
                    required=True, help="Path to dataset file.")
parser.add_argument("-dn", "--dataset-name", type=str,
                    required=True, help="Name of the dataset.")
parser.add_argument("-dtc", "--dataset-target-column", type=str,
                    required=True, help="Dataset target column.")
parser.add_argument("-dd", "--dataset-delimiter", type=str,
                    default=",", help="Dataset delimiter.")

parser.add_argument("-m", "--model", required=True, type=str, choices=["GRU", "CNN", "CNNGRU"],
                    help="Model class to use.")

parser.add_argument("-d", "--device", type=str, required=True,
                    help="If available, will use this device for training.")

parser.add_argument("-ts", "--test-split", type=float, default=0.2,
                    help="Fraction of window used for testing.")

parser.add_argument("-vs", "--val-split", type=float, default=0.25,
                    help="Fraction of window used for validation.")

args = parser.parse_args()

my_class = None
my_config = None
if args.model == "GRU":
    my_class = MyGRU
    my_config = MyGRUConfig
elif args.model == "CNN":
    my_class = MyCNN
    my_config = MyCNNConfig
elif args.model == "CNNGRU":
    my_class = MyCNNGRU
    my_config = MyCNNGRUConfig

job_identifier: str = f"{args.dataset_name}_{args.model}_{args.device}"

# read data
base_df = pd.read_csv(args.dataset_path, delimiter=args.dataset_delimiter, index_col=0, parse_dates=True)
orig_values = base_df[args.dataset_target_column].values.reshape(-1, 1)

# extract sub-arrays
train = orig_values[:int(len(base_df) * (1 - args.test_split))]
test = orig_values[int(len(base_df) * (1 - args.test_split)):]
train_val = train[:int(len(train) * (1 - args.val_split))]

# perform optimization
optimizer_instance: HyperOptimizer = HyperOptimizer(my_config, job_identifier, args.device)
checkpoint: Mapping = optimizer_instance.perform_optimization(optimizer_instance, train, train_val)
# save checkpoint
create_dirs(GeneralConfig.result_dir)
torch.save(checkpoint, os.path.join(GeneralConfig.result_dir, f"{job_identifier}_checkpoint.pt"))

# update specs with best config
wrapper = BaseWrapper(my_class, my_config, checkpoint, device=args.device)

# create test dataset tensor
test_data = create_tensor_dataset(*create_dataset(test,
                                                  seq_length=wrapper.model_args["input_dim"],
                                                  target_length=wrapper.model_args["output_dim"],
                                                  device=args.device))
# predict / test
pred_values, true_values = wrapper.predict(test_data)
print("pred_values", pred_values)
print("true_values", true_values)
# TODO: do what you need to do with the predictions
