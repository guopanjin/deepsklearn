import argparse
import json

from deepsklearn.config import criteo_config
from deepsklearn.features import FeaturePipeline
from deepsklearn.utils import Logger,set_seed,prevent_sleep
from deepsklearn.datasets import TorchStreamingDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from deepsklearn.trainer import DiscriminativeTrainer
from deepsklearn.models.models_registry import get_model
logger=Logger.get_logger()
set_seed(42)
import os

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=str,required=True)
    args = parser.parse_args()
    with open(args.config,"r",encoding="utf-8") as f:
        config=json.load(f) #return python dict
    return config


class TrainConfig:
    def __init__(self,**train_config):
        self.model_name=train_config["model_name"]
        self.train_data_path=train_config["train_data_path"]
        self.validation_data_path=train_config["validation_data_path"]
        self.train_batch_size = int(train_config.get("train_batch_size",1000))
        self.validation_batch_size = int(train_config.get("validation_batch_size", 1000))
        self.device=train_config.get("device", "cpu")
        with open(os.path.expanduser(train_config["feature_config"]),"r",encoding="utf-8") as f:
            self.feature_config=json.load(f)
        with open(os.path.expanduser(train_config["label_config"]),"r",encoding="utf-8") as f:
            self.label_config=json.load(f)["label"]

class TrainModel:
    def __init__(self,train_config:TrainConfig):
        self.feature_config=train_config.feature_config
        self.model_name=train_config.model_name
        self.train_data_path = train_config.train_data_path
        self.validation_data_path = train_config.validation_data_path
        self.train_batch_size = train_config.train_batch_size
        self.validation_batch_size = train_config.validation_batch_size
        self.feature_config=train_config.feature_config
        self.label_config=train_config.label_config
        self.device=train_config.device

    def train(self):
        feature_columns = FeaturePipeline(self.feature_config).get_feature_columns()
        config={}
        config["model_name"]=self.model_name
        config["args"]={"feature_columns":feature_columns,
                                        "norm":False,
                                        "dropout":0,
                                        "customize_init_embedding":True
                                        }
        model:nn.Module=get_model(config)
        train_dataset = TorchStreamingDataset(
            data_path=self.train_data_path,
            feature_configs=self.feature_config,
            label_configs=self.label_config,
            batch_size=self.train_batch_size
        )
        train_dataLoader = DataLoader(
            train_dataset,
            batch_size=None
        )
        validation_dataset = TorchStreamingDataset(data_path=self.validation_data_path,
                                                   feature_configs=self.feature_config,
                                                   label_configs=self.label_config,
                                                   batch_size=self.validation_batch_size
                                                   )
        validation_dataLoader = DataLoader(
            validation_dataset,
            batch_size=None
        )
        trainer = DiscriminativeTrainer(
            model_name=self.model_name,
            model=model,
            train_dataloader=train_dataLoader,
            validation_dataloader=validation_dataLoader,
            use_warm_up=True,
            warm_up_steps=100,
            device=self.device
        )
        trainer.train()




if __name__ == '__main__':
    train_config=parse_args()
    logger.info(f"train_config:{train_config}")
    train_model_config=TrainConfig(**train_config)
    logger.info(f"train_model_config:{train_model_config.__dict__}")
    TrainModel(train_model_config).train()

