from deepsklearn.models import MMOE
from deepsklearn.config import aliexpress_NL_config
from deepsklearn.datasets import TorchStreamingMTLDataset
from torch.utils.data import DataLoader
from deepsklearn.trainer import MultiTaskTrainer
from deepsklearn.utils import Logger,set_seed,prevent_sleep
import torch.nn as nn
'''

'''
logger=Logger.get_logger()
set_seed(42)
prevent_sleep() #prevent_sleep

############setu up config##########
train_data_path=aliexpress_NL_config.train_data_path
validation_data_path=aliexpress_NL_config.test_data_path
label_columns=aliexpress_NL_config.label_columns
categorical_feature_columns=aliexpress_NL_config.categorical_feature_columns
numerical_feature_columns=aliexpress_NL_config.numerical_feature_columns

train_batch_size=10000
validation_batch_size=100000
device='cpu'
epoch_number = 3
use_warm_up = True
warm_up_steps = 10
log_steps=20
validation_steps = 2000
use_early_stop = True
#define model
model_name = "mtl_mmoe"

model = MMOE(
             categorical_feature_columns=categorical_feature_columns,
             numerical_feature_columns=numerical_feature_columns,
             embed_dim=32)

def main(model:nn.Module):
    train_dataset = TorchStreamingMTLDataset(
        data_path=train_data_path,
        categorical_feature_columns=categorical_feature_columns,
        numerical_feature_columns=numerical_feature_columns,
        label_columns=label_columns,
        batch_size=train_batch_size)
    train_dataloader = DataLoader(train_dataset, batch_size=None)

    validation_dataset =TorchStreamingMTLDataset(
        data_path=validation_data_path,
        categorical_feature_columns=categorical_feature_columns,
        numerical_feature_columns=numerical_feature_columns,
        label_columns=label_columns,
        batch_size=validation_batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=None)
    trainer = MultiTaskTrainer(
        loss_fn=nn.CrossEntropyLoss(),
        model_name=model_name,
        model=model,
        device=device,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        epoch_number=epoch_number,
        use_warm_up=use_warm_up,
        warm_up_steps=warm_up_steps,
        validation_steps=validation_steps,
        log_steps=log_steps,
        use_early_stop=use_early_stop
    )
    trainer.train()

if __name__ == '__main__':
    main(model)









