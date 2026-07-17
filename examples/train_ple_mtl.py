from deepsklearn.models import PLE
from deepsklearn.config import aliexpress_NL_config
from deepsklearn.datasets import TorchStreamingMTLDataset
from torch.utils.data import DataLoader
from deepsklearn.trainer import MultiTaskTrainer
from deepsklearn.utils import Logger,set_seed,prevent_sleep
import torch.nn as nn
'''
2026-07-15 21:12:21 | INFO | multitask_trainer.py:111 | {'model': 'mtl_ple', 'duration': '23.605min', 'stage': 'training', 'epoch': 2, 'step_size': 10000, 'step_loss': 0.0778871700167656, 'ema_loss': 0.09157844019098203, 'global_size': 36195788, 'global_step': 3620, 'ctr_loss': 0.0749, 'ctcvr_loss': 0.003, 'step_ctr_auc': 0.7938316091285985, 'step_ctcvr_auc': 0.9339069534767384}
2026-07-15 21:12:27 | INFO | multitask_trainer.py:111 | {'model': 'mtl_ple', 'duration': '23.705min', 'stage': 'training', 'epoch': 2, 'step_size': 10000, 'step_loss': 0.09287388622760773, 'ema_loss': 0.09033817456837082, 'global_size': 36395788, 'global_step': 3640, 'ctr_loss': 0.0871, 'ctcvr_loss': 0.0057, 'step_ctr_auc': 0.7827009457255004, 'step_ctcvr_auc': 0.8404348478783026}
2026-07-15 21:13:45 | INFO | multitask_trainer.py:192 | {'stage': 'validation', 'model_name': 'mtl_ple', 'epoch': 2, 'validation_number': 5559301, 'validation_loss': 0.1145, 'normal_loss': 0.0, 'ctr_auc': 0.7213745728791888, 'ctcvr_auc': 0.8541920391110636}
2026-07-15 21:13:45 | INFO | multitask_trainer.py:148 | restore the best model weight to the current model

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
model_name = "mtl_ple"

model = PLE(
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









