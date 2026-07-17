from deepsklearn.models import ESMM
from deepsklearn.config import aliexpress_NL_config
from deepsklearn.datasets import TorchStreamingMTLDataset
from torch.utils.data import DataLoader
from deepsklearn.trainer import MultiTaskTrainer
from deepsklearn.utils import Logger,set_seed,prevent_sleep
import torch.nn as nn
'''
2026-07-15 20:45:16 | INFO | multitask_trainer.py:111 | {'model': 'mtl_esmm', 'duration': '12.537min', 'stage': 'training', 'epoch': 2, 'step_size': 10000, 'step_loss': 0.08718512952327728, 'ema_loss': 0.08997971433718824, 'global_size': 35995788, 'global_step': 3600, 'ctr_loss': 0.0817, 'ctcvr_loss': 0.0055, 'step_ctr_auc': 0.8012207968148767, 'step_ctcvr_auc': 0.9292746697357885}
2026-07-15 20:45:20 | INFO | multitask_trainer.py:111 | {'model': 'mtl_esmm', 'duration': '12.594min', 'stage': 'training', 'epoch': 2, 'step_size': 10000, 'step_loss': 0.0780835822224617, 'ema_loss': 0.0917933149044586, 'global_size': 36195788, 'global_step': 3620, 'ctr_loss': 0.0752, 'ctcvr_loss': 0.0029, 'step_ctr_auc': 0.7881583217094812, 'step_ctcvr_auc': 0.9529364682341172}
2026-07-15 20:45:23 | INFO | multitask_trainer.py:111 | {'model': 'mtl_esmm', 'duration': '12.649min', 'stage': 'training', 'epoch': 2, 'step_size': 10000, 'step_loss': 0.09304310381412506, 'ema_loss': 0.09066214628381705, 'global_size': 36395788, 'global_step': 3640, 'ctr_loss': 0.0873, 'ctcvr_loss': 0.0058, 'step_ctr_auc': 0.7809046409954513, 'step_ctcvr_auc': 0.8356435148118495}
2026-07-15 20:46:27 | INFO | multitask_trainer.py:192 | {'stage': 'validation', 'model_name': 'mtl_esmm', 'epoch': 2, 'validation_number': 5559301, 'validation_loss': 0.1148, 'normal_loss': 0.0, 'ctr_auc': 0.7165439945133315, 'ctcvr_auc': 0.8525021330965767}
2026-07-15 20:46:27 | INFO | multitask_trainer.py:148 | restore the best model weight to the current model

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
model_name = "mtl_esmm"

model = ESMM(
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









