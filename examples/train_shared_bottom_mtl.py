from deepsklearn.models import SharedBottom
from deepsklearn.config import aliexpress_NL_config
from deepsklearn.datasets import TorchStreamingMTLDataset
from torch.utils.data import DataLoader
from deepsklearn.trainer import MultiTaskTrainer
from deepsklearn.utils import Logger,set_seed,prevent_sleep
import torch.nn as nn
'''
2026-07-15 20:27:44 | INFO | multitask_trainer.py:111 | {'model': 'mtl_shared_bottom', 'duration': '11.772min', 'stage': 'training', 'epoch': 2, 'step_size': 10000, 'step_loss': 0.08980295062065125, 'ema_loss': 0.09370763581752216, 'global_size': 35595788, 'global_step': 3560, 'ctr_loss': 0.0858, 'ctcvr_loss': 0.004, 'step_ctr_auc': 0.7935235303135408, 'step_ctcvr_auc': 0.815847923961981}
2026-07-15 20:27:47 | INFO | multitask_trainer.py:111 | {'model': 'mtl_shared_bottom', 'duration': '11.823min', 'stage': 'training', 'epoch': 2, 'step_size': 10000, 'step_loss': 0.08510968089103699, 'ema_loss': 0.09291639114189, 'global_size': 35795788, 'global_step': 3580, 'ctr_loss': 0.0821, 'ctcvr_loss': 0.003, 'step_ctr_auc': 0.8001424432641236, 'step_ctcvr_auc': 0.9630852340936374}
2026-07-15 20:27:50 | INFO | multitask_trainer.py:111 | {'model': 'mtl_shared_bottom', 'duration': '11.872min', 'stage': 'training', 'epoch': 2, 'step_size': 10000, 'step_loss': 0.08740410953760147, 'ema_loss': 0.09018106338196227, 'global_size': 35995788, 'global_step': 3600, 'ctr_loss': 0.0817, 'ctcvr_loss': 0.0057, 'step_ctr_auc': 0.8020747333423796, 'step_ctcvr_auc': 0.9032663630904724}
2026-07-15 20:27:53 | INFO | multitask_trainer.py:111 | {'model': 'mtl_shared_bottom', 'duration': '11.923min', 'stage': 'training', 'epoch': 2, 'step_size': 10000, 'step_loss': 0.07821179181337357, 'ema_loss': 0.09197493447295278, 'global_size': 36195788, 'global_step': 3620, 'ctr_loss': 0.0753, 'ctcvr_loss': 0.0029, 'step_ctr_auc': 0.7857633238405208, 'step_ctcvr_auc': 0.9533766883441721}
2026-07-15 20:27:56 | INFO | multitask_trainer.py:111 | {'model': 'mtl_shared_bottom', 'duration': '11.974min', 'stage': 'training', 'epoch': 2, 'step_size': 10000, 'step_loss': 0.09307943284511566, 'ema_loss': 0.09084827027742773, 'global_size': 36395788, 'global_step': 3640, 'ctr_loss': 0.0873, 'ctcvr_loss': 0.0057, 'step_ctr_auc': 0.7799531782865446, 'step_ctcvr_auc': 0.8223829063250601}
2026-07-15 20:28:58 | INFO | multitask_trainer.py:192 | {'stage': 'validation', 'model_name': 'mtl_shared_bottom', 'epoch': 2, 'validation_number': 5559301, 'validation_loss': 0.115, 'normal_loss': 0.0, 'ctr_auc': 0.7152432016342786, 'ctcvr_auc': 0.8329570542617007}
2026-07-15 20:28:58 | INFO | multitask_trainer.py:148 | restore the best model weight to the current model

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
model_name = "mtl_shared_bottom"

model = SharedBottom(
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









