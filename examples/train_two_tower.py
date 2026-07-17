from deepsklearn.config import amazon_beauty_config
from deepsklearn.models import  TwoTower
from deepsklearn.datasets import TorchStreamingRetrievalDataset
from torch.utils.data import DataLoader
from deepsklearn.trainer import RetrivealTrainer
from deepsklearn.utils import Logger,set_seed,prevent_sleep
import torch.nn as nn
'''
2026-07-13 19:13:00 | INFO | retriveval_trainer.py:171 | {'stage': 'validation', 'model_name': 'two tower', 'epoch': 5, 'validation_number': 22363, 'validation_loss': 7.6002, 'normal_loss': 0.8084, 'validation_ppl': 1998.5956, 'num_classes': 12102}
2026-07-13 19:13:00 | INFO | retriveval_trainer.py:107 | {'model': 'two tower', 'duration': '0.043min', 'stage': 'training', 'epoch': 6, 'step_size': 1000, 'step_loss': 6.264137268066406, 'ema_loss': 6.151403324776163, 'global_size': 136178, 'global_step': 140, 'num_classes': 12102, 'step_loss_ppl': 525.3881, 'ema_loss_ppl': 469.3756}
2026-07-13 19:13:00 | INFO | retriveval_trainer.py:171 | {'stage': 'validation', 'model_name': 'two tower', 'epoch': 6, 'validation_number': 22363, 'validation_loss': 7.4335, 'normal_loss': 0.7907, 'validation_ppl': 1691.7182, 'num_classes': 12102}
2026-07-13 19:13:00 | INFO | retriveval_trainer.py:203 | early stop,stop training,best_loss:6.702, bad_round:5, min_delta:0.0005
2026-07-13 19:13:00 | INFO | retriveval_trainer.py:131 | early stop trigged,epoch 6
2026-07-13 19:13:00 | INFO | retriveval_trainer.py:143 | restore the best model weight to the current model

'''
logger=Logger.get_logger()
set_seed(42)
prevent_sleep() #prevent_sleep

############setu up config##########
generative_train_data_path=amazon_beauty_config.generative_train_data
generative_validation_data_path=amazon_beauty_config.generative_validation_data
generative_feature_columns=amazon_beauty_config.generative_feature_columns
generative_label_column=amazon_beauty_config.generative_label_column
generative_sequence_columns=amazon_beauty_config.generative_sequence_columns
train_batch_size=1000
validation_batch_size=1000
device='cpu'
epoch_number = 100
use_warm_up = True
warm_up_steps = 10
validation_steps = 20
use_early_stop = True
#define model
model_name = "two_tower"
num_items = amazon_beauty_config.n_items
num_classes=num_items+1
seq_len = amazon_beauty_config.generative_seq_len
model = TwoTower(
    num_items=num_items,
    embed_dim=32)

def main(model:nn.Module):
    train_dataset = TorchStreamingRetrievalDataset(
        data_path=generative_train_data_path,
        feature_columns=generative_feature_columns,
        sequence_columns=generative_sequence_columns,
        label_column=generative_label_column,
        batch_size=train_batch_size
    )
    train_dataloader = DataLoader(train_dataset, batch_size=None)
    validation_dataset = TorchStreamingRetrievalDataset(
        data_path=generative_validation_data_path,
        feature_columns=generative_feature_columns,
        sequence_columns=generative_sequence_columns,
        label_column=generative_label_column,
        batch_size=validation_batch_size
    )
    validation_dataloader = DataLoader(validation_dataset, batch_size=None)
    trainer = RetrivealTrainer(
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
        use_early_stop=use_early_stop,
        num_classes=num_classes
    )
    trainer.train()

if __name__ == '__main__':
    main(model)









