from deepsklearn.trainer import RetrivealTrainer
from deepsklearn.utils import Logger, set_seed, prevent_sleep
from deepsklearn.config import amazon_beauty_config
from deepsklearn.datasets import TorchStreamingItem2vecDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from deepsklearn.models import Item2vec
'''
before initialize embedding table (std=0.01)
2026-07-13 11:11:05 | INFO | retriveval_trainer.py:108 | {'model': 'item2vec', 'duration': '0.481min', 'stage': 'training', 'epoch': 19, 'step_size': 6306, 'step_loss': 1.435644507408142, 'ema_loss': 1.4275869161234978, 'global_size': 11126120, 'global_step': 1120, 'num_classes': 12103, 'step_loss_ppl': 4.2024, 'ema_loss_ppl': 4.1686}
2026-07-13 11:11:05 | INFO | retriveval_trainer.py:169 | {'stage': 'validation', 'model_name': 'item2vec', 'epoch': 19, 'validation_number': 44726, 'validation_loss': 1.4776, 'normal_loss': 0.1572, 'validation_ppl': 4.3824, 'num_classes': 12103}
2026-07-13 11:11:05 | INFO | retriveval_trainer.py:169 | {'stage': 'validation', 'model_name': 'item2vec', 'epoch': 19, 'validation_number': 44726, 'validation_loss': 1.4852, 'normal_loss': 0.158, 'validation_ppl': 4.4158, 'num_classes': 12103}
2026-07-13 11:11:05 | INFO | retriveval_trainer.py:141 | restore the best model weight to the current model

after initialize embedding table (std=0.01)
2026-07-13 11:23:47 | INFO | retriveval_trainer.py:111 | {'model': 'item2vec', 'duration': '0.48min', 'stage': 'training', 'epoch': 19, 'step_size': 6306, 'step_loss': 0.3898337185382843, 'ema_loss': 0.3853980038921381, 'global_size': 11126120, 'global_step': 1120, 'num_classes': 12103, 'step_loss_ppl': 1.4767, 'ema_loss_ppl': 1.4702}
2026-07-13 11:23:48 | INFO | retriveval_trainer.py:172 | {'stage': 'validation', 'model_name': 'item2vec', 'epoch': 19, 'validation_number': 44726, 'validation_loss': 0.4217, 'normal_loss': 0.0449, 'validation_ppl': 1.5246, 'num_classes': 12103}
2026-07-13 11:23:48 | INFO | retriveval_trainer.py:172 | {'stage': 'validation', 'model_name': 'item2vec', 'epoch': 19, 'validation_number': 44726, 'validation_loss': 0.4219, 'normal_loss': 0.0449, 'validation_ppl': 1.5249, 'num_classes': 12103}
2026-07-13 11:23:48 | INFO | retriveval_trainer.py:144 | restore the best model weight to the current model

'''
logger = Logger.get_logger()
set_seed(42)
prevent_sleep()  # prevent_sleep
item2vec_train_data_path = amazon_beauty_config.item2vec_train_parquet_path
item2vec_validation_data_path = amazon_beauty_config.item2vec_validation_parquet_path
item2vec_sampler_weight_path = amazon_beauty_config.item2vec_sampler_weights_path
item2vec_feature_item = amazon_beauty_config.item2vec_feature_item
item2vec_label_item = amazon_beauty_config.item2vec_label_item
n_items = amazon_beauty_config.n_items

train_batch_size = 10000
validation_batch_size = 10000
sample_k = 5
device = 'cpu'
epoch_number = 20
use_warm_up = True
warm_up_steps = 10
validation_steps = 20
use_early_stop = True
# define model
model_name = "item2vec"
num_items = amazon_beauty_config.n_items
num_classes = num_items + 2
seq_len = amazon_beauty_config.generative_seq_len
model = Item2vec(
    num_items=num_items,
    embed_dim=32)


def main(model: nn.Module):
    train_dataset = TorchStreamingItem2vecDataset(
        data_path=item2vec_train_data_path,
        sampler_weight_path=item2vec_sampler_weight_path,
        feature_item=item2vec_feature_item,
        label_item=item2vec_label_item,
        batch_size=train_batch_size,
        sample_k=sample_k
    )
    train_dataloader = DataLoader(train_dataset, batch_size=None)

    validation_dataset = TorchStreamingItem2vecDataset(
        data_path=item2vec_validation_data_path,
        sampler_weight_path=item2vec_sampler_weight_path,
        feature_item=item2vec_feature_item,
        label_item=item2vec_label_item,
        batch_size=validation_batch_size,
        sample_k=sample_k
    )
    validation_dataloader = DataLoader(validation_dataset, batch_size=None)

    trainer = RetrivealTrainer(
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
