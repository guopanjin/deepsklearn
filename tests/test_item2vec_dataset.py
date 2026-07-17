from deepsklearn.config import amazon_beauty_config
from deepsklearn.datasets import TorchStreamingItem2vecDataset
from torch.utils.data import DataLoader

item2vec_train_data_path=amazon_beauty_config.item2vec_train_parquet_path
item2vec_validation_data_path=amazon_beauty_config.item2vec_validation_parquet_path
item2vec_sampler_weight_path=amazon_beauty_config.item2vec_sampler_weights_path
item2vec_feature_item=amazon_beauty_config.item2vec_feature_item
item2vec_label_item=amazon_beauty_config.item2vec_label_item
n_items=amazon_beauty_config.n_items


train_dataset=TorchStreamingItem2vecDataset(
             data_path=item2vec_train_data_path,
             sampler_weight_path=item2vec_sampler_weight_path,
             feature_item=item2vec_feature_item,
             label_item=item2vec_label_item,
             batch_size = 10,
             sample_k = 5
             )

train_dataloader=DataLoader(train_dataset,batch_size=None)









i=0
for features in train_dataset:
    i+=1
    if i>2:
        break;
    print("=====features====")
    print(features)
    for k,v in features.items():
        print(k,v.shape)

