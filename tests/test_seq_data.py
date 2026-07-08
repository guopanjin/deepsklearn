import os

from deepsklearn.config import amazon_beauty_config
from deepsklearn.datasets.seq_dataset import TorchStreamingSeqDataset
from torch.utils.data import DataLoader

train_data_path=amazon_beauty_config.train_dataset
feature_columns=amazon_beauty_config.feature_columns
label_columns=amazon_beauty_config.label_columns



train_dataset=TorchStreamingSeqDataset(data_path=train_data_path,
                         feature_columns=feature_columns,
                         label_columns=label_columns,
                         batch_size=2
                         )
i=0
for features,labels in train_dataset:
    i+=1
    if(i>2):
        break
    print(features)
    print(labels)


print("=================dataloader===============")

train_loader=DataLoader(train_dataset,batch_size=None)


i=0
for features,labels in train_loader:
    i+=1
    if(i>2):
        break
    print(features)
    print(labels)
    for feature in feature_columns:
       print(feature,features[feature].shape,features[feature].dtype)
