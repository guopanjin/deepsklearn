from deepsklearn.config import aliexpress_NL_config
from deepsklearn.datasets import TorchStreamingMTLDataset
from torch.utils.data import DataLoader

train_data_path=aliexpress_NL_config.train_data_path
validation_data_path=aliexpress_NL_config.test_data_path
label_columns=aliexpress_NL_config.label_columns

categorical_feature_columns=aliexpress_NL_config.categorical_feature_columns
numerical_feature_columns=aliexpress_NL_config.numerical_feature_columns


train_dataset= TorchStreamingMTLDataset(
    data_path=train_data_path,
    categorical_feature_columns=categorical_feature_columns,
    numerical_feature_columns=numerical_feature_columns,
    label_columns=label_columns,
    batch_size=3)

train_dataloader=DataLoader(train_dataset,batch_size=None)

i=0
for feature_dict,label_dict in train_dataloader:
    i+=1
    if i>3:
        break;
    print(feature_dict)
    print(label_dict)