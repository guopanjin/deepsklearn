import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset,get_worker_info
from deepsklearn.utils import Logger
import pyarrow.parquet as pq
'''
Build the streamingDataset based on the pytorch API
return (feature_dict,label_dict)
'''
logger=Logger.get_logger()

class  TorchStreamingItem2vecDataset(IterableDataset):
   def __init__(self,data_path,
                sampler_weight_path,
                feature_item,
                label_item,
                batch_size=1000,
                sample_k: int = 5
                ):
       self.data_path=data_path
       self.sampler_weight_path=sampler_weight_path
       self.batch_size=batch_size
       self.feature_item=feature_item
       self.label_item=label_item
       self.sample_k=sample_k
       self.file_list=sorted(self.__get_file_list())# make sure the dataset is stable
       self.item_list_tensor,self.weight_tensor=self._get_sampler_weights()
   def _get_sampler_weights(self):
       sampler_weight=pd.read_parquet(self.sampler_weight_path)
       item_list_tensor=torch.tensor(sampler_weight["item_id"].to_numpy())
       weight_tensor=torch.tensor(sampler_weight["frequency"].to_numpy(),dtype=torch.float)
       return item_list_tensor,weight_tensor

   def __get_file_list(self):
       file_list=[]
       if os.path.isdir(self.data_path):
           for root, dirs, files in os.walk(self.data_path):
               for file in files:
                   file_list.append(os.path.join(root, file))
       else:
           file_list.append(self.data_path)
       logger.info(f"file_list:{file_list}")
       return file_list

   def __parse_data(self,file):
       parquet_file = pq.ParquetFile(os.path.expanduser(file))
       for batch in parquet_file.iter_batches(batch_size=self.batch_size):
           batch_df = batch.to_pandas()
           #return 1D array shape=(N,)
           feature_item=torch.tensor(np.stack(batch_df[self.feature_item].to_numpy(),axis=0))
           positive_items=torch.tensor(np.stack(batch_df[self.label_item].to_numpy(),axis=0))
           step_size=positive_items.shape[0]
           sample_indices=torch.multinomial(self.weight_tensor,
                             num_samples=step_size*self.sample_k,
                             replacement=True
                             )
           #TODO need to remove the itemids that are same with the postive items
           negative_items=self.item_list_tensor[sample_indices].reshape(step_size,self.sample_k)

           feature_dict={"features":feature_item,#(batch_size,)
                         "positive_items":positive_items,#(batch_size,)
                         "negative_items":negative_items,#(batch_size,k)
                         }
           yield feature_dict
   def __iter__(self):
       worker_info=get_worker_info()
       if worker_info is None:
           for file in self.file_list:
               yield from self.__parse_data(file)
       else:
           worker_number= worker_info.num_workers
           worker_id= worker_info.id
           for index,file in enumerate(self.file_list):
               if index%worker_number==worker_id:
                   yield from self.__parse_data(file)
