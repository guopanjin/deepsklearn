import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset,get_worker_info
from deepsklearn.utils.hash_utils import xxhash_encoder
from deepsklearn.utils import Logger
'''
Build the streamingDataset based on the pytorch API
return (feature_dict,label_dict)
'''
logger=Logger.get_logger()
class CategoricalProcessor():
    '''
    return 1D-array shape=(N,)
    '''
    @classmethod
    def transform(cls,series:pd.Series,bucket_size):
        #fill the missing data
        x=series.fillna("MISSING").astype(str).values
        x=np.array([
            xxhash_encoder(v)%bucket_size for v in x
        ])
        return x

class  TorchStreamingMTLDataset(IterableDataset):
   def __init__(self,data_path,
                categorical_feature_columns:dict,
                numerical_feature_columns:list,
                label_columns:list,
                batch_size=1000):
       self.data_path=data_path
       self.batch_size=batch_size
       self.categorical_feature_columns=categorical_feature_columns
       self.numerical_feature_columns=numerical_feature_columns
       self.label_columns=label_columns
       self.file_list=sorted(self.__get_file_list())# make sure the dataset is stable
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
       for batch in pd.read_csv(file,chunksize=self.batch_size):
           feature_dict={}
           label_dict={}
           for category_feature_name,num_embeddings in self.categorical_feature_columns.items():
               feature_dict[category_feature_name]=CategoricalProcessor.transform(batch[category_feature_name],num_embeddings)
           for numerical_feature_column in self.numerical_feature_columns:
               #errors="coerce" if can not covert will convert none
               numerical_feature_data=pd.to_numeric(batch[numerical_feature_column],errors="coerce").fillna(-1.0)
               numerical_feature_np=numerical_feature_data.values
               sign=np.where(numerical_feature_np>0,1,0)
               numerical_feature_result=sign*np.log1p(np.abs(numerical_feature_np))
               # the default dtype of numpy is float64,but the default dtype of torch is float32
               # so nedd to convert to float32 before feeded to torch
               feature_dict[numerical_feature_column]=torch.tensor(numerical_feature_result,dtype=torch.float32)
           for label in self.label_columns:
               #the default dtype of numpy is float64,but the default dtype of torch is float32
               #so nedd to convert to float32 before feeded to torch
               label_dict[label]=torch.tensor(batch[label].values,dtype=torch.float32)
           yield (feature_dict,label_dict)
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

