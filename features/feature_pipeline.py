import pandas as pd
from features.build_processor import BuildProcessor

class FeaturePipeline:
    def __init__(self,feature_configs:dict):
        self.feature_configs=feature_configs
        self.processor_dict=self.__initialize_process(self.feature_configs)
    def __initialize_process(self,feature_configs:dict):
        processor_dict={}
        for feature_name,cfg in feature_configs.items():
            processor_dict[feature_name]=BuildProcessor.registry_processor(cfg)
        return processor_dict

    # for streaming training,so we need to input data here
    def transform(self,batch:pd.DataFrame):
        feature_dict={}
        for feature_name,feature_process in self.processor_dict.items():
            feature_dict[feature_name]=feature_process.transform(batch[feature_name])
        return feature_dict

