from deepsklearn.models import LR,DNN,FM,WideDeep,DeepFM,DCN,DCNV2,DLRM,PNN
_Models_REGISTRY={
     "deepfm":DeepFM
}

def get_model(cfg):
    if cfg['model_name'] not in _Models_REGISTRY:
        raise ValueError(f"there is no such type:{cfg['model_name']},available:{sorted(_Models_REGISTRY.keys())}")
    cls=_Models_REGISTRY[cfg['model_name']]
    args=cfg['args'] # this is a dict type
    return cls(**args) # initialize class here
