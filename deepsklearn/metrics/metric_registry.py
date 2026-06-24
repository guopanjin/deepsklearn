from deepsklearn.metrics import metrics
from deepsklearn.config.constants import Constant

_METRICS_REGISTRY={
    Constant.Metrics.precision:metrics.precision_score,
    Constant.Metrics.recall:metrics.recall,
    Constant.Metrics.f1_score:metrics.f1_score,
    Constant.Metrics.auc:metrics.auc,
    Constant.Metrics.gauc:metrics.gauc,
    Constant.Metrics.recall_at_k:metrics.recall_at_k,
    Constant.Metrics.precision_at_k:metrics.precision_at_k,
    Constant.Metrics.ndcg_at_k:metrics.ndcg_at_k,
    Constant.Metrics.hr_at_k:metrics.hr_at_k,
    Constant.Metrics.mrr_at_k:metrics.mrr_at_k
}


def get_metrics(metric_name:str,**kwargs):
    if metric_name not in _METRICS_REGISTRY.keys():
        raise ValueError(f"this is no such:{metric_name}, availabel:{sorted(_METRICS_REGISTRY.keys())}")
    function=_METRICS_REGISTRY.get(metric_name)
    return function(**kwargs)

