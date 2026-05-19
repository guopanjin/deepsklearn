from .processors import ContinuousProcessor,CategoricalProcessor
# TODO use annotation
REGISTRY={
     "continuous":ContinuousProcessor,
     "catagorical":CategoricalProcessor
}