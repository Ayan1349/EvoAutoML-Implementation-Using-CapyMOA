from capymoa.stream.preprocessing import MOATransformer
from capymoa.stream.preprocessing.pipeline import (ClassifierPipeline, ClassifierPipelineElement, TransformerPipelineElement)

from capymoa.classifier import HoeffdingTree, HoeffdingAdaptiveTree, NaiveBayes, AdaptiveRandomForestClassifier, KNN

from capymoa.EvOAutoML.pipelinehelper import (
    PipelineHelperClassifier,
    PipelineHelperTransformer,
)

from moa.streams.filters import NormalisationFilter, StandardisationFilter
from capymoa.stream.preprocessing import ClassifierPipeline
from capymoa.datasets import Covtype

dataset= Covtype()


model_list = [
            ("HT", HoeffdingTree(schema=dataset.get_schema())),
            ('HAT', HoeffdingAdaptiveTree(schema=dataset.get_schema())),
            ("NB", NaiveBayes(schema=dataset.get_schema())),
            ("KNN", KNN(schema=dataset.get_schema())),
        ]

# Create the pipeline
SF = MOATransformer(schema=dataset.get_schema(), moa_filter=StandardisationFilter())
NF = MOATransformer(schema=dataset.get_schema(), moa_filter=NormalisationFilter())

transformer_pe = TransformerPipelineElement(PipelineHelperTransformer(
        [
            ("StandardScaler",SF),
            ("MinMaxScaler", NF),
        ],
        schema=dataset.get_schema()
    ))


classifier_pe = ClassifierPipelineElement(PipelineHelperClassifier(
        [
            ("HT", HoeffdingTree(schema=transformer_pe.transformer.get_schema())),
            ('HAT', HoeffdingAdaptiveTree(schema=transformer_pe.transformer.get_schema())),
            ("NB", NaiveBayes(schema=transformer_pe.transformer.get_schema())),
            ("KNN", KNN(schema=transformer_pe.transformer.get_schema())),
        ],
        schema=transformer_pe.transformer.get_schema()
    ))

AUTOML_CLASSIFICATION_PIPELINE = ClassifierPipeline(model_list=model_list, pipeline_elements=[
    transformer_pe,
    classifier_pe
])
# Map names to pipeline elements
pipeline_elements = {
    "Scaler": AUTOML_CLASSIFICATION_PIPELINE.elements[0],  # First element
    "Classifier": AUTOML_CLASSIFICATION_PIPELINE.elements[1],  # Second element
}

# Generate parameter grid
CLASSIFICATION_PARAM_GRID = {
    "Scaler": pipeline_elements["Scaler"].generate({}),
    "Classifier": pipeline_elements["Classifier"].generate(
        {
            "HT__nb_threshold": [5, 10, 20, 30],
            "HT__grace_period": [10, 100, 200],
            "HT__tie_threshold": [0.001, 0.01, 0.05, 0.1],
            "HAT__grace_period": [10, 100, 200],
            "KNN__k": [1, 5, 20],
            "KNN__window_size": [100, 500, 1000],
        }
    ),
}
