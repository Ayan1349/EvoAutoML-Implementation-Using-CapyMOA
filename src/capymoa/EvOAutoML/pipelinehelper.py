import typing

import pandas as pd
from capymoa import base, Base
from capymoa.Base import Estimator
from capymoa.stream import preprocessing
from capymoa.base import MOAClassifier, MOARegressor
from capymoa.stream.preprocessing import transformer
from moa.streams.filters import NormalisationFilter, StandardisationFilter

from capymoa.EvOAutoML.base.utils import PipelineHelper
from capymoa import type_alias, classifier
from capymoa.stream.preprocessing.transformer import MOATransformer
import moa.streams.filters
from capymoa.datasets import Covtype

dataset = Covtype()

class PipelineHelperClassifier(PipelineHelper, MOAClassifier):
    """
    This class is used to create a pipeline, where multiple classifiers are
    able to used in parallel. The selected classifier (`selected_model`) is
    used to make predictions as well as for training.
    The other classifiers are not trained in parallel.

    Parameters
    ----------
    models: dict
        A dictionary of models that can be used in the pipeline.
    selected_model: Estimator
        the model that is used for training and prediction.
    """
    def __init__(self, models, selected_model=None, seed=42, schema=None, CLI=None):

        PipelineHelper.__init__(self, models, selected_model, seed=seed)
        MOAClassifier.__init__(self, moa_learner=self.selected_model.moa_learner, schema=schema, CLI=None, random_seed=seed)

    @classmethod
    def _unit_test_params(cls,self):
        models = [
            ("HT", HoeffdingTree(schema=self.schema)),   
            ("EFDT", EFDT(self.schema)),
        ]
        yield {
            "models": models,
        }

    def train(
        self, instance, **kwargs
    ) -> Estimator:     
        self.selected_model.train(instance, **kwargs)

    def predict(self, instance, **kwargs) -> type_alias.Label:
        self.selected_model= self.selected_model.__class__(schema=dataset.get_schema())
        return self.selected_model.predict(instance)


    def predict_proba(
        self, instance
    ) -> typing.Dict[type_alias.LabelProbabilities, float]:
        return self.selected_model.predict_proba(instance)


class PipelineHelperTransformer(PipelineHelper, MOATransformer):
    """
    Add some Text here
    """

    def __init__(self, models, selected_model=None, seed=42, schema=None, CLI=None, moa_filter: moa.streams.filters.StreamFilter | None = None):

        PipelineHelper.__init__(self, models, selected_model, seed=seed)
        MOATransformer.__init__(self, moa_filter=self.selected_model.moa_filter, schema=schema, CLI=None)


    @classmethod
    def _unit_test_params(cls):
        models = [
            ("MinMax", MOATransformer(moa_fiter=NormalisationFilter())),
            ("NORM", MOATransformer(moa_filter=StandardisationFilter())),
        ]
        yield {
            "models": models,
        }

    @property
    def _supervised(self):
        if self.selected_model._supervised:
            return True
        else:
            return False

    def transform_instance(self, instance) -> dict:
        """

        Args:
            instance

        Returns:

        """
        self.selected_model = self.selected_model.__class__(schema=self.schema, moa_filter=self.moa_filter)
        return self.selected_model.transform_instance(instance)


