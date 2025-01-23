from __future__ import annotations

import abc

from capymoa import base
from capymoa.stream import preprocessing
from capymoa import instance
from . import Baseclass

class Estimator(Baseclass.Base, abc.ABC):
    """An estimator."""

    def __init__(self):
        self.moa_model = None  # Placeholder for the MOA algorithm
        self.is_supervised = True  # Default to supervised

    @property
    def _tags(self):
        """Return tags about the estimator's capabilities.
        Tags can be used to specify what kind of inputs an estimator is able to process. For
        instance, some estimators can handle text, whilst others don't. Inheriting from
        `base.Estimator` will imply a set of default tags which can be overridden by implementing
        the `_more_tags` property."""
        return {
            "supports_online_learning": True,
            "requires_supervised_data": self.is_supervised,
        }

    def __or__(self, other):
        """Merge with another Transformer into a Pipeline."""

        if isinstance(other, preprocessing.BasePipeline):
            return other.__ror__(self)
        return preprocessing.BasePipeline(self, other)

    def __ror__(self, other):
        """Merge with another Transformer into a Pipeline."""

        if isinstance(other, preprocessing.BasePipeline):
            return other.__or__(self)
        return preprocessing.BasePipeline(other, self)

    def _more_tags(self):
        return set()
    
    '''def _initialize_moa_model(self, moa_class_name, **params):
        """Helper to initialize a MOA model through CapyMOA."""
        self.moa_model = instance.Instance(moa_class_name, **params)'''

    @classmethod
    def _unit_test_params(self):
        """Indicates which parameters to use during unit testing.

        Most estimators have a default value for each of their parameters. However, in some cases,
        no default value is specified. This class method allows to circumvent this issue when the
        model has to be instantiated during unit testing.

        This can also be used to override default parameters that are computationally expensive,
        such as the number of base models in an ensemble.

        """
        yield {}

    def _unit_test_skips(self):
        """Indicates which checks to skip during unit testing.

        Most estimators pass the full test suite. However, in some cases, some estimators might not
        be able to pass certain checks.

        """
        return set()