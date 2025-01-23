import collections

#from river import base, metrics, tree
from capymoa import base, evaluation
from capymoa.classifier import (
    HoeffdingTree,
    NoChange,
    MajorityClass,
    OnlineBagging,
    LeveragingBagging,
    AdaptiveRandomForestClassifier,
    StreamingRandomPatches,
    OnlineSmoothBoost,
    OzaBoost,
    StreamingGradientBoostedTrees
)

from capymoa.EvOAutoML.base.evolution import (
    EvolutionaryBaggingEstimator,
)
from capymoa.EvOAutoML.config import (
    AUTOML_CLASSIFICATION_PIPELINE,
    CLASSIFICATION_PARAM_GRID,
)


class EvolutionaryBaggingClassifier(
    EvolutionaryBaggingEstimator, base.Classifier
):
    """
    Evolutionary Bagging Classifier follows the Oza Bagging approach to update
    the population of estimator pipelines.

    Parameters
    ----------
    model
        A model or model pipeline that can be configured
        by the parameter grid.
    param_grid
        A parameter grid, that represents the configuration space of the model.
    population_size
        The population size estimates the size of the population as
        well as the size of the ensemble used for the prediction.
    sampling_size
        The sampling size estimates how many models are mutated
        within one mutation step.
    metric
        The metric that should be optimised.
    sampling_rate
        The sampling rate estimates the number of samples that are executed
        before a mutation step takes place.
    seed
        Random number generator seed for reproducibility.
    """

    def __init__(
        self,
        schema,
        model_list,
        model=AUTOML_CLASSIFICATION_PIPELINE,
        param_grid=CLASSIFICATION_PARAM_GRID,
        population_size=10,
        sampling_size=1,
        metric="accuracy",
        sampling_rate=1000,
        seed=42,
    ):

        super().__init__(
            model=model,
            param_grid=param_grid,
            population_size=population_size,
            sampling_size=sampling_size,
            metric=metric,
            model_list=model_list,
            sampling_rate=sampling_rate,
            seed=seed,
            schema=schema,
        )
        self.schema=schema
        

    @classmethod
    def _unit_test_params(cls, self):
        model = HoeffdingTree(schema=self.schema) 
        param_grid = {
            'grace_period': [500, 300, 1000, 100, 150, 200],
        }

        yield {
            "model": model,
            "param_grid": param_grid,
        }

    @classmethod
    def _unit_test_skips(self) -> set:
        """
        Indicates which checks to skip during unit testing.
        Most estimators pass the full test suite. However, in some cases,
        some estimators might not be able to pass certain checks.
        Returns
        -------
        set
            Set of checks to skip during unit testing.
        """
        return {"check_init_default_params_are_not_mutable"}
    

    def predict(self, instance):
        """
        Averages the predictions of each classifier and returns the most
        likely label.

        Parameters
        ----------
        instance

        Returns
        -------
        The predicted label with the highest probability.
        """
        y_pred_proba = self.predict_proba(instance)
        if y_pred_proba:
            return max(y_pred_proba, key=y_pred_proba.get)
        return None


    def predict_proba(self, instance):
        """Averages the predictions of each classifier."""
        
        k=[]
        for classifier in self:
            k.append(int(classifier.predict(instance)))
        y_pred = collections.Counter(k)

        total = sum(y_pred.values())
        if total > 0:
            return {label: proba / total for label, proba in y_pred.items()}
        return y_pred
