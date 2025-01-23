import copy
import capymoa
import numpy as np
from capymoa import base
from sklearn.model_selection import ParameterSampler
from capymoa.evaluation import evaluation
from capymoa import type_alias
from capymoa import Base
from capymoa import base
from capymoa.evaluation import ClassificationEvaluator
from capymoa.classifier import HoeffdingAdaptiveTree
from capymoa.stream.preprocessing.pipeline import BasePipeline

class EvolutionaryBaggingEstimator(Base.Wrapper.Wrapper, Base.Ensemble.Ensemble):
    def __init__(
        self,
        model,
        param_grid,
        metric,
        schema,
        model_list,     #List of models with tuple values storing model acronym and model object initialized with schema
        evaluator=None, #ClassificationEvaluator/Regressor/etc #changes made
        population_size=10,
        sampling_size=1,
        sampling_rate=1000,
        seed=42,
    ):
        self._rng = np.random.RandomState(seed)
        self.schema = schema
        schemas = []
        for i in range(population_size):
            schemas.append(schema)
        param_iter = ParameterSampler(
            param_grid, population_size, random_state=self._rng
        )
        if evaluator is None:
            self.evaluator = ClassificationEvaluator(schema=self.schema, window_size=1000)
        else:
            self.evaluator = evaluator
        param_list = list(param_iter) 
        param_list = [{k: v for (k, v) in d.items()} for d in param_list] ## list of pipeline dictionaries
        print("PARAM LIST",param_list)
        Base.Ensemble.Ensemble.__init__(
            self, models=[
                self._initialize_model(model, model_list=model_list, params=params)
                for params in param_list
            ], 
        )
        print("model.models",self.models)
        self.model_list = model_list
        self.param_grid = param_grid
        self.population_size = population_size
        self.sampling_size = sampling_size
        self.metric = metric
        self.sampling_rate = sampling_rate
        self.n_models = population_size
        self.model = model
        self.seed = seed
        self._i = 0
        self.evaluator.update(y_target_index=0, y_pred_index=0)
        self.metric_index = self.evaluator.metrics_header().index(metric)
        self._population_metrics = []
        for i in range(self.n_models):
            # Create a new ClassificationEvaluator instance
            evaluatori = ClassificationEvaluator(schema=schemas[i], window_size=1000)
            
            # Update with dummy data to initialize internal MOA objects
            evaluatori.update(y_target_index=0, y_pred_index=0)
            
            # Add the evaluator to the list
            self._population_metrics.append(evaluatori)


    @property
    def _wrapped_model(self):
        return self.model

    def _initialize_model(self, model:BasePipeline, params, model_list):
        modeli = model    #-------------------------------- should be a pipeline
        modeli._set_params(schema=self.schema, model_list=model_list, new_params=params)
        return modeli

    def train(self, instance, **kwargs):
        y = instance.y_index
        if self._i % self.sampling_rate == 0:
            scores=[float(be.metrics()[self.metric_index]) for be in self._population_metrics]
            idx_best = scores.index(max(scores))
            idx_worst = scores.index(min(scores))
            child = self._mutate_estimator(estimator=self[idx_best])
            self.models[idx_worst] = child
            self._population_metrics[idx_worst] = ClassificationEvaluator(schema=self.schema, window_size=1000)

        for idx, modell in enumerate(self):
            k = modell
            self._population_metrics[idx].update(
                y, k.predict(instance)
            )
            for _ in range(self._rng.poisson(6)):
                modell.train(instance)
                
        self._i += 1
        return self

    def reset(self):
        """Resets the estimator to its initial state.

        Returns
        -------
            self

        """
        self._i = 0
        return self

    def _mutate_estimator(self, estimator) -> (base.Classifier):
        child_estimator = estimator
        key_to_change = list(self.param_grid.keys())[1]
        value_to_change = self.param_grid[key_to_change][
            self._rng.choice(range(len(self.param_grid[key_to_change])))
        ]
        child_estimator._set_params(schema=self.schema, new_params={key_to_change: value_to_change}, model_list=self.model_list)
        return child_estimator

    def clone(self):
        """Return a fresh estimator with the same parameters."""
        return self