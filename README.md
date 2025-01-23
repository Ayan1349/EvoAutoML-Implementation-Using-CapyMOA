# EvoAutoML Implementation Using CapyMOA

This repository contains an implementation of the EvoAutoML algorithm using the CapyMOA framework. The original source code is available at [https://github.com/kulbachcedric/EvOAutoML.git](https://github.com/kulbachcedric/EvOAutoML.git). The primary focus of this project is on classification tasks, with experiments conducted on the Covtype dataset. The implementation can be extended to other datasets and models by modifying the parameter grid and dataset configurations.

---

## Prerequisites

- Python 3.11 or higher  
- Conda for virtual environment management  

---

## Running the Code

1. *Run the Script*:  
   Use the provided run.py file to execute the EvoAutoML pipeline. Before running the script:
   - Modify the project_dir path in the run.py file to point to the capymoa folder inside the src directory of this repository. This ensures that all necessary packages are correctly imported.
   - Adjust the dataset name, the list of models to be used in the pipeline, and the corresponding parameters as needed.

2. *Output*:  
   - The script processes the specified number of instances of the chosen dataset.  
   - Accuracy is recorded and plotted against the number of processed instances.  
   - The total execution time is displayed as a bar plot.

---

## Features

- *Classification Tasks*: Currently implemented for classification using the Covtype dataset.  
- *Extensibility*: The dataset and models in the parameter grid can be customized as required.  
- *CapyMOA Integration*: The CapyMOA package is used in editable mode, with relevant modifications made to both CapyMOA and EvoAutoML to ensure seamless execution.  
- *Pipeline Definition*: Flexible pipelines with support for:
  - Data preprocessing using transformers like StandardisationFilter and NormalisationFilter.  
  - Multiple classifiers such as HoeffdingTree, HoeffdingAdaptiveTree, NaiveBayes, and KNN.
- *Hyperparameter Tuning*: An evolutionary algorithm dynamically selects optimal models and hyperparameters.  
- *Visualization*:
  - Accuracy vs Number of Instances Plot.  
  - Execution Time Bar Plot.

---

## Modifications

- **CapyMOA/src**: Added EvoAutoML folder.  
- **CapyMOA/src/capymoa/base**: Added Baseclass.py, Estimator.py, Ensemble.py, and Wrapper.py to define:
  - A Base class.
  - An Estimator class (parent class for all estimators).
  - An Ensemble class (code adapted from the River source code available at [https://github.com/online-ml/river.git](https://github.com/online-ml/river.git)).
- **CapyMOA/src/capymoa/stream/preprocessing**: Made modifications in pipeline.py and transformer.py.  
- **CapyMOA/src/capymoa/jar**: Added sizeofag-1.0.0.jar to handle MOA-related errors.

---

## Future Work

- Extend support to regression tasks.  
- Explore additional datasets and classification algorithms.  
- Optimize the mutation operation for improved pipeline generation.

---

## Acknowledgments

- The authors of the EvoAutoML paper for the original implementation.  
- The developers of the CapyMOA framework for providing a comprehensive API for MOA integration in Python.
