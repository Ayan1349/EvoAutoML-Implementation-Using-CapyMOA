import jpype
import sys
import os
import matplotlib.pyplot as plt
import time


# Specify the actual path to the "project" directory
project_dir = r"C:\Users\Akshita Kumar\Desktop\IP\M2 DS\Data Stream Processing\project_F\CapyMOA\src\capymoa"  # Update this path as per your system

if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

try:
    # Import capymoa specifically from the project directory
    import capymoa
    from capymoa.stream.preprocessing.pipeline import (ClassifierPipeline, ClassifierPipelineElement, TransformerPipelineElement)
    from capymoa.stream.preprocessing import pipeline, transformer
    from capymoa.Base import Ensemble
    from capymoa import evaluation
    from capymoa.base import Classifier
    from capymoa.datasets import Covtype
finally:
    # Restore sys.path to its original state
    if project_dir in sys.path:
        sys.path.remove(project_dir)


from capymoa.EvOAutoML import classification, pipelinehelper
from capymoa.EvOAutoML.pipelinehelper import (
    PipelineHelperClassifier,
    PipelineHelperTransformer,
)
from capymoa.classifier import HoeffdingTree, HoeffdingAdaptiveTree, NaiveBayes, AdaptiveRandomForestClassifier, KNN
from moa.streams.filters import NormalisationFilter, StandardisationFilter
from capymoa.stream.preprocessing import MOATransformer


dataset = Covtype()


model_list = [
            ("HT", HoeffdingTree(schema=dataset.get_schema())),
            ('HAT', HoeffdingAdaptiveTree(schema=dataset.get_schema())),
            ("NB", NaiveBayes(schema=dataset.get_schema())),
            ("KNN", KNN(schema=dataset.get_schema())),
        ]

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

model_pipeline = ClassifierPipeline(model_list=model_list, pipeline_elements=[
    transformer_pe,
    classifier_pe
])

# Map names to pipeline elements
pipeline_elements = {
    "Scaler": model_pipeline.elements[0],  # First element
    "Classifier": model_pipeline.elements[1],  # Second element
}

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

model_list = [
            ("HT", HoeffdingTree(schema=transformer_pe.transformer.get_schema())),
            ('HAT', HoeffdingAdaptiveTree(schema=transformer_pe.transformer.get_schema())),
            ("NB", NaiveBayes(schema=transformer_pe.transformer.get_schema())),
            ("KNN", KNN(schema=transformer_pe.transformer.get_schema())),
        ]


model = classification.EvolutionaryBaggingClassifier(
    model=model_pipeline,
    param_grid=CLASSIFICATION_PARAM_GRID, model_list=model_list,
    seed=43, schema=dataset.get_schema(),
)
print("MODEL", model)
ev = evaluation.ClassificationEvaluator(schema=dataset.get_schema(), window_size=1000)
ev.update(y_target_index=0, y_pred_index=0)
acc_index = ev.metrics_header().index("accuracy")
i=0

# Initialize lists to store accuracy and instance count
accuracy_list = []
instance_count = []

# Start timing the execution
start_time = time.time()
for i in range(50000):
    print("INSTANCE",i)
    instance = dataset.next_instance()
    prediction = int(model.predict(instance))
    ev.update(instance.y_index, prediction)
    if model == None:
        print("Model is None!!!!!!!!!")
    model.train(instance)

    # Record accuracy every 10,000 instances
    if i % 100 == 0:
        current_accuracy = ev.accuracy()
        accuracy_list.append(current_accuracy)
        instance_count.append(i)
        print(f"Accuracy at instance {i}: {current_accuracy}")

# End timing
end_time = time.time()
execution_time = end_time - start_time

# Final accuracy
print("Final Accuracy:", ev.accuracy())

# Plot accuracy vs number of instances
plt.figure(figsize=(10, 5))
plt.plot(instance_count, accuracy_list, marker='o', label='Accuracy')
plt.xlabel('Number of Instances')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Instances')
plt.legend()
plt.grid()
plt.show()

# Bar plot for execution time
plt.figure(figsize=(6, 5))
plt.bar(['Model Execution Time'], [execution_time], color='blue')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time for Model Training and Prediction')
plt.show()

print(ev.accuracy())

