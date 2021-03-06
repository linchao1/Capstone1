
# Udacity Azure ML Nanodegree Capstone Project Using Heart Failure Dataset

![Diagram 1 Heart Failure](screenshots/heart_failure.png?raw=true "heart failure")

## Table of contents
* [Introduction](#Introduction)
* [Project Architecture ](#Project-Architecture)
* [Dataset](#Dataset-Heart-Failure-Dataset)
* [AutoML](#Automated-ML)
* [HyperparameterTuning](#Hyperparameter-Tuning)
* [Model Deployment](#Model-Deployment)
* [Screen Recording](#Screen-Recording)
* [Description of Future Improvements](#Description-of-Future-Improvements)
* [Summary](#Summary)


## Introduction
This project is my capstone project for Uadacity's Azure Machine Learning Nanodegree.  

The dataset I chose is from Kaggle's Heart Failure dataset. I am doing this project in honor of my mother who passed away from heart failure, this month, five years ago. I am very interested to learn the factors that influence heart failure, and to see if I might be at risk,too, for this disease.  I also wish to mention that heart failure is a very specific form of heart disease, where the heart is too weak to pump the necessary blood flow. In the dataset, the 'ejection_fraction' is an important attribute which indicates the strength of each heart pump.  Ejection_fraction is a number that shows how strong the heart can pump blood. It is normally obtained by using MRI or echocardiogram. 

## Project Architecture
I set this project up both using the Virtual Machine provided by Udacity, as well as using portal.azure.com.  Some days one platform performed better than the other. 
This project's architecture is defined as a devops project. This nanodegree course is focused on these steps.
![Diagram 2 Dev Ops ](screenshots/dev_ops.png?raw=true "Dev Ops Architecture").

For this capstone project, the main focus of steps is on training, packaging, validating, and deploying the models.  We didnot focus on monitoring, but it would be an essenatil part of dev ops.
## Dataset- Heart Failure Dataset

## Overview
The dataset I chose is from Kaggle's https://www.kaggle.com/andrewmvd/heart-failure-clinical-data.  This dataset has the following variables, with DEATH_EVENT as the desired prediction.  The following tables shows the data fields for a person.  
Field         | Second Header
------------- | -------------
age   |                      float64
anaemia    |                   int64
creatinine_phosphokinase   |   int64
diabetes            |          int64
ejection_fraction    |         int64
high_blood_pressure   |        int64
platelets           |        float64
serum_creatinine    |        float64
serum_sodium       |           int64
sex                |           int64
smoking            |           int64
time               |           int64
DEATH_EVENT        |           int64

### Task

The specific task is classification, and the features I am detecting is "DEATH_EVENT". For the primary metric, which is the key variable that I use to train for the best models, I chose Accuracy. For my future work, I describe possible other primary metrics, such as recall, that may be better for the determination of those at risk for heart failure. This is because with physical diagnosis of disease, there is a debate as to whether it is okay to have either false positives, or false negatives. In the case of heart failure,  I believe it is better to be tested positive, though scarcy, further tests can rules out the disease. 



### Access
I accessed the dataset through this public, externally available url https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv. 

For the Hyperdrive progam, the following code was used, with TabularDatasetFactory Class, and the METHOD was 
from_delimited_files, which creates a TabularDataset to represent tabular data in delimited files (e.g. CSV and TSV).
````# Try to load the dataset from the Workspace. Otherwise, create it from the file
# NOTE: update the key to match the dataset name
found = False
key = "Heartdataset"
description_text = "Heart DataSet"

if key in ws.datasets.keys(): 
        found = True
        dataset = ws.datasets[key]

if not found:
        # Create AML Dataset and register it into Workspace
        data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv'
        dataset = Dataset.Tabular.from_delimited_files(data_url)        
        #Register Dataset in Workspace
        dataset = dataset.register(workspace=ws,
                                   name=key,
                                   description=description_text)

````
## Automated ML
Here is the `automl` settings and configuration I used for this experiment.

    automl_settings = {
    "experiment_timeout_minutes": 20,
    "max_concurrent_iterations": 4,
    "primary_metric" : 'accuracy',
    "iteration_timeout_minutes": 5
    }
    automl_config = AutoMLConfig(compute_target=compute_target,
                             task = "classification",
                             training_data=train,
                             label_column_name="DEATH_EVENT",   
                             path = project_folder,
                             enable_early_stopping= True,
                             featurization= 'auto',
                             verbosity = logging.INFO,
                             debug_log = "automl_errors.log",
                             **automl_settings
                                )
                            
 Comments on my AutoML settings and configurations:
 The key items to note are that the task is a classification, with DEATH_EVENT as the value to be predicted, and Accuracy is the primary metric to be maximized. Based on my discussion above about false positives and false negatives,  I used the confusion matrix to show my point.  The matrix is shown below with the primary metric as Accuracy.
 
 ![Diagram 2 Best Model Trained](screenshots/final_confusion_matrix.png?raw=true "Confusion Matrix")

For discussion purposes, I suggest that it is better to be false postive (7), than to be false negative (4). This is because further tests can be done on false positive people, whereas false negative people may assume they're okay, and skip further tests.  I believe recall may be another good primary parameter to use for this reason, since it catches all positive instances. 
 
### AutoML Results - Showing Resuls of 86.62%
The best result with AutoML was with voting ensemble with 86.62%.  The parameters were as described above.
````
best_run.get_metrics(name='accuracy')
{'accuracy': 0.8662055335968379}
````

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.
Screen shot of the `RunDetails` widget
![Diagram 1 RunDetails Widget](screenshots/run_complete_final.png?raw=true "RunDetails")

Screen shot of the best model with it's runID  with it's parameters.
![Diagram 2 Best Model Trained](screenshots/automl_best_run_id_new.png?raw=true "Best Model")

Best Model in AutoML:
    
    Best Model: Pipeline(memory=None,
         steps=[('datatransformer',
                 DataTransformer(enable_dnn=None, enable_feature_sweeping=None,
                                 feature_sweeping_config=None,
                                 feature_sweeping_timeout=None,
                                 featurization_config=None, force_text_dnn=None,
                                 is_cross_validation=None,
                                 is_onnx_compatible=None, logger=None,
                                 observer=None, task=None, working_dir=None)),
                ('prefittedsoftvotingclassifier',...
                                                                                                subsample=0.9405263157894738,
                                                                                                subsample_for_bin=200000,
                                                                                                subsample_freq=0,
                                                                                                verbose=-10))],
                                                                     verbose=False))],
                                               flatten_transform=None,
                                               weights=[0.13333333333333333,
                                                        0.13333333333333333,
                                                        0.06666666666666667,
                                                        0.13333333333333333,
                                                        0.06666666666666667,
                                                        0.13333333333333333,
                                                        0.06666666666666667,
                                                        0.06666666666666667,
                                                        0.06666666666666667,
                                                        0.06666666666666667,
                                                        0.06666666666666667]))],
             verbose=False)

    
Screenshot of the best model being registered.
![Diagram 4 Best Model Registered](screenshots/Automl_registered_and_model_ID.png?raw=true "Best Model")


## Hyperparameter Tuning

For hyperparameter tuning, I used Azure Machine Learning HyperDrive package to tune hyperparameters with the Azure SDK. Here are the basic steps

1. Define the parameter search space
2. Specify a primary metric to optimize
3. Specify early termination policy for low-performing runs
4. Allocate resources
5. Launch an experiment with the defined configuration
6. Visualize the training runs
7. Select the best configuration for my model

The parameter sampling method to use over the hyperparameter space includes Random sampling, Grid sampling, & Bayesian sampling. I chose random sampling with continuous hyperparameters.

````
#ps 
ps = RandomParameterSampling({"--C": uniform(0.2, 1),
                             "--max_iter": choice(50, 100, 150)})
                                                         
````

This will define a search space with two parameters, --C ( continuous parameters over a continuous range of values) and --max_iter. The --C can have a uniform distribution with 0.2 as a minimum value and 1 as a maximum value, and the max_iter will be a choice of [50,100,150].

Here are my hyperdrive hyperparameter policy and estimator.

For my early termination policy, I chose *Bandit Policy*, as this automatically terminate poorly performing runs with an early termination policy. Early termination improves computational efficiency. Bandit policy is based on slack factor/slack amount and evaluation interval. Bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run. 

````
# Specify a Policy- Bandit Policy
#policy = ### YOUR CODE HERE ###
policy = BanditPolicy(slack_factor = 0.1, evaluation_interval=2, delay_evaluation=5)

````
My SKLearn Estimator is train2.py
````
    
#create a SKLearn estimator for train2.py

est = SKLearn(source_directory = './',
                     entry_script = 'train2.py',
                     compute_target = compute_target)

````
My  hyperdrive configuration has the following: the primary metric is accuracy, and the goal is to maximize for accuracy, with the max total runs as 20, and max concurrent runs at 4.  I chose a maximum total of 20 runs so that the model would finish in about 30 minutes. The max concurrent runs of 4 was defined by the workspace. 
````
# Create a HyperDriveConfig using the estimator, hyperparameter sampler, and policy.
#hyperdrive_config = ### YOUR CODE HERE ###
hyperdrive_config = HyperDriveConfig(estimator = est,
                             hyperparameter_sampling=ps,
                             policy=policy,
                             primary_metric_name='Accuracy',
                             primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                             max_total_runs=20,
                             max_concurrent_runs=4)
                             
                             
````
### My Reasons for choosing the values and configuration. 
For the parameter Sampler, as I mentioned above, I chose Random sampling, as it supports discrete and continuous hyperparameters. I really liked that it supports early termination of low-performance runs. In random sampling, hyperparameter values are randomly selected from the defined search space.

The primary metric I wanted the hyperparameter tuning to optimize is accuracy. Each training run is evaluated for the primary metric. The early termination policy uses the primary metric to identify low-performance runs.

Early termination policy is Bandit
This automatically terminate poorly performing runs with an early termination policy. Early termination improves computational efficiency. Bandit policy is based on slack factor/slack amount and evaluation interval. Bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run.

The hyperdrive configuartion has the following: 
the primary metric is accuracy, and the goal is to maximize for accuracy, with the max total runs as 20, and max concurrent runs at 4.


### Results

I did two experiments with the same sampling parameters, and here are my accuracy results.

Experiments | Accuracy
------------ | -------------
Experiment 1| 89.33%
Experiment 2 | 86.66%

I attribute the differences to my use of parameter sampling method of Random sampling. In random sampling, hyperparameter values are randomly selected from the defined search space.  Hence this attributed to the differences in accuracy results.

## HD EXPERIMENT 1 - Showing Accuracy of 89.33% 

The results of my model was: 
```
best_run.get_metrics(name = 'Accuracy')
Out[30]:
{'Accuracy': 0.8933333333333333}

```

Screen shot of the `RunDetails` widget
![Diagram 3 RunDetails Widget](screenshots/hyperdrive_run_complete_1.png?raw=true "RunDetails")

Screen shot of the best model trained with it's parameters.
![Diagram 4 Best Model Trained](screenshots/hyperparamter_outputScreen_1.png?raw=true "Best Model")

Screen shot of the best model trained with --C  and max parameters.
![Diagram 6 HD --C](screenshots/hd_best_model_c_max_iter_new_feb9.png?raw=true "Max and --C")



## HD EXPERIMENT 2 - Showing Accuracy of 86.66% 

```
Best Run ID: HD_14d87b4c-d86d-4296-a11f-65ad319d8d9f_2
Best run metrics for accuracy is: 0.8666666666666667
```

Screen shot of the `RunDetails` widget
![Diagram 3 RunDetails Widget](screenshots/HD_runcomplete_latest.png?raw=true "RunDetails")

Screen shot of the best model trained with it's parameters.
![Diagram 4 Best Model Trained](screenshots/HD_new_new_results_best_runrpng.png?raw=true "Best Model")

Here's the code showing the model was registered and saved.
````
#TODO: Register the best model
# Save the best model

model = best_run.register_model(model_name = 'HeartRateModel', 
                                           model_path = './outputs/model.joblib',
                                           model_framework= Model.Framework.SCIKITLEARN)
print("Model saved successfully")
Model saved successfully
````

## Model Deployment
Here's my overview of the deployed model and instructions on how to query the endpoint with a sample input.  I deployed the AutoML model since the Accuracy for AutoML and Hyperparameter Tuning, were less than .03% apart ( accurarcy of 86.33% for AutoML vs. 86.66% Hyperparameter tuning Experiment 2) so I deployed AutoML using the ACI service.

Here are the steps for deployment.

<ul>
    <li>Register the model ( optional)</li>
    <li>Prepare an inference configuration.</li>
    <li>Prepare an entry script.</li>
    <li>Choose a compute target.</li>
    <li>Deploy the model to the compute target.
    <li>Test the resulting web service.</li>
</ul>

I followed these steps, and my deployed model worked.  I queried the endpoint with the first two data rows in dataset, and the query replied with the predictions which were correct.

here's the conda environment information

![Diagram 7 Best Model Trained](screenshots/conda_new_feb10.png?raw=true "Best Model")

Below is how I deployed the model to the compute target.

````
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.webservice import Webservice
from azureml.core.model import Model
from azureml.core.environment import Environment

environment = best_run.get_environment()

inference_config = InferenceConfig(entry_script=script_file_name, environment = environment)

aciconfig = AciWebservice.deploy_configuration(cpu_cores = 1, 
                                               memory_gb = 1, 
                                               enable_app_insights = True,
                                               auth_enabled = True,
                                               tags = {'area': "bmData", 'type': "automl_classification"}, 
                                               description = 'sample service for Automl Classification')

aci_service_name = 'aci-automl-heart-service'
print(aci_service_name)
aci_service = Model.deploy(ws, aci_service_name, [model], inference_config, aciconfig)
aci_service.wait_for_deployment(True)
print(aci_service.state)

)
````
Here is the healthy deployment screenshot

![Diagram 8 Final Healthy](screenshots/final_endpoint.png?raw=true "Best Model")


Here's the output:
````
aci-automl-heart-service
Tips: You can try get_logs(): https://aka.ms/debugimage#dockerlog or local deployment: https://aka.ms/debugimage#debug-locally to debug if deployment takes longer than 10 minutes.
Running.........................................
Succeeded
ACI service creation operation finished, operation "Succeeded"
Healthy
````
Here are the scoring uri, and the key
````
#get scoring uri and primary authentication key
primary, secondary = aci_service.get_keys()
print ('Service state:' + aci_service.state)
print ('Service scoring URI: ' + aci_service.scoring_uri)
print ('Service Swagger URI:' + aci_service.swagger_uri)
print ('Service primary authentication key:' + primary)
````
Here are the print outputs
````
Service state:Healthy
Service scoring URI: http://03f074b7-b00f-4202-b072-ea3eb4427cd0.southcentralus.azurecontainer.io/score
Service Swagger URI:http://03f074b7-b00f-4202-b072-ea3eb4427cd0.southcentralus.azurecontainer.io/swagger.json
Service primary authentication key:x6WpaCh2SPfU4HIpGpVZq2x142mU88Is
````
I used endpoint1.py to make the prediction and the output was:
````
"{\"result\": [1, 1]}"
result: [1, 1], where '1' means the result in the 'DEATH_EVENT' column
````
Here's the screen shot
![Diagram 10 Predictiond](screenshots/final_prediction.png?raw=true "Confusion Matrix")

 Update:   I've added the file containing the environment details.
 
 Update: I've checked my phyton code to be PEP8 compliant with this tool
 http://pep8online.com/checkresult 
 

Reference: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?tabs=azcli

## Screen Recording
<ul>
My screencast demonstrates:
<li>- A Working models of one AutoML.</li>
<li>- Demo of the deployed model with ACI using the AutoML model.</li>
<li>- Demo of a sample request sent to the endpoint and its response </li>
</ul>

**Screencast** https://drive.google.com/file/d/1zeUQoSkBDezqUGhR3FhKI9Dd7qWwiB0p/view?usp=sharing

## Description of Future Improvements
I delved into the specific aspects of which primary metric to use. I concluded the following which is that other primary metrics may provide more robust predictions for people, especially doctors, that may wish to predict the likeliness of a patient having heart failure.  I'll discuss Accuracy, Precision and Recall.

(1) **Accuracy**-  Through out this nano degree, accuracy has been the primary metric that we used in both Project 1 and Project 2. Accuracy is the ratio of predictions that exactly match the true class labels.  This metric is straight forward.

Objective: Closer to 1 the better
Range: [0, 1]

(2) **Precision**	Precision is the ability of a model to avoid labeling negative samples as positive. 

Objective: Closer to 1 the better
Range: [0, 1]

Supported metric names include,
precision_score_macro, the arithmetic mean of precision for each class.
precision_score_micro, computed globally by counting the total true positives and false positives.
precision_score_weighted, the arithmetic mean of precision for each class, weighted by number of true instances in each class.
Comment: this one is a very interesting one, as it is detecting false positives. Once I patient has been labeled as a possible heart failure candiate, they can go through more tests to determine the truth.

(3) **Recall**	Recall is the ability of a model to detect all positive samples.

Objective: Closer to 1 the better
Range: [0, 1]

Supported metric names include,
recall_score_macro: the arithmetic mean of recall for each class.
recall_score_micro: computed globally by counting the total true positives, false negatives and false positives.
recall_score_weighted: the arithmetic mean of recall for each class, weighted by number of true instances in each class.

**Comment**- Recall metric is also interesting to me, as it is focused on making sure to find all positive results, in other words, if someone has heart failure likelihood, this model will not leave anyone out, or guessing.

Reference: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-understand-automated-ml
## Summary
My summary findings are that machine learning can be an assistance to doctors to help with their patients to determine if they have heart failure, especially with the machine learning's model to be tuned for false positives and false negatives. Given the opportunities to probe deeper into a patients health, with false positives, the risk of a wrong diagnosis may be mitigated with further tests.  

I hope my mother is proud of this capstone project I did in her honor.  Thank you, Mom, for being my inspiration. 

