# FAQ

Here we will try to answer some of the most common questions about drift detection and the Frouros library.

## What is the difference between *concept drift* and *data drift*?

Concept drift refers to changes in the underlying concept being modeled, such as changes in the relationship between 
the input features and the target variable. It can be caused by changes in the conditional probability $P(y|X)$ with or 
without a change in $P(X)$. Data drift, on the other hand, refers to changes in the distribution of the input features 
$P(X)$, such as changes in the feature distributions over time. It focuses on detecting when the incoming data no longer 
resembles the data the model was trained on, potentially leading to decreased performance or reliability.

## What is the difference between *out-of-distribution* detection and *data drift* detection?

Out-of-distribution detection focuses on identifying samples that fall outside the training distribution, often used 
to detect anomalies or novel data. It aims to detect instances that differ significantly from the data the model was 
trained on. Data drift detection, on the other hand, is concerned with identifying shifts or changes in the 
distribution of the data over time.

## How can I detect *concept drift* without having access to the ground truth labels at inference time?

In cases where ground truth labels are not available at inference time or the verification latency is high, it may not 
be possible to directly detect concept drift using traditional methods. In such cases, it may be necessary to use 
alternative techniques, such as data drift detection, to monitor changes in the feature distributions and identify 
potential drift. By monitoring the feature distributions, it may be possible to detect when the incoming data no 
longer resembles the data the model was trained on, even in the absence of ground truth labels.

## Why do I need to use a *drift* detector?

One of the main mistakes when deploying a machine learning model for consumption is to assume that the data used for 
inference will come from the same distribution as the data on which the model was trained, i.e., that the data will be 
stationary. It may also be the case that the data used at inference time is still similar to those used for training, 
but the concept of what was learned in the first instance has changed over time, making the model obsolete in terms of 
performance.

Drift detectors make it possible to monitor model performance or feature distributions to detect significant deviations 
that can cause model performance decay. By using them, it is possible to know when it is necessary to replace the 
current model with a new one trained on more recent data.

## Is *model drift* the same as *concept drift*?

Model drift is a term used to describe the degradation of a model's performance over time. This can be caused by a 
variety of factors, including concept drift, data drift, or other issues such as model aging. Concept drift, on the 
other hand, refers specifically to changes in the underlying concept being modeled, such as changes in the relationship 
between the input features and the target variable. While concept drift can lead to model drift, model drift can also be
caused by other factors and may not always be directly related to changes in the underlying concept.

## What actions should I take if *drift* is detected in my model?

If drift is detected in your model, it is important to take action to address the underlying cause of the drift. 
This may involve retraining the model on more recent data, updating the model's features or architecture, or taking 
other steps to ensure that the model remains accurate and reliable. In some cases, it may also be necessary to 
re-evaluate the model's performance and consider whether it is still suitable for its intended use case.

## Can Frouros be integrated with popular machine learning frameworks such as TensorFlow or PyTorch?

Yes, Frouros is designed to be compatible with any machine learning frameworks such as TensorFlow or PyTorch. It is 
framework-agnostic and can be used with any machine learning model or pipeline.

For instance, we provide an [example](./examples/data_drift/MMD_advance.html) that shows how to integrate Frouros with a PyTorch model to detect data 
drift for a computer vision use case. In addition, there is an [example](./examples/concept_drift/DDM_advance.html) that shows how to integrate Frouros with 
scikit-learn to detect concept drift in a streaming manner.

## How frequently should I run *drift* detection checks in my machine learning pipeline?

The frequency of drift detection checks will depend on the specific use case and the nature of the data being 
processed. In general, it is a good practice to run drift detection checks regularly, such as after each batch of 
data or at regular intervals, to ensure that any drift is detected and addressed in a timely manner.

## What are some common causes of *drift* in machine learning models?

Drift in machine learning models can be caused by a variety of factors, including changes in the underlying concept 
being modeled, changes in the distribution of the input features, changes in the relationship between the input 
features and the target variable, and other issues such as model aging or degradation. It is important to monitor 
models for drift and take action to address any detected drift to maintain model accuracy and reliability.

## How can I contribute to the development of Frouros or report issues?

The [contribute section](./contribute.html#how-to-contribute) provides information on how to contribute to the development of Frouros, 
including guidelines for reporting issues, submitting feature requests, and contributing code or documentation.

## Does Frouros provide visualization tools for *drift* detection results?

Frouros does not currently provide built-in visualization tools for drift detection results, but it is planned to 
include them in future releases.
