# Concepts

Some concepts related to the drift detection field must be explained in order use `frouros` in a correct manner and at its fully potential.

## What is drift detection?

Can be defined as the process of trying to detect a significant change in the concept previously learned by a model (*concept drift*), or a change related to the feature/covariate distributions (*data drift*) that can end up producing a performance decay in model's performance.

Traditionally there has been little consensus on the terminology and definitions of the
different types of drift, as stated in {cite}`moreno2012unifying`. In order to adopt some
clear definitions, we apply those used in {cite}`gama2014survey` for the *concept drift* part, in combination with those used in {cite}`rabanser2019failing`'s work
for detecting *dataset shift* using only the feature/covariate distributions.

Therefore, the problem statement can be defined as follows:

Given a time period ${[0, t]}$, a set of sample-pairs ${D=\{(X_{0}, y_{0}),...,(X_{t}, y_{t})\}}$, where ${X_{i} \in \mathbb{R}^{m}}$ is the ${m}$-dimensional feature vector and ${y_{i} \in \mathbb{R}^{k}}$ is the ${k}$-class vector (using *one-hot encoding*) if we are dealing with a classification problem or ${y_{i} \in \mathbb{R}}$ is a scalar if it is a regression problem, ${D}$ is used to fit ${\hat{f} \colon X \to Y}$ (known as model) to be as close as possible to the unknown ${{f} \colon X \to Y}$. *Machine learning* algorithms are typically used for this fitting procedure. 
${(X_{i}, y_{i}) \notin D}$ samples obtained in ${[t+1, \infty)}$ and used by ${\hat{f}}$ may start to differ with respect to ${D}$ pairs from a statistical point of view. It is also possible that some changes occur in terms of concept of the problem (change in ${f}$).

Since ${P(y, X) = P(y|X) P(X)}$ {cite}`moreno2012unifying`, a change in the joint distribution between two different times that can produce some performance degradation can be described as follows:

$$
P_{[0, t]}(X, y) \neq P_{[t+1, \infty)}(X, y)
$$

The different types of changes that are considered as a form of drift can be categorized in the following types:

- **Concept drift**: There is a change in the conditional probability $P(y|X)$ with or without a change in ${P(X)}$. Thus, it can be defined as ${P_{[0, t]}(y|X) \neq P_{[t+1, \infty)}(y|X)}$. [Concept drift methods](#concept-drift) aim to detect this type of drift. Also known as *real concept drift* {cite}`gama2014survey`.

- **Data drift**: There is a change in ${P(X)}$. Therefore, this type of drift only focuses in the distribution of the covariates ${P(X)}$, so
${P_{[0, t]}(X) \neq P_{[t+1, \infty)}(X)}$. [Data drift methods](#data-drift) are designed to try to detect this type drift. Unlike *concept drift* taking place, the presence of *data drift* does not guarantee that model's performance is being affected, but it is highly probable that is happening. We have renamed *dataset shift* {cite}`rabanser2019failing` to *data drift*
in order to maintain consistency with the *concept drift* definition. These *data drift* methods can also be used to detect *label drift*, also known as *prior probability shift* {cite}`storkey2009training`, where the label distribution ${P(Y)}$ is the one that changes over time, in such a way that ${P_{[0, t]}(Y) \neq P_{[t+1, \infty)}(Y)}$.

## Verification latency or delay

According to {cite}`dos2016fast`, is defined as the period between a model's prediction and the availability of the ground-truth label (in case of a classification problem) or the target value (in case of a regression problem).
In real-world cases, the *verification latency* is highly dependent on the application domain and even in some problems it is no possible to finally obtain the ground-truth/target value, which makes it impossible to detect the *concept drift* using concept drift methods, therefore other techniques can to be used, such as [data drift methods](#data-drift) that only focus on covariate distributions.

## Drift detection methods

Drift detection methods can be classified according to the type of drift they can detect and how they detect it.

### Concept drift

Their main objective is to **detect concept drift**. They are closely related to data stream mining, online and incremental learning. 

At the time of writing this, Frouros only implements *concept drift* detectors that work in a {doc}`streaming </api_reference/detectors/concept_drift/streaming>` manner. This means that the detector can only be updated with a single sample each time. 

### Data drift

On the other hand, there are problems where it is very costly or even impossible to obtain labels in a reasonable amount of time (see [verification latency](#verification-latency-or-delay)). In this case, is not possible to directly check if *concept drift* is occurring, so **detect data drift** becomes the main objective of these type of methods.

At the time of writing this, Frouros implements detectors that are capable to work in {doc}`batch </api_reference/detectors/data_drift/batch>` or {doc}`streaming </api_reference/detectors/data_drift/streaming>` mode. In addition, we can difference between univariate and multivariate data drift detectors, according to the type of feature/covariate distributions used.
```{bibliography}
:filter: docname in docnames
```
