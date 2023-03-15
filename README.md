[//]: # (![Frouros logo]&#40;logo.svg&#41;)

<p align="center">
  <!-- CI -->
  <a href="https://github.com/IFCA/frouros/actions/workflows/ci.yml">
    <img src="https://github.com/IFCA/frouros/actions/workflows/ci.yml/badge.svg?style=flat-square" alt="ci"/>
  </a>
  <!-- Code coverage -->
  <a href="https://codecov.io/gh/IFCA/frouros">
    <img src="https://codecov.io/gh/IFCA/frouros/branch/main/graph/badge.svg?token=DLKQSWYTYM" alt="coverage"/>
  </a>
  <!-- Documentation -->
  <a href="https://frouros.readthedocs.io/">
    <img src="https://readthedocs.org/projects/frouros/badge/?version=latest" alt="documentation"/>
  </a>
  <!-- PyPI -->
  <a href="https://pypi.org/project/frouros">
    <img src="https://img.shields.io/pypi/v/frouros.svg?label=release&color=blue" alt="pypi">
  </a>
  <!-- License -->
  <a href="https://opensource.org/licenses/BSD-3-Clause">
    <img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg" alt="bsd_3_license">
  </a>
</p>

<p align="center">Frouros is a Python library for drift detection in machine learning systems.</p>

Frouros provides a combination of classical and more recent algorithms for drift detection, both for detecting concept and data drift.

## ⚡️ Quickstart

### Concept drift

As a quick example, we can use the wine dataset to which concept drift it is induced in order to show the use of a concept drift detector like DDM (Drift Detection Method).

```python
import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from frouros.detectors.concept_drift import DDM, DDMConfig

np.random.seed(seed=31)

# Load wine dataset
X, y = load_wine(return_X_y=True)

# Split train (70%) and test (30%)
(
    X_train,
    X_test,
    y_train,
    y_test,
) = train_test_split(X, y, train_size=0.7, random_state=31)

# IMPORTANT: Induce/simulate concept drift in the last part (20%)
# of y_test by modifying some labels (50% approx). Therefore, changing P(y|X))
drift_size = int(y_test.shape[0] * 0.2)
y_test_drift = y_test[-drift_size:]
modify_idx = np.random.rand(*y_test_drift.shape) <= 0.5
y_test_drift[modify_idx] = (y_test_drift[modify_idx] + 1) % len(np.unique(y_test))
y_test[-drift_size:] = y_test_drift

# Define and fit model
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression()),
    ]
)
pipeline.fit(X=X_train, y=y_train)

# Detector configuration and instantiation
config = DDMConfig(warning_level=2.0,
                   drift_level=3.0,
                   min_num_instances=30,)
detector = DDM(config=config)

# Simulate data stream (assuming test label available after prediction)
for i, (X, y) in enumerate(zip(X_test, y_test)):
    y_pred = pipeline.predict(X.reshape(1, -1))
    error = 1 - int(y_pred == y)
    detector.update(value=error)
    status = detector.status
    if status["drift"]:
        print(f"Drift detected at index {i}")
        break

# >> Drift detected at index 44
```

More concept drift examples can be found [here](https://frouros.readthedocs.io/en/latest/examples.html#data-drift).

### Data drift

As a quick example, we can generate two normal distributions in order to use a data drift detector like Kolmogorov-Smirnov. This method tries to verify if generated samples come from the same distribution or not. If they come from different distributions, it means that there is data drift.

```python
import numpy as np
from frouros.detectors.data_drift import KSTest

np.random.seed(31)
# X samples from a normal distribution with mean=2 and std=2
x_mean = 2
x_std = 2
# Y samples a normal distribution with mean=1 and std=2
y_mean = 1
y_std = 2

num_samples = 10000
X_ref = np.random.normal(x_mean, x_std, num_samples)
X_test = np.random.normal(y_mean, y_std, num_samples)

alpha = 0.01  # significance level for the hypothesis test

detector = KSTest()
detector.fit(X=X_ref)
statistic, p_value = detector.compare(X=X_test)

p_value < alpha
>> > True  # Drift detected. We can reject H0, so both samples come from different distributions.
```

More data drift examples can be found [here](https://frouros.readthedocs.io/en/latest/examples.html#data-drift).

## 🛠 Installation

Frouros supports Python 3.8, 3.9 and 3.10 versions. It can be installed via pip:

```bash
pip install frouros
```

## 🕵🏻‍♂️️ Drift detection methods

The currently implemented detectors are listed in the following diagram.

![Detectors diagram](images/detectors.png)

## 👍 Contributing

Check out the [contribution](https://github.com/IFCA/frouros/blob/main/CONTRIBUTING.md) section.

## 💬 Citation

Although Frouros paper is still in preprint, if you want to cite it you can use the [preprint](https://arxiv.org/abs/2208.06868) version (to be replaced by the paper once is published).

```bibtex
@article{cespedes2022frouros,
  title={Frouros: A Python library for drift detection in Machine Learning problems},
  author={C{\'e}spedes Sisniega, Jaime and L{\'o}pez Garc{\'\i}a, {\'A}lvaro },
  journal={arXiv preprint arXiv:2208.06868},
  year={2022}
}
```

## 📝 License

Frouros is an open-source software licensed under the [BSD-3-Clause license](https://github.com/IFCA/frouros/blob/main/LICENSE).