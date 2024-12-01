<p align="center">
  <img height="115px" src="https://raw.githubusercontent.com/IFCA-Advanced-Computing/frouros/main/images/logo.png" alt="logo">
</p>

---

<p align="center">
  <!-- CI -->
  <a href="https://github.com/IFCA-Advanced-Computing/frouros/actions/workflows/ci.yml">
    <img src="https://github.com/IFCA-Advanced-Computing/frouros/actions/workflows/ci.yml/badge.svg?style=flat-square" alt="ci"/>
  </a>
  <!-- Code coverage -->
  <a href="https://codecov.io/gh/IFCA-Advanced-Computing/frouros">
    <img src="https://codecov.io/gh/IFCA-Advanced-Computing/frouros/graph/badge.svg?token=DLKQSWYTYM" alt="coverage"/>
  </a>
  <!-- Documentation -->
  <a href="https://frouros.readthedocs.io/">
    <img src="https://readthedocs.org/projects/frouros/badge/?version=latest" alt="documentation"/>
  </a>
  <!-- Downloads -->
  <a href="https://pepy.tech/project/frouros">
    <img src="https://static.pepy.tech/badge/frouros" alt="downloads"/>
  </a>
  <!-- Platform -->
  <a href="https://github.com/IFCA-Advanced-Computing/frouros">
    <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-blue.svg" alt="downloads"/>
  </a>
  <!-- PyPI -->
  <a href="https://pypi.org/project/frouros">
    <img src="https://img.shields.io/pypi/v/frouros.svg?label=release&color=blue" alt="pypi">
  </a>
  <!-- Python -->
  <a href="https://pypi.org/project/frouros">
    <img src="https://img.shields.io/pypi/pyversions/frouros" alt="python">
  </a>
  <!-- License -->
  <a href="https://opensource.org/licenses/BSD-3-Clause">
    <img src="https://img.shields.io/badge/license-BSD%203--Clause-blue.svg" alt="bsd_3_license">
  </a>
  <!-- Journal -->
  <a href="https://doi.org/10.1016/j.softx.2024.101733">
    <img src="https://img.shields.io/badge/SoftwareX-10.1016%2Fj.softx.2024.101733-blue.svg" alt="SoftwareX">
  </a>
</p>

Frouros is a Python library for drift detection in machine learning systems that provides a combination of classical and more recent algorithms for both concept and data drift detection.

<p align="center">
    <i>
        "Everything changes and nothing stands still"
    </i>
</p>
<p align="center">
    <i>
        "You could not step twice into the same river"
    </i>
</p>
<div align="center" style="width: 70%;">
    <p align="right">
        <i>
            Heraclitus of Ephesus (535-475 BCE.)
        </i>
    </p>
</div>

----

## ⚡️ Quickstart

### 🔄 Concept drift

As a quick example, we can use the breast cancer dataset to which concept drift it is induced and show the use of a concept drift detector like DDM (Drift Detection Method). We can see how concept drift affects the performance in terms of accuracy.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from frouros.detectors.concept_drift import DDM, DDMConfig
from frouros.metrics import PrequentialError

np.random.seed(seed=31)

# Load breast cancer dataset
X, y = load_breast_cancer(return_X_y=True)

# Split train (70%) and test (30%)
(
    X_train,
    X_test,
    y_train,
    y_test,
) = train_test_split(X, y, train_size=0.7, random_state=31)

# Define and fit model
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression()),
    ]
)
pipeline.fit(X=X_train, y=y_train)

# Detector configuration and instantiation
config = DDMConfig(
    warning_level=2.0,
    drift_level=3.0,
    min_num_instances=25,  # minimum number of instances before checking for concept drift
)
detector = DDM(config=config)

# Metric to compute accuracy
metric = PrequentialError(alpha=1.0)  # alpha=1.0 is equivalent to normal accuracy

def stream_test(X_test, y_test, y, metric, detector):
    """Simulate data stream over X_test and y_test. y is the true label."""
    drift_flag = False
    for i, (X, y) in enumerate(zip(X_test, y_test)):
        y_pred = pipeline.predict(X.reshape(1, -1))
        error = 1 - (y_pred.item() == y.item())
        metric_error = metric(error_value=error)
        _ = detector.update(value=error)
        status = detector.status
        if status["drift"] and not drift_flag:
            drift_flag = True
            print(f"Concept drift detected at step {i}. Accuracy: {1 - metric_error:.4f}")
    if not drift_flag:
        print("No concept drift detected")
    print(f"Final accuracy: {1 - metric_error:.4f}\n")

# Simulate data stream (assuming test label available after each prediction)
# No concept drift is expected to occur
stream_test(
    X_test=X_test,
    y_test=y_test,
    y=y,
    metric=metric,
    detector=detector,
)
# >> No concept drift detected
# >> Final accuracy: 0.9766

# IMPORTANT: Induce/simulate concept drift in the last part (20%)
# of y_test by modifying some labels (50% approx). Therefore, changing P(y|X))
drift_size = int(y_test.shape[0] * 0.2)
y_test_drift = y_test[-drift_size:]
modify_idx = np.random.rand(*y_test_drift.shape) <= 0.5
y_test_drift[modify_idx] = (y_test_drift[modify_idx] + 1) % len(np.unique(y_test))
y_test[-drift_size:] = y_test_drift

# Reset detector and metric
detector.reset()
metric.reset()

# Simulate data stream (assuming test label available after each prediction)
# Concept drift is expected to occur because of the label modification
stream_test(
    X_test=X_test,
    y_test=y_test,
    y=y,
    metric=metric,
    detector=detector,
)
# >> Concept drift detected at step 142. Accuracy: 0.9510
# >> Final accuracy: 0.8480
```

More concept drift examples can be found [here](https://frouros.readthedocs.io/en/latest/examples/concept_drift.html).

### 📊 Data drift

As a quick example, we can use the iris dataset to which data drift is induced and show the use of a data drift detector like Kolmogorov-Smirnov test.

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from frouros.detectors.data_drift import KSTest

np.random.seed(seed=31)

# Load iris dataset
X, y = load_iris(return_X_y=True)

# Split train (70%) and test (30%)
(
    X_train,
    X_test,
    y_train,
    y_test,
) = train_test_split(X, y, train_size=0.7, random_state=31)

# Set the feature index to which detector is applied
feature_idx = 0

# IMPORTANT: Induce/simulate data drift in the selected feature of y_test by
# applying some gaussian noise. Therefore, changing P(X))
X_test[:, feature_idx] += np.random.normal(
    loc=0.0,
    scale=3.0,
    size=X_test.shape[0],
)

# Define and fit model
model = DecisionTreeClassifier(random_state=31)
model.fit(X=X_train, y=y_train)

# Set significance level for hypothesis testing
alpha = 0.001
# Define and fit detector
detector = KSTest()
_ = detector.fit(X=X_train[:, feature_idx])

# Apply detector to the selected feature of X_test
result, _ = detector.compare(X=X_test[:, feature_idx])

# Check if drift is taking place
if result.p_value <= alpha:
    print(f"Data drift detected at feature {feature_idx}")
else:
    print(f"No data drift detected at feature {feature_idx}")
# >> Data drift detected at feature 0
# Therefore, we can reject H0 (both samples come from the same distribution).
```

More data drift examples can be found [here](https://frouros.readthedocs.io/en/latest/examples/data_drift.html).

## 🛠 Installation

Frouros can be installed via pip:

```bash
pip install frouros
```

## 🕵🏻‍♂️️ Drift detection methods

The currently implemented detectors are listed in the following table.

<table style="width: 100%; text-align: center; border-collapse: collapse; border: 1px solid grey;">
  <thead>
    <tr>
    <th style="text-align: center; border: 1px solid grey; padding: 4px;">Drift detector</th>
    <th style="text-align: center; border: 1px solid grey; padding: 4px;">Type</th>
    <th style="text-align: center; border: 1px solid grey; padding: 4px;">Family</th>
    <th style="text-align: center; border: 1px solid grey; padding: 4px;">Univariate (U) / Multivariate (M)</th>
    <th style="text-align: center; border: 1px solid grey; padding: 4px;">Numerical (N) / Categorical (C)</th>
    <th style="text-align: center; border: 1px solid grey; padding: 4px;">Method</th>
    <th style="text-align: center; border: 1px solid grey; padding: 4px;">Reference</th>
    </tr>
  </thead>
  <tbody>
  <tr>
    <td rowspan="13" style="text-align: center; border: 1px solid grey; padding: 8px;">Concept drift</td>
    <td rowspan="13" style="text-align: center; border: 1px solid grey; padding: 8px;">Streaming</td>
    <td rowspan="4" style="text-align: center; border: 1px solid grey; padding: 8px;">Change detection</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">BOCD</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.48550/arXiv.0710.3742">Adams and MacKay (2007)</a></td>
  </tr>
  <tr>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">CUSUM</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.2307/2333009">Page (1954)</a></td>
  </tr>
  <tr>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">Geometric moving average</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.2307/1266443">Roberts (1959)</a></td>
  </tr>
  <tr>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">Page Hinkley</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.2307/2333009">Page (1954)</a></td>
  </tr>
  <tr>
    <td rowspan="6" style="text-align: center; border: 1px solid grey; padding: 8px;">Statistical process control</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">DDM</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.1007/978-3-540-28645-5_29">Gama et al. (2004)</a></td>
  </tr>
  <tr>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">ECDD-WT</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.1016/j.patrec.2011.08.019">Ross et al. (2012)</a></td>
  </tr>
  <tr>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">EDDM</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://www.researchgate.net/publication/245999704_Early_Drift_Detection_Method">Baena-Garcıa et al. (2006)</a></td>
  </tr>
  <tr>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">HDDM-A</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.1109/TKDE.2014.2345382">Frias-Blanco et al. (2014)</a></td>
  </tr>
  <tr>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">HDDM-W</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.1109/TKDE.2014.2345382">Frias-Blanco et al. (2014)</a></td>
  </tr>
  <tr>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">RDDM</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.1016/j.eswa.2017.08.023">Barros et al. (2017)</a></td>
  </tr>
  <tr>
    <td rowspan="3" style="text-align: center; border: 1px solid grey; padding: 8px;">Window based</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">ADWIN</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.1137/1.9781611972771.42">Bifet and Gavalda (2007)</a></td>
  </tr>
  <tr>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">KSWIN</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.1016/j.neucom.2019.11.111">Raab et al. (2020)</a></td>
  </tr>
  <tr>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">STEPD</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.1007/978-3-540-75488-6_27">Nishida and Yamauchi (2007)</a></td>
  </tr>
  <tr>
    <td rowspan="19" style="text-align: center; border: 1px solid grey; padding: 8px;">Data drift</td>
    <td rowspan="17" style="text-align: center; border: 1px solid grey; padding: 8px;">Batch</td>
    <td rowspan="9" style="text-align: center; border: 1px solid grey; padding: 8px;">Distance based</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">M</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">Bhattacharyya distance</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://www.jstor.org/stable/25047882">Bhattacharyya (1946)</a></td>
  </tr>
  <tr>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">Earth Mover's distance</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.1023/A:1026543900054">Rubner et al. (2000)</a></td>
  </tr>
  <tr>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">Energy distance</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.1016/j.jspi.2013.03.018">Székely et al. (2013)</a></td>
  </tr>
  <tr>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">Hellinger distance</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.1515/CRLL.1909.136.210">Hellinger (1909)</a></td>
  </tr>
  <tr>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">Histogram intersection normalized complement</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.1007/BF00130487">Swain and Ballard (1991)</a></td>
  </tr>
  <tr>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">Jensen-Shannon distance</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.1109/18.61115">Lin (1991)</a></td>
  </tr>
  <tr>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">Kullback-Leibler divergence</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.1214/aoms/1177729694">Kullback and Leibler (1951)</a></td>
  </tr>
  <tr>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">M</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">Maximum Mean Discrepancy</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://dl.acm.org/doi/10.5555/2188385.2188410">Gretton et al. (2012)</a></td>
  </tr>
  <tr>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">Population Stability Index</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.1057/jors.2008.144">Wu and Olson (2010)</a></td>
  </tr>
  <tr>
    <td rowspan="8" style="text-align: center; border: 1px solid grey; padding: 8px;">Statistical test</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">Anderson-Darling test</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.2307/2288805">Scholz and Stephens (1987)</a></td>
  </tr>
  <tr>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">Baumgartner-Weiss-Schindler test</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.2307/2533862">Baumgartner et al. (1998)</a></td>
  </tr>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">C</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">Chi-square test</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.1080/14786440009463897">Pearson (1900)</a></td>
  </tr>
  <tr>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">Cramér-von Mises test</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.1080/03461238.1928.10416862">Cramér (1902)</a></td>
  </tr>
  <tr>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">Kolmogorov-Smirnov test</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.2307/2280095">Massey Jr (1951)</a></td>
  </tr>
  <tr>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">Kuiper's test</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.1016/S1385-7258(60)50006-0">Kuiper (1960)</a></td>
  </tr>
  <tr>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">Mann-Whitney U test</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.1214/aoms/1177730491">Mann and Whitney (1947)</a></td>
  </tr>
  <tr>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">Welch's t-test</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.2307/2332510">Welch (1947)</a></td>
  </tr>
  <tr>
    <td rowspan="2" style="text-align: center; border: 1px solid grey; padding: 8px;">Streaming</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">Distance based</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">M</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">Maximum Mean Discrepancy</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://dl.acm.org/doi/10.5555/2188385.2188410">Gretton et al. (2012)</a></td>
  </tr>
  <tr>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">Statistical test</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">Incremental Kolmogorov-Smirnov test</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.1145/2939672.2939836">dos Reis et al. (2016)</a></td>
  </tr>
</tbody>
</table>

## ❗ What is and what is not Frouros?

Unlike other libraries that in addition to provide drift detection algorithms, include other functionalities such as anomaly/outlier detection, adversarial detection, imbalance learning, among others, Frouros has and will **ONLY** have one purpose: **drift detection**.

We firmly believe that machine learning related libraries or frameworks should not follow [Jack of all trades, master of none](https://en.wikipedia.org/wiki/Jack_of_all_trades,_master_of_none) principle. Instead, they should be focused on a single task and do it well.

## ✅ Who is using Frouros?

Frouros is actively being used by the following projects to implement drift
detection in machine learning pipelines:

 * [AI4EOSC](https://ai4eosc.eu).
 * [iMagine](https://imagine-ai.eu).

If you want your project listed here, do not hesitate to send us a pull request.

## 👍 Contributing

Check out the [contribution](https://github.com/IFCA/frouros/blob/main/CONTRIBUTING.md) section.

## 💬 Citation

If you want to cite Frouros you can use the [SoftwareX publication](https://doi.org/10.1016/j.softx.2024.101733).

```bibtex
@article{CESPEDESSISNIEGA2024101733,
title = {Frouros: An open-source Python library for drift detection in machine learning systems},
journal = {SoftwareX},
volume = {26},
pages = {101733},
year = {2024},
issn = {2352-7110},
doi = {https://doi.org/10.1016/j.softx.2024.101733},
url = {https://www.sciencedirect.com/science/article/pii/S2352711024001043},
author = {Jaime {Céspedes Sisniega} and Álvaro {López García}},
keywords = {Machine learning, Drift detection, Concept drift, Data drift, Python},
abstract = {Frouros is an open-source Python library capable of detecting drift in machine learning systems. It provides a combination of classical and more recent algorithms for drift detection, covering both concept and data drift. We have designed it to be compatible with any machine learning framework and easily adaptable to real-world use cases. The library is developed following best development and continuous integration practices to ensure ease of maintenance and extensibility.}
}
```

## 📝 License

Frouros is an open-source software licensed under the [BSD-3-Clause license](https://github.com/IFCA/frouros/blob/main/LICENSE).

## 🙏 Acknowledgements

Frouros has received funding from the Agencia Estatal de Investigación, Unidad de Excelencia María de Maeztu, ref. MDM-2017-0765.
