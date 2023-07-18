<p align="center">
  <img height="115px" src="https://raw.githubusercontent.com/IFCA/frouros/main/images/logo.png" alt="logo">
</p>

---

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
  <!-- Downloads -->
  <a href="https://pepy.tech/project/frouros">
    <img src="https://static.pepy.tech/badge/frouros" alt="downloads"/>
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
  <!-- arXiv -->
  <a href="https://arxiv.org/abs/2208.06868">
    <img src="https://img.shields.io/badge/arXiv-2208.06868-blue.svg" alt="arxiv">
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

>> Drift detected at index 44
```

More concept drift examples can be found [here](https://frouros.readthedocs.io/en/latest/examples.html#data-drift).

### Data drift

As a quick example, we can use the iris dataset to which data drift in order to show the use of a data drift detector like Kolmogorov-Smirnov test.

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
dim_idx = 0

# IMPORTANT: Induce/simulate data drift in the selected feature of y_test by
# applying some gaussian noise. Therefore, changing P(X))
X_test[:, dim_idx] += np.random.normal(
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
detector.fit(X=X_train[:, dim_idx])

# Apply detector to the selected feature of X_test
result = detector.compare(X=X_test[:, dim_idx])

# Check if drift is taking place
result[0].p_value < alpha
>> True # Data drift detected.
# Therefore, we can reject H0 (both samples come from the same distribution).
```

More data drift examples can be found [here](https://frouros.readthedocs.io/en/latest/examples.html#data-drift).

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
    <td rowspan="16" style="text-align: center; border: 1px solid grey; padding: 8px;">Data drift</td>
    <td rowspan="14" style="text-align: center; border: 1px solid grey; padding: 8px;">Batch</td>
    <td rowspan="9" style="text-align: center; border: 1px solid grey; padding: 8px;">Distance based</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">Anderson-Darling test</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.2307/2288805">Scholz and Stephens (1987)</a></td>
  </tr>
  <tr>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
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
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">MMD</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://dl.acm.org/doi/10.5555/2188385.2188410">Gretton et al. (2012)</a></td>
  </tr>
  <tr>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">U</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">N</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">PSI</td>
    <td style="text-align: center; border: 1px solid grey; padding: 8px;"><a href="https://doi.org/10.1057/jors.2008.144">Wu and Olson (2010)</a></td>
  </tr>
  <tr>
    <td rowspan="5" style="text-align: center; border: 1px solid grey; padding: 8px;">Statistical test</td>
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
    <td style="text-align: center; border: 1px solid grey; padding: 8px;">MMD</td>
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

Although Frouros paper is still in preprint, if you want to cite it you can use the [preprint](https://arxiv.org/abs/2208.06868) version (to be replaced by the paper once is published).

```bibtex
@article{cespedes2022frouros,
  title={Frouros: A Python library for drift detection in machine learning systems},
  author={C{\'e}spedes-Sisniega, Jaime and L{\'o}pez-Garc{\'\i}a, {\'A}lvaro },
  journal={arXiv preprint arXiv:2208.06868},
  year={2022}
}
```

## 📝 License

Frouros is an open-source software licensed under the [BSD-3-Clause license](https://github.com/IFCA/frouros/blob/main/LICENSE).

## 🙏 Acknowledgements

Frouros has received funding from the Agencia Estatal de Investigación, Unidad de Excelencia María de Maeztu, ref. MDM-2017-0765.
