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
  <!-- License -->
  <a href="https://opensource.org/licenses/BSD-3-Clause">
    <img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg" alt="bsd_3_license">
  </a>
</p>

<p align="center">Frouros is a Python library for drift detection in Machine Learning problems.</p>

Frouros provides a combination of classical and more recent algorithms for drift detection, both for detecting concept and data drift.

## Quickstart

As a quick and easy example, we can generate two normal distributions in order to use a data drift detector like Kolmogorov-Smirnov. This method tries to verify if generated samples come from the same distribution or not. If they come from different distributions, it means that there is data drift.

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

More examples can be found [here](https://frouros.readthedocs.io/en/latest/examples.html).

## Installation

Frouros supports Python 3.8, 3.9 and 3.10 versions. It can be installed via pip:

```bash
pip install frouros
```

Latest main branch modifications can be installed via:
```bash
pip install git+https://github.com/IFCA/frouros.git
```

## Drift detection methods

The currently supported methods are listed in the following table. They are divided in three main categories depending on the type of drift that they are capable of detecting and how they detect it.

<table class="center">
<thead>
<tr>
    <th>Type</th>
    <th>Subtype</th>
    <th>Method</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="12">
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/concept_drift/base.py"> 
            <div style="height:100%;width:100%">
                Concept drift
            </div>
        </a>
    </td>
    <td rowspan="3">
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/supervised/cusum_based/base.py"> 
            <div style="height:100%;width:100%">
                CUSUM Based
            </div>
        </a>
    </td>
   <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/supervised/cusum_based/cusum.py"> 
            <div style="height:100%;width:100%">
                CUSUM
            </div>
        </a>
    </td>
  <tr>
    <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/supervised/cusum_based/geometric_moving_average.py">  
            <div style="height:100%;width:100%">
                Geometric Moving Average
            </div>
        </a>
    </td>
  </tr>
  <tr>
    <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/supervised/cusum_based/page_hinkley.py">  
            <div style="height:100%;width:100%">
                Page Hinkley
            </div>
        </a>
    </td>
  </tr>
    <td rowspan="7">
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/supervised/ddm_based/base.py">  
            <div style="height:100%;width:100%">
                DDM Based
            </div>
        </a>
    </td>
    <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/supervised/ddm_based/ddm.py">  
            <div style="height:100%;width:100%">
                DDM
            </div>
        </a>
    </td>
  <tr>
    <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/supervised/ddm_based/ecdd.py">  
            <div style="height:100%;width:100%">
                ECDD-WT
            </div>
        </a>
    </td>
  </tr>
  <tr>
    <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/supervised/ddm_based/eddm.py">  
            <div style="height:100%;width:100%">
                EDDM
            </div>
        </a>
    </td>
  </tr>
  <tr>
    <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/supervised/ddm_based/hddm.py">  
            <div style="height:100%;width:100%">
                HDDM-A
            </div>
        </a>
    </td>
  </tr>
  <tr>
    <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/supervised/ddm_based/hddm.py">  
            <div style="height:100%;width:100%">
                HDDM-W
            </div>
        </a>
    </td>
  </tr>
  <tr>
    <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/supervised/ddm_based/rddm.py">  
            <div style="height:100%;width:100%">
                RDDM
            </div>
        </a>
    </td>
  </tr>
  <tr>
    <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/supervised/ddm_based/stepd.py">  
            <div style="height:100%;width:100%">
                STEPD
            </div>
        </a>
    </td>
  </tr>
  <td rowspan="2">
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/supervised/window_based/base.py">  
            <div style="height:100%;width:100%">
                Window Based
            </div>
        </a>
    </td>
  <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/supervised/window_based/adwin.py">  
            <div style="height:100%;width:100%">
                ADWIN
            </div>
        </a>
    </td>
  <tr>
  <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/supervised/window_based/kswin.py">  
            <div style="height:100%;width:100%">
                KSWIN
            </div>
        </a>
    </td>
  </tr>
  </tr>
  <tr>
    <td rowspan="10">
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/data_drift/base.py"> 
            <div style="height:100%;width:100%">
                Data drift
            </div>
        </a>
    </td>
    <td rowspan="6">
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/unsupervised/distance_based/base.py"> 
            <div style="height:100%;width:100%">
                Distance Based
            </div>
        </a>
    </td>
    <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/unsupervised/distance_based/emd.py"> 
            <div style="height:100%;width:100%">
                EMD
            </div>
        </a>
    </td>
  </tr>
  <tr>
    <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/unsupervised/distance_based/histogram_intersection.py"> 
            <div style="height:100%;width:100%">
                Histogram Intersection
            </div>
        </a>
    </td>
  </tr>
  <tr>
    <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/unsupervised/distance_based/js.py"> 
            <div style="height:100%;width:100%">
                JS
            </div>
        </a>
    </td>
  </tr>
  <tr>
    <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/unsupervised/distance_based/kl.py"> 
            <div style="height:100%;width:100%">
                KL
            </div>
        </a>
    </td>
  </tr>
  <tr>
      <td>
      <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/unsupervised/distance_based/mmd.py"> 
                <div style="height:100%;width:100%">
                    MMD
                </div>
            </a>
      </td>
  </tr>
  <tr>
      <td>
      <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/unsupervised/distance_based/psi.py"> 
                <div style="height:100%;width:100%">
                    PSI
                </div>
            </a>
      </td>
  </tr>
  <tr>
    <td rowspan="4">
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/unsupervised/statistical_test/base.py"> 
            <div style="height:100%;width:100%">
                Statistical Test
            </div>
        </a>
    </td>
    <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/unsupervised/statistical_test/chisquare.py"> 
            <div style="height:100%;width:100%">
                Chi-Square
            </div>
        </a>
    </td>
  </tr>
  <tr>
    <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/unsupervised/statistical_test/cvm.py">
            <div style="height:100%;width:100%">
                CVM
            </div>    
        </a>
    </td>
  </tr>
  <tr>
    <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/unsupervised/statistical_test/ks.py">
            <div style="height:100%;width:100%">
                KS
            </div>    
        </a>
    </td>
  </tr>
  <tr>
    <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/unsupervised/statistical_test/welch_t_test.py">
            <div style="height:100%;width:100%">
                Welch's T-test
            </div>    
        </a>
    </td>
  </tr>
</tbody>
</table>

## Datasets

Some well-known datasets and synthetic generators are provided and listed in the following table.

<table class="center">
<thead>
<tr>
    <th>Type</th>
    <th>Dataset</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/datasets/real.py"> 
            <div style="height:100%;width:100%">
                Real
            </div>
        </a>
    </td>
    <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/datasets/real.py">  
            <div style="height:100%;width:100%">
                Elec2
            </div>
        </a>
    </td>
  </tr>
  <tr>
    <td rowspan="3">
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/datasets/synthetic.py"> 
            <div style="height:100%;width:100%">
                Synthetic
            </div>
        </a>
    </td>
    <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/datasets/synthetic.py">  
            <div style="height:100%;width:100%">
                SEA
            </div>
        </a>
    </td>
  </tr>
</tbody>
</table>