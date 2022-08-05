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

It provides a set of algorithms for drift detection, both for the supervised and unsupervised parts, as well as some semi-supervised algorithms. It is design with the intention of being integrated easily with the [scikit-learn](https://github.com/scikit-learn/scikit-learn) library. This integration allows Frouros to be used in machine learning problem pipelines, in the implementation of new drift detection algorithms and could be used to compare performance between detectors, as a benchmark.

## Quickstart

As a quick and easy example, we can generate two bivariate normal distribution in order to use an unsupervised method like MMD (Maximum Mean Discrepancy). This method tries to verify if generated samples come from the same distribution or not. If they come from different distributions, it means that there is covariate drift.

```python
from sklearn.gaussian_process.kernels import RBF
import numpy as np
from frouros.unsupervised.distance_based import MMD

np.random.seed(31)
# X samples from a normal distribution with mean = [1. 1.] and cov = [[2. 0.][0. 2.]]
x_mean = np.ones(2)
x_cov = 2*np.eye(2)
# Y samples a normal distribution with mean = [0. 0.] and cov = [[2. 1.][1. 2.]]
y_mean = np.zeros(2)
y_cov = np.eye(2) + 1

num_samples = 200
X_ref = np.random.multivariate_normal(x_mean, x_cov, num_samples)
X_test = np.random.multivariate_normal(y_mean, y_cov, num_samples)

alpha = 0.01  # significance level for the hypothesis test

detector = MMD(num_permutations=1000, kernel=RBF(length_scale=1.0), random_state=31)
detector.fit(X=X_ref)
detector.transform(X=X_test)
mmd, p_value = detector.distance

p_value < alpha
>>> True  # Drift detected. We can reject H0, so both samples come from different distributions.
```

More advance examples can be found [here](https://frouros.readthedocs.io).

## Installation

Frouros supports Python 3.8, 3.9 and 3.10 versions. It can be installed via pip:

```bash
pip install frouros
```
there is also the option to use [PyTorch](https://github.com/pytorch/pytorch) models with the help of [skorch](https://github.com/skorch-dev/skorch):
```bash
pip install frouros[pytorch]
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
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/supervised/base.py"> 
            <div style="height:100%;width:100%">
                Supervised
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
    <td rowspan="2">
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/semi_supervised/base.py"> 
            <div style="height:100%;width:100%">
                Semi-supervised
            </div>
        </a>
    </td>
    <td rowspan="2">
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/semi_supervised/margin_density_based/base.py"> 
            <div style="height:100%;width:100%">
                Margin Density Based
            </div>
        </a>
    </td>
    <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/semi_supervised/margin_density_based/md3.py"> 
            <div style="height:100%;width:100%">
                MD3-SVM
            </div>
        </a>
    </td>
  <tr>
    <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/supervised/margin_density_based/md3.py">  
            <div style="height:100%;width:100%">
                MD3-RS
            </div>
        </a>
    </td>
  </tr>
  </tr>
  <tr>
    <td rowspan="10">
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/unsupervised/base.py"> 
            <div style="height:100%;width:100%">
                Unsupervised
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