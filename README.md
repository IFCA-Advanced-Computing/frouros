# Frouros

<p style="text-align: center;">
  <!-- CI -->
  <a href="https://github.com/jaime-cespedes-sisniega/frouros/actions/workflows/ci.yml">
    <img src="https://github.com/jaime-cespedes-sisniega/frouros/actions/workflows/ci.yml/badge.svg?style=flat-square" alt="ci"/>
  </a>
  <!-- Code coverage -->
  <a href="https://codecov.io/gh/jaime-cespedes-sisniega/frouros">
    <img src="https://codecov.io/gh/jaime-cespedes-sisniega/frouros/branch/main/graph/badge.svg?token=DLKQSWYTYM" alt="coverage"/>
  </a>

[//]: # (  <!-- Documentation -->)

[//]: # (  <a href="">)

[//]: # (    <img src="" alt="documentation">)

[//]: # (  </a>)

[//]: # (  <!-- Roadmap -->)

[//]: # (  <a href="">)

[//]: # (    <img src="" alt="roadmap">)

[//]: # (  </a>)

[//]: # (  <!-- PyPI -->)

[//]: # (  <a href="">)

[//]: # (    <img src="" alt="pypi">)

[//]: # (  </a>)

[//]: # (  <!-- PePy -->)

[//]: # (  <a href="">)

[//]: # (    <img src="" alt="pepy">)

[//]: # (  </a>)
  <!-- License -->
  <a href="https://opensource.org/licenses/BSD-3-Clause">
    <img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg" alt="bsd_3_license">
  </a>
</p>

Frouros is a Python library for drift detection.

## Drift detection methods

<table class="tg">
<thead>
<tr>
    <th>Type</th>
    <th>Subtype</th>
    <th>Method</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/supervised/base.py"> 
            <div style="height:100%;width:100%">
                Supervised
            </div>
        </a>
    </td>
    <td rowspan="2">
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/supervised/ddm_based/base.py">  
            <div style="height:100%;width:100%">
                DDM Based
            </div>
        </a>
    </td>
    <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/supervised/distance_based/ddm.py">  
            <div style="height:100%;width:100%">
                DDM
            </div>
        </a>
    </td>
  <tr>
    <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/supervised/distance_based/eddm.py">  
            <div style="height:100%;width:100%">
                EDDM
            </div>
        </a>
    </td>
  </tr>
  <tr>
    <td rowspan="4">
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/unsupervised/base.py"> 
            <div style="height:100%;width:100%">
                Unsupervised
            </div>
        </a>
    </td>
    <td rowspan="2">
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
      <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/unsupervised/distance_based/psi.py"> 
                <div style="height:100%;width:100%">
                    PSI
                </div>
            </a>
      </td>
  </tr>
  <tr>
    <td rowspan="2">
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/unsupervised/statistical_test/base.py"> 
            <div style="height:100%;width:100%">
                Statistical Test
            </div>
        </a>
    </td>
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
</tbody>
</table>

## Datasets

<table class="tg">
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
        <a> 
            <div style="height:100%;width:100%">
                Synthetic
            </div>
        </a>
    </td>
  </tr>
</tbody>
</table>