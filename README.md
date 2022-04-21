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
    <td>Supervised</td>
    <td>DDM Based</td>
    <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/supervised/ddm.py">
        </a>        
        <div style="height:100%;width:100%">
            DDM
        </div>    
    </td>
  </tr>
  <tr>
    <td colspan="3">&nbsp;</td>
  </tr>
  <tr>
    <td rowspan="3">Unsupervised</td>
    <td rowspan="1">Distance Based</td>
    <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/unsupervised/distance_based/emd.py">
        </a>        
        <div style="height:100%;width:100%">
            EMD
        </div>    
    </td>
  </tr>
  <tr>
    <td rowspan="2">Statistical Test</td>
    <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/unsupervised/statistical_test/cvm.py">
        </a>        
        <div style="height:100%;width:100%">
            CVM
        </div>    
    </td>
  </tr>
  <tr>
    <td>
        <a href="https://github.com/jaime-cespedes-sisniega/frouros/blob/main/frouros/unsupervised/statistical_test/ks.py">
        </a>        
        <div style="height:100%;width:100%">
            KS
        </div>    
    </td>
  </tr>
</tbody>
</table>