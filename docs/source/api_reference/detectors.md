# Detectors

The {mod}`frouros.detectors` module contains drift detection algorithms.

```{eval-rst}
.. automodule:: frouros.detectors
    :no-members:
    :no-inherited-members:
```

```{currentmodule} frouros.detectors
```

## Concept drift

```{eval-rst}
.. automodule:: frouros.detectors.concept_drift
    :no-members:
    :no-inherited-members:
```

```{currentmodule} frouros.detectors.concept_drift
```

### CUSUM Test

```{eval-rst}
.. automodule:: frouros.detectors.concept_drift.cusum_based
    :no-members:
    :no-inherited-members:
```

```{eval-rst}
.. autosummary::
    :toctree: auto_generated/
    :template: class.md

    CUSUM
    GeometricMovingAverage
    PageHinkley
```

### DDM Based

```{eval-rst}
.. automodule:: frouros.detectors.concept_drift.ddm_based
    :no-members:
    :no-inherited-members:
```

```{eval-rst}
.. autosummary::
    :toctree: auto_generated/
    :template: class.md

    DDM
    ECDDWT
    EDDM
    HDDMA
    HDDMW
    RDDM
    STEPD
```

### Window Based

```{eval-rst}
.. automodule:: frouros.detectors.concept_drift.window_based
    :no-members:
    :no-inherited-members:
```

```{eval-rst}
.. autosummary::
    :toctree: auto_generated/
    :template: class.md

    ADWIN
    KSWIN
```

## Data drift

```{eval-rst}
.. automodule:: frouros.detectors.data_drift
    :no-members:
    :no-inherited-members:
```

```{currentmodule} frouros.detectors.data_drift
```

### Batch

```{eval-rst}
.. automodule:: frouros.detectors.data_drift.batch
    :no-members:
    :no-inherited-members:
```

```{currentmodule} frouros.detectors.data_drift.batch
```

#### Distance Based

```{eval-rst}
.. automodule:: frouros.detectors.data_drift.batch.distance_based
    :no-members:
    :no-inherited-members:
```

```{eval-rst}
.. autosummary::
    :toctree: auto_generated/
    :template: class.md

    EMD
    HistogramIntersection
    JS
    KL
    MMD
    PSI
```

#### Statistical Test

```{eval-rst}
.. automodule:: frouros.detectors.data_drift.batch.statistical_test
    :no-members:
    :no-inherited-members:
```

```{eval-rst}
.. autosummary::
    :toctree: auto_generated/
    :template: class.md

    ChiSquareTest
    CVMTest
    KSTest
    WelchTTest
```

### Streaming

```{eval-rst}
.. automodule:: frouros.detectors.data_drift.streaming
    :no-members:
    :no-inherited-members:
```

```{currentmodule} frouros.detectors.data_drift.streaming
```

#### Statistical Test

```{eval-rst}
.. automodule:: frouros.detectors.data_drift.streaming.statistical_test
    :no-members:
    :no-inherited-members:
```

```{eval-rst}
.. autosummary::
    :toctree: auto_generated/
    :template: class.md

    IncrementalKSTest
```