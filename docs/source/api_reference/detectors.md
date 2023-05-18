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

### Streaming

```{eval-rst}
.. automodule:: frouros.detectors.concept_drift.streaming
    :no-members:
    :no-inherited-members:
```

```{currentmodule} frouros.detectors.concept_drift.streaming
```

#### CUSUM Test

```{eval-rst}
.. automodule:: frouros.detectors.concept_drift.streaming.cusum_based
    :no-members:
    :no-inherited-members:
```

```{eval-rst}
.. autosummary::
    :toctree: auto_generated/
    :template: class.md

    CUSUM
    CUSUMConfig
    GeometricMovingAverage
    GeometricMovingAverageConfig
    PageHinkley
    PageHinkleyConfig
```

#### Statistical Process Control

```{eval-rst}
.. automodule:: frouros.detectors.concept_drift.streaming.statistical_process_control
    :no-members:
    :no-inherited-members:
```

```{eval-rst}
.. autosummary::
    :toctree: auto_generated/
    :template: class.md

    DDM
    DDMConfig
    ECDDWT
    ECDDWTConfig
    EDDM
    EDDMConfig
    HDDMA
    HDDMAConfig
    HDDMW
    HDDMWConfig
    RDDM
    RDDMConfig
```

#### Window Based

```{eval-rst}
.. automodule:: frouros.detectors.concept_drift.streaming.window_based
    :no-members:
    :no-inherited-members:
```

```{eval-rst}
.. autosummary::
    :toctree: auto_generated/
    :template: class.md

    ADWIN
    ADWINConfig
    KSWIN
    KSWINConfig
    STEPD
    STEPDConfig
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

    BhattacharyyaDistance
    EMD
    HellingerDistance
    HINormalizedComplement
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

#### Distance Based

```{eval-rst}
.. automodule:: frouros.detectors.data_drift.streaming.distance_based
    :no-members:
    :no-inherited-members:
```

```{eval-rst}
.. autosummary::
    :toctree: auto_generated/
    :template: class.md

    MMD
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